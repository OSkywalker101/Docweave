"""Text region detection and pluggable label readers.

Split into two halves on purpose:

* **Region detection** is geometric and always runs. We always know *where*
  text is, even when we cannot read it.
* **Reading** is pluggable and lazy. An OCR engine is imported only when a
  caller actually asks for labels and an engine is available.

That split is what lets the engine reconstruct structure — the actual goal —
without any OCR dependency at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from docweave.diagram.shapes import Box, binarize

#: A text line is usually wider than tall and not tiny.
MIN_TEXT_HEIGHT_RATIO = 0.008
MAX_TEXT_HEIGHT_RATIO = 0.16
#: Stroke density of glyphs is high; solid blocks are fills, not text.
MAX_TEXT_FILL_RATIO = 0.72
#: A label is a run of glyphs. A single connected blob is far more likely to be
#: an arrowhead or a rule, and treating it as text would erase it before the
#: arrowhead detector ever sees it.
MIN_TEXT_GLYPHS = 2
#: Identical blob heights across several blobs mean a dashed rule, not a word.
#: A real label mixes ascenders and descenders; a dashed connector does not,
#: and accepting one would erase the connector before the line detector ran.
MIN_RULE_GLYPHS = 3


@dataclass
class TextRegion:
    """A detected block of text-like ink."""

    box: Box
    #: Fraction of the box covered by ink.
    fill: float
    #: Estimated glyph height, used to rank likely label lines.
    height: int
    #: Associated node id once association runs.
    node_id: str | None = None
    #: "shape" when inside a shape, "edge" when on a connector, "free" otherwise.
    placement: str = "free"
    text: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    def to_dict(self) -> dict:
        return {
            "box": list(self.box),
            "fill": round(self.fill, 4),
            "placement": self.placement,
            "node_id": self.node_id,
            "text": self.text,
        }


def detect_text_regions(image: np.ndarray, *, min_height: int | None = None) -> list[TextRegion]:
    """Locate text-like regions with a morphological gradient.

    The morphological gradient responds to edges, so it highlights glyph
    strokes and ignores the interior of a filled box. Horizontal closing then
    merges a word into a line.
    """
    cv2 = _import_cv2()
    binary = binarize(image)
    height_px, width_px = binary.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel, iterations=1)
    _, closed = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, connect, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    floor = min_height or max(6.0, height_px * MIN_TEXT_HEIGHT_RATIO)
    ceiling = max(floor + 1.0, height_px * MAX_TEXT_HEIGHT_RATIO)
    ink_mask = (binary > 0).astype("uint8")

    regions: list[TextRegion] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < floor or h > ceiling:
            continue
        if w < 3:
            continue
        aspect = w / h if h else 0.0
        if aspect < 0.18:
            continue  # a single glyph or a vertical rule
        area = float(cv2.contourArea(contour))
        fill = area / float(w * h) if w * h else 0.0
        if fill <= 0 or fill > MAX_TEXT_FILL_RATIO:
            continue
        heights = _blob_heights(ink_mask[y : y + h, x : x + w])
        if len(heights) < MIN_TEXT_GLYPHS:
            continue
        if _is_uniform_rule(heights):
            continue
        regions.append(TextRegion(box=(int(x), int(y), int(w), int(h)), fill=fill, height=int(h)))

    return sorted(regions, key=lambda r: (r.box[1], r.box[0]))


def _blob_heights(mask) -> list[int]:
    """Heights of the connected ink components inside a candidate region."""
    cv2 = _import_cv2()
    if mask.size == 0:
        return []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    return [int(stats[index][3]) for index in range(1, count) if int(stats[index][4]) >= 2]


def _is_uniform_rule(heights: Sequence[int]) -> bool:
    """True when every blob is the same height — a dashed rule, not a word."""
    return len(heights) >= MIN_RULE_GLYPHS and max(heights) == min(heights)


def group_regions(regions: Sequence[TextRegion], line_gap_ratio: float = 0.5) -> list[TextRegion]:
    """Merge vertically adjacent regions into multi-line text blocks."""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: (r.box[0] // 24, r.box[1]))
    merged: list[list[TextRegion]] = [[ordered[0]]]
    for region in ordered[1:]:
        current = merged[-1]
        last = current[-1]
        same_column = abs(region.box[0] - last.box[0]) <= max(last.box[2], 12) * 0.6
        gap = region.box[1] - (last.box[1] + last.box[3])
        if same_column and 0 <= gap <= last.height * (1 + line_gap_ratio):
            current.append(region)
        else:
            merged.append([region])
    return [_merge(group) for group in merged]


def _merge(group: Sequence[TextRegion]) -> TextRegion:
    x0 = min(r.box[0] for r in group)
    y0 = min(r.box[1] for r in group)
    x1 = max(r.box[0] + r.box[2] for r in group)
    y1 = max(r.box[1] + r.box[3] for r in group)
    box = (x0, y0, x1 - x0, y1 - y0)
    area = float((x1 - x0) * (y1 - y0))
    ink = sum(r.fill * r.box[2] * r.box[3] for r in group)
    return TextRegion(
        box=box,
        fill=ink / area if area else 0.0,
        height=y1 - y0,
    )


# --------------------------------------------------------------------------
# label readers
# --------------------------------------------------------------------------


class LabelReader(Protocol):
    """Reads the characters inside a region.

    Implementations must be safe to construct cheaply and must not import their
    engine until :meth:`available` or :meth:`read` is called.
    """

    name: str

    def available(self) -> bool: ...

    def read(self, image: np.ndarray, box: Box) -> tuple[str | None, float]: ...


class NullLabelReader:
    """Used when no OCR engine is available.

    It reports that it cannot read, so callers mark the region unread and attach
    the "install an OCR engine" hint. The region itself is still reported, which
    is what makes the spatial association useful without OCR.
    """

    name = "none"

    def available(self) -> bool:
        return False

    def read(self, image: np.ndarray, box: Box) -> tuple[str | None, float]:
        return None, 0.0


class TesseractLabelReader:
    """Tesseract via pytesseract. Imported lazily, on first use."""

    name = "tesseract"

    def __init__(self, language: str = "eng", psm: int = 7) -> None:
        self.language = language
        self.psm = psm

    def available(self) -> bool:
        try:
            import pytesseract
        except ImportError:
            return False
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False
        return True

    def read(self, image: np.ndarray, box: Box) -> tuple[str | None, float]:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:  # pragma: no cover - guarded by available()
            return None, 0.0
        x, y, w, h = box
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + w + pad)
        y1 = min(image.shape[0], y + h + pad)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None, 0.0
        try:
            data = pytesseract.image_to_data(
                crop, lang=self.language, config=f"--psm {self.psm}", output_type=Output.DICT
            )
        except Exception:
            return None, 0.0
        words: list[str] = []
        confidences: list[float] = []
        # Tesseract returns text and confidence as parallel arrays. They are required
        # to align: truncating to the shorter one would drop recognised words
        # without saying so, which is the one failure this engine must not make
        # quietly. A mismatch raises, and the surrounding handler reports the
        # page as unreadable.
        for text, conf in zip(data.get("text", []), data.get("conf", []), strict=True):
            text = (text or "").strip()
            if not text:
                continue
            words.append(text)
            try:
                value = float(conf)
            except (TypeError, ValueError):
                value = -1.0
            if value >= 0:
                confidences.append(value)
        if not words:
            return None, 0.0
        score = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        return " ".join(words), score


_WHITESPACE = re.compile(r"\s+")


def default_label_reader():
    """The label reader, which is Tesseract -- deliberately not Indra.

    Page OCR uses Indra: it is more accurate, an order of magnitude faster,
    and returns labelled layout blocks. Diagram labels are the opposite case,
    and the reason is measured rather than assumed.

    A label is a 63x12 crop. Indra's line detector returns nothing for it at
    4x, 8x, 12x and 16x upscale, with and without padding. Tesseract in
    single-line mode (``--psm 7``) reads the same crop correctly and
    immediately. A page pipeline pointed at one word does not become a line
    recogniser.

    So the faster, more accurate engine keeps the pages and the one that
    actually works keeps the labels. Recorded in docs/architecture.md so the
    experiment is not repeated.
    """
    reader = TesseractLabelReader()
    if reader.available():
        return reader
    return NullLabelReader()


def clean_label(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = _WHITESPACE.sub(" ", text).strip()
    return cleaned or None


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - declared dependency
        from docweave.exceptions import DependencyMissingError

        raise DependencyMissingError("opencv-python", "diagram analysis", "diagrams") from exc
    return cv2
