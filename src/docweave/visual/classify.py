"""Decide what an image is, from its pixels.

This is the gate the rest of the pipeline sits behind. It is deliberately
heuristic and deliberately dependency-light: classification has to be cheap
enough to run on every image in a document, and it has to work without a model.
A model that is slower than the thing it is routing is not a routing decision,
it is a tax.

The signals are all cheap and all auditable, and every decision ships the
numbers that produced it. That matters more than the thresholds: a reader who
disagrees with a classification can see exactly why it was made and what would
have to change, instead of trusting a black box.

Signals used:

- geometry -- width, height, aspect
- ink and background -- how much of the frame is dark, how much is empty
- colour -- colourfulness and saturation, because a photo and a line drawing
  differ enormously here and it costs almost nothing to measure
- structure -- how many closed primitives the contour detector finds, and how
  many straight runs the line detector finds
- text -- how much OCR text the image carries

Nothing here is a trained model, which means nothing here can be wrong in a way
we cannot explain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VisualKind(str, Enum):
    """What an image appears to be.

    ``UNKNOWN`` exists so that an unclassifiable image is reported rather than
    silently coerced into the nearest category -- the same reasoning as
    ``ShapeType.UNKNOWN`` in the diagram engine.
    """

    TEXT_PAGE = "text_page"
    DIAGRAM = "diagram"
    TABLE_IMAGE = "table_image"
    CHART = "chart"
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    UNKNOWN = "unknown"


#: A diagram is line art: mostly background, low colourfulness, and its
#: structure is made of closed outlines. These thresholds are the ones the
#: synthetic corpus and the real diagrams in this project were checked against.
_MAX_DIAGRAM_COLOURFULNESS = 22.0
_MIN_SHAPES_FOR_DIAGRAM = 2
_MIN_LINES_FOR_DIAGRAM = 2
#: A table in an image is a grid: many parallel runs crossing each other.
_MIN_GRID_CELLS = 9
#: A photo fills the frame with varied colour and carries little text.
_MIN_PHOTO_COLOURFULNESS = 30.0
_MIN_PHOTO_INK = 0.03
_MAX_PHOTO_INK = 0.75
#: A screenshot is structured and text-heavy but not a printed page: many small
#: text blocks rather than a few large ones.
_MIN_SCREENSHOT_BLOCKS = 6
_MAX_SCREENSHOT_TEXT_DENSITY = 6.0


@dataclass
class VisualClassification:
    """What the image is, how sure, and why.

    ``evidence`` holds the raw signals. It is not decoration: the point of a
    heuristic classifier is that its decisions can be audited, and a reader who
    disagrees with ``kind`` should be able to see the numbers that produced it.
    """

    kind: VisualKind
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def is_diagram(self) -> bool:
        return self.kind is VisualKind.DIAGRAM

    @property
    def needs_vision_model(self) -> bool:
        """True for kinds where a model adds information that pixels cannot.

        Photos and screenshots are the two: a pixel histogram says nothing about
        what a photograph *depicts*, and that is the whole question.
        """
        return self.kind in (VisualKind.PHOTO, VisualKind.SCREENSHOT)


def classify_image(
    image: Any,
    *,
    ocr_text: str = "",
    shapes: int | None = None,
    text_blocks: int = 0,
) -> VisualClassification:
    """Classify an image into a :class:`VisualKind`.

    ``ocr_text`` and ``shapes`` are supplied by the caller when it already has
    them, because a document conversion has usually read the text layer and can
    run the contour detector; recomputing both here would double the work on
    every page. When they are absent the classifier falls back to what the
    pixels alone say and reports a lower confidence rather than pretending
    otherwise.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a diagrams extra
        return VisualClassification(
            kind=VisualKind.UNKNOWN,
            confidence=0.0,
            reasons=["numpy is not installed, so the image cannot be measured"],
        )

    array = np.asarray(image)
    if array.ndim == 2:
        height, width = array.shape
        channels = 1
        gray = array
    elif array.ndim == 3:
        height, width = array.shape[:2]
        channels = array.shape[2]
        gray = array[:, :, :3].mean(axis=2)
    else:  # pragma: no cover - defensive
        return VisualClassification(
            kind=VisualKind.UNKNOWN,
            confidence=0.0,
            reasons=["image is not a 2-D or 3-D array"],
        )

    area = float(max(1, width * height))
    evidence: dict[str, Any] = {
        "width": width,
        "height": height,
        "aspect": round(width / height, 3) if height else 0.0,
    }

    ink = (gray < 128).mean()
    background = (gray > 240).mean()
    evidence["ink_fraction"] = round(float(ink), 4)
    evidence["background_fraction"] = round(float(background), 4)

    colourfulness, saturation, n_colours = _colour_stats(array, channels, np)
    evidence["colourfulness"] = round(colourfulness, 2)
    evidence["saturation"] = round(saturation, 2)
    evidence["distinct_colours"] = n_colours

    lines = _line_runs(gray, np)
    evidence["horizontal_runs"] = len(lines["horizontal"])
    evidence["vertical_runs"] = len(lines["vertical"])
    evidence["grid_cells"] = lines["cells"]

    if shapes is None:
        shapes = _count_shapes(gray)
    evidence["shapes"] = shapes

    text_len = len((ocr_text or "").strip())
    evidence["text_chars"] = text_len
    evidence["text_density"] = round(text_len / (area / 10000.0), 3)

    return _decide(
        evidence, colourfulness, saturation, shapes, lines, text_len, text_blocks
    )


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------


def _colour_stats(array, channels: int, np) -> tuple[float, float, int]:
    """Colourfulness, mean saturation and a count of distinct colours.

    Colourfulness is the Hasler-Süsstrunk metric -- the spread of pixel
    differences -- which separates a line drawing from a photograph far more
    reliably than a raw colour count, and needs no colour space conversion.
    """
    if channels < 3:
        return 0.0, 0.0, 1
    rgb = array[:, :, :3].astype(np.float32)
    rg = rgb[:, :, 0] - rgb[:, :, 2]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 2]) - rgb[:, :, 1]
    colourfulness = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    saturation = float((mx - mn).mean()) / 255.0 if mx.size else 0.0

    # Quantise to 3 bits per channel: enough to see "two colours" vs "a
    # photograph", small enough that antialiasing does not inflate the count.
    quantised = (array[:, :, :3].astype(np.uint16) >> 5).reshape(-1, 3)
    n_colours = len(np.unique(quantised, axis=0))
    return colourfulness, saturation, n_colours


def _line_runs(gray, np) -> dict:
    """Long straight runs, split by orientation, and how many cells they grid.

    A table is a grid and a diagram is not, and the difference is countable:
    a grid has many horizontal and vertical runs that cross, a flowchart has a
    handful of connectors that mostly do not.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is a diagrams extra
        return {"horizontal": [], "vertical": [], "cells": 0}

    height, width = gray.shape
    edges = cv2.Canny(gray.astype(np.uint8), 60, 160, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=max(24, int(width * 0.05)),
        maxLineGap=12,
    )
    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = (float(v) for v in line)
            if abs(y2 - y1) <= abs(x2 - x1) * 0.2 and abs(x2 - x1) > 24:
                horizontal.append((round((y1 + y2) / 2, 1), min(x1, x2), max(x1, x2)))
            elif abs(x2 - x1) <= abs(y2 - y1) * 0.2 and abs(y2 - y1) > 24:
                vertical.append((round((x1 + x2) / 2, 1), min(y1, y2), max(y1, y2)))

    # A table grid is made of rules that span the table. A dotted or decorative
    # background also produces many runs, but they are short, so only runs
    # crossing a real share of the frame count towards a grid. Without this the
    # user's pastel diagram, whose whole background is a dot grid, reads as a
    # 284-cell table.
    min_span_h = max(24.0, width * 0.25)
    min_span_v = max(24.0, height * 0.25)
    grid_h = {y for y, low, high in horizontal if high - low >= min_span_h}
    grid_v = {x for x, low, high in vertical if high - low >= min_span_v}

    # Long lines are not enough: the clip art in a real diagram has outlines
    # that span the frame too. A table's rows and columns are *regular*, which
    # is what separates a grid from decoration.
    regular = (
        _regular_spacing(grid_h) and _regular_spacing(grid_v) and len(grid_h) >= 3 and len(grid_v) >= 3
    )
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "grid_horizontal": len(grid_h),
        "grid_vertical": len(grid_v),
        "grid_regular": regular,
        "cells": len(grid_h) * len(grid_v) if regular else 0,
    }


def _regular_spacing(positions, *, min_count: int = 3, max_cv: float = 0.6) -> bool:
    """True when positions are evenly spaced, which is what a grid looks like.

    ``max_cv`` is the coefficient of variation of the gaps. A table's rows are
    near-uniform; decorative or illustrative lines are not, and a high spread
    is the signal that the runs are not a table.
    """
    ordered = sorted(positions)
    if len(ordered) < min_count:
        return False
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if len(gaps) < min_count - 1:
        return False
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return False
    variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return math.sqrt(variance) / mean <= max_cv


def _count_shapes(gray) -> int:
    """How many closed primitives the diagram detector finds, if available.

    Imported lazily and never required: the classifier is useful on a machine
    that has no diagram extra, and a diagram it cannot measure is still
    measurable as "structure with low colourfulness".
    """
    try:
        import cv2
        import numpy as np

        from docweave.diagram.shapes import detect_shapes
    except ImportError:
        return 0

    bgr = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return len(detect_shapes(bgr))


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


def _decide(
    evidence: dict,
    colourfulness: float,
    saturation: float,
    shapes: int,
    lines: dict,
    text_len: int,
    text_blocks: int,
) -> VisualClassification:
    """Apply the rules in order and say which one fired.

    Order matters. A diagram is checked before text because a flowchart can
    carry a lot of OCR text and is not therefore a page. A photo is checked
    only after the structural kinds, because colourfulness alone would call a
    coloured diagram a photograph.
    """
    cells = lines["cells"]
    density = evidence["text_density"]
    reasons: list[str] = []

    # A diagram: closed outlines, connectors, and the low colourfulness that
    # goes with line art. The user's real diagrams fail this on colourfulness,
    # which is exactly the signal that keeps them out of the diagram engine.
    if (
        shapes >= _MIN_SHAPES_FOR_DIAGRAM
        and len(lines["horizontal"]) + len(lines["vertical"]) >= _MIN_LINES_FOR_DIAGRAM
        and colourfulness <= _MAX_DIAGRAM_COLOURFULNESS
    ):
        reasons.append(
            f"{shapes} closed shapes and line structure with colourfulness "
            f"{colourfulness:.1f}, which is line art"
        )
        return VisualClassification(VisualKind.DIAGRAM, 0.8, reasons, evidence)

    # A table in an image: a grid, counted rather than guessed.
    if cells >= _MIN_GRID_CELLS:
        reasons.append(
            f"{lines['grid_horizontal']} horizontal and "
            f"{lines['grid_vertical']} vertical rules spanning the frame, "
            f"forming {cells} cells"
        )
        return VisualClassification(VisualKind.TABLE_IMAGE, 0.7, reasons, evidence)

    # A chart: structure and some text, but not a grid and not line art. This
    # is the weakest of the rules and says so through its confidence.
    if text_len > 20 and cells >= 4 and colourfulness > _MAX_DIAGRAM_COLOURFULNESS:
        reasons.append(
            f"text-bearing structure with {cells} grid cells and "
            f"colourfulness {colourfulness:.1f}"
        )
        return VisualClassification(VisualKind.CHART, 0.5, reasons, evidence)

    # A page of text. High density is the whole test: a page is mostly words.
    if density > 0.5:
        reasons.append(f"{text_len} characters of text at density {density}")
        return VisualClassification(VisualKind.TEXT_PAGE, 0.75, reasons, evidence)

    # A screenshot: structured, text-heavy, but broken into many small blocks
    # rather than the long runs of a printed page.
    if text_blocks >= _MIN_SCREENSHOT_BLOCKS and density <= _MAX_SCREENSHOT_TEXT_DENSITY:
        reasons.append(
            f"{text_blocks} separate text blocks at density {density}, which is "
            "interface rather than prose"
        )
        return VisualClassification(VisualKind.SCREENSHOT, 0.5, reasons, evidence)

    # A photo or illustration: colour, moderate ink, and no structure that
    # anything else would claim. This is what the user's real diagrams are.
    if (
        colourfulness > _MIN_PHOTO_COLOURFULNESS
        and _MIN_PHOTO_INK <= evidence["ink_fraction"] <= _MAX_PHOTO_INK
    ):
        reasons.append(
            f"colourfulness {colourfulness:.1f} with {evidence['ink_fraction']:.2f} "
            "ink and no diagram structure"
        )
        return VisualClassification(VisualKind.PHOTO, 0.55, reasons, evidence)

    reasons.append(
        "no rule matched: "
        + ", ".join(
            f"{k}={v}"
            for k, v in evidence.items()
            if k in ("shapes", "colourfulness", "grid_cells", "text_density", "ink_fraction")
        )
    )
    return VisualClassification(VisualKind.UNKNOWN, 0.0, reasons, evidence)
