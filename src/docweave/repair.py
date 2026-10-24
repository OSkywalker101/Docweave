"""Targeted OCR repair.

This is Phase 3 of the pipeline, and the only place the engine goes back and
re-does work it already decided was missing. Three rules govern it:

1. **Only repair what was measured to be broken.** Pages are selected from the
   signals the first pass collected. A document with a complete text layer is
   never rasterised.
2. **Never overwrite good content.** Original extraction is retained on every
   repaired node, and a repair that produces *less* text than the original is
   rejected rather than accepted.
3. **Disclose the repair.** Every repaired page produces a diagnostic naming the
   engine, the confidence and the before/after character counts.

OCR is optional and never imported until a page actually needs it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from docweave.config import ConversionConfig
from docweave.models import Diagnostic, Document, Origin, Page, Paragraph, SourceInfo
from docweave.models.diagnostics import Category, Code, Severity

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from docweave.models import SourceRef


class OcrEngine(Protocol):
    """Anything that can turn a page image into text."""

    name: str

    def available(self) -> bool: ...

    def read(self, page: RenderedPage) -> tuple[str | None, float]: ...


class IndraEngine:
    """Indra OCR (datalab-to/indra), via ONNX Runtime.

    Chosen over Tesseract for accuracy and speed: on OmniParseBench its text
    score is 41.3 against Tesseract's 19.3, at roughly 200 ms a page on four
    cores rather than five seconds. It needs no PyTorch and about 18 MB of
    model weights, and unlike Tesseract it returns layout blocks with labels
    and a reading order, which is information this engine can use rather than
    discard.

    Indra is a *document* pipeline, not a line recogniser. The repair loop asks
    for one page at a time, so results are cached per page: without that, every
    call would re-parse the file and re-warm the models.

    The weights are licensed separately from the code (modified OpenRAIL-M).
    See docs/dependencies.md -- this is a licensing decision, not just a
    technical one.
    """

    name = "indra"

    def __init__(self, *, threads: int = 2, mode: str = "auto") -> None:
        self.threads = threads
        self.mode = mode
        self._conv = None
        self._cache: dict[int, tuple[str | None, float]] = {}
        self._labels: dict[int, list[dict]] = {}

    def available(self) -> bool:
        try:
            import indra_ocr  # noqa: F401
        except ImportError:
            return False
        return True

    def read(self, page: RenderedPage) -> tuple[str | None, float]:
        if page.page_number in self._cache:
            return self._cache[page.page_number]
        result = self._read_uncached(page)
        self._cache[page.page_number] = result
        return result

    def blocks_for(self, page_number: int) -> list[dict]:
        """Layout blocks with labels for a page, if the engine produced any."""
        return list(self._labels.get(page_number, ()))

    def _read_uncached(self, page: RenderedPage) -> tuple[str | None, float]:
        try:
            import indra_ocr
        except ImportError:
            return None, 0.0
        try:
            # Held in a local so the type checker can narrow it: assigning to
            # ``self._conv`` and then using the attribute does not narrow.
            conv = self._conv
            if conv is None:
                conv = indra_ocr.load(threads=self.threads, mode=self.mode)
                self._conv = conv
            for index, converted in conv.convert(page.source_path):
                # 0-based index from the engine, 1-based page number here.
                if index + 1 != page.page_number:
                    continue
                self._labels[page.page_number] = list(converted.blocks or [])
                text = (converted.text or "").strip()
                if not text:
                    return None, 0.0
                confidences = [
                    float(getattr(line, "conf", 0.0)) / 100.0
                    for line in (converted.lines or ())
                    if getattr(line, "text", None)
                ]
                confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )
                return text, confidence
        except Exception:
            return None, 0.0
        return None, 0.0


@dataclass
class RenderedPage:
    """A page rasterised at a chosen resolution, ready for OCR."""

    page_number: int
    #: numpy array; typed loosely so numpy stays an optional import here.
    image: object
    dpi: int = 200
    #: Where the page came from. Document-level engines read the file rather
    #: than a raster; line-level engines ignore this.
    source_path: str = ""


@dataclass
class RepairReport:
    """What the repair pass did, and what it could not do."""

    repaired_pages: list[int] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)
    rejected_pages: list[dict] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    elapsed_ms: float = 0.0
    engine: str = "none"

    @property
    def repaired_count(self) -> int:
        return len(self.repaired_pages)


class TesseractEngine:
    """Tesseract through pytesseract. Imported on first use, never at module load."""

    name = "tesseract"

    def __init__(self, language: str = "eng") -> None:
        self.language = language

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

    def read(self, page: RenderedPage) -> tuple[str | None, float]:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:  # pragma: no cover - guarded by available()
            return None, 0.0
        try:
            data = pytesseract.image_to_data(
                page.image,
                lang=self.language,
                config="--psm 6",
                output_type=Output.DICT,
            )
        except Exception:
            return None, 0.0

        lines: dict[tuple[int, int, int], list[str]] = {}
        confidences: list[float] = []
        for text, conf, page_num, block, par, line in zip(
            data.get("text", []),
            data.get("conf", []),
            data.get("page_num", []),
            data.get("block_num", []),
            data.get("par_num", []),
            data.get("line_num", []),
            strict=True,
        ):
            text = (text or "").strip()
            if not text:
                continue
            lines.setdefault((page_num, block, par, line), []).append(text)
            try:
                value = float(conf)
            except (TypeError, ValueError):
                value = -1.0
            if value >= 0:
                confidences.append(value)

        if not lines:
            return None, 0.0
        ordered = sorted(lines)
        rendered = "\n".join(" ".join(lines[key]) for key in ordered)
        score = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        return rendered, score


def select_pages(document: Document, config: ConversionConfig) -> list[int]:
    """Choose the pages that measurement says are broken.

    A page qualifies when it has a raster layer and too little text to be a real
    text page. This is the same signal the validator reports, so the repair set
    and the report can never disagree.
    """
    selected: list[int] = []
    for page in document.pages:
        signals = page.signals
        heavy = signals.image_count > 0 and signals.media_area_ratio >= config.image_heavy_area_ratio
        if signals.char_count <= config.scanned_max_chars and (heavy or signals.image_count > 0):
            selected.append(page.index)
    return selected


class OcrRepairer:
    """Runs OCR over the pages the first pass could not read."""

    def __init__(self, config: ConversionConfig | None = None, engine: OcrEngine | None = None):
        self.config = config or ConversionConfig()
        self.engine = engine if engine is not None else default_engine()

    def available(self) -> bool:
        try:
            return bool(self.engine.available())
        except Exception:
            return False

    def repair(
        self,
        document: Document,
        render_page,
        *,
        source: SourceInfo | None = None,
    ) -> RepairReport:
        """Repair the document in place. ``render_page`` yields a page image.

        The original extraction is kept on each node's sources, so a consumer can
        always see what came from the text layer and what came from OCR.
        """
        report = RepairReport(engine=self.engine.name)
        started = time.perf_counter()

        targets = select_pages(document, self.config)
        if not targets:
            report.elapsed_ms = (time.perf_counter() - started) * 1000.0
            return report

        if not self.available():
            report.skipped_pages = list(targets)
            report.diagnostics.append(
                Diagnostic(
                    code=Code.OCR_UNAVAILABLE,
                    category=Category.INTERVENTION,
                    severity=Severity.WARNING,
                    message=(
                        f"{len(targets)} page(s) need OCR but no OCR engine is available; "
                        "their content is missing from the output."
                    ),
                    reason=(
                        "Tesseract is not installed, or 'docweave[ocr]' is missing."
                    ),
                    hint="Install Tesseract and 'docweave[ocr]', then re-run to repair these pages.",
                    evidence={"pages": targets, "engine": self.engine.name},
                )
            )
            report.elapsed_ms = (time.perf_counter() - started) * 1000.0
            return report

        by_index = {page.index: page for page in document.pages}
        for page_number in targets:
            page = by_index.get(page_number)
            if page is None:
                continue
            rendered = render_page(page_number)
            if rendered is None:
                report.skipped_pages.append(page_number)
                continue
            text, confidence = self.engine.read(rendered)
            cleaned = (text or "").strip()

            if not cleaned:
                report.rejected_pages.append(
                    {"page": page_number, "reason": "OCR returned no text", "confidence": confidence}
                )
                report.diagnostics.append(
                    Diagnostic(
                        code=Code.OCR_EMPTY,
                        category=Category.UNVERIFIED,
                        severity=Severity.WARNING,
                        message=f"OCR returned no text for page {page_number}.",
                        reason="The page may be blank, or the scan may be too low resolution to read.",
                        scope_kind="page",
                        scope_index=page_number,
                        evidence={"confidence": round(confidence, 4)},
                    )
                )
                continue

            original_chars = page.signals.char_count
            if len(cleaned) < original_chars:
                # Rule 2: a repair that loses content is worse than no repair.
                report.rejected_pages.append(
                    {
                        "page": page_number,
                        "reason": "OCR produced less text than the existing extraction",
                        "before": original_chars,
                        "after": len(cleaned),
                    }
                )
                report.diagnostics.append(
                    Diagnostic(
                        code=Code.OCR_REJECTED,
                        category=Category.STRUCTURAL,
                        severity=Severity.INFO,
                        message=(
                            f"OCR on page {page_number} produced {len(cleaned)} characters, "
                            f"fewer than the {original_chars} already extracted. The original was kept."
                        ),
                        reason="Overwriting a better extraction with a worse one would lose content.",
                        scope_kind="page",
                        scope_index=page_number,
                        evidence={"before": original_chars, "after": len(cleaned)},
                    )
                )
                continue

            self._apply(page, page_number, cleaned, confidence, source)
            report.repaired_pages.append(page_number)
            report.diagnostics.append(
                Diagnostic(
                    code=Code.OCR_REPAIRED,
                    category=Category.UNVERIFIED,
                    severity=Severity.INFO,
                    message=(
                        f"Page {page_number} was repaired by OCR: {original_chars} -> "
                        f"{len(cleaned)} characters."
                    ),
                    reason=f"{self.engine.name} read the raster layer; the text layer had none.",
                    hint="Spot-check OCR output before relying on it; it is never exact.",
                    scope_kind="page",
                    scope_index=page_number,
                    evidence={
                        "engine": self.engine.name,
                        "confidence": round(confidence, 4),
                        "before_chars": original_chars,
                        "after_chars": len(cleaned),
                    },
                )
            )

        if report.repaired_pages:
            document.metadata["ocr_repaired_pages"] = list(report.repaired_pages)
        report.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return report

    def _apply(
        self,
        page: Page,
        page_number: int,
        text: str,
        confidence: float,
        source: SourceInfo | None,
    ) -> None:
        """Attach OCR text while preserving the original provenance.

        When the engine reports layout blocks, they are turned into nodes of the
        right kind rather than collapsed into one paragraph. An OCR engine that
        can tell a heading from a list has done work worth keeping: the whole
        point of this project is that structure survives conversion, and a flat
        blob of page text throws away exactly what was recovered.
        """

        blocks = getattr(self.engine, "blocks_for", None)
        labelled = blocks(page_number) if callable(blocks) else []
        nodes = (
            _nodes_from_blocks(labelled, page_number, confidence, self.engine.name)
            if labelled
            else [_ocr_paragraph(text, page_number, confidence, self.engine.name)]
        )

        # Keep whatever the text layer did produce, clearly subordinate.
        produced = nodes
        for node in produced:
            for existing in page.blocks:
                node.sources.extend(existing.sources)
        page.blocks = list(produced) + list(page.blocks)
        page.signals.char_count = max(page.signals.char_count, len(text))


def _ocr_source(
    page_number: int, confidence: float, engine: str, note: str
) -> SourceRef:  # noqa: F821 - resolved under TYPE_CHECKING below
    from docweave.models import Confidence, Locator, SourceRef

    return SourceRef(
        adapter=f"ocr:{engine}",
        origin=Origin.INFERRED,
        confidence=Confidence.MEDIUM if confidence >= 0.6 else Confidence.LOW,
        locator=Locator(kind="page", page=page_number),
        note=note,
    )


def _ocr_paragraph(text: str, page_number: int, confidence: float, engine: str):
    """One node, for engines that report only flat text."""

    return Paragraph(
        id=f"p{page_number}.ocr",
        text=text,
        sources=[
            _ocr_source(
                page_number,
                confidence,
                engine,
                (
                    f"Recovered by {engine} at {confidence:.0%} confidence. "
                    "OCR text is never exact and must be verified."
                ),
            )
        ],
    )


#: How an engine's layout labels map onto IR node kinds. Only labels that mean
#: something structural are listed; anything unknown becomes body text rather
#: than being dropped or guessed at.
_LABEL_KINDS = {
    "section-header": "heading",
    "heading": "heading",
    "list-group": "list",
    "list": "list",
}


_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")


def _strip_bullet(line: str) -> str:
    """Remove a leading bullet or number, leaving the words."""
    return _BULLET.sub("", line).strip()


def _looks_ordered(text: str) -> bool:
    """True when the first item carries a number, so the list is numbered."""
    for line in text.splitlines():
        if line.strip():
            return bool(re.match(r"^\s*\d+[.)]\s+", line))
    return False


def _nodes_from_blocks(
    blocks: list[dict], page_number: int, confidence: float, engine: str
) -> list:
    """Build nodes from labelled layout blocks, in reading order."""
    from docweave.models import Heading, ListBlock, ListItem

    nodes: list = []
    heading_level = 0
    for index, block in enumerate(blocks):
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        label = str(block.get("label") or "").strip().lower()
        kind = _LABEL_KINDS.get(label, "text")
        note = (
            f"Recovered by {engine} at {confidence:.0%} confidence; "
            f"layout label {label or 'unlabelled'}. OCR text is never exact."
        )
        source = _ocr_source(page_number, confidence, engine, note)

        if kind == "heading":
            # A step down the outline when a deeper header appears, capped at
            # six levels, which is deeper than any reader renders.
            heading_level = 1 if heading_level == 0 else min(heading_level + 1, 6)
            nodes.append(
                Heading(
                    id=f"p{page_number}.ocr.h{len(nodes)}",
                    text=text,
                    level=heading_level,
                    sources=[source],
                )
            )
            continue

        if kind == "list":
            # The engine emits the group as text, one item per line, with any
            # bullet or number already attached. Strip it rather than trusting
            # it: a leading "- " twice is harmless, a missed one is not.
            items = [
                ListItem(text=_strip_bullet(line))
                for line in text.splitlines()
                if line.strip()
            ]
            if not items:
                continue
            nodes.append(
                ListBlock(
                    id=f"p{page_number}.ocr.l{len(nodes)}",
                    ordered=_looks_ordered(text),
                    items=items,
                    sources=[source],
                )
            )
            continue

        nodes.append(
            Paragraph(
                id=f"p{page_number}.ocr.b{index}",
                text=text,
                sources=[source],
            )
        )
    return nodes


def default_engine() -> OcrEngine:
    """Pick the best OCR backend that is actually installed.

    Indra first: it is several times more accurate and an order of magnitude
    faster than Tesseract. Tesseract remains supported rather than removed --
    it is a single binary with no model download, and a caller who needs no
    network on first run still wants it. Order is therefore a preference, not
    a requirement, and both report honestly when neither is present.
    """
    for candidate in (IndraEngine, TesseractEngine):
        engine = candidate()
        if engine.available():
            return engine
    return TesseractEngine()


class RenderedPageShim:
    """Adapter so an engine can take a plain image plus a page number."""

    def __init__(self, image, page_number: int, dpi: int = 200) -> None:
        self.page_number = page_number
        self.image = image
        self.dpi = dpi


def repair_document(
    path,
    document: Document,
    config: ConversionConfig | None = None,
    engine: OcrEngine | None = None,
):
    """Convenience wrapper: rasterise the needed pages, then repair.

    Returns ``None`` when the document has no pages to repair, so callers can
    skip the whole path cheaply.
    """
    config = config or ConversionConfig()
    targets = select_pages(document, config)
    if not targets:
        return None

    engine = engine if engine is not None else default_engine()

    # Indra reads the file itself; only a raster engine needs the renderer.
    needs_raster = getattr(engine, "name", "") != "indra"
    if needs_raster:
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            return None

    def render_page(page_number: int):
        # Carrying the path lets a document-level engine read the file itself
        # instead of re-encoding an image it could have opened directly.
        source = str(path)
        if not needs_raster:
            return RenderedPage(page_number=page_number, image=None, source_path=source)
        with pymupdf.open(str(path)) as doc:
            page = doc[page_number - 1]
            pixmap = page.get_pixmap(dpi=200)
            import numpy as np

            return RenderedPage(
                page_number=page_number,
                image=np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                ),
                source_path=source,
            )

    return OcrRepairer(config, engine).repair(document, render_page)
