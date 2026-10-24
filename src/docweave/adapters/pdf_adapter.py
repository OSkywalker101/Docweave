"""PDF adapter built on PyMuPDF.

Extraction strategy, in cost order:

1. Native text with geometry (one ``get_text("dict")`` pass per page).
2. Cheap structural signals derived from that same pass -- no second parse.
3. Optional table recovery, attempted only on pages that look like they contain
   a ruled table, and only when ``pdf_tables`` allows it.
4. Reading-order repair from bounding boxes when a column gutter is detected.

OCR is deliberately absent from this path. When a page looks scanned we emit an
unverified-content diagnostic instead of silently invoking a slow, lossy engine.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import clean_text, measure
from docweave.exceptions import (
    EncryptedDocumentError,
    LimitExceededError,
    MalformedDocumentError,
)
from docweave.models import (
    Cell,
    Confidence,
    Document,
    HeaderFooterBlock,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Locator,
    Origin,
    Page,
    PageSignals,
    Paragraph,
    SourceInfo,
    Table,
)
from docweave.models.diagnostics import Category, Code

_BULLET_RE = re.compile(r"^\s*([\u2022\u2023\u25aa\u25cf\u2043\u2219\-\u2013\u2014])\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+(.*)$")
_HYPHEN_BREAK_RE = re.compile(r"(\w)[-\u2010]\s*$")
_RASTER_IMAGE_EXT = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "jpx": "image/jp2", "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff"}
#: Characters that are separators or artifacts rather than content. See
#: ``_join_lines`` for why the controls become a space instead of being deleted.
#: Zero-width characters are removed outright: they are invisible, so leaving
#: them in only makes the text harder to search and to compare, and a reader
#: never sees them.
_SPAN_NOISE_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|[\u00ad\u200b\u200c\u200d\u2060\ufeff]"
)
_SPAN_SPACE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
_MAX_IMAGES_PER_PAGE = 200
_HIST_BINS = 240


class PdfAdapter(Adapter):
    name = "pymupdf"
    formats = ("pdf",)
    requires = {"fitz": "pymupdf", "pymupdf": "pymupdf"}
    capabilities = (
        "native text with per-span geometry",
        "page-accurate provenance (page number + bounding box)",
        "mixed native/scanned document handling via per-page signals",
        "column-aware reading order",
        "ruled table recovery",
        "link and embedded image capture",
        "repeated header/footer detection",
    )
    limitations = (
        "no OCR on the default path: scanned pages are reported, not converted",
        "table recovery is geometry-based and can mis-merge dense layouts",
        "heading levels are inferred from relative font size, not from semantics",
        "math is not converted to LaTeX; equation images are preserved as assets",
    )
    cost = "moderate"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        fitz = _import_fitz()

        try:
            doc = fitz.open(filename=str(ctx.path))
        except Exception as exc:
            raise MalformedDocumentError(f"PDF could not be opened: {exc}") from exc

        with doc:
            if doc.needs_pass or getattr(doc, "is_encrypted", False):
                raise EncryptedDocumentError(
                    "This PDF is password protected. Supply an unlocked copy; "
                    "DocWeave does not attempt to brute-force or bypass the password."
                )
            if doc.page_count == 0:
                raise MalformedDocumentError("PDF contains zero pages.")

            if doc.page_count > ctx.config.max_pages:
                raise LimitExceededError(
                    f"Document has {doc.page_count} pages, above the configured limit of "
                    f"{ctx.config.max_pages}. Raise ConversionConfig.max_pages to convert it."
                )

            if getattr(doc, "is_repaired", False):
                # MuPDF silently rebuilds a damaged file. The conversion may then
                # "succeed" while content is missing, which is exactly the failure
                # mode we refuse to hide.
                ctx.warn(
                    Code.PARSER_FALLBACK,
                    "The PDF was damaged and had to be repaired before parsing. "
                    "Some content may be missing from the output.",
                    category=Category.INTERVENTION,
                    reason="The parser's repair pass reported a modified file structure.",
                    hint="Check the page list against the original; re-export the PDF if pages are missing.",
                    evidence={"pages_after_repair": doc.page_count},
                )

            pages: list[Page] = []
            header_candidates: dict[str, list[tuple[int, str]]] = {}
            footer_candidates: dict[str, list[tuple[int, str]]] = {}

            for index in range(doc.page_count):
                ctx.check_deadline(f"PDF page {index + 1}")
                page = doc[index]
                built = self._build_page(ctx, doc, page, index)
                pages.append(built)
                self._collect_repeating_bands(page, built, header_candidates, footer_candidates)

            header_texts, footer_texts = _repeating_bands(pages, header_candidates, footer_candidates)
            if header_texts or footer_texts:
                self._apply_running_bands(ctx, pages, header_texts, footer_texts)

            # Page-level judgements live in the validator, which owns the
            # diagnostic codes, so that every code has exactly one producer.

            doc_md = _pdf_metadata(doc)
            total_chars = sum(p.signals.char_count for p in pages)

        document = Document(
            adapter=self.name,
            container_kind="pages",
            pages=pages,
            metadata={
                **doc_md,
                "page_count": len(pages),
                "character_count": total_chars,
                "tables_recovered": sum(p.signals.table_count for p in pages),
                "images_found": sum(p.signals.image_count for p in pages),
            },
        )
        return ExtractionResult(
            document=document,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="pages",
        )

    # -- per page ----------------------------------------------------------

    def _build_page(self, ctx: ExtractionContext, doc, page, index: int) -> Page:
        page_number = index + 1
        rect = page.rect
        signals = PageSignals(rotation=int(getattr(page, "rotation", 0) or 0))

        try:
            text_dict = page.get_text("dict")
        except Exception as exc:
            ctx.warn(
                Code.PARSER_FALLBACK,
                f"Text extraction failed on page {page_number}: {exc}",
                category=Category.INTERVENTION,
                reason="The page could not be parsed by the native text engine.",
                hint="This page may need a repair pass; its content is missing from the output.",
                scope_kind="page",
                scope_index=page_number,
            )
            signals.image_count = _count_images(page)
            return Page(
                index=page_number,
                width=float(rect.width),
                height=float(rect.height),
                rotation=signals.rotation,
                signals=signals,
            )

        raw_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]
        raw_blocks = [b for b in raw_blocks if _block_text(b).strip()]

        table_regions: list[tuple[float, float, float, float]] = []
        recovered: list[tuple[tuple[float, float, float, float], Table]] = []
        if self._should_try_tables(ctx, page, raw_blocks):
            recovered = self._extract_tables(ctx, page, page_number)
            table_regions = [bbox for bbox, _ in recovered]

        kept = [b for b in raw_blocks if not _inside_any(_block_bbox(b), table_regions)]
        signals.table_count = len(table_regions)

        ordered, column_count = _reading_order(kept, float(rect.width), ctx.config.column_gutter_ratio)
        signals.column_count = column_count
        # The column count travels to the validator in ``signals``; the validator
        # owns the MULTI_COLUMN code so it is reported exactly once.

        page_id = f"p{page_number}"
        parent = page_id
        median_size = _median_span_size(raw_blocks)
        signals.font_size_median = median_size

        heading_sizes = [s for s, _ in _candidate_headings(raw_blocks, median_size)]
        signals.inferred_heading_sizes = sorted(set(round(s, 2) for s in heading_sizes), reverse=True)

        # Text blocks and recovered tables are merged back into one vertical order
        # so a table appears where it sits on the page rather than at the end.
        positioned: list[tuple[float, int, object]] = []
        for block in ordered:
            node = self._node_for_block(
                ctx, block, parent, page_number, median_size, heading_sizes
            )
            if node is not None:
                positioned.append((_block_bbox(block)[1], 0, node))
        for bbox, table in recovered:
            positioned.append((bbox[1], 1, table))
        positioned.sort(key=lambda item: (item[0], item[1]))
        blocks = [node for _, _, node in positioned]

        images, image_count = self._images(ctx, doc, page, page_number, parent)
        signals.image_count = image_count

        links = page.get_links() if ctx.config.include_links else []
        signals.link_count = len(links)
        if links:
            self._attach_links(blocks, links)

        page_text = " ".join(_block_text(b) for b in raw_blocks)
        stats = measure(page_text)
        signals.char_count = len(page_text.strip())
        signals.block_count = len(raw_blocks)
        signals.replacement_chars = stats.replacement
        signals.control_chars = stats.control
        page_area = max(float(rect.width) * float(rect.height), 1.0)
        signals.text_area_ratio = min(
            sum(_bbox_area(_block_bbox(b)) for b in raw_blocks) / page_area, 1.0
        )
        signals.media_area_ratio = min(sum(_bbox_area(b) for b in _image_rects(page, image_count)) / page_area, 1.0)

        return Page(
            index=page_number,
            width=float(rect.width),
            height=float(rect.height),
            rotation=signals.rotation,
            blocks=blocks,
            images=images,
            signals=signals,
        )

    def _node_for_block(
        self,
        ctx: ExtractionContext,
        block: dict,
        parent: str,
        page_number: int,
        median_size: float | None,
        heading_sizes: list[float],
    ):
        text = _block_text(block)
        cleaned, stats = clean_text(text, normalize=False)
        cleaned = _join_lines(block, cleaned)
        if not cleaned.strip():
            return None

        bbox = _block_bbox(block)
        locator = Locator(kind="page", page=page_number, bbox=bbox)
        warnings: list[str] = []
        if stats.replacement:
            warnings.append("replacement-characters")

        size = _block_font_size(block)
        # Font size is kept as metadata: the renderer ignores it, but the
        # running-header detector and downstream chunkers use it.
        font_meta: dict = {"font_size": size} if size else {}

        if _is_heading(cleaned, size, median_size, heading_sizes):
            level = _heading_level(size, heading_sizes) or 1
            node = Heading(
                id=ctx.ids.child(parent, "h"),
                level=level,
                text=cleaned.strip(),
                warnings=warnings,
                meta=dict(font_meta),
                sources=[
                    _src(locator, Origin.INFERRED, Confidence.MEDIUM,
                         note=f"Heading level inferred from font size {size} vs body median {median_size}.")
                ],
            )
            return node

        bullet = _BULLET_RE.match(cleaned) or _ORDERED_RE.match(cleaned)
        if bullet and len(cleaned.strip()) < 4000:
            ordered = bool(_ORDERED_RE.match(cleaned))
            # Both patterns are ``marker, text``; group(2) is the item content.
            item_text = bullet.group(2).strip() or cleaned.strip()
            ordinal = int(_ORDERED_RE.match(cleaned).group(1)) if ordered else None
            return ListBlock(
                id=ctx.ids.child(parent, "list"),
                ordered=ordered,
                items=[
                    ListItem(
                        id=ctx.ids.child(parent, "li"),
                        text=item_text.strip(),
                        ordinal=ordinal,
                        warnings=warnings,
                        sources=[
                            _src(locator, Origin.INFERRED, Confidence.MEDIUM,
                                 note="List membership inferred from a leading marker glyph.")
                        ],
                    )
                ],
                warnings=warnings,
                meta=dict(font_meta),
                sources=[_src(locator, Origin.OBSERVED, Confidence.HIGH)],
            )

        return Paragraph(
            id=ctx.ids.child(parent, "b"),
            text=cleaned.strip(),
            warnings=warnings,
            meta=dict(font_meta),
            sources=[_src(locator, Origin.OBSERVED, Confidence.HIGH)],
        )

    # -- tables ------------------------------------------------------------

    def _should_try_tables(self, ctx: ExtractionContext, page, blocks: list[dict]) -> bool:
        mode = ctx.config.pdf_tables
        if mode == "never":
            return False
        if mode == "always":
            return True
        return _looks_like_ruled_table(page, blocks)

    def _extract_tables(
        self, ctx: ExtractionContext, page, page_number: int
    ) -> list[tuple[tuple[float, float, float, float], Table]]:
        ctx.check_deadline(f"table detection on page {page_number}")
        try:
            finder = page.find_tables()
            tables = list(finder)
        except Exception as exc:
            ctx.warn(
                Code.PARSER_FALLBACK,
                f"Table detection failed on page {page_number}: {exc}",
                category=Category.HEURISTIC,
                reason="Geometry-based table detection raised an error.",
                scope_kind="page",
                scope_index=page_number,
            )
            return []

        results: list[tuple[tuple[float, float, float, float], Table]] = []
        for table in tables:
            try:
                extracted = table.extract()
            except Exception:
                continue
            rows = _normalize_rows(extracted)
            if len(rows) < 2:
                continue
            raw_bbox = getattr(table, "bbox", None)
            if not raw_bbox:
                continue
            bbox = tuple(float(v) for v in raw_bbox)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            results.append((bbox, _table_node(ctx.ids, self.name, page_number, bbox, rows)))
        return results

    # -- images ------------------------------------------------------------

    def _images(
        self, ctx: ExtractionContext, doc, page, page_number: int, parent: str
    ) -> tuple[list[ImageBlock], int]:
        if not ctx.config.include_images:
            return [], _count_images(page)
        count = _count_images(page)
        out: list[ImageBlock] = []
        seen_xrefs: set[int] = set()
        try:
            images = page.get_images(full=True)
        except Exception:
            return [], count
        for entry in images[:_MAX_IMAGES_PER_PAGE]:
            xref = int(entry[0])
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            rects = []
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            bbox = tuple(float(v) for v in rects[0]) if rects else None
            media_type = None
            blob = None
            if ctx.config.extract_assets or ctx.config.assets_dir is not None:
                try:
                    info = doc.extract_image(xref)
                    blob = info.get("image")
                    media_type = _RASTER_IMAGE_EXT.get(info.get("ext", ""), f"image/{info.get('ext','bin')}")
                except Exception:
                    blob = None
            asset_id = ctx.ids.child(parent, "asset")
            asset_ref = ctx.assets.add(
                blob if ctx.config.extract_assets else None,
                media_type=media_type,
                extension=None if media_type else None,
                locator=Locator(kind="page", page=page_number, bbox=bbox),
                asset_id=asset_id,
            )
            out.append(
                ImageBlock(
                    id=ctx.ids.child(parent, "img"),
                    text="",
                    asset_id=asset_ref.asset_id,
                    media_type=media_type,
                    width=int(bbox[2] - bbox[0]) if bbox else None,
                    height=int(bbox[3] - bbox[1]) if bbox else None,
                    sources=[
                        _src(
                            Locator(kind="page", page=page_number, bbox=bbox),
                            Origin.OBSERVED,
                            Confidence.HIGH,
                        )
                    ],
                )
            )
        if len(images) > _MAX_IMAGES_PER_PAGE:
            ctx.warn(
                Code.LIMIT_EXCEEDED,
                f"Page {page_number} embeds {len(images)} images; only the first "
                f"{_MAX_IMAGES_PER_PAGE} were recorded.",
                category=Category.INTERVENTION,
                reason="Per-page image cap reached to bound memory.",
                scope_kind="page",
                scope_index=page_number,
                evidence={"image_count": len(images)},
            )
        return out, count

    def _attach_links(self, blocks: list, links: list) -> None:
        for link in links:
            uri = link.get("uri")
            if not uri:
                continue
            origin = link.get("from")
            if origin is None:
                continue
            rect = (float(origin.x0), float(origin.y0), float(origin.x1), float(origin.y1))
            for block in blocks:
                for locator in [s.locator for s in block.sources]:
                    if locator.bbox and _bbox_overlaps(locator.bbox, rect, min_ratio=0.4):
                        block.meta.setdefault("links", []).append(uri)
                        break

    # -- running headers / footers -----------------------------------------

    def _collect_repeating_bands(
        self, page, built: Page, headers: dict, footers: dict
    ) -> None:
        rect = page.rect
        if not rect.height:
            return
        top_limit = float(rect.height) * 0.08
        bottom_limit = float(rect.height) * 0.92
        median = built.signals.font_size_median
        for block in built.blocks:
            locators = [s.locator for s in block.sources if s.locator.bbox]
            if not locators:
                continue
            bbox = locators[0].bbox
            assert bbox is not None
            text = block.text_of().strip()
            if not text:
                continue
            # Running furniture is set smaller than body text. Without this
            # guard, a document title repeated at the top of every page would be
            # mistaken for a header and silently deleted from every page.
            size = block.meta.get("font_size")
            if median and isinstance(size, (int, float)) and size > median * 1.1:
                continue
            if bbox[1] <= top_limit:
                headers.setdefault(text, []).append((built.index, bbox))
            elif bbox[3] >= bottom_limit:
                footers.setdefault(text, []).append((built.index, bbox))

    def _apply_running_bands(
        self, ctx: ExtractionContext, pages: list[Page], headers: set[str], footers: set[str]
    ) -> None:
        for page in pages:
            kept = []
            for block in page.blocks:
                text = block.text_of().strip()
                role = "header" if text in headers else ("footer" if text in footers else None)
                if role is None:
                    kept.append(block)
                    continue
                if ctx.config.include_headers_footers:
                    kept.append(
                        HeaderFooterBlock(
                            id=block.id,
                            text=text,
                            role=role,
                            sources=block.sources,
                            warnings=list(block.warnings),
                        )
                    )
            page.blocks = kept


# --------------------------------------------------------------------------
# Geometry and text helpers
# --------------------------------------------------------------------------


def _import_fitz():
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            from docweave.exceptions import DependencyMissingError

            raise DependencyMissingError("pymupdf", "PDF conversion", "pdf") from exc
    _quiet_pymupdf(fitz)
    return fitz


_MESSAGES_SILENCED = False


def _quiet_pymupdf(fitz) -> None:
    """Keep PyMuPDF's advisory messages off stdout.

    MuPDF writes hints like "Consider using the pymupdf_layout package" straight
    to stdout, which would corrupt ``--json-output`` and piped Markdown. Our own
    diagnostics already cover anything worth reporting.
    """
    global _MESSAGES_SILENCED
    if _MESSAGES_SILENCED:
        return
    try:
        # ``stream`` must be a writable file-like object; a sink keeps the
        # message off stdout entirely.
        fitz.set_messages(stream=_NullSink())
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
        # PyMuPDF also prints a one-off upsell to stdout from find_tables(),
        # bypassing set_messages. It has a dedicated opt-out.
        no_recommend = getattr(fitz, "no_recommend_layout", None)
        if callable(no_recommend):
            no_recommend()
        _MESSAGES_SILENCED = True
    except Exception:  # pragma: no cover - older PyMuPDF builds
        pass


class _NullSink:
    """Write-only sink that discards PyMuPDF advisory messages."""

    def write(self, _data: str) -> int:
        return 0

    def flush(self) -> None:
        return None

    def close(self) -> None:  # pragma: no cover - PyMuPDF never closes it
        return None


def _pdf_metadata(doc) -> dict:
    try:
        meta = dict(doc.metadata or {})
    except Exception:
        meta = {}
    cleaned = {k: v for k, v in meta.items() if v not in (None, "")}
    cleaned.pop("encryption", None)
    return cleaned


def _block_text(block: dict) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return "".join(parts)


def _block_bbox(block: dict) -> tuple[float, float, float, float]:
    bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in bbox)  # type: ignore[return-value]


def _bbox_area(bbox: Iterable[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_overlaps(a, b, min_ratio: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    if not ox or not oy:
        return False
    area = ox * oy
    ref = min(_bbox_area(a), _bbox_area(b))
    return ref > 0 and area / ref >= min_ratio


def _inside_any(bbox, regions) -> bool:
    for region in regions:
        if _bbox_overlaps(bbox, region, min_ratio=0.55):
            return True
    return False


def _src(locator: Locator, origin: Origin, confidence: Confidence, note: str | None = None):
    from docweave.models import SourceRef

    return SourceRef(
        adapter="pymupdf", origin=origin, confidence=confidence, locator=locator, note=note
    )


def _count_images(page) -> int:
    try:
        return len(page.get_images(full=True))
    except Exception:
        return 0


def _image_rects(page, limit: int) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    try:
        entries = page.get_images(full=True)
    except Exception:
        return rects
    for entry in entries[:limit]:
        try:
            for r in page.get_image_rects(int(entry[0])):
                rects.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
        except Exception:
            continue
    return rects


def _join_lines(block: dict, text: str) -> str:
    """Re-join the characters of a text block with sensible line breaks.

    PyMuPDF concatenates spans without separators, which destroys word spaces and
    hyphenated line breaks. We rebuild them from span geometry.

    That rebuild reads the raw spans rather than the cleaned text, so this is
    also where characters that never belong in output are caught. A real 37-page
    PDF put ``\\x08`` between a contents entry and its page number; it arrived in
    Markdown as an invisible control character. Controls become a space rather
    than being deleted, because here they act as the separator PyMuPDF dropped,
    and deleting would glue ``Introduction`` to its page number. A soft hyphen
    is a discretionary break point, so it is removed rather than spaced.
    """
    lines = block.get("lines", [])
    if not lines:
        return text
    rendered: list[str] = []
    heights = [
        float(line.get("bbox", (0, 0, 0, 0))[3]) - float(line.get("bbox", (0, 0, 0, 0))[1])
        for line in lines
    ]
    heights = [h for h in heights if h > 0]
    median_h = statistics.median(heights) if heights else 0.0

    previous_bottom: float | None = None
    for line in lines:
        spans = line.get("spans", [])
        if not spans:
            continue
        span_text = "".join(s.get("text", "") for s in spans)
        if not span_text:
            continue
        line_bbox = line.get("bbox", (0, 0, 0, 0))
        top = float(line_bbox[1])
        bottom = float(line_bbox[3])
        gap = (top - previous_bottom) if previous_bottom is not None else 0.0
        previous_bottom = bottom
        if rendered:
            last = rendered[-1]
            if _HYPHEN_BREAK_RE.search(last) and span_text[:1].islower():
                rendered[-1] = _HYPHEN_BREAK_RE.sub(r"\1", last) + span_text
                continue
            if median_h and gap > median_h * 1.35:
                rendered.append("\n" + span_text)
            else:
                rendered.append(" " + span_text)
        else:
            rendered.append(span_text)
    joined = "".join(rendered).strip()
    # Controls acted as the separator PyMuPDF dropped, so they become a space;
    # zero-width characters are simply removed. Runs of spaces then collapse:
    # a double space is not content either, and it is one of the artifacts a
    # reader notices immediately.
    joined = _ZERO_WIDTH_RE.sub("", joined)
    return re.sub(r" {2,}", " ", _SPAN_SPACE_RE.sub(" ", joined))


def _span_sizes(block: dict) -> list[float]:
    sizes: list[float] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("text", "").strip():
                size = span.get("size")
                if isinstance(size, (int, float)) and size > 0:
                    sizes.append(float(size))
    return sizes


def _median_span_size(blocks: list[dict]) -> float | None:
    sizes: list[float] = []
    for block in blocks:
        sizes.extend(_span_sizes(block))
    return statistics.median(sizes) if sizes else None


def _block_font_size(block: dict) -> float | None:
    sizes = _span_sizes(block)
    return statistics.median(sizes) if sizes else None


def _candidate_headings(blocks: list[dict], median_size: float | None) -> list[tuple[float, str]]:
    if median_size is None:
        return []
    out: list[tuple[float, str]] = []
    for block in blocks:
        size = _block_font_size(block)
        text = _block_text(block).strip()
        if size is None or not text or len(text) > 240:
            continue
        if size >= median_size * 1.15 and not text.endswith((".", ",", ";", ":")):
            out.append((size, text))
    return out


def _heading_level(size: float | None, sizes: Sequence[float]) -> int | None:
    if size is None or not sizes:
        return None
    distinct = sorted(set(round(s, 2) for s in sizes), reverse=True)
    for level, candidate in enumerate(distinct[:6], start=1):
        if abs(candidate - size) < 0.6:
            return level
    return min(len(distinct), 6)


def _is_heading(text: str, size: float | None, median_size: float | None, heading_sizes: list[float]) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 240:
        return False
    if size is None or median_size is None:
        return False
    if size < median_size * 1.15:
        return False
    if stripped.endswith((".", ",", ";", ":", "。", "，")):
        return False
    if _BULLET_RE.match(stripped) or _ORDERED_RE.match(stripped):
        return False
    words = stripped.split()
    if len(words) > 20:
        return False
    return any(abs(size - s) < 0.6 for s in heading_sizes)


def _looks_like_ruled_table(page, blocks: list[dict]) -> bool:
    """Cheap pre-check: a page must contain several long thin vector lines."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return False
    long_lines = 0
    for drawing in drawings:
        for item in drawing.get("items", []):
            if item[0] == "l":
                p0, p1 = item[1], item[2]
                length = abs(p1.x - p0.x) + abs(p1.y - p0.y)
                if length > 60:
                    long_lines += 1
            elif item[0] == "re":
                rect = item[1]
                w = float(rect.width)
                h = float(rect.height)
                if (w > 60 and h <= 3) or (h > 60 and w <= 3):
                    long_lines += 1
    return long_lines >= 4


def _normalize_rows(extracted) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for raw in extracted:
        if raw is None:
            continue
        rows.append([None if c is None else str(c) for c in raw])
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    return [r + [None] * (width - len(r)) for r in rows]


def _table_node(ids, adapter: str, page_number: int, bbox, rows: list[list[str | None]]) -> Table:
    cells_rows: list[list[Cell]] = []
    for r, row in enumerate(rows):
        cells: list[Cell] = []
        for c, value in enumerate(row):
            text = (value or "").strip()
            cells.append(
                Cell(
                    id=ids.child(None, "c"),
                    text=text,
                    row=r,
                    column=c,
                    is_header=(r == 0),
                    locator=Locator(kind="page", page=page_number, bbox=bbox),
                )
            )
        cells_rows.append(cells)
    return Table(
        id=ids.child(None, "table"),
        text="",
        rows=cells_rows,
        header_rows=1 if rows else 0,
        structure="grid",
        sources=[
            _src(
                Locator(kind="page", page=page_number, bbox=bbox),
                Origin.RECONSTRUCTED,
                Confidence.MEDIUM,
                note="Table recovered from ruling-line geometry; spans are not represented.",
            )
        ],
    )


def _reading_order(blocks: list[dict], page_width: float, gutter_ratio: float):
    """Order blocks, splitting into columns when a wide vertical gutter exists."""
    if len(blocks) < 3 or page_width <= 0:
        return sorted(blocks, key=lambda b: (_block_bbox(b)[1], _block_bbox(b)[0])), 1

    boxes = [_block_bbox(b) for b in blocks]
    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    width = right - left
    if width <= 0:
        return blocks, 1

    bins = [0] * _HIST_BINS
    for x0, _, x1, _ in boxes:
        i0 = max(0, min(_HIST_BINS - 1, int((x0 - left) / width * _HIST_BINS)))
        i1 = max(0, min(_HIST_BINS - 1, int((x1 - left) / width * _HIST_BINS)))
        for i in range(i0, i1 + 1):
            bins[i] += 1

    min_gutter = max(1, int(gutter_ratio * _HIST_BINS))
    gutters: list[tuple[int, int]] = []
    i = 0
    while i < _HIST_BINS:
        if bins[i] == 0:
            j = i
            while j < _HIST_BINS and bins[j] == 0:
                j += 1
            # Ignore leading/trailing whitespace: only interior gutters split columns.
            if i > 0 and j < _HIST_BINS and (j - i) >= min_gutter:
                gutters.append((i, j))
            i = j
        else:
            i += 1

    if not gutters:
        return sorted(blocks, key=lambda b: (_block_bbox(b)[1], _block_bbox(b)[0])), 1

    # Keep the most significant gutters (largest), then rebuild column indices.
    gutters.sort(key=lambda g: g[1] - g[0], reverse=True)
    gutters = sorted(gutters[:2])

    def column_of(bbox) -> int:
        center = (bbox[0] + bbox[2]) / 2 - left
        idx = int(center / width * _HIST_BINS)
        col = 0
        for start, _ in gutters:
            if idx >= start:
                col += 1
        return col

    decorated = [(column_of(_block_bbox(b)), b) for b in blocks]
    decorated.sort(key=lambda pair: (pair[0], _block_bbox(pair[1])[1], _block_bbox(pair[1])[0]))
    return [pair[1] for pair in decorated], len(gutters) + 1


def _repeating_bands(
    pages: list[Page], headers: dict, footers: dict
) -> tuple[set[str], set[str]]:
    """Pick text that repeats across pages in the top or bottom band."""
    if len(pages) < 2:
        return set(), set()
    threshold = max(2, int(len(pages) * 0.6))

    def pick(candidates: dict) -> set[str]:
        return {
            text
            for text, hits in candidates.items()
            if len({page_index for page_index, _ in hits}) >= threshold
        }

    return pick(headers), pick(footers)
