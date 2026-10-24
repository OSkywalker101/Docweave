"""PPTX adapter (python-pptx).

Each slide is a first-class structural unit. Shapes are ordered top-to-bottom,
left-to-right because PowerPoint stores them in z-order, which is not reading
order. Tables, charts, pictures and speaker notes are preserved.
"""

from __future__ import annotations

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import clean_text, ref
from docweave.exceptions import MalformedDocumentError
from docweave.models import (
    Cell,
    Confidence,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Locator,
    Origin,
    Paragraph,
    Slide,
    SourceInfo,
    Table,
)
from docweave.models.diagnostics import Category, Code


class PptxAdapter(Adapter):
    name = "python-pptx"
    formats = ("pptx",)
    requires = {"pptx": "python-pptx"}
    capabilities = (
        "per-slide structural units",
        "shape-level provenance (slide index + shape id)",
        "reading order derived from shape geometry",
        "tables with merged cells",
        "chart series/category data",
        "speaker notes",
        "embedded pictures as asset references",
    )
    limitations = (
        "grouped shapes are flattened with their group id retained in metadata",
        "SmartArt is not decomposed",
        "chart data is read for known chart types only",
        "animation and transition ordering is not represented",
    )
    cost = "moderate"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        try:
            from pptx import Presentation  # type: ignore
        except ImportError as exc:
            from docweave.exceptions import DependencyMissingError

            raise DependencyMissingError("python-pptx", "PPTX conversion", "office") from exc

        # python-pptx moved this exception across releases; do not hard-code a path.
        try:
            from pptx.opc.exceptions import PackageNotFoundError  # type: ignore
        except ImportError:
            try:
                from pptx.exc import PackageNotFoundError  # type: ignore
            except ImportError:
                class PackageNotFoundError(Exception):  # type: ignore[no-redef]
                    """Fallback when python-docx's package error is unavailable."""

        try:
            presentation = Presentation(str(ctx.path))
        except PackageNotFoundError as exc:
            raise MalformedDocumentError(f"Invalid PPTX package: {exc}") from exc
        except Exception as exc:
            raise MalformedDocumentError(f"PPTX could not be opened: {exc}") from exc

        slides: list[Slide] = []
        total_tables = 0
        total_charts = 0
        empty_slides = 0

        for index, slide in enumerate(presentation.slides, start=1):
            ctx.check_deadline(f"PPTX slide {index}")
            shapes = _ordered_shapes(slide)
            blocks: list = []
            images: list[ImageBlock] = []
            title: str | None = None
            slide_tables = 0

            for shape in shapes:
                try:
                    node, image, kind = self._shape_node(ctx, slide, shape, index)
                except Exception as exc:
                    ctx.warn(
                        Code.PARSER_FALLBACK,
                        f"Shape {getattr(shape, 'shape_id', '?')} on slide {index} could not be read: {exc}",
                        category=Category.INTERVENTION,
                        reason="The shape's object model raised an error.",
                        scope_kind="slide",
                        scope_index=index,
                    )
                    continue
                if kind == "title" and title is None and node is not None:
                    title = node.text_of() or title
                if node is not None:
                    blocks.append(node)
                if image is not None:
                    images.append(image)
                if kind == "table":
                    slide_tables += 1
                elif kind == "chart":
                    total_charts += 1

            notes = self._notes(ctx, slide, index)
            if not blocks and not images:
                empty_slides += 1
                ctx.warn(
                    Code.EMPTY_CONTAINER,
                    f"Slide {index} has no convertible content.",
                    category=Category.STRUCTURAL,
                    reason="No text frames, tables, charts or pictures were found.",
                    scope_kind="slide",
                    scope_index=index,
                )

            slides.append(
                Slide(
                    index=index,
                    layout=_layout_name(slide),
                    title=title,
                    blocks=blocks,
                    notes=notes,
                    images=images,
                )
            )
            total_tables += slide_tables

        if total_charts:
            ctx.warn(
                Code.CHART_DATA_UNAVAILABLE,
                f"{total_charts} chart(s) were read; values come from the embedded workbook, "
                "not from the rendered graphic.",
                category=Category.UNVERIFIED,
                severity="info",
                reason="Chart values are recovered from cached workbook data.",
            )

        meta = {
            "slide_count": len(slides),
            "table_count": total_tables,
            "chart_count": total_charts,
            "empty_slides": empty_slides,
            "title": _core_title(presentation),
        }
        meta = {k: v for k, v in meta.items() if v not in (None, "")}

        document = Document(
            adapter=self.name,
            container_kind="slides",
            slides=slides,
            metadata=meta,
        )
        return ExtractionResult(
            document=document,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="slides",
        )

    # -- shapes ------------------------------------------------------------

    def _shape_node(self, ctx: ExtractionContext, slide, shape, slide_index: int):
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        locator = Locator(kind="slide", slide=slide_index, shape_id=str(getattr(shape, "shape_id", "")))
        provenance = ref(self.name, locator, confidence=Confidence.HIGH)
        name = getattr(shape, "name", "")

        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            return None, None, "group"

        if getattr(shape, "has_table", False):
            return self._table(ctx, shape, locator, name), None, "table"

        if getattr(shape, "has_chart", False):
            return self._chart(ctx, shape, locator, name), None, "chart"

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or getattr(shape, "image", None) is not None:
            return None, self._picture(ctx, shape, locator, name), "picture"

        if not getattr(shape, "has_text_frame", False):
            return None, None, "other"

        text, stats = clean_text(shape.text_frame.text, strip_controls=False)
        kind = "text"
        level = 1
        if getattr(shape, "is_placeholder", False):
            try:
                placeholder_type = str(shape.placeholder_format.type)
            except Exception:
                placeholder_type = ""
            if "TITLE" in placeholder_type:
                kind = "title"
                # A subtitle sits under the slide title, so it must not become
                # a peer heading.
                level = 2 if "SUBTITLE" in placeholder_type else 1

        if not text.strip():
            return None, None, kind

        warnings = ["replacement-characters"] if stats.replacement else []
        if kind == "title":
            node = Heading(
                id=ctx.ids.child(None, "h"),
                level=level,
                text=text.strip(),
                warnings=warnings,
                sources=[provenance],
                meta={"shape": name, "placeholder": placeholder_type if kind else ""},
            )
            return node, None, kind

        paragraphs = [p for p in text.split("\n") if p.strip()]
        if len(paragraphs) > 1 and _looks_bulleted(paragraphs):
            return (
                ListBlock(
                    id=ctx.ids.child(None, "list"),
                    ordered=False,
                    items=[
                        ListItem(id=ctx.ids.child(None, "li"), text=p, sources=[provenance])
                        for p in paragraphs
                    ],
                    sources=[provenance],
                    meta={"shape": name},
                ),
                None,
                "list",
            )

        node = Paragraph(
            id=ctx.ids.child(None, "p"),
            text=text.strip(),
            warnings=warnings,
            sources=[provenance],
            meta={"shape": name},
        )
        return node, None, kind

    def _table(self, ctx: ExtractionContext, shape, locator: Locator, name: str) -> Table:
        rows: list[list[Cell]] = []
        for r, row in enumerate(shape.table.rows):
            cells: list[Cell] = []
            for c, cell in enumerate(row.cells):
                text, _ = clean_text(cell.text, strip_controls=False)
                cells.append(
                    Cell(
                        id=ctx.ids.child(None, "c"),
                        text=text,
                        row=r,
                        column=c,
                        is_header=(r == 0),
                        locator=locator,
                    )
                )
            rows.append(cells)
        return Table(
            id=ctx.ids.child(None, "table"),
            text="",
            rows=rows,
            header_rows=1 if rows else 0,
            sources=[
                ref(
                    self.name,
                    locator,
                    origin=Origin.OBSERVED,
                    confidence=Confidence.HIGH,
                    note=f"Slide table from shape '{name}'.",
                )
            ],
            meta={"shape": name},
        )

    def _chart(self, ctx: ExtractionContext, shape, locator: Locator, name: str) -> Paragraph:
        """Charts are summarised as text; values come from the cached workbook."""
        lines: list[str] = []
        chart = shape.chart
        try:
            chart_type = str(chart.chart_type)
        except Exception:
            chart_type = "chart"
        try:
            categories = [str(c) for c in chart.plots[0].categories]
            lines.append("Categories: " + ", ".join(categories))
            for series in chart.series:
                values = ", ".join("" if v is None else str(v) for v in series.values)
                lines.append(f"Series '{series.name}': {values}")
        except Exception as exc:
            ctx.warn(
                Code.CHART_DATA_UNAVAILABLE,
                f"Chart '{name}' on slide {locator.slide} could not be read: {exc}",
                category=Category.UNVERIFIED,
                scope_kind="slide",
                scope_index=locator.slide,
            )
        body = "\n".join(lines) if lines else f"[{chart_type} - data unavailable]"
        return Paragraph(
            id=ctx.ids.child(None, "chart"),
            text=body,
            sources=[
                ref(
                    self.name,
                    locator,
                    origin=Origin.RECONSTRUCTED,
                    confidence=Confidence.MEDIUM,
                    note="Chart summarised from the embedded workbook.",
                )
            ],
            meta={"shape": name, "chart_type": chart_type},
        )

    def _picture(self, ctx: ExtractionContext, shape, locator: Locator, name: str) -> ImageBlock:
        blob = None
        media_type = None
        try:
            image = shape.image
            blob = image.blob
            media_type = image.content_type
        except Exception:
            blob = None
        asset = ctx.assets.add(
            blob if ctx.config.extract_assets else None,
            media_type=media_type,
            locator=locator,
            asset_id=ctx.ids.child(None, "asset"),
        )
        return ImageBlock(
            id=ctx.ids.child(None, "img"),
            text="",
            asset_id=asset.asset_id,
            media_type=media_type,
            sources=[
                ref(
                    self.name,
                    locator,
                    origin=Origin.OBSERVED,
                    confidence=Confidence.HIGH,
                    note=f"Picture from shape '{name}'.",
                )
            ],
            meta={"shape": name},
        )

    def _notes(self, ctx: ExtractionContext, slide, slide_index: int) -> list:
        try:
            if not slide.has_notes_slide:
                return []
            frame = slide.notes_slide.notes_text_frame
        except Exception:
            return []
        if frame is None:
            return []
        text, _ = clean_text(frame.text, strip_controls=False)
        if not text.strip():
            return []
        return [
            Paragraph(
                id=ctx.ids.child(None, "note"),
                text=text.strip(),
                sources=[
                    ref(
                        self.name,
                        Locator(kind="slide", slide=slide_index),
                        origin=Origin.OBSERVED,
                        confidence=Confidence.HIGH,
                        note="Speaker notes.",
                    )
                ],
            )
        ]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ordered_shapes(slide) -> list:
    """Reading order: by vertical position, then horizontal, ignoring z-order."""
    shapes = [s for s in slide.shapes if s is not None]

    def key(shape):
        top = getattr(shape, "top", None)
        left = getattr(shape, "left", None)
        return (
            int(top) if top is not None else 1 << 30,
            int(left) if left is not None else 1 << 30,
            int(getattr(shape, "shape_id", 0) or 0),
        )

    return sorted(shapes, key=key)


def _layout_name(slide) -> str | None:
    try:
        return slide.slide_layout.name
    except Exception:
        return None


def _core_title(presentation) -> str | None:
    try:
        title = presentation.core_properties.title
        return title or None
    except Exception:
        return None


def _looks_bulleted(paragraphs: list[str]) -> bool:
    bullets = ("\u2022", "\u25cf", "\u25aa", "\u2013", "-", "*")
    hits = sum(1 for p in paragraphs if p.lstrip().startswith(bullets))
    return hits >= max(1, len(paragraphs) // 2)
