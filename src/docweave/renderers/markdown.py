"""Markdown renderer.

Markdown is the most lossy target, so the renderer is explicit about what it
cannot express: complex tables fall back to HTML (which keeps spans), images that
were not extracted become visible placeholders rather than silent omissions, and
anything with an unresolved warning is marked in a trailer block.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from docweave.config import ConversionConfig
from docweave.models import (
    AssetRef,
    Cell,
    CodeBlock,
    DiagnosticReport,
    Document,
    Equation,
    HeaderFooterBlock,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Paragraph,
    Quote,
    Section,
    Table,
)

_PIPE_RE = re.compile(r"\|")
_BACKSLASH = "\\"
_PIPE_ESCAPE = "\\|"


class MarkdownRenderer:
    def __init__(
        self,
        config: ConversionConfig | None = None,
        assets: list[AssetRef] | None = None,
    ) -> None:
        self.config = config or ConversionConfig()
        self.assets = {a.asset_id: a for a in (assets or [])}

    def _asset(self, asset_id: str | None) -> AssetRef | None:
        return self.assets.get(asset_id) if asset_id else None

    def render(
        self,
        document: Document,
        diagnostics: DiagnosticReport | None = None,
        assets: list[AssetRef] | None = None,
    ) -> str:
        if assets:
            self.assets = {a.asset_id: a for a in assets}
        renderer = _Ctx(self.config, self.assets)
        parts: list[str] = []
        front = renderer.front_matter(document)
        if front:
            parts.append(front)

        if document.container_kind == "pages":
            parts.extend(renderer.render_pages(document))
        elif document.container_kind == "slides":
            parts.extend(renderer.render_slides(document))
        elif document.container_kind == "sheets":
            parts.extend(renderer.render_sheets(document))
        else:
            parts.extend(renderer.blocks(document.body))

        if self.config.markdown_include_diagnostics and diagnostics and diagnostics.diagnostics:
            parts.append(renderer.render_diagnostics(diagnostics))

        return _join(parts) + "\n"

    # -- document level ----------------------------------------------------

    def render_pages(self, document: Document) -> list[str]:
        out: list[str] = []
        for page in document.pages:
            body = self.blocks(page.blocks)
            images = [self.image(i) for i in page.images]
            combined = _join(body + [i for i in images if i])
            if not combined:
                continue
            out.append(f"<!-- page {page.index} -->")
            out.append(combined)
        return out

    def render_slides(self, document: Document) -> list[str]:
        out: list[str] = []
        for slide in document.slides:
            out.append(f"<!-- slide {slide.index} -->")
            has_title_node = any(isinstance(b, Heading) for b in slide.blocks)
            if slide.title and not has_title_node:
                out.append(f"# {slide.title}")
            out.extend(self.blocks(slide.blocks))
            out.extend(self.image(i) for i in slide.images)
            if slide.notes:
                out.append("> **Speaker notes**")
                out.extend(f"> {line}" for line in _notes_text(slide.notes))
        return out

    def render_sheets(self, document: Document) -> list[str]:
        out: list[str] = []
        for sheet in document.sheets:
            marker = " (hidden)" if sheet.hidden else ""
            out.append(f"## Worksheet: {sheet.name}{marker}")
            if sheet.table is None:
                out.append("_No cell data._")
                continue
            if sheet.truncated:
                out.append(
                    f"> Truncated: only part of this worksheet was converted "
                    f"({sheet.table.meta.get('converted_rows')} rows)."
                )
            if sheet.formula_count:
                out.append(
                    f"> {sheet.formula_count} formula cell(s). Formulas are preserved; "
                    "cached results are shown when the file contains them."
                )
            out.append(self.table(sheet.table))
        return out

    def render_diagnostics(self, report: DiagnosticReport) -> str:
        lines = ["---", "", "<!-- docweave:diagnostics -->", ""]
        lines.append(f"**Extraction report** — {report.summary()}")
        lines.append("")
        for d in report.sorted()[:50]:
            lines.append(f"- `{d.code}` ({d.category.value}) {d.location()}: {d.message}")
        remaining = len(report.diagnostics) - 50
        if remaining > 0:
            lines.append(f"- … and {remaining} more")
        return "\n".join(lines)

    # -- nodes -------------------------------------------------------------

    def front_matter(self, document: Document) -> str:
        """Optional provenance header.

        Off by default: the JSON envelope is the canonical place for source
        metadata, and an LLM-facing Markdown file does not need a metadata table.
        """
        if not self.config.markdown_include_front_matter:
            return ""
        meta = document.metadata
        rows = [
            ("source", document.source.name),
            ("format", document.source.detected_format),
            ("adapter", document.adapter),
        ]
        for key in ("page_count", "slide_count", "sheet_count", "character_count", "title", "author"):
            if meta.get(key) not in (None, ""):
                rows.append((key, meta[key]))
        if not rows:
            return ""
        lines = ["| field | value |", "| --- | --- |"]
        for key, value in rows:
            lines.append(f"| {key} | {_cell(str(value))} |")
        return "\n".join(lines)

    def blocks(self, nodes: Iterable) -> list[str]:
        out: list[str] = []
        for node in nodes:
            rendered = self.node(node)
            if not rendered:
                continue
            if self.config.emit_stable_ids and not isinstance(node, (ListBlock, ListItem)):
                # Stable anchors let a downstream chunker map slices of this file
                # back to the exact node they came from.
                rendered = f"<!-- {node.id} -->\n{rendered}"
            out.append(rendered)
        return out

    def node(self, node) -> str:
        # Nodes sourced from Markdown already contain Markdown. Escaping them a
        # second time would surface literal backslashes to the reader.
        verbatim = bool(node.meta.get("verbatim"))

        if isinstance(node, Heading):
            return f"{'#' * _clamp(node.level)} {_inline(node.text, verbatim)}"
        if isinstance(node, Paragraph):
            return _inline(node.text, verbatim)
        if isinstance(node, Quote):
            return "\n".join(f"> {line}" for line in node.text.split("\n"))
        if isinstance(node, CodeBlock):
            lang = node.language or ""
            return f"```{lang}\n{node.text}\n```"
        if isinstance(node, ListBlock):
            return self.list_block(node)
        if isinstance(node, ListItem):
            return self.list_block(ListBlock(ordered=node.ordinal is not None, items=[node]))
        if isinstance(node, Table):
            return self.table(node)
        if isinstance(node, ImageBlock):
            return self.image(node)
        if isinstance(node, Equation):
            return f"$$ {_latex_or_placeholder(node)} $$"
        if isinstance(node, HeaderFooterBlock):
            return f"<!-- {node.role}: {_escape_inline(node.text)} -->"
        if isinstance(node, Section):
            parts = []
            if node.title:
                parts.append(f"{'#' * _clamp(node.level or 1)} {_escape_inline(node.title)}")
            parts.extend(self.blocks(node.children))
            return _join(parts)
        text = getattr(node, "text", "")
        return _inline(text, verbatim) if text else ""

    def list_block(self, node: ListBlock) -> str:
        lines: list[str] = []
        verbatim = any(item.meta.get("verbatim") for item in node.items)
        for index, item in enumerate(node.items, start=1):
            marker = f"{item.ordinal or index}." if node.ordered else "-"
            body = _inline(item.text, bool(item.meta.get("verbatim")) or verbatim)
            lines.append(f"{marker} {body}")
        return "\n".join(lines)

    def table(self, table: Table) -> str:
        rows = [row for row in table.rows if any(c.text.strip() for c in row)]
        if not rows:
            return "_(empty table)_"
        mode = self.config.table_rendering
        if mode == "markdown":
            if table.sparse:
                # A sparse sheet's rows do not start at column A. Padding them
                # into a grid would put values under the wrong headings, so we
                # emit cell references instead of a misleading alignment.
                return self._table_list(table, rows)
            if not table.is_rectangular():
                # Ragged rows lose column identity in a Markdown grid.
                mode = "html"

        if mode == "html":
            return self._table_html(table, rows)
        if mode == "list":
            return self._table_list(table, rows)
        if mode == "both":
            return self._table_html(table, rows) + "\n\n" + self._table_markdown(table, rows)
        return self._table_markdown(table, rows)

    def _table_markdown(self, table: Table, rows: list[list[Cell]]) -> str:
        header_rows = max(0, min(table.header_rows, len(rows)))
        body = rows[header_rows:]
        cols = max((len(r) for r in rows), default=0)
        if not cols:
            return "_(empty table)_"

        if header_rows:
            head_cells = [c.text for c in rows[0]]
            lines = ["| " + " | ".join(_cell(c) for c in _pad(head_cells, cols)) + " |"]
            lines.append("| " + " | ".join("---" for _ in range(cols)) + " |")
            body = rows[header_rows:]
        else:
            # Markdown requires a header row. An empty one is used rather than
            # promoting real data, which would silently reclassify content.
            lines = ["| " + " | ".join("" for _ in range(cols)) + " |"]
            lines.append("| " + " | ".join("---" for _ in range(cols)) + " |")
            body = rows

        for row in body:
            cells = [c.text for c in row]
            lines.append("| " + " | ".join(_cell(c) for c in _pad(cells, cols)) + " |")
        return "\n".join(lines)

    def _table_html(self, table: Table, rows: list[list[Cell]]) -> str:
        parts = ["<table>"]
        header_rows = max(0, min(table.header_rows, len(rows)))
        for r, row in enumerate(rows):
            tag = "th" if r < header_rows else "td"
            parts.append("  <tr>")
            for cell in row:
                span = ""
                if cell.column_span > 1:
                    span += f' colspan="{cell.column_span}"'
                if cell.row_span > 1:
                    span += f' rowspan="{cell.row_span}"'
                parts.append(f"    <{tag}{span}>{_escape_html(cell.text)}</{tag}>")
            parts.append("  </tr>")
        parts.append("</table>")
        return "\n".join(parts)

    def _table_list(self, table: Table, rows: list[list[Cell]]) -> str:
        lines: list[str] = []
        for row in rows:
            if not any(c.text.strip() for c in row):
                continue
            pairs = []
            for cell in row:
                reference = cell.locator.cell if cell.locator else None
                label = reference if reference else f"col {cell.column + 1}"
                pairs.append(f"{label}: {_quote_ambiguous(_cell(cell.text, escape_pipe=False))}")
            lines.append("- " + "; ".join(pairs))
        return "\n".join(lines)

    def image(self, image: ImageBlock) -> str:
        mode = self.config.image_rendering
        if mode == "omit":
            return ""
        asset = self._asset(image.asset_id)
        # Prefer a real description. Failing that, the asset id is more useful
        # than a bare "image": it is traceable back to the source document.
        label = image.alt or image.media_type or image.asset_id or "image"
        if mode == "link" and asset and asset.path:
            return f"![{label}]({asset.path})"
        if mode == "path" and asset and asset.path:
            return f"[image: {asset.path}]"
        placeholder = f"![{label}](image://{image.asset_id or image.id})"
        note = "" if asset and asset.extracted else " _(asset not extracted; run with --extract-assets)_"
        return f"{placeholder}{note}"


class _Ctx:
    """Page/slide/sheet rendering that reuses the block-level Markdown rules."""

    def __init__(self, config: ConversionConfig, assets: dict) -> None:
        self.config = config
        self.assets = assets
        self._md = MarkdownRenderer(config, list(assets.values()))

    def blocks(self, nodes) -> list[str]:
        return self._md.blocks(nodes)

    def front_matter(self, document: Document) -> str:
        return self._md.front_matter(document)

    def render_pages(self, document: Document) -> list[str]:
        return self._md.render_pages(document)

    def render_slides(self, document: Document) -> list[str]:
        return self._md.render_slides(document)

    def render_sheets(self, document: Document) -> list[str]:
        return self._md.render_sheets(document)

    def render_diagnostics(self, report: DiagnosticReport) -> str:
        return self._md.render_diagnostics(report)

    def image(self, image: ImageBlock) -> str:
        return self._md.image(image)


def _notes_text(nodes) -> list[str]:
    for node in nodes:
        text = getattr(node, "text", "")
        if text:
            return text.split("\n")
    return []


def _clamp(level: int) -> int:
    return max(1, min(int(level or 1), 6))


def _pad(values: list[str], width: int) -> list[str]:
    return values + [""] * max(0, width - len(values))


def _cell(value: str, *, escape_pipe: bool = True) -> str:
    """Clean one cell's text for rendering.

    ``escape_pipe`` matters: inside a grid table a pipe would end the cell, but
    in the list format cells are separated by ``; `` and there is no pipe to
    escape. Escaping there corrupts the value for every reader -- ``0|`` came
    out as ``0\\|``, which no consumer reads back as the number it was.
    """
    text = value.replace("\n", " ")
    if escape_pipe:
        text = text.replace("|", _PIPE_ESCAPE)
    return text.strip() or " "


#: The list format separates cells with ``"; "``, so a value that itself contains
#: a separator followed by a cell label is indistinguishable from another cell.
#: ``A1`` holding ``harbour; B1: docks`` would otherwise render exactly like a
#: two-cell row and quietly change the cell count.
_AMBIGUOUS_SEPARATOR_RE = re.compile(r";\s*(?:[A-Z]{1,3}\d+\s*:|col \d+\s*:)")


def _quote_ambiguous(value: str) -> str:
    """Wrap a cell value that would otherwise read as extra cells.

    Quoting is applied only where it is needed, so ordinary values with a
    semicolon in them (``alpha; beta``) stay readable. Embedded quotes are
    doubled, as in CSV.
    """
    if not _AMBIGUOUS_SEPARATOR_RE.search(value):
        return value
    return '"' + value.replace('"', '""') + '"'


def _inline(text: str, verbatim: bool = False) -> str:
    return text if verbatim else _escape_inline(text)


def _escape_inline(text: str) -> str:
    """Escape only what would change Markdown structure.

    Deliberately conservative: over-escaping mangles prose (``--`` becomes
    ``\\-\\-``), so only genuinely structural characters are touched.
    """
    out: list[str] = []
    for ch in text:
        if ch in ("`", "*", "[", "]"):
            out.append(_BACKSLASH + ch)
        else:
            out.append(ch)
    return "".join(out)


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _latex_or_placeholder(node: Equation) -> str:
    if node.latex:
        return node.latex
    return "equation not recovered"


def _join(parts: Iterable[str]) -> str:
    cleaned = [p.rstrip() for p in parts if p is not None and p.strip()]
    return "\n\n".join(cleaned)
