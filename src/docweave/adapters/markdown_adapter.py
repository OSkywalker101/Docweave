"""Markdown adapter.

Markdown is DocWeave's most common *output*, so parsing it back into the IR is
not an afterthought: it makes the engine round-trippable and lets us validate
that a renderer is not silently dropping structure.
"""

from __future__ import annotations

import re

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import char_locator, clean_text, read_text, ref
from docweave.models import (
    Cell,
    CodeBlock,
    Confidence,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Origin,
    Paragraph,
    Quote,
    SourceInfo,
    Table,
)
from docweave.models.diagnostics import Category, Code

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")
_BULLET_RE = re.compile(r"^([-*+])\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\d{1,9})[.)]\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]*)(?:\s+\"(?P<title>[^\"]*)\")?\)")


class MarkdownAdapter(Adapter):
    name = "markdown"
    formats = ("markdown",)
    capabilities = (
        "ATX headings",
        "fenced code blocks with language",
        "ordered and unordered lists",
        "pipe tables with header detection",
        "block and inline images",
    )
    limitations = (
        "does not resolve reference-style links or footnotes",
        "does not evaluate raw HTML blocks",
        "does not interpret inline emphasis as structure (text is preserved verbatim)",
    )
    cost = "cheap"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        raw, encoding, lossy = read_text(ctx.path)
        text, stats = clean_text(raw, strip_controls=False)
        if lossy or stats.replacement:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"{stats.replacement} replacement character(s) in decoded markdown.",
                category=Category.STRUCTURAL,
                evidence=stats.as_evidence(),
            )

        blocks = _parse_blocks(ctx, text.split("\n"), self.name)
        doc = Document(
            adapter=self.name,
            container_kind="body",
            body=blocks,
            metadata={
                "encoding": encoding,
                "line_count": text.count("\n") + 1,
                "character_count": len(text),
            },
        )
        return ExtractionResult(
            document=doc,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )


def _parse_blocks(ctx: ExtractionContext, lines: list[str], adapter: str) -> list:
    blocks: list = []
    i = 0
    n = len(lines)
    offset = 0
    offsets: list[int] = []
    running = 0
    for ln in lines:
        offsets.append(running)
        running += len(ln) + 1
    del offset

    def make_locator(i: int, length: int):
        return char_locator(offsets[i] if i < n else 0, length)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block
        fence = _FENCE_RE.match(stripped)
        if fence:
            marker = fence.group(1)
            lang = fence.group(2) or None
            body: list[str] = []
            start = i
            i += 1
            while i < n:
                closing = _FENCE_RE.match(lines[i].strip())
                if closing and closing.group(1)[0] == marker[0] and len(closing.group(1)) >= len(marker):
                    i += 1
                    break
                body.append(lines[i])
                i += 1
            content = "\n".join(body)
            blocks.append(
                CodeBlock(
                    id=ctx.ids.child(None, "code"),
                    text=content,
                    language=lang,
                    sources=[
                        ref(adapter,
                            make_locator(start, len(content)),
                            origin=Origin.OBSERVED,
                            confidence=Confidence.HIGH,
                        )
                    ],
                )
            )
            continue

        # Table: header row + separator row
        if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            table, consumed = _parse_table(ctx, lines, i, make_locator, adapter)
            if table is not None:
                blocks.append(table)
                i = consumed
                continue

        heading = _ATX_RE.match(stripped)
        if heading:
            content = heading.group(2)
            blocks.append(
                Heading(
                    id=ctx.ids.child(None, "h"),
                    level=len(heading.group(1)),
                    text=content,
                    sources=[ref(adapter, make_locator(i, len(content)), confidence=Confidence.HIGH)],
                )
            )
            i += 1
            continue

        if stripped.startswith(">") or stripped.startswith("&gt;"):
            # No annotation: ``body`` is already bound above in this scope, and
            # re-annotating the same name is a redefinition, not a redeclaration.
            body = []
            start = i
            while i < n and lines[i].strip().startswith(">"):
                body.append(lines[i].strip().lstrip(">").strip())
                i += 1
            content = "\n".join(body)
            blocks.append(
                Quote(
                    id=ctx.ids.child(None, "quote"),
                    text=content,
                    sources=[ref(adapter, make_locator(start, len(content)))],
                )
            )
            continue

        list_item = _BULLET_RE.match(stripped) or _ORDERED_RE.match(stripped)
        if list_item:
            start = i
            items: list[ListItem] = []
            ordered = bool(_ORDERED_RE.match(stripped))
            while i < n:
                m_bullet = _BULLET_RE.match(lines[i].strip())
                m_ordered = _ORDERED_RE.match(lines[i].strip())
                m = m_ordered if ordered else m_bullet
                if not m:
                    if lines[i].strip() and lines[i].startswith(("  ", "\t")) and items:
                        items[-1].text += "\n" + lines[i].strip()
                        i += 1
                        continue
                    break
                content = (m_ordered or m_bullet).group(2) if not ordered else m_ordered.group(2)
                ordinal = int(m_ordered.group(1)) if m_ordered else None
                items.append(
                    ListItem(
                        id=ctx.ids.child(None, "li"),
                        text=content,
                        ordinal=ordinal,
                        sources=[ref(adapter, make_locator(i, len(content)))],
                    )
                )
                i += 1
            blocks.append(
                ListBlock(
                    id=ctx.ids.child(None, "list"),
                    ordered=ordered,
                    items=items,
                    sources=[ref(adapter, make_locator(start, 0))],
                )
            )
            continue

        # Standalone image
        image = _IMAGE_RE.fullmatch(stripped)
        if image:
            src = image.group("src")
            alt = image.group("alt") or None
            if not src.startswith(("http://", "https://")):
                ctx.warn(
                    Code.EXTERNAL_LINK,
                    f"Image reference '{src}' points outside the document; the asset is not embedded here.",
                    category=Category.UNVERIFIED,
                    reason="Markdown referenced a local or remote image that DocWeave cannot resolve.",
                    evidence={"src": src},
                )
            blocks.append(
                ImageBlock(
                    id=ctx.ids.child(None, "img"),
                    text="",
                    alt=alt,
                    meta={"src": src},
                    sources=[ref(adapter, make_locator(i, len(stripped)), confidence=Confidence.MEDIUM)],
                )
            )
            i += 1
            continue

        # Paragraph: consume until blank line or a new block starter
        start = i
        para: list[str] = []
        while i < n:
            candidate = lines[i].strip()
            if not candidate:
                break
            if _ATX_RE.match(candidate) or _FENCE_RE.match(candidate):
                break
            if para and (_BULLET_RE.match(candidate) or _ORDERED_RE.match(candidate)):
                break
            if candidate.startswith(">"):
                break
            para.append(candidate)
            i += 1
        content = "\n".join(para)
        if content:
            blocks.append(
                Paragraph(
                    id=ctx.ids.child(None, "p"),
                    text=content,
                    sources=[ref(adapter, make_locator(start, len(content)))],
                )
            )

    if not blocks:
            return blocks

    # Everything produced here is already Markdown. Flagging it stops the
    # renderer from escaping syntax a second time and showing literal slashes.
    for block in blocks:
        block.meta.setdefault("verbatim", True)
        if isinstance(block, ListBlock):
            for item in block.items:
                item.meta.setdefault("verbatim", True)

    return blocks


def _parse_table(ctx, lines: list[str], start: int, make_locator, adapter: str):
    """Parse a pipe table starting at ``lines[start]``."""
    header_cells = _split_row(lines[start])
    sep = _split_row(lines[start + 1])
    if len(sep) != len(header_cells) or not sep:
        return None, start

    rows: list[list[Cell]] = []
    header_rows = 0
    header_cells_out = _make_cells(ctx, header_cells, 0, header=True, make_locator=make_locator, line=start)
    rows.append(header_cells_out)
    header_rows = 1

    i = start + 2
    r = 1
    while i < len(lines) and "|" in lines[i] and lines[i].strip():
        cells = _make_cells(ctx, _split_row(lines[i]), r, header=False, make_locator=make_locator, line=i)
        rows.append(cells)
        r += 1
        i += 1

    table = Table(
        id=ctx.ids.child(None, "table"),
        text="",
        rows=rows,
        header_rows=header_rows,
        structure="grid",
        sources=[
            ref(
                adapter,
                make_locator(start, 0),
                origin=Origin.RECONSTRUCTED,
                confidence=Confidence.MEDIUM,
                note="Pipe table parsed from markdown source; alignment/span information is not present in markdown.",
            )
        ],
    )
    return table, i


def _make_cells(ctx, texts: list[str], row: int, *, header: bool, make_locator, line: int):
    cells: list[Cell] = []
    for col, text in enumerate(texts):
        cells.append(
            Cell(
                id=ctx.ids.child(None, "c"),
                text=text,
                row=row,
                column=col,
                is_header=header,
                locator=make_locator(line, len(text)),
            )
        )
    return cells


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in s:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def parse_markdown_text(ctx: ExtractionContext, text: str, adapter: str = "markdown") -> list:
    """Public helper used by the HTML adapter for embedded Markdown."""
    return _parse_blocks(ctx, text.split("\n"), adapter)


def guess_level(text: str) -> int | None:  # pragma: no cover - trivial helper
    m = _ATX_RE.match(text.strip())
    return len(m.group(1)) if m else None
