"""HTML / XHTML adapter.

Built on BeautifulSoup with the pure-Python ``html.parser`` backend: it does not
execute scripts and does not load external entities, which matters because every
input is untrusted. Structural elements map to typed IR nodes; unknown wrappers
are traversed rather than discarded so their text is never lost.
"""

from __future__ import annotations

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import clean_text, decode_bytes
from docweave.models import (
    Cell,
    CodeBlock,
    Confidence,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Locator,
    Origin,
    Paragraph,
    Quote,
    SourceInfo,
    Table,
)
from docweave.models.diagnostics import Category, Code

_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
    "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "table", "ul",
}
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head", "iframe", "object"}
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class HtmlAdapter(Adapter):
    name = "html"
    formats = ("html", "xhtml")
    requires = {"bs4": "beautifulsoup4"}
    capabilities = (
        "headings, paragraphs, lists, tables, code, images, blockquotes",
        "configurable element stripping",
        "css-path provenance per node",
        "safe: no script execution, no external entity resolution",
    )
    limitations = (
        "does not lay out CSS; visual order is assumed to match DOM order",
        "does not execute JavaScript, so script-rendered content is absent",
        "does not resolve relative image URLs against the filesystem",
    )
    cost = "cheap"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        from bs4 import BeautifulSoup, Comment

        data = ctx.path.read_bytes()
        text, encoding, lossy = decode_bytes(data)
        text, stats = clean_text(text, strip_controls=False)

        soup = BeautifulSoup(text, "html.parser")

        if ctx.config.html_strip_comments:
            for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
                comment.extract()
        for selector in ctx.config.html_strip_selectors:
            for element in soup.select(selector):
                element.decompose()

        title_el = soup.find("title")
        title = title_el.get_text(strip=True) if title_el else None

        body = soup.body or soup
        walker = _Walker(ctx, self.name)
        blocks = walker.walk_children(body)

        if lossy or stats.replacement:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"{stats.replacement} replacement character(s) after decoding as {encoding}.",
                category=Category.STRUCTURAL,
                reason="The declared charset could not be used strictly.",
                hint="Ensure the document declares UTF-8 (or a charset meta tag).",
                evidence=stats.as_evidence(),
            )

        images_seen = sum(1 for n in blocks if isinstance(n, ImageBlock))
        if images_seen:
            ctx.warn(
                Code.IMAGE_WITHOUT_TEXT,
                f"{images_seen} image(s) referenced; their visual content was not interpreted.",
                category=Category.UNVERIFIED,
                severity="info",
                reason="DocWeave does not run a vision model by default.",
                hint="Install 'docweave[ocr]' and run the optional OCR pass for pixel text.",
                evidence={"image_count": images_seen},
            )

        doc = Document(
            adapter=self.name,
            container_kind="body",
            body=blocks,
            metadata={
                "encoding": encoding,
                "title": title,
                "node_count": len(blocks),
                "table_count": sum(1 for b in blocks if isinstance(b, Table)),
            },
        )
        return ExtractionResult(
            document=doc,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )


class _Walker:
    """Depth-first DOM walk that turns elements into IR nodes."""

    def __init__(self, ctx: ExtractionContext, adapter: str) -> None:
        self.ctx = ctx
        self.adapter = adapter

    def walk_children(self, element) -> list:
        out: list = []
        for child in element.children:
            node = self.visit(child)
            if node is None:
                continue
            if isinstance(node, list):
                out.extend(node)
            else:
                out.append(node)
        return out

    def visit(self, element):
        from bs4 import NavigableString, Tag

        if isinstance(element, NavigableString):
            text = str(element).strip()
            if not text:
                return None
            return self._paragraph(text, element)
        if not isinstance(element, Tag):
            return None

        name = (element.name or "").lower()
        if name in _SKIP_TAGS:
            return None

        path = _css_path(element)

        if name in _HEADING_TAGS:
            text = text_of(element)
            if not text:
                return None
            return Heading(
                id=self.ctx.ids.child(None, "h"),
                level=_HEADING_TAGS[name],
                text=text,
                sources=[self._ref(path, locator_offset(text))],
            )

        if name == "table":
            return self._table(element, path)

        if name in ("ul", "ol"):
            return self._list(element, path)

        if name == "pre":
            text = text_of(element)
            code_el = element.find("code")
            language = None
            if code_el is not None:
                classes = code_el.get("class") or []
                for cls in classes:
                    if str(cls).startswith("language-"):
                        language = str(cls)[len("language-"):]
            return CodeBlock(
                id=self.ctx.ids.child(None, "code"),
                text=text.rstrip("\n"),
                language=language,
                sources=[self._ref(path, locator_offset(text))],
            )

        if name == "blockquote":
            text = text_of(element)
            if not text:
                return None
            return Quote(
                id=self.ctx.ids.child(None, "quote"),
                text=text,
                sources=[self._ref(path, locator_offset(text))],
            )

        if name == "hr":
            return None

        if name == "img":
            return self._image(element, path)

        if name == "br":
            return None

        if name == "a":
            text = text_of(element)
            href = element.get("href")
            if text and href:
                node = self._paragraph(text, element)
                node.meta["href"] = href
                return node

        if name == "p" or (name in _BLOCK_TAGS and not self._has_block_child(element)):
            text = text_of(element)
            if not text:
                return None
            node = self._paragraph(text, element)
            # Inline anchors live inside the paragraph we just flattened, so
            # harvest them here instead of dropping them.
            links = [
                a.get("href")
                for a in element.find_all("a", href=True)
                if a.get("href") and a.get("href") != "#"
            ]
            if links:
                node.meta["links"] = links
            return node

        if name in _BLOCK_TAGS:
            children = self.walk_children(element)
            if children:
                return children
            text = text_of(element)
            if text:
                return self._paragraph(text, element)
            return None

        # Inline / unknown element: keep the text.
        text = text_of(element)
        if text:
            return self._paragraph(text, element)
        return None

    # -- helpers -----------------------------------------------------------

    def _ref(self, css_path: str, text: str, **kwargs) -> object:
        from docweave.adapters.common import ref

        return ref(
            self.adapter,
            Locator(kind="html", css_path=css_path),
            **kwargs,
        )

    def _paragraph(self, text: str, element) -> Paragraph:
        return Paragraph(
            id=self.ctx.ids.child(None, "p"),
            text=text,
            sources=[self._ref(_css_path(element), text)],
        )

    def _list(self, element, path: str) -> ListBlock | None:
        items: list[ListItem] = []
        for li in element.find_all("li", recursive=False):
            text = text_of(li)
            if not text:
                continue
            items.append(
                ListItem(
                    id=self.ctx.ids.child(None, "li"),
                    text=text,
                    ordinal=len(items) + 1 if element.name == "ol" else None,
                    sources=[self._ref(_css_path(li), text)],
                )
            )
        if not items:
            return None
        return ListBlock(
            id=self.ctx.ids.child(None, "list"),
            ordered=element.name == "ol",
            items=items,
            sources=[self._ref(path, "")],
        )

    def _table(self, element, path: str) -> Table | None:
        rows: list[list[Cell]] = []
        header_rows = 0
        for r, tr in enumerate(element.find_all("tr")):
            cells: list[Cell] = []
            for c, cell in enumerate(tr.find_all(["th", "td"], recursive=False)):
                text = text_of(cell)
                span = _int_attr(cell, "colspan", 1)
                row_span = _int_attr(cell, "rowspan", 1)
                cells.append(
                    Cell(
                        id=self.ctx.ids.child(None, "c"),
                        text=text,
                        row=r,
                        column=c,
                        is_header=cell.name == "th",
                        column_span=span,
                        row_span=row_span,
                        locator=Locator(kind="html", css_path=_css_path(cell)),
                    )
                )
            if cells:
                rows.append(cells)
                if r == 0 and any(c.is_header for c in cells):
                    header_rows = 1
        if not rows:
            return None
        return Table(
            id=self.ctx.ids.child(None, "table"),
            text="",
            rows=rows,
            header_rows=header_rows,
            caption=_table_caption(element),
            sources=[
                self._ref(path, "", origin=Origin.OBSERVED, confidence=Confidence.HIGH)
            ],
        )

    def _image(self, element, path: str) -> ImageBlock | None:
        src = element.get("src")
        if not src:
            return None
        alt = element.get("alt") or None
        asset_ref = None
        if self.ctx.config.extract_assets and not src.startswith(("http://", "https://", "data:")):
            candidate = (self.ctx.path.parent / src).resolve()
            try:
                if candidate.is_file() and self.ctx.path.parent.resolve() in candidate.parents:
                    asset_ref = self.ctx.assets.add(
                        candidate.read_bytes(),
                        media_type=None,
                        locator=Locator(kind="html", css_path=path),
                        asset_id=self.ctx.ids.child(None, "asset"),
                    )
            except OSError:
                asset_ref = None
        return ImageBlock(
            id=self.ctx.ids.child(None, "img"),
            text="",
            alt=alt,
            asset_id=asset_ref.asset_id if asset_ref else None,
            width=_int_or_none(element.get("width")),
            height=_int_or_none(element.get("height")),
            meta={"src": src},
            sources=[self._ref(path, src, confidence=Confidence.MEDIUM)],
        )

    def _has_block_child(self, element) -> bool:
        from bs4 import Tag

        return any(
            isinstance(c, Tag) and (c.name or "").lower() in _BLOCK_TAGS for c in element.children
        )


def _int_attr(element, name: str, default: int) -> int:
    raw = element.get(name)
    try:
        value = int(str(raw))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


_WS_RE = None


def text_of(element) -> str:
    """Flatten an element's text without inventing word breaks.

    ``get_text(" ")`` inserts a separator everywhere, which turns
    ``<a>link</a>.`` into ``link .``. Joining with an empty separator keeps the
    source's own whitespace, then we collapse runs.
    """
    import re

    global _WS_RE
    if _WS_RE is None:
        _WS_RE = re.compile(r"\s+")
    return _WS_RE.sub(" ", element.get_text("", strip=False)).strip()


def _int_or_none(raw) -> int | None:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def _table_caption(element) -> str | None:
    caption = element.find("caption")
    if caption is not None:
        text = text_of(caption)
        return text or None
    return None


def locator_offset(text: str):
    return Locator(kind="html", char_length=len(text)) if text else Locator()


def _css_path(element) -> str:
    """Build a short, deterministic CSS-ish path for provenance."""
    parts: list[str] = []
    node = element
    while node is not None and getattr(node, "name", None) and len(parts) < 6:
        parent = node.parent
        if parent is None:
            parts.append(node.name)
            break
        siblings = [c for c in parent.children if getattr(c, "name", None) == node.name]
        if len(siblings) > 1:
            parts.append(f"{node.name}:nth-of-type({siblings.index(node) + 1})")
        else:
            parts.append(node.name)
        node = parent
    return " > ".join(reversed(parts))
