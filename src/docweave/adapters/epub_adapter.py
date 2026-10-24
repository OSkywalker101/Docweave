"""EPUB adapter.

Uses ebooklib to parse EPUB 2/3 files. Extracts:
- Spine order (reading order)
- Chapter content (HTML)
- Metadata (Dublin Core)
- Table of contents (NCX/NAV)
- Images and fonts as assets
- CSS styles (referenced, not applied)
"""

from __future__ import annotations

from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.config import ConversionConfig
from docweave.models import Cell, Document, Origin, Paragraph, Section, SourceInfo, SourceRef, Table
from docweave.models.diagnostics import Category, Code, Severity


class EpubAdapter(Adapter):
    name = "epub"
    formats = ("epub",)
    requires = {"ebooklib": "epub"}
    capabilities = (
        "EPUB 2 and 3",
        "spine-ordered reading sequence",
        "Dublin Core metadata",
        "NCX and NAV table of contents",
        "images/fonts as assets",
        "CSS referenced (not applied)",
        "MathML in content",
    )
    limitations = (
        "CSS not applied to output",
        "JavaScript/interactive content not executed",
        "fixed-layout EPUB not fully supported",
        "DRM-protected files not readable",
    )
    cost = "moderate"
    streaming = False

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        import ebooklib
        from ebooklib import epub

        config: ConversionConfig = ctx.config
        path: Path = ctx.path

        ctx.check_deadline("parse EPUB")
        book = epub.read_epub(str(path), options={"ignore_ncx": False})

        document = Document()
        document.container_kind = "body"

        # Metadata section
        meta_section = Section(title="EPUB Metadata", level=1)
        meta_section.id = ctx.ids.next("section")

        meta_table = Table()
        meta_table.id = ctx.ids.next("table")
        meta_table.header_rows = 1
        meta_table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

        # Dublin Core metadata
        for elem in book.get_metadata("DC", "title"):
            meta_table.rows.append([Cell(text="Title"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "creator"):
            meta_table.rows.append([Cell(text="Creator"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "subject"):
            meta_table.rows.append([Cell(text="Subject"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "description"):
            meta_table.rows.append([Cell(text="Description"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "publisher"):
            meta_table.rows.append([Cell(text="Publisher"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "date"):
            meta_table.rows.append([Cell(text="Date"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "identifier"):
            meta_table.rows.append([Cell(text="Identifier"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "language"):
            meta_table.rows.append([Cell(text="Language"), Cell(text=elem[0])])
        for elem in book.get_metadata("DC", "rights"):
            meta_table.rows.append([Cell(text="Rights"), Cell(text=elem[0])])

        # Additional metadata
        if book.get_metadata("DC", "contributor"):
            contribs = [e[0] for e in book.get_metadata("DC", "contributor")]
            meta_table.rows.append([Cell(text="Contributors"), Cell(text=", ".join(contribs))])

        meta_section.children.append(meta_table)
        document.body.append(meta_section)

        # Table of Contents
        toc_section = Section(title="Table of Contents", level=1)
        toc_section.id = ctx.ids.next("section")

        toc_table = Table()
        toc_table.id = ctx.ids.next("table")
        toc_table.header_rows = 1
        toc_table.rows = [[Cell(text="Level", is_header=True), Cell(text="Title", is_header=True), Cell(text="Href", is_header=True)]]

        def add_toc_items(items, level=1):
            for item in items:
                if isinstance(item, tuple):
                    # (link, title, [children]) or (link, title)
                    link = item[0]
                    title = item[1]
                    toc_table.rows.append([Cell(text=str(level)), Cell(text=title), Cell(text=link.href if hasattr(link, 'href') else str(link))])
                    if len(item) > 2 and item[2]:
                        add_toc_items(item[2], level + 1)
                elif isinstance(item, epub.Link):
                    toc_table.rows.append([Cell(text=str(level)), Cell(text=item.title), Cell(text=item.href)])
                elif isinstance(item, list):
                    add_toc_items(item, level)

        try:
            add_toc_items(book.toc)
        except Exception as exc:
            ctx.warn(
                Code.UNSUPPORTED_STRUCTURE,
                f"Could not parse table of contents: {exc}",
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                scope_kind="epub",
                scope_name="toc",
            )

        toc_section.children.append(toc_table)
        document.body.append(toc_section)

        # Spine content (reading order)
        content_section = Section(title="Content", level=1)
        content_section.id = ctx.ids.next("section")

        spine_items = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        for i, item in enumerate(spine_items):
            ctx.check_deadline(f"spine item {i}")
            item_section = Section(title=item.get_name() or f"Chapter {i+1}", level=2)
            item_section.id = ctx.ids.next("section")

            # Parse HTML content
            content = item.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8", "replace")

            # Extract text from HTML
            from docweave.adapters.html_adapter import HtmlAdapter
            HtmlAdapter()
            from docweave.assets import AssetCollector
            from docweave.ids import IdFactory

            ExtractionContext(
                path=path,
                config=config,
                ids=IdFactory(),
                assets=AssetCollector(directory=None, enabled=False),
            )
            # We'll extract text directly here
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")

            # Remove script/style
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            # Get text with some structure
            for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "pre", "ul", "ol", "table"]):
                text = elem.get_text(" ", strip=True)
                if not text:
                    continue

                if elem.name.startswith("h"):
                    level = int(elem.name[1])
                    heading = Section(title=text, level=min(level + 2, 4))  # Offset by 2
                    heading.id = ctx.ids.next("section")
                    heading.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"spine item: {item.get_name()}",
                        )
                    ]
                    item_section.children.append(heading)
                elif elem.name == "p":
                    para = Paragraph(text=text)
                    para.id = ctx.ids.next("paragraph")
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"spine item: {item.get_name()}",
                        )
                    ]
                    item_section.children.append(para)
                elif elem.name in ("ul", "ol"):
                    list_block = self._parse_list(elem, ctx, f"spine item: {item.get_name()}")
                    if list_block:
                        item_section.children.append(list_block)
                elif elem.name == "table":
                    table = self._parse_table(elem, ctx, f"spine item: {item.get_name()}")
                    if table:
                        item_section.children.append(table)
                elif elem.name == "blockquote":
                    para = Paragraph(text=text)
                    para.id = ctx.ids.next("paragraph")
                    para.meta = {"blockquote": True}
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"spine item: {item.get_name()}",
                        )
                    ]
                    item_section.children.append(para)
                elif elem.name == "pre":
                    code_block = elem.get_text()
                    from docweave.models import CodeBlock
                    cb = CodeBlock(text=code_block)
                    cb.id = ctx.ids.next("code")
                    cb.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"spine item: {item.get_name()}",
                        )
                    ]
                    item_section.children.append(cb)

            content_section.children.append(item_section)

        document.body.append(content_section)

        # Images as assets
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            asset = AssetRef(
                asset_id=ctx.ids.next("asset"),
                media_type=item.media_type or "image",
                extension=Path(item.file_name).suffix.lower() if item.file_name else None,
                byte_size=len(item.content) if item.content else 0,
                path=f"epub:{path.name}#{item.get_name()}",
                extracted=False,
                description=f"EPUB image: {item.get_name()}",
            )
            document.assets.append(asset)
            ctx.assets.add(asset)

        # Fonts as assets
        for item in book.get_items_of_type(ebooklib.ITEM_FONT):
            asset = AssetRef(
                asset_id=ctx.ids.next("asset"),
                media_type=item.media_type or "font",
                extension=Path(item.file_name).suffix.lower() if item.file_name else None,
                byte_size=len(item.content) if item.content else 0,
                path=f"epub:{path.name}#{item.get_name()}",
                extracted=False,
                description=f"EPUB font: {item.get_name()}",
            )
            document.assets.append(asset)
            ctx.assets.add(asset)

        document.metadata["epub"] = {
            "version": book.version,
            "spine_count": len(book.spine),
            "toc_items": len(book.toc) if book.toc else 0,
            "images": len(list(book.get_items_of_type(ebooklib.ITEM_IMAGE))),
            "fonts": len(list(book.get_items_of_type(ebooklib.ITEM_FONT))),
        }

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",
            detected_format="epub",
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")

    def _parse_list(self, elem, ctx: ExtractionContext, note: str):
        from docweave.models import ListBlock, ListItem
        list_block = ListBlock(ordered=elem.name == "ol")
        list_block.id = ctx.ids.next("list")
        for li in elem.find_all("li", recursive=False):
            text = li.get_text(" ", strip=True)
            if text:
                item = ListItem(text=text)
                item.id = ctx.ids.next("item")
                item.sources = [
                    SourceRef(
                        adapter=self.name,
                        origin=Origin.OBSERVED,
                        confidence="high",
                        note=note,
                    )
                ]
                list_block.items.append(item)
        return list_block if list_block.items else None

    def _parse_table(self, elem, ctx: ExtractionContext, note: str):
        from docweave.models import Cell, Table
        table = Table()
        table.id = ctx.ids.next("table")
        for row_elem in elem.find_all("tr"):
            row = []
            for cell_elem in row_elem.find_all(["td", "th"]):
                text = cell_elem.get_text(" ", strip=True)
                cell = Cell(text=text, is_header=cell_elem.name == "th")
                cell.sources = [
                    SourceRef(
                        adapter=self.name,
                        origin=Origin.OBSERVED,
                        confidence="high",
                        note=note,
                    )
                ]
                row.append(cell)
            if row:
                table.rows.append(row)
        return table if table.rows else None
