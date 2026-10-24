"""RSS/Atom feed adapter.

Uses feedparser to parse RSS 1.0/2.0 and Atom 1.0 feeds.
Extracts channel metadata, entries with content, enclosures, and categories.
"""

from __future__ import annotations

import html
from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.models import Cell, Document, Origin, Paragraph, Section, SourceInfo, SourceRef, Table
from docweave.models.diagnostics import Category, Code, Severity


class RssAdapter(Adapter):
    name = "rss"
    formats = ("rss", "atom")
    requires = {"feedparser": "web"}
    capabilities = (
        "RSS 0.91/1.0/2.0 and Atom 1.0",
        "channel/feed metadata",
        "entries with full content",
        "enclosures (podcasts, media)",
        "categories/tags",
        "author information",
    )
    limitations = (
        "no authentication for private feeds",
        "rate limiting not handled",
        "malformed feeds may lose data",
    )
    cost = "cheap"
    streaming = True

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        import feedparser

        path: Path = ctx.path

        ctx.check_deadline("parse feed")
        feed = feedparser.parse(str(path))

        if feed.bozo and feed.bozo_exception:
            ctx.warn(
                Code.MALFORMED_DOCUMENT,
                f"Feed parsing had errors: {feed.bozo_exception}",
                category=Category.STRUCTURAL,
                severity=Severity.WARNING,
                scope_kind="feed",
                scope_name="root",
                evidence={"bozo": True, "exception": str(feed.bozo_exception)},
            )

        document = Document()
        document.container_kind = "body"

        # Feed metadata section
        meta_section = Section(title="Feed Metadata", level=1)
        meta_section.id = ctx.ids.next("section")

        feed_info = feed.feed
        meta_table = Table()
        meta_table.id = ctx.ids.next("table")
        meta_table.header_rows = 1
        meta_table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

        feed_fields = [
            ("Title", feed_info.get("title")),
            ("Subtitle/Description", feed_info.get("subtitle") or feed_info.get("description")),
            ("Link", feed_info.get("link")),
            ("Language", feed_info.get("language")),
            ("Updated", feed_info.get("updated") or feed_info.get("published")),
            ("Generator", feed_info.get("generator")),
            ("Rights", feed_info.get("rights")),
            ("Author", feed_info.get("author") or feed_info.get("author_detail", {}).get("name")),
            ("Image", feed_info.get("image", {}).get("href") if feed_info.get("image") else None),
        ]

        for key, val in feed_fields:
            if val:
                meta_table.rows.append([Cell(text=key), Cell(text=str(val))])

        meta_section.children.append(meta_table)
        document.body.append(meta_section)

        # Entries section
        entries_section = Section(title=f"Entries ({len(feed.entries)})", level=1)
        entries_section.id = ctx.ids.next("section")

        entries_table = Table()
        entries_table.id = ctx.ids.next("table")
        entries_table.header_rows = 1
        entries_table.rows = [[
            Cell(text="Title", is_header=True),
            Cell(text="Published", is_header=True),
            Cell(text="Author", is_header=True),
            Cell(text="Link", is_header=True),
            Cell(text="Categories", is_header=True),
            Cell(text="Enclosures", is_header=True),
        ]]

        for i, entry in enumerate(feed.entries):
            ctx.check_deadline(f"entry {i}")

            # Entry detail section
            entry_section = Section(title=entry.get("title", f"Entry {i+1}"), level=2)
            entry_section.id = ctx.ids.next("section")

            # Entry metadata table
            e_table = Table()
            e_table.id = ctx.ids.next("table")
            e_table.header_rows = 1
            e_table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

            # Published date
            pub_date = entry.get("published") or entry.get("updated") or entry.get("created")
            if pub_date:
                e_table.rows.append([Cell(text="Published"), Cell(text=str(pub_date))])

            # Author
            author = entry.get("author") or entry.get("author_detail", {}).get("name")
            if author:
                e_table.rows.append([Cell(text="Author"), Cell(text=str(author))])

            # Link
            link = entry.get("link")
            if link:
                e_table.rows.append([Cell(text="Link"), Cell(text=str(link))])

            # Categories
            cats = entry.get("tags", [])
            if cats:
                cat_text = ", ".join(t.get("term", "") for t in cats if t.get("term"))
                if cat_text:
                    e_table.rows.append([Cell(text="Categories"), Cell(text=cat_text)])

            # Enclosures
            enclosures = entry.get("enclosures", [])
            if enclosures:
                enc_text = ", ".join(f"{e.get('href', '')} ({e.get('type', '')})" for e in enclosures)
                e_table.rows.append([Cell(text="Enclosures"), Cell(text=enc_text)])
                # Add enclosures as assets
                for enc in enclosures:
                    asset = AssetRef(
                        asset_id=ctx.ids.next("asset"),
                        media_type=enc.get("type", "application/octet-stream"),
                        extension=Path(enc.get("href", "")).suffix.lower() if enc.get("href") else None,
                        byte_size=enc.get("length", 0),
                        path=enc.get("href"),
                        extracted=False,
                        description=f"Feed enclosure: {entry.get('title', 'entry')}",
                    )
                    document.assets.append(asset)
                    ctx.assets.add(asset)

            entry_section.children.append(e_table)

            # Entry content
            content = entry.get("content", [])
            if content:
                for c in content:
                    if c.get("value"):
                        text = self._clean_html(c["value"])
                        if text:
                            para = Paragraph(text=text)
                            para.id = ctx.ids.next("paragraph")
                            para.sources = [
                                SourceRef(
                                    adapter=self.name,
                                    origin=Origin.OBSERVED,
                                    confidence="high",
                                    note=f"entry content: {entry.get('title', 'untitled')}",
                                )
                            ]
                            entry_section.children.append(para)
            else:
                # Fallback to summary/description
                summary = entry.get("summary") or entry.get("description")
                if summary:
                    text = self._clean_html(summary)
                    if text:
                        para = Paragraph(text=text)
                        para.id = ctx.ids.next("paragraph")
                        para.sources = [
                            SourceRef(
                                adapter=self.name,
                                origin=Origin.OBSERVED,
                                confidence="high",
                                note=f"entry summary: {entry.get('title', 'untitled')}",
                            )
                        ]
                        entry_section.children.append(para)

            entries_section.children.append(entry_section)

            # Add to summary table
            entries_table.rows.append([
                Cell(text=entry.get("title", "Untitled")[:100]),
                Cell(text=str(pub_date) if pub_date else ""),
                Cell(text=str(author) if author else ""),
                Cell(text=str(link) if link else ""),
                Cell(text=cat_text if cats else ""),
                Cell(text=str(len(enclosures)) if enclosures else "0"),
            ])

        entries_section.children.insert(0, entries_table)
        document.body.append(entries_section)

        document.metadata["feed"] = {
            "title": feed_info.get("title"),
            "link": feed_info.get("link"),
            "entry_count": len(feed.entries),
            "feed_type": feed.version,
            "bozo": feed.bozo,
        }

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",
            detected_format=ctx.detection.format,
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")

    def _clean_html(self, text: str) -> str:
        """Strip HTML tags and decode entities."""
        import re
        # Decode HTML entities
        text = html.unescape(text)
        # Remove script/style
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
