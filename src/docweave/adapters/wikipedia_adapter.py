"""Wikipedia article adapter.

Uses wikipedia-api to fetch article content, sections, links, and metadata.
Structures the article by heading hierarchy for faithful representation.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.models import Cell, Document, Origin, Paragraph, Section, SourceInfo, SourceRef, Table
from docweave.models.diagnostics import Category, Code, Severity


class WikipediaAdapter(Adapter):
    name = "wikipedia"
    formats = ("wikipedia",)
    requires = {"wikipediaapi": "web"}
    capabilities = (
        "fetches article content via MediaWiki API",
        "preserves heading hierarchy",
        "extracts links and references",
        "categories and templates",
        "page images as assets",
        "multi-language support",
    )
    limitations = (
        "requires network access",
        "rate limited by Wikipedia API",
        "some extensions (math, scores) not rendered",
        "redirect pages not auto-followed",
    )
    cost = "moderate"
    streaming = False

    # URL patterns
    URL_PATTERNS = [
        r"^https?://([a-z]{2,3})\.wikipedia\.org/wiki/([^#?]+)",
        r"^https?://([a-z]{2,3})\.w\.org/wiki/([^#?]+)",
    ]

    def __init__(self):
        super().__init__()
        self._wiki = None

    def _get_wiki(self, lang: str = "en"):
        import wikipediaapi
        if self._wiki is None or self._wiki.language != lang:
            self._wiki = wikipediaapi.Wikipedia(
                language=lang,
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent="DocWeave/0.1 (https://github.com/docweave/docweave)",
            )
        return self._wiki

    @classmethod
    def extract_page_title(cls, url: str) -> tuple[str, str] | None:
        """Returns (lang, title) from Wikipedia URL."""
        for pattern in cls.URL_PATTERNS:
            import re
            m = re.match(pattern, url)
            if m:
                lang = m.group(1)
                title = unquote(m.group(2)).replace("_", " ")
                return lang, title
        return None

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:

        # The "path" for Wikipedia is a URL
        url = ctx.detection.media_type or ctx.detection.declared_extension or str(ctx.path)
        extracted = self.extract_page_title(url)
        if not extracted:
            raise ValueError(f"Could not extract Wikipedia page from: {url}")
        lang, title = extracted

        ctx.check_deadline("fetch article")
        wiki = self._get_wiki(lang)
        page = wiki.page(title)

        if not page.exists():
            ctx.warn(
                Code.UNSUPPORTED_STRUCTURE,
                f"Wikipedia page does not exist: {title}",
                category=Category.STRUCTURAL,
                severity=Severity.ERROR,
                scope_kind="wikipedia",
                scope_name=title,
            )
            # Return minimal document
            document = Document()
            document.container_kind = "body"
            para = Paragraph(text=f"[Wikipedia page not found: {title}]")
            para.id = ctx.ids.next("paragraph")
            document.body.append(para)

            source = SourceInfo(
                name=f"wikipedia:{lang}:{title}",
                byte_size=0,
                sha256="",
                detected_format="wikipedia",
                detection_method="url",
                detection_confidence=0.0,
                declared_extension=None,
                extension_mismatch=False,
            )
            return ExtractionResult(document=document, source=source, container_kind="body")

        document = Document()
        document.container_kind = "body"

        # Page metadata section
        meta_section = Section(title="Page Metadata", level=1)
        meta_section.id = ctx.ids.next("section")

        meta_table = Table()
        meta_table.id = ctx.ids.next("table")
        meta_table.header_rows = 1
        meta_table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

        meta_fields = [
            ("Title", page.title),
            ("Page ID", str(page.pageid)),
            ("Language", lang),
            ("Full URL", page.fullurl),
            ("Categories", ", ".join(list(page.categories.keys())[:20])),
            ("Links (count)", str(len(page.links))),
            ("References (count)", str(len(page.references))),
            ("Images (count)", str(len(page.images))),
        ]

        for key, val in meta_fields:
            if val:
                meta_table.rows.append([Cell(text=key), Cell(text=str(val))])

        meta_section.children.append(meta_table)
        document.body.append(meta_section)

        # Content sections - recursively process heading hierarchy
        def process_sections(sections: list, parent_section: Section | None = None, level: int = 2):
            for sec in sections:
                ctx.check_deadline(f"section {sec.title}")
                sec_section = Section(title=sec.title, level=level)
                sec_section.id = ctx.ids.next("section")

                # Section text
                if sec.text:
                    for para_text in sec.text.split("\n\n"):
                        para_text = para_text.strip()
                        if para_text:
                            para = Paragraph(text=para_text)
                            para.id = ctx.ids.next("paragraph")
                            para.sources = [
                                SourceRef(
                                    adapter=self.name,
                                    origin=Origin.OBSERVED,
                                    confidence="high",
                                    note=f"section: {sec.title}",
                                )
                            ]
                            sec_section.children.append(para)

                if parent_section:
                    parent_section.children.append(sec_section)
                else:
                    document.body.append(sec_section)

                # Recurse
                if sec.sections:
                    process_sections(sec.sections, sec_section, level + 1)

        process_sections(page.sections)

        # Images as assets
        for img_url in list(page.images)[:50]:  # Limit to 50
            asset = AssetRef(
                asset_id=ctx.ids.next("asset"),
                media_type="image",
                extension=Path(img_url).suffix.lower() or ".jpg",
                byte_size=0,
                path=img_url,
                extracted=False,
                description=f"Wikipedia image from {title}",
            )
            document.assets.append(asset)
            ctx.assets.add(asset)

        document.metadata["wikipedia"] = {
            "title": page.title,
            "pageid": page.pageid,
            "language": lang,
            "url": page.fullurl,
            "categories": list(page.categories.keys()),
            "sections": len(page.sections),
        }

        source = SourceInfo(
            name=f"wikipedia:{lang}:{title}",
            byte_size=len(page.text) if page.text else 0,
            sha256="",
            detected_format="wikipedia",
            detection_method="url",
            detection_confidence=1.0,
            declared_extension=None,
            extension_mismatch=False,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")
