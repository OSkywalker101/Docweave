"""Bing Search API adapter.

Fetches search results from Bing Search API and structures them as a document.
Requires BING_SEARCH_API_KEY environment variable or config.
"""

from __future__ import annotations

import os

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.config import ConversionConfig
from docweave.exceptions import DependencyMissingError
from docweave.models import Cell, Document, Origin, Paragraph, Section, SourceInfo, SourceRef, Table


class BingAdapter(Adapter):
    name = "bing"
    formats = ("bing",)
    requires = {"requests": "web"}
    capabilities = (
        "Bing Web Search API v7",
        "structured results with snippets",
        "answer packs (news, videos, entities)",
        "safe search filtering",
        "market/language targeting",
    )
    limitations = (
        "requires Azure Bing Search subscription",
        "rate limited by subscription tier",
        "no access to raw HTML of result pages",
        "costs money per query",
    )
    cost = "expensive"
    streaming = False

    def __init__(self):
        super().__init__()
        self._api_key = None
        self._endpoint = "https://api.bing.microsoft.com/v7.0/search"

    def _get_credentials(self, config: ConversionConfig) -> tuple[str, str]:
        """Get API key and endpoint from config or environment."""
        api_key = getattr(config, "bing_api_key", None) or os.getenv("BING_SEARCH_API_KEY")
        endpoint = getattr(config, "bing_endpoint", None) or os.getenv("BING_SEARCH_ENDPOINT", self._endpoint)
        if not api_key:
            raise DependencyMissingError(
                "BING_SEARCH_API_KEY",
                "The bing adapter",
                "web (requires Azure Bing Search API key)",
            )
        return api_key, endpoint

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        import requests

        config: ConversionConfig = ctx.config
        # The "path" for Bing is a query string or URL with query params
        query = ctx.detection.media_type or ctx.detection.declared_extension or str(ctx.path)

        api_key, endpoint = self._get_credentials(config)

        ctx.check_deadline("search request")
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {
            "q": query,
            "count": getattr(config, "bing_count", 50),
            "offset": getattr(config, "bing_offset", 0),
            "mkt": getattr(config, "bing_market", "en-US"),
            "safesearch": getattr(config, "bing_safesearch", "Moderate"),
            "responseFilter": "Webpages,News,Images,Videos,RelatedSearches",
        }

        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        document = Document()
        document.container_kind = "body"

        # Query metadata
        meta_section = Section(title=f"Search Results: {query}", level=1)
        meta_section.id = ctx.ids.next("section")

        meta_table = Table()
        meta_table.id = ctx.ids.next("table")
        meta_table.header_rows = 1
        meta_table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

        meta_fields = [
            ("Query", query),
            ("Total Estimated Matches", f"{data.get('webPages', {}).get('totalEstimatedMatches', 0):,}"),
            ("Market", params["mkt"]),
            ("SafeSearch", params["safesearch"]),
            ("Response Types", ", ".join(k for k in data.keys() if k not in ("_type", "queryContext"))),
        ]

        for key, val in meta_fields:
            if val:
                meta_table.rows.append([Cell(text=key), Cell(text=str(val))])

        meta_section.children.append(meta_table)
        document.body.append(meta_section)

        # Web pages
        web_pages = data.get("webPages", {}).get("value", [])
        if web_pages:
            wp_section = Section(title=f"Web Pages ({len(web_pages)})", level=1)
            wp_section.id = ctx.ids.next("section")

            wp_table = Table()
            wp_table.id = ctx.ids.next("table")
            wp_table.header_rows = 1
            wp_table.rows = [[
                Cell(text="Title", is_header=True),
                Cell(text="URL", is_header=True),
                Cell(text="Snippet", is_header=True),
                Cell(text="Display URL", is_header=True),
                Cell(text="Date", is_header=True),
            ]]

            for page in web_pages:
                wp_table.rows.append([
                    Cell(text=page.get("name", "")[:100]),
                    Cell(text=page.get("url", "")),
                    Cell(text=page.get("snippet", "")[:200]),
                    Cell(text=page.get("displayUrl", "")),
                    Cell(text=page.get("dateLastCrawled", "")[:10] if page.get("dateLastCrawled") else ""),
                ])

            wp_section.children.append(wp_table)
            document.body.append(wp_section)

        # News
        news = data.get("news", {}).get("value", [])
        if news:
            news_section = Section(title=f"News ({len(news)})", level=1)
            news_section.id = ctx.ids.next("section")

            news_table = Table()
            news_table.id = ctx.ids.next("table")
            news_table.header_rows = 1
            news_table.rows = [[
                Cell(text="Title", is_header=True),
                Cell(text="Source", is_header=True),
                Cell(text="Description", is_header=True),
                Cell(text="URL", is_header=True),
                Cell(text="Published", is_header=True),
            ]]

            for item in news:
                news_table.rows.append([
                    Cell(text=item.get("name", "")[:100]),
                    Cell(text=item.get("provider", [{}])[0].get("name", "") if item.get("provider") else ""),
                    Cell(text=item.get("description", "")[:200]),
                    Cell(text=item.get("url", "")),
                    Cell(text=item.get("datePublished", "")[:10] if item.get("datePublished") else ""),
                ])

            news_section.children.append(news_table)
            document.body.append(news_section)

        # Images
        images = data.get("images", {}).get("value", [])
        if images:
            img_section = Section(title=f"Images ({len(images)})", level=1)
            img_section.id = ctx.ids.next("section")

            img_table = Table()
            img_table.id = ctx.ids.next("table")
            img_table.header_rows = 1
            img_table.rows = [[
                Cell(text="Thumbnail", is_header=True),
                Cell(text="Content URL", is_header=True),
                Cell(text="Title", is_header=True),
                Cell(text="Size", is_header=True),
            ]]

            for img in images[:20]:  # Limit
                img_table.rows.append([
                    Cell(text=img.get("thumbnailUrl", "")[:80]),
                    Cell(text=img.get("contentUrl", "")[:80]),
                    Cell(text=img.get("name", "")[:80]),
                    Cell(text=f"{img.get('width', '')}x{img.get('height', '')}"),
                ])

                # Add as asset
                asset = AssetRef(
                    asset_id=ctx.ids.next("asset"),
                    media_type=img.get("encodingFormat", "image"),
                    extension=f".{img.get('encodingFormat', 'jpg')}",
                    byte_size=img.get("contentSize", 0),
                    path=img.get("contentUrl"),
                    extracted=False,
                    description=f"Bing image result: {img.get('name', 'image')}",
                )
                document.assets.append(asset)
                ctx.assets.add(asset)

            img_section.children.append(img_table)
            document.body.append(img_section)

        # Videos
        videos = data.get("videos", {}).get("value", [])
        if videos:
            vid_section = Section(title=f"Videos ({len(videos)})", level=1)
            vid_section.id = ctx.ids.next("section")

            vid_table = Table()
            vid_table.id = ctx.ids.next("table")
            vid_table.header_rows = 1
            vid_table.rows = [[
                Cell(text="Title", is_header=True),
                Cell(text="URL", is_header=True),
                Cell(text="Duration", is_header=True),
                Cell(text="View Count", is_header=True),
            ]]

            for vid in videos[:20]:
                vid_table.rows.append([
                    Cell(text=vid.get("name", "")[:100]),
                    Cell(text=vid.get("contentUrl", "")[:80]),
                    Cell(text=self._format_duration(vid.get("duration", ""))),
                    Cell(text=f"{vid.get('viewCount', 0):,}" if vid.get("viewCount") else ""),
                ])

            vid_section.children.append(vid_table)
            document.body.append(vid_section)

        # Related searches
        related = data.get("relatedSearches", {}).get("value", [])
        if related:
            rel_section = Section(title="Related Searches", level=1)
            rel_section.id = ctx.ids.next("section")
            for r in related:
                para = Paragraph(text=r.get("text", ""))
                para.id = ctx.ids.next("paragraph")
                para.sources = [
                    SourceRef(
                        adapter=self.name,
                        origin=Origin.OBSERVED,
                        confidence="high",
                        note="related search",
                    )
                ]
                rel_section.children.append(para)
            document.body.append(rel_section)

        document.metadata["bing"] = {
            "query": query,
            "web_results": len(web_pages),
            "news_results": len(news),
            "image_results": len(images),
            "video_results": len(videos),
            "related_searches": len(related),
        }

        source = SourceInfo(
            name=f"bing:search:{query[:50]}",
            byte_size=0,
            sha256="",
            detected_format="bing",
            detection_method="query",
            detection_confidence=1.0,
            declared_extension=None,
            extension_mismatch=False,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")

    def _format_duration(self, iso_duration: str) -> str:
        """Convert ISO 8601 duration (PT1H30M15S) to readable format."""
        import re
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
        if not match:
            return iso_duration
        h, m, s = match.groups()
        parts = []
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        if s:
            parts.append(f"{s}s")
        return " ".join(parts) if parts else "0s"
