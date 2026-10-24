"""YouTube transcript adapter.

Fetches video metadata and transcripts using yt-dlp. Supports:
- Automatic captions (multiple languages)
- Manual transcripts
- Video metadata (title, description, duration, chapters, etc.)
- Embedded thumbnails as assets

Note: This adapter works with YouTube URLs, not local files. The detection
happens at the converter level for URLs matching youtube.com/watch?v=...
"""

from __future__ import annotations

import re

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.models import Document, Origin, Paragraph, Section, SourceInfo, SourceRef
from docweave.models.diagnostics import Category, Code, Severity


class YouTubeAdapter(Adapter):
    name = "youtube"
    formats = ("youtube",)
    requires = {"yt_dlp": "youtube"}
    capabilities = (
        "fetches video metadata",
        "downloads automatic/manual transcripts",
        "multiple language support",
        "chapter/segment extraction",
        "thumbnail as asset",
    )
    limitations = (
        "requires network access",
        "no transcript available for some videos",
        "rate limited by YouTube",
        "age-restricted videos need cookies",
    )
    cost = "moderate"
    streaming = False

    # URL patterns
    URL_PATTERNS = [
        r"^https?://(?:www\.)?youtube\.com/watch\?v=([\w-]+)",
        r"^https?://youtu\.be/([\w-]+)",
        r"^https?://(?:www\.)?youtube\.com/embed/([\w-]+)",
        r"^https?://(?:www\.)?youtube\.com/v/([\w-]+)",
    ]

    @classmethod
    def extract_video_id(cls, url: str) -> str | None:
        for pattern in cls.URL_PATTERNS:
            m = re.match(pattern, url)
            if m:
                return m.group(1)
        return None

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        import yt_dlp

        # The "path" for YouTube is actually a URL passed via detection
        url = ctx.detection.media_type or ctx.detection.declared_extension or str(ctx.path)
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {url}")

        ctx.check_deadline("fetch metadata")
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],  # Prefer English
            "subtitlesformat": "vtt",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        document = Document()
        document.container_kind = "body"

        # Video metadata section
        meta_section = Section(title="Video Metadata", level=1)
        meta_section.id = ctx.ids.next("section")

        meta_table = self._build_metadata_table(info, ctx)
        meta_section.children.append(meta_table)

        # Transcript section
        transcript_section = Section(title="Transcript", level=1)
        transcript_section.id = ctx.ids.next("section")

        transcript_text = self._fetch_transcript(info, video_id, ctx)
        if transcript_text:
            # Split into paragraphs by timestamps or blank lines
            paras = self._segment_transcript(transcript_text)
            for para_text in paras:
                para = Paragraph(text=para_text)
                para.id = ctx.ids.next("paragraph")
                para.sources = [
                    SourceRef(
                        adapter=self.name,
                        origin=Origin.OBSERVED,
                        confidence="high",
                        note=f"YouTube transcript for {video_id}",
                    )
                ]
                transcript_section.children.append(para)
        else:
            ctx.warn(
                Code.IMAGE_WITHOUT_TEXT,
                "No transcript available for this video",
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                hint="Video may not have captions enabled",
                scope_kind="youtube",
                scope_name=video_id,
            )
            para = Paragraph(text="[No transcript available]")
            para.id = ctx.ids.next("paragraph")
            transcript_section.children.append(para)

        document.body.extend([meta_section, transcript_section])

        # Thumbnail as asset
        thumb_url = info.get("thumbnail")
        if thumb_url:
            asset = AssetRef(
                asset_id=ctx.ids.next("asset"),
                media_type="image/jpeg",
                extension=".jpg",
                byte_size=0,  # Not downloaded
                path=thumb_url,
                extracted=False,
                description=f"YouTube thumbnail for {video_id}",
            )
            document.assets.append(asset)
            ctx.assets.add(asset)

        # Chapters
        chapters = info.get("chapters")
        if chapters:
            chapters_section = Section(title="Chapters", level=1)
            chapters_section.id = ctx.ids.next("section")
            chap_table = Table()
            chap_table.id = ctx.ids.next("table")
            chap_table.header_rows = 1
            chap_table.rows = [[
                Cell(text="Start", is_header=True),
                Cell(text="End", is_header=True),
                Cell(text="Title", is_header=True),
            ]]
            for ch in chapters:
                chap_table.rows.append([
                    Cell(text=self._format_time(ch.get("start_time", 0))),
                    Cell(text=self._format_time(ch.get("end_time", 0))),
                    Cell(text=ch.get("title", "")),
                ])
            chapters_section.children.append(chap_table)
            document.body.append(chapters_section)

        document.metadata["youtube"] = {
            "video_id": video_id,
            "title": info.get("title"),
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "has_transcript": bool(transcript_text),
            "languages": list(info.get("subtitles", {}).keys()) + list(info.get("automatic_captions", {}).keys()),
        }

        source = SourceInfo(
            name=f"youtube:{video_id}",
            byte_size=0,
            sha256="",
            detected_format="youtube",
            detection_method="url",
            detection_confidence=1.0,
            declared_extension=None,
            extension_mismatch=False,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")

    def _build_metadata_table(self, info: dict, ctx: ExtractionContext):
        table = Table()
        table.id = ctx.ids.next("table")
        table.header_rows = 1
        table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

        fields = [
            ("Title", info.get("title")),
            ("Duration", self._format_time(info.get("duration", 0))),
            ("Uploader", info.get("uploader")),
            ("Channel", info.get("channel")),
            ("View Count", f"{info.get('view_count', 0):,}" if info.get("view_count") else "N/A"),
            ("Like Count", f"{info.get('like_count', 0):,}" if info.get("like_count") else "N/A"),
            ("Upload Date", info.get("upload_date")),
            ("Description", info.get("description", "")[:500] + "..." if info.get("description", "") else ""),
            ("Tags", ", ".join(info.get("tags", [])[:20])),
            ("Categories", ", ".join(info.get("categories", []))),
        ]
        for key, val in fields:
            table.rows.append([Cell(text=key), Cell(text=str(val) if val else "N/A")])
        return table

    def _fetch_transcript(self, info: dict, video_id: str, ctx: ExtractionContext) -> str | None:
        # Try manual subtitles first, then automatic
        subs = info.get("subtitles") or {}
        auto_subs = info.get("automatic_captions") or {}

        for lang in ["en", "en-US", "en-GB"]:
            if lang in subs:
                return self._download_subtitle(subs[lang])
            if lang in auto_subs:
                return self._download_subtitle(auto_subs[lang])

        # Fallback to any language
        for lang, sublist in {**subs, **auto_subs}.items():
            return self._download_subtitle(sublist)

        return None

    def _download_subtitle(self, sublist: list) -> str | None:
        import yt_dlp
        for sub in sublist:
            if sub.get("ext") in ("vtt", "srt", "ttml"):
                url = sub.get("url")
                if url:
                    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                        return ydl.urlopen(url).read().decode("utf-8")
        return None

    def _segment_transcript(self, vtt_text: str) -> list[str]:
        """Parse VTT and segment into paragraphs."""
        import re
        # Remove VTT headers
        lines = vtt_text.split("\n")
        content_lines = []
        in_content = False
        for line in lines:
            if line.startswith("WEBVTT"):
                continue
            if line.strip() == "":
                continue
            if "-->" in line:
                in_content = True
                continue
            if in_content:
                content_lines.append(line)

        # Merge into paragraphs (simple: split on blank lines in original)
        text = "\n".join(content_lines)
        # Clean up VTT markup
        text = re.sub(r"<[^>]+>", "", text)  # Remove tags
        text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\]", "", text)  # Remove timestamps
        # Split on double newline
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paras[:100]  # Limit for sanity

    def _format_time(self, seconds: float) -> str:
        if seconds is None:
            return "N/A"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
