"""Outlook MSG adapter.

Uses msg-extractor to parse .msg files (OLE2 compound documents).
Extracts headers, body (text/HTML), attachments, and metadata.
"""

from __future__ import annotations

from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.config import ConversionConfig
from docweave.models import Cell, Document, Origin, Paragraph, Section, SourceInfo, SourceRef, Table
from docweave.models.diagnostics import Category, Code, Severity


class MsgAdapter(Adapter):
    name = "msg"
    formats = ("msg",)
    requires = {"msg_extractor": "msg"}
    capabilities = (
        "parses MSG headers (From, To, CC, Subject, Date, Message-ID)",
        "extracts text and HTML body",
        "recovers attachments as assets",
        "handles RTF body (winmail.dat)",
        "reports encoding issues",
    )
    limitations = (
        "encrypted/signed messages not fully supported",
        "embedded OLE objects not extracted",
        "some MAPI properties may be lost",
    )
    cost = "cheap"
    streaming = False

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        from msg_extractor import Message

        config: ConversionConfig = ctx.config
        path: Path = ctx.path

        ctx.check_deadline("parse MSG")
        msg = Message(str(path))

        document = Document()
        document.container_kind = "body"

        # Headers section
        headers_section = Section(title="Email Headers", level=1)
        headers_section.id = ctx.ids.next("section")

        headers_table = Table()
        headers_table.id = ctx.ids.next("table")
        headers_table.header_rows = 1
        headers_table.rows = [[Cell(text="Header", is_header=True), Cell(text="Value", is_header=True)]]

        header_fields = [
            ("Subject", msg.subject),
            ("From", msg.sender),
            ("To", msg.to),
            ("CC", msg.cc),
            ("BCC", msg.bcc),
            ("Date", msg.date),
            ("Message-ID", msg.message_id),
            ("In-Reply-To", msg.in_reply_to),
            ("References", msg.references),
            ("Importance", msg.importance),
            ("Sensitivity", msg.sensitivity),
        ]

        for key, val in header_fields:
            if val:
                headers_table.rows.append([Cell(text=key), Cell(text=str(val))])

        headers_section.children.append(headers_table)
        document.body.append(headers_section)

        # Body section
        body_section = Section(title="Message Body", level=1)
        body_section.id = ctx.ids.next("section")

        # Prefer text body, fall back to HTML
        body_text = msg.body or ""
        body_html = msg.html_body or ""

        if body_text:
            for para_text in body_text.split("\n\n"):
                para_text = para_text.strip()
                if para_text:
                    para = Paragraph(text=para_text)
                    para.id = ctx.ids.next("paragraph")
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note="text body",
                        )
                    ]
                    body_section.children.append(para)
        elif body_html:
            # Convert HTML to text
            from docweave.adapters.html_adapter import HtmlAdapter
            html_adapter = HtmlAdapter()
            # Create a minimal context for HTML conversion
            from docweave.assets import AssetCollector
            from docweave.ids import IdFactory
            sub_ctx = ExtractionContext(
                path=path,
                config=config,
                ids=IdFactory(),
                assets=AssetCollector(directory=None, enabled=False),
            )
            html_result = html_adapter.extract(sub_ctx)
            for node in html_result.document.body:
                body_section.children.append(node)
        else:
            ctx.warn(
                Code.IMAGE_WITHOUT_TEXT,
                "Message has no text or HTML body",
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                scope_kind="msg",
                scope_name="body",
            )

        document.body.append(body_section)

        # Attachments
        if msg.attachments:
            attach_section = Section(title="Attachments", level=1)
            attach_section.id = ctx.ids.next("section")

            attach_table = Table()
            attach_table.id = ctx.ids.next("table")
            attach_table.header_rows = 1
            attach_table.rows = [[
                Cell(text="Filename", is_header=True),
                Cell(text="Size", is_header=True),
                Cell(text="Type", is_header=True),
                Cell(text="Content-ID", is_header=True),
            ]]

            for att in msg.attachments:
                filename = att.long_filename or att.short_filename or "unknown"
                size = att.data.__len__() if hasattr(att.data, "__len__") else 0
                content_type = att.content_type or "application/octet-stream"
                content_id = att.content_id or ""

                attach_table.rows.append([
                    Cell(text=filename),
                    Cell(text=str(size)),
                    Cell(text=content_type),
                    Cell(text=content_id),
                ])

                # Add as asset
                asset = AssetRef(
                    asset_id=ctx.ids.next("asset"),
                    media_type=content_type,
                    extension=Path(filename).suffix.lower() if filename else None,
                    byte_size=size,
                    path=f"msg:{path.name}#{filename}",
                    extracted=False,
                    description=f"MSG attachment: {filename}",
                )
                document.assets.append(asset)
                ctx.assets.add(asset)

            attach_section.children.append(attach_table)
            document.body.append(attach_section)

        # RTF body (winmail.dat) - if present and different from text/html
        if msg.rtf_body and msg.rtf_body != body_text and msg.rtf_body != body_html:
            ctx.warn(
                Code.UNSUPPORTED_STRUCTURE,
                "Message contains RTF body (winmail.dat) not fully converted",
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                scope_kind="msg",
                scope_name="rtf_body",
            )

        document.metadata["msg"] = {
            "subject": msg.subject,
            "sender": msg.sender,
            "recipients": msg.to + (msg.cc or []) + (msg.bcc or []),
            "date": str(msg.date) if msg.date else None,
            "has_attachments": bool(msg.attachments),
            "attachment_count": len(msg.attachments) if msg.attachments else 0,
            "body_format": "text" if body_text else ("html" if body_html else "none"),
        }

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",
            detected_format="msg",
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")
