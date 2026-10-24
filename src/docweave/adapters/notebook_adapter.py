"""Jupyter Notebook (.ipynb) adapter.

Parses notebook JSON structure and converts to DocWeave IR preserving:
- Markdown cells as headings/paragraphs
- Code cells with syntax highlighting
- Output cells (text, HTML, images, errors)
- Execution metadata
- Notebook-level metadata (kernelspec, language_info)
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.models import Cell, CodeBlock, Document, Origin, Paragraph, Section, SourceInfo, SourceRef, Table


class NotebookAdapter(Adapter):
    name = "notebook"
    formats = ("ipynb",)
    requires = {"nbformat": "notebook"}
    capabilities = (
        "parses nbformat v4 notebooks",
        "markdown cells → headings/paragraphs",
        "code cells → code blocks with language",
        "outputs: text, HTML, images, JSON, errors",
        "execution metadata (timing, counts)",
        "notebook metadata (kernelspec, language_info)",
        "cell attachments as assets",
    )
    limitations = (
        "widget state not preserved",
        "interactive outputs (JavaScript) not rendered",
        "large base64 images increase memory",
    )
    cost = "cheap"
    streaming = False

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        import nbformat

        path: Path = ctx.path

        ctx.check_deadline("parse notebook")
        with path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        document = Document()
        document.container_kind = "body"

        # Notebook metadata section
        meta_section = Section(title="Notebook Metadata", level=1)
        meta_section.id = ctx.ids.next("section")

        meta_table = Table()
        meta_table.id = ctx.ids.next("table")
        meta_table.header_rows = 1
        meta_table.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

        # Kernelspec
        kernelspec = nb.metadata.get("kernelspec", {})
        if kernelspec:
            meta_table.rows.append([Cell(text="Kernel"), Cell(text=kernelspec.get("display_name", kernelspec.get("name", "unknown")))])
            meta_table.rows.append([Cell(text="Language"), Cell(text=kernelspec.get("language", "unknown"))])

        # Language info
        lang_info = nb.metadata.get("language_info", {})
        if lang_info:
            meta_table.rows.append([Cell(text="Language Version"), Cell(text=lang_info.get("version", "unknown"))])
            meta_table.rows.append([Cell(text="Mimetype"), Cell(text=lang_info.get("mimetype", "unknown"))])

        # Cell counts
        code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
        markdown_cells = sum(1 for c in nb.cells if c.cell_type == "markdown")
        raw_cells = sum(1 for c in nb.cells if c.cell_type == "raw")
        meta_table.rows.append([Cell(text="Total Cells"), Cell(text=str(len(nb.cells)))])
        meta_table.rows.append([Cell(text="Code Cells"), Cell(text=str(code_cells))])
        meta_table.rows.append([Cell(text="Markdown Cells"), Cell(text=str(markdown_cells))])
        meta_table.rows.append([Cell(text="Raw Cells"), Cell(text=str(raw_cells))])

        meta_section.children.append(meta_table)
        document.body.append(meta_section)

        # Cells
        cells_section = Section(title=f"Notebook Cells ({len(nb.cells)})", level=1)
        cells_section.id = ctx.ids.next("section")

        for i, cell in enumerate(nb.cells):
            ctx.check_deadline(f"cell {i}")
            cell_section = Section(title=f"Cell {i+1} [{cell.cell_type}]", level=2)
            cell_section.id = ctx.ids.next("section")

            # Cell metadata table
            cell_meta = Table()
            cell_meta.id = ctx.ids.next("table")
            cell_meta.header_rows = 1
            cell_meta.rows = [[Cell(text="Property", is_header=True), Cell(text="Value", is_header=True)]]

            if cell.cell_type == "code":
                execution_count = cell.get("execution_count")
                if execution_count is not None:
                    cell_meta.rows.append([Cell(text="Execution Count"), Cell(text=str(execution_count))])

                # Language from kernelspec or metadata
                lang = kernelspec.get("language", "python").lower()
                cell_meta.rows.append([Cell(text="Language"), Cell(text=lang)])

                # Source code
                source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
                if source.strip():
                    code_block = CodeBlock(text=source, language=lang)
                    code_block.id = ctx.ids.next("code")
                    code_block.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"code cell {i+1}",
                        )
                    ]
                    cell_section.children.append(code_block)

                # Outputs
                for j, output in enumerate(cell.outputs):
                    self._process_output(output, ctx, cell_section, f"cell {i+1}, output {j+1}")

            elif cell.cell_type == "markdown":
                source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
                if source.strip():
                    # Parse markdown for headings
                    for para_text in source.split("\n\n"):
                        para_text = para_text.strip()
                        if not para_text:
                            continue
                        # Check for heading
                        if para_text.startswith("#"):
                            level = len(para_text) - len(para_text.lstrip("#"))
                            heading_text = para_text.lstrip("# ").strip()
                            # Add as paragraph with heading metadata
                            para = Paragraph(text=heading_text)
                            para.id = ctx.ids.next("paragraph")
                            para.meta = {"heading_level": level, "original": para_text}
                            para.sources = [
                                SourceRef(
                                    adapter=self.name,
                                    origin=Origin.OBSERVED,
                                    confidence="high",
                                    note=f"markdown cell {i+1}",
                                )
                            ]
                            cell_section.children.append(para)
                        else:
                            para = Paragraph(text=para_text)
                            para.id = ctx.ids.next("paragraph")
                            para.sources = [
                                SourceRef(
                                    adapter=self.name,
                                    origin=Origin.OBSERVED,
                                    confidence="high",
                                    note=f"markdown cell {i+1}",
                                )
                            ]
                            cell_section.children.append(para)

            elif cell.cell_type == "raw":
                source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
                if source.strip():
                    para = Paragraph(text=source)
                    para.id = ctx.ids.next("paragraph")
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"raw cell {i+1}",
                        )
                    ]
                    cell_section.children.append(para)

            # Cell metadata
            if cell.metadata:
                for key, val in cell.metadata.items():
                    if key not in ("slideshow",):  # Skip verbose metadata
                        cell_meta.rows.append([Cell(text=key), Cell(text=json.dumps(val)[:200])])

            if len(cell_meta.rows) > 1:
                cell_section.children.insert(0, cell_meta)

            cells_section.children.append(cell_section)

        document.body.append(cells_section)

        document.metadata["notebook"] = {
            "nbformat": nb.nbformat,
            "nbformat_minor": nb.nbformat_minor,
            "kernelspec": kernelspec,
            "language_info": lang_info,
            "cell_count": len(nb.cells),
            "code_cells": code_cells,
            "markdown_cells": markdown_cells,
        }

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",
            detected_format="ipynb",
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")

    def _process_output(self, output: dict, ctx: ExtractionContext, section: Section, note: str):
        """Process a notebook cell output."""
        output_type = output.get("output_type")

        if output_type == "stream":
            text = output.get("text", "")
            if text.strip():
                for line in text.strip().split("\n"):
                    para = Paragraph(text=line)
                    para.id = ctx.ids.next("paragraph")
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"{note} (stdout/stderr)",
                        )
                    ]
                    section.children.append(para)

        elif output_type == "execute_result":
            data = output.get("data", {})
            self._process_mime_data(data, ctx, section, note)

        elif output_type == "display_data":
            data = output.get("data", {})
            self._process_mime_data(data, ctx, section, note)

        elif output_type == "error":
            ename = output.get("ename", "Error")
            evalue = output.get("evalue", "")
            traceback = output.get("traceback", [])
            error_text = f"{ename}: {evalue}\n" + "\n".join(traceback)
            para = Paragraph(text=error_text)
            para.id = ctx.ids.next("paragraph")
            para.sources = [
                SourceRef(
                    adapter=self.name,
                    origin=Origin.OBSERVED,
                    confidence="high",
                    note=f"{note} (error)",
                )
            ]
            section.children.append(para)

    def _process_mime_data(self, data: dict, ctx: ExtractionContext, section: Section, note: str):
        """Process MIME-typed output data."""
        # Priority order for display
        mime_order = [
            "application/vnd.jupyter.widget-view+json",
            "application/vnd.jupyter.widget-state+json",
            "application/javascript",
            "text/html",
            "text/markdown",
            "text/plain",
            "image/png",
            "image/jpeg",
            "image/svg+xml",
            "image/gif",
            "application/json",
            "application/geo+json",
        ]

        for mime in mime_order:
            if mime in data:
                value = data[mime]
                if mime.startswith("text/"):
                    text = "".join(value) if isinstance(value, list) else value
                    if text.strip():
                        if mime == "text/html":
                            # Convert HTML to text
                            from docweave.adapters.html_adapter import HtmlAdapter
                            HtmlAdapter()
                            from docweave.assets import AssetCollector
                            from docweave.config import ConversionConfig
                            from docweave.ids import IdFactory
                            ExtractionContext(
                                path=Path(""),
                                config=ConversionConfig(),
                                ids=IdFactory(),
                                assets=AssetCollector(directory=None, enabled=False),
                            )
                            # We'll just extract text from HTML
                            import re
                            text = re.sub(r"<[^>]+>", "", text)
                            text = re.sub(r"\s+", " ", text).strip()
                        para = Paragraph(text=text)
                        para.id = ctx.ids.next("paragraph")
                        para.sources = [
                            SourceRef(
                                adapter=self.name,
                                origin=Origin.OBSERVED,
                                confidence="high",
                                note=f"{note} ({mime})",
                            )
                        ]
                        section.children.append(para)
                elif mime.startswith("image/"):
                    # Base64 encoded image
                    if isinstance(value, str):
                        img_data = base64.b64decode(value)
                    elif isinstance(value, dict) and "base64" in value:
                        img_data = base64.b64decode(value["base64"])
                    else:
                        continue

                    ext = mimetypes.guess_extension(mime) or ".png"
                    asset = AssetRef(
                        asset_id=ctx.ids.next("asset"),
                        media_type=mime,
                        extension=ext,
                        byte_size=len(img_data),
                        path=f"notebook:{note}#{ctx.ids.next('asset')}{ext}",
                        extracted=False,
                        description=f"Notebook output image: {note}",
                    )
                    document.assets.append(asset)
                    ctx.assets.add(asset)

                    # Add placeholder paragraph
                    para = Paragraph(text=f"[Image: {mime}, {len(img_data)} bytes]")
                    para.id = ctx.ids.next("paragraph")
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"{note} (image)",
                        )
                    ]
                    section.children.append(para)
                elif mime == "application/json":
                    text = json.dumps(value, indent=2) if not isinstance(value, str) else value
                    para = Paragraph(text=text)
                    para.id = ctx.ids.next("paragraph")
                    para.sources = [
                        SourceRef(
                            adapter=self.name,
                            origin=Origin.OBSERVED,
                            confidence="high",
                            note=f"{note} (JSON)",
                        )
                    ]
                    section.children.append(para)
                break  # Only process first matching mime type
