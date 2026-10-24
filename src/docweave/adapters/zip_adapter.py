"""ZIP archive adapter.

Iterates over archive entries and converts each supported file using the
registry. Produces a structured document with a table of contents and
per-entry extraction results.

This is a compound adapter that delegates to other adapters.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.config import ConversionConfig
from docweave.converter import Converter
from docweave.models import Cell, Document, Section, SourceInfo, Table
from docweave.models.diagnostics import Category, Code, Severity


class ZipAdapter(Adapter):
    name = "zip"
    formats = ("zip",)
    requires = {}  # Uses stdlib zipfile; patoolib optional for nested archives
    capabilities = (
        "lists archive contents",
        "extracts and converts supported entries",
        "preserves directory structure",
        "reports skipped/failed entries",
    )
    limitations = (
        "does not recurse into nested archives by default",
        "encrypted archives not supported",
        "memory usage scales with extracted content",
    )
    cost = "moderate"
    streaming = False

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        config: ConversionConfig = ctx.config
        path: Path = ctx.path

        # Create a sub-converter with the same config but asset extraction enabled
        sub_config = config.merged(extract_assets=True)
        converter = Converter(config=sub_config, registry=ctx.registry)

        document = Document()
        document.container_kind = "body"

        # Build table of contents
        toc_section = Section(title="Archive Contents", level=1)
        toc_section.id = ctx.ids.child(None, "section")

        toc_table = Table()
        toc_table.id = ctx.ids.child(toc_section.id, "table")
        toc_table.header_rows = 1
        toc_table.rows = [[
            Cell(text="Path", is_header=True),
            Cell(text="Size", is_header=True),
            Cell(text="Format", is_header=True),
            Cell(text="Status", is_header=True),
            Cell(text="Chars", is_header=True),
        ]]

        entries_processed = 0
        entries_failed = 0
        entries_skipped = 0

        # Try to open as zip file
        try:
            zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            # Not a valid zip file - this shouldn't happen if detection worked,
            # but handle gracefully
            ctx.warn(
                Code.UNSUPPORTED_STRUCTURE,
                "File has zip magic bytes but is not a valid zip archive",
                category=Category.STRUCTURAL,
                severity=Severity.ERROR,
                scope_kind="archive",
                scope_name=path.name,
            )
            # Return empty document with error
            document = Document()
            document.container_kind = "body"
            source = SourceInfo(
                name=path.name,
                byte_size=path.stat().st_size,
                sha256="",
                detected_format="zip",
                detection_method=ctx.detection.method if ctx.detection else "unknown",
                detection_confidence=ctx.detection.confidence if ctx.detection else 0.0,
                declared_extension=ctx.detection.declared_extension if ctx.detection else None,
                extension_mismatch=ctx.detection.extension_mismatch if ctx.detection else False,
            )
            return ExtractionResult(document=document, source=source, container_kind="body")

        with zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]  # Skip directories
            names.sort()

            for name in names:
                ctx.check_deadline(f"entry {name}")
                info = zf.getinfo(name)

                # Extract to temp location
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    extracted_path = Path(tmpdir) / name
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    zf.extract(name, tmpdir)

                    # Detect format
                    try:
                        result = converter.convert(extracted_path)
                    except Exception as exc:
                        entries_failed += 1
                        status = f"failed: {type(exc).__name__}"
                        fmt = "unknown"
                        chars = 0
                        # Add diagnostic
                        ctx.warn(
                            Code.UNSUPPORTED_STRUCTURE,
                            f"Archive entry '{name}' could not be converted: {exc}",
                            category=Category.UNVERIFIED,
                            severity=Severity.WARNING,
                            scope_kind="archive_entry",
                            scope_name=name,
                            evidence={"error": str(exc)},
                        )
                    else:
                        entries_processed += 1
                        fmt = result.detection.format
                        chars = result.document.text_length()
                        status = "ok" if result.ok else "partial"

                        # Merge assets
                        for asset in result.document.assets:
                            asset.path = f"zip:{name}#{asset.asset_id}"
                            document.assets.append(asset)

                        # Merge body content under a subsection
                        if result.document.body:
                            subsection = Section(title=name, level=2)
                            subsection.id = ctx.ids.child(toc_section.id, "section")
                            for node in result.document.body:
                                # Update source references to include archive path
                                for src in node.sources:
                                    src.note = f"archive entry: {name}; {src.note or ''}"
                                subsection.children.append(node)
                            toc_section.children.append(subsection)

                        # Merge diagnostics
                        for diag in result.diagnostics.diagnostics:
                            diag.scope_kind = "archive_entry"
                            diag.scope_name = name
                            ctx.diagnostics.append(diag)

                    # Add to TOC
                    toc_table.rows.append([
                        Cell(text=name),
                        Cell(text=str(info.file_size)),
                        Cell(text=fmt),
                        Cell(text=status),
                        Cell(text=str(chars)),
                    ])

                    entries_skipped += 1 if status == "skipped" else 0

        toc_section.children.insert(0, toc_table)
        document.body.append(toc_section)

        document.metadata["archive"] = {
            "total_entries": len(names),
            "processed": entries_processed,
            "failed": entries_failed,
            "skipped": entries_skipped,
        }

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",
            detected_format="zip",
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")
