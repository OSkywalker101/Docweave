"""Validation: turn extraction signals into an honest, classified report.

The rules here are deliberately cheap and explainable. Each check answers one
question a downstream consumer actually has: *did this conversion lose anything,
and how confident should I be about that?*

What this module will never do: claim that the absence of a diagnostic proves
the content is correct. Absence of evidence is not evidence of absence.
"""

from __future__ import annotations

from docweave.config import ConversionConfig
from docweave.models import Diagnostic, DiagnosticReport, Document, Equation
from docweave.models.diagnostics import Category, Code, Severity

#: A document shorter than this is treated as a stub, not a conversion failure.
_MIN_REASONABLE_CHARS = 1


def validate(document: Document, config: ConversionConfig) -> DiagnosticReport:
    """Run every applicable check against an extracted document."""
    report = DiagnosticReport()

    _check_empty(document, report)
    _check_pages(document, config, report)
    _check_tables(document, report)
    _check_images(document, report)
    _check_equations(document, report)
    _check_node_provenance(document, report)
    return report


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def _check_empty(document: Document, report: DiagnosticReport) -> None:
    chars = document.text_length()
    nodes = document.node_count()
    if nodes == 0 or chars < _MIN_REASONABLE_CHARS:
        report.add(
            Diagnostic(
                code=Code.NO_CONTENT,
                category=Category.STRUCTURAL,
                severity=Severity.ERROR,
                message="No text content was extracted from this document.",
                reason=f"{nodes} node(s) and {chars} character(s) were produced.",
                hint="If this document is a scan, install 'docweave[ocr]' and run the OCR repair pass.",
                evidence={"node_count": nodes, "character_count": chars},
            )
        )


def _check_pages(document: Document, config: ConversionConfig, report: DiagnosticReport) -> None:
    """Per-page judgements, driven entirely by signals the adapter collected.

    This function is the single owner of page-scoped codes, so a given problem
    is reported once rather than once per layer that noticed it.
    """
    if not document.pages:
        return
    total = len(document.pages)

    for page in document.pages:
        s = page.signals
        has_blocks = bool(page.blocks) or bool(page.images)
        heavy_images = s.image_count > 0 and s.media_area_ratio >= config.image_heavy_area_ratio
        scanned = (
            config.detect_scanned_pages
            and s.char_count <= config.scanned_max_chars
            and (heavy_images or s.image_count > 0)
        )

        if page.rotation:
            report.add(
                Diagnostic(
                    code=Code.READING_ORDER_INFERRED,
                    category=Category.HEURISTIC,
                    severity=Severity.INFO,
                    message=f"Page {page.index} is rotated by {page.rotation}°; text order may be affected.",
                    reason="Rotation changes how blocks map onto the reading order.",
                    scope_kind="page",
                    scope_index=page.index,
                    evidence={"rotation": page.rotation},
                )
            )

        if s.column_count > 1:
            _add_multi_column(report, page)

        if s.replacement_chars:
            ratio = s.replacement_chars / max(s.char_count, 1)
            if ratio > config.replacement_char_ratio:
                report.add(
                    Diagnostic(
                        code=Code.REPLACEMENT_CHARS,
                        category=Category.STRUCTURAL,
                        severity=Severity.WARNING,
                        message=(
                            f"Page {page.index} contains {s.replacement_chars} replacement "
                            f"character(s) ({ratio:.2%}); some glyphs could not be mapped to "
                            "Unicode."
                        ),
                        reason="The embedded font encoding is incomplete or non-standard.",
                        scope_kind="page",
                        scope_index=page.index,
                        evidence={
                            "replacement_chars": s.replacement_chars,
                            "char_count": s.char_count,
                            "ratio": round(ratio, 6),
                            "threshold": config.replacement_char_ratio,
                        },
                    )
                )

        if scanned:
            report.add(
                Diagnostic(
                    code=Code.LIKELY_SCANNED,
                    category=Category.UNVERIFIED,
                    severity=Severity.WARNING,
                    message=(
                        f"Page {page.index} of {total} has almost no text ({s.char_count} chars) "
                        f"but {s.image_count} image(s) covering {s.media_area_ratio:.0%} of the "
                        "page. OCR was not run, so pixel text is missing from the output."
                    ),
                    reason="A raster layer with no usable text layer is the signature of a scan.",
                    hint=(
                        "Install 'docweave[ocr]' and Tesseract, then run the optional OCR repair "
                        "pass for this document."
                    ),
                    scope_kind="page",
                    scope_index=page.index,
                    evidence={
                        "char_count": s.char_count,
                        "image_count": s.image_count,
                        "media_area_ratio": round(s.media_area_ratio, 4),
                    },
                )
            )
            continue

        if not has_blocks and s.char_count == 0 and s.image_count == 0:
            report.add(
                Diagnostic(
                    code=Code.EMPTY_CONTAINER,
                    category=Category.STRUCTURAL,
                    severity=Severity.WARNING,
                    message=f"Page {page.index} of {total} produced no text and no images.",
                    reason="The page appears blank, or its content stream is unreadable.",
                    scope_kind="page",
                    scope_index=page.index,
                )
            )
            continue

        if 0 < s.char_count < config.min_chars_per_page:
            report.add(
                Diagnostic(
                    code=Code.LOW_TEXT_DENSITY,
                    category=Category.HEURISTIC,
                    severity=Severity.INFO,
                    message=(
                        f"Page {page.index} yielded {s.char_count} characters, below the "
                        f"heuristic threshold of {config.min_chars_per_page}."
                    ),
                    reason="Low text yield is normal for cover, divider and figure-only pages.",
                    scope_kind="page",
                    scope_index=page.index,
                    evidence={"char_count": s.char_count, "threshold": config.min_chars_per_page},
                )
            )


def _add_multi_column(report: DiagnosticReport, page) -> None:
    s = page.signals
    report.add(
        Diagnostic(
            code=Code.MULTI_COLUMN,
            category=Category.HEURISTIC,
            severity=Severity.INFO,
            message=(
                f"Page {page.index} appears to have {s.column_count} text columns; reading "
                "order was reconstructed from bounding boxes."
            ),
            reason="A vertical whitespace gutter splits the text into columns.",
            hint="Spot-check paragraph order on multi-column pages.",
            scope_kind="page",
            scope_index=page.index,
            evidence={"column_count": s.column_count},
        )
    )


def _check_tables(document: Document, report: DiagnosticReport) -> None:
    for table in document.tables():
        if not table.rows:
            report.add(
                Diagnostic(
                    code=Code.EMPTY_CONTAINER,
                    category=Category.STRUCTURAL,
                    severity=Severity.WARNING,
                    message="A table was recovered with no rows.",
                    reason="Table detection produced an empty grid.",
                    node_id=table.id,
                )
            )
            continue
        if not table.sparse and not table.is_rectangular():
            widths = sorted({len(r) for r in table.rows})
            report.add(
                Diagnostic(
                    code=Code.RAGGED_TABLE,
                    category=Category.STRUCTURAL,
                    severity=Severity.WARNING,
                    message=(
                        f"Table has {len(table.rows)} rows with differing column counts {widths}."
                    ),
                    reason="Ragged rows make column identity ambiguous in Markdown output.",
                    hint="Use the JSON output, which keeps per-cell row/column coordinates.",
                    node_id=table.id,
                    evidence={"row_widths": widths, "row_count": len(table.rows)},
                )
            )
        empty = sum(1 for row in table.rows for cell in row if not cell.text.strip())
        total = sum(len(row) for row in table.rows)
        if total and empty / total > 0.5:
            report.add(
                Diagnostic(
                    code=Code.EMPTY_CELLS,
                    category=Category.HEURISTIC,
                    severity=Severity.INFO,
                    message=f"{empty} of {total} table cells are empty.",
                    reason="A majority of empty cells usually indicates a mis-detected table region.",
                    node_id=table.id,
                    evidence={"empty_cells": empty, "total_cells": total},
                )
            )


def _check_images(document: Document, report: DiagnosticReport) -> None:
    described = 0
    total = 0
    for image in document.images():
        total += 1
        if image.alt:
            described += 1
    if not total:
        return
    if report.has(Code.LIKELY_SCANNED):
        # The per-page scanned diagnostic already says image content was not
        # interpreted; repeating it at document level would just be noise.
        return
    if described == 0:
        report.add(
            Diagnostic(
                code=Code.IMAGE_WITHOUT_TEXT,
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                message=(
                    f"{total} image(s) are referenced but none carry a description; "
                    "their visual content was not interpreted."
                ),
                reason="DocWeave does not run a vision model by default.",
                hint="Supply alt text upstream, or enable the optional OCR pass for pixel text.",
                evidence={"image_count": total, "with_alt": described},
            )
        )


def _check_equations(document: Document, report: DiagnosticReport) -> None:
    for node in document.iter_nodes():
        if isinstance(node, Equation) and not node.latex:
            report.add(
                Diagnostic(
                    code=Code.EQUATION_NOT_RECOVERED,
                    category=Category.UNVERIFIED,
                    severity=Severity.WARNING,
                    message="An equation was detected but no LaTeX could be recovered.",
                    reason="Math recognition is not part of the default local path.",
                    node_id=node.id,
                )
            )


def _check_node_provenance(document: Document, report: DiagnosticReport) -> None:
    missing = [n for n in document.iter_nodes() if not n.sources]
    if not missing:
        return
    report.add(
        Diagnostic(
            code=Code.UNSUPPORTED_STRUCTURE,
            category=Category.UNVERIFIED,
            severity=Severity.INFO,
            message=f"{len(missing)} node(s) have no source reference attached.",
            reason="Nodes without provenance cannot be traced back to a page, slide or cell.",
            evidence={"node_count": len(missing)},
        )
    )
