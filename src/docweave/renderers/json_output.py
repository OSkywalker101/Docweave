"""Structured JSON and JSONL renderers.

The JSON envelope is the versioned contract for downstream integrations:
``schema_version``, tool identity, source identity, the IR, and the diagnostics
report. Nothing non-deterministic is emitted unless explicitly requested.
"""

from __future__ import annotations

import json
from typing import Any

from docweave.config import ConversionConfig
from docweave.exceptions import RenderError
from docweave.models import DiagnosticReport, Document, Node, Table

SCHEMA_ENVELOPE = "docweave.document/1.0"
SCHEMA_RECORD = "docweave.record/1.0"


def tool_identity() -> dict[str, str]:
    from docweave import __version__

    return {"name": "docweave", "version": __version__}


def render_json(
    document: Document,
    diagnostics: DiagnosticReport,
    config: ConversionConfig | None = None,
    *,
    timing: dict[str, float] | None = None,
) -> str:
    envelope: dict[str, Any] = {
        "schema": SCHEMA_ENVELOPE,
        "schema_version": document.schema_version,
        "tool": tool_identity(),
        "config": (config or ConversionConfig()).describe(),
        "source": document.source.model_dump(mode="json"),
        "adapter": document.adapter,
        "metadata": document.metadata,
        "container_kind": document.container_kind,
        "document": document.model_dump(mode="json", exclude={"body", "pages", "slides", "sheets"}),
        "diagnostics": [d.model_dump(mode="json") for d in diagnostics.sorted()],
        "diagnostic_summary": {
            "total": len(diagnostics.diagnostics),
            "counts": diagnostics.counts(),
            "worst_severity": (
                diagnostics.worst_severity().value if diagnostics.worst_severity() else None
            ),
        },
    }
    if document.body:
        envelope["body"] = [n.model_dump(mode="json") for n in document.body]
    if document.pages:
        envelope["pages"] = [p.model_dump(mode="json") for p in document.pages]
    if document.slides:
        envelope["slides"] = [s.model_dump(mode="json") for s in document.slides]
    if document.sheets:
        envelope["sheets"] = [s.model_dump(mode="json") for s in document.sheets]
    if document.assets:
        envelope["assets"] = [a.model_dump(mode="json") for a in document.assets]
    if timing:
        # Only ever emitted when explicitly requested: it is the one field that
        # is not reproducible across runs.
        envelope["timing"] = timing

    try:
        return json.dumps(envelope, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise RenderError(f"Document could not be serialised to JSON: {exc}") from exc


def iter_records(document: Document) -> list[dict[str, Any]]:
    """One record per content node, in canonical reading order.

    This is the shape RAG pipelines usually want: small, streamable records that
    keep their provenance and their container identity.
    """
    records: list[dict[str, Any]] = []
    owner_kind, owner = _first_owner(document)
    del owner_kind, owner

    for kind, container, node in document.iter_nodes_with_owner():
        record: dict[str, Any] = {
            "schema": SCHEMA_RECORD,
            "id": node.id,
            "type": _type_name(node),
            "container": kind,
            "origin": node.origin.value,
            "confidence": node.confidence.value,
            "text": node.text_of(),
        }
        if kind == "page":
            record["page"] = getattr(container, "index", None)
            record["bbox"] = _bbox(node)
        elif kind == "slide":
            record["slide"] = getattr(container, "index", None)
            record["slide_title"] = getattr(container, "title", None)
        elif kind == "sheet":
            record["sheet"] = getattr(container, "name", None)
        if node.warnings:
            record["warnings"] = list(node.warnings)
        if node.meta:
            record["meta"] = node.meta
        if isinstance(node, Table):
            record["table"] = _table_payload(node)
        records.append(record)
    return records


def render_jsonl(document: Document) -> str:
    lines: list[str] = []
    for record in iter_records(document):
        try:
            lines.append(json.dumps(record, ensure_ascii=False))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise RenderError(f"Record could not be serialised: {exc}") from exc
    return "\n".join(lines) + ("\n" if lines else "")


def render_report(document: Document, diagnostics: DiagnosticReport) -> str:
    payload = {
        "schema": "docweave.report/1.0",
        "tool": tool_identity(),
        "source": document.source.model_dump(mode="json"),
        "container_kind": document.container_kind,
        "counts": {
            "nodes": document.node_count(),
            "pages": len(document.pages),
            "slides": len(document.slides),
            "sheets": len(document.sheets),
            "tables": sum(1 for _ in document.tables()),
            "images": sum(1 for _ in document.images()),
            "characters": document.text_length(),
            "assets": len(document.assets),
        },
        "summary": diagnostics.summary(),
        "diagnostic_counts": diagnostics.counts(),
        "diagnostics": [d.model_dump(mode="json") for d in diagnostics.sorted()],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _table_payload(table: Table) -> dict[str, Any]:
    return {
        "header_rows": table.header_rows,
        "structure": table.structure,
        "row_count": len(table.rows),
        "column_count": table.column_count(),
        "rectangular": table.is_rectangular(),
        "rows": [
            [
                {
                    "text": c.text,
                    "row": c.row,
                    "column": c.column,
                    "is_header": c.is_header,
                    "row_span": c.row_span,
                    "column_span": c.column_span,
                    "formula": c.formula,
                    "value": c.value,
                    "number_format": c.number_format,
                    "reference": c.reference,
                }
                for c in row
            ]
            for row in table.rows
        ],
    }


def _type_name(node: Node) -> str:
    return type(node).__name__.lower()


def _bbox(node: Node) -> list[float] | None:
    for source in node.sources:
        if source.locator.bbox:
            return [round(v, 3) for v in source.locator.bbox]
    return None


def _first_owner(document: Document):
    for kind, container in document.owned_containers():
        return kind, container
    return None, None
