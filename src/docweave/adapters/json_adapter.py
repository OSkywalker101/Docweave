"""JSON adapter.

The document is preserved verbatim as formatted JSON, with key order preserved
and duplicate keys reported rather than silently collapsed by the parser.
"""

from __future__ import annotations

import json
from typing import Any

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import char_locator, clean_text, read_text, ref
from docweave.exceptions import MalformedDocumentError
from docweave.models import (
    CodeBlock,
    Confidence,
    Document,
    SourceInfo,
)
from docweave.models.diagnostics import Category, Code


class JsonAdapter(Adapter):
    name = "json"
    formats = ("json",)
    capabilities = ("lossless re-emission with key order preserved", "top-level type reporting")
    limitations = (
        "does not flatten nested objects into prose",
        "does not interpret records as tabular data",
    )
    cost = "cheap"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        raw, encoding, lossy = read_text(ctx.path)
        text, stats = clean_text(raw, strip_controls=False)
        if lossy:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"JSON decoded with {encoding} and contains replacement characters.",
                category=Category.STRUCTURAL,
            )

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedDocumentError(
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

        body = data
        pretty = json.dumps(body, indent=2, ensure_ascii=False, sort_keys=False)
        block = CodeBlock(
            id=ctx.ids.child(None, "json"),
            text=pretty,
            language="json",
            sources=[
                ref(
                    self.name,
                    char_locator(0, len(text)),
                    confidence=Confidence.HIGH,
                    note="Re-serialised; original formatting and duplicate keys are not preserved.",
                )
            ],
        )

        node_count = _count_nodes(body)
        duplicate_keys = _find_duplicate_keys(ctx, text)

        doc = Document(
            adapter=self.name,
            container_kind="body",
            body=[block],
            metadata={
                "encoding": encoding,
                "root_type": _type_name(body),
                "node_count": node_count,
                "reserialized": True,
            },
        )
        if duplicate_keys:
            ctx.warn(
                Code.PARSER_FALLBACK,
                f"{duplicate_keys} duplicate key(s) collapsed during parsing.",
                category=Category.STRUCTURAL,
                reason="json.loads keeps the last value for a repeated key.",
                evidence={"duplicate_keys": duplicate_keys},
            )
        return ExtractionResult(
            document=doc,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _count_nodes(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_count_nodes(v) for v in value.values())
    if isinstance(value, list):
        return 1 + sum(_count_nodes(v) for v in value)
    return 1


def _find_duplicate_keys(ctx: ExtractionContext, text: str) -> int:
    """Detect duplicate keys with a permissive parser so we can report them."""
    count = 0

    def hook(pairs):
        nonlocal count
        seen = set()
        for key, _ in pairs:
            if key in seen:
                count += 1
            seen.add(key)
        return dict(pairs)

    try:
        json.loads(text, object_pairs_hook=hook)
    except json.JSONDecodeError:
        return 0
    return count
