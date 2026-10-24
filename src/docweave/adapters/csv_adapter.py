"""Delimited text adapter (CSV / TSV).

Tables are the point: rows, columns and header relationships are preserved as a
real table, not flattened into prose. Delimiter is taken from the detected
format but verified against the content, because a ``.csv`` full of tabs is a
real failure mode.
"""

from __future__ import annotations

import csv
import io

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import char_locator, clean_text, read_text, ref
from docweave.detect import FMT_TSV
from docweave.models import (
    Cell,
    Confidence,
    Document,
    Origin,
    SourceInfo,
    Table,
)
from docweave.models.diagnostics import Category, Code


class DelimitedAdapter(Adapter):
    name = "delimited"
    formats = ("csv", "tsv")
    capabilities = (
        "RFC 4180 quoting",
        "header row detection",
        "ragged row preservation",
        "streamed row by row with a configurable cap",
        "delimiter verified against content rather than trusted from the extension",
    )
    limitations = (
        "no type inference: every value stays a string",
        "no dialect detection beyond delimiter confirmation",
        "a ragged file may be reported as table-shaped even when it is prose",
    )
    streaming = True
    cost = "cheap"

    def __init__(self, detected_format: str = "csv") -> None:
        #: Set by the registry per detected format, so a TSV is never parsed
        #: with a comma and vice versa.
        self.detected_format = detected_format

    def delimiter_for(self, ctx: ExtractionContext) -> str:
        return "\t" if self.detected_format == FMT_TSV else ","

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        raw, encoding, lossy = read_text(ctx.path)
        delimiter = self.delimiter_for(ctx)

        # Verify the declared delimiter actually appears.
        head = raw[:8192]
        if delimiter not in head:
            alt = "\t" if delimiter == "," else ","
            if alt in head:
                ctx.warn(
                    Code.PARSER_FALLBACK,
                    f"Delimiter switched to {repr(alt)}: {repr(delimiter)} does not occur in the content.",
                    category=Category.STRUCTURAL,
                    reason="File extension/format disagrees with the bytes on disk.",
                    evidence={"declared": delimiter, "used": alt},
                )
                delimiter = alt

        max_rows = ctx.config.max_csv_rows
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        rows: list[list[Cell]] = []
        header_rows = 0
        offset = 0
        truncated = False
        try:
            for r, values in enumerate(reader):
                ctx.check_deadline(f"CSV row {r}")
                if r >= max_rows:
                    truncated = True
                    break
                is_header = r == 0
                cells = _make_cells(ctx, values, r, is_header, offset)
                rows.append(cells)
                if is_header:
                    header_rows = 1
                offset += sum(len(v) for v in values) + len(values)
        except csv.Error as exc:
            ctx.warn(
                Code.PARSER_FALLBACK,
                f"Delimited parsing stopped early: {exc}",
                category=Category.STRUCTURAL,
                reason="The file is not valid delimited text.",
            )

        if truncated:
            ctx.warn(
                Code.LIMIT_EXCEEDED,
                f"Only the first {max_rows} rows were converted.",
                category=Category.INTERVENTION,
                severity="warning",
                reason="Row budget reached.",
                hint="Raise ConversionConfig.max_csv_rows to include the remaining rows.",
                evidence={"max_csv_rows": max_rows},
            )

        if lossy:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"Input decoded with {encoding} and required replacement characters.",
                category=Category.STRUCTURAL,
            )

        # Ragged rows are reported once, by the validator, which owns that code.

        table = Table(
            id=ctx.ids.child(None, "table"),
            text="",
            rows=rows,
            header_rows=header_rows,
            structure="grid",
            sources=[
                ref(
                    self.name,
                    char_locator(0, len(raw)),
                    origin=Origin.OBSERVED,
                    confidence=Confidence.HIGH,
                )
            ],
        )
        doc = Document(
            adapter=self.name,
            container_kind="body",
            body=[table],
            metadata={
                "encoding": encoding,
                "delimiter": delimiter,
                "row_count": len(rows),
                "column_count": table.column_count(),
                "truncated": truncated,
            },
        )
        return ExtractionResult(
            document=doc,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )


def _make_cells(ctx, values: list[str], row: int, is_header: bool, offset: int) -> list[Cell]:
    cells: list[Cell] = []
    for col, value in enumerate(values):
        text, _ = clean_text(value, normalize=False)
        cells.append(
            Cell(
                id=ctx.ids.child(None, "c"),
                text=text,
                row=row,
                column=col,
                is_header=is_header,
                locator=char_locator(offset, len(text)),
            )
        )
    return cells


def _check_ragged(ctx: ExtractionContext, rows: list[list[Cell]]) -> None:
    if len(rows) < 2:
        return
    widths = {len(r) for r in rows}
    if len(widths) == 1:
        return
    most_common = max(widths)
    offenders = [r for r in rows if len(r) != most_common]
    ctx.warn(
        Code.RAGGED_TABLE,
        f"{len(offenders)} of {len(rows)} rows have a column count different from {most_common}.",
        category=Category.STRUCTURAL,
        reason="Row field counts disagree, so column alignment is ambiguous.",
        hint="Pad the rows in the source file, or inspect the table in the JSON output.",
        evidence={"observed_widths": sorted(widths), "row_count": len(rows)},
    )
