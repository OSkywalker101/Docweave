"""Legacy Excel (.xls) adapter.

Uses xlrd 2.0+ to parse BIFF-format Excel files (Excel 97-2003).
Extracts sheets, cells, formulas, merged cells, and formatting hints.

Note: xlrd 2.0+ only supports .xls format, not .xlsx.
"""

from __future__ import annotations

from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.models import Cell, Document, SourceInfo, Table, Worksheet


class XlsAdapter(Adapter):
    name = "xls"
    formats = ("xls",)
    requires = {"xlrd": "xls"}
    capabilities = (
        "Excel 97-2003 (.xls) BIFF format",
        "multiple worksheets",
        "cell values, types, and formulas",
        "merged cells",
        "sheet visibility",
        "named ranges (metadata only)",
        "cell formatting hints (font, alignment, borders)",
    )
    limitations = (
        "no .xlsx support (use xlsx adapter)",
        "no VBA/macro extraction",
        "no chart/pivot table extraction",
        "formula results not calculated (only stored cached values)",
        "limited formatting fidelity",
    )
    cost = "cheap"
    streaming = False

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        import xlrd

        path: Path = ctx.path

        ctx.check_deadline("open workbook")
        book = xlrd.open_workbook(str(path), formatting_info=True)

        document = Document()
        document.container_kind = "sheets"

        for sheet_index in range(book.nsheets):
            ctx.check_deadline(f"sheet {sheet_index}")
            sheet = book.sheet_by_index(sheet_index)

            # Skip empty sheets
            if sheet.nrows == 0 and sheet.ncols == 0:
                ws = Worksheet(name=sheet.name, index=sheet_index + 1, hidden=sheet.visibility != 0)
                ws.id = ctx.ids.next("sheet")
                document.sheets.append(ws)
                continue

            ws = Worksheet(name=sheet.name, index=sheet_index + 1, hidden=sheet.visibility != 0)
            ws.id = ctx.ids.next("sheet")

            # Dimension
            if sheet.nrows > 0 and sheet.ncols > 0:
                ws.dimension = f"A1:{self._col_to_a1(sheet.ncols - 1)}{sheet.nrows}"

            # Build table
            table = Table()
            table.id = ctx.ids.next("table")
            table.sparse = True  # XLS sheets are inherently sparse

            # Process merged cells
            merged = {}
            for crange in sheet.merged_cells:
                rlo, rhi, clo, chi = crange
                for r in range(rlo, rhi):
                    for c in range(clo, chi):
                        if r == rlo and c == clo:
                            merged[(r, c)] = (rhi - rlo, chi - clo)
                        else:
                            merged[(r, c)] = None  # Mark as covered

            formula_count = 0
            for row_idx in range(sheet.nrows):
                row_cells = []
                for col_idx in range(sheet.ncols):
                    cell_obj = sheet.cell(row_idx, col_idx)
                    cell = self._convert_cell(cell_obj, sheet, row_idx, col_idx, merged, book)
                    if cell:
                        if cell.formula:
                            formula_count += 1
                        row_cells.append(cell)
                    else:
                        # Empty cell - still add for coordinate preservation in sparse mode
                        cell = Cell(
                            id=ctx.ids.next("cell"),
                            text="",
                            row=row_idx,
                            column=col_idx,
                            locator=self._make_locator(sheet.name, row_idx, col_idx),
                        )
                        row_cells.append(cell)

                if row_cells:
                    table.rows.append(row_cells)

            # Detect header row (first non-empty row with mostly text)
            if table.rows:
                table.header_rows = 1

            ws.table = table
            ws.formula_count = formula_count
            ws.merged_range_count = len(sheet.merged_cells)

            # Add sheet-level metadata
            ws.meta = {
                "sheet_index": sheet_index,
                "nrows": sheet.nrows,
                "ncols": sheet.ncols,
            }

            document.sheets.append(ws)

        # Named ranges (metadata only)
        if book.name_map:
            document.metadata["xls_named_ranges"] = {
                name: [ref for ref in refs]
                for name, refs in book.name_map.items()
            }

        document.metadata["xls"] = {
            "sheet_count": book.nsheets,
            "sheets": [s.name for s in book.sheets()],
            "code_name": book.code_name,
        }

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",
            detected_format="xls",
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="sheets")

    def _convert_cell(self, cell_obj, sheet, row_idx: int, col_idx: int, merged: dict, book) -> Cell | None:
        """Convert xlrd cell to DocWeave Cell."""
        from docweave.ids import IdFactory

        # Create a temporary ID factory for cell IDs
        ids = IdFactory()

        ctype = cell_obj.ctype
        value = cell_obj.value

        # Determine cell text
        if ctype == xlrd.XL_CELL_EMPTY:
            text = ""
        elif ctype == xlrd.XL_CELL_TEXT:
            text = str(value)
        elif ctype == xlrd.XL_CELL_NUMBER:
            # Format number appropriately
            if value == int(value):
                text = str(int(value))
            else:
                text = str(value)
        elif ctype == xlrd.XL_CELL_DATE:
            # Convert Excel date to ISO format
            try:
                dt = xlrd.xldate_as_datetime(value, book.datemode)
                text = dt.isoformat()
            except Exception:
                text = str(value)
        elif ctype == xlrd.XL_CELL_BOOLEAN:
            text = "TRUE" if value else "FALSE"
        elif ctype == xlrd.XL_CELL_ERROR:
            text = f"#ERROR({value})"
        elif ctype == xlrd.XL_CELL_BLANK:
            text = ""
        else:
            text = str(value)

        # Check for formula
        formula = None
        if hasattr(cell_obj, 'formula') and cell_obj.formula:
            formula = cell_obj.formula

        # Check merged
        row_span = 1
        col_span = 1
        if (row_idx, col_idx) in merged:
            span = merged[(row_idx, col_idx)]
            if span:
                row_span, col_span = span

        cell = Cell(
            id=ids.next("cell"),
            text=text,
            row=row_idx,
            column=col_idx,
            is_header=row_idx == 0 and table.header_rows > 0,
            row_span=row_span,
            column_span=col_span,
            formula=formula,
            number_format=self._get_number_format(cell_obj, book),
            locator=self._make_locator(sheet.name, row_idx, col_idx),
        )

        return cell

    def _get_number_format(self, cell_obj, book) -> str | None:
        """Extract number format string from cell."""
        try:
            if hasattr(cell_obj, 'xf_index') and cell_obj.xf_index is not None:
                xf = book.xf_list[cell_obj.xf_index]
                fmt = book.format_map.get(xf.format_key)
                if fmt:
                    return fmt.format_str
        except Exception:
            pass
        return None

    def _make_locator(self, sheet_name: str, row: int, col: int):
        from docweave.models import Locator
        return Locator(
            kind="sheet",
            sheet=sheet_name,
            cell=self._col_to_a1(col) + str(row + 1),
            row=row,
            column=col,
        )

    def _col_to_a1(self, col: int) -> str:
        """Convert 0-based column index to A1 notation."""
        result = ""
        col += 1
        while col > 0:
            col, remainder = divmod(col - 1, 26)
            result = chr(65 + remainder) + result
        return result
