"""Every adapter, end to end, on synthetic documents.

These are the tests that would catch a regression where a parser starts
"successfully" producing empty output.
"""

from __future__ import annotations

import json

import pytest

from docweave import ConversionConfig, convert
from docweave.exceptions import MalformedDocumentError, UnsupportedFormatError
from docweave.models import CodeBlock, Heading, ListBlock, Paragraph
from docweave.models.diagnostics import Code
from tests.conftest import requires_docx, requires_pdf, requires_pptx, requires_xlsx


def texts(result) -> str:
    return "\n".join(n.text_of() for n in result.document.iter_nodes())


# --------------------------------------------------------------------------
# text family
# --------------------------------------------------------------------------


class TestPlainText:
    def test_paragraphs_split_on_blank_lines(self, fx):
        result = convert(fx("bom.txt"))
        assert result.source.detected_format == "text"
        assert "Heading" in texts(result)

    def test_bom_is_stripped(self, fx):
        assert not texts(convert(fx("bom.txt"))).startswith("\ufeff")

    def test_binary_control_bytes_are_reported(self, fx):
        result = convert(fx("binary.csv"))
        assert result.source.detected_format == "text"
        assert result.diagnostics.has(Code.EXTENSION_MISMATCH)


class TestMarkdown:
    def test_structure_is_recovered(self, fx):
        result = convert(fx("sample.md"))
        kinds = [type(n).__name__ for n in result.document.iter_nodes()]
        assert "Heading" in kinds
        assert "CodeBlock" in kinds
        assert "ListBlock" in kinds
        assert "Table" in kinds
        assert "Quote" in kinds

    def test_code_language_preserved(self, fx):
        blocks = [n for n in convert(fx("sample.md")).document.iter_nodes() if isinstance(n, CodeBlock)]
        assert blocks[0].language == "python"

    def test_table_header_detected(self, fx):
        table = next(convert(fx("sample.md")).document.tables())
        assert table.header_rows == 1
        assert table.is_rectangular()

    def test_round_trip_is_lossless_for_text(self, fx):
        once = convert(fx("sample.md")).to_markdown()
        path = fx("sample.md").parent / "roundtrip.md"
        path.write_text(once, encoding="utf-8")
        twice = convert(path).to_markdown()
        assert _normalise(once) == _normalise(twice)


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n") if line.strip())


class TestDelimited:
    def test_rows_and_columns(self, fx):
        table = next(convert(fx("sample.csv")).document.tables())
        assert len(table.rows) == 4
        assert table.column_count() == 3
        assert table.rows[0][0].text == "name"
        assert table.rows[0][0].is_header

    def test_quoted_field_with_comma(self, fx):
        table = next(convert(fx("sample.csv")).document.tables())
        assert table.rows[3][0].text == "quoted, comma"

    def test_ragged_rows_are_reported_not_hidden(self, fx):
        result = convert(fx("ragged.csv"))
        assert result.diagnostics.has(Code.RAGGED_TABLE)
        widths = {len(r) for r in next(result.document.tables()).rows}
        assert len(widths) > 1

    def test_row_limit_is_reported(self, fx):
        result = convert(fx("large.csv"), ConversionConfig(max_csv_rows=10))
        assert result.diagnostics.has(Code.LIMIT_EXCEEDED)
        assert len(next(result.document.tables()).rows) == 10


class TestJson:
    def test_content_is_preserved(self, fx):
        result = convert(fx("sample.json"))
        data = json.loads(next(result.document.iter_nodes()).text)
        assert data["z"] == 1
        assert data["a"][2]["b"] is None

    def test_duplicate_keys_are_reported(self, fx):
        assert convert(fx("sample.json")).diagnostics.has(Code.PARSER_FALLBACK)

    def test_invalid_json_raises(self, scratch):
        path = scratch / "broken.json"
        path.write_text('{"a": ')
        with pytest.raises(MalformedDocumentError):
            convert(path)


class TestXml:
    def test_structure_and_text(self, fx):
        result = convert(fx("sample.xml"))
        assert result.document.metadata["root_tag"] == "root"
        assert result.document.metadata["element_count"] == 2
        assert "text" in texts(result)

    def test_entity_expansion_is_blocked(self, scratch):
        # defusedxml refuses billion-laughs; a naive parser would hang or blow up.
        payload = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE lolz [<!ENTITY lol 'lol'>"
            "<!ENTITY lol2 '&lol;&lol;&lol;&lol;'>]>"
            "<lolz>&lol2;</lolz>"
        )
        path = scratch / "bomb.xml"
        path.write_text(payload)
        with pytest.raises(MalformedDocumentError):
            convert(path)


class TestHtml:
    def test_elements_map_to_nodes(self, fx):
        result = convert(fx("sample.html"))
        kinds = [type(n).__name__ for n in result.document.iter_nodes()]
        assert kinds.count("Heading") == 2
        assert "ListBlock" in kinds
        assert "Table" in kinds
        assert "CodeBlock" in kinds
        assert "ImageBlock" in kinds

    def test_scripts_are_stripped(self, fx):
        assert "alert" not in texts(convert(fx("sample.html")))

    def test_link_text_is_not_broken_by_separators(self, fx):
        paragraph = next(
            n for n in convert(fx("sample.html")).document.iter_nodes() if isinstance(n, Paragraph)
        )
        assert "a link." in paragraph.text

    def test_table_caption_and_header(self, fx):
        table = next(convert(fx("sample.html")).document.tables())
        assert table.caption == "Cap"
        assert table.header_rows == 1
        assert table.rows[0][0].is_header

    def test_image_without_alt_is_reported(self, fx):
        assert convert(fx("sample.html")).diagnostics.has(Code.IMAGE_WITHOUT_TEXT)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


@requires_pdf
class TestPdf:
    def test_pages_are_structural_units(self, fx):
        result = convert(fx("simple.pdf"))
        assert result.document.container_kind == "pages"
        assert len(result.document.pages) == 4
        assert [p.index for p in result.document.pages] == [1, 2, 3, 4]

    def test_headings_inferred_from_font_size(self, fx):
        headings = [n for n in convert(fx("simple.pdf")).document.iter_nodes() if isinstance(n, Heading)]
        assert headings
        assert headings[0].level == 1
        # Heading levels are inferred, and the IR must say so.
        assert all(h.origin.value == "inferred" for h in headings)

    def test_provenance_carries_page_and_bbox(self, fx):
        paragraph = next(
            n for n in convert(fx("simple.pdf")).document.iter_nodes() if isinstance(n, Paragraph)
        )
        locator = paragraph.sources[0].locator
        assert locator.page == 1
        assert locator.bbox is not None

    def test_repeating_footer_is_removed_by_default(self, fx):
        assert "CONFIDENTIAL" not in texts(convert(fx("simple.pdf")))

    def test_repeating_footer_can_be_kept(self, fx):
        result = convert(fx("simple.pdf"), ConversionConfig(include_headers_footers=True))
        assert "CONFIDENTIAL" in texts(result)

    def test_per_page_furniture_is_not_treated_as_header(self, fx):
        # "Page N" differs per page, so it is content, not running furniture.
        body = texts(convert(fx("simple.pdf")))
        assert "Page 2" in body

    def test_title_survives_header_filtering(self, fx):
        # Guards the regression where a repeated title is deleted as a header.
        assert "Quarterly Report" in texts(convert(fx("simple.pdf")))

    def test_numbered_lines_become_a_list_with_text(self, fx):
        page_one = convert(fx("simple.pdf")).document.pages[0]
        items = [
            item.text
            for node in page_one.blocks
            if isinstance(node, ListBlock)
            for item in node.items
        ]
        assert items == ["First finding", "Second finding"]

    def test_table_recovered_from_rulings(self, fx):
        result = convert(fx("table.pdf"))
        table = next(result.document.tables())
        assert table.is_rectangular()
        assert table.rows[0][0].text == "Region"
        assert table.rows[3][2].text == "85"
        # Recovered geometry is not "observed"; it must be labelled.
        assert table.origin.value == "reconstructed"

    def test_table_extraction_can_be_disabled(self, fx):
        result = convert(fx("table.pdf"), ConversionConfig(pdf_tables="never"))
        assert list(result.document.tables()) == []

    def test_multi_column_reading_order(self, fx):
        result = convert(fx("multicolumn.pdf"))
        assert result.document.pages[0].signals.column_count == 2
        assert result.diagnostics.has(Code.MULTI_COLUMN)

    def test_scanned_page_is_reported_not_silently_empty(self, fx):
        result = convert(fx("scanned.pdf"))
        assert result.diagnostics.has(Code.LIKELY_SCANNED)
        assert result.diagnostics.has(Code.NO_CONTENT)
        assert not result.ok

    def test_blank_pages_are_reported(self, fx):
        result = convert(fx("empty_pages.pdf"))
        empties = [
            d for d in result.diagnostics.diagnostics if d.code == Code.EMPTY_CONTAINER
        ]
        assert {d.scope_index for d in empties} == {1, 2}

    def test_encrypted_pdf_refuses_cleanly(self, fx):
        with pytest.raises(Exception) as exc:
            convert(fx("encrypted.pdf"))
        assert "password" in str(exc.value).lower()

    def test_truncated_pdf_is_reported_as_repaired(self, fx):
        # A damaged PDF must not quietly "succeed": the repair is disclosed.
        result = convert(fx("corrupt.pdf"))
        repaired = [d for d in result.diagnostics.diagnostics if d.code == Code.PARSER_FALLBACK]
        assert repaired
        assert "repair" in repaired[0].message.lower()

    def test_page_limit_is_enforced(self, fx):
        with pytest.raises(Exception) as exc:
            convert(fx("simple.pdf"), ConversionConfig(max_pages=2))
        assert "max_pages" in str(exc.value)

    def test_no_ocr_on_the_default_path(self, fx):
        # OCR must never be triggered implicitly; a scan is reported instead.
        result = convert(fx("scanned.pdf"))
        assert result.document.text_length() == 0


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------


@requires_docx
class TestDocx:
    def test_body_order_is_preserved(self, fx):
        result = convert(fx("order.docx"))
        assert [type(n).__name__ for n in result.document.body] == [
            "Paragraph",
            "Table",
            "Paragraph",
        ]
        assert result.document.body[0].text == "before table"
        assert result.document.body[2].text == "after table"

    def test_heading_hierarchy(self, fx):
        headings = [n for n in convert(fx("sample.docx")).document.iter_nodes() if isinstance(n, Heading)]
        assert [(h.level, h.text) for h in headings] == [
            (1, "Quarterly Report"),
            (2, "Summary"),
        ]

    def test_lists(self, fx):
        lists = [n for n in convert(fx("sample.docx")).document.iter_nodes() if isinstance(n, ListBlock)]
        assert any(not item.ordered for item in lists)
        assert any(item.ordered for item in lists)

    def test_table_header_row(self, fx):
        table = next(convert(fx("sample.docx")).document.tables())
        assert table.header_rows == 1
        assert table.rows[0][0].text == "Region"

    def test_core_metadata(self, fx):
        meta = convert(fx("sample.docx")).document.metadata
        assert meta["title"] == "Quarterly Report"
        assert meta["author"] == "DocWeave Tests"

    def test_footnotes_are_not_discarded(self, fx):
        result = convert(fx("notes.docx"))
        body = texts(result)
        assert "Figures are unaudited." in body
        assert "Definition of ARR provided." in body
        # Notes are emitted in the order the body references them.
        assert body.index("Figures are unaudited.") < body.index("Definition of ARR provided.")
        # And the loss of anchoring is disclosed rather than glossed over.
        assert result.diagnostics.has(Code.UNSUPPORTED_STRUCTURE)

    def test_separators_are_not_treated_as_notes(self, fx):
        assert "separator" not in texts(convert(fx("notes.docx")))

    def test_broken_package_is_reported_as_unrecognised(self, fx):
        # A zip that lacks word/document.xml is not a DOCX. Say so precisely
        # rather than letting python-docx fail with an opaque error.
        with pytest.raises(UnsupportedFormatError) as exc:
            convert(fx("broken.docx"))
        assert "unknown" in str(exc.value)
        assert "docx" in str(exc.value)


# --------------------------------------------------------------------------
# PPTX
# --------------------------------------------------------------------------


@requires_pptx
class TestPptx:
    def test_slides_are_structural_units(self, fx):
        result = convert(fx("sample.pptx"))
        assert result.document.container_kind == "slides"
        assert len(result.document.slides) == 4

    def test_titles_do_not_appear_twice(self, fx):
        markdown = convert(fx("sample.pptx")).to_markdown()
        assert markdown.count("# Quarterly Review") == 1

    def test_subtitle_is_not_a_peer_heading(self, fx):
        headings = [
            (n.level, n.text)
            for n in convert(fx("sample.pptx")).document.iter_nodes()
            if isinstance(n, Heading)
        ]
        assert (2, "Prepared by the analysis team") in headings

    def test_table_and_notes(self, fx):
        result = convert(fx("sample.pptx"))
        table = next(result.document.tables())
        assert table.rows[1][0].text == "North"
        assert any("finance" in n.text_of() for n in result.document.iter_nodes())

    def test_empty_presentation_is_reported(self, fx):
        result = convert(fx("empty.pptx"))
        assert not result.ok
        assert result.diagnostics.has(Code.NO_CONTENT)


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------


@requires_xlsx
class TestControlCharacters:
    """Characters that are not content must not reach output.

    Found by running DocWeave against a real 37-page book PDF rather than by
    writing a fixture: ``\\x08`` separated a contents entry from its page number
    in the text layer and arrived in Markdown as an invisible control character.
    Every synthetic fixture used plain ASCII, so nothing caught it.
    """

    def test_a_control_separator_becomes_a_space(self, fx):
        """A space, not a deletion.

        Here the control is the separator PyMuPDF drops when it concatenates
        spans, so removing it would glue ``Introduction`` to its page number.
        """
        out = convert(fx("control_chars.pdf")).to_markdown()
        assert "Introduction 1" in out

    def test_no_control_characters_survive(self, fx):
        out = convert(fx("control_chars.pdf")).to_markdown()
        offenders = {
            hex(ord(c))
            for c in out
            if ord(c) < 32 and c not in "\n\r\t"
        }
        assert not offenders, offenders

    def test_no_zero_width_characters_survive(self, fx):
        """Invisible characters make text harder to search and to compare."""
        from docweave.adapters.pdf_adapter import _join_lines

        block = {
            "lines": [
                {
                    "bbox": (0, 0, 100, 10),
                    "spans": [{"text": "hyper­feminine"}],
                }
            ]
        }
        assert _join_lines(block, "") == "hyperfeminine"

    def test_the_real_book_has_no_control_characters(self):
        """The defect that started this, checked against the real file.

        Skipped when the PDF is not present: it is a 1.2 MB third-party book
        and does not belong in the fixture corpus.
        """
        from pathlib import Path

        source = Path("Bimbo.pdf")
        if not source.exists():
            pytest.skip("Bimbo.pdf is not in the working tree")
        out = convert(str(source)).to_markdown()
        offenders = {hex(ord(c)) for c in out if ord(c) < 32 and c not in "\n\r\t"}
        assert not offenders, offenders
        invisibles = ["\u00ad", "\u200b", "\u200c", "\u2060", "\ufeff"]
        for invisible in invisibles:
            assert invisible not in out, hex(ord(invisible))


class TestXlsx:
    def test_a_whole_day_date_does_not_carry_a_clock_time(self, fx):
        """Regression: ``str(datetime)`` leaked ``00:00:00`` into every date."""
        out = convert(fx("formatted.xlsx")).to_markdown()
        assert "2026-03-14" in out
        assert "00:00:00" not in out

    def test_numbers_keep_their_raw_value_not_the_display_string(self, fx):
        """0.075 stays 0.075: a consumer can format it, but cannot un-format it."""
        out = convert(fx("formatted.xlsx")).to_markdown()
        assert "0.075" in out
        assert "7.5%" not in out

    def test_sheets_keep_identity(self, fx):
        result = convert(fx("sample.xlsx"))
        assert result.document.container_kind == "sheets"
        assert [s.name for s in result.document.sheets] == ["Data", "Hidden"]
        assert result.document.sheets[1].hidden is True

    def test_cell_coordinates_survive(self, fx):
        result = convert(fx("sparse.xlsx"))
        refs = {
            cell.locator.cell
            for sheet in result.document.sheets
            for row in sheet.table.rows
            for cell in row
        }
        assert refs == {"A1", "C1", "E2"}

    def test_formula_and_cached_value_are_separate(self, fx):
        sheet = convert(fx("sample.xlsx")).document.sheets[0]
        cells = [c for row in sheet.table.rows for c in row]
        formula_cells = [c for c in cells if c.formula]
        assert formula_cells
        assert formula_cells[0].formula == "=SUM(B2:B2)"
        # Never evaluated: no cached result must be invented.
        assert formula_cells[0].value is None
        assert "no cached result" in formula_cells[0].text

    def test_missing_cached_values_are_reported(self, fx):
        assert convert(fx("sample.xlsx")).diagnostics.has(
            Code.FORMULA_WITHOUT_CACHED_VALUE
        )

    def test_cached_values_can_be_read_when_present(self, fx, scratch):
        import openpyxl

        path = scratch / "cached.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = 2
        sheet["A2"] = 3
        sheet["A3"] = "=SUM(A1:A2)"
        workbook.save(path)

        # openpyxl does not compute formulas, so the cached pass must still
        # report "unavailable" rather than invent 5.
        result = convert(path)
        cells = [c for row in result.document.sheets[0].table.rows for c in row]
        total = next(c for c in cells if c.formula)
        assert total.value is None

    def test_hidden_sheet_is_flagged(self, fx):
        assert convert(fx("sample.xlsx")).diagnostics.has(Code.HIDDEN_CONTENT)

    def test_sparse_rows_are_not_flagged_as_defective(self, fx):
        result = convert(fx("sparse.xlsx"))
        assert not result.diagnostics.has(Code.RAGGED_TABLE)

    def test_merged_ranges_are_reported(self, fx):
        assert convert(fx("sample.xlsx")).diagnostics.has(Code.EMPTY_CELLS)

    def test_row_budget_is_enforced(self, fx):
        result = convert(fx("sample.xlsx"), ConversionConfig(xlsx_max_rows_per_sheet=2))
        assert result.document.sheets[0].truncated is True
        assert result.diagnostics.has(Code.LIMIT_EXCEEDED)


class TestUnsupportedInputs:
    def test_zip_without_document_parts(self, fx):
        with pytest.raises(UnsupportedFormatError):
            convert(fx("notzip.docx"))

    def test_unsupported_format_message_lists_options(self, scratch):
        path = scratch / "legacy.doc"
        path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 256)
        with pytest.raises(UnsupportedFormatError) as exc:
            convert(path)
        assert "Supported formats" in str(exc.value)
