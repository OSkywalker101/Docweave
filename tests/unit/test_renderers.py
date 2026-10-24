"""Renderers: Markdown, JSON, JSONL, and the determinism guarantee."""

from __future__ import annotations

import json

import pytest

from docweave import ConversionConfig, convert
from docweave.models import (
    Cell,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Paragraph,
    SourceRef,
    Table,
)
from docweave.renderers import MarkdownRenderer, iter_records, render_report


def _ref():
    return SourceRef(adapter="test", origin="observed", confidence="high")


def _document(nodes) -> Document:
    return Document(adapter="test", body=nodes)


class TestMarkdownNodes:
    def test_headings(self):
        out = MarkdownRenderer().render(_document([Heading(id="h", level=2, text="Title", sources=[_ref()])]))
        assert out.strip() == "## Title"

    def test_heading_level_is_clamped(self):
        out = MarkdownRenderer().render(_document([Heading(id="h", level=9, text="T", sources=[_ref()])]))
        assert out.strip() == "###### T"

    def test_code_block_with_language(self):
        from docweave.models import CodeBlock

        out = MarkdownRenderer().render(
            _document([CodeBlock(id="c", text="print(1)", language="python", sources=[_ref()])])
        )
        assert "```python" in out
        assert "print(1)" in out

    def test_unordered_and_ordered_lists(self):
        block = ListBlock(
            id="l",
            ordered=True,
            items=[ListItem(id="i0", text="first", ordinal=1), ListItem(id="i1", text="second", ordinal=2)],
            sources=[_ref()],
        )
        assert MarkdownRenderer().render(_document([block])).strip() == "1. first\n2. second"

    def test_quote(self):
        from docweave.models import Quote

        out = MarkdownRenderer().render(_document([Quote(id="q", text="a\nb", sources=[_ref()])]))
        assert out.strip() == "> a\n> b"

    def test_inline_markup_is_escaped(self):
        out = MarkdownRenderer().render(_document([Paragraph(id="p", text="a * b [c]", sources=[_ref()])]))
        assert r"\*" in out and r"\[c\]" in out

    def test_markup_sourced_text_is_not_escaped_twice(self):
        node = Paragraph(id="p", text="**bold** and [l](u)", sources=[_ref()], meta={"verbatim": True})
        out = MarkdownRenderer().render(_document([node]))
        assert out.strip() == "**bold** and [l](u)"
        assert "\\" not in out

    def test_prose_dashes_are_not_mangled(self):
        out = MarkdownRenderer().render(
            _document([Paragraph(id="p", text="a -- b and snake_case_name", sources=[_ref()])])
        )
        assert "--" in out
        assert "snake_case_name" in out


class TestMarkdownTables:
    def _table(self, rows, header_rows=1, **kwargs):
        cells = [
            [
                Cell(id=f"c{r}{c}", text=v, row=r, column=c, is_header=r < header_rows)
                for c, v in enumerate(row)
            ]
            for r, row in enumerate(rows)
        ]
        return Table(id="t", rows=cells, header_rows=header_rows, sources=[_ref()], **kwargs)

    def test_simple_table(self):
        table = self._table([["a", "b"], ["1", "2"]])
        out = MarkdownRenderer().render(_document([table]))
        assert "| a | b |" in out
        assert "| 1 | 2 |" in out

    def test_table_without_header_gets_an_empty_header_row(self):
        table = self._table([["a", "b"], ["1", "2"]], header_rows=0)
        out = MarkdownRenderer().render(_document([table]))
        assert out.splitlines()[0].strip() == "|  |  |"
        # Data must not be promoted into the header position.
        assert "| a | b |" in out.splitlines()[2]

    def test_pipes_in_cells_are_escaped(self):
        table = self._table([["a|b", "c"]])
        assert r"a\|b" in MarkdownRenderer().render(_document([table]))

    def test_ragged_table_falls_back_to_html(self):
        table = self._table([["a", "b", "c"], ["1", "2"]], header_rows=0)
        out = MarkdownRenderer().render(_document([table]))
        assert out.strip().startswith("<table>")

    def test_sparse_table_uses_cell_references(self):
        cells = [
            [Cell(id="c0", text="v", row=0, column=3, locator={"kind": "sheet", "cell": "D1"})]
        ]
        table = Table(id="t", rows=cells, header_rows=0, sparse=True, sources=[_ref()])
        out = MarkdownRenderer().render(_document([table]))
        assert "D1: v" in out
        # Padding a sparse row into a grid would misplace the value.
        assert "|" not in out

    def test_html_rendering_keeps_spans(self):
        cells = [[Cell(id="c0", text="span", row=0, column=0, column_span=2, row_span=3)]]
        table = Table(id="t", rows=cells, header_rows=0, sources=[_ref()])
        out = MarkdownRenderer(ConversionConfig(table_rendering="html")).render(_document([table]))
        assert 'colspan="2"' in out and 'rowspan="3"' in out

    def test_both_rendering_emits_html_then_markdown(self):
        table = self._table([["a"], ["1"]])
        out = MarkdownRenderer(ConversionConfig(table_rendering="both")).render(_document([table]))
        assert "<table>" in out and "| a |" in out

    def test_empty_table(self):
        assert "empty table" in MarkdownRenderer().render(_document([Table(id="t", sources=[_ref()])]))

    def _sparse(self, values):
        """A sparse one-row table, the shape a spreadsheet row renders as."""
        cells = [
            Cell(id=f"c{i}", text=v, row=0, column=i, locator={"kind": "sheet", "cell": f"{chr(65 + i)}1"})
            for i, v in enumerate(values)
        ]
        return Table(id="t", rows=[cells], header_rows=0, sparse=True, sources=[_ref()])

    def test_sparse_row_separator_inside_a_value_is_quoted(self):
        """One cell whose text looks like two must not render as two cells.

        The list format joins cells with ``"; "``, so the value
        ``harbour; B1: docks`` would otherwise be indistinguishable from a row
        of A1 and B1 -- silently changing the cell count for any consumer.
        """
        out = MarkdownRenderer().render(
            _document([self._sparse(["harbour; B1: docks", "next"])])
        )
        assert '- A1: "harbour; B1: docks"; B1: next' in out

    def test_quoting_is_not_applied_unnecessarily(self):
        """A semicolon that cannot be mistaken for a separator stays readable."""
        out = MarkdownRenderer().render(_document([self._sparse(["alpha; beta", "next"])]))
        assert "- A1: alpha; beta; B1: next" in out

    def test_embedded_quotes_are_doubled(self):
        out = MarkdownRenderer().render(_document([self._sparse(['say "hi"; B1: x', "n"])]))
        assert '- A1: "say ""hi""; B1: x"; B1: n' in out


class TestMarkdownImages:
    def test_placeholder_marks_unextracted_assets(self):
        node = ImageBlock(id="i", asset_id="missing", sources=[_ref()])
        out = MarkdownRenderer().render(_document([node]))
        assert "image://missing" in out
        assert "not extracted" in out

    def test_omit_mode_drops_images(self):
        node = ImageBlock(id="i", asset_id="a", sources=[_ref()])
        assert MarkdownRenderer(ConversionConfig(image_rendering="omit")).render(_document([node])).strip() == ""

    def test_alt_text_is_used_as_the_label(self):
        node = ImageBlock(id="i", alt="Company logo", sources=[_ref()])
        assert "Company logo" in MarkdownRenderer().render(_document([node]))


class TestMarkdownContainers:
    def test_pages_get_anchors(self, fx):
        result = convert(fx("simple.pdf"))
        assert "<!-- page 1 -->" in result.to_markdown()
        assert "<!-- page 4 -->" in result.to_markdown()

    def test_slides_get_anchors(self, fx):
        assert "<!-- slide 1 -->" in convert(fx("sample.pptx")).to_markdown()

    def test_hidden_sheets_are_labelled(self, fx):
        assert "(hidden)" in convert(fx("sample.xlsx")).to_markdown()

    def test_diagnostics_trailer_can_be_disabled(self, fx):
        result = convert(fx("sample.html"))
        with_block = MarkdownRenderer(ConversionConfig(markdown_include_diagnostics=True)).render(
            result.document, result.diagnostics
        )
        without = MarkdownRenderer(ConversionConfig(markdown_include_diagnostics=False)).render(
            result.document, result.diagnostics
        )
        assert "docweave:diagnostics" in with_block
        assert "docweave:diagnostics" not in without

    def test_front_matter_is_opt_in(self, fx):
        result = convert(fx("simple.pdf"))
        assert "| field | value |" not in result.to_markdown()
        with_meta = MarkdownRenderer(ConversionConfig(markdown_include_front_matter=True)).render(result.document)
        assert "| field | value |" in with_meta


class TestJsonEnvelope:
    def test_envelope_shape(self, fx):
        result = convert(fx("sample.docx"))
        payload = json.loads(result.to_json())
        assert payload["schema"] == "docweave.document/1.0"
        assert payload["schema_version"] == "1.0"
        assert payload["tool"]["name"] == "docweave"
        assert payload["source"]["detected_format"] == "docx"
        assert "body" in payload
        assert "diagnostics" in payload
        assert payload["diagnostic_summary"]["total"] >= 0

    def test_source_has_content_hash(self, fx):
        payload = json.loads(convert(fx("sample.md")).to_json())
        assert len(payload["source"]["sha256"]) == 64
        assert payload["source"]["byte_size"] > 0

    def test_timing_is_opt_in(self, fx):
        result = convert(fx("sample.md"))
        assert "timing" not in json.loads(result.to_json())
        assert "timing" in json.loads(result.to_json(timing=True))

    def test_envelope_is_valid_json_for_every_format(self, fx):
        for name in (
            "simple.pdf",
            "sample.docx",
            "sample.pptx",
            "sample.xlsx",
            "sample.html",
            "sample.csv",
            "sample.json",
            "sample.xml",
            "sample.md",
        ):
            json.loads(convert(fx(name)).to_json())

    def test_table_payload_keeps_coordinates(self, fx):
        payload = json.loads(convert(fx("sparse.xlsx")).to_json())
        table = payload["sheets"][0]["table"]
        references = [c["reference"] for row in table["rows"] for c in row]
        assert references == ["A1", "C1", "E2"]
        assert table["sparse"] is True


class TestJsonl:
    def test_one_record_per_node(self, fx):
        lines = [json.loads(line) for line in convert(fx("sample.docx")).to_jsonl().splitlines()]
        assert lines
        assert all(line["schema"] == "docweave.record/1.0" for line in lines)

    def test_records_carry_container_and_provenance_grade(self, fx):
        records = iter_records(convert(fx("simple.pdf")).document)
        assert {r["container"] for r in records} == {"page"}
        assert all(r["origin"] in {"observed", "reconstructed", "inferred", "absent"} for r in records)
        assert all(r["confidence"] in {"high", "medium", "low", "none"} for r in records)

    def test_page_and_bbox_are_present(self, fx):
        records = iter_records(convert(fx("simple.pdf")).document)
        assert any(r.get("page") == 1 for r in records)
        assert any(r.get("bbox") for r in records)

    def test_sheet_records_name_the_sheet(self, fx):
        records = iter_records(convert(fx("sample.xlsx")).document)
        assert {r["sheet"] for r in records} == {"Data", "Hidden"}

    def test_slide_records_include_the_title(self, fx):
        records = iter_records(convert(fx("sample.pptx")).document)
        assert any(r.get("slide_title") for r in records)

    def test_table_record_has_cell_payload(self, fx):
        records = iter_records(convert(fx("sample.csv")).document)
        tables = [r for r in records if r["type"] == "table"]
        assert tables and "rows" in tables[0]["table"]


class TestReport:
    def test_report_counts(self, fx):
        payload = json.loads(render_report(*_report_parts(fx("sample.docx"))))
        assert payload["schema"] == "docweave.report/1.0"
        assert payload["counts"]["nodes"] > 0
        assert payload["counts"]["tables"] >= 1

    def test_report_of_a_failed_document_still_renders(self, fx):
        payload = json.loads(render_report(*_report_parts(fx("scanned.pdf"))))
        assert payload["counts"]["pages"] == 1
        assert payload["diagnostics"]


def _report_parts(fx):
    result = convert(fx)
    return result.document, result.diagnostics


class TestDeterminism:
    @pytest.mark.parametrize(
        "name",
        ["simple.pdf", "sample.docx", "sample.pptx", "sample.xlsx", "sample.html", "sample.md", "sample.csv"],
    )
    def test_markdown_is_byte_identical(self, fx, name):
        first = convert(fx(name)).to_markdown()
        second = convert(fx(name)).to_markdown()
        assert first == second

    @pytest.mark.parametrize(
        "name", ["simple.pdf", "sample.xlsx", "sample.pptx", "sample.docx"]
    )
    def test_json_is_byte_identical(self, fx, name):
        assert convert(fx(name)).to_json() == convert(fx(name)).to_json()

    def test_jsonl_is_byte_identical(self, fx):
        assert convert(fx("simple.pdf")).to_jsonl() == convert(fx("simple.pdf")).to_jsonl()

    def test_ids_are_stable_across_runs(self, fx):
        a = [n.id for n in convert(fx("sample.docx")).document.iter_nodes()]
        b = [n.id for n in convert(fx("sample.docx")).document.iter_nodes()]
        assert a == b


class TestDiagnosticRendering:
    def test_escape_of_dangerous_table_content(self, fx):
        result = convert(fx("sample.html"))
        rendered = result.to_markdown()
        assert "<script>" not in rendered

    def test_untrusted_text_cannot_inject_a_pipe(self, scratch):
        path = scratch / "x.csv"
        path.write_text("a,b\nc|d,e\n")
        out = convert(path).to_markdown()
        assert r"c\|d" in out
