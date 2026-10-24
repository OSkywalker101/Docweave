"""Validation engine: honest classification of extraction problems."""

from __future__ import annotations

import pytest

from docweave import ConversionConfig, convert
from docweave.models import (
    Cell,
    Document,
    Paragraph,
    SourceRef,
    Table,
)
from docweave.models.diagnostics import Category, Code, Severity
from docweave.validation import validate


def _ref():
    return SourceRef(adapter="test", origin="observed", confidence="high")


def _document(nodes) -> Document:
    return Document(adapter="test", body=nodes)


class TestClassification:
    def test_no_content_is_an_error(self):
        report = validate(_document([]), ConversionConfig())
        assert report.has(Code.NO_CONTENT)
        assert report.errors()
        assert report.errors()[0].category is Category.STRUCTURAL

    def test_content_present_means_no_content_error(self):
        report = validate(_document([Paragraph(id="p", text="hello", sources=[_ref()])]), ConversionConfig())
        assert not report.has(Code.NO_CONTENT)

    def test_table_only_document_is_not_reported_empty(self):
        table = Table(id="t", rows=[[Cell(id="c", text="x", row=0, column=0)]], sources=[_ref()])
        report = validate(_document([table]), ConversionConfig())
        assert not report.has(Code.NO_CONTENT)

    def test_ragged_table_is_structural(self):
        table = Table(
            id="t",
            rows=[[Cell(id="c0", text="a", row=0, column=0)], [Cell(id="c1", text="b", row=1, column=0)]],
            sources=[_ref()],
        )
        table.rows[1].append(Cell(id="c2", text="c", row=1, column=1))
        report = validate(_document([table]), ConversionConfig())
        diagnostic = next(d for d in report.diagnostics if d.code == Code.RAGGED_TABLE)
        assert diagnostic.category is Category.STRUCTURAL
        assert diagnostic.severity is Severity.WARNING

    def test_sparse_tables_are_not_ragged(self):
        table = Table(
            id="t",
            sparse=True,
            rows=[[Cell(id="c0", text="a", row=0, column=0)], [Cell(id="c1", text="b", row=1, column=2)]],
            sources=[_ref()],
        )
        assert not validate(_document([table]), ConversionConfig()).has(Code.RAGGED_TABLE)

    def test_nodes_without_provenance_are_reported(self):
        report = validate(_document([Paragraph(id="p", text="orphan")]), ConversionConfig())
        assert report.has(Code.UNSUPPORTED_STRUCTURE)


class TestThresholds:
    def test_replacement_char_threshold_is_configurable(self):
        from docweave.models import Page, PageSignals

        page = Page(
            index=1,
            signals=PageSignals(char_count=1000, replacement_chars=5),
            blocks=[Paragraph(id="p", text="x" * 1000, sources=[_ref()])],
        )
        document = Document(adapter="test", container_kind="pages", pages=[page])
        # 0.5% of characters is above the 0.1% default.
        assert validate(document, ConversionConfig()).has(Code.REPLACEMENT_CHARS)
        relaxed = validate(document, ConversionConfig(replacement_char_ratio=0.05))
        assert not relaxed.has(Code.REPLACEMENT_CHARS)

    def test_cell_budget_is_enforced(self, scratch):
        import openpyxl

        from docweave.exceptions import LimitExceededError

        path = scratch / "wide.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        for r in range(1, 60):
            for c in range(1, 20):
                sheet.cell(row=r, column=c, value=f"{r}-{c}")
        workbook.save(path)

        with pytest.raises(LimitExceededError) as exc:
            from docweave import convert

            convert(path, ConversionConfig(max_cells=50))
        assert "cell budget" in str(exc.value)

    def test_emit_stable_ids_adds_anchors(self, fx):
        from docweave import convert

        result = convert(fx("sample.md"), ConversionConfig(emit_stable_ids=True))
        markdown = result.to_markdown()
        first_id = next(iter(result.document.iter_nodes())).id
        assert f"<!-- {first_id} -->" in markdown

    def test_empty_table_is_reported(self):
        report = validate(_document([Table(id="t", sources=[_ref()])]), ConversionConfig())
        assert report.has(Code.EMPTY_CONTAINER)

    def test_low_density_threshold_is_configurable(self):
        from docweave.models import Page, PageSignals

        page = Page(
            index=1,
            signals=PageSignals(char_count=5),
            blocks=[Paragraph(id="p", text="tiny", sources=[_ref()])],
        )
        document = Document(adapter="test", container_kind="pages", pages=[page])
        strict = validate(document, ConversionConfig(min_chars_per_page=100))
        lenient = validate(document, ConversionConfig(min_chars_per_page=1))
        assert strict.has(Code.LOW_TEXT_DENSITY)
        assert not lenient.has(Code.LOW_TEXT_DENSITY)

    def test_scanned_page_threshold_is_configurable(self):
        from docweave.models import ImageBlock, Page, PageSignals

        page = Page(
            index=1,
            signals=PageSignals(char_count=0, image_count=1, media_area_ratio=0.9),
            images=[ImageBlock(id="i", sources=[_ref()])],
        )
        document = Document(adapter="test", container_kind="pages", pages=[page])
        assert validate(document, ConversionConfig()).has(Code.LIKELY_SCANNED)
        # Raising the bar means the page is only flagged as image-without-text.
        relaxed = validate(document, ConversionConfig(detect_scanned_pages=False))
        assert not relaxed.has(Code.LIKELY_SCANNED)

    def test_scanned_page_outranks_the_document_level_image_warning(self, fx):
        report = convert(fx("scanned.pdf")).diagnostics
        codes = report.codes()
        assert codes.count(Code.IMAGE_WITHOUT_TEXT) == 0


class TestReporting:
    def test_every_diagnostic_explains_itself(self, fx):
        result = convert(fx("simple.pdf"))
        for diagnostic in result.diagnostics.diagnostics:
            assert diagnostic.message
            assert diagnostic.category in Category

    def test_intervention_diagnostics_carry_a_hint(self, fx):
        result = convert(fx("scanned.pdf"))
        scanned = next(d for d in result.diagnostics.diagnostics if d.code == Code.LIKELY_SCANNED)
        assert scanned.category is Category.UNVERIFIED
        assert "ocr" in (scanned.hint or "").lower()

    def test_diagnostics_are_deduplicated(self, fx):
        report = convert(fx("simple.pdf")).diagnostics
        keys = [(d.code, d.scope_index, d.message) for d in report.diagnostics]
        assert len(keys) == len(set(keys))

    def test_summary_is_human_readable(self, fx):
        summary = convert(fx("sample.html")).diagnostics.summary()
        assert "diagnostic(s)" in summary

    def test_evidence_is_machine_readable(self, fx):
        scanned = next(
            d
            for d in convert(fx("scanned.pdf")).diagnostics.diagnostics
            if d.code == Code.LIKELY_SCANNED
        )
        assert isinstance(scanned.evidence, dict)
        assert "char_count" in scanned.evidence

    def test_report_serialises(self, fx):
        import json

        report = convert(fx("sample.docx")).diagnostics
        json.dumps([d.model_dump(mode="json") for d in report.sorted()])


class TestNoFalseClean:
    def test_a_clean_document_reports_nothing_structural(self, fx):
        report = convert(fx("sample.docx")).diagnostics
        assert not report.by_category(Category.STRUCTURAL)

    def test_unverified_is_distinct_from_structural(self, fx):
        report = convert(fx("sample.xlsx")).diagnostics
        unverified = report.by_category(Category.UNVERIFIED)
        assert unverified
        assert all(d.category is Category.UNVERIFIED for d in unverified)

    def test_exit_signal_reflects_errors_only(self, fx):
        # Warnings and unverified content must not fail a conversion outright.
        assert convert(fx("sample.xlsx")).ok
        assert not convert(fx("scanned.pdf")).ok
