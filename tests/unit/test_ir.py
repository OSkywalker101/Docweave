"""The canonical IR: provenance, confidence, origin and stable ids."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from docweave import ConversionConfig, convert
from docweave.ids import IdFactory
from docweave.models import (
    Cell,
    Confidence,
    Diagnostic,
    DiagnosticReport,
    Document,
    Origin,
    Paragraph,
    SourceRef,
    Table,
)
from docweave.models.diagnostics import Category, Severity


class TestIds:
    def test_ids_are_hierarchical_and_deterministic(self):
        ids = IdFactory()
        first = ids.child(None, "p")
        second = ids.child(None, "p")
        nested = ids.child(first, "c")
        assert (first, second, nested) == ("d.p0", "d.p1", "d.p0.c0")

    def test_reset_repeats_the_sequence(self):
        ids = IdFactory()
        before = [ids.child(None, "x") for _ in range(3)]
        ids.reset()
        after = [ids.child(None, "x") for _ in range(3)]
        assert before == after

    def test_node_ids_are_unique_within_a_document(self, fx):
        result = convert(fx("simple.pdf"))
        ids = [n.id for n in result.document.iter_nodes()]
        assert len(ids) == len(set(ids))


class TestProvenance:
    def test_every_node_carries_a_source(self, fx):
        for name in ("simple.pdf", "sample.docx", "sample.pptx", "sample.xlsx", "sample.html"):
            result = convert(fx(name))
            for node in result.document.iter_nodes():
                assert node.sources, f"{name}: {node.id} has no provenance"

    def test_origin_is_the_weakest_source(self):
        node = Paragraph(
            id="x",
            sources=[
                SourceRef(adapter="a", origin=Origin.OBSERVED, confidence=Confidence.HIGH),
                SourceRef(adapter="b", origin=Origin.INFERRED, confidence=Confidence.MEDIUM),
            ],
        )
        assert node.origin is Origin.INFERRED
        assert node.confidence is Confidence.MEDIUM

    def test_absent_content_is_representable(self):
        node = Paragraph(
            id="x",
            sources=[SourceRef(adapter="a", origin=Origin.ABSENT, confidence=Confidence.NONE)],
            text="",
        )
        assert node.origin is Origin.ABSENT
        assert node.confidence is Confidence.NONE

    def test_locators_are_format_specific(self, fx):
        pdf = convert(fx("simple.pdf"))
        assert any(s.locator.page for n in pdf.document.iter_nodes() for s in n.sources)

        pptx = convert(fx("sample.pptx"))
        assert any(s.locator.slide for n in pptx.document.iter_nodes() for s in n.sources)

        html = convert(fx("sample.html"))
        assert any(s.locator.css_path for n in html.document.iter_nodes() for s in n.sources)

    def test_cells_keep_a1_references(self, fx):
        xlsx = convert(fx("sparse.xlsx"))
        refs = {
            cell.locator.cell
            for sheet in xlsx.document.sheets
            for row in sheet.table.rows
            for cell in row
        }
        assert refs == {"A1", "C1", "E2"}


class TestDocumentModel:
    def test_text_length_counts_table_cells(self):
        table = Table(
            id="t",
            rows=[[Cell(id="c0", text="alpha", row=0, column=0), Cell(id="c1", text="beta", row=0, column=1)]],
        )
        document = Document(adapter="test", body=[table])
        assert document.text_length() == len("alpha") + len("beta")

    def test_a_table_only_document_is_not_empty(self, fx):
        result = convert(fx("sample.csv"))
        assert result.document.text_length() > 0
        assert result.ok

    def test_traversal_is_stable(self, fx):
        document = convert(fx("simple.pdf")).document
        assert [n.id for n in document.iter_nodes()] == [
            n.id for n in document.iter_nodes()
        ]

    def test_sheet_tables_are_traversed(self, fx):
        document = convert(fx("sample.xlsx")).document
        assert any(isinstance(n, Table) for n in document.iter_nodes())

    def test_digest_is_content_based(self, fx):
        a = convert(fx("simple.pdf")).document
        b = convert(fx("simple.pdf")).document
        assert a.digest() == b.digest()

    def test_unknown_fields_are_rejected(self):
        # Narrow on purpose: a blind ``Exception`` would also pass if the model
        # raised something entirely unrelated, which is the failure mode a
        # strict-schema test exists to catch.
        with pytest.raises(ValidationError):
            Paragraph(id="x", nonsense=1)


class TestSerialisation:
    def test_document_round_trips_through_json(self, fx):
        document = convert(fx("sample.xlsx")).document
        payload = document.model_dump(mode="json")
        restored = Document.model_validate(payload)
        assert restored.text_length() == document.text_length()
        assert restored.schema_version == document.schema_version

    def test_schema_version_is_declared(self, fx):
        assert convert(fx("sample.md")).document.schema_version == "1.0"


class TestDiagnosticReport:
    def test_sorting_is_deterministic_and_error_first(self):
        report = DiagnosticReport()
        report.add(
            Diagnostic(code="b", category=Category.HEURISTIC, severity=Severity.INFO, message="i")
        )
        report.add(
            Diagnostic(code="a", category=Category.STRUCTURAL, severity=Severity.ERROR, message="e")
        )
        report.add(
            Diagnostic(code="c", category=Category.HEURISTIC, severity=Severity.WARNING, message="w")
        )
        assert [d.code for d in report.sorted()] == ["a", "c", "b"]

    def test_counts_by_category(self):
        report = DiagnosticReport()
        report.add(
            Diagnostic(code="x", category=Category.STRUCTURAL, severity=Severity.ERROR, message="m")
        )
        report.add(
            Diagnostic(code="y", category=Category.HEURISTIC, severity=Severity.WARNING, message="m")
        )
        counts = report.counts()
        assert counts[Category.STRUCTURAL.value] == 1
        assert counts[Category.HEURISTIC.value] == 1
        assert counts[Category.UNVERIFIED.value] == 0

    def test_worst_severity(self):
        report = DiagnosticReport()
        assert report.worst_severity() is None
        report.add(
            Diagnostic(code="x", category=Category.HEURISTIC, severity=Severity.INFO, message="m")
        )
        assert report.worst_severity() is Severity.INFO
        report.add(
            Diagnostic(code="y", category=Category.STRUCTURAL, severity=Severity.ERROR, message="m")
        )
        assert report.worst_severity() is Severity.ERROR

    def test_to_line_includes_location_and_code(self):
        diagnostic = Diagnostic(
            code="c",
            category=Category.HEURISTIC,
            message="m",
            scope_kind="page",
            scope_index=7,
        )
        assert "page 7" in diagnostic.to_line()
        assert "c" in diagnostic.to_line()


class TestConfig:
    def test_unknown_key_is_rejected_with_a_helpful_message(self):
        from docweave.exceptions import InputError

        with pytest.raises(InputError) as exc:
            ConversionConfig.from_mapping({"nope": 1})
        assert "nope" in str(exc.value)

    def test_json_file_round_trip(self, scratch):
        path = scratch / "cfg.json"
        path.write_text(json.dumps({"pdf_tables": "never", "max_pages": 10}), encoding="utf-8")
        config = ConversionConfig.from_json_file(path)
        assert config.pdf_tables == "never"
        assert config.max_pages == 10

    def test_missing_config_file(self, scratch):
        from docweave.exceptions import InputError

        with pytest.raises(InputError):
            ConversionConfig.from_json_file(scratch / "absent.json")

    def test_describe_is_stable(self):
        a = ConversionConfig().describe()
        b = ConversionConfig().describe()
        assert a == b
        assert "pdf_tables" in a

    def test_merged_ignores_none(self):
        config = ConversionConfig().merged(pdf_tables=None, max_pages=7)
        assert config.pdf_tables == "auto"
        assert config.max_pages == 7
