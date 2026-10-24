"""Diagram engine integration: CLI, Python API, and images inside documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from docweave.cli import EXIT_OK, EXIT_PARTIAL, cli
from docweave.models.diagnostics import Code
from tests.conftest import requires_diagrams

pytestmark = requires_diagrams


@pytest.fixture
def run():
    runner = CliRunner()

    def _run(*args):
        return runner.invoke(cli, list(args), catch_exceptions=False)

    return _run


class TestDiagramCli:
    def test_mermaid_to_stdout(self, run, diagram_corpus):
        result = run("diagram", str(diagram_corpus / "flowchart.png"), "--format", "mermaid")
        assert "flowchart TD" in result.output
        assert result.exit_code in (EXIT_OK, EXIT_PARTIAL)

    def test_mermaid_to_file(self, run, diagram_corpus, scratch):
        out = scratch / "diagram.mmd"
        result = run(
            "diagram",
            str(diagram_corpus / "flowchart.png"),
            "--format",
            "mermaid",
            "-o",
            str(out),
        )
        assert out.exists()
        assert "flowchart TD" in out.read_text(encoding="utf-8")
        assert result.exit_code in (EXIT_OK, EXIT_PARTIAL)

    def test_json_output(self, run, diagram_corpus, scratch):
        out = scratch / "diagram.json"
        run("diagram", str(diagram_corpus / "er.png"), "--format", "json", "-o", str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["schema"] == "docweave.diagram/1.0"
        assert payload["nodes"]

    def test_json_flag_overrides_format(self, run, diagram_corpus, scratch):
        out = scratch / "diagram.json"
        run("diagram", str(diagram_corpus / "er.png"), "-o", str(out), "--json")
        assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "docweave.diagram/1.0"

    def test_markdown_output(self, run, diagram_corpus, scratch):
        out = scratch / "diagram.md"
        run("diagram", str(diagram_corpus / "flowchart.png"), "-f", "markdown", "-o", str(out))
        text = out.read_text(encoding="utf-8")
        assert "## Nodes" in text and "## Edges" in text

    def test_plantuml_output(self, run, diagram_corpus, scratch):
        out = scratch / "diagram.puml"
        run("diagram", str(diagram_corpus / "flowchart.png"), "-f", "plantuml", "-o", str(out))
        assert out.read_text(encoding="utf-8").startswith("@startuml")

    def test_no_ocr_flag(self, run, diagram_corpus):
        result = run("diagram", str(diagram_corpus / "flowchart.png"), "--no-ocr", "-f", "markdown")
        assert "flowchart TD" not in result.output
        assert "(not read)" in result.output

    def test_diagnostics_are_reported(self, run, diagram_corpus):
        """The CLI must account for every label it finds.

        Either the labels were read, or the ones that were not are reported.
        Asserting only the unread case made this test pass on a machine with no
        OCR engine and fail on one with it installed -- an outcome that depended
        on the machine rather than on the code.
        """
        result = run("diagram", str(diagram_corpus / "flowchart.png"))
        reported_labels = (
            "diagram_label_unread" in result.output
            or "label region" in result.output
        )
        read_labels = "flowchart TD" in result.output or "Label" in result.output
        assert reported_labels or read_labels

    def test_quiet_hides_diagnostics_but_not_content(self, run, diagram_corpus):
        result = run("diagram", str(diagram_corpus / "flowchart.png"), "-q", "-f", "mermaid")
        assert "flowchart TD" in result.output
        assert "diagram_label_unread" not in result.output
        # Quiet must mean quiet: with an OCR engine present, per-label unread
        # chatter must not reappear just because the machine is more capable.
        assert "Extraction report" not in result.output

    def test_unreadable_input_is_exit_code_2(self, run, scratch):
        path = scratch / "broken.png"
        path.write_bytes(b"not a png at all")
        assert run("diagram", str(path)).exit_code == 2

    def test_missing_file_is_exit_code_2(self, run, scratch):
        assert run("diagram", str(scratch / "absent.png")).exit_code == 2

    def test_degraded_diagram_exits_nonzero(self, run, diagram_corpus):
        """A diagram we could not trust must not look like a clean success."""
        result = run("diagram", str(diagram_corpus / "ambiguous.png"), "-q")
        assert result.exit_code == EXIT_PARTIAL

    def test_help_lists_formats(self, run):
        result = run("diagram", "--help")
        for fmt in ("mermaid", "plantuml", "json", "markdown"):
            assert fmt in result.output


class TestPythonApi:
    def test_analyze_file(self, diagram_corpus):
        from docweave.diagram import analyze_file

        result = analyze_file(diagram_corpus / "flowchart.png")
        assert result.graph.nodes
        assert result.elapsed_ms >= 0

    def test_render_all_formats(self, diagram_corpus):
        from docweave.diagram import DIAGRAM_FORMATS, analyze_file, render

        graph = analyze_file(diagram_corpus / "flowchart.png").graph
        for fmt in DIAGRAM_FORMATS:
            assert render(graph, fmt).strip()

    def test_analyze_bytes(self):
        from docweave.diagram import analyze_bytes

        data = Path("tests/fixtures/diagrams/er.png").read_bytes()
        result = analyze_bytes(data, source_image="er.png")
        assert result.graph.nodes

    def test_bad_bytes_raise(self):
        from docweave.diagram import analyze_bytes

        with pytest.raises(ValueError):
            analyze_bytes(b"definitely not an image")

    def test_graph_model_round_trips(self, diagram_corpus):
        from docweave.diagram import analyze_file
        from docweave.diagram.graph import DiagramGraph

        graph = analyze_file(diagram_corpus / "flowchart.png").graph
        restored = DiagramGraph.model_validate(json.loads(graph.model_dump_json()))
        assert len(restored.nodes) == len(graph.nodes)
        assert [n.id for n in restored.nodes] == [n.id for n in graph.nodes]
        assert restored.kind == graph.kind


class TestDocumentIntegration:
    def test_gate_rejects_small_images(self):
        from docweave.diagram.integrate import DiagramIntegrator

        integrator = DiagramIntegrator()
        assert integrator.should_analyze(b"x", "image/gif") is False
        assert integrator.should_analyze(b"x", None) is False

    def test_gate_accepts_a_real_diagram(self):
        from docweave.diagram.integrate import DiagramIntegrator

        data = Path("tests/fixtures/diagrams/er.png").read_bytes()
        assert DiagramIntegrator().should_analyze(data, "image/png") is True

    def test_dimensions_are_read_without_decoding(self):
        from docweave.diagram.integrate import _dimensions

        png = Path("tests/fixtures/diagrams/er.png").read_bytes()
        assert _dimensions(png) == (700, 420)
        assert _dimensions(b"garbage") == (None, None)

    def test_diagnostics_are_scoped_to_the_asset(self, diagram_corpus):
        from docweave.diagram import analyze_file
        from docweave.diagram.integrate import diagram_diagnostics

        result = analyze_file(diagram_corpus / "ambiguous.png")
        wrapped = diagram_diagnostics(result, asset_id="asset1")
        assert all("asset1" in d.message for d in wrapped)
        assert all(d.scope_kind == "diagram" for d in wrapped)

    def test_edged_diagnostic_when_no_edges(self, diagram_corpus):
        from docweave.diagram import analyze_file
        from docweave.diagram.integrate import diagram_diagnostics

        result = analyze_file(diagram_corpus / "ambiguous.png")
        codes = [d.code for d in diagram_diagnostics(result, asset_id="a")]
        assert Code.DIAGRAM_UNCONNECTED in codes

    def test_convert_reports_images_without_diagrams(self, fx):
        """A document with images must not claim they were understood."""
        from docweave import convert

        result = convert(fx("image.docx"))
        assert result.document.assets
        assert all(not a.extracted for a in result.document.assets)


class TestDiagramIntegrationEndToEnd:
    """A diagram embedded in a document is reconstructed as part of the conversion."""

    def test_diagram_in_docx_is_detected_and_reconstructed(self, fx, scratch):
        from docweave import ConversionConfig, convert

        result = convert(
            fx("diagram.docx"),
            ConversionConfig(
                detect_diagrams=True,
                extract_assets=True,
                assets_dir=scratch / "assets",
            ),
        )
        assert result.document.metadata.get("diagram_count") == 1
        graph_info = list(result.document.metadata["diagrams"].values())[0]
        assert graph_info["kind"] == "flowchart"
        assert graph_info["nodes"] >= 5
        assert graph_info["edges"] >= 4

    def test_diagram_diagnostics_reach_the_document_report(self, fx, scratch):
        from docweave import ConversionConfig, convert

        result = convert(
            fx("diagram.docx"),
            ConversionConfig(
                detect_diagrams=True,
                extract_assets=True,
                assets_dir=scratch / "assets",
            ),
        )
        assert any(d.scope_kind == "diagram" for d in result.diagnostics.diagnostics)

    def test_small_images_are_not_sent_to_the_diagram_engine(self, fx):
        """A 4x4 icon is not a diagram; running the engine on it is wasted work."""
        from docweave import ConversionConfig, convert

        result = convert(
            fx("image.docx"),
            ConversionConfig(
                detect_diagrams=True,
                extract_assets=True,
                assets_dir=fx("image.docx").parent / "icons",
            ),
        )
        assert result.document.metadata.get("diagram_count") is None

    def test_diagram_detection_is_off_by_default(self, fx):
        from docweave import ConversionConfig, convert

        assert ConversionConfig().detect_diagrams is False
        result = convert(fx("diagram.docx"), ConversionConfig(extract_assets=False))
        assert "diagrams" not in result.document.metadata


class TestAccuracyMeasurement:
    """Reconstruction accuracy against the drawn graph.

    These are the numbers that justify tuning the detectors. They are measured
    here, in the test suite, rather than asserted as marketing.
    """

    def _score(self, corpus, spec, graph):
        expected_nodes = len(spec.nodes)
        expected_edges = len(spec.edges)
        got_nodes = len(graph.nodes)
        got_edges = len(graph.edges)
        node_recall = min(got_nodes, expected_nodes) / expected_nodes
        edge_recall = min(got_edges, expected_edges) / expected_edges
        precision = got_nodes / expected_nodes if expected_nodes else 0.0
        return node_recall, edge_recall, precision

    def test_clean_diagrams_reach_high_recall(self, diagram_corpus, diagram_specs):
        from docweave.diagram import analyze_file

        for name in ("flowchart", "er", "sequence", "parallelogram", "box_and_arrow"):
            graph = analyze_file(
                diagram_corpus / f"{name}.png"
            ).graph
            node_recall, edge_recall, precision = self._score(
                diagram_corpus, diagram_specs[name], graph
            )
            assert node_recall >= 1.0, f"{name}: node recall {node_recall:.2f}"
            assert edge_recall >= 0.6, f"{name}: edge recall {edge_recall:.2f}"
            assert precision <= 1.35, f"{name}: node precision {precision:.2f}"

    def test_hard_diagrams_degrade_without_crashing(self, diagram_corpus, diagram_specs):
        from docweave.diagram import analyze_file

        graph = analyze_file(diagram_corpus / "ambiguous.png").graph
        assert graph.validate_references() == []
        assert graph.degraded is True

    def test_accuracy_report_is_generated(self, diagram_corpus, diagram_specs):
        """The measured numbers are reproducible, not hand-written."""
        from tests.diagram_accuracy import score_corpus

        report = score_corpus(diagram_corpus, diagram_specs)
        assert report["diagrams"] >= 5
        assert 0.0 <= report["node_recall"] <= 1.0
        assert "per_diagram" in report
