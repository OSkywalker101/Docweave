"""The MarkItDown comparison harness itself.

The harness is tested for what it *refuses* to claim as much as for what it
measures. A benchmark that quietly invents an accuracy number is worse than no
benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.accuracy import strip_markup  # noqa: E402
from benchmarks.compare_markitdown import (  # noqa: E402
    compare_file,
    discover,
    environment,
    format_report,
    measure_docweave,
    summarise,
)
from tests.ground_truth import build_registry  # noqa: E402
from tests.unit.test_renderers import _document, _ref  # noqa: E402


def _truth(name: str):
    return build_registry()[name]


class TestMeasurement:
    def test_docweave_measurement(self, fx):
        result = measure_docweave(fx("sample.md"), runs=1)
        assert result.ok
        assert result.characters > 0
        assert result.latency_ms >= 0
        assert result.peak_bytes > 0

    def test_failure_is_captured_not_raised(self, fx):
        result = measure_docweave(fx("empty.bin"), runs=1)
        assert not result.ok
        assert "empty" in (result.error or "")

    def test_markitdown_measurement(self, fx):
        from benchmarks.compare_markitdown import measure_markitdown

        result = measure_markitdown(fx("sample.md"), runs=1)
        # Either it worked, or it reported why. Both are acceptable outcomes.
        assert result.ok or result.error


class TestComparison:
    def test_compare_produces_both_sides(self, fx):
        comparison = compare_file(fx("sample.md"), runs=1)
        assert comparison.docweave is not None
        assert comparison.markitdown is not None

    def test_incomparable_files_are_noted(self, fx):
        comparison = compare_file(fx("empty.bin"), runs=1)
        if not (comparison.docweave and comparison.docweave.ok):
            assert comparison.notes


class TestHonesty:
    def test_unscored_files_say_so_explicitly(self, fx):
        """A file with no ground truth must not silently count as accurate."""
        summary = summarise([compare_file(fx("bad.xml"), runs=1)])
        assert summary["accuracy"]["measured"] is False
        assert "No ground truth" in summary["accuracy"]["reason"]

    def test_structure_comparison_is_qualified(self, fx):
        summary = summarise([compare_file(fx("sample.md"), runs=1)])
        assert "not comparable" in summary["structure"]["note"]

    def test_report_states_the_scope_limitation(self, fx):
        summary = summarise([compare_file(fx("sample.md"), runs=1, truth=_truth("sample.md"))])
        summary["environment"] = environment()
        text = format_report(summary, [compare_file(fx("sample.md"), runs=1, truth=_truth("sample.md"))])
        assert "lower bound" in text
        assert "author-provided ground truth" in text

    def test_environment_is_recorded(self):
        info = environment()
        for key in ("python", "platform", "machine", "cpu_count", "runs_per_file"):
            assert key in info

    def test_summary_is_json_serialisable(self, fx):
        summary = summarise([compare_file(fx("sample.md"), runs=1, truth=_truth("sample.md"))])
        summary["environment"] = environment()
        json.dumps(summary)

    def test_scope_limitation_is_stated(self, fx):
        summary = summarise([compare_file(fx("sample.md"), runs=1, truth=_truth("sample.md"))])
        accuracy = summary["accuracy"]
        assert accuracy["measured"] is True
        assert "lower bound" in accuracy["scope"]
        assert accuracy["caveats"]


class TestAccuracyScoring:
    def _score(self, fx, name):
        return compare_file(fx(name), runs=1, truth=_truth(name))

    def test_ground_truth_is_authored_not_guessed(self):
        truth = _truth("sample.csv")
        assert truth.fmt == "csv"
        assert truth.tables and truth.tables[0].header == ["name", "qty", "price"]

    def test_a_scored_file_reports_accuracy(self, fx):
        comparison = self._score(fx, "sample.md")
        assert comparison.scored
        assert comparison.docweave.accuracy is not None
        assert comparison.docweave.accuracy.word_f1 > 0.8

    def test_both_systems_are_scored_identically(self, fx):
        comparison = self._score(fx, "simple.pdf")
        assert comparison.docweave.accuracy is not None
        assert comparison.markitdown.accuracy is not None

    def test_table_cells_are_scored(self, fx):
        comparison = self._score(fx, "sample.csv")
        score = comparison.docweave.accuracy
        assert score.tables.expected == 1
        assert score.tables.cell_recall > 0.9

    def test_a_scan_is_not_rewarded_for_inventing_text(self, fx):
        """Correct output for a text-free scan is empty output."""
        comparison = self._score(fx, "scanned.pdf")
        assert comparison.docweave.accuracy.word_f1 <= 0.999

    def test_running_furniture_is_excluded_from_ground_truth(self):
        truth = _truth("simple.pdf")
        assert "CONFIDENTIAL" in truth.excluded
        assert all("CONFIDENTIAL" not in b for b in truth.blocks)

    def test_scaffold_is_not_scored_as_content(self, fx):
        """Engine prose must not be charged to the converter as hallucination."""
        result = measure_docweave(fx("sample.xlsx"), runs=1)
        cleaned = strip_markup(result.clean_markdown)
        assert "formula cell" not in cleaned.lower()
        assert "worksheet:" not in cleaned.lower()

    def test_normalisation_does_not_hide_omission(self):
        """A dropped word must lower recall, not normalise away."""
        from benchmarks.accuracy import multiset_scores

        recall, _, _ = multiset_scores("alpha beta gamma delta", "alpha beta")
        assert recall == pytest.approx(0.5)

    def test_normalisation_does_not_hide_invention(self):
        from benchmarks.accuracy import multiset_scores

        _, precision, _ = multiset_scores("alpha beta", "alpha beta gamma delta invented")
        assert precision < 0.5

    def test_table_extractor_terminates_on_html_without_a_closing_row(self):
        """Regression: an unmatched <tr> on a later line looped forever."""
        from benchmarks.accuracy import tables_from_markdown

        tables = tables_from_markdown("intro\n<table>\n<tr><td>a</td></tr>\n</table>\ntail")
        assert len(tables) == 1
        assert tables[0]["cells"] == ["a"]

    def test_table_extractor_terminates_on_a_plain_bullet_list(self):
        """Regression: a bullet list took the spreadsheet branch and spun.

        The branch was reached by the line pattern alone, produced no rows, and
        looped without advancing. Any document with bullets would hang the run.
        """
        from benchmarks.accuracy import tables_from_markdown

        assert tables_from_markdown("- one\n- two\n- three\n") == []

    @pytest.mark.parametrize(
        "value",
        ["harbour; B1: docks", 'say "hi"; B1: x', "alpha; beta", "col 2: ghost"],
    )
    def test_a_quoted_cell_survives_a_round_trip(self, value):
        """The renderer's quoting must be undone by whoever reads it.

        A value that looks like another cell is quoted so it cannot be miscounted.
        A consumer that does not unquote it has merely moved the bug. The test
        goes through the real renderer rather than hand-building the escaped
        string, so it cannot drift from what is actually emitted.
        """
        from benchmarks.accuracy import _parse_sheet_cells
        from docweave.models import Cell, Table
        from docweave.renderers import MarkdownRenderer

        rendered = MarkdownRenderer().render(
            _document(
                [
                    Table(
                        id="t",
                        rows=[
                            [
                                Cell(
                                    id="c0",
                                    text=value,
                                    row=0,
                                    column=0,
                                    locator={"kind": "sheet", "cell": "A1"},
                                )
                            ]
                        ],
                        header_rows=0,
                        sparse=True,
                        sources=[_ref()],
                    )
                ]
            )
        )
        row = next(l for l in rendered.splitlines() if l.startswith("- "))
        assert _parse_sheet_cells(row[2:]) == [value]


class TestDiscovery:
    def test_only_comparable_suffixes(self, fx):
        from benchmarks.compare_markitdown import COMPARABLE_SUFFIXES

        found = discover(fx("sample.md").parent)
        assert found
        assert all(p.suffix.lower() in COMPARABLE_SUFFIXES for p in found)
        assert any(p.name == "sample.md" for p in found)

    def test_diagrams_are_not_in_the_comparison_set(self, fx):
        """Image files are not comparable documents and must not enter the run."""
        from benchmarks.compare_markitdown import COMPARABLE_SUFFIXES

        assert ".png" not in COMPARABLE_SUFFIXES

    def test_empty_corpus_yields_nothing(self, scratch):
        assert discover(scratch) == []


class TestDiagramBenchmarkHarness:
    """The diagram accuracy report is a measurement, so it is testable."""

    def test_report_scores_the_corpus(self, diagram_corpus, diagram_specs):
        from tests.diagram_accuracy import format_report, score_corpus

        report = score_corpus(diagram_corpus, diagram_specs)
        assert report["diagrams"] == len(diagram_specs)
        text = format_report(report)
        assert "node recall" in text and "edge recall" in text

    @pytest.mark.parametrize(
        "metric",
        [
            "node_recall",
            "node_precision",
            "edge_recall",
            "edge_precision",
            "kind_accuracy",
            "direction_proof_rate",
        ],
    )
    def test_every_metric_is_a_proportion(self, diagram_corpus, diagram_specs, metric):
        """Scores must stay in [0, 1].

        The harness once defined precision as detected/expected, which is an
        unbounded over-generation ratio and reported 1.202. Precision is now
        matched/detected, and this is the guard that keeps it from drifting
        back into a number that can exceed a perfect score.
        """
        from tests.diagram_accuracy import score_corpus

        report = score_corpus(diagram_corpus, diagram_specs)
        assert 0.0 <= report[metric] <= 1.0, f"{metric}={report[metric]}"

    def test_inventing_structure_costs_precision(self, diagram_corpus, diagram_specs):
        """Overshoot must be visible as lost precision, not as a ratio > 1."""
        from tests.diagram_accuracy import score_corpus

        report = score_corpus(diagram_corpus, diagram_specs)
        for entry in report["per_diagram"]:
            if entry["edges"]["detected"] > entry["edges"]["expected"]:
                assert entry["edge_precision"] < 1.0, entry["name"]

    def test_per_diagram_entries_exist(self, diagram_corpus, diagram_specs):
        from tests.diagram_accuracy import score_corpus

        report = score_corpus(diagram_corpus, diagram_specs)
        names = {entry["name"] for entry in report["per_diagram"]}
        assert names == set(diagram_specs)

    def test_precision_is_reported_separately_from_recall(self, diagram_corpus, diagram_specs):
        """A detector that invents edges must not look good on recall alone."""
        from tests.diagram_accuracy import score_corpus

        report = score_corpus(diagram_corpus, diagram_specs)
        for entry in report["per_diagram"]:
            assert "node_precision" in entry
            assert "edge_precision" in entry

    def test_measurements_are_stable(self, diagram_corpus, diagram_specs):
        from tests.diagram_accuracy import score_corpus

        first = score_corpus(diagram_corpus, diagram_specs)
        second = score_corpus(diagram_corpus, diagram_specs)
        assert first["per_diagram"] == second["per_diagram"]
