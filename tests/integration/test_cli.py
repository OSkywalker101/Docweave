"""CLI: exit codes, output routing, batch behaviour and machine-readable output."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from docweave.cli import (
    EXIT_INPUT,
    EXIT_OK,
    EXIT_PARTIAL,
    cli,
)


@pytest.fixture
def run():
    runner = CliRunner()
    def _run(*args, **kwargs):
        return runner.invoke(cli, list(args), catch_exceptions=False, **kwargs)
    return _run


class TestBasics:
    def test_help(self, run):
        result = run("--help")
        assert result.exit_code == 0
        assert "convert" in result.output
        assert "capabilities" in result.output

    def test_version(self, run):
        result = run("--version")
        assert result.exit_code == EXIT_OK
        assert "docweave" in result.output

    def test_capabilities_lists_limitations(self, run):
        result = run("capabilities")
        assert result.exit_code == EXIT_OK
        assert "cannot do" in result.output
        assert "pymupdf" in result.output

    def test_capabilities_json(self, run):
        result = run("capabilities", "--json")
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert all("capabilities" in entry for entry in payload)


class TestConvertSingle:
    def test_markdown_to_stdout(self, run, fx):
        result = run("convert", str(fx("sample.md")))
        assert result.exit_code == EXIT_OK
        assert "# Title" in result.output

    def test_write_to_file(self, run, fx, scratch):
        out = scratch / "out.md"
        result = run("convert", str(fx("sample.md")), "-o", str(out))
        assert result.exit_code == EXIT_OK
        assert out.exists()
        assert "# Title" in out.read_text(encoding="utf-8")

    def test_format_from_extension(self, run, fx, scratch):
        out = scratch / "out.json"
        run("convert", str(fx("sample.md")), "-o", str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "docweave.document/1.0"

    def test_explicit_format_flag(self, run, fx, scratch):
        out = scratch / "out.jsonl"
        result = run("convert", str(fx("sample.md")), "--format", "jsonl", "-o", str(out))
        assert result.exit_code == EXIT_OK
        assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])

    def test_unknown_format_is_rejected(self, run, fx):
        result = run("convert", str(fx("sample.md")), "--format", "pdf")
        assert result.exit_code != EXIT_OK

    def test_report_format(self, run, fx, scratch):
        out = scratch / "report.json"
        run("convert", str(fx("sample.docx")), "--format", "report", "-o", str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "docweave.report/1.0"

    def test_missing_input_is_exit_code_2(self, run, scratch):
        result = run("convert", str(scratch / "absent.md"))
        assert result.exit_code == EXIT_INPUT

    def test_encrypted_pdf_is_exit_code_2(self, run, fx):
        result = run("convert", str(fx("encrypted.pdf")))
        assert result.exit_code == EXIT_INPUT

    def test_failed_document_is_exit_code_5(self, run, fx):
        result = run("convert", str(fx("scanned.pdf")), "--quiet")
        assert result.exit_code == EXIT_PARTIAL

    def test_diagnostics_are_printed(self, run, fx):
        result = run("convert", str(fx("scanned.pdf")))
        assert "heuristic_likely_scanned" in result.output

    def test_quiet_suppresses_the_document_body(self, run, fx):
        result = run("convert", str(fx("sample.md")), "--quiet")
        assert "# Title" not in result.output

    def test_config_file_is_applied(self, run, fx, scratch):
        cfg = scratch / "cfg.json"
        cfg.write_text(json.dumps({"pdf_tables": "never"}), encoding="utf-8")
        out = scratch / "o.md"
        result = run("convert", str(fx("table.pdf")), "-c", str(cfg), "-o", str(out))
        assert result.exit_code == EXIT_OK
        assert "| Region |" not in out.read_text(encoding="utf-8")

    def test_bad_config_file_is_reported(self, run, fx, scratch):
        cfg = scratch / "cfg.json"
        cfg.write_text('{"nonsense": 1}', encoding="utf-8")
        result = run("convert", str(fx("sample.md")), "-c", str(cfg))
        assert result.exit_code == EXIT_INPUT
        assert "nonsense" in result.output


class TestBatch:
    def test_directory_conversion(self, run, scratch, corpus):
        out = scratch / "out"
        result = run("convert", str(corpus), "-o", str(out), "-j", "4", "--quiet")
        assert result.exit_code == EXIT_PARTIAL  # corpus contains deliberately broken files
        assert (out / "sample.md").exists()
        assert (out / "simple.md").exists()

    def test_colliding_stems_do_not_overwrite(self, run, scratch):
        src = scratch / "in"
        src.mkdir()
        (src / "report.md").write_text("# from markdown\n", encoding="utf-8")
        (src / "report.html").write_text("<h1>from html</h1>", encoding="utf-8")
        (src / "report.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        out = scratch / "out"
        run("convert", str(src), "-o", str(out), "--quiet")
        written = sorted(p.name for p in out.iterdir())
        assert len(written) == 3
        assert len(set(written)) == 3
        contents = "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
        assert "from markdown" in contents and "from html" in contents

    def test_batch_survives_a_bad_file(self, run, scratch):
        src = scratch / "in"
        src.mkdir()
        (src / "ok.md").write_text("# fine\n", encoding="utf-8")
        (src / "bad.bin").write_bytes(b"")
        out = scratch / "out"
        result = run("convert", str(src), "-o", str(out))
        assert result.exit_code == EXIT_PARTIAL
        assert (out / "ok.md").exists()
        assert not (out / "bad.md").exists()

    def test_file_output_with_many_inputs_is_rejected(self, run, corpus, scratch):
        result = run(
            "convert", str(corpus / "sample.md"), str(corpus / "sample.csv"), "-o", str(scratch / "x.md")
        )
        assert result.exit_code == EXIT_INPUT

    def test_recursive_flag(self, run, scratch, corpus):
        nested = scratch / "in"
        (nested / "deep").mkdir(parents=True)
        (nested / "deep" / "a.md").write_text("# deep\n", encoding="utf-8")
        out = scratch / "out"
        run("convert", str(nested), "-r", "-o", str(out), "--quiet")
        assert (out / "a.md").exists()

    def test_empty_selection_is_exit_code_2(self, run, scratch):
        empty = scratch / "empty"
        empty.mkdir()
        assert run("convert", str(empty)).exit_code == EXIT_INPUT

    def test_batch_json_summary(self, run, scratch, corpus):
        result = run(
            "convert",
            str(corpus / "sample.md"),
            str(corpus / "sample.csv"),
            "--json-output",
        )
        payload = json.loads(result.output)
        assert payload["total"] == 2
        assert payload["succeeded"] == 2


class TestInspect:
    def test_human_output(self, run, fx):
        result = run("inspect", str(fx("sample.xlsx")))
        assert result.exit_code == EXIT_OK
        assert "xlsx" in result.output

    def test_json_output(self, run, fx):
        result = run("inspect", str(fx("simple.pdf")), "--json")
        payload = json.loads(result.output)
        assert payload[0]["detected_format"] == "pdf"
        assert len(payload[0]["sha256"]) == 64

    def test_extension_mismatch_is_visible(self, run, fx):
        result = run("inspect", str(fx("liar.pdf")), "--json")
        payload = json.loads(result.output)
        assert payload[0]["detected_format"] == "csv"
        assert payload[0]["extension_mismatch"] is True

    def test_missing_file_exits_nonzero(self, run, scratch):
        assert run("inspect", str(scratch / "nope.pdf")).exit_code == EXIT_PARTIAL


class TestValidate:
    def test_reports_diagnostics(self, run, fx):
        result = run("validate", str(fx("scanned.pdf")))
        assert "heuristic_likely_scanned" in result.output

    def test_json_report(self, run, fx):
        result = run("validate", str(fx("sample.docx")), "--json")
        payload = json.loads(result.output)
        assert payload["total"] == 1
        assert payload["succeeded"] == 1
        assert "files" in payload
        assert isinstance(payload["files"][0]["diagnostics"], list)

    def test_clean_document_exits_zero(self, run, fx):
        result = run("validate", str(fx("sample.docx")), "--quiet")
        assert result.exit_code == EXIT_OK

    def test_failing_document_exits_nonzero(self, run, fx):
        assert run("validate", str(fx("scanned.pdf")), "--quiet").exit_code == EXIT_PARTIAL

    def test_batch_validation(self, run, corpus):
        result = run("validate", str(corpus), "--json")
        payload = json.loads(result.output)
        assert payload["total"] > 5
