"""Integration: assets, batch processing, limits, and the Python API surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docweave import (
    ConversionConfig,
    ConversionResult,
    Converter,
    capabilities,
    convert,
    convert_to,
    discover,
    inspect,
)
from docweave.assets import AssetCollector
from docweave.exceptions import DeadlineExceededError, LimitExceededError


class TestPythonApi:
    def test_module_level_convert(self, fx):
        result = convert(fx("sample.md"))
        assert isinstance(result, ConversionResult)
        assert "Title" in result.to_markdown()

    def test_convert_to_string(self, fx):
        assert convert_to(fx("sample.md"), "markdown").startswith("# Title")
        assert json.loads(convert_to(fx("sample.md"), "json"))["schema"] == "docweave.document/1.0"
        assert convert_to(fx("sample.md"), "jsonl").count("\n") >= 1
        assert json.loads(convert_to(fx("sample.md"), "report"))["schema"] == "docweave.report/1.0"

    def test_unknown_output_format_is_rejected(self, fx):
        from docweave.exceptions import UnsupportedFormatError

        with pytest.raises(UnsupportedFormatError):
            convert(fx("sample.md")).render("pdf")

    def test_converter_is_reusable(self, fx):
        converter = Converter()
        first = converter.convert(fx("sample.md"))
        second = converter.convert(fx("sample.csv"))
        assert first.document is not second.document
        assert first.source.sha256 != second.source.sha256

    def test_inspect_without_conversion(self, fx):
        result = inspect(fx("sample.xlsx"))
        assert result.detection.format == "xlsx"
        assert result.readable

    def test_capabilities_reports_every_adapter(self):
        entries = capabilities()
        names = {e["adapter"] for e in entries}
        assert {"pymupdf", "python-docx", "python-pptx", "openpyxl", "html", "markdown", "delimited"} <= names

    def test_capabilities_declare_limitations(self):
        for entry in capabilities():
            assert entry["limitations"], f"{entry['adapter']} must declare limitations"
            assert entry["cost"] in {"cheap", "moderate", "expensive", "unknown"}

    def test_discover_finds_files(self, corpus):
        found = discover(corpus)
        assert len(found) > 5
        assert all(p.is_file() for p in found)

    def test_discover_single_file(self, fx):
        assert discover(fx("sample.md")) == [fx("sample.md")]

    def test_no_server_dependency(self):
        import sys

        convert  # noqa: B018 - importing must not pull in a web framework
        assert "fastapi" not in sys.modules
        assert "uvicorn" not in sys.modules

    def test_heavy_parsers_are_not_imported_by_the_core(self, fx, monkeypatch):
        # Importing docweave and converting a text file must not load the PDF,
        # Word or Excel libraries.
        import subprocess
        import sys

        code = (
            "import sys; import docweave; "
            f"docweave.convert(r'{fx('sample.md')}'); "
            "print(','.join(m for m in ('pymupdf','fitz','docx','pptx','openpyxl') if m in sys.modules))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == ""


class TestAssets:
    def test_images_are_referenced_by_default(self, fx):
        result = convert(fx("image.docx"))
        assert result.document.assets
        assert all(not a.extracted for a in result.document.assets)

    def test_extraction_writes_files(self, fx, scratch):
        out = scratch / "assets"
        result = convert(
            fx("image.docx"), ConversionConfig(extract_assets=True, assets_dir=out)
        )
        extracted = [a for a in result.document.assets if a.extracted]
        assert extracted
        for asset in extracted:
            assert asset.path and Path(asset.path).exists()
            assert asset.sha256 and len(asset.sha256) == 64

    def test_asset_budget_is_enforced(self, fx, scratch):
        config = ConversionConfig(
            extract_assets=True, assets_dir=scratch / "a", max_asset_bytes=1
        )
        with pytest.raises(LimitExceededError):
            convert(fx("image.docx"), config)

    def test_collector_without_data_records_a_reference_only(self):
        collector = AssetCollector()
        ref = collector.add(None, media_type="image/png", asset_id="a1")
        assert ref.media_type == "image/png"
        assert ref.extension == ".png"
        assert not ref.extracted
        assert ref.sha256 is None

    def test_collector_is_deterministic(self, scratch):
        digests = []
        for _ in range(2):
            collector = AssetCollector(directory=scratch / "assets", enabled=True)
            collector.add(b"\x89PNGfake", media_type="image/png", asset_id="a1")
            digests.append(collector.refs()[0].sha256)
        assert digests[0] == digests[1]

    def test_markdown_can_link_extracted_assets(self, fx, scratch):
        config = ConversionConfig(
            extract_assets=True,
            assets_dir=scratch / "a",
            image_rendering="path",
        )
        out = convert(fx("image.docx"), config).to_markdown()
        assert "[image:" in out


class TestBatch:
    def test_batch_continues_past_failures(self, corpus, scratch):
        targets = [corpus / "sample.md", corpus / "empty.bin", corpus / "sample.csv"]
        report = Converter().convert_many(targets)
        assert report.summary()["total"] == 3
        assert len(report.succeeded) == 2
        assert len(report.failed) == 1
        assert "empty" in report.failed[0].path.name

    def test_batch_summary_shape(self, corpus):
        report = Converter().convert_many([corpus / "sample.md", corpus / "sample.csv"])
        summary = report.summary()
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0
        assert "diagnostics" in summary
        assert isinstance(summary["codes"], list)

    def test_concurrency_does_not_change_results(self, corpus):
        targets = [corpus / n for n in ("sample.md", "sample.csv", "sample.html", "sample.json")]
        serial = Converter().convert_many(targets, concurrency=1)
        parallel = Converter().convert_many(targets, concurrency=4)
        assert [i.result.document.digest() for i in serial.succeeded] == [
            i.result.document.digest() for i in parallel.succeeded
        ]

    def test_empty_batch(self):
        report = Converter().convert_many([])
        assert report.summary()["total"] == 0

    def test_convert_safe_never_raises(self, corpus):
        result, error, kind = Converter().convert_safe(corpus / "empty.bin")
        assert result is None
        assert error and "empty" in error
        assert kind == "EmptyFileError"


class TestLimits:
    def test_page_limit(self, fx):
        with pytest.raises(LimitExceededError):
            convert(fx("simple.pdf"), ConversionConfig(max_pages=1))

    def test_file_size_limit(self, scratch):
        path = scratch / "big.md"
        path.write_text("# t\n\n" + ("word " * 250_000), encoding="utf-8")
        assert path.stat().st_size > 1024 * 1024
        with pytest.raises(Exception) as exc:
            convert(path, ConversionConfig(max_file_size_mb=1))
        assert "exceeds the configured limit" in str(exc.value)

    def test_pptx_images_are_referenced(self, fx):
        assert convert(fx("image.pptx")).document.assets

    def test_deadline_is_enforced(self, fx):
        with pytest.raises(DeadlineExceededError):
            convert(
                fx("large.csv"),
                ConversionConfig(deadline_seconds=0.0000001),
            )

    def test_limits_are_reported_not_hidden(self, fx):
        result = convert(fx("large.csv"), ConversionConfig(max_csv_rows=5))
        assert result.diagnostics.has("requires_larger_limits")


class TestRobustness:
    def test_hostile_inputs_do_not_crash_the_engine(self, corpus):
        for name in sorted(p.name for p in corpus.iterdir()):
            result, error, kind = Converter().convert_safe(corpus / name)
            assert (result is None) != (error is None)

    def test_path_with_spaces(self, scratch, fx):
        target = scratch / "a file with spaces.md"
        target.write_bytes(fx("sample.md").read_bytes())
        assert "Title" in convert(target).to_markdown()

    def test_unicode_survives(self, scratch):
        path = scratch / "u.md"
        path.write_text("# 日本語\n\nEmoji: ✅\nRTL: مرحبا\n", encoding="utf-8")
        out = convert(path).to_markdown()
        assert "日本語" in out and "✅" in out and "مرحبا" in out

    def test_output_is_utf8(self, scratch, fx):
        target = scratch / "out.md"
        target.write_text(convert(fx("sample.html")).to_markdown(), encoding="utf-8")
        assert target.read_bytes().decode("utf-8")

    def test_result_is_json_serialisable(self, fx):
        result = convert(fx("sample.xlsx"))
        json.dumps(
            {
                "document": result.document.model_dump(mode="json"),
                "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics.sorted()],
            }
        )
