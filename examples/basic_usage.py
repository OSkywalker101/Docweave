"""Python API usage examples.

Run: python examples/basic_usage.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from docweave import ConversionConfig, Converter, convert


def example_simple_conversion() -> None:
    """The 90% case: convert a document to Markdown."""
    source = Path("tests/fixtures/sample.docx")
    if not source.exists():
        print("(run `python tests/fixtures_builder.py` first)")
        return

    result = convert(source)
    print(result.to_markdown())


def example_structured_output() -> None:
    """The RAG-pipeline case: keep structure and provenance."""
    source = Path("tests/fixtures/simple.pdf")
    if not source.exists():
        return

    result = convert(source)
    payload = json.loads(result.to_json())

    for page in payload["pages"]:
        print(f"page {page['index']}: {len(page['blocks'])} block(s)")

    # Every node knows where it came from.
    for record in [json.loads(line) for line in result.to_jsonl().splitlines()][:5]:
        print(
            f"  {record['type']:<10} page={record.get('page')} "
            f"origin={record['origin']} bbox={record.get('bbox')}"
        )


def example_diagnostics() -> None:
    """The differentiator: find out what the engine could not read."""
    source = Path("tests/fixtures/scanned.pdf")
    if not source.exists():
        return

    result = convert(source)
    for diagnostic in result.diagnostics.sorted():
        print(diagnostic.to_line())
        if diagnostic.reason:
            print(f"    reason: {diagnostic.reason}")
        if diagnostic.hint:
            print(f"    hint:   {diagnostic.hint}")

    print(f"\nexit signal ok={result.ok}")


def example_configuration() -> None:
    """Tune behaviour for a specific downstream consumer."""
    source = Path("tests/fixtures/sample.xlsx")
    if not source.exists():
        return

    config = ConversionConfig(
        table_rendering="list",      # A1-referenced cells instead of a fake grid
        include_headers_footers=False,
        markdown_include_diagnostics=True,
    )
    print(convert(source, config).to_markdown())


def example_batch() -> None:
    """Batch conversion with per-file error isolation."""
    corpus = Path("tests/fixtures")
    if not corpus.exists():
        return

    report = Converter().convert_many(sorted(corpus.glob("*.pdf")) + sorted(corpus.glob("*.md")))
    print(json.dumps(report.summary(), indent=2))

    for item in report.failed:
        print(f"  failed: {item.path.name} -> {item.reason}")


def example_assets() -> None:
    """Extract embedded images to disk."""
    source = Path("tests/fixtures/image.docx")
    if not source.exists():
        return

    with tempfile.TemporaryDirectory() as tmp:
        config = ConversionConfig(extract_assets=True, assets_dir=Path(tmp))
        result = convert(source, config)
        for asset in result.document.assets:
            state = "extracted" if asset.extracted else "referenced"
            print(f"{asset.asset_id} [{state}] {asset.media_type} {asset.byte_size} bytes")


if __name__ == "__main__":
    example_simple_conversion()
    print("\n" + "=" * 70)
    example_structured_output()
    print("\n" + "=" * 70)
    example_diagnostics()
    print("\n" + "=" * 70)
    example_configuration()
    print("\n" + "=" * 70)
    example_batch()
    print("\n" + "=" * 70)
    example_assets()
