"""Reproducible comparison against Microsoft MarkItDown.

Run:

    python benchmarks/compare_markitdown.py --corpus tests/fixtures
    python benchmarks/compare_markitdown.py --corpus tests/fixtures --json

What this measures, and what it deliberately does not
----------------------------------------------------

Measured here, on identical inputs, in the same process:

* wall-clock latency (median of N runs)
* peak resident memory (via ``psutil`` / ``tracemalloc``)
* extracted character and word counts
* node, page, sheet and table coverage
* how much content each system *reports* as missing

**Not** measured: extraction accuracy. Accuracy needs a ground-truth
transcription of each document, and this corpus does not have one. Producing an
accuracy number anyway — from an LLM, from a fuzzy match against itself, from
MarkItDown's own output — would be circular and is refused. The harness reports
the limitation explicitly rather than filling the hole with a guess.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.accuracy import (  # noqa: E402
    DocumentScore,
    aggregate,
    score_document,
    strip_comment_noise,
    tables_from_markdown,
)
from docweave import convert  # noqa: E402
from tests.ground_truth import GroundTruth, build_registry  # noqa: E402

#: Formats both systems are expected to handle.
COMPARABLE_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv", ".json", ".xml", ".md", ".txt"}
RUNS = 3


@dataclass
class SampleResult:
    name: str
    format: str
    ok: bool
    error: str | None = None
    latency_ms: float = 0.0
    latency_median_ms: float = 0.0
    peak_bytes: int = 0
    characters: int = 0
    words: int = 0
    lines: int = 0
    nodes: int = 0
    pages: int = 0
    sheets: int = 0
    tables: int = 0
    diagnostics: int = 0
    structural_problems: int = 0
    unverified: int = 0
    #: Markdown with comment noise and diagnostic trailers removed.
    clean_markdown: str = ""
    accuracy: DocumentScore | None = None


@dataclass
class Comparison:
    file: str
    docweave: SampleResult | None = None
    markitdown: SampleResult | None = None
    #: True when this file has author-provided ground truth and was scored.
    scored: bool = False
    notes: list[str] = field(default_factory=list)


def _words(text: str) -> int:
    return len(text.split())


def measure_docweave(path: Path, runs: int = RUNS) -> SampleResult:
    """Time and measure a DocWeave conversion, taking the median of ``runs``."""
    timings: list[float] = []
    result = None
    peak = 0
    for _ in range(runs):
        tracemalloc.start()
        started = time.perf_counter()
        try:
            result = convert(path)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            timings.append(elapsed)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return SampleResult(
                name=path.name,
                format=path.suffix.lstrip(".").lower(),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round(elapsed, 3),
                latency_median_ms=round(statistics.median(timings), 3),
                peak_bytes=peak,
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        timings.append(elapsed)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    text = result.to_markdown()
    clean = strip_comment_noise(text)
    document = result.document
    counts = result.diagnostics.counts()
    return SampleResult(
        name=path.name,
        format=path.suffix.lstrip(".").lower(),
        ok=True,
        latency_ms=round(timings[-1], 3),
        latency_median_ms=round(statistics.median(timings), 3),
        peak_bytes=peak,
        characters=len(text),
        words=_words(text),
        lines=text.count("\n") + 1,
        nodes=document.node_count(),
        pages=len(document.pages),
        sheets=len(document.sheets),
        tables=sum(1 for _ in document.tables()),
        diagnostics=len(result.diagnostics.diagnostics),
        structural_problems=counts["structural_problem"],
        unverified=counts["unverified_content"],
        clean_markdown=clean,
    )


def measure_markitdown(path: Path, runs: int = RUNS) -> SampleResult:
    """Time and measure the MarkItDown baseline, or report why it is unavailable."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        return SampleResult(
            name=path.name,
            format=path.suffix.lstrip(".").lower(),
            ok=False,
            error=f"markitdown is not importable: {exc}",
        )

    timings: list[float] = []
    text = ""
    peak = 0
    error: str | None = None
    converter = MarkItDown()
    for _ in range(runs):
        tracemalloc.start()
        started = time.perf_counter()
        try:
            text = converter.convert(str(path)).text_content or ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            text = ""
        timings.append((time.perf_counter() - started) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if error:
            break

    if error:
        return SampleResult(
            name=path.name,
            format=path.suffix.lstrip(".").lower(),
            ok=False,
            error=error,
            latency_ms=round(timings[-1], 3),
            latency_median_ms=round(statistics.median(timings), 3),
            peak_bytes=peak,
        )

    return SampleResult(
        name=path.name,
        format=path.suffix.lstrip(".").lower(),
        ok=True,
        latency_ms=round(timings[-1], 3),
        latency_median_ms=round(statistics.median(timings), 3),
        peak_bytes=peak,
        characters=len(text),
        words=_words(text),
        lines=text.count("\n") + 1,
        nodes=0,  # MarkItDown exposes no intermediate structure
        pages=0,
        sheets=0,
        tables=text.count("|") and text.count("\n|") or 0,
        diagnostics=0,
        structural_problems=0,
        unverified=0,
        clean_markdown=strip_comment_noise(text),
    )


def discover(corpus: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in corpus.glob(pattern) if p.is_file() and p.suffix.lower() in COMPARABLE_SUFFIXES
    )


def compare_file(
    path: Path, runs: int = RUNS, truth: GroundTruth | None = None
) -> Comparison:
    ours = measure_docweave(path, runs)
    theirs = measure_markitdown(path, runs)
    comparison = Comparison(file=path.name, docweave=ours, markitdown=theirs)

    if truth is not None and ours.ok and theirs.ok:
        comparison.scored = True
        ours.accuracy = score_document(
            truth,
            ours.clean_markdown,
            tables_from_markdown(ours.clean_markdown),
            containers_found=ours.pages or ours.sheets,
        )
        theirs.accuracy = score_document(
            truth,
            theirs.clean_markdown,
            tables_from_markdown(theirs.clean_markdown),
            containers_found=0,
        )
        comparison.notes.extend(truth.notes)
        # Container coverage is only meaningful from a structured extractor.
        if truth.pages or truth.slides or truth.sheets:
            theirs.accuracy.container_coverage = None
    elif not ours.ok or not theirs.ok:
        comparison.notes.append("at least one system failed; see per-system error")
    return comparison


def environment() -> dict:
    import os

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "runs_per_file": RUNS,
    }
    try:
        import markitdown

        info["markitdown"] = getattr(markitdown, "__version__", "installed")
    except Exception:
        info["markitdown"] = "not installed"
    import docweave

    info["docweave"] = docweave.__version__
    return info


def summarise(comparisons: Iterable[Comparison]) -> dict:
    rows = list(comparisons)
    both = [c for c in rows if c.docweave and c.markitdown and c.docweave.ok and c.markitdown.ok]

    def median(values: list[float]) -> float:
        return round(statistics.median(values), 3) if values else 0.0

    summary: dict = {
        "files": len(rows),
        "comparable": len(both),
        "docweave": {
            "median_latency_ms": median([c.docweave.latency_median_ms for c in both]),
            "total_chars": sum(c.docweave.characters for c in both),
            "total_words": sum(c.docweave.words for c in both),
            "files_failed": sum(1 for c in rows if c.docweave and not c.docweave.ok),
        },
        "markitdown": {
            "median_latency_ms": median([c.markitdown.latency_median_ms for c in both]),
            "total_chars": sum(c.markitdown.characters for c in both),
            "total_words": sum(c.markitdown.words for c in both),
            "files_failed": sum(1 for c in rows if c.markitdown and not c.markitdown.ok),
        },
        "structure": {
            "note": (
                "MarkItDown emits Markdown only, so node/page/sheet counts are not "
                "comparable. DocWeave's counts describe its intermediate representation."
            )
        },
    }

    scored = [c for c in rows if c.scored and c.docweave.accuracy and c.markitdown.accuracy]
    if scored:
        ours = aggregate([c.docweave.accuracy for c in scored])
        theirs = aggregate([c.markitdown.accuracy for c in scored])
        summary["accuracy"] = {
            "measured": True,
            "scored_documents": len(scored),
            "docweave": ours,
            "markitdown": theirs,
            "ground_truth": (
                "Author-provided: every fixture is rendered from a declarative spec in "
                "tests/ground_truth.py, so the expected extraction is exact by construction."
            ),
            "scope": (
                "Synthetic, author-controlled corpus: known boundaries, no scanning noise, "
                "no font substitution. This is a lower bound on real-world accuracy and not a "
                "substitute for a labelled public corpus."
            ),
            "caveats": [
                "CER alone hides omission and invention; word recall and word precision are "
                "reported alongside it for exactly that reason.",
                "Container coverage is only scored for DocWeave, which exposes an "
                "intermediate representation; MarkItDown has no equivalent to score.",
                "A document with no ground-truth text (a scan) scores 0 for any system that "
                "reports text, because inventing text is scored as error.",
            ],
        }
    else:
        summary["accuracy"] = {
            "measured": False,
            "reason": (
                "No ground truth was available for the compared files. Character and word "
                "counts describe volume, not correctness: a system that emits fewer "
                "characters is not thereby more accurate."
            ),
        }
    return summary


def format_report(summary: dict, comparisons: list[Comparison]) -> str:
    lines = [
        "DocWeave vs MarkItDown",
        "=" * 92,
        f"environment: python {summary['environment']['python']} on "
        f"{summary['environment']['platform']}",
        f"docweave {summary['environment']['docweave']} / markitdown "
        f"{summary['environment']['markitdown']} / "
        f"{summary['environment']['runs_per_file']} runs per file",
        "",
        f"{'file':<26}{'dw ms':>9}{'mid ms':>9}{'dw chars':>10}{'mid chars':>11}",
        "-" * 92,
    ]
    for comparison in comparisons:
        ours, theirs = comparison.docweave, comparison.markitdown
        if not (ours and theirs and ours.ok and theirs.ok):
            lines.append(f"{comparison.file:<26}  not comparable")
            continue
        lines.append(
            f"{comparison.file:<26}{ours.latency_ms:>9.1f}{ours.latency_median_ms:>9.1f}"
            f"{ours.characters:>10d}{theirs.characters:>11d}"
        )
    lines.append("-" * 92)
    lines.append(
        f"median latency: docweave {summary['docweave']['median_latency_ms']:.1f} ms, "
        f"markitdown {summary['markitdown']['median_latency_ms']:.1f} ms"
    )
    lines.append(
        f"total characters: docweave {summary['docweave']['total_chars']}, "
        f"markitdown {summary['markitdown']['total_chars']}"
    )

    accuracy = summary["accuracy"]
    lines.append("")
    if not accuracy.get("measured"):
        lines.append("ACCURACY: not measured.")
        lines.append(f"  {accuracy['reason']}")
        return "\n".join(lines)

    ours, theirs = accuracy["docweave"], accuracy["markitdown"]
    lines.append(f"ACCURACY vs author-provided ground truth ({accuracy['scored_documents']} documents)")
    lines.append("-" * 92)
    lines.append(f"{'metric':<28}{'docweave':>14}{'markitdown':>14}{'winner':>12}")
    for key, _better_low in (
        ("word_recall", True),
        ("word_precision", True),
        ("word_f1", True),
        ("reading_order", True),
        ("table_cell_recall", True),
        ("table_cell_precision", True),
    ):
        a, b = ours.get(key, 0.0), theirs.get(key, 0.0)
        winner = "docweave" if a > b else ("markitdown" if b > a else "tie")
        lines.append(f"{key:<28}{a:>14.4f}{b:>14.4f}{winner:>12}")
    a, b = ours.get("cer", 0.0), theirs.get("cer", 0.0)
    winner = "docweave" if a < b else ("markitdown" if b < a else "tie")
    lines.append(f"{'cer (lower is better)':<28}{a:>14.4f}{b:>14.4f}{winner:>12}")

    lines.append("")
    lines.append(f"{'per-document word F1':<26}{'docweave':>12}{'markitdown':>12}")
    for comparison in comparisons:
        if not comparison.scored:
            continue
        ours_score = comparison.docweave.accuracy
        theirs_score = comparison.markitdown.accuracy
        lines.append(
            f"{comparison.file:<26}{ours_score.word_f1:>12.3f}{theirs_score.word_f1:>12.3f}"
        )

    lines.append("")
    lines.append(f"scope: {accuracy['scope']}")
    for caveat in accuracy["caveats"]:
        lines.append(f"caveat: {caveat}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    global RUNS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="tests/fixtures", help="Directory to benchmark.")
    parser.add_argument("--runs", type=int, default=RUNS, help="Runs per file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--output", help="Write the JSON report here.")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 2

    files = discover(corpus)
    if not files:
        print(f"no comparable files under {corpus}", file=sys.stderr)
        return 2

    RUNS = args.runs

    comparisons = [
        compare_file(path, args.runs, build_registry().get(path.name)) for path in files
    ]
    summary = summarise(comparisons)
    summary["environment"] = environment()
    summary["per_file"] = [
        {"file": c.file, "docweave": asdict(c.docweave), "markitdown": asdict(c.markitdown)}
        for c in comparisons
        if c.docweave and c.markitdown
    ]

    if args.json:
        payload = json.dumps(summary, indent=2)
        print(payload)
    else:
        print(format_report(summary, comparisons))

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
