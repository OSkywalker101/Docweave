"""Fail when a published metric regresses.

The README quotes numbers produced by ``benchmarks/compare_markitdown.py`` and
``tests/diagram_accuracy.py``. Those numbers were reproducible by hand, but
nothing stopped a change from quietly making them worse -- and a benchmark that
can only be run manually is a benchmark nobody runs before shipping.

So the current values are recorded in ``benchmarks/baseline.json`` and this
check compares against them. It fails on a regression, prints what improved, and
tells you to regenerate the baseline deliberately when a change is intentional.

Timing is excluded deliberately: latency depends on the machine, so a CI runner
would either fail constantly or force a threshold loose enough to be useless.
Accuracy numbers are machine-independent by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "benchmarks" / "baseline.json"

#: metric -> direction. True means higher is better.
DIRECTION = {
    "cer": False,
    "word_recall": True,
    "word_precision": True,
    "word_f1": True,
    "reading_order": True,
    "table_cell_recall": True,
    "table_cell_precision": True,
}

#: Accuracy is allowed to wobble a little between runs; structural metrics are
#: deterministic and must not move at all.
TOLERANCE = {
    "cer": 0.02,
    "word_recall": 0.01,
    "word_precision": 0.01,
    "word_f1": 0.01,
    "reading_order": 0.01,
    "table_cell_recall": 0.0,
    "table_cell_precision": 0.0,
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


DIAGRAM_METRICS = (
    "node_recall",
    "node_precision",
    "edge_recall",
    "edge_precision",
    "kind_accuracy",
    "direction_proof_rate",
)


def compare(baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    regressions: list[str] = []
    improvements: list[str] = []

    def note(key: str, before: float, after: float, higher_is_better: bool, slack: float) -> None:
        worse = after < before - slack if higher_is_better else after > before + slack
        better = after > before + slack if higher_is_better else after < before - slack
        if worse:
            regressions.append(f"{key}: {before} -> {after}")
        elif better:
            improvements.append(f"{key}: {before} -> {after}")

    old, new = baseline.get("document_accuracy", {}), current.get("document_accuracy", {})
    for metric, higher in DIRECTION.items():
        for system in ("docweave", "markitdown"):
            before = old.get(system, {}).get(metric)
            after = new.get(system, {}).get(metric)
            if before is None or after is None:
                continue
            note(
                f"document_accuracy.{system}.{metric}",
                before,
                after,
                higher,
                TOLERANCE.get(metric, 0.0),
            )

    old, new = baseline.get("diagram_accuracy", {}), current.get("diagram_accuracy", {})
    for metric in DIAGRAM_METRICS:
        before, after = old.get(metric), new.get(metric)
        if before is None or after is None:
            continue
        # CER-like lower-is-better does not appear here: every diagram metric
        # is a proportion, so higher is better for all of them.
        note(f"diagram_accuracy.{metric}", before, after, True, 0.005)

    return regressions, improvements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        help="Path to a markitdown comparison JSON; omit to check only the diagram report.",
    )
    parser.add_argument("--write", action="store_true", help="Regenerate the baseline.")
    parser.add_argument("--diagram", help="Path to a diagram accuracy JSON.")
    args = parser.parse_args()

    current: dict = {}

    if args.report:
        report = load(Path(args.report))
        accuracy = report.get("accuracy", {})
        if not accuracy.get("measured"):
            print("comparison report carries no accuracy section; refusing to baseline it")
            return 1
        current["document_accuracy"] = {
            "docweave": accuracy.get("docweave", {}),
            "markitdown": accuracy.get("markitdown", {}),
        }

    if args.diagram:
        current["diagram_accuracy"] = load(Path(args.diagram))

    if not current:
        print("nothing to check: pass --report and/or --diagram", file=sys.stderr)
        return 2

    if args.write:
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written with {len(current)} section(s)")
        return 0

    if not BASELINE.exists():
        print(f"no baseline at {BASELINE}; run with --write to create one")
        return 1

    regressions, improvements = compare(load(BASELINE), current)
    for item in improvements:
        print(f"improved  {item}")
    if improvements:
        print("\nIf these are intended, regenerate with --write.\n")

    if regressions:
        print(f"{len(regressions)} REGRESSION(S):")
        for item in regressions:
            print(f"  ! {item}")
        print(
            "\nA published number got worse. Fix it, or -- if the change is "
            "deliberate\nand the old number was wrong -- regenerate with --write "
            "and say why in\nthe commit message."
        )
        return 1

    print("no metric regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
