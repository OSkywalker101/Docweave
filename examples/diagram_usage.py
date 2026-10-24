"""Diagram understanding: from pixels to a graph.

Run: python examples/diagram_usage.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docweave.diagram import analyze_file, render
from docweave.diagram.text import NullLabelReader


def _fixture(name: str) -> Path:
    path = Path("tests/fixtures/diagrams") / f"{name}.png"
    if not path.exists():
        print("run `python tests/diagram_corpus.py` first")
        raise SystemExit(1)
    return path


def example_structure_not_text() -> None:
    """The point: we reconstruct the graph, not the words on it."""
    result = analyze_file(_fixture("flowchart"))
    graph = result.graph

    print(f"kind: {graph.kind.value}")
    for node in graph.nodes:
        label = node.label.text or "(label not read)"
        print(f"  {node.id}  {node.kind.value:<14} shape={node.shape_type:<12} {label}")
    print()
    for edge in graph.edges:
        arrow = edge.arrowhead or "inferred"
        print(f"  {edge.source} -> {edge.target}   ({arrow})")


def example_directions_are_proven_or_admitted() -> None:
    """An arrowhead settles a direction. Without one, we say we guessed."""
    result = analyze_file(_fixture("flowchart"))
    proven = [e for e in result.graph.edges if not e.direction_inferred]
    inferred = [e for e in result.graph.edges if e.direction_inferred]
    print(f"direction proven by arrowhead : {len(proven)}")
    print(f"direction inferred from layout: {len(inferred)}")
    for edge in inferred:
        print(f"  {edge.id}: {edge.source} -> {edge.target} ({edge.evidence['direction']})")


def example_render_targets() -> None:
    graph = analyze_file(_fixture("er")).graph
    print(render(graph, "mermaid"))
    print(render(graph, "plantuml"))


def example_lossless_output() -> None:
    """JSON keeps everything, including what could not be reconstructed."""
    graph = analyze_file(_fixture("ambiguous"), reader=NullLabelReader()).graph
    payload = json.loads(render(graph, "json"))
    print(f"kind      : {payload['kind']}  (degraded={payload['degraded']})")
    print(f"nodes     : {len(payload['nodes'])}")
    print(f"edges     : {len(payload['edges'])}")
    print(f"unassigned: {len(payload['unassigned_shapes'])} shapes")


def example_diagnostics() -> None:
    result = analyze_file(_fixture("flowchart"), reader=NullLabelReader())
    for diagnostic in result.diagnostics:
        print(diagnostic.to_line())
        if diagnostic.reason:
            print(f"    reason: {diagnostic.reason}")


if __name__ == "__main__":
    example_structure_not_text()
    print("\n" + "=" * 70)
    example_directions_are_proven_or_admitted()
    print("\n" + "=" * 70)
    example_render_targets()
    print("\n" + "=" * 70)
    example_lossless_output()
    print("\n" + "=" * 70)
    example_diagnostics()
