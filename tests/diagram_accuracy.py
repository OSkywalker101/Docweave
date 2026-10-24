"""Reconstruction accuracy against the synthetic corpus.

Every diagram in the corpus ships the graph that was drawn, so accuracy is a
measurement rather than an opinion. This module is used both by the test suite
and by ``python tests/diagram_accuracy.py`` for a human-readable report.

It measures what the engine claims, and reports what it misses. It does not
round in our favour: precision is reported separately from recall so a detector
that invents edges cannot look good.
"""

from __future__ import annotations

import json
from pathlib import Path


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union else 0.0


def _match_nodes(spec, graph) -> set[int]:
    """Indices of detected nodes that correspond to a drawn node."""
    boxes = [(n.box.x, n.box.y, n.box.width, n.box.height) for n in graph.nodes]
    matched: set[int] = set()
    for node in spec.nodes:
        best, best_iou, best_index = 0.5, -1.0, -1
        for index, box in enumerate(boxes):
            score = _iou(node.box, box)
            if score > best_iou:
                best, best_iou, best_index = best, score, index
        if best_index >= 0 and best_iou >= best:
            matched.add(best_index)
    return matched


def _match_edges(spec, graph) -> set[int]:
    """Indices of detected edges that correspond to a drawn connector.

    Matched by endpoint geometry, not by name: labels are unread without an OCR
    engine, and matching on counts alone cannot tell a true edge from ink that
    was promoted into one -- which is precisely the failure being measured.
    """
    by_id = {
        n.id: (n.box.x, n.box.y, n.box.width, n.box.height) for n in graph.nodes
    }
    detected = []
    for edge in graph.edges:
        source, target = by_id.get(edge.source), by_id.get(edge.target)
        if source and target:
            detected.append((source, target))

    matched: set[int] = set()
    for edge in spec.edges:
        a, b = spec.node(edge.source).box, spec.node(edge.target).box
        best, best_score, best_index = 0.5, -1.0, -1
        for index, (source, target) in enumerate(detected):
            forward = min(_iou(a, source), _iou(b, target))
            reverse = min(_iou(a, target), _iou(b, source))
            score = max(forward, reverse)
            if score > best_score:
                best, best_score, best_index = best, score, index
        if best_index >= 0 and best_score >= best:
            matched.add(best_index)
    return matched


def score_one(spec, graph) -> dict:
    """Compare a reconstructed graph against the graph that was drawn."""
    expected_nodes = len(spec.nodes)
    expected_edges = len(spec.edges)
    got_nodes = len(graph.nodes)
    got_edges = len(graph.edges)

    node_hits = len(_match_nodes(spec, graph))
    edge_hits = len(_match_edges(spec, graph))

    def ratio(hits: int, total: int) -> float:
        return hits / total if total else 1.0

    # Recall: of what was drawn, how much was recovered.
    # Precision: of what was published, how much was really there. Overshooting
    # must cost precision, which is why it is measured against detections and
    # not against the expected count.
    node_recall = ratio(node_hits, expected_nodes)
    edge_recall = ratio(edge_hits, expected_edges)
    node_precision = ratio(node_hits, got_nodes)
    edge_precision = ratio(edge_hits, got_edges)

    directed = sum(1 for edge in graph.edges if edge.directed)
    inferred = sum(1 for edge in graph.edges if edge.direction_inferred)
    # "Proven" means the direction came from an arrowhead, not from layout.
    proven = got_edges - inferred

    return {
        "name": spec.name,
        "expected_kind": spec.kind,
        "detected_kind": graph.kind.value,
        "kind_correct": graph.kind.value == spec.kind,
        "nodes": {"expected": expected_nodes, "detected": got_nodes, "matched": node_hits},
        "edges": {"expected": expected_edges, "detected": got_edges, "matched": edge_hits},
        "node_recall": round(node_recall, 4),
        "edge_recall": round(edge_recall, 4),
        "node_precision": round(node_precision, 4),
        "edge_precision": round(edge_precision, 4),
        "directed_edges": directed,
        "direction_inferred": inferred,
        "direction_proof_rate": round(proven / got_edges, 4) if got_edges else 1.0,
        "unassigned_shapes": len(graph.unassigned_shapes),
        "degraded": graph.degraded,
    }


def score_corpus(corpus_dir: Path, specs: dict) -> dict:
    """Score every diagram in a corpus directory."""
    from docweave.diagram import analyze_file

    per_diagram: list[dict] = []
    for name in sorted(specs):
        path = Path(corpus_dir) / f"{name}.png"
        if not path.exists():
            continue
        graph = analyze_file(path).graph
        per_diagram.append(score_one(specs[name], graph))

    if not per_diagram:
        return {"diagrams": 0, "per_diagram": []}

    def mean(key: str) -> float:
        return round(sum(entry[key] for entry in per_diagram) / len(per_diagram), 4)

    return {
        "diagrams": len(per_diagram),
        "node_recall": mean("node_recall"),
        "edge_recall": mean("edge_recall"),
        "node_precision": mean("node_precision"),
        "edge_precision": mean("edge_precision"),
        "direction_proof_rate": mean("direction_proof_rate"),
        "kind_accuracy": round(
            sum(1 for entry in per_diagram if entry["kind_correct"]) / len(per_diagram), 4
        ),
        "per_diagram": per_diagram,
    }


def format_report(report: dict) -> str:
    lines = [
        f"diagrams: {report['diagrams']}",
        f"node recall     : {report['node_recall']:.3f}",
        f"node precision  : {report['node_precision']:.3f}",
        f"edge recall     : {report['edge_recall']:.3f}",
        f"edge precision  : {report['edge_precision']:.3f}",
        f"kind accuracy   : {report['kind_accuracy']:.3f}",
        f"direction proven: {report['direction_proof_rate']:.3f}",
        "",
        f"{'diagram':<16}{'kind':<14}{'nodes':<12}{'edges':<12}{'n-rec':<8}{'e-rec':<8}degraded",
    ]
    for entry in report["per_diagram"]:
        lines.append(
            f"{entry['name']:<16}"
            f"{entry['detected_kind']:<14}"
            f"{str(entry['nodes']['expected']) + '/' + str(entry['nodes']['detected']):<12}"
            f"{str(entry['edges']['expected']) + '/' + str(entry['edges']['detected']):<12}"
            f"{entry['node_recall']:<8.3f}{entry['edge_recall']:<8.3f}"
            f"{entry['degraded']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual measurement
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.diagram_corpus import build_all, specs_by_name

    directory = Path("tests/fixtures/diagrams")
    build_all(directory)
    report = score_corpus(directory, specs_by_name())

    # Machine-readable output for CI. tools/check_metrics.py compares it
    # against benchmarks/baseline.json so a published number cannot silently
    # regress between releases.
    json_path = sys.argv[1] if len(sys.argv) > 1 else None
    if json_path:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {target}")

    print(format_report(report))
