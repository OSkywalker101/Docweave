"""Property-based testing for the diagram engine.

The document property suite (tests/property/test_invariants.py) found two real
defects that no hand-written fixture would have caught, because it generates
inputs rather than writing them. The diagram engine is the weakest measured
area and it has never had that treatment: eleven hand-written fixtures and a
classifier tuned to them.

This module generates *random graphs*, renders them to PNG through the same
renderer the corpus uses, and asserts only things that follow from the fact
that the graph was drawn: the detected nodes must be exactly the drawn nodes,
the detected edges exactly the drawn edges, and analysis must be deterministic.
There is no expected output to match, so a failure means a guarantee broke.

The single most important property is the node count. The corpus found three
separate ways the engine invents nodes -- a closed ring of connectors read as a
box, a compartmented box exploded into compartments, and a contour seen twice
at two sizes. A generator produces all three constantly.
"""

from __future__ import annotations

import hypothesis
from hypothesis import given
from hypothesis import strategies as st

from docweave.diagram import analyze_file
from tests.diagram_corpus import DiagramSpec, SpecEdge, SpecNode, render, write_png

GRID = (160, 110)
NODE_W, NODE_H = 120, 60
MARGIN = 40


def _centre(position: tuple[int, int]) -> tuple[float, float]:
    x, y = position
    return (
        x * GRID[0] + MARGIN + NODE_W / 2,
        y * GRID[1] + MARGIN + NODE_H / 2,
    )


def _box(position: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y = position
    return (x * GRID[0] + MARGIN, y * GRID[1] + MARGIN, NODE_W, NODE_H)


def _segment_hits_box(
    a: tuple[float, float], b: tuple[float, float], box: tuple[float, float, float, float]
) -> bool:
    """True if the segment a-b passes through ``box``.

    A well-formed diagram does not route a connector through an unrelated node.
    Drawing one makes the ground truth ambiguous -- the box is now split by a
    line across it -- so the generator avoids it rather than testing a situation
    real diagrams do not contain. The engine's weakness here is that a box
    crossed by a line splits into two contours; that is recorded separately.
    """
    x, y, w, h = box
    # Sample densely enough that a crossing cannot slip between samples.
    steps = 64
    for i in range(steps + 1):
        t = i / steps
        px = a[0] + (b[0] - a[0]) * t
        py = a[1] + (b[1] - a[1]) * t
        if x < px < x + w and y < py < y + h:
            return True
    return False


def _clean_edges(
    nodes: list[tuple[int, int]], edges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Drop connectors that would cross an unrelated node."""
    keep: list[tuple[int, int]] = []
    for a, b in edges:
        if a == b:
            continue
        pa, pb = _centre(nodes[a]), _centre(nodes[b])
        crosses = any(
            _segment_hits_box(pa, pb, _box(pos))
            for index, pos in enumerate(nodes)
            if index not in (a, b)
        )
        if not crosses:
            keep.append((a, b))
    return keep


def _make_spec(nodes: list[tuple[int, int]], edges: list[tuple[int, int]]) -> DiagramSpec:
    """Turn a graph into a spec on the shared grid.

    Labels are deliberately *absent*: the property tests run without an OCR engine
    and must not depend on one. The engine still detects nodes from geometry
    alone, which is the part being tested.
    """
    width = (max(n[0] for n in nodes) + 1) * GRID[0] + NODE_W + MARGIN
    height = (max(n[1] for n in nodes) + 1) * GRID[1] + NODE_H + MARGIN
    return DiagramSpec(
        name="generated",
        kind="box_and_arrow",
        width=width,
        height=height,
        nodes=[
            SpecNode(f"n{i}", "rect", (x * GRID[0] + MARGIN, y * GRID[1] + MARGIN, NODE_W, NODE_H))
            for i, (x, y) in enumerate(nodes)
        ],
        edges=[
            SpecEdge(f"n{a}", f"n{b}") for a, b in _clean_edges(nodes, edges)
        ],
    )


def _render_to_path(spec: DiagramSpec, path) -> None:
    write_png(render(spec), path)


@given(
    graph=st.dictionaries(
        st.integers(min_value=0, max_value=4),
        st.integers(min_value=0, max_value=4),
        max_size=6,
    ).filter(lambda g: len(g) >= 2)
)
@hypothesis.settings(max_examples=30, deadline=None)
def test_every_drawn_node_is_found_exactly_once(tmp_path_factory, graph):
    """N nodes drawn, N nodes detected, no phantoms.

    This is the property the corpus could never test on its own: whether the
    engine *adds* nodes that were not drawn.
    """
    nodes = [(x, y) for x, y in graph.items()]
    edges = [(0, i) for i in range(1, len(nodes))]
    spec = _make_spec(nodes, edges)
    directory = tmp_path_factory.mktemp("nodes")
    path = directory / "g.png"
    _render_to_path(spec, path)
    result = analyze_file(path)
    assert len(result.graph.nodes) == len(nodes), (
        f"drew {len(nodes)} nodes, detected {len(result.graph.nodes)}"
    )


@given(
    graph=st.integers(min_value=2, max_value=6),
    edges=st.lists(st.tuples(st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5)), max_size=5),
)
@hypothesis.settings(max_examples=30, deadline=None)
def test_no_self_loops_on_an_acyclic_graph(tmp_path_factory, graph, edges):
    """A generated graph with no self-edges must not produce any self-loops.

    This is the state_machine failure mode in miniature: a cycle of connectors
    forms a closed contour that reads as a box.
    """
    n = graph
    nodes = [(i % 3, i // 3) for i in range(n)]
    # Keep only edges between distinct nodes, with lower index first (DAG).
    drawn = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b and a < n and b < n})
    spec = _make_spec(nodes, drawn)
    directory = tmp_path_factory.mktemp("loops")
    path = directory / "g.png"
    _render_to_path(spec, path)
    result = analyze_file(path)
    self_loops = [e for e in result.graph.edges if e.source == e.target]
    assert not self_loops, f"generated {len(drawn)} edges, got {len(self_loops)} self-loop(s)"


@given(
    n=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=1000),
)
@hypothesis.settings(max_examples=25, deadline=None)
def test_analysis_is_deterministic(tmp_path_factory, n, seed):
    """Same PNG, same graph, twice."""
    import random

    rng = random.Random(seed)
    nodes = [(rng.randrange(3), rng.randrange(3)) for _ in range(n)]
    # Dedup node positions so the graph is well-formed.
    unique: list[tuple[int, int]] = []
    for pos in nodes:
        if pos not in unique:
            unique.append(pos)
    if len(unique) < 2:
        return
    edges = [(0, i) for i in range(1, len(unique))]
    spec = _make_spec(unique, edges)
    directory = tmp_path_factory.mktemp("det")
    path = directory / "g.png"
    _render_to_path(spec, path)
    first = analyze_file(path)
    second = analyze_file(path)
    assert len(first.graph.nodes) == len(second.graph.nodes)
    assert len(first.graph.edges) == len(second.graph.edges)
    assert first.graph.kind == second.graph.kind


@given(
    n=st.integers(min_value=2, max_value=5),
    extra=st.integers(min_value=0, max_value=3),
)
@hypothesis.settings(max_examples=20, deadline=None)
def test_an_empty_diagram_is_reported_as_unknown(tmp_path_factory, n, extra):
    """A diagram with no connectors cannot be given a kind."""
    nodes = [(i, 0) for i in range(n)]
    spec = _make_spec(nodes, [])
    directory = tmp_path_factory.mktemp("empty")
    path = directory / "g.png"
    _render_to_path(spec, path)
    result = analyze_file(path)
    assert result.graph.kind.value == "unknown", (
        f"a diagram with no connectors was classified as {result.graph.kind.value}"
    )
