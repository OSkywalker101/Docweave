"""Semantic inference: geometry to meaning.

Two questions are answered here, and both are answered with the evidence that
produced them:

1. *What is this shape for?* A diamond is a decision, a cylinder is a data
   store. That mapping is a convention, not a certainty, so the convention and
   the measured geometry are both recorded on the node.
2. *What kind of diagram is this?* Flowchart, architecture, ER, sequence, UML
   class, or a plain box-and-arrow graph. When the evidence is thin the engine
   says ``unknown`` rather than picking the most likely category.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from docweave.diagram.graph import DiagramKind, NodeKind
from docweave.diagram.shapes import Shape, ShapeType

#: Drawn shape -> conventional flowchart/UML meaning.
SHAPE_TO_KIND: dict[ShapeType, NodeKind] = {
    ShapeType.RECTANGLE: NodeKind.PROCESS,
    ShapeType.ROUNDED_RECTANGLE: NodeKind.START,
    ShapeType.DIAMOND: NodeKind.DECISION,
    ShapeType.ELLIPSE: NodeKind.TERMINATOR,
    ShapeType.PARALLELOGRAM: NodeKind.INPUT_OUTPUT,
    ShapeType.CYLINDER: NodeKind.DATA_STORE,
    ShapeType.TRIANGLE: NodeKind.UNKNOWN,
    ShapeType.CONTAINER: NodeKind.GROUP,
    ShapeType.UNKNOWN: NodeKind.UNKNOWN,
}

#: ER and UML-class drawings use plain rectangles for entities/classes.
ER_LIKE_KINDS = (NodeKind.ENTITY,)

#: Shapes that carry flowchart meaning on their own, without a diamond. A
#: cylinder is deliberately absent: a datastore is just as likely to be a
#: component in a service topology as a step in a flow, and including it
#: reclassified an architecture diagram as a flowchart the moment the fixture
#: started drawing its database correctly.
FLOWCHART_SHAPES = frozenset({NodeKind.DECISION, NodeKind.INPUT_OUTPUT})


def kind_for_shape(shape: Shape, *, has_children: bool) -> tuple[NodeKind, dict]:
    """Map a primitive to a semantic kind, recording why."""
    evidence = {
        "shape_type": shape.type.value,
        "shape_confidence": round(shape.confidence, 3),
        "fill_ratio": shape.measures.get("fill_ratio"),
        "vertices": shape.measures.get("vertices"),
        "aspect": shape.measures.get("aspect"),
        "flat_top": shape.measures.get("flat_top"),
    }
    if has_children:
        return NodeKind.GROUP, evidence
    return SHAPE_TO_KIND.get(shape.type, NodeKind.UNKNOWN), evidence


def resolve_terminal_kinds(graph, *, layout: Layout) -> None:
    """Promote terminals to start/end using conventional placement.

    Flowcharts put the entry point at the top (or far left) and the exit at the
    bottom (or far right). That is a convention, so the decision is recorded on
    the node and only applies when a terminal has no other role.
    """
    terminals = [n for n in graph.nodes if n.kind is NodeKind.TERMINATOR]
    if len(terminals) < 2:
        return
    terminals.sort(key=lambda n: (layout.reading_rank(n.box.center)))
    first, last = terminals[0], terminals[-1]
    for node, kind in ((first, NodeKind.START), (last, NodeKind.END)):
        if node.kind is NodeKind.TERMINATOR:
            node.kind = kind
            node.evidence["terminal_role"] = "layout convention (reading order)"


@dataclass
class Layout:
    """Reading-order helper shared by inference steps."""

    width: int
    height: int

    def reading_rank(self, point: tuple[float, float]) -> tuple[float, float]:
        """Primary axis is vertical; near-equal rows are ordered left to right."""
        row = round(point[1] / max(self.height / 8.0, 1.0))
        return (row, point[0])

    def leftmost(self, nodes: Sequence) -> object:
        return min(nodes, key=lambda n: n.box.center[0])

    def rightmost(self, nodes: Sequence) -> object:
        return max(nodes, key=lambda n: n.box.center[0])


PROCESS_LIKE = {
    NodeKind.PROCESS,
    NodeKind.START,
    NodeKind.END,
    NodeKind.COMPONENT,
    NodeKind.UNKNOWN,
}


def infer_diagram_kind(
    graph,
    *,
    dashed_edges: int = 0,
    directed_edges: int = 0,
    lifelines: int = 0,
) -> tuple[DiagramKind, dict]:
    """Classify the diagram, with the evidence that led to it."""
    kinds = graph.kinds()
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    evidence = {
        "nodes": node_count,
        "edges": edge_count,
        "decision_nodes": sum(1 for n in graph.nodes if n.kind is NodeKind.DECISION),
        "terminator_nodes": sum(
            1
            for n in graph.nodes
            if n.kind in (NodeKind.TERMINATOR, NodeKind.START, NodeKind.END)
        ),
        "dashed_edges": dashed_edges,
        "directed_edges": directed_edges,
        "lifelines": lifelines,
    }

    # A box drawn with internal dividers -- name / attributes / operations --
    # is a UML class or a data record. This is checked before the ER branch
    # because a class diagram is otherwise indistinguishable from one: plain
    # boxes, undirected associations, no arrows. The compartments are what
    # break the tie, and they are measured, not guessed.
    compartmented = sum(1 for n in graph.nodes if getattr(n, "compartments", 1) > 1)
    evidence["compartmented_nodes"] = compartmented

    unattached = sum(1 for n in graph.nodes if not graph.outgoing(n.id) and not graph.incoming(n.id))
    degrees = [len(graph.outgoing(n.id)) + len(graph.incoming(n.id)) for n in graph.nodes]
    hub_degree = max(degrees) if degrees else 0
    evidence["unattached_nodes"] = unattached
    evidence["hub_degree"] = hub_degree

    if node_count == 0:
        return DiagramKind.UNKNOWN, evidence

    if lifelines >= 2 and not compartmented:
        return DiagramKind.SEQUENCE, evidence

    # Not enough connectivity to name a kind. A third of the content sitting
    # unattached means the relationships that would distinguish an ER diagram
    # from a generic one simply are not there. Naming one anyway would be a
    # guess presented as a reading, which is the one thing this engine must not
    # do -- so it says so instead.
    if unattached * 3 >= node_count:
        return DiagramKind.UNKNOWN, evidence

    # A parallelogram or a cylinder is a flowchart shape in its own right. A
    # small flowchart of "read input -> transform" has no diamond and no
    # terminator, so demanding both used to push it into box_and_arrow.
    if kinds & FLOWCHART_SHAPES and kinds & {
        NodeKind.TERMINATOR,
        NodeKind.START,
        NodeKind.END,
        NodeKind.PROCESS,
        NodeKind.INPUT_OUTPUT,
        NodeKind.DATA_STORE,
    }:
        return DiagramKind.FLOWCHART, evidence

    if kinds <= PROCESS_LIKE or kinds <= PROCESS_LIKE | {NodeKind.GROUP}:
        if compartmented >= 2 and compartmented * 2 >= node_count:
            # Compartmented boxes throughout: a class or record model. The
            # dividers are decisive on their own, and requiring undirected
            # associations as well made the rule hostage to arrowhead false
            # positives elsewhere in the pipeline.
            return DiagramKind.UML_CLASS, evidence
        if dashed_edges == 0 and directed_edges == 0:
            # Undirected, undashed associations between plain boxes is the
            # signature of an ER or class diagram.
            return DiagramKind.ER, evidence

    # Architecture means a *topology*: a hub fanning out to several peers, or an
    # explicit boundary. Arrows alone do not make a diagram architecture -- a
    # linear chain of boxes is just a flow, and calling it architecture would
    # claim a reading the geometry does not support.
    #
    # This deliberately sits outside the plain-box test above: a datastore is
    # as ordinary in a service topology as in a flowchart, so requiring every
    # node to be a plain box hid the hub that decides the kind.
    if hub_degree >= 3 or kinds & {NodeKind.GROUP}:
        return DiagramKind.ARCHITECTURE, evidence
    return DiagramKind.BOX_AND_ARROW, evidence


def lifelines_from_segments(segments, *, width: int, height: int, min_length_ratio: float = 0.22) -> int:
    """Count long vertical connectors, which are sequence-diagram lifelines."""
    threshold = max(height * min_length_ratio, 40.0)
    verticals = [s for s in segments if s.is_vertical and s.length >= threshold]
    # Group near-parallel lines that overlap vertically.
    groups: list[list] = []
    for segment in sorted(verticals, key=lambda s: s.start[0]):
        placed = False
        for group in groups:
            if abs(group[0].start[0] - segment.start[0]) < max(width * 0.02, 8):
                group.append(segment)
                placed = True
                break
        if not placed:
            groups.append([segment])
    return len(groups)


def direction_from_layout(
    source_box, target_box, *, tie_break_horizontal: bool = True
) -> str:
    """Fallback direction when no arrowhead resolved it.

    Flowcharts and dataflow read top-to-bottom then left-to-right. This is a
    convention, so callers must mark the resulting edge as inferred.
    """
    sx, sy = source_box
    tx, ty = target_box
    dy = ty - sy
    dx = tx - sx
    if tie_break_horizontal and abs(dy) < abs(dx) * 0.25:
        return "left_to_right" if dx >= 0 else "right_to_left"
    return "top_to_bottom" if dy >= 0 else "bottom_to_top"


def distance_point_to_shape(point: tuple[float, float], shape: Shape) -> float:
    """Shortest distance from a point to a shape's bounding box (0 if inside)."""
    x, y, w, h = shape.box
    dx = max(x - point[0], 0.0, point[0] - (x + w))
    dy = max(y - point[1], 0.0, point[1] - (y + h))
    return math.hypot(dx, dy)


def distance_point_to_boundary(point: tuple[float, float], shape: Shape) -> float:
    """Distance from a point to the nearest edge of a shape's bounding box.

    Zero for a point inside the box, which is the same as
    :func:`distance_point_to_shape`, so this only breaks ties between shapes
    that all contain the point.
    """
    x, y, w, h = shape.box
    return min(
        abs(point[0] - x),
        abs(point[0] - (x + w)),
        abs(point[1] - y),
        abs(point[1] - (y + h)),
    )


def snap_endpoint(
    point: tuple[float, float],
    shapes: Sequence[Shape],
    *,
    tolerance: float,
    exclude: str = "",
) -> tuple[Shape, float] | None:
    """Find the shape a connector endpoint belongs to.

    Prefers shapes the endpoint actually touches, then shapes within tolerance,
    and never returns a container: a connector terminates on a node, not on the
    frame drawn around it.

    Where one shape sits inside another, both report distance zero and the first
    one encountered used to win arbitrarily -- which turned a connector into a
    self-loop whenever the wrong shape happened to be drawn first. The tie is
    broken by distance to the boundary, then by area: a connector endpoint lies
    on the edge of the shape it terminates on.
    """
    best: tuple[Shape, float] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for shape in shapes:
        if shape.type in (ShapeType.CONTAINER, ShapeType.ARROWHEAD, ShapeType.LINE):
            continue
        if shape.id == exclude:
            continue
        distance = distance_point_to_shape(point, shape)
        if distance > tolerance:
            continue
        x, y, w, h = shape.box
        key = (distance, distance_point_to_boundary(point, shape), float(w * h), shape.id)
        if best_key is None or key < best_key:
            best, best_key = (shape, distance), key
    return best
