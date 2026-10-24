"""Diagram analysis pipeline.

Order matters and is deliberate:

1. shapes   — what is drawn
2. text     — where the words are (always; reading them is optional)
3. text is erased, connectors are found
4. arrowheads are found by ink thickness
5. edges are built only from segments that reach two different shapes
6. labels are associated with shapes, then read if an engine is available
7. semantics are inferred, and anything unresolved is reported

Step 5 is what keeps precision honest. A glyph stroke or a stray mark can
produce a line, but it cannot terminate on two separate shapes, so it never
becomes an edge. The engine does not need to guess: it only publishes
relationships it can point at.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime only
    from docweave.visual.classify import VisualClassification

from docweave.config import ConversionConfig
from docweave.diagram.graph import (
    BoundingBox,
    DiagramEdge,
    DiagramGraph,
    DiagramNode,
    EdgeStyle,
    Label,
    NodeKind,
)
from docweave.diagram.infer import (
    Layout,
    direction_from_layout,
    infer_diagram_kind,
    kind_for_shape,
    lifelines_from_segments,
    resolve_terminal_kinds,
    snap_endpoint,
)
from docweave.diagram.shapes import (
    Segment,
    Shape,
    ShapeType,
    detect_arrowheads,
    detect_segments,
    detect_shapes,
    image_size,
    point_segment_distance,
)
from docweave.diagram.text import (
    NullLabelReader,
    TextRegion,
    clean_label,
    default_label_reader,
    detect_text_regions,
    group_regions,
)
from docweave.models import Diagnostic
from docweave.models.diagnostics import Category, Code, Severity

#: How far from a connector endpoint an arrowhead may sit and still count.
#: Hough frequently stops a line short of the head it is drawn over.
ARROWHEAD_ENDPOINT_TOLERANCE = 22.0


@dataclass
class DiagramResult:
    """A reconstructed diagram plus everything that could not be trusted."""

    graph: DiagramGraph
    diagnostics: list[Diagnostic] = field(default_factory=list)
    elapsed_ms: float = 0.0
    shapes: list[Shape] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    arrowheads: list[Shape] = field(default_factory=list)
    text_regions: list[TextRegion] = field(default_factory=list)
    #: What the visual classifier said, when it ran. ``None`` means the gate was
    #: not consulted -- not that the image was confirmed to be a diagram.
    classification: VisualClassification | None = None

    @property
    def ok(self) -> bool:
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)


class DiagramAnalyzer:
    """Reconstructs a typed graph from a diagram image.

    Deterministic: the same image always produces the same graph, with the same
    node ids, in the same order.
    """

    def __init__(self, config: ConversionConfig | None = None) -> None:
        self.config = config or ConversionConfig()

    def analyze(
        self,
        image: np.ndarray,
        *,
        source_image: str | None = None,
        reader=None,
    ) -> DiagramResult:
        started = time.perf_counter()
        diagnostics: list[Diagnostic] = []

        width, height = image_size(image)
        graph = DiagramGraph(width=width, height=height, source_image=source_image)

        shapes = detect_shapes(image)
        raw_text = detect_text_regions(image)
        text_regions = group_regions(raw_text)
        text_boxes = [region.box for region in raw_text]

        segments = detect_segments(image, shapes, text_boxes)
        snap = max(18.0, min(width, height) * 0.06)

        # Restrict attention to lines that actually join two shapes. Text and
        # stray marks produce segments too, and hunting for arrowheads along
        # them would attribute glyph blobs to connectors.
        connectors = [s for s in segments if _connects_two_shapes(s, shapes, snap)]
        arrowheads = detect_arrowheads(image, shapes, connectors, text_boxes)

        nodes, shape_index_to_node = _build_nodes(shapes)

        # A box drawn with internal dividers -- a UML class, a packet, a message
        # group -- is detected as one contour per compartment. Those are not
        # separate things: the dividers are structure inside a single node, and
        # promoting them invents both nodes and relationships between them.
        merged, compartments, remap = _merge_compartments(nodes)
        graph.nodes = merged
        for index, node_id in list(shape_index_to_node.items()):
            shape_index_to_node[index] = remap.get(node_id, node_id)

        graph.edges = _build_edges(
            shapes,
            segments,
            arrowheads,
            shape_index_to_node,
            snap,
            diagnostics,
            compartments=compartments,
        )

        _associate_text(text_regions, shapes, segments)
        _read_labels(image, text_regions, graph, reader, diagnostics)
        _attach_edge_labels(image, text_regions, graph, segments, reader, diagnostics)
        _attach_unassigned(graph, shapes, text_regions, shape_index_to_node)

        lifelines = lifelines_from_segments(segments, width=width, height=height)
        kind, evidence = infer_diagram_kind(
            graph,
            directed_edges=sum(1 for edge in graph.edges if edge.directed),
            dashed_edges=sum(1 for edge in graph.edges if edge.style is EdgeStyle.DASHED),
            lifelines=lifelines,
        )
        graph.kind = kind
        graph.evidence = evidence
        resolve_terminal_kinds(graph, layout=Layout(width=width, height=height))

        _quality_diagnostics(graph, diagnostics)
        graph.degraded = any(
            diagnostic.severity in (Severity.WARNING, Severity.ERROR)
            for diagnostic in diagnostics
        )

        return DiagramResult(
            graph=graph,
            diagnostics=diagnostics,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            shapes=shapes,
            segments=segments,
            arrowheads=arrowheads,
            text_regions=text_regions,
        )


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


def _connects_two_shapes(segment: Segment, shapes: Sequence[Shape], snap: float) -> bool:
    """True when both endpoints of a line land on distinct shapes."""
    start = snap_endpoint(segment.start, shapes, tolerance=snap)
    end = snap_endpoint(segment.end, shapes, tolerance=snap)
    return start is not None and end is not None and start[0].id != end[0].id


def _build_nodes(shapes: Sequence[Shape]) -> tuple[list[DiagramNode], dict[int, str]]:
    nodes: list[DiagramNode] = []
    shape_index_to_node: dict[int, str] = {}

    for index, shape in enumerate(shapes):
        x, y, w, h = shape.box
        kind, evidence = kind_for_shape(shape, has_children=shape.type is ShapeType.CONTAINER)
        node = DiagramNode(
            id=f"n{len(nodes)}",
            kind=kind,
            box=BoundingBox(x=x, y=y, width=w, height=h),
            shape_type=shape.type.value,
            shape_confidence=shape.confidence,
            evidence=evidence,
        )
        nodes.append(node)
        shape_index_to_node[index] = node.id

    groups = [node for node in nodes if node.kind is NodeKind.GROUP]
    if groups:
        for node in nodes:
            if node.kind is NodeKind.GROUP:
                continue
            parent = _smallest_containing(node, groups)
            if parent is not None:
                parent.contains.append(node.id)

    return nodes, shape_index_to_node


def _smallest_containing(node: DiagramNode, groups: Sequence[DiagramNode]) -> DiagramNode | None:
    best: DiagramNode | None = None
    best_area = float("inf")
    for group in groups:
        box = group.box
        if (
            box.x <= node.box.x
            and box.y <= node.box.y
            and box.x + box.width >= node.box.x + node.box.width
            and box.y + box.height >= node.box.y + node.box.height
        ):
            area = box.width * box.height
            if area < best_area:
                best, best_area = group, area
    return best


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------


def _merge_compartments(
    nodes: list[DiagramNode], *, x_tolerance: float = 3.0, y_tolerance: float = 4.0
) -> tuple[list[DiagramNode], set[tuple[str, str]], dict[str, str]]:
    """Fold vertically stacked, edge-aligned compartments into one node.

    Detected contours that share a left and right edge and touch vertically are
    compartments of a single box. Real diagrams do stack things, but they do
    not usually stack them at exactly the same width with no gap -- which is
    precisely what a divider looks like.

    Returns the surviving nodes, the pairs that were merged, and a remap from
    every absorbed id to its survivor. Without the remap, edges keep pointing at
    nodes that no longer exist.
    """
    ordered = sorted(nodes, key=lambda n: (n.box.y, n.box.x))
    consumed: set[str] = set()
    internal: set[tuple[str, str]] = set()
    survivors: list[DiagramNode] = []
    remap: dict[str, str] = {}

    for index, upper in enumerate(ordered):
        if upper.id in consumed:
            continue
        members = [upper]
        cursor = upper
        for lower in ordered[index + 1 :]:
            if lower.id in consumed:
                continue
            if abs(lower.box.x - cursor.box.x) > x_tolerance:
                continue
            if (
                abs(
                    (cursor.box.x + cursor.box.width)
                    - (lower.box.x + lower.box.width)
                )
                > x_tolerance
            ):
                continue
            # Touching, or overlapping by a hair: a divider leaves no gap.
            gap = lower.box.y - (cursor.box.y + cursor.box.height)
            if not (-y_tolerance <= gap <= y_tolerance):
                continue
            members.append(lower)
            cursor = lower

        if len(members) < 2:
            survivors.append(upper)
            continue

        for member in members:
            consumed.add(member.id)
            internal.add(tuple(sorted((upper.id, member.id))))
        for member in members:
            remap[member.id] = upper.id
        top = min(m.box.y for m in members)
        bottom = max(m.box.y + m.box.height for m in members)
        left = min(m.box.x for m in members)
        right = max(m.box.x + m.box.width for m in members)
        upper.box = BoundingBox(x=left, y=top, width=right - left, height=bottom - top)
        upper.compartments = len(members)
        survivors.append(upper)

    return survivors, internal, remap


def _extend_endpoints(
    segment: Segment, *, reach: float = 0.4
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Push a connector's endpoints outward along its own direction.

    Bounded by a fraction of the segment's length, so this closes a dash gap or
    a slightly ragged stop without dragging a short decorative line across the
    diagram to whatever it happens to be nearest to.
    """
    length = segment.length
    if length < 1.0:
        return segment.start, segment.end
    step = length * reach
    ux = (segment.end[0] - segment.start[0]) / length
    uy = (segment.end[1] - segment.start[1]) / length
    return (
        (segment.start[0] - ux * step, segment.start[1] - uy * step),
        (segment.end[0] + ux * step, segment.end[1] + uy * step),
    )


def _build_edges(
    shapes: Sequence[Shape],
    segments: Sequence[Segment],
    arrowheads: Sequence[Shape],
    shape_index_to_node: dict[int, str],
    snap: float,
    diagnostics: list[Diagnostic],
    compartments: set[tuple[str, str]] = frozenset(),
) -> list[DiagramEdge]:
    """Group segments by the pair of shapes they join, then emit one edge each.

    A connector is often drawn as several strokes, and the arrowhead frequently
    sits on a fragment other than the one Hough returned first. Grouping before
    deciding direction means the arrowhead is found wherever it is.
    """
    index_of = {shape.id: index for index, shape in enumerate(shapes)}
    groups: dict[tuple[str, str], list[tuple[Segment, Shape, Shape]]] = {}
    dangling = 0
    self_loops = 0

    for segment in segments:
        start_hit = snap_endpoint(segment.start, shapes, tolerance=snap)
        end_hit = snap_endpoint(segment.end, shapes, tolerance=snap)
        if start_hit is None or end_hit is None:
            # A dashed connector's ink legitimately stops short: the final gap in
            # the dash pattern can leave the drawn line well clear of the shape
            # it serves. Re-snap the endpoints extended along the connector
            # rather than discarding a relationship that is plainly drawn.
            extended = _extend_endpoints(segment)
            start_hit = snap_endpoint(extended[0], shapes, tolerance=snap)
            end_hit = snap_endpoint(extended[1], shapes, tolerance=snap)
        if start_hit is None or end_hit is None:
            dangling += 1
            continue
        a, b = start_hit[0], end_hit[0]
        if a.id == b.id:
            self_loops += 1
            continue
        key = tuple(sorted((a.id, b.id)))
        groups.setdefault(key, []).append((segment, a, b))

    edges: list[DiagramEdge] = []
    for key in sorted(groups):
        members = groups[key]
        source, target, arrow = _resolve_direction(members, arrowheads)
        source_node = shape_index_to_node[index_of[source.id]]
        target_node = shape_index_to_node[index_of[target.id]]
        if (source_node, target_node) in compartments or (
            target_node,
            source_node,
        ) in compartments:
            # A divider between two compartments of one box. Real structure, but
            # internal to the node, not a relationship between two things.
            continue
        edges.append(
            DiagramEdge(
                id=f"e{len(edges)}",
                source=source_node,
                target=target_node,
                style=(
                    EdgeStyle.DASHED
                    if any(s.dashed for s, _, _ in members)
                    else (EdgeStyle.SOLID if arrow else EdgeStyle.ASSOCIATION)
                ),
                directed=arrow is not None,
                direction_inferred=arrow is None,
                arrowhead=arrow,
                waypoints=_waypoints(members),
                evidence={
                    "direction": "arrowhead" if arrow else "layout convention",
                    "fragments": len(members),
                    "length": round(sum(s.length for s, _, _ in members), 2),
                },
            )
        )

    if dangling:
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_UNCONNECTED,
                category=Category.UNVERIFIED,
                severity=Severity.WARNING if dangling > len(edges) else Severity.INFO,
                message=f"{dangling} line(s) did not connect two shapes and were not turned into edges.",
                reason="A relationship is published only when both endpoints land on a shape.",
                evidence={"unconnected_segments": dangling, "edges": len(edges)},
            )
        )
    if self_loops:
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_UNCONNECTED,
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                message=f"{self_loops} connector(s) return to the shape they left.",
                reason="A self-loop and a decorative line are not distinguishable here.",
                evidence={"self_loops": self_loops},
            )
        )
    return edges


def _waypoints(members: Sequence[tuple[Segment, Shape, Shape]]) -> list[tuple[float, float]]:
    """Bounding endpoints of all the strokes that make up one connector."""
    xs = [point for segment, _, _ in members for point in (segment.start[0], segment.end[0])]
    ys = [point for segment, _, _ in members for point in (segment.start[1], segment.end[1])]
    if not xs:
        return []
    return [(round(min(xs), 2), round(min(ys), 2)), (round(max(xs), 2), round(max(ys), 2))]


def _resolve_direction(
    members: Sequence[tuple[Segment, Shape, Shape]],
    arrowheads: Sequence[Shape],
) -> tuple[Shape, Shape, str | None]:
    """Decide which way an edge points.

    An arrowhead settles it. With no arrowhead we fall back to reading order and
    the edge is marked direction-inferred, so nobody mistakes a convention for
    evidence.
    """
    sides: list[str] = []
    for segment, _source, _target in members:
        tolerance = max(ARROWHEAD_ENDPOINT_TOLERANCE, segment.length * 0.12)
        if _arrowhead_near(arrowheads, segment.start, tolerance):
            sides.append("start")
        if _arrowhead_near(arrowheads, segment.end, tolerance):
            sides.append("end")

    segment, a, b = members[0]
    if "start" in sides and "end" in sides:
        return a, b, "both"
    if "end" in sides:
        return a, b, "head"
    if "start" in sides:
        return b, a, "tail"

    center_a = (a.box[0] + a.box[2] / 2, a.box[1] + a.box[3] / 2)
    center_b = (b.box[0] + b.box[2] / 2, b.box[1] + b.box[3] / 2)
    flow = direction_from_layout(center_a, center_b)
    if flow in ("top_to_bottom", "left_to_right"):
        return a, b, None
    return b, a, None


def _arrowhead_near(
    arrowheads: Sequence[Shape], point: tuple[float, float], tolerance: float
) -> Shape | None:
    for head in arrowheads:
        center = head.measures.get("center") or head.center
        if ((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) ** 0.5 <= tolerance:
            return head
    return None


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def _associate_text(
    regions: Sequence[TextRegion],
    shapes: Sequence[Shape],
    segments: Sequence[Segment],
) -> None:
    """Attach each region to the shape containing it, or to a connector."""
    for region in regions:
        center = region.center
        best: Shape | None = None
        for shape in shapes:
            if shape.type in (ShapeType.CONTAINER, ShapeType.ARROWHEAD):
                continue
            if not shape.contains_point(*center, margin=-1):
                continue
            if best is None or shape.area < best.area:
                best = shape
        if best is not None:
            region.placement = "shape"
            continue

        nearest: Segment | None = None
        nearest_distance = 14.0
        for segment in segments:
            distance = point_segment_distance(center, segment.start, segment.end)
            if distance < nearest_distance:
                nearest, nearest_distance = segment, distance
        region.placement = "edge" if nearest is not None else "free"


def _read_labels(
    image: np.ndarray,
    regions: Sequence[TextRegion],
    graph: DiagramGraph,
    reader,
    diagnostics: list[Diagnostic],
) -> None:
    """Run the label reader over each region and store what it found.

    The image is passed explicitly rather than held in module state: a
    package-level mutable would make concurrent analysis unsafe.
    """
    if reader is None:
        reader = default_label_reader()
    try:
        available = bool(reader.available())
    except Exception:
        available = False

    readable: list[DiagramNode] = []
    unreadable = 0
    for region in regions:
        if region.placement != "shape":
            continue
        node = _node_at(graph, region)
        if node is None:
            continue
        x, y, w, h = region.box
        text: str | None = None
        score = 0.0
        if available:
            try:
                text, score = reader.read(image, region.box)
            except Exception:
                text, score = None, 0.0
        cleaned = clean_label(text)
        if cleaned:
            node.label = Label(
                text=cleaned,
                region=BoundingBox(x=x, y=y, width=w, height=h),
                reader=reader.name,
                confidence=score,
            )
            node.evidence["label_source"] = reader.name
            readable.append(node)
        else:
            unreadable += 1
            node.label = Label(
                text=None,
                region=BoundingBox(x=x, y=y, width=w, height=h),
                unreadable=True,
            )

    if unreadable:
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_LABEL_UNREAD,
                category=Category.UNVERIFIED,
                severity=Severity.INFO if available else Severity.WARNING,
                message=(
                    f"{unreadable} label region(s) were located but not read."
                    if available
                    else f"{unreadable} label region(s) were located; no OCR engine is available to read them."
                ),
                reason="Structure does not depend on reading labels, so the graph is still usable.",
                hint=None if available else "Install 'docweave[ocr]' and Tesseract, then re-run.",
                evidence={"unreadable_regions": unreadable, "reader": reader.name},
            )
        )
    del readable


def _node_at(graph: DiagramGraph, region: TextRegion) -> DiagramNode | None:
    center = region.center
    for node in graph.nodes:
        box = node.box
        if box.x <= center[0] <= box.x + box.width and box.y <= center[1] <= box.y + box.height:
            return node
    return None


def _attach_edge_labels(
    image: np.ndarray,
    regions: Sequence[TextRegion],
    graph: DiagramGraph,
    segments: Sequence[Segment],
    reader,
    diagnostics: list[Diagnostic],
) -> None:
    """Attach labels that sit on a connector to the edge that carries it.

    Edge labels matter more than node labels for meaning: ``yes``/``no`` on a
    branch or ``1:N`` on an association is the part that changes the reading.
    """
    if reader is None:
        reader = NullLabelReader()
    try:
        available = bool(reader.available())
    except Exception:
        available = False
    if not available:
        return

    for region in regions:
        if region.placement != "edge":
            continue
        center = region.center
        nearest: Segment | None = None
        nearest_distance = 16.0
        for segment in segments:
            distance = point_segment_distance(center, segment.start, segment.end)
            if distance < nearest_distance:
                nearest, nearest_distance = segment, distance
        if nearest is None:
            continue
        edge = _edge_matching(graph, nearest)
        if edge is None:
            continue
        try:
            text, score = reader.read(image, region.box)
        except Exception:
            text, score = None, 0.0
        cleaned = clean_label(text)
        if cleaned:
            x, y, w, h = region.box
            edge.label = Label(
                text=cleaned,
                region=BoundingBox(x=x, y=y, width=w, height=h),
                reader=reader.name,
                confidence=score,
            )


def _edge_matching(graph: DiagramGraph, segment: Segment) -> DiagramEdge | None:
    """Find the edge whose waypoints coincide with a segment."""
    for edge in graph.edges:
        for index in range(len(edge.waypoints) - 1):
            a = edge.waypoints[index]
            b = edge.waypoints[index + 1]
            if _close(a, segment.start) and _close(b, segment.end):
                return edge
            if _close(b, segment.start) and _close(a, segment.end):
                return edge
    return None


def _close(a: tuple[float, float], b: tuple[float, float], tolerance: float = 3.0) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def _attach_unassigned(
    graph: DiagramGraph,
    shapes: Sequence[Shape],
    regions: Sequence[TextRegion],
    shape_index_to_node: dict[int, str],
) -> None:
    """Keep whatever could not be placed. Nothing is discarded silently."""
    for index, shape in enumerate(shapes):
        if index in shape_index_to_node:
            continue
        graph.unassigned_shapes.append(
            {
                "type": shape.type.value,
                "box": list(shape.box),
                "confidence": round(shape.confidence, 3),
            }
        )
    for region in regions:
        if region.placement == "free":
            graph.unassigned_text.append(
                {"box": list(region.box), "placement": region.placement, "text": region.text}
            )


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------


def _quality_diagnostics(graph: DiagramGraph, diagnostics: list[Diagnostic]) -> None:
    if not graph.nodes and not graph.unassigned_shapes:
        # Silence here reads as success, and it is the one outcome that must
        # never be silent: the file opened, pages were analysed, and nothing
        # was recognised.
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_NOTHING_DETECTED,
                category=Category.STRUCTURAL,
                severity=Severity.ERROR,
                message=(
                    "No shapes were detected, so no diagram was reconstructed. "
                    "The image was read successfully; nothing in it matched."
                ),
                reason=(
                    "The detector works on outlined primitives on a plain "
                    "background. Filled or coloured nodes, illustration, or a "
                    "decorative background are outside what it looks for."
                ),
                hint=(
                    "Try a cleaner export: unfilled shapes, higher contrast, no "
                    "background pattern."
                ),
                evidence={
                    "source": graph.source_image,
                    "width": graph.width,
                    "height": graph.height,
                    "shapes": 0,
                },
            )
        )
    dangling = graph.validate_references()
    if dangling:
        diagnostics.append(
            Diagnostic(
                code=Code.UNSUPPORTED_STRUCTURE,
                category=Category.STRUCTURAL,
                severity=Severity.ERROR,
                message=f"{len(dangling)} edge(s) reference a node that does not exist.",
                reason="Edge construction produced a dangling reference.",
                evidence={"edges": dangling},
            )
        )

    isolated = [
        node.id
        for node in graph.nodes
        if not graph.incoming(node.id) and not graph.outgoing(node.id)
    ]
    if isolated:
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_ISOLATED_NODE,
                category=Category.HEURISTIC,
                severity=Severity.INFO,
                message=f"{len(isolated)} node(s) have no detected connection.",
                reason="A missed connector is the usual cause; an intentionally standalone box is the alternative.",
                evidence={"nodes": isolated},
            )
        )

    unknown = [node.id for node in graph.nodes if node.kind is NodeKind.UNKNOWN]
    if unknown:
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_UNTYPED_SHAPE,
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                message=f"{len(unknown)} shape(s) could not be given a semantic kind.",
                reason="The primitive matched no known convention; the shape is preserved untyped.",
                evidence={"nodes": unknown},
            )
        )

    if graph.unassigned_shapes:
        diagnostics.append(
            Diagnostic(
                code=Code.DIAGRAM_UNASSIGNED_SHAPE,
                category=Category.UNVERIFIED,
                severity=Severity.INFO,
                message=f"{len(graph.unassigned_shapes)} detected shape(s) are not nodes in the graph.",
                reason="They are retained in the output rather than discarded.",
                evidence={"count": len(graph.unassigned_shapes)},
            )
        )
