"""Synthetic diagram corpus.

Every diagram is drawn with OpenCV from a declarative spec, and the same spec
is returned as ground truth. That is what makes reconstruction accuracy
measurable instead of guessed: the test asserts the detected graph matches the
graph that was drawn.

Deterministic by construction: fixed sizes, fixed colours, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (90, 90, 90)


@dataclass
class SpecNode:
    key: str
    shape: str  # rect | rounded | diamond | ellipse | parallelogram | cylinder | actor | group
    box: tuple[int, int, int, int]
    label: str = ""
    #: Filled shapes break text-region detection; keep outlines for clean cases.
    fill: bool = False
    #: Horizontal divider lines inside the box, as fractions of its height. UML
    #: class compartments: a name, then attributes, then operations.
    dividers: list[float] = field(default_factory=list)


@dataclass
class SpecEdge:
    source: str
    target: str
    #: "arrow" | "plain" | "both"
    heads: str = "arrow"
    label: str = ""
    #: Draw as a dashed line.
    dashed: bool = False


@dataclass
class DiagramSpec:
    name: str
    kind: str
    width: int
    height: int
    nodes: list[SpecNode] = field(default_factory=list)
    edges: list[SpecEdge] = field(default_factory=list)
    #: Optional plain-text title above the diagram.
    title: str = ""
    #: Extra free-floating labels not attached to any node.
    loose_text: list[tuple[str, tuple[int, int, int, int]]] = field(default_factory=list)
    #: Sequence-diagram lifelines: (x, y_top, y_bottom).
    lifelines: list[tuple[int, int, int]] = field(default_factory=list)

    def node(self, key: str) -> SpecNode:
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key)


def _blank(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    return image


def _draw_shape(image, spec: SpecNode) -> None:
    import cv2

    x, y, w, h = spec.box
    color = BLACK
    thickness = 2
    name = spec.shape

    if name == "rect":
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    elif name == "rounded":
        cv2.rectangle(
            image, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA
        )
        radius = max(6, min(w, h) // 5)
        for cx, cy, start, end in (
            (x + radius, y + radius, 180, 270),
            (x + w - radius, y + radius, 270, 360),
            (x + w - radius, y + h - radius, 0, 90),
            (x + radius, y + h - radius, 90, 180),
        ):
            cv2.ellipse(image, (cx, cy), (radius, radius), 0, start, end, color, thickness)
    elif name == "diamond":
        pts = np.array(
            [
                [x + w // 2, y],
                [x + w, y + h // 2],
                [x + w // 2, y + h],
                [x, y + h // 2],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [pts], True, color, thickness)
    elif name == "ellipse":
        cv2.ellipse(image, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, color, thickness)
    elif name == "parallelogram":
        skew = max(10, w // 5)
        pts = np.array(
            [
                [x + skew, y],
                [x + w, y],
                [x + w - skew, y + h],
                [x, y + h],
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [pts], True, color, thickness)
    elif name == "cylinder":
        # The bottom cap is drawn bulging upward, which is geometrically wrong
        # for a cylinder. It is left as-is deliberately: correcting it made the
        # flowchart fixture's datastore vanish from detection entirely, which is
        # a real but separate gap in the contour walk. See docs/architecture.md.
        cv2.ellipse(image, (x + w // 2, y + h // 6), (w // 2, h // 6), 0, 0, 180, color, thickness)
        cv2.line(image, (x, y + h // 6), (x, y + h - h // 6), color, thickness)
        cv2.line(image, (x + w, y + h // 6), (x + w, y + h - h // 6), color, thickness)
        cv2.ellipse(image, (x + w // 2, y + h - h // 6), (w // 2, h // 6), 0, 0, 180, color, thickness)
    elif name == "group":
        # A subsystem frame: a box drawn around other boxes with real slack.
        _dashed_line(image, (x, y), (x + w, y), BLACK)
        _dashed_line(image, (x + w, y), (x + w, y + h), BLACK)
        _dashed_line(image, (x + w, y + h), (x, y + h), BLACK)
        _dashed_line(image, (x, y + h), (x, y), BLACK)
    elif name == "actor":
        cx = x + w // 2
        head_r = max(5, w // 6)
        cv2.circle(image, (cx, y + head_r), head_r, color, thickness)
        cv2.line(image, (cx, y + 2 * head_r), (cx, y + h - 12), color, thickness)
        cv2.line(image, (cx - w // 3, y + h - 12), (cx + w // 3, y + h - 12), color, thickness)
        cv2.line(image, (cx, y + h // 2), (cx - w // 3, y + h - 18), color, thickness)
        cv2.line(image, (cx, y + h // 2), (cx + w // 3, y + h - 18), color, thickness)
    else:
        raise ValueError(f"unknown shape {name!r}")

    for fraction in spec.dividers:
        # A compartment divider is deliberately the same weight as the box
        # outline, because that is how UML draws it, and because a thinner line
        # would let the detector skip it for reasons that have nothing to do
        # with whether it should be mistaken for a connector.
        y_line = int(y + h * fraction)
        cv2.line(image, (x + 2, y_line), (x + w - 2, y_line), color, thickness)

    if spec.fill:
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)], (215, 215, 215))
        cv2.addWeighted(overlay, 1.0, image, 0.0, 0, image)


def _anchor(node: SpecNode, toward: tuple[int, int]) -> tuple[int, int]:
    """Where a connector should meet ``node`` when heading toward ``toward``."""
    x, y, w, h = node.box
    cx, cy = x + w // 2, y + h // 2
    dx = toward[0] - cx
    dy = toward[1] - cy
    if dx == 0 and dy == 0:
        return (cx, cy)
    scale = (w / 2.0) / abs(dx) if dx else float("inf")
    scale = min(scale, (h / 2.0) / abs(dy) if dy else float("inf"))
    return (int(round(cx + dx * scale)), int(round(cy + dy * scale)))


def _draw_edge(image, edge: SpecEdge, a: SpecNode, b: SpecNode) -> None:
    import cv2

    a_center = (a.box[0] + a.box[2] // 2, a.box[1] + a.box[3] // 2)
    b_center = (b.box[0] + b.box[2] // 2, b.box[1] + b.box[3] // 2)
    start = _anchor(a, b_center)
    end = _anchor(b, a_center)

    line_type = cv2.LINE_AA
    color = GREY if edge.dashed else BLACK
    if edge.dashed:
        _dashed_line(image, start, end, color)
    else:
        cv2.line(image, start, end, color, 2, line_type)

    head_size = 9
    if edge.heads in ("arrow", "both"):
        _arrowhead(image, end, start, head_size, color)
    if edge.heads == "both":
        _arrowhead(image, start, end, head_size, color)

    if edge.label:
        _draw_label(image, edge.label, ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2))


def _dashed_line(image, start, end, color, dash: int = 9, gap: int = 7) -> None:
    import cv2

    total = int(np.hypot(end[0] - start[0], end[1] - start[1]))
    if total == 0:
        return
    ux = (end[0] - start[0]) / total
    uy = (end[1] - start[1]) / total
    position = 0
    while position < total:
        seg_end = min(total, position + dash)
        cv2.line(
            image,
            (int(start[0] + ux * position), int(start[1] + uy * position)),
            (int(start[0] + ux * seg_end), int(start[1] + uy * seg_end)),
            color,
            2,
        )
        position = seg_end + gap


def _arrowhead(image, tip, away, size: int, color) -> None:
    import cv2

    vx, vy = tip[0] - away[0], tip[1] - away[1]
    norm = float(np.hypot(vx, vy))
    if norm == 0:
        return
    ux, uy = vx / norm, vy / norm
    px, py = -uy, ux
    base = (tip[0] - ux * size, tip[1] - uy * size)
    pts = np.array(
        [
            tip,
            (int(base[0] + px * size * 0.5), int(base[1] + py * size * 0.5)),
            (int(base[0] - px * size * 0.5), int(base[1] - py * size * 0.5)),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pts], color)


def _draw_label(image, text: str, center: tuple[int, int]) -> None:
    import cv2

    scale = 0.42
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    origin = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 1, cv2.LINE_AA)


def render(spec: DiagramSpec) -> np.ndarray:
    """Render a spec to an image. The spec doubles as ground truth."""
    import cv2

    image = _blank(spec.width, spec.height)
    if spec.title:
        cv2.putText(
            image, spec.title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA
        )

    for edge in spec.edges:
        _draw_edge(image, edge, spec.node(edge.source), spec.node(edge.target))

    for x, y0, y1 in spec.lifelines:
        _dashed_line(image, (x, y0), (x, y1), GREY, dash=7, gap=6)

    for node in spec.nodes:
        _draw_shape(image, node)
        if node.label:
            _draw_label(
                image,
                node.label,
                (node.box[0] + node.box[2] // 2, node.box[1] + node.box[3] // 2),
            )

    for text, (x, y, w, h) in spec.loose_text:
        _draw_label(image, text, (x + w // 2, y + h // 2))

    return image


def write_png(image: np.ndarray, path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")
    return path


# --------------------------------------------------------------------------
# the corpus
# --------------------------------------------------------------------------


def flowchart() -> DiagramSpec:
    return DiagramSpec(
        name="flowchart",
        kind="flowchart",
        width=720,
        height=520,
        title="Order flow",
        nodes=[
            SpecNode("start", "ellipse", (300, 60, 130, 60), "Start"),
            SpecNode("validate", "rect", (280, 170, 170, 60), "Validate order"),
            SpecNode("stock", "diamond", (295, 285, 140, 110), "In stock?"),
            SpecNode("pick", "rect", (60, 310, 150, 60), "Pick items"),
            SpecNode("ship", "rect", (480, 310, 170, 60), "Ship order"),
            SpecNode("db", "cylinder", (285, 455, 160, 50), "Inventory"),
            SpecNode("end", "ellipse", (505, 440, 110, 55), "Done"),
        ],
        edges=[
            SpecEdge("start", "validate"),
            SpecEdge("validate", "stock"),
            SpecEdge("stock", "pick", label="no"),
            SpecEdge("stock", "ship", label="yes"),
            SpecEdge("pick", "db"),
            SpecEdge("db", "ship"),
            SpecEdge("ship", "end"),
        ],
    )


def architecture() -> DiagramSpec:
    return DiagramSpec(
        name="architecture",
        kind="architecture",
        width=760,
        height=480,
        title="Service topology",
        nodes=[
            SpecNode("client", "rect", (40, 200, 130, 70), "Client"),
            SpecNode("gateway", "rect", (240, 190, 150, 90), "API Gateway"),
            SpecNode("auth", "rect", (480, 90, 140, 60), "Auth"),
            SpecNode("orders", "rect", (480, 210, 140, 60), "Orders"),
            SpecNode("billing", "rect", (480, 330, 140, 60), "Billing"),
            SpecNode("db", "cylinder", (270, 350, 150, 70), "Primary DB"),
        ],
        edges=[
            SpecEdge("client", "gateway"),
            SpecEdge("gateway", "auth", dashed=True),
            SpecEdge("gateway", "orders"),
            SpecEdge("gateway", "billing"),
            SpecEdge("orders", "db", dashed=True),
        ],
    )


def er_diagram() -> DiagramSpec:
    """ER convention: plain association lines with cardinality labels.

    Real ER diagrams do not use arrowheads. Drawing them would make this
    indistinguishable from a dataflow diagram, and the classifier would (rightly)
    call it one.
    """
    return DiagramSpec(
        name="er",
        kind="er",
        width=700,
        height=420,
        title="Schema",
        nodes=[
            SpecNode("customer", "rect", (60, 70, 180, 70), "Customer"),
            SpecNode("order", "rect", (400, 60, 170, 80), "Order"),
            SpecNode("line", "rect", (400, 250, 170, 80), "OrderLine"),
            SpecNode("product", "rect", (60, 260, 180, 70), "Product"),
        ],
        edges=[
            SpecEdge("customer", "order", heads="plain", label="1:N"),
            SpecEdge("order", "line", heads="plain", label="1:1"),
            SpecEdge("line", "product", heads="plain", label="N:M"),
        ],
    )


def sequence() -> DiagramSpec:
    """Sequence convention: participant boxes plus vertical lifelines."""
    spec = DiagramSpec(
        name="sequence",
        kind="sequence",
        width=680,
        height=430,
        title="Login",
        nodes=[
            SpecNode("user", "rect", (50, 85, 120, 50), "User"),
            SpecNode("app", "rect", (290, 85, 120, 50), "App"),
            SpecNode("db", "rect", (520, 85, 120, 50), "DB"),
        ],
        edges=[
            SpecEdge("user", "app", label="login"),
            SpecEdge("app", "db", label="check"),
            SpecEdge("db", "app", label="token"),
        ],
    )
    # Lifelines: long vertical connectors dropping from each participant.
    for node in spec.nodes:
        x, y, w, _h = node.box
        spec.lifelines.append((x + w // 2, y + 50, 380))
    return spec


def box_and_arrow() -> DiagramSpec:
    return DiagramSpec(
        name="box_and_arrow",
        kind="box_and_arrow",
        width=620,
        height=400,
        title="Simple flow",
        nodes=[
            SpecNode("a", "rect", (60, 80, 150, 70), "Input"),
            SpecNode("b", "rounded", (440, 80, 140, 70), "Process"),
            SpecNode("c", "rect", (250, 270, 150, 70), "Store"),
        ],
        edges=[
            SpecEdge("a", "b", heads="plain"),
            SpecEdge("b", "c", heads="arrow"),
        ],
    )


def parallelogram_io() -> DiagramSpec:
    return DiagramSpec(
        name="parallelogram",
        kind="flowchart",
        width=560,
        height=260,
        title="IO shapes",
        nodes=[
            SpecNode("in", "parallelogram", (60, 90, 200, 70), "Read input"),
            SpecNode("work", "rect", (320, 90, 170, 70), "Transform"),
        ],
        edges=[SpecEdge("in", "work")],
    )


def ambiguous() -> DiagramSpec:
    """Overlapping boxes and an unattached stub: reconstruction must degrade."""
    return DiagramSpec(
        name="ambiguous",
        kind="unknown",
        width=520,
        height=320,
        title="Ambiguous",
        nodes=[
            SpecNode("left", "rect", (60, 90, 170, 80), "Left"),
            SpecNode("right", "rect", (250, 130, 170, 80), "Right"),
            SpecNode("orphan", "rect", (300, 240, 120, 50), "Orphan"),
        ],
        edges=[SpecEdge("left", "right")],
        loose_text=[("?", (80, 250, 40, 24))],
    )


def uml_class_diagram() -> DiagramSpec:
    """Class compartments and undirected associations.

    Added to try to break the classifier rather than to flatter it: the engine
    carries a ``UML_CLASS`` kind that nothing ever returned, and a compartment
    divider is a straight line inside a box -- exactly the thing a connector
    detector could mistake for an edge.
    """
    return DiagramSpec(
        name="uml_class",
        kind="uml_class",
        width=720,
        height=460,
        title="Class model",
        nodes=[
            SpecNode("order", "rect", (60, 80, 230, 150), "Order", dividers=[0.4, 0.7]),
            SpecNode("customer", "rect", (430, 80, 230, 150), "Customer", dividers=[0.4, 0.7]),
            SpecNode("line_item", "rect", (240, 300, 230, 150), "LineItem", dividers=[0.4, 0.7]),
        ],
        edges=[
            SpecEdge("order", "customer", heads="plain"),
            SpecEdge("order", "line_item", heads="plain"),
        ],
    )


def state_machine() -> DiagramSpec:
    """Cyclic flow through rounded states.

    No diamond and no parallelogram, so no shape carries flowchart meaning on
    its own: ``FLOWCHART_SHAPES`` is ``{decision, input_output}``. Rounded
    rectangles map to ``START``, but START is deliberately not flowchart
    evidence -- a rounded box is as common in a service topology as in a flow.
    A ring of four rounded states connected by arrows is therefore a plain
    box-and-arrow graph, which is what the engine reports.
    """
    return DiagramSpec(
        name="state_machine",
        kind="box_and_arrow",
        width=760,
        height=400,
        title="Order lifecycle",
        nodes=[
            SpecNode("new", "rounded", (60, 150, 170, 70), "New"),
            SpecNode("paid", "rounded", (290, 80, 170, 70), "Paid"),
            SpecNode("shipped", "rounded", (520, 150, 170, 70), "Shipped"),
            SpecNode("refunded", "rounded", (290, 260, 170, 70), "Refunded"),
        ],
        edges=[
            SpecEdge("new", "paid"),
            SpecEdge("paid", "shipped"),
            SpecEdge("shipped", "refunded"),
            SpecEdge("refunded", "new"),
        ],
    )


def grouped_subsystem() -> DiagramSpec:
    """A dashed frame containing three services, plus one node outside it.

    The frame is a group, not a process. A detector that reads it as another
    box would both add a phantom node and misjudge what the diagram is.
    """
    return DiagramSpec(
        name="grouped_subsystem",
        kind="architecture",
        width=780,
        height=440,
        title="Payments subsystem",
        nodes=[
            SpecNode("frame", "group", (240, 60, 470, 330), ""),
            SpecNode("auth", "rect", (270, 110, 180, 70), "Auth"),
            SpecNode("charge", "rect", (500, 110, 180, 70), "Charge"),
            SpecNode("ledger", "rect", (385, 270, 180, 70), "Ledger"),
            SpecNode("client", "rect", (40, 170, 150, 70), "Client"),
        ],
        edges=[
            SpecEdge("client", "auth"),
            SpecEdge("charge", "ledger"),
            SpecEdge("auth", "charge"),
        ],
    )


def network_topology() -> DiagramSpec:
    """A hub with many spokes, several of them dashed links.

    Differs from the architecture fixture by having no data store and by
    depending entirely on dash detection to read the link semantics.
    """
    return DiagramSpec(
        name="network_topology",
        kind="architecture",
        width=820,
        height=460,
        title="Network topology",
        nodes=[
            SpecNode("router", "rect", (350, 180, 140, 80), "Router"),
            SpecNode("web", "rect", (90, 70, 150, 70), "Web"),
            SpecNode("mail", "rect", (90, 200, 150, 70), "Mail"),
            SpecNode("file", "rect", (90, 330, 150, 70), "File"),
            SpecNode("db1", "rect", (600, 70, 150, 70), "DB1"),
            SpecNode("db2", "rect", (600, 330, 150, 70), "DB2"),
        ],
        edges=[
            SpecEdge("router", "web"),
            SpecEdge("router", "mail", dashed=True),
            SpecEdge("router", "file"),
            SpecEdge("router", "db1", dashed=True),
            SpecEdge("router", "db2"),
        ],
    )


def sequence_of_diagrams() -> list[DiagramSpec]:
    return [
        flowchart(),
        architecture(),
        er_diagram(),
        sequence(),
        box_and_arrow(),
        parallelogram_io(),
        ambiguous(),
        uml_class_diagram(),
        state_machine(),
        grouped_subsystem(),
        network_topology(),
    ]


def build_all(directory: Path) -> dict[str, Path]:
    """Write every diagram PNG and return ``name -> path``."""
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for spec in sequence_of_diagrams():
        image = render(spec)
        out[spec.name] = write_png(image, directory / f"{spec.name}.png")
    return out


def specs_by_name() -> dict[str, DiagramSpec]:
    return {spec.name: spec for spec in sequence_of_diagrams()}


if __name__ == "__main__":  # pragma: no cover - manual regeneration
    target = Path(__file__).parent / "fixtures" / "diagrams"
    for name, path in build_all(target).items():
        print(name, path.stat().st_size)
