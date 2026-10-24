"""Diagram graph model.

This is the diagram engine's canonical output: a typed, provenance-carrying
graph. It is deliberately *not* derived from any particular renderer — Mermaid
and PlantUML are both projections of this model, and the JSON form is the
contract.

Two invariants:

1. Every edge names real node ids. There is no "some relationship exists".
2. Anything the engine could not establish carries ``uncertain`` set rather than
   a guessed value.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DIAGRAM_SCHEMA_VERSION = "1.0"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class NodeKind(str, Enum):
    """Semantic role inferred from geometry.

    ``UNKNOWN`` is load-bearing: an unrecognised box becomes a generic node with
    an honest kind, never a guessed one.
    """

    START = "start"
    END = "end"
    TERMINATOR = "terminator"
    PROCESS = "process"
    DECISION = "decision"
    INPUT_OUTPUT = "input_output"
    DATA_STORE = "data_store"
    ACTOR = "actor"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    NOTE = "note"
    GROUP = "group"
    COMPONENT = "component"
    UNKNOWN = "unknown"


class EdgeStyle(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    BIDIRECTIONAL = "bidirectional"


class DiagramKind(str, Enum):
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    ER = "er"
    SEQUENCE = "sequence"
    UML_CLASS = "uml_class"
    BOX_AND_ARROW = "box_and_arrow"
    UNKNOWN = "unknown"


class BoundingBox(_Base):
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class Label(_Base):
    """A text region and whatever we could read from it.

    ``text`` is only populated when a reader actually read it. ``region`` is
    known even when the text is not, which is what makes the spatial
    association useful without OCR.
    """

    text: str | None = None
    region: BoundingBox | None = None
    #: Which reader produced ``text`` ("tesseract", "none").
    reader: str | None = None
    confidence: float = 0.0
    #: True when a region was found but nothing could be read from it.
    unreadable: bool = False

    @property
    def readable(self) -> bool:
        return bool(self.text)


class DiagramNode(_Base):
    id: str
    kind: NodeKind = NodeKind.UNKNOWN
    label: Label = Field(default_factory=Label)
    box: BoundingBox
    #: Raw geometry shape that produced this node.
    shape_type: str = "unknown"
    shape_confidence: float = 0.0
    #: Geometry that drove the semantic classification.
    evidence: dict[str, Any] = Field(default_factory=dict)
    #: Shape ids of contained sub-diagrams, when this node is a group.
    contains: list[str] = Field(default_factory=list)
    #: How many stacked contours were merged into this one. Greater than one
    #: means the source drew internal dividers -- a compartmented box.
    compartments: int = 1
    #: Mermaid/PlantUML node id; stable and safe to reference.
    ref: str = ""

    def __post_init__(self) -> None:
        if not self.ref:
            object.__setattr__(self, "ref", self.id)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class DiagramEdge(_Base):
    id: str
    source: str
    target: str
    label: Label = Field(default_factory=Label)
    style: EdgeStyle = EdgeStyle.SOLID
    #: True when an arrowhead resolved the direction.
    directed: bool = True
    #: Set when direction could only be inferred from layout, not from geometry.
    direction_inferred: bool = False
    arrowhead: str | None = None  # "head", "tail", "both", None
    waypoints: list[tuple[float, float]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class DiagramGraph(_Base):
    """A reconstructed diagram."""

    schema_version: str = DIAGRAM_SCHEMA_VERSION
    kind: DiagramKind = DiagramKind.UNKNOWN
    width: int = 0
    height: int = 0
    title: str | None = None
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    #: Measurements behind the diagram-kind classification.
    evidence: dict[str, Any] = Field(default_factory=dict)
    #: Shapes and text regions that could not be assigned to a node or edge.
    #: Retained so nothing is silently discarded.
    unassigned_shapes: list[dict] = Field(default_factory=list)
    unassigned_text: list[dict] = Field(default_factory=list)
    #: Source image reference (path or asset id) for provenance.
    source_image: str | None = None
    #: True when the reconstruction is not trustworthy enough to publish as fact.
    degraded: bool = False

    # -- queries -----------------------------------------------------------

    def node(self, node_id: str) -> DiagramNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def outgoing(self, node_id: str) -> list[DiagramEdge]:
        return [e for e in self.edges if e.source == node_id]

    def incoming(self, node_id: str) -> list[DiagramEdge]:
        return [e for e in self.edges if e.target == node_id]

    def labelled_nodes(self) -> list[DiagramNode]:
        return [n for n in self.nodes if n.label.readable]

    def kinds(self) -> set[NodeKind]:
        return {n.kind for n in self.nodes}

    def connectivity(self) -> dict:
        """Cheap health metrics, used to judge reconstruction quality."""
        isolated = [n.id for n in self.nodes if not self.incoming(n.id) and not self.outgoing(n.id)]
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "isolated_nodes": len(isolated),
            "readable_labels": len(self.labelled_nodes()),
            "unassigned_shapes": len(self.unassigned_shapes),
        }

    def validate_references(self) -> list[str]:
        """Edges must reference real nodes. Returns dangling edge ids."""
        known = {n.id for n in self.nodes}
        return [e.id for e in self.edges if e.source not in known or e.target not in known]
