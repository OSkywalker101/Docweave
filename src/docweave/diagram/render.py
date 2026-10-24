"""Diagram renderers: Mermaid, PlantUML, JSON and Markdown.

The graph is the contract. Each renderer is a projection of it and is explicit
about what it cannot express: when Mermaid has no equivalent for a construct,
the renderer says so in the output rather than dropping it silently. The
Markdown and JSON forms are always lossless.
"""

from __future__ import annotations

import json
import re

from docweave.diagram.graph import DiagramGraph, DiagramKind, DiagramNode, NodeKind

MERMAID = "mermaid"
PLANTUML = "plantuml"
JSON = "json"
MARKDOWN = "markdown"
DIAGRAM_FORMATS = (MERMAID, PLANTUML, JSON, MARKDOWN)

#: Node kind -> Mermaid node delimiters. Stored as a (open, close) pair rather
#: than a ``str.format`` template, because Mermaid's diamond syntax is itself
#: full of braces that ``format`` would try to interpret.
NODE_SHAPES: dict[NodeKind, tuple[str, str]] = {
    NodeKind.START: ("([", "])"),
    NodeKind.END: ("([", "])"),
    NodeKind.TERMINATOR: ("([", "])"),
    NodeKind.DECISION: ("{", "}"),
    NodeKind.PROCESS: ("[", "]"),
    NodeKind.COMPONENT: ("[", "]"),
    NodeKind.INPUT_OUTPUT: ("[", "]"),
    NodeKind.DATA_STORE: ("[(", ")]"),
    NodeKind.ACTOR: ("[", "]"),
    NodeKind.ENTITY: ("[", "]"),
    NodeKind.RELATIONSHIP: ("[", "]"),
    NodeKind.NOTE: ("[", "]"),
    NodeKind.GROUP: ("[", "]"),
    NodeKind.UNKNOWN: ("[", "]"),
}

#: PlantUML body delimiters for a node kind. Also a (open, close) pair: both
#: target languages use braces as syntax, which ``str.format`` would eat.
PLANTUML_SHAPES: dict[NodeKind, tuple[str, str]] = {
    NodeKind.START: ("(", ")"),
    NodeKind.END: ("(", ")"),
    NodeKind.TERMINATOR: ("(", ")"),
    NodeKind.DECISION: ("{", "}"),
    NodeKind.PROCESS: ("[", "]"),
    NodeKind.COMPONENT: ("[", "]"),
    NodeKind.INPUT_OUTPUT: ("/", "/"),
    NodeKind.DATA_STORE: ("(", ")"),
    NodeKind.ACTOR: ("[", "]"),
    NodeKind.ENTITY: ("[", "]"),
    NodeKind.RELATIONSHIP: ("[", "]"),
    NodeKind.NOTE: ('note "', '"'),
    NodeKind.GROUP: ("[", "]"),
    NodeKind.UNKNOWN: ("[", "]"),
}

_ARROW = {
    "head": "-->",
    "tail": "<--",
    "both": "<-->",
    None: "---",
}

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


class RenderNote:
    """Something the target format cannot express, surfaced rather than dropped."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)

    def as_comment(self, prefix: str = "%%") -> list[str]:
        return [f"{prefix} {message}" for message in self.messages]


def render(graph: DiagramGraph, output_format: str) -> str:
    """Render a graph. Raises :class:`ValueError` for an unknown format."""
    if output_format == JSON:
        return render_json(graph)
    if output_format == MARKDOWN:
        return render_markdown(graph)
    if output_format == MERMAID:
        return render_mermaid(graph)
    if output_format == PLANTUML:
        return render_plantuml(graph)
    raise ValueError(
        f"unknown diagram format {output_format!r}; choose one of {', '.join(DIAGRAM_FORMATS)}"
    )


# --------------------------------------------------------------------------
# identifiers and labels
# --------------------------------------------------------------------------


def _node_ref(node: DiagramNode) -> str:
    """A stable, Mermaid/PlantUML-safe identifier derived from the node id."""
    return _IDENT_RE.sub("_", node.ref or node.id)


def _label(node: DiagramNode) -> str:
    """What to show for a node.

    When a label could not be read the *shape* is still meaningful, so the
    placeholder names the kind rather than leaving an empty box.
    """
    if node.label.text:
        return node.label.text
    return f"{node.kind.value.replace('_', ' ')} ({node.id})"


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ").replace("[", "(").replace("]", ")")


def _escape_plantuml(text: str) -> str:
    return text.replace("\n", " ").replace(":", " ")


# --------------------------------------------------------------------------
# Mermaid
# --------------------------------------------------------------------------


def render_mermaid(graph: DiagramGraph) -> str:
    notes = RenderNote()
    lines: list[str] = []

    if graph.kind is DiagramKind.SEQUENCE:
        lines.extend(_mermaid_sequence(graph, notes))
    elif graph.kind is DiagramKind.ER and _er_is_renderable(graph):
        lines.extend(_mermaid_er(graph, notes))
    else:
        lines.extend(_mermaid_flowchart(graph, notes))

    comments = notes.as_comment()
    if comments:
        lines = comments + lines
    return "\n".join(lines) + "\n"


def _mermaid_flowchart(graph: DiagramGraph, notes: RenderNote) -> list[str]:
    lines = ["flowchart TD"]
    for node in graph.nodes:
        open_delim, close_delim = NODE_SHAPES.get(node.kind, NODE_SHAPES[NodeKind.UNKNOWN])
        label = _escape_mermaid(_label(node))
        lines.append(f"    {_node_ref(node)}{open_delim}{label}{close_delim}")

    for edge in graph.edges:
        arrow = _ARROW.get(edge.arrowhead, _ARROW[None])
        if edge.label.text:
            text = _escape_mermaid(edge.label.text)
            lines.append(f"    {_node_ref(graph.node(edge.source))} -- {text} {arrow} {_node_ref(graph.node(edge.target))}")
        else:
            lines.append(f"    {_node_ref(graph.node(edge.source))} {arrow} {_node_ref(graph.node(edge.target))}")
        if edge.direction_inferred:
            notes.add(
                f"Edge {edge.id} ({edge.source} -> {edge.target}) has no arrowhead in the image; "
                "its direction was inferred from layout."
            )

    for node in graph.nodes:
        for child_id in node.contains:
            child = graph.node(child_id)
            if child is not None:
                lines.append(f"    {child.id} -.-> {node.id}")
                notes.add(
                    f"Mermaid has no nested-node construct; containment {child.id} in {node.id} "
                    "is drawn as a dotted edge."
                )
    return lines


def _er_is_renderable(graph: DiagramGraph) -> bool:
    """ER needs entities and associations; anything else would be a lie."""
    if not graph.edges:
        return False
    kinds = graph.kinds()
    return kinds <= {NodeKind.PROCESS, NodeKind.ENTITY, NodeKind.UNKNOWN, NodeKind.GROUP}


def _mermaid_er(graph: DiagramGraph, notes: RenderNote) -> list[str]:
    lines = ["erDiagram"]
    for node in graph.nodes:
        lines.append(f"    {_node_ref(node)}")
        label = _label(node)
        if not node.label.text:
            notes.add(f"Entity {node.id} has no readable name; Mermaid ER uses the identifier as the name.")
        else:
            notes.add(f"Entity {node.id} is named '{label}'; Mermaid ER renders it as '{_node_ref(node)}'.")
    for edge in graph.edges:
        source = _node_ref(graph.node(edge.source))
        target = _node_ref(graph.node(edge.target))
        label = _escape_mermaid(edge.label.text) if edge.label.text else "relates"
        lines.append(f"    {source} \"{label}\" {target}")
    if any(edge.direction_inferred for edge in graph.edges):
        notes.add("Some associations had no arrowhead; ER relationships are undirected by convention.")
    return lines


def _mermaid_sequence(graph: DiagramGraph, notes: RenderNote) -> list[str]:
    lines = ["sequenceDiagram"]
    actors = list(graph.nodes)
    for node in actors:
        lines.append(f"    participant {_node_ref(node)} as {_escape_mermaid(_label(node))}")
    for edge in graph.edges:
        source = _node_ref(graph.node(edge.source))
        target = _node_ref(graph.node(edge.target))
        label = _escape_mermaid(edge.label.text) if edge.label.text else "step"
        arrow = "-->>" if edge.arrowhead != "tail" else "<<-"
        lines.append(f"    {source}{arrow}{target}: {label}")
    if len(actors) < 2:
        notes.add("A sequence diagram needs at least two participants.")
    return lines


# --------------------------------------------------------------------------
# PlantUML
# --------------------------------------------------------------------------


def render_plantuml(graph: DiagramGraph) -> str:
    notes = RenderNote()
    lines = ["@startuml"]

    if graph.kind is DiagramKind.ER and _er_is_renderable(graph):
        lines.append("hide circle")
    lines.append("skinparam dpi 120")

    for node in graph.nodes:
        open_delim, close_delim = PLANTUML_SHAPES.get(node.kind, ("[", "]"))
        label = _escape_plantuml(_label(node))
        lines.append(f"{_node_ref(node)} as {open_delim}{label}{close_delim}")

    for edge in graph.edges:
        source = _node_ref(graph.node(edge.source))
        target = _node_ref(graph.node(edge.target))
        if edge.arrowhead in ("head", "both"):
            connector = "<--" if edge.arrowhead == "both" else "-->"
        elif edge.arrowhead == "tail":
            connector = "<--"
        else:
            connector = "--"
        if edge.label.text:
            lines.append(f"{source} {connector} {target} : {_escape_plantuml(edge.label.text)}")
        else:
            lines.append(f"{source} {connector} {target}")
        if edge.direction_inferred:
            notes.add(f"Edge {edge.id} direction was inferred from layout, not from an arrowhead.")

    for node in graph.nodes:
        for child_id in node.contains:
            child = graph.node(child_id)
            if child is not None:
                lines.append(f"{child.id} ..> {node.id} : contained in")
                notes.add(f"PlantUML packages are not inferred; containment {child.id} in {node.id} drawn as an edge.")

    lines.extend(notes.as_comment())
    lines.append("@enduml")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def render_markdown(graph: DiagramGraph) -> str:
    """Lossless human-readable rendering, including the uncertain parts."""
    lines: list[str] = [f"# Diagram: {graph.kind.value}", ""]

    stats = graph.connectivity()
    lines.append(
        f"{stats['node_count']} node(s), {stats['edge_count']} edge(s), "
        f"{stats['readable_labels']} label(s) read. "
        f"Source image: {graph.source_image or '(not recorded)'}"
    )
    lines.append("")
    if graph.kind is DiagramKind.UNKNOWN:
        lines.append(
            "> The diagram type could not be determined confidently. "
            "Treat the structure below as a plain node/edge graph."
        )
        lines.append("")

    if graph.nodes:
        lines.append("## Nodes")
        lines.append("")
        lines.append("| id | kind | shape | label | position |")
        lines.append("| --- | --- | --- | --- | --- |")
        for node in graph.nodes:
            label = node.label.text or "(not read)"
            position = f"{node.box.x},{node.box.y} {node.box.width}x{node.box.height}"
            lines.append(
                f"| {node.id} | {node.kind.value} | {node.shape_type} | "
                f"{_escape_pipe(label)} | {position} |"
            )
        lines.append("")

    if graph.edges:
        lines.append("## Edges")
        lines.append("")
        lines.append("| id | from | to | label | direction | style |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for edge in graph.edges:
            direction = edge.arrowhead or "none"
            if edge.direction_inferred:
                direction = "inferred from layout"
            lines.append(
                f"| {edge.id} | {edge.source} | {edge.target} | "
                f"{_escape_pipe(edge.label.text or '')} | {direction} | {edge.style.value} |"
            )
        lines.append("")

    if graph.unassigned_shapes or graph.unassigned_text:
        lines.append("## Not reconstructed")
        lines.append("")
        for item in graph.unassigned_shapes:
            lines.append(f"- shape `{item['type']}` at {item['box']} is not a node in the graph")
        for item in graph.unassigned_text:
            lines.append(f"- text region at {item['box']} could not be attached to a shape or edge")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def render_json(graph: DiagramGraph) -> str:
    payload = {
        "schema": "docweave.diagram/1.0",
        "schema_version": graph.schema_version,
        "kind": graph.kind.value,
        "evidence": graph.evidence,
        "width": graph.width,
        "height": graph.height,
        "source_image": graph.source_image,
        "degraded": graph.degraded,
        "nodes": [node.to_dict() for node in graph.nodes],
        "edges": [edge.to_dict() for edge in graph.edges],
        "unassigned_shapes": graph.unassigned_shapes,
        "unassigned_text": graph.unassigned_text,
        "summary": graph.connectivity(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _escape_pipe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
