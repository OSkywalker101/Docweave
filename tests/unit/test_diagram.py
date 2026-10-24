"""Diagram engine: geometric detection, graph construction and honesty.

The synthetic corpus ships a spec alongside every image, so these tests compare
the reconstructed graph against the graph that was actually drawn. That is what
makes "reconstruction accuracy" a measurement rather than an opinion.
"""

from __future__ import annotations

import json

import pytest

from docweave.diagram import (
    DiagramAnalyzer,
    DiagramKind,
    NodeKind,
    analyze_file,
    load_image,
    render,
)
from docweave.diagram.graph import EdgeStyle
from docweave.diagram.render import JSON, MARKDOWN, MERMAID, PLANTUML
from docweave.diagram.shapes import (
    box_iou,
    detect_segments,
    detect_shapes,
    point_segment_distance,
    vertex_angles,
)
from docweave.diagram.text import NullLabelReader, detect_text_regions, group_regions
from docweave.models.diagnostics import Code, Severity
from tests.conftest import requires_diagrams, requires_tesseract

pytestmark = requires_diagrams


def _graph(diagram_corpus, name):
    return analyze_file(
        diagram_corpus / f"{name}.png", reader=NullLabelReader()
    ).graph


def _image(diagram_corpus, name):
    return load_image(diagram_corpus / f"{name}.png")


class TestEdgeStyle:
    """Dash detection.

    ``EdgeStyle.DASHED`` existed in the IR and ``dashed_edges`` fed the
    ER-versus-architecture branch, but nothing ever set either: the value was
    structurally always zero. These tests exist because that was invisible.
    """

    def test_a_dashed_connector_is_reported_as_dashed(self, diagram_corpus):
        graph = _graph(diagram_corpus, "architecture")
        assert any(e.style is EdgeStyle.DASHED for e in graph.edges)

    def test_dash_state_reaches_the_kind_evidence(self, diagram_corpus):
        graph = _graph(diagram_corpus, "architecture")
        assert graph.evidence["dashed_edges"] >= 1

    def test_a_solid_diagram_reports_no_dashes(self, diagram_corpus):
        assert _graph(diagram_corpus, "er").evidence["dashed_edges"] == 0

    def test_style_belongs_to_the_edge_not_the_last_segment(self, diagram_corpus):
        """Regression: every edge inherited the final segment's flag.

        The style was read from ``segment``, a leaked loop variable from the
        pass that grouped segments, so all edges shared one value.
        """
        graph = _graph(diagram_corpus, "architecture")
        styles = {e.style for e in graph.edges}
        assert len(styles) > 1, "a mixed diagram must not report a single style"

    def test_undirected_associations_keep_their_own_style(self, diagram_corpus):
        graph = _graph(diagram_corpus, "er")
        assert all(e.style is EdgeStyle.ASSOCIATION for e in graph.edges)


class TestDiagramKind:
    """A kind must be claimed only on evidence, never by default."""

    def test_flowchart_shapes_alone_imply_a_flowchart(self, diagram_corpus):
        """A parallelogram is a flowchart shape without any diamond present."""
        assert _graph(diagram_corpus, "parallelogram").kind is DiagramKind.FLOWCHART

    def test_an_unconnected_diagram_is_not_given_a_kind(self, diagram_corpus):
        """Too little structure to read: say so instead of guessing."""
        assert _graph(diagram_corpus, "ambiguous").kind is DiagramKind.UNKNOWN

    def test_a_linear_chain_is_not_called_architecture(self, diagram_corpus):
        """Arrows do not make a topology. Claiming architecture would overstate."""
        graph = _graph(diagram_corpus, "box_and_arrow")
        assert graph.kind is DiagramKind.BOX_AND_ARROW
        assert graph.evidence["hub_degree"] < 3

    def test_fan_out_topology_is_reported_as_architecture(self, diagram_corpus):
        graph = _graph(diagram_corpus, "architecture")
        assert graph.kind is DiagramKind.ARCHITECTURE
        assert graph.evidence["hub_degree"] >= 3


# --------------------------------------------------------------------------
# geometry primitives
# --------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_vertex_angles_of_a_square(self):
        import numpy as np

        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        angles = vertex_angles(square)
        assert len(angles) == 4
        assert all(abs(a - 90) < 1 for a in angles)

    def test_iou(self):
        assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert box_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
        assert 0 < box_iou((0, 0, 10, 10), (5, 0, 10, 10)) < 1

    def test_point_segment_distance(self):
        assert point_segment_distance((5, 5), (0, 0), (10, 0)) == pytest.approx(5.0)
        assert point_segment_distance((5, 0), (0, 0), (10, 0)) == pytest.approx(0.0)
        # Beyond the end of the segment, distance is to the endpoint.
        assert point_segment_distance((15, 0), (0, 0), (10, 0)) == pytest.approx(5.0)


class TestShapeDetection:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("er", 4),
            ("box_and_arrow", 3),
            ("sequence", 3),
            ("parallelogram", 2),
            ("flowchart", 7),
            ("state_machine", 4),
        ],
    )
    def test_node_counts_match_the_drawn_graph(self, diagram_corpus, name, expected):
        """Every drawn shape must be found. Extra shapes are caught separately."""
        graph = _graph(diagram_corpus, name)
        assert len(graph.nodes) == expected

    def test_a_ring_of_connectors_invents_no_node(self, diagram_corpus):
        """The blank a ring of connectors encloses is not a drawn shape.

        Four states joined in a cycle enclose a central region. It fills a
        coarse polygon well enough to pass the primitive test, so without the
        enclosure filter it becomes a fifth node that absorbs the connectors.
        """
        graph = _graph(diagram_corpus, "state_machine")
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 4

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("flowchart", 7),
            ("architecture", 5),
            ("er", 3),
            ("box_and_arrow", 2),
            ("parallelogram", 1),
            ("uml_class", 2),
            ("network_topology", 5),
        ],
    )
    def test_edge_counts_match_the_drawn_graph(self, diagram_corpus, name, expected):
        """Every drawn connector must be found."""
        graph = _graph(diagram_corpus, name)
        assert len(graph.edges) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="the only connector is a 21px hop, below the length floor of ~35px",
    )
    def test_ambiguous_short_connector_is_found(self, diagram_corpus):
        """Two boxes drawn nearly touching are still joined by a connector."""
        assert len(_graph(diagram_corpus, "ambiguous").edges) == 1

    @pytest.mark.xfail(
        strict=True,
        reason="without an arrowhead the two hops are indistinguishable and are merged",
    )
    def test_sequence_return_hop_is_found(self, diagram_corpus):
        """A message and its reply are two edges, not one."""
        assert len(_graph(diagram_corpus, "sequence").edges) == 3

    def test_sequence_message_and_reply_stay_separate_lines(self, diagram_corpus):
        """The reply runs just below the message; both strokes must survive."""
        image = _image(diagram_corpus, "sequence")
        shapes = detect_shapes(image)
        text_boxes = [region.box for region in detect_text_regions(image)]
        segments = detect_segments(image, shapes, text_boxes)
        horizontals = [
            s
            for s in segments
            if s.is_horizontal and s.length > 80 and s.start[0] > 390 and 100 <= s.start[1] <= 130
        ]
        assert len(horizontals) == 2

    def test_flowchart_semantics(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        kinds = {node.kind for node in graph.nodes}
        assert NodeKind.DECISION in kinds
        assert NodeKind.START in kinds
        assert NodeKind.END in kinds

    def test_start_and_end_are_chosen_by_layout(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        starts = [n for n in graph.nodes if n.kind is NodeKind.START]
        ends = [n for n in graph.nodes if n.kind is NodeKind.END]
        assert len(starts) == 1 and len(ends) == 1
        assert starts[0].box.y < ends[0].box.y

    def test_parallelogram_is_not_a_diamond(self, diagram_corpus):
        graph = _graph(diagram_corpus, "parallelogram")
        kinds = [n.kind for n in graph.nodes]
        assert NodeKind.INPUT_OUTPUT in kinds
        assert NodeKind.DECISION not in kinds

    def test_shapes_carry_measured_evidence(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        for node in graph.nodes:
            assert node.evidence["shape_type"]
            assert node.box.width > 0 and node.box.height > 0
            assert node.shape_confidence > 0

    def test_no_duplicate_nodes_from_stroked_outlines(self, diagram_corpus):
        """An outline yields two contours; only one may become a node."""
        graph = _graph(diagram_corpus, "er")
        boxes = [(n.box.x, n.box.y, n.box.width, n.box.height) for n in graph.nodes]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                assert box_iou(a, b) < 0.8, "two nodes overlap almost completely"


class TestTextRegions:
    def test_regions_are_found_inside_shapes(self, diagram_corpus):
        image = load_image(diagram_corpus / "er.png")
        regions = group_regions(detect_text_regions(image))
        assert regions
        assert any(r.box[2] > 10 for r in regions)

    def test_arrowheads_are_not_mistaken_for_text(self, diagram_corpus):
        """A solid triangle is one ink blob; a label is several."""
        image = load_image(diagram_corpus / "sequence.png")
        regions = detect_text_regions(image)
        for region in regions:
            assert region.box[2] / max(region.box[3], 1) > 0.18

    def test_null_reader_yields_no_text(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        assert all(not node.label.readable for node in graph.nodes)
        assert any(node.label.unreadable for node in graph.nodes)


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------


class TestEdges:
    def test_flowchart_reconstructs_all_relationships(self, diagram_corpus, diagram_specs):
        graph = _graph(diagram_corpus, "flowchart")
        spec = diagram_specs["flowchart"]
        assert len(graph.edges) == len(spec.edges)

    def test_edges_only_reference_real_nodes(self, diagram_corpus):
        for name in ("flowchart", "er", "architecture", "sequence", "ambiguous"):
            graph = _graph(diagram_corpus, name)
            assert graph.validate_references() == []

    def test_direction_is_proven_or_marked_inferred(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        for edge in graph.edges:
            if edge.direction_inferred:
                assert edge.arrowhead is None
            else:
                assert edge.arrowhead in ("head", "tail", "both")

    def test_arrows_point_the_way_the_walk_does(self, diagram_corpus):
        """The flowchart is drawn top-to-bottom, so edges must flow downward."""
        graph = _graph(diagram_corpus, "flowchart")
        downward = 0
        for edge in graph.edges:
            source = graph.node(edge.source)
            target = graph.node(edge.target)
            if target.box.y > source.box.y:
                downward += 1
        assert downward >= len(graph.edges) // 2

    def test_undirected_edges_are_marked(self, diagram_corpus):
        graph = _graph(diagram_corpus, "er")
        assert any(not edge.directed for edge in graph.edges)

    def test_unconnected_lines_are_reported_not_invented(self, diagram_corpus):
        result = analyze_file(diagram_corpus / "ambiguous.png", reader=NullLabelReader())
        codes = [d.code for d in result.diagnostics]
        assert Code.DIAGRAM_UNCONNECTED in codes
        # Overlapping boxes leave no unambiguous connector: we publish no edge.
        assert result.graph.edges == []


class TestCompartmentedBoxes:
    """A box with internal dividers is one node, not several.

    Added after the UML fixture showed a single class box detected as three
    compartment nodes joined by two invented edges.
    """

    def test_dividers_do_not_become_nodes(self, diagram_corpus):
        graph = _graph(diagram_corpus, "uml_class")
        assert len(graph.nodes) == 3

    def test_dividers_do_not_become_edges(self, diagram_corpus):
        graph = _graph(diagram_corpus, "uml_class")
        assert len(graph.edges) == 2, "internal dividers are not relationships"

    def test_merged_node_records_its_compartment_count(self, diagram_corpus):
        graph = _graph(diagram_corpus, "uml_class")
        assert all(n.compartments == 3 for n in graph.nodes)

    def test_no_edge_points_at_a_node_that_no_longer_exists(self, diagram_corpus):
        """Regression: merging left edges referencing absorbed ids."""
        graph = _graph(diagram_corpus, "uml_class")
        known = {n.id for n in graph.nodes}
        assert all(e.source in known and e.target in known for e in graph.edges)

    def test_compartmented_boxes_identify_a_class_model(self, diagram_corpus):
        assert _graph(diagram_corpus, "uml_class").kind is DiagramKind.UML_CLASS


class TestHonesty:
    def test_missing_labels_are_reported(self, diagram_corpus):
        result = analyze_file(diagram_corpus / "flowchart.png", reader=NullLabelReader())
        unread = [d for d in result.diagnostics if d.code == Code.DIAGRAM_LABEL_UNREAD]
        assert unread
        assert unread[0].hint and "ocr" in unread[0].hint.lower()

    def test_source_image_is_preserved(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        assert graph.source_image == "flowchart.png"

    def test_degraded_flag_when_uncertain(self, diagram_corpus):
        clean = analyze_file(diagram_corpus / "flowchart.png", reader=NullLabelReader())
        messy = analyze_file(diagram_corpus / "ambiguous.png", reader=NullLabelReader())
        assert messy.graph.degraded is True
        assert clean.graph.degraded in (True, False)

    def test_isolated_nodes_are_surfaced(self, diagram_corpus):
        result = analyze_file(diagram_corpus / "ambiguous.png", reader=NullLabelReader())
        assert any(d.code == Code.DIAGRAM_ISOLATED_NODE for d in result.diagnostics)

    def test_nothing_is_silently_dropped(self, diagram_corpus):
        """Whatever the detector saw, the graph accounts for."""
        for name in ("flowchart", "er", "sequence", "architecture", "ambiguous"):
            result = analyze_file(
                diagram_corpus / f"{name}.png", reader=NullLabelReader()
            )
            accounted = len(result.graph.nodes) + len(result.graph.unassigned_shapes)
            assert accounted == len(result.shapes)


class TestDeterminism:
    @pytest.mark.parametrize(
        "name", ["flowchart", "er", "sequence", "architecture", "ambiguous"]
    )
    def test_graph_is_byte_identical_across_runs(self, diagram_corpus, name):
        first = render(_graph(diagram_corpus, name), JSON)
        second = render(_graph(diagram_corpus, name), JSON)
        assert first == second

    def test_node_ids_are_stable(self, diagram_corpus):
        a = [n.id for n in _graph(diagram_corpus, "flowchart").nodes]
        b = [n.id for n in _graph(diagram_corpus, "flowchart").nodes]
        assert a == b


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------


class TestRenderers:
    def test_mermaid_flowchart(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "flowchart"), MERMAID)
        assert "flowchart TD" in out
        assert out.count("\n") > len(_graph(diagram_corpus, "flowchart").nodes)

    def test_mermaid_decision_uses_braces(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "flowchart"), MERMAID)
        assert "{" in out and "}" in out

    def test_mermaid_sequence(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "sequence"), MERMAID)
        assert out.startswith("sequenceDiagram")
        assert "participant" in out

    def test_mermaid_er(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "er"), MERMAID)
        assert "erDiagram" in out

    def test_plantuml_round_trips(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "flowchart"), PLANTUML)
        assert out.startswith("@startuml")
        assert out.rstrip().endswith("@enduml")

    def test_plantuml_uses_uml_shapes(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "flowchart"), PLANTUML)
        assert "{" in out and "}" in out  # decision

    def test_json_is_lossless(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        payload = json.loads(render(graph, JSON))
        assert payload["schema"] == "docweave.diagram/1.0"
        assert len(payload["nodes"]) == len(graph.nodes)
        assert len(payload["edges"]) == len(graph.edges)
        assert payload["kind"] == graph.kind.value

    def test_markdown_tables_every_node_and_edge(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        out = render(graph, MARKDOWN)
        assert "## Nodes" in out and "## Edges" in out
        for node in graph.nodes:
            assert f"| {node.id} |" in out

    def test_markdown_states_unread_labels(self, diagram_corpus):
        out = render(_graph(diagram_corpus, "flowchart"), MARKDOWN)
        assert "(not read)" in out

    def test_markdown_shows_inferred_directions(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        if any(edge.direction_inferred for edge in graph.edges):
            assert "inferred from layout" in render(graph, MARKDOWN)

    def test_mermaid_comments_uncertainty(self, diagram_corpus):
        graph = _graph(diagram_corpus, "flowchart")
        out = render(graph, MERMAID)
        if any(edge.direction_inferred for edge in graph.edges):
            assert "%%" in out

    def test_unknown_format_is_rejected(self, diagram_corpus):
        with pytest.raises(ValueError):
            render(_graph(diagram_corpus, "flowchart"), "dot")

    def test_labels_are_escaped_in_output(self, diagram_corpus, scratch):
        """A label containing Mermaid syntax must not break the output."""
        from docweave.diagram.graph import BoundingBox, DiagramGraph, DiagramNode, Label

        graph = DiagramGraph(
            kind=DiagramKind.FLOWCHART,
            nodes=[
                DiagramNode(
                    id="n0",
                    kind=NodeKind.PROCESS,
                    box=BoundingBox(x=0, y=0, width=10, height=10),
                    label=Label(text="evil [x] & more"),
                ),
                DiagramNode(
                    id="n1",
                    kind=NodeKind.PROCESS,
                    box=BoundingBox(x=100, y=0, width=10, height=10),
                ),
            ],
        )
        out = render(graph, MERMAID)
        assert "evil" in out
        assert out.count("[") >= 2  # only the node delimiters survive


# --------------------------------------------------------------------------
# resilience
# --------------------------------------------------------------------------


class TestResilience:
    def test_blank_image_yields_an_empty_graph(self, scratch):
        import numpy as np

        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        result = DiagramAnalyzer().analyze(blank, reader=NullLabelReader())
        assert result.graph.nodes == []
        assert not result.graph.edges

    def test_noise_does_not_crash(self):
        import numpy as np

        rng = np.random.default_rng(1234)
        noise = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        result = DiagramAnalyzer().analyze(noise, reader=NullLabelReader())
        assert result.graph.schema_version

    def test_tiny_image_is_handled(self):
        import numpy as np

        tiny = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = DiagramAnalyzer().analyze(tiny, reader=NullLabelReader())
        assert result.graph.width == 4

    def test_unreadable_file_raises_value_error(self, scratch):
        path = scratch / "not-an-image.png"
        path.write_bytes(b"this is not a png")
        with pytest.raises(ValueError):
            load_image(path)


# --------------------------------------------------------------------------
# optional OCR
# --------------------------------------------------------------------------


class TestNothingDetected:
    """An empty result must never look like a success.

    Found by pointing the engine at a real hand-drawn diagram: six filled
    pastel ellipses on a dotted background. It detected nothing and still
    printed "Diagram reconstructed with no uncertainties flagged" in green.
    """

    def test_an_unrecognisable_image_reports_an_error(self, scratch):
        import numpy as np

        path = scratch / "blank.png"
        import cv2

        cv2.imwrite(str(path), np.full((200, 300, 3), 255, np.uint8))
        result = analyze_file(path)
        codes = {d.code for d in result.diagnostics}
        assert Code.DIAGRAM_NOTHING_DETECTED in codes
        assert any(d.severity is Severity.ERROR for d in result.diagnostics)

    def test_a_filled_diagram_is_not_reported_as_clean(self, diagram_corpus):
        """The green all-clear must mean something was actually found."""
        from docweave.diagram.text import NullLabelReader

        result = analyze_file(diagram_corpus / "er.png", reader=NullLabelReader())
        assert result.graph.nodes, "fixture should yield nodes"

    def test_the_error_explains_what_was_looked_for(self, scratch):
        import cv2
        import numpy as np

        path = scratch / "empty.png"
        cv2.imwrite(str(path), np.full((150, 150, 3), 255, np.uint8))
        diagnostic = next(
            d
            for d in analyze_file(path).diagnostics
            if d.code == Code.DIAGRAM_NOTHING_DETECTED
        )
        assert diagnostic.reason
        assert diagnostic.hint
        assert "outline" in (diagnostic.hint or "").lower() or "shape" in (
            diagnostic.hint or ""
        ).lower()


class TestPdfInput:
    """A diagram wrapped in a PDF is an ordinary thing to produce.

    ``docweave diagram`` refused the file outright with exit 2 while
    ``docweave convert`` opened it perfectly well.
    """

    def test_a_pdf_diagram_is_analysed_not_refused(self, tmp_path):
        import pymupdf

        source = tmp_path / "diagram.pdf"
        document = pymupdf.open()
        page = document.new_page(width=400, height=300)
        page.draw_rect(pymupdf.Rect(40, 40, 160, 110), color=(0, 0, 0))
        page.draw_rect(pymupdf.Rect(240, 180, 360, 250), color=(0, 0, 0))
        page.draw_line((160, 75), (240, 215), color=(0, 0, 0))
        document.save(str(source))
        document.close()

        graph = analyze_file(source).graph
        assert len(graph.nodes) >= 2, "a PDF-wrapped diagram must be detected"
        assert "page1" in (graph.source_image or "")

    def test_a_missing_page_is_reported_not_silently_page_one(self, tmp_path):
        import pymupdf

        source = tmp_path / "one_page.pdf"
        document = pymupdf.open()
        document.new_page(width=200, height=200)
        document.save(str(source))
        document.close()

        with pytest.raises(ValueError, match="page 4 does not exist"):
            analyze_file(source, page=4)


class TestOptionalOcr:
    def test_tesseract_reader_reports_availability(self):
        from docweave.diagram.text import TesseractLabelReader

        available = TesseractLabelReader().available()
        assert isinstance(available, bool)

    def test_the_default_reader_is_tesseract_when_available(self):
        """Indra reads pages, not words; see ``default_label_reader``.

        Asserting the choice rather than just that *something* is returned,
        because the obvious "swap in the better engine" change is wrong here
        and the evidence is in that docstring.
        """
        from docweave.diagram.text import (
            NullLabelReader,
            TesseractLabelReader,
            default_label_reader,
        )

        reader = default_label_reader()
        if TesseractLabelReader().available():
            assert isinstance(reader, TesseractLabelReader)
        else:
            assert isinstance(reader, NullLabelReader)

    @requires_tesseract
    def test_labels_are_read_when_ocr_is_available(self, diagram_corpus):
        from docweave.diagram.text import default_label_reader

        reader = default_label_reader()
        graph = analyze_file(diagram_corpus / "flowchart.png", reader=reader).graph
        readable = [n for n in graph.nodes if n.label.readable]
        assert readable, "expected at least one label to be read"
        assert all(n.label.reader == reader.name for n in readable)

    @requires_tesseract
    def test_read_labels_populate_the_graph(self, diagram_corpus):
        from docweave.diagram.text import default_label_reader

        graph = analyze_file(
            diagram_corpus / "er.png", reader=default_label_reader()
        ).graph
        texts = {n.label.text.lower() for n in graph.nodes if n.label.readable}
        assert texts & {"customer", "order", "product", "orderline"}
