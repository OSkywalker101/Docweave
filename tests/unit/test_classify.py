"""Visual classifier: the gate in front of the diagram engine.

Every decision must ship the numbers that produced it, and an image that cannot
be classified must say so rather than being coerced into the nearest category.
These tests hold the classifier to that, and one of them is a regression test for
a swallowed ``NameError`` that made the shape count silently zero.
"""

from __future__ import annotations

import sys
from unittest import mock

from docweave.visual import VisualKind, classify_image
from docweave.visual.classify import VisualClassification, _count_shapes
from tests.conftest import requires_diagrams, requires_docx

pytestmark = requires_diagrams


def _gradient(height: int = 300, width: int = 300):
    import numpy as np

    xs = np.linspace(0, 255, width)
    ys = np.linspace(0, 255, height)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx, yy, (xx + yy) / 2.0], axis=-1).astype(np.uint8)


class TestDiagramDetection:
    def test_real_diagram_is_a_diagram(self, diagram_corpus):
        import cv2

        image = cv2.imread(str(diagram_corpus / "flowchart.png"), cv2.IMREAD_COLOR)
        classification = classify_image(image)
        assert classification.kind is VisualKind.DIAGRAM
        assert classification.is_diagram is True
        assert classification.needs_vision_model is False
        assert classification.reasons, "a decision must explain itself"

    def test_count_shapes_is_not_silently_zero(self, diagram_corpus):
        """Regression: ``_count_shapes`` used ``np`` without importing it.

        The broad ``except Exception`` turned that ``NameError`` into ``0``, so
        the diagram rule (``shapes >= 2``) could never fire on the pixels alone.
        """
        import cv2

        image = cv2.imread(str(diagram_corpus / "flowchart.png"), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        assert _count_shapes(gray) >= 2


class TestFallbacks:
    def test_numpy_missing_returns_unknown(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            classification = classify_image(b"not even an image")
        assert classification.kind is VisualKind.UNKNOWN
        assert classification.confidence == 0.0
        assert any("numpy" in reason for reason in classification.reasons)

    def test_photo_needs_vision_model(self):
        classification = classify_image(_gradient())
        assert classification.kind in (VisualKind.PHOTO, VisualKind.SCREENSHOT)
        assert classification.needs_vision_model is True

    def test_screenshot_from_many_blocks(self):
        import numpy as np

        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        classification = classify_image(image, text_blocks=8)
        assert classification.kind is VisualKind.SCREENSHOT
        assert classification.needs_vision_model is True

    def test_text_page_from_density(self):
        import numpy as np

        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        classification = classify_image(image, ocr_text="word " * 40)
        assert classification.kind is VisualKind.TEXT_PAGE

    def test_unknown_keeps_its_evidence(self):
        import numpy as np

        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        classification = classify_image(image)
        assert classification.kind is VisualKind.UNKNOWN
        assert classification.reasons
        # The measurements that led to "unknown" are preserved, not discarded.
        for key in ("width", "height", "ink_fraction", "colourfulness", "shapes"):
            assert key in classification.evidence


class TestClassificationContract:
    def test_decision_is_auditable(self):
        classification = classify_image(_gradient())
        assert isinstance(classification, VisualClassification)
        assert 0.0 <= classification.confidence <= 1.0
        assert classification.reasons
        assert classification.evidence


class TestPipelineGate:
    """The classifier must change what the pipeline *does*, not just what it says."""

    def _asset(self, tmp_path, name):
        import cv2

        from docweave.models import AssetRef

        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), _gradient())
        return AssetRef(asset_id=name, media_type="image/png", path=str(path))

    def test_gate_skips_a_photo(self, tmp_path):
        from docweave.diagram.integrate import DiagramIntegrator
        from docweave.diagram.text import NullLabelReader
        from docweave.models.diagnostics import Code

        asset = self._asset(tmp_path, "photo")

        off = DiagramIntegrator(classify_images=False, reader=NullLabelReader())
        result_off = off.analyze_asset(asset)
        assert result_off is not None
        assert result_off.classification is None  # the gate was not consulted

        on = DiagramIntegrator(classify_images=True, reader=NullLabelReader())
        result_on = on.analyze_asset(asset)
        assert result_on is not None
        assert result_on.classification is not None
        assert result_on.classification.needs_vision_model is True
        assert {d.code for d in result_on.diagnostics} >= {Code.DIAGRAM_SKIPPED_NON_DIAGRAM}
        # Nothing was reconstructed, and the report says why.
        assert result_on.graph.nodes == []

    def test_gate_lets_a_diagram_through(self, diagram_corpus):
        from docweave.diagram.integrate import DiagramIntegrator
        from docweave.diagram.text import NullLabelReader
        from docweave.models import AssetRef

        asset = AssetRef(
            asset_id="flow",
            media_type="image/png",
            path=str(diagram_corpus / "flowchart.png"),
        )
        integrator = DiagramIntegrator(classify_images=True, reader=NullLabelReader())
        result = integrator.analyze_asset(asset)
        assert result is not None
        assert result.classification is not None
        assert result.classification.kind is VisualKind.DIAGRAM
        assert result.graph.nodes, "a confirmed diagram must still be reconstructed"


class TestConverterWiring:
    """The gate must be reachable from the real ``convert`` entry point."""

    @requires_docx
    def test_convert_records_the_classification(self, tmp_path):
        import cv2
        from docx import Document

        from docweave import convert
        from docweave.config import ConversionConfig

        png = tmp_path / "photo.png"
        cv2.imwrite(str(png), _gradient(600, 600))
        docx = tmp_path / "photo.docx"
        document = Document()
        document.add_picture(str(png))
        document.save(str(docx))

        config = ConversionConfig(
            detect_diagrams=True,
            classify_images=True,
            extract_assets=True,
            assets_dir=tmp_path / "assets",
        )
        result = convert(str(docx), config=config)

        entries = result.document.metadata.get("diagrams") or {}
        assert entries, "the embedded image should have been classified"
        (entry,) = entries.values()
        assert entry["classification"] in ("photo", "screenshot")
        assert entry["nodes"] == 0  # geometry was skipped, and said so


