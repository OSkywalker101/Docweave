"""Diagram engine integration with document conversion.

When ``ConversionConfig.diagrams`` is enabled, raster images extracted from a
PDF, DOCX or PPTX are handed to the diagram engine and their reconstructed
graphs are attached to the document. Images that are photographs, logos or
screenshots produce no meaningful graph; those are reported rather than forced.

This is opt-in: it costs OpenCV plus a reconstruction pass per image, and most
documents do not need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from docweave.models import AssetRef, Diagnostic
from docweave.models.diagnostics import Category, Code, Severity

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime only
    from docweave.diagram.analyze import DiagramResult

#: Images smaller than this are icons, not diagrams.
MIN_DIAGRAM_PIXELS = 120
MIN_DIAGRAM_AREA = 40_000


class DiagramIntegrator:
    """Runs the diagram engine over a document's extracted images."""

    def __init__(
        self,
        *,
        reader=None,
        min_pixels: int = MIN_DIAGRAM_PIXELS,
        classify_images: bool = False,
    ) -> None:
        from docweave.diagram import default_reader

        self.reader = reader if reader is not None else default_reader()
        self.min_pixels = min_pixels
        self.classify_images = classify_images

    def should_analyze(self, data: bytes | None, media_type: str | None) -> bool:
        """Cheap gate: is this raster image plausibly a diagram?

        ``data`` may be absent: the asset was registered without its bytes when
        asset extraction is disabled, and the ``if not data`` guard below is
        exactly that case rather than an accident.
        """
        if not data or not media_type or not media_type.startswith("image/"):
            return False
        if media_type in ("image/gif", "image/webp"):
            return False
        width, height = _dimensions(data)
        if width is None or height is None:
            return False
        if width < self.min_pixels or height < self.min_pixels:
            return False
        return width * height >= MIN_DIAGRAM_AREA

    def classify_asset(self, asset: AssetRef):
        """Classify an extracted image.

        Returns ``(classification, decoded_image)``; both are ``None`` when the
        asset carries no readable bytes. The decoded image is returned so the
        caller does not have to decode it a second time to reconstruct it.
        """
        from docweave.visual import classify_image

        data = _read_asset(asset)
        if data is None:
            return None, None
        image = _decode(data)
        if image is None:
            return None, None
        return classify_image(image), image

    def analyze_asset(self, asset: AssetRef) -> DiagramResult | None:
        """Return a :class:`DiagramResult`, or ``None`` if it is not a diagram."""
        from docweave.diagram import analyze_image

        data = _read_asset(asset)
        if data is None:
            return None
        image = _decode(data)
        if image is None:
            return None

        classification = None
        if self.classify_images:
            from docweave.visual import classify_image

            try:
                classification = classify_image(image)
            except Exception:
                # The classifier is a gate, not a requirement: a failure to
                # measure the image must not stop reconstruction. The engine
                # still runs below and reports honestly what it found.
                classification = None
            else:
                if classification.needs_vision_model:
                    # Geometry cannot read a photograph or a screenshot. Saying
                    # so is more useful than an empty graph that looks like a
                    # failed reconstruction.
                    return _skipped_result(asset.asset_id, classification)

        result = analyze_image(image, source_image=asset.asset_id, reader=self.reader)
        result.classification = classification
        return result


def _read_asset(asset: AssetRef) -> bytes | None:
    if asset.path:
        try:
            return Path(asset.path).read_bytes()
        except OSError:
            return None
    return None


def _decode(data: bytes):
    """Decode image bytes to a BGR array, or ``None`` when unreadable."""
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _skipped_result(asset_id: str, classification) -> DiagramResult:
    """A result that records *why* reconstruction was skipped.

    Deliberately not ``None``: ``None`` means "not a diagram, move along", and
    silently dropping a photograph the classifier correctly identified would
    hide the one decision this whole pipeline exists to make.
    """
    from docweave.diagram.analyze import DiagramResult
    from docweave.diagram.graph import DiagramGraph

    evidence = ", ".join(
        f"{key}={value}"
        for key, value in list(classification.evidence.items())[:4]
    )
    diagnostic = Diagnostic(
        code=Code.DIAGRAM_SKIPPED_NON_DIAGRAM,
        category=Category.UNVERIFIED,
        severity=Severity.INFO,
        message=(
            f"Diagram reconstruction skipped for '{asset_id}': classified as "
            f"{classification.kind.value}, not a diagram."
        ),
        reason=f"Visual classifier measured: {evidence or 'no signals'}.",
        hint=(
            "The original image is preserved. Reading it would need a vision "
            "model, which this build deliberately does not ship."
        ),
        scope_kind="diagram",
        scope_name=asset_id,
        evidence={
            "classification": classification.kind.value,
            "confidence": classification.confidence,
            "needs_vision_model": classification.needs_vision_model,
        },
    )
    return DiagramResult(
        graph=DiagramGraph(width=0, height=0, source_image=asset_id),
        diagnostics=[diagnostic],
        classification=classification,
    )


def _dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Read pixel dimensions from image bytes without a full decode."""
    import struct

    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 24:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB):
                height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
                return int(width), int(height)
            if offset + 4 > len(data):
                break
            offset += 2 + struct.unpack(">H", data[offset + 2 : offset + 4])[0]
    return None, None


def diagram_diagnostics(result, *, asset_id: str) -> list[Diagnostic]:
    """Convert a diagram result into document-level diagnostics."""
    out: list[Diagnostic] = []
    graph = result.graph
    for diagnostic in result.diagnostics:
        out.append(
            Diagnostic(
                code=diagnostic.code,
                category=diagnostic.category,
                severity=diagnostic.severity,
                message=f"Diagram asset '{asset_id}': {diagnostic.message}",
                reason=diagnostic.reason,
                hint=diagnostic.hint,
                scope_kind="diagram",
                scope_name=asset_id,
                evidence=diagnostic.evidence,
            )
        )
    if not graph.edges and len(graph.nodes) > 1:
        out.append(
            Diagnostic(
                code=Code.DIAGRAM_UNCONNECTED,
                category=Category.UNVERIFIED,
                severity=Severity.WARNING,
                message=(
                    f"Diagram asset '{asset_id}' yielded {len(graph.nodes)} shapes but no edges; "
                    "its relationships were not reconstructed."
                ),
                reason="No line terminated cleanly on two shapes.",
                hint="The source image is preserved; inspect it directly.",
                scope_kind="diagram",
                scope_name=asset_id,
                evidence={"nodes": len(graph.nodes), "edges": 0},
            )
        )
    return out
