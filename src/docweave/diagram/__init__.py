"""Diagram understanding engine.

Reconstructs the *structure and meaning* of a diagram — nodes, edges,
directions and labels — rather than dumping the text drawn on it.

    from docweave.diagram import analyze_file

    result = analyze_file("architecture.png")
    print(result.graph.kind)
    print(result.graph.to_markdown())

Design commitments:

* Deterministic geometric detection. No machine learning is involved, so the
  same image always yields the same graph and every classification can be traced
  to the measurements that produced it.
* Structure first, text second. Text *regions* are always located; reading them
  requires an optional OCR engine and its absence never costs us the graph.
* Never invent a relationship. An edge is published only when a line terminates
  on two distinct shapes. Anything else is reported as unassigned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docweave.config import ConversionConfig
from docweave.diagram.analyze import DiagramAnalyzer, DiagramResult
from docweave.diagram.graph import (
    DIAGRAM_SCHEMA_VERSION,
    BoundingBox,
    DiagramEdge,
    DiagramGraph,
    DiagramKind,
    DiagramNode,
    EdgeStyle,
    Label,
    NodeKind,
)
from docweave.diagram.render import (
    DIAGRAM_FORMATS,
    JSON,
    MARKDOWN,
    MERMAID,
    PLANTUML,
    render,
)

__all__ = [
    "DIAGRAM_SCHEMA_VERSION",
    "DIAGRAM_FORMATS",
    "MERMAID",
    "PLANTUML",
    "JSON",
    "MARKDOWN",
    "DiagramAnalyzer",
    "DiagramResult",
    "DiagramGraph",
    "DiagramNode",
    "DiagramEdge",
    "DiagramKind",
    "NodeKind",
    "EdgeStyle",
    "Label",
    "BoundingBox",
    "analyze_file",
    "analyze_image",
    "render",
    "load_image",
    "default_reader",
]


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR, raising a clear error for anything unreadable."""
    import cv2

    data = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"could not read image: {Path(path).name}")
    return data


def default_reader(ocr: bool = True):
    """Pick a label reader.

    With ``ocr=False`` (or when no engine is installed) the null reader is used:
    text regions are still located, and the graph is unaffected.
    """
    if not ocr:
        from docweave.diagram.text import NullLabelReader

        return NullLabelReader()
    from docweave.diagram.text import TesseractLabelReader

    reader = TesseractLabelReader()
    return reader if reader.available() else reader


def analyze_image(
    image: np.ndarray,
    *,
    source_image: str | None = None,
    config: ConversionConfig | None = None,
    reader=None,
) -> DiagramResult:
    """Reconstruct a graph from an already-loaded image."""
    if reader is None:
        reader = default_reader()
    return DiagramAnalyzer(config).analyze(
        image, source_image=source_image, reader=reader
    )


def analyze_file(
    path: str | Path,
    *,
    config: ConversionConfig | None = None,
    reader=None,
    dpi: int = 200,
    page: int | None = None,
) -> DiagramResult:
    """Reconstruct a graph from an image or a PDF page on disk.

    A PDF is accepted because wrapping a diagram in one is the most ordinary
    thing a person can do -- exporting from a browser, from a document, or from
    any "save as PDF" menu -- and having the command refuse the file while
    ``docweave convert`` opens it perfectly well was a plain inconsistency.

    The page is rasterised rather than the embedded image being extracted: a
    diagram in a PDF has usually been scaled, rotated or re-encoded, and the
    pixel dimensions the detector sees should be the ones a reader would.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf" or _looks_like_pdf(path):
        return _analyze_pdf(path, config=config, reader=reader, dpi=dpi, page=page)
    image = load_image(path)
    return analyze_image(image, source_image=path.name, config=config, reader=reader)


def _looks_like_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _analyze_pdf(
    path: Path, *, config, reader, dpi: int, page: int | None
) -> DiagramResult:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ValueError(
            f"{path.name} is a PDF; reading one needs the pdf extra "
            "(pip install 'docweave[pdf]')"
        ) from exc

    with pymupdf.open(str(path)) as document:
        index = 0 if page is None else max(0, page - 1)
        if index >= document.page_count:
            raise ValueError(
                f"{path.name} has {document.page_count} page(s); page {index + 1} "
                "does not exist"
            )
        pixmap = document[index].get_pixmap(dpi=dpi)
        import numpy as np

        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        name = f"{path.name}#page{index + 1}"
    return analyze_image(image, source_image=name, config=config, reader=reader)


def analyze_bytes(
    data: bytes,
    *,
    source_image: str | None = None,
    config: ConversionConfig | None = None,
    reader=None,
) -> DiagramResult:
    """Reconstruct a graph from image bytes.

    This is the entry point for images already extracted from a PDF, DOCX or
    PPTX, where only the byte stream exists.
    """
    import cv2
    import numpy as np

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return analyze_image(image, source_image=source_image, config=config, reader=reader)
