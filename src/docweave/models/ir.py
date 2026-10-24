"""Typed canonical intermediate representation (IR).

Design goals
------------
1. Structure survives extraction. Tables are tables, spreadsheet cells keep their
   coordinates, slides keep their identity. Markdown is a *rendering* of this
   model, never the storage format.
2. Provenance is mandatory. Every node records where it came from and whether it
   was observed directly or reconstructed from geometry/structure.
3. Confidence is an explicit *band*, not a probability. The model never pretends
   to be statistically calibrated.
4. Unresolved content is representable. ``Origin.ABSENT`` marks content we know
   we could not read, so it can be reported instead of silently dropped.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

IR_SCHEMA_VERSION = "1.0"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Origin(str, Enum):
    """How a piece of content came to exist.

    ``OBSERVED``      read directly from the source bytes / object model.
    ``RECONSTRUCTED`` rebuilt from structure or geometry (e.g. table cells from
                      ruling lines, reading order from bounding boxes).
    ``INFERRED``      produced by a heuristic (e.g. heading level guessed from
                      relative font size).
    ``ABSENT``        known to exist in the source but not recovered.
    """

    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    INFERRED = "inferred"
    ABSENT = "absent"


class Confidence(str, Enum):
    """Heuristic confidence band. Not a calibrated probability."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

    @property
    def rank(self) -> int:
        return _CONFIDENCE_RANK[self]


_CONFIDENCE_RANK = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1, Confidence.NONE: 0}


# --------------------------------------------------------------------------
# Locators / provenance
# --------------------------------------------------------------------------


class Locator(_Base):
    """Where a node lives in its source document.

    Only the fields relevant to a format are populated. Spreadsheets use
    ``sheet``/``cell``, presentations use ``slide``, paginated formats use
    ``page``/``bbox``, text formats use ``char_offset``, HTML uses ``css_path``.
    """

    kind: str = "none"  # none | page | slide | sheet | char | html
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    slide: int | None = None
    shape_id: str | None = None
    sheet: str | None = None
    cell: str | None = None
    row: int | None = None
    column: int | None = None
    char_offset: int | None = None
    char_length: int | None = None
    css_path: str | None = None


class SourceRef(_Base):
    """Provenance for a single node."""

    adapter: str
    origin: Origin = Origin.OBSERVED
    confidence: Confidence = Confidence.HIGH
    locator: Locator = Field(default_factory=Locator)
    note: str | None = None


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


class Node(_Base):
    """Base for every content node.

    ``id`` is deterministic: derived from the structural path during assembly
    (``p3-b2-t1-r0c2``) so two runs over the same input emit the same ids.
    """

    id: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def origin(self) -> Origin:
        """The weakest provenance of any source.

        A node with one observed and one inferred source is, from a consumer's
        point of view, inferred: the weakest link decides what we may claim.
        """
        if not self.sources:
            return Origin.INFERRED
        return max((s.origin for s in self.sources), key=lambda o: _ORIGIN_RANK[o])

    @property
    def confidence(self) -> Confidence:
        """The lowest confidence band of any source (again: weakest link)."""
        if not self.sources:
            return Confidence.LOW
        return min((s.confidence for s in self.sources), key=lambda c: c.rank)

    def text_of(self) -> str:
        return getattr(self, "text", "") or ""


_ORIGIN_RANK = {
    Origin.OBSERVED: 0,
    Origin.RECONSTRUCTED: 1,
    Origin.INFERRED: 2,
    Origin.ABSENT: 3,
}


class Block(Node):
    """A node that carries text."""

    text: str = ""


class Paragraph(Block):
    pass


class Heading(Block):
    level: int = 1


class CodeBlock(Block):
    language: str | None = None


class Quote(Block):
    pass


class ListItem(Block):
    ordinal: int | None = None


class ListBlock(Node):
    ordered: bool = False
    items: list[ListItem] = Field(default_factory=list)


class Cell(_Base):
    id: str = ""
    text: str = ""
    row: int = 0
    column: int = 0
    is_header: bool = False
    row_span: int = 1
    column_span: int = 1
    formula: str | None = None
    #: Cached formula result when the source file had one. Never computed by us.
    value: str | None = None
    number_format: str | None = None
    locator: Locator | None = None
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reference(self) -> str | None:
        """A1-style address when the source format has one (spreadsheets)."""
        return self.locator.cell if self.locator else None

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_reference(cls, data: Any) -> Any:
        """Let the model round-trip its own JSON output.

        ``reference`` is derived, so re-validating a dumped cell must not look
        like an unknown field. Everything else stays strictly forbidden.
        """
        if isinstance(data, dict) and "reference" in data:
            data = {k: v for k, v in data.items() if k != "reference"}
        return data


class Table(Block):
    rows: list[list[Cell]] = Field(default_factory=list)
    header_rows: int = 0
    caption: str | None = None
    structure: str = "grid"  # grid | keyvalue | flow
    #: True when rows are *expected* to differ in width because each cell keeps
    #: its own coordinates (spreadsheet sheets). Raggedness is then data, not a
    #: defect, so the validator does not report it.
    sparse: bool = False

    def column_count(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def is_rectangular(self) -> bool:
        if not self.rows:
            return True
        width = len(self.rows[0])
        return all(len(r) == width for r in self.rows)


class ImageBlock(Block):
    """An embedded or referenced image.

    ``asset_id`` links to :class:`AssetRef` on the document when assets were
    extracted. ``alt`` is only populated when the source provided a description;
    it is never invented.
    """

    asset_id: str | None = None
    media_type: str | None = None
    width: int | None = None
    height: int | None = None
    alt: str | None = None


class Equation(Block):
    """Mathematical content. ``latex`` is populated only when actually recovered."""

    latex: str | None = None
    asset_id: str | None = None


class HeaderFooterBlock(Block):
    role: str = "header"  # header | footer


class Container(Node):
    """A node that owns ordered children (sections, slides' shape groups)."""

    children: list[Node] = Field(default_factory=list)


class Section(Container):
    title: str | None = None
    level: int | None = None


#: Node types the renderers and validator understand.
CONTENT_NODE_TYPES = (
    Paragraph,
    Heading,
    CodeBlock,
    Quote,
    ListBlock,
    ListItem,
    Table,
    ImageBlock,
    Equation,
    HeaderFooterBlock,
    Section,
)

AnyNode = Union[
    Paragraph,
    Heading,
    CodeBlock,
    Quote,
    ListBlock,
    ListItem,
    Table,
    ImageBlock,
    Equation,
    HeaderFooterBlock,
    Section,
]

Node.model_rebuild()
Container.model_rebuild()
Section.model_rebuild()
ListBlock.model_rebuild()


# --------------------------------------------------------------------------
# Format specific containers
# --------------------------------------------------------------------------


class PageSignals(_Base):
    """Cheap, per-page observations collected during extraction."""

    char_count: int = 0
    block_count: int = 0
    image_count: int = 0
    link_count: int = 0
    replacement_chars: int = 0
    control_chars: int = 0
    table_count: int = 0
    column_count: int = 1
    rotation: int = 0
    media_area_ratio: float = 0.0
    text_area_ratio: float = 0.0
    font_size_median: float | None = None
    inferred_heading_sizes: list[float] = Field(default_factory=list)


class Page(_Base):
    index: int  # 1-based
    width: float | None = None
    height: float | None = None
    rotation: int = 0
    blocks: list[AnyNode] = Field(default_factory=list)
    images: list[ImageBlock] = Field(default_factory=list)
    signals: PageSignals = Field(default_factory=PageSignals)


class Slide(_Base):
    index: int  # 1-based
    layout: str | None = None
    title: str | None = None
    blocks: list[AnyNode] = Field(default_factory=list)
    notes: list[AnyNode] = Field(default_factory=list)
    images: list[ImageBlock] = Field(default_factory=list)


class Worksheet(_Base):
    name: str
    index: int  # 1-based
    hidden: bool = False
    dimension: str | None = None
    #: The sheet's grid. Cells carry A1 references, so sparse sheets stay exact.
    table: Table | None = None
    #: Any non-tabular content attached to the sheet (kept for other formats).
    blocks: list[AnyNode] = Field(default_factory=list)
    formula_count: int = 0
    merged_range_count: int = 0
    truncated: bool = False


class AssetRef(_Base):
    """A binary asset discovered in the source document."""

    asset_id: str
    media_type: str | None = None
    extension: str | None = None
    byte_size: int | None = None
    sha256: str | None = None
    path: str | None = None
    extracted: bool = False
    locator: Locator = Field(default_factory=Locator)
    description: str | None = None


class SourceInfo(_Base):
    """Identity of the input, independent of any filesystem path leakage."""

    name: str = ""
    byte_size: int = 0
    sha256: str = ""
    detected_format: str = "unknown"
    detection_method: str = "signature"
    detection_confidence: float = 0.0
    declared_extension: str | None = None
    extension_mismatch: bool = False
    encrypted: bool = False


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


class Document(_Base):
    """The canonical document representation."""

    schema_version: str = IR_SCHEMA_VERSION
    source: SourceInfo = Field(default_factory=SourceInfo)
    adapter: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Ordered content for formats without pagination/containers (txt, md,
    #: html, csv, json, xml).
    body: list[AnyNode] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    slides: list[Slide] = Field(default_factory=list)
    sheets: list[Worksheet] = Field(default_factory=list)
    assets: list[AssetRef] = Field(default_factory=list)
    container_kind: str = "body"  # body | pages | slides | sheets

    # -- traversal ---------------------------------------------------------

    def owned_containers(self) -> Iterator[tuple[str, Any]]:
        if self.container_kind == "pages":
            for page in self.pages:
                yield "page", page
        elif self.container_kind == "slides":
            for slide in self.slides:
                yield "slide", slide
        elif self.container_kind == "sheets":
            for sheet in self.sheets:
                yield "sheet", sheet

    def iter_nodes(self) -> Iterator[AnyNode]:
        """Yield every node in canonical reading order."""
        owners = list(self.owned_containers())
        if not owners:
            yield from self.body
            return
        for _, container in owners:
            yield from _walk(container.blocks)
            if isinstance(container, Slide):
                yield from _walk(container.notes)
            elif isinstance(container, Worksheet) and container.table is not None:
                yield container.table

    def iter_nodes_with_owner(self) -> Iterator[tuple[str, Any, AnyNode]]:
        owners = list(self.owned_containers())
        if not owners:
            for node in self.body:
                yield "body", None, node
            return
        for kind, container in owners:
            for node in _walk(container.blocks):
                yield kind, container, node
            if isinstance(container, Slide):
                for node in _walk(container.notes):
                    yield "notes", container, node
            elif isinstance(container, Worksheet) and container.table is not None:
                yield kind, container, container.table

    def node_count(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    def tables(self) -> Iterator[Table]:
        for node in self.iter_nodes():
            if isinstance(node, Table):
                yield node

    def images(self) -> Iterator[ImageBlock]:
        for node in self.iter_nodes():
            if isinstance(node, ImageBlock):
                yield node
        for page in self.pages:
            yield from page.images
        for slide in self.slides:
            yield from slide.images

    def text_length(self) -> int:
        """Total extracted characters, counting table cell content.

        A document that is mostly tables has almost no ``node.text``; counting
        only that would make a perfectly good table-only conversion look empty.
        """
        total = 0
        for node in self.iter_nodes():
            if isinstance(node, Table):
                total += sum(len(c.text) for row in node.rows for c in row)
            else:
                total += len(node.text_of())
        return total

    def digest(self) -> str:
        """Content hash of the extracted text, independent of asset ordering."""
        h = hashlib.sha256()
        h.update(f"{self.adapter}:{self.container_kind}".encode())
        for node in self.iter_nodes():
            h.update(b"\x1e")
            h.update(node.id.encode("utf-8"))
            h.update(b"\x1f")
            if isinstance(node, Table):
                for row in node.rows:
                    for cell in row:
                        h.update(cell.text.encode("utf-8"))
                        h.update(b"\x1f")
            else:
                h.update(node.text_of().encode("utf-8"))
        return h.hexdigest()


def _walk(nodes: list[AnyNode]) -> Iterator[AnyNode]:
    """Yield nested nodes. Table cells are reached via ``Table.rows``, not here,
    because they are attributes of the table rather than reading-order nodes."""
    for node in nodes:
        yield node
        if isinstance(node, Section):
            yield from _walk(node.children)
        elif isinstance(node, ListBlock):
            yield from node.items
