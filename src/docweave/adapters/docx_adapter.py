"""DOCX adapter (python-docx).

Body order is taken from the underlying ``w:body`` XML rather than from
``document.paragraphs``, because python-docx exposes paragraphs and tables as two
separate collections and concatenating them silently destroys the interleaving.
"""

from __future__ import annotations

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import clean_text, ref
from docweave.exceptions import MalformedDocumentError
from docweave.models import (
    Cell,
    CodeBlock,
    Confidence,
    Document,
    Heading,
    ImageBlock,
    ListBlock,
    ListItem,
    Locator,
    Origin,
    Paragraph,
    Quote,
    SourceInfo,
    Table,
)
from docweave.models.diagnostics import Category, Code

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class DocxAdapter(Adapter):
    name = "python-docx"
    formats = ("docx",)
    requires = {"docx": "python-docx"}
    capabilities = (
        "exact paragraph and table order",
        "heading hierarchy from paragraph styles",
        "list numbering from numPr",
        "hyperlinks (python-docx >= 1.1)",
        "inline and anchored images",
        "headers/footers per section",
        "footnotes and endnotes read from the package parts, in reference order",
        "core document metadata",
    )
    limitations = (
        "does not reconstruct text boxes, SmartArt or shapes",
        "does not evaluate fields; cached field results are kept as-is",
        "does not convert OMML equations to LaTeX",
        "note text is appended in its own section, not inlined at the reference",
    )
    cost = "moderate"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        try:
            import docx  # type: ignore
            from docx.opc.exceptions import PackageNotFoundError
        except ImportError as exc:
            from docweave.exceptions import DependencyMissingError

            raise DependencyMissingError("python-docx", "DOCX conversion", "office") from exc

        try:
            document = docx.Document(str(ctx.path))
        except PackageNotFoundError as exc:
            raise MalformedDocumentError(f"Invalid DOCX package: {exc}") from exc
        except Exception as exc:
            raise MalformedDocumentError(f"DOCX could not be opened: {exc}") from exc

        blocks: list = []
        # Embedded image parts are registered as assets here, but they are not
        # emitted as nodes: python-docx also surfaces each inline drawing, and
        # emitting both would put every image in the output twice.
        assets = _collect_image_parts(ctx, document)

        counters = {"heading": 0, "p": 0, "table": 0, "list": 0, "code": 0, "quote": 0}
        table_count = 0
        asset_cursor = {"next": 0}
        for element in document.element.body:
            tag = element.tag
            if tag == f"{_W_NS}p":
                node = self._paragraph_node(
                    ctx, document, element, counters, assets, asset_cursor
                )
                if node is not None:
                    if isinstance(node, ListBlock):
                        _merge_list(blocks, node)
                    else:
                        blocks.append(node)
            elif tag == f"{_W_NS}tbl":
                node = self._table_node(ctx, document, element)
                if node is not None:
                    blocks.append(node)
                    table_count += 1

        if ctx.config.include_headers_footers:
            blocks.extend(self._headers_footers(ctx, document))

        notes = self._notes(ctx, document)
        if notes:
            # Detected notes must not be discarded: they are appended as their
            # own section so the content survives even though the inline
            # anchors do not.
            blocks.append(
                Heading(
                    id=ctx.ids.child(None, "h"),
                    level=2,
                    text="Notes",
                    sources=[ref(self.name, Locator(kind="char"))],
                )
            )
            for text in notes:
                blocks.append(
                    Paragraph(
                        id=ctx.ids.child(None, "p"),
                        text=text,
                        sources=[
                            ref(
                                self.name,
                                Locator(kind="char", char_length=len(text)),
                                confidence=Confidence.MEDIUM,
                                note="Footnote/endnote appended without its inline anchor.",
                            )
                        ],
                    )
                )
        unsupported = _unsupported_summary(document)
        if unsupported:
            ctx.warn(
                Code.UNSUPPORTED_STRUCTURE,
                "Some DOCX structures were not converted: " + ", ".join(sorted(unsupported)) + ".",
                category=Category.UNVERIFIED,
                severity="info",
                reason="python-docx does not expose these object models.",
                evidence={"features": sorted(unsupported)},
            )

        props = document.core_properties
        meta = {
            "title": props.title,
            "author": props.author,
            "subject": props.subject,
            "created": props.created.isoformat() if props.created else None,
            "modified": props.modified.isoformat() if props.modified else None,
            "revision": props.revision,
            "table_count": table_count,
            "paragraph_count": counters["p"],
        }
        meta = {k: v for k, v in meta.items() if v not in (None, "")}

        doc_model = Document(
            adapter=self.name,
            container_kind="body",
            body=blocks,
            metadata=meta,
        )
        return ExtractionResult(
            document=doc_model,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )

    # -- block builders ----------------------------------------------------

    def _paragraph_node(
        self, ctx, document, element, counters: dict, assets: list, asset_cursor: dict
    ):
        from docx.text.paragraph import Paragraph as DocxParagraph

        paragraph = DocxParagraph(element, document)
        text, stats = clean_text(paragraph.text, strip_controls=False)
        style_name = (paragraph.style.name if paragraph.style is not None else "") or ""
        locator = Locator(kind="char", char_length=len(text))

        if not text.strip():
            if _has_drawing(element):
                return _image_node(ctx, element, text, assets, asset_cursor)
            return None

        provenance = ref(self.name, locator, confidence=Confidence.HIGH)

        heading_level = _heading_level(style_name)
        if heading_level:
            counters["heading"] += 1
            return Heading(
                id=ctx.ids.child(None, "h"),
                level=heading_level,
                text=text.strip(),
                warnings=_unicode_warnings(stats),
                sources=[provenance],
                meta={"style": style_name},
            )

        if _is_code_style(style_name):
            counters["code"] += 1
            return CodeBlock(
                id=ctx.ids.child(None, "code"),
                text=text,
                language=None,
                sources=[provenance],
                meta={"style": style_name},
            )

        if style_name.strip().lower().startswith("quote"):
            counters["quote"] += 1
            return Quote(id=ctx.ids.child(None, "quote"), text=text, sources=[provenance])

        list_kind = _list_kind(element, style_name)
        if list_kind:
            counters["list"] += 1
            return ListBlock(
                id=ctx.ids.child(None, "list"),
                ordered=list_kind == "ordered",
                items=[
                    ListItem(
                        id=ctx.ids.child(None, "li"),
                        text=text,
                        sources=[provenance],
                    )
                ],
                sources=[provenance],
                meta={"style": style_name},
            )

        counters["p"] += 1
        node = Paragraph(
            id=ctx.ids.child(None, "p"),
            text=text,
            warnings=_unicode_warnings(stats),
            sources=[provenance],
            meta={"style": style_name} if style_name else {},
        )
        hyperlinks = _hyperlinks(element)
        if hyperlinks:
            node.meta["links"] = hyperlinks
        return node

    def _table_node(self, ctx: ExtractionContext, document, element) -> Table | None:
        from docx.table import Table as DocxTable

        table = DocxTable(element, document)
        rows: list[list[Cell]] = []
        for r, row in enumerate(table.rows):
            cells: list[Cell] = []
            for c, cell in enumerate(row.cells):
                text, _ = clean_text(cell.text, strip_controls=False)
                grid_span = _grid_span(cell)
                cells.append(
                    Cell(
                        id=ctx.ids.child(None, "c"),
                        text=text,
                        row=r,
                        column=c,
                        is_header=(r == 0 and _row_is_header(row)),
                        column_span=grid_span,
                        locator=Locator(kind="char", char_length=len(text)),
                    )
                )
            rows.append(cells)
        if not rows:
            return None
        header_rows = 1 if any(c.is_header for c in rows[0]) else 0
        return Table(
            id=ctx.ids.child(None, "table"),
            text="",
            rows=rows,
            header_rows=header_rows,
            sources=[
                ref(
                    self.name,
                    Locator(kind="char"),
                    origin=Origin.OBSERVED,
                    confidence=Confidence.HIGH,
                    note="DOCX table; merged cells are expanded to the full grid.",
                )
            ],
        )

    def _headers_footers(self, ctx: ExtractionContext, document) -> list:
        from docweave.models import HeaderFooterBlock

        out: list = []
        seen: set[str] = set()
        for index, section in enumerate(document.sections, start=1):
            for part, role in (
                (getattr(section, "header", None), "header"),
                (getattr(section, "footer", None), "footer"),
            ):
                if part is None:
                    continue
                text = "\n".join(p.text for p in part.paragraphs).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(
                    HeaderFooterBlock(
                        id=ctx.ids.child(None, "hf"),
                        text=text,
                        role=role,
                        sources=[ref(self.name, Locator(kind="char", char_length=len(text)))],
                        meta={"section": index},
                    )
                )
        return out

    def _notes(self, ctx: ExtractionContext, document) -> list:
        """Footnote and endnote text, in the order the body references it.

        python-docx does not expose footnotes as an object model, so the
        package parts are read directly. Separator entries (id -1 and 0) are
        structural and are skipped.
        """
        texts = _read_note_parts(document)
        if not texts:
            return []

        order: list[str] = []
        for element in document.element.body.iter():
            if element.tag in (f"{_W_NS}footnoteReference", f"{_W_NS}endnoteReference"):
                note_id = element.get(f"{_W_NS}id")
                if note_id and note_id not in order and note_id in texts:
                    order.append(note_id)
        # Anything not referenced from the body still exists; do not drop it.
        for note_id in texts:
            if note_id not in order:
                order.append(note_id)

        resolved = [texts[n] for n in order if texts[n].strip()]
        if resolved:
            ctx.warn(
                Code.UNSUPPORTED_STRUCTURE,
                f"{len(resolved)} note(s) were read and appended as text, "
                "but their inline anchors were not reconstructed.",
                category=Category.UNVERIFIED,
                severity="info",
                reason="Note anchoring requires walking the body reference graph.",
                evidence={"note_count": len(resolved)},
            )
        return resolved


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _read_note_parts(document) -> dict[str, str]:
    """Extract ``{note_id: text}`` from ``word/footnotes.xml`` / ``endnotes.xml``."""
    out: dict[str, str] = {}
    try:
        parts = list(document.part.package.iter_parts())
    except Exception:
        return out

    for part in parts:
        name = str(getattr(part, "partname", ""))
        if not name.endswith(("footnotes.xml", "endnotes.xml")):
            continue
        try:
            from defusedxml.ElementTree import fromstring

            root = fromstring(part.blob)
        except Exception:
            continue
        tag = "footnote" if name.endswith("footnotes.xml") else "endnote"
        for node in root.findall(f"{_W_NS}{tag}"):
            note_id = node.get(f"{_W_NS}id")
            if note_id is None or int(note_id) < 1:
                continue  # -1 and 0 are the separator/continuation entries
            text, _ = clean_text(
                "".join(t.text or "" for t in node.iter(f"{_W_NS}t")), strip_controls=False
            )
            if text.strip():
                out[note_id] = text.strip()
    return out


def _collect_image_parts(ctx: ExtractionContext, document) -> list:
    """Register every embedded image part as an asset.

    Returns the registered references so inline drawings can link to them. No
    node is emitted here: the body walk produces one node per drawing, and
    emitting here as well would put every image in the output twice.
    """
    out: list = []
    try:
        rels = document.part.rels
    except Exception:
        return out
    for _rel_id, rel in sorted(rels.items()):
        if "image" not in rel.reltype:
            continue
        try:
            part = rel.target_part
            blob = part.blob
        except Exception:
            continue
        out.append(
            ctx.assets.add(
                blob if ctx.config.extract_assets else None,
                media_type=getattr(part, "content_type", None),
                locator=Locator(kind="char"),
                asset_id=ctx.ids.child(None, "asset"),
            )
        )
    return out


def _heading_level(style_name: str) -> int | None:
    name = style_name.strip().lower()
    if name.startswith("heading "):
        tail = name.split(" ", 1)[1]
        if tail.isdigit():
            return max(1, min(int(tail), 6))
    if name in ("title",):
        return 1
    if name.startswith("subtitle"):
        return 2
    return None


def _is_code_style(style_name: str) -> bool:
    name = style_name.strip().lower()
    return "code" in name or "preformatted" in name or "macro" in name


def _list_kind(element, style_name: str) -> str | None:
    ppr = element.find(f"{_W_NS}pPr")
    if ppr is not None and ppr.find(f"{_W_NS}numPr") is not None:
        numpr = ppr.find(f"{_W_NS}numPr")
        numid = numpr.find(f"{_W_NS}numId") if numpr is not None else None
        if numid is not None and numid.get(f"{_W_NS}val") not in (None, "0"):
            return "ordered" if _numid_is_ordered(element) else "unordered"
    name = style_name.strip().lower()
    if name.startswith("list number") or "numbered" in name:
        return "ordered"
    if name.startswith("list bullet") or "bullet" in name:
        return "unordered"
    return None


def _numid_is_ordered(element) -> bool:
    """Best-effort: a numbering format of 'bullet' means unordered."""
    numbering = getattr(element, "part", None)
    if numbering is None:
        return False
    try:
        npart = numbering.numbering_part  # type: ignore[attr-defined]
        num_id_el = element.find(f"{_W_NS}pPr/{_W_NS}numPr/{_W_NS}numId")
        if num_id_el is None:
            return False
        target = num_id_el.get(f"{_W_NS}val")
        for num in npart.element.findall(f"{_W_NS}num"):
            if num.get(f"{_W_NS}numId") != target:
                continue
            abstract = num.find(f"{_W_NS}abstractNumId")
            if abstract is None:
                continue
            aid = abstract.get(f"{_W_NS}val")
            for anum in npart.element.findall(f"{_W_NS}abstractNum"):
                if anum.get(f"{_W_NS}abstractNumId") != aid:
                    continue
                level = anum.find(f"{_W_NS}lvl")
                if level is not None:
                    fmt = level.find(f"{_W_NS}numFmt")
                    if fmt is not None and fmt.get(f"{_W_NS}val", "").startswith("bullet"):
                        return False
                return True
    except Exception:
        return False
    return False


def _row_is_header(row) -> bool:
    trpr = getattr(row._tr, "trPr", None)
    if trpr is None:
        return False
    return trpr.find(f"{_W_NS}tblHeader") is not None


def _grid_span(cell) -> int:
    tc_pr = getattr(cell._tc, "tcPr", None)
    if tc_pr is None:
        return 1
    span = tc_pr.find(f"{_W_NS}gridSpan")
    if span is None:
        return 1
    try:
        value = int(span.get(f"{_W_NS}val", "1"))
    except ValueError:
        return 1
    return max(1, value)


def _has_drawing(element) -> bool:
    return element.find(f".//{_W_NS}drawing") is not None or element.find(f".//{_W_NS}pict") is not None


def _image_node(ctx: ExtractionContext, element, text: str, assets: list, cursor: dict):
    """An inline drawing, linked to the asset registered for it.

    Linking by registration order is a deliberate simplification: python-docx
    exposes no stable mapping from a drawing to its image part. When the parts
    and the drawings disagree in count the node simply carries no asset, which
    is visible in the output rather than silently wrong.
    """
    index = cursor["next"]
    cursor["next"] += 1
    asset = assets[index] if index < len(assets) else None
    return ImageBlock(
        id=ctx.ids.child(None, "img"),
        text=text,
        asset_id=asset.asset_id if asset else None,
        media_type=asset.media_type if asset else None,
        sources=[
            ref(
                "python-docx",
                Locator(kind="char"),
                origin=Origin.OBSERVED,
                confidence=Confidence.HIGH,
                note="Inline drawing in the document body.",
            )
        ],
    )


def _hyperlinks(element) -> list[str]:
    out: list[str] = []
    for link in element.findall(f".//{_W_NS}hyperlink"):
        rid = link.get(f"{_R_NS}id")
        anchor = link.get(f"{_W_NS}anchor")
        if rid:
            out.append(f"rId:{rid}")
        elif anchor:
            out.append(f"#{anchor}")
    return out


def _merge_list(blocks: list, node: ListBlock) -> None:
    """Fold consecutive list paragraphs into one list block."""
    if blocks and isinstance(blocks[-1], ListBlock) and blocks[-1].ordered == node.ordered:
        blocks[-1].items.extend(node.items)
        return
    blocks.append(node)


def _unicode_warnings(stats) -> list[str]:
    return ["replacement-characters"] if stats.replacement else []


def _unsupported_summary(document) -> set[str]:
    found: set[str] = set()
    body_xml = document.element.body.xml
    if "w:txbxContent" in body_xml or "v:textbox" in body_xml:
        found.add("text boxes")
    if "a:graphicData" in body_xml and "diagram" in body_xml.lower():
        found.add("SmartArt")
    if "w:object" in body_xml or "w:oleObject" in body_xml:
        found.add("embedded OLE objects")
    if "m:oMath" in body_xml:
        found.add("OMML equations")
    if "w:pict" in body_xml:
        found.add("VML images")
    return found
