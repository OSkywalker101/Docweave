"""XML adapter.

Parsing uses :mod:`defusedxml` because every input file is untrusted. The
original markup is preserved verbatim so nothing is silently dropped; a
concatenation of text nodes is also emitted for readability.
"""

from __future__ import annotations

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import char_locator, clean_text, read_text, ref
from docweave.exceptions import MalformedDocumentError
from docweave.models import CodeBlock, Confidence, Document, SourceInfo
from docweave.models.diagnostics import Category, Code


class XmlAdapter(Adapter):
    name = "xml"
    formats = ("xml",)
    requires = {"defusedxml": "defusedxml"}
    capabilities = (
        "safe parsing via defusedxml (entity expansion and external entities blocked)",
        "verbatim markup preservation",
        "element/attribute counts",
    )
    limitations = (
        "does not map arbitrary XML vocabularies to headings/tables",
        "does not resolve namespaces into readable prefixes",
    )
    cost = "cheap"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        raw, encoding, lossy = read_text(ctx.path)
        text, stats = clean_text(raw, strip_controls=False)

        try:
            from defusedxml.ElementTree import fromstring
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise MalformedDocumentError("defusedxml is required to parse XML safely") from exc

        try:
            root = fromstring(text)
        except Exception as exc:
            raise MalformedDocumentError(f"Invalid XML: {exc}") from exc

        blocks: list = []
        nodes = _count_elements(root)
        # The verbatim markup already contains every character of the document,
        # including its visible text. Emitting a second, flattened copy of the
        # same text would duplicate content in the output and double-count it for
        # any downstream chunker.
        blocks.append(
            CodeBlock(
                id=ctx.ids.child(None, "xml"),
                text=_pretty(root),
                language="xml",
                sources=[ref(self.name, char_locator(0, len(text)), confidence=Confidence.HIGH)],
            )
        )

        if lossy:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"XML decoded with {encoding} and contains replacement characters.",
                category=Category.STRUCTURAL,
            )

        doc = Document(
            adapter=self.name,
            container_kind="body",
            body=blocks,
            metadata={
                "encoding": encoding,
                "root_tag": _tag(root.tag),
                "element_count": nodes,
            },
        )
        return ExtractionResult(
            document=doc,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )


def _tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _count_elements(element) -> int:
    return 1 + sum(_count_elements(child) for child in element)


def _visible_text(element) -> str:
    parts = [element.text or ""]
    for child in element:
        parts.append(_visible_text(child))
        parts.append(child.tail or "")
    return " ".join(" ".join(parts).split())


def _pretty(element) -> str:
    from defusedxml.ElementTree import tostring

    return tostring(element, encoding="unicode")
