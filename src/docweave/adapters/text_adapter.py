"""Plain text adapter."""

from __future__ import annotations

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.adapters.common import char_locator, clean_text, read_text, ref
from docweave.models import (
    Confidence,
    Document,
    Paragraph,
    SourceInfo,
)
from docweave.models.diagnostics import Category, Code


class PlainTextAdapter(Adapter):
    name = "plain-text"
    formats = ("text",)
    capabilities = ("byte-exact text", "unicode safe", "zero optional dependencies")
    limitations = ("no structure beyond blank-line separated blocks",)
    cost = "cheap"

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        raw, encoding, lossy = read_text(ctx.path)
        text, stats = clean_text(raw)

        if lossy or stats.replacement:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"Some bytes could not be decoded as {encoding}; "
                f"{stats.replacement} replacement character(s) appear in the output.",
                category=Category.STRUCTURAL,
                reason=f"Strict decode failed for the declared charset '{encoding}'.",
                hint="Re-save the file as UTF-8 to avoid data loss.",
                evidence=stats.as_evidence(),
            )
        if stats.c1:
            ctx.warn(
                Code.REPLACEMENT_CHARS,
                f"{stats.c1} C1 control character(s) found; the file may be mis-encoded.",
                category=Category.STRUCTURAL,
                evidence=stats.as_evidence(),
            )

        source = ref(self.name, char_locator(0, len(text)), confidence=Confidence.HIGH)

        blocks = []
        offset = 0
        for i, chunk in enumerate(text.split("\n\n")):
            stripped = chunk.strip()
            if stripped:
                blocks.append(
                    Paragraph(
                        id=ctx.ids.child(None, f"b{i}"),
                        text=stripped,
                        sources=[ref(self.name, char_locator(offset, len(stripped)))],
                    )
                )
            offset += len(chunk) + 2
        del source

        doc = Document(
            adapter=self.name,
            container_kind="body",
            body=blocks,
            metadata={
                "encoding": encoding,
                "line_count": text.count("\n") + 1,
                "character_count": len(text),
            },
        )
        return ExtractionResult(
            document=doc,
            source=SourceInfo(name=ctx.path.name, byte_size=ctx.path.stat().st_size),
            container_kind="body",
        )
