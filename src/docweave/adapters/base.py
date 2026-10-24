"""Adapter interface and registry.

An adapter converts one source file into the canonical IR. Adapters are
independently testable, replaceable, and declare their own capabilities,
limitations and optional dependencies so ``docweave capabilities`` can report
the truth without importing anything heavy.
"""

from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docweave.assets import AssetCollector
from docweave.config import ConversionConfig
from docweave.exceptions import (
    DeadlineExceededError,
    DependencyMissingError,
    LimitExceededError,
    MalformedDocumentError,
    UnsupportedFormatError,
)
from docweave.ids import IdFactory
from docweave.models import Diagnostic, Document, Origin, SourceInfo, SourceRef
from docweave.models.diagnostics import Category, Code, Severity


@dataclass
class ExtractionContext:
    """Everything an adapter needs, plus the guards it must respect."""

    path: Path
    config: ConversionConfig
    ids: IdFactory
    assets: AssetCollector
    diagnostics: list[Diagnostic] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    deadline: float | None = None
    registry: Any = None
    detection: Any = None
    #: Adapter-reported facts merged into Document.metadata.
    metadata: dict = field(default_factory=dict)

    def check_deadline(self, stage: str = "extraction") -> None:
        """Cooperative cancellation between units of work."""
        if self.deadline is None:
            return
        if time.perf_counter() > self.deadline:
            raise DeadlineExceededError(
                f"Conversion exceeded the configured deadline of "
                f"{self.config.deadline_seconds:g}s during {stage}."
            )

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def warn(
        self,
        code: str,
        message: str,
        *,
        category: Category = Category.HEURISTIC,
        severity: Severity = Severity.WARNING,
        reason: str | None = None,
        hint: str | None = None,
        scope_kind: str | None = None,
        scope_index: int | None = None,
        scope_name: str | None = None,
        node_id: str | None = None,
        evidence: dict | None = None,
    ) -> Diagnostic:
        d = Diagnostic(
            code=code,
            category=category,
            severity=severity,
            message=message,
            reason=reason,
            hint=hint,
            scope_kind=scope_kind,
            scope_index=scope_index,
            scope_name=scope_name,
            node_id=node_id,
            evidence=evidence or {},
        )
        self.diagnostics.append(d)
        return d

    def ref(
        self,
        locator=None,
        *,
        origin: Origin = Origin.OBSERVED,
        confidence: str = "high",
        note: str | None = None,
    ) -> SourceRef:
        from docweave.models import Confidence, Locator

        return SourceRef(
            adapter="",
            origin=origin,
            confidence=Confidence(confidence),
            locator=locator or Locator(),
            note=note,
        )


@dataclass
class ExtractionResult:
    """What an adapter returns."""

    document: Document
    source: SourceInfo
    container_kind: str = "body"


class Adapter(ABC):
    """Base class for every format adapter."""

    #: Stable adapter identifier, recorded in provenance.
    name: str = "adapter"
    #: Detected formats this adapter claims.
    formats: tuple[str, ...] = ()
    #: Import name -> pip package name, for optional dependencies.
    requires: dict[str, str] = {}
    #: What this adapter does well.
    capabilities: tuple[str, ...] = ()
    #: What it does not do. Rendered by ``docweave capabilities``.
    limitations: tuple[str, ...] = ()
    #: Whether the adapter streams (bounded memory) or loads the whole file.
    streaming: bool = False
    #: Rough relative cost hint for routing documentation.
    cost: str = "cheap"  # cheap | moderate | expensive

    def check_dependencies(self) -> None:
        """Raise :class:`DependencyMissingError` when a requirement is absent."""
        import importlib.util

        for module, package in self.requires.items():
            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                extra = _EXTRA_FOR.get(package, package)
                raise DependencyMissingError(package, f"The {self.name} adapter", extra)

    @abstractmethod
    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        """Convert the source file and return the IR."""

    # -- helpers available to all adapters ---------------------------------

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Adapter {self.name} formats={self.formats}>"


_EXTRA_FOR = {
    "pymupdf": "pdf",
    "python-docx": "office",
    "python-pptx": "office",
    "openpyxl": "office",
    "docx": "office",
    "pptx": "office",
    "xlrd": "xls",
    "pytesseract": "ocr",
    "indra_ocr": "ocr",
    "faster_whisper": "audio",
    "feedparser": "web",
    "wikipediaapi": "web",
    "yt_dlp": "youtube",
    "msg_extractor": "msg",
    "nbformat": "notebook",
    "ebooklib": "epub",
}


class AdapterRegistry:
    """Lazy adapter lookup. Adapters are constructed on first use only."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Adapter]] = {}
        self._instances: dict[str, Adapter] = {}
        self._load_errors: dict[str, str] = {}

    def register(self, fmt: str, factory: Callable[[], Adapter]) -> None:
        self._factories[fmt] = factory

    def formats(self) -> list[str]:
        return sorted(self._factories)

    def has(self, fmt: str) -> bool:
        return fmt in self._factories

    def get(self, fmt: str) -> Adapter:
        if fmt not in self._factories:
            raise UnsupportedFormatError(
                f"No adapter is registered for format '{fmt}'. "
                f"Supported: {', '.join(self.formats())}",
                detected=fmt,
            )
        if fmt not in self._instances:
            try:
                adapter = self._factories[fmt]()
            except Exception as exc:  # pragma: no cover - defensive
                self._load_errors[fmt] = str(exc)
                raise
            adapter.check_dependencies()
            self._instances[fmt] = adapter
        return self._instances[fmt]

    def capabilities(self) -> list[dict]:
        """Report every registered adapter, including unavailable ones.

        Uses ``find_spec`` rather than importing, so this stays cheap. Adapters
        that serve several formats are grouped under their own name so their
        capabilities are not attributed to just one of them.
        """
        import importlib.util

        grouped: dict[str, dict] = {}
        for fmt in sorted(self._factories):
            try:
                adapter = self._factories[fmt]()
                name = adapter.name
                requires = dict(adapter.requires)
                caps = list(adapter.capabilities)
                lims = list(adapter.limitations)
                cost = adapter.cost
                streaming = adapter.streaming
            except Exception as exc:  # pragma: no cover - defensive
                name = _fallback_name(fmt)
                requires, caps, lims = {}, [], []
                cost, streaming = "unknown", False
                self._load_errors.setdefault(fmt, str(exc))

            entry = grouped.get(name)
            if entry is None:
                entry = {
                    "adapter": name,
                    "formats": [],
                    "requires": requires,
                    "missing_dependencies": [],
                    "available": True,
                    "capabilities": [],
                    "limitations": [],
                    "cost": cost,
                    "streaming": streaming,
                }
                grouped[name] = entry

            entry["formats"].append(fmt)
            for item in caps:
                if item not in entry["capabilities"]:
                    entry["capabilities"].append(item)
            for item in lims:
                if item not in entry["limitations"]:
                    entry["limitations"].append(item)

        for entry in grouped.values():
            missing = []
            for module, package in entry["requires"].items():
                try:
                    ok = importlib.util.find_spec(module) is not None
                except (ImportError, ValueError):
                    ok = False
                if not ok:
                    missing.append(package)
            entry["missing_dependencies"] = missing
            entry["available"] = not missing

        return [grouped[k] for k in sorted(grouped)]


def _fallback_name(fmt: str) -> str:
    return fmt


def build_registry() -> AdapterRegistry:
    """Construct the registry with every adapter wired in."""
    from docweave.adapters import (
        audio_adapter,
        bing_adapter,
        csv_adapter,
        docx_adapter,
        epub_adapter,
        html_adapter,
        json_adapter,
        markdown_adapter,
        msg_adapter,
        notebook_adapter,
        pdf_adapter,
        pptx_adapter,
        rss_adapter,
        text_adapter,
        wikipedia_adapter,
        xls_adapter,
        xlsx_adapter,
        xml_adapter,
        youtube_adapter,
        zip_adapter,
    )

    registry = AdapterRegistry()
    registry.register("pdf", lambda: pdf_adapter.PdfAdapter())
    registry.register("docx", lambda: docx_adapter.DocxAdapter())
    registry.register("pptx", lambda: pptx_adapter.PptxAdapter())
    registry.register("xlsx", lambda: xlsx_adapter.XlsxAdapter())
    registry.register("xls", lambda: xls_adapter.XlsAdapter())
    registry.register("html", lambda: html_adapter.HtmlAdapter())
    registry.register("xhtml", lambda: html_adapter.HtmlAdapter())
    registry.register("text", lambda: text_adapter.PlainTextAdapter())
    registry.register("markdown", lambda: markdown_adapter.MarkdownAdapter())
    registry.register("csv", lambda: csv_adapter.DelimitedAdapter("csv"))
    registry.register("tsv", lambda: csv_adapter.DelimitedAdapter("tsv"))
    registry.register("json", lambda: json_adapter.JsonAdapter())
    registry.register("xml", lambda: xml_adapter.XmlAdapter())
    registry.register("zip", lambda: zip_adapter.ZipAdapter())
    registry.register("audio", lambda: audio_adapter.AudioAdapter())
    registry.register("youtube", lambda: youtube_adapter.YouTubeAdapter())
    registry.register("msg", lambda: msg_adapter.MsgAdapter())
    registry.register("rss", lambda: rss_adapter.RssAdapter())
    registry.register("atom", lambda: rss_adapter.RssAdapter())
    registry.register("wikipedia", lambda: wikipedia_adapter.WikipediaAdapter())
    registry.register("bing", lambda: bing_adapter.BingAdapter())
    registry.register("ipynb", lambda: notebook_adapter.NotebookAdapter())
    registry.register("epub", lambda: epub_adapter.EpubAdapter())
    return registry


# Re-exported for adapter modules.
__all__ = [
    "Adapter",
    "AdapterRegistry",
    "ExtractionContext",
    "ExtractionResult",
    "build_registry",
    "LimitExceededError",
    "MalformedDocumentError",
    "Category",
    "Code",
    "Severity",
    "Document",
    "importlib",
    "Iterable",
]
