"""Conversion orchestration and the public Python API.

Flow: detect -> analyse (implicit, during extraction) -> extract -> validate ->
render. Every stage records what it did; nothing is silently discarded.

This module is the only place that knows the whole pipeline. It deliberately does
not start a web server, spawn a subprocess, or reach the network.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from docweave import detect as detect_mod
from docweave.adapters import AdapterRegistry, ExtractionContext, build_registry
from docweave.assets import AssetCollector
from docweave.config import ConversionConfig
from docweave.detect import Detection, InspectionResult
from docweave.detect import inspect as inspect_file
from docweave.exceptions import DocWeaveError, UnsupportedFormatError
from docweave.ids import IdFactory
from docweave.models import Diagnostic, DiagnosticReport, Document, SourceInfo
from docweave.models.diagnostics import Category, Code, Severity
from docweave.renderers import (
    MarkdownRenderer,
    render_json,
    render_jsonl,
    render_report,
)
from docweave.validation import validate

OutputFormat = str  # "markdown" | "json" | "jsonl" | "report"

MARKDOWN = "markdown"
JSON = "json"
JSONL = "jsonl"
REPORT = "report"
OUTPUT_FORMATS = (MARKDOWN, JSON, JSONL, REPORT)

PathLike = Union[str, Path]


@dataclass
class ConversionResult:
    """Everything a caller needs about one document."""

    document: Document
    diagnostics: DiagnosticReport
    source: SourceInfo
    detection: Detection
    elapsed_ms: float = 0.0
    adapter: str = ""
    #: The configuration this result was produced with. Rendering honours it, so
    #: ``convert(f, cfg).to_markdown()`` matches ``convert(f, cfg).render(...)``.
    config: ConversionConfig | None = None

    @property
    def ok(self) -> bool:
        return not any(d.severity.value == "error" for d in self.diagnostics.diagnostics)

    def to_markdown(self) -> str:
        return MarkdownRenderer(self.config).render(
            self.document, self.diagnostics, self.document.assets
        )

    def to_json(self, config: ConversionConfig | None = None, *, timing: bool = False) -> str:
        return render_json(
            self.document,
            self.diagnostics,
            config or self.config,
            timing={"total_ms": round(self.elapsed_ms, 3)} if timing else None,
        )

    def to_jsonl(self) -> str:
        return render_jsonl(self.document)

    def to_report(self) -> str:
        return render_report(self.document, self.diagnostics)

    def render(self, output_format: OutputFormat, config: ConversionConfig | None = None) -> str:
        if output_format == MARKDOWN:
            return self.to_markdown()
        if output_format == JSON:
            return self.to_json(config)
        if output_format == JSONL:
            return self.to_jsonl()
        if output_format == REPORT:
            return self.to_report()
        raise UnsupportedFormatError(
            f"Unknown output format '{output_format}'. Choose one of: {', '.join(OUTPUT_FORMATS)}."
        )


@dataclass
class BatchItem:
    path: Path
    result: ConversionResult | None = None
    error: str | None = None
    error_kind: str | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.result is not None
            and self.result.ok
        )

    @property
    def reason(self) -> str:
        """Why this item is not a success. Never empty."""
        if self.error:
            return self.error
        if self.result is None:
            return "no result"
        errors = self.result.diagnostics.errors()
        if errors:
            return errors[0].message
        return "unknown failure"


@dataclass
class BatchReport:
    items: list[BatchItem] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def succeeded(self) -> list[BatchItem]:
        return [i for i in self.items if i.ok]

    @property
    def failed(self) -> list[BatchItem]:
        return [i for i in self.items if not i.ok]

    def summary(self) -> dict:
        diagnostics = DiagnosticReport()
        for item in self.succeeded:
            assert item.result is not None
            diagnostics.extend(item.result.diagnostics.diagnostics)
        return {
            "total": len(self.items),
            "succeeded": len(self.succeeded),
            "failed": len(self.failed),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "diagnostics": diagnostics.counts(),
            "codes": sorted(set(diagnostics.codes())),
        }


class Converter:
    """Reusable converter. Create once, convert many documents."""

    def __init__(
        self,
        config: ConversionConfig | None = None,
        registry: AdapterRegistry | None = None,
    ) -> None:
        self.config = config or ConversionConfig()
        self.registry = registry or build_registry()

    # -- single document ---------------------------------------------------

    def convert(self, source: PathLike) -> ConversionResult:
        path = Path(source)
        config = self.config

        detection = detect_mod.detect_format(
            path,
            max_file_size_mb=config.max_file_size_mb,
            sniff=config.sniff_bytes,
        )
        if not self.registry.has(detection.format):
            hint = ""
            if detection.extension_mismatch:
                hint = (
                    f" The extension says '{detection.declared_extension}' but the content is "
                    f"'{detection.format}'."
                )
            raise UnsupportedFormatError(
                f"Format '{detection.format}' is not supported by this build.{hint} "
                f"Supported formats: {', '.join(self.registry.formats())}",
                detected=detection.format,
            )

        digest, size = _hash_file(path)
        ids = IdFactory()
        assets = AssetCollector(
            directory=config.assets_dir,
            enabled=config.extract_assets,
            max_bytes=config.max_asset_bytes,
        )
        deadline = (
            time.perf_counter() + config.deadline_seconds
            if config.deadline_seconds is not None
            else None
        )
        ctx = ExtractionContext(
            path=path,
            config=config,
            ids=ids,
            assets=assets,
            deadline=deadline,
            registry=self.registry,
            detection=detection,
        )

        adapter = self.registry.get(detection.format)
        started = time.perf_counter()
        extracted = adapter.extract(ctx)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if detection.extension_mismatch:
            ctx.warn(
                Code.EXTENSION_MISMATCH,
                f"File extension is '.{detection.declared_extension}' but the content is "
                f"'{detection.format}'. Content won.",
                category=Category.HEURISTIC,
                severity=Severity.INFO,
                reason="Detection is driven by file signatures, never by the extension alone.",
                evidence={
                    "declared_extension": detection.declared_extension,
                    "detected_format": detection.format,
                    "detection_confidence": detection.confidence,
                },
            )

        document = extracted.document
        document.adapter = adapter.name
        document.assets = assets.refs()
        # Deliberately NOT recording conversion_ms in metadata: the JSON envelope
        # must be byte-identical across runs for the same input.
        document.source = SourceInfo(
            name=path.name,
            byte_size=size,
            sha256=digest,
            detected_format=detection.format,
            detection_method=detection.method,
            detection_confidence=detection.confidence,
            declared_extension=detection.declared_extension,
            extension_mismatch=detection.extension_mismatch,
            encrypted=detection.encrypted,
        )
        if detection.notes:
            document.metadata.setdefault("detection_notes", list(detection.notes))

        report = validate(document, config)
        report.extend(ctx.diagnostics)

        if config.detect_diagrams or config.classify_images:
            self._analyze_diagrams(path, document, config, report)

        if config.ocr_repair:
            # Repair runs after validation so it operates on the same page set
            # the report describes, and so its own diagnostics join the report.
            repair = self._repair(path, document, config)
            if repair is not None:
                report.extend(repair.diagnostics)
                document.metadata["ocr"] = {
                    "engine": repair.engine,
                    "repaired_pages": list(repair.repaired_pages),
                    "skipped_pages": list(repair.skipped_pages),
                    "rejected_pages": len(repair.rejected_pages),
                    "elapsed_ms": round(repair.elapsed_ms, 3),
                }

        report = _dedupe(report)

        return ConversionResult(
            document=document,
            diagnostics=report,
            source=document.source,
            detection=detection,
            elapsed_ms=elapsed_ms,
            adapter=adapter.name,
            config=config,
        )

    def _analyze_diagrams(
        self, path: Path, document: Document, config: ConversionConfig, report: DiagnosticReport
    ) -> None:
        """Reconstruct any diagram found among the document's images.

        Images are only analysed when they were extracted to disk, because the
        engine must not hold every embedded image in memory just to look at it.
        """
        from docweave.diagram.integrate import DiagramIntegrator, diagram_diagnostics

        integrator = DiagramIntegrator(classify_images=config.classify_images)
        found = 0
        for asset in document.assets:
            if not integrator.should_analyze(_asset_bytes(asset), asset.media_type):
                continue
            try:
                result = integrator.analyze_asset(asset)
            except Exception as exc:
                # Never swallow a reconstruction failure into a clean-looking
                # result: record what broke and why, with the image preserved.
                report.add(
                    Diagnostic(
                        code=Code.DIAGRAM_ANALYSIS_FAILED,
                        category=Category.UNVERIFIED,
                        severity=Severity.WARNING,
                        message=(
                            f"Diagram asset '{asset.asset_id}' could not be analysed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        reason="An unexpected error was raised by the diagram engine.",
                        hint="The source image is preserved; inspect it directly.",
                        scope_kind="diagram",
                        scope_name=asset.asset_id,
                    )
                )
                continue
            if result is None:
                continue
            found += 1
            report.extend(diagram_diagnostics(result, asset_id=asset.asset_id))
            entry: dict[str, Any] = {
                "kind": result.graph.kind.value,
                "nodes": len(result.graph.nodes),
                "edges": len(result.graph.edges),
                "degraded": result.graph.degraded,
            }
            if result.classification is not None:
                entry["classification"] = result.classification.kind.value
                entry["classification_confidence"] = result.classification.confidence
            document.metadata.setdefault("diagrams", {})[asset.asset_id] = entry
        if found:
            document.metadata["diagram_count"] = found

    def _repair(self, path: Path, document: Document, config: ConversionConfig):
        """Run the targeted OCR repair pass. Returns ``None`` when not needed."""
        from docweave.repair import repair_document

        # Let the repair module choose: it prefers the most accurate engine
        # that is installed, and hardcoding one here meant a machine with both
        # installed silently used the worse one.
        return repair_document(path, document, config)

    def convert_safe(self, source: PathLike) -> tuple[ConversionResult | None, str | None, str | None]:
        """Convert without raising. Returns ``(result, error, error_kind)``."""
        try:
            return self.convert(source), None, None
        except DocWeaveError as exc:
            return None, str(exc), type(exc).__name__
        except Exception as exc:  # pragma: no cover - unexpected adapter bug
            return None, f"Unexpected {type(exc).__name__}: {exc}", type(exc).__name__

    # -- batch -------------------------------------------------------------

    def convert_many(
        self,
        sources: Sequence[PathLike],
        *,
        concurrency: int = 1,
        progress: Callable[[BatchItem], None] | None = None,
    ) -> BatchReport:
        """Convert many files. One failure never aborts the run."""
        started = time.perf_counter()
        items = [BatchItem(path=Path(s)) for s in sources]
        if not items:
            return BatchReport(items=[], elapsed_ms=0.0)

        if concurrency <= 1:
            for item in items:
                self._convert_item(item, progress)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(self._convert_item, item, None): item for item in items}
                for future in as_completed(futures):
                    item = futures[future]
                    future.result()
                    if progress:
                        progress(item)

        return BatchReport(items=items, elapsed_ms=(time.perf_counter() - started) * 1000.0)

    def _convert_item(
        self, item: BatchItem, progress: Callable[[BatchItem], None] | None
    ) -> None:
        t0 = time.perf_counter()
        result, error, kind = self.convert_safe(item.path)
        item.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        item.result = result
        item.error = error
        item.error_kind = kind
        if progress:
            progress(item)

    # -- inspection --------------------------------------------------------

    def inspect(self, source: PathLike) -> InspectionResult:
        return inspect_file(source, max_file_size_mb=self.config.max_file_size_mb)

    def capabilities(self) -> list[dict]:
        return self.registry.capabilities()


# --------------------------------------------------------------------------
# module-level convenience API (no server, no globals with state)
# --------------------------------------------------------------------------


def convert(
    source: PathLike,
    config: ConversionConfig | None = None,
    *,
    registry: AdapterRegistry | None = None,
) -> ConversionResult:
    """Convert one document. The entry point for RAG pipeline integration."""
    return Converter(config, registry).convert(source)


def convert_to(
    source: PathLike,
    output_format: OutputFormat = MARKDOWN,
    config: ConversionConfig | None = None,
    **kwargs: Any,
) -> str:
    """Convert one document straight to a string in the requested format."""
    config = config or ConversionConfig()
    result = convert(source, config, **kwargs)
    return result.render(output_format, config)


def inspect(source: PathLike, config: ConversionConfig | None = None) -> InspectionResult:
    return Converter(config).inspect(source)


def validate_document(source: PathLike, config: ConversionConfig | None = None) -> ConversionResult:
    """Convert and return only the diagnostics. Useful as a pre-flight check."""
    return convert(source, config)


def capabilities(config: ConversionConfig | None = None) -> list[dict]:
    return Converter(config).capabilities()


def discover(
    root: PathLike,
    *,
    recursive: bool = True,
    limit: int | None = None,
) -> list[Path]:
    """List candidate documents under ``root`` without reading them."""
    base = Path(root)
    if base.is_file():
        return [base]
    pattern = "**/*" if recursive else "*"
    out = [p for p in sorted(base.glob(pattern)) if p.is_file()]
    if limit is not None:
        out = out[:limit]
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _asset_bytes(asset) -> bytes | None:
    if not asset.path:
        return None
    try:
        return Path(asset.path).read_bytes()
    except OSError:
        return None


def _hash_file(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            size += len(block)
            h.update(block)
    return h.hexdigest(), size


def _dedupe(report: DiagnosticReport) -> DiagnosticReport:
    """Collapse diagnostics that repeat verbatim, keeping counts in evidence."""
    seen: dict[tuple, Diagnostic] = {}
    out = DiagnosticReport()
    for d in report.diagnostics:
        key = (d.code, d.scope_kind, d.scope_index, d.scope_name, d.message)
        existing = seen.get(key)
        if existing is None:
            seen[key] = d
            out.add(d)
            continue
        count = int(existing.evidence.get("occurrences", 1)) + 1
        existing.evidence["occurrences"] = count
    return out
