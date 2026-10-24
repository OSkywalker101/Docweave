"""Diagnostics: machine-readable, severity-classified extraction problems.

The four categories mirror the contract in the spec: a consumer must be able to
tell apart a *confirmed structural defect*, a *heuristic suspicion*, content we
*could not verify*, and something that *needs a human*.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Category(str, Enum):
    #: A structural problem we can point at concretely.
    STRUCTURAL = "structural_problem"
    #: A suspicion produced by a rule; may be a false positive.
    HEURISTIC = "heuristic_warning"
    #: Content exists in the source but could not be verified as correctly read.
    UNVERIFIED = "unverified_content"
    #: The operator has to intervene (missing dependency, encryption, limits).
    INTERVENTION = "user_intervention_required"


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    category: Category
    severity: Severity = Severity.WARNING
    message: str
    #: Why the engine believes this. Explanation is part of the contract.
    reason: str | None = None
    #: Suggested remediation, when one exists.
    hint: str | None = None
    #: page / slide / sheet index, when the problem is localised.
    scope_kind: str | None = None
    scope_index: int | None = None
    scope_name: str | None = None
    node_id: str | None = None
    #: Measured evidence backing the diagnostic (counts, ratios, thresholds).
    evidence: dict[str, Any] = Field(default_factory=dict)

    def location(self) -> str:
        if self.scope_kind == "page" and self.scope_index is not None:
            return f"page {self.scope_index}"
        if self.scope_kind == "slide" and self.scope_index is not None:
            return f"slide {self.scope_index}"
        if self.scope_kind == "sheet" and self.scope_name:
            return f"sheet '{self.scope_name}'"
        if self.scope_name:
            return self.scope_name
        return self.scope_kind or "document"

    def to_line(self) -> str:
        """One plain-text line for logs.

        Deliberately avoids ``[brackets]``: console renderers treat them as
        markup and would silently drop the diagnostic code.
        """
        prefix = {
            Severity.ERROR: "ERROR",
            Severity.WARNING: "WARN",
            Severity.INFO: "INFO",
        }[self.severity]
        return f"{prefix} {self.code} {self.location()}: {self.message}"


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostics: list[Diagnostic] = Field(default_factory=list)

    def add(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def extend(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    # -- queries -----------------------------------------------------------

    def by_category(self, category: Category) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.category == category]

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]

    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.WARNING]

    def codes(self) -> list[str]:
        return [d.code for d in self.diagnostics]

    def has(self, code: str) -> bool:
        return any(d.code == code for d in self.diagnostics)

    def worst_severity(self) -> Severity | None:
        if any(d.severity == Severity.ERROR for d in self.diagnostics):
            return Severity.ERROR
        if any(d.severity == Severity.WARNING for d in self.diagnostics):
            return Severity.WARNING
        if self.diagnostics:
            return Severity.INFO
        return None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {c.value: 0 for c in Category}
        for d in self.diagnostics:
            out[d.category.value] += 1
        return out

    def sorted(self) -> list[Diagnostic]:
        """Deterministic ordering: errors first, then by scope, then code."""
        sev_rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        return sorted(
            self.diagnostics,
            key=lambda d: (
                sev_rank[d.severity],
                d.scope_kind or "",
                d.scope_index if d.scope_index is not None else -1,
                d.code,
                d.message,
            ),
        )

    def summary(self) -> str:
        counts = self.counts()
        return (
            f"{len(self.diagnostics)} diagnostic(s): "
            f"{counts[Category.STRUCTURAL.value]} structural, "
            f"{counts[Category.HEURISTIC.value]} heuristic, "
            f"{counts[Category.UNVERIFIED.value]} unverified, "
            f"{counts[Category.INTERVENTION.value]} requiring intervention"
        )


# Diagnostic codes, centralised so tests and docs stay in sync.
class Code:
    # structural
    NO_CONTENT = "no_content_extracted"
    EMPTY_CONTAINER = "empty_container"
    RAGGED_TABLE = "table_ragged_rows"
    CONTROL_CHARS = "malformed_control_characters"
    REPLACEMENT_CHARS = "malformed_replacement_characters"
    BROKEN_XML = "malformed_xml"
    # heuristic
    LOW_TEXT_DENSITY = "heuristic_low_text_density"
    IMAGE_HEAVY_PAGE = "heuristic_image_heavy_page"
    MULTI_COLUMN = "heuristic_multi_column"
    LIKELY_SCANNED = "heuristic_likely_scanned"
    READING_ORDER_INFERRED = "heuristic_reading_order_inferred"
    EMPTY_CELLS = "heuristic_significant_empty_cells"
    # unverified
    IMAGE_WITHOUT_TEXT = "unverified_image_content"
    EQUATION_NOT_RECOVERED = "unverified_equation_not_recovered"
    CHART_DATA_UNAVAILABLE = "unverified_chart_data_unavailable"
    FORMULA_WITHOUT_CACHED_VALUE = "unverified_formula_without_cached_value"
    HIDDEN_CONTENT = "unverified_hidden_content"
    EXTERNAL_LINK = "unverified_external_reference"
    # intervention
    ENCRYPTED = "requires_decryption"
    MISSING_DEPENDENCY = "requires_optional_dependency"
    LIMIT_EXCEEDED = "requires_larger_limits"
    PARSER_FALLBACK = "requires_parser_review"
    UNSUPPORTED_STRUCTURE = "requires_adapter_support"
    EXTENSION_MISMATCH = "input_extension_mismatch"
    # Diagram engine
    DIAGRAM_UNCONNECTED = "diagram_unconnected_line"
    DIAGRAM_ISOLATED_NODE = "diagram_isolated_node"
    DIAGRAM_UNTYPED_SHAPE = "diagram_untyped_shape"
    DIAGRAM_UNASSIGNED_SHAPE = "diagram_unassigned_shape"
    DIAGRAM_LABEL_UNREAD = "diagram_label_unread"
    DIAGRAM_KIND_UNKNOWN = "diagram_kind_unknown"
    DIAGRAM_NOTHING_DETECTED = "diagram_nothing_detected"
    DIAGRAM_SKIPPED_NON_DIAGRAM = "diagram_skipped_non_diagram"
    DIAGRAM_ANALYSIS_FAILED = "diagram_analysis_failed"
    # OCR repair
    OCR_REPAIRED = "page_repaired_by_ocr"
    OCR_EMPTY = "ocr_returned_no_text"
    OCR_REJECTED = "ocr_rejected_less_content"
    OCR_UNAVAILABLE = "ocr_engine_unavailable"
