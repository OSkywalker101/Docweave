"""Conversion configuration.

Every knob that affects output or cost lives here so that behaviour is
inspectable, serialisable to JSON and diffable between runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docweave.exceptions import InputError


class ConversionConfig(BaseModel):
    """User-facing conversion options."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # -- resource limits ---------------------------------------------------
    max_file_size_mb: int = Field(default=256, ge=1)
    max_pages: int = Field(default=5000, ge=1)
    max_cells: int = Field(default=500_000, ge=1)
    max_csv_rows: int = Field(default=200_000, ge=1)
    max_asset_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    #: Cooperative deadline checked between pages/rows. Not a hard kill.
    deadline_seconds: float | None = Field(default=600.0, gt=0)
    #: Bytes sniffed from the head of a file for format detection.
    sniff_bytes: int = Field(default=8192, ge=64)

    # -- extraction --------------------------------------------------------
    include_headers_footers: bool = False
    include_links: bool = True
    include_images: bool = True
    #: Write image bytes to disk. Off by default: it costs I/O and memory.
    extract_assets: bool = False
    assets_dir: Path | None = None
    #: "auto" only attempts table recovery on pages that look like they have one.
    pdf_tables: Literal["auto", "always", "never"] = "auto"
    #: Load the workbook a second time to recover cached formula results.
    xlsx_cached_values: bool = True
    xlsx_max_rows_per_sheet: int = Field(default=50_000, ge=1)
    html_strip_selectors: list[str] = Field(
        default_factory=lambda: ["script", "style", "noscript", "template", "svg"]
    )
    html_strip_comments: bool = True

    # -- routing thresholds (heuristic, tunable, evaluated on held-out data) -
    min_chars_per_page: int = Field(default=40, ge=0)
    #: Master switch for the scanned-page heuristic. Turning it off makes the
    #: engine stop *warning*; it never changes what is extracted, because OCR is
    #: not on the default path either way.
    detect_scanned_pages: bool = True
    #: A page whose images cover more than this fraction of the page area and
    #: has almost no text is probably scanned and needs OCR.
    image_heavy_area_ratio: float = Field(default=0.5, gt=0, le=1)
    scanned_max_chars: int = Field(default=20, ge=0)
    #: Column split is attempted when a vertical whitespace gutter wider than
    #: this fraction of the text width exists.
    column_gutter_ratio: float = Field(default=0.04, gt=0, le=1)
    #: Ratio of replacement characters above which Unicode damage is reported.
    replacement_char_ratio: float = Field(default=0.001, ge=0, le=1)

    # -- repair ------------------------------------------------------------
    #: Run the OCR repair pass over pages the first pass could not read.
    #: Off by default: it is slow, it needs an external binary, and OCR output
    #: is never exact. Only pages measured to be missing text are touched.
    ocr_repair: bool = False
    ocr_language: str = "eng"
    #: Rendering resolution for the pages sent to OCR.
    ocr_dpi: int = Field(default=200, ge=72, le=600)

    # -- diagrams ----------------------------------------------------------
    #: Reconstruct diagrams found among the document's images. Opt-in: it
    #: costs OpenCV plus a reconstruction pass per image, and most images in a
    #: document are not diagrams.
    detect_diagrams: bool = False
    #: Run the visual classifier before the diagram engine and skip images it
    #: says are photos or screenshots. Costs one cheap classifier pass per
    #: image; without it every qualifying image is handed to the geometry
    #: engine regardless of what it actually is.
    classify_images: bool = False

    # -- rendering ---------------------------------------------------------
    table_rendering: Literal["markdown", "html", "both", "list"] = "markdown"
    image_rendering: Literal["placeholder", "path", "link", "omit"] = "placeholder"
    #: Emit ``docweave-image-N`` markers; useful for downstream chunkers.
    emit_stable_ids: bool = False
    #: Prepend a small source/format table. Off by default because the JSON
    #: envelope already carries provenance and Markdown is the LLM-facing target.
    markdown_include_front_matter: bool = False
    #: Include per-adapter timing in the JSON envelope (non-deterministic).
    include_timing: bool = False
    #: Emit a trailing diagnostics summary block in Markdown output.
    markdown_include_diagnostics: bool = True

    # -- loading / saving --------------------------------------------------

    @classmethod
    def from_json_file(cls, path: str | Path) -> ConversionConfig:
        p = Path(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise InputError(f"Config file not found: {p.name}") from exc
        except json.JSONDecodeError as exc:
            raise InputError(f"Config file is not valid JSON: {p.name} ({exc})") from exc
        if not isinstance(raw, dict):
            raise InputError(f"Config file must contain a JSON object: {p.name}")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ConversionConfig:
        known = set(cls.model_fields)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise InputError(
                "Unknown configuration key(s): "
                + ", ".join(unknown)
                + ". Valid keys: "
                + ", ".join(sorted(known))
            )
        if "assets_dir" in raw and raw["assets_dir"] is not None:
            raw = dict(raw)
            raw["assets_dir"] = Path(str(raw["assets_dir"]))
        return cls(**raw)

    def merged(self, **overrides: Any) -> ConversionConfig:
        clean = {k: v for k, v in overrides.items() if v is not None}
        if "assets_dir" in clean and clean["assets_dir"] is not None:
            clean["assets_dir"] = Path(str(clean["assets_dir"]))
        return self.model_copy(update=clean)

    def to_mapping(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        return data

    def describe(self) -> str:
        """Stable, human-readable fingerprint of the settings that affect output."""
        payload = self.model_dump(mode="json")
        keep = (
            "include_headers_footers",
            "include_links",
            "include_images",
            "extract_assets",
            "pdf_tables",
            "xlsx_cached_values",
            "table_rendering",
            "image_rendering",
        )
        return json.dumps({k: payload[k] for k in keep}, sort_keys=True)
