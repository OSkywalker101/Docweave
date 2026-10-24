"""Shared helpers for adapters: encoding handling, text hygiene, node builders."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from docweave.models import (
    Confidence,
    Locator,
    Origin,
    Paragraph,
    SourceRef,
)

#: Unicode replacement char and other markers of lossy decoding.
_REPLACEMENT = "\ufffd"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
#: C1 control range, which shows up when Latin-1 is misdetected as UTF-8.
_C1_RE = re.compile(r"[\x80-\x9f]")
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

_HTML_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)


@dataclass
class TextStats:
    """Character-level hygiene counters for a decoded blob."""

    total: int = 0
    replacement: int = 0
    control: int = 0
    c1: int = 0
    normalized: int = 0

    def as_evidence(self) -> dict:
        return {
            "characters": self.total,
            "replacement_chars": self.replacement,
            "control_chars": self.control,
            "c1_control_chars": self.c1,
            "unicode_normalized": self.normalized,
        }


def decode_bytes(data: bytes, *, declared: str | None = None) -> tuple[str, str, bool]:
    """Decode ``data`` to text.

    Returns ``(text, encoding, lossy)``. Tries, in order: an explicit encoding,
    a BOM, an HTML ``<meta charset>``, then a strict utf-8 decode, then
    cp1252/latin-1 with a replacement pass. The boolean reports whether any
    bytes had to be replaced, which the validator reports as unverified content.
    """
    encoding = declared
    if encoding is None:
        for bom, enc in _BOM_ENCODINGS:
            if data.startswith(bom):
                encoding = enc
                break
    if encoding is None:
        match = _HTML_META_CHARSET_RE.search(data[:4096])
        if match:
            candidate = match.group(1).decode("ascii", "replace")
            if candidate.lower() in {"utf-8", "utf8"}:
                candidate = "utf-8"
            encoding = candidate
    if encoding is not None:
        try:
            return data.decode(encoding), encoding, False
        except (UnicodeDecodeError, LookupError):
            pass
    try:
        return data.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        pass
    for fallback in ("cp1252", "latin-1"):
        text = data.decode(fallback, errors="replace")
        return text, fallback, _REPLACEMENT in text
    raise UnicodeDecodeError("utf-8", data, 0, 1, "undecodable")  # pragma: no cover


def read_text(path: Path, *, declared: str | None = None) -> tuple[str, str, bool]:
    return decode_bytes(path.read_bytes(), declared=declared)


def measure(text: str) -> TextStats:
    stats = TextStats(total=len(text))
    stats.replacement = text.count(_REPLACEMENT)
    stats.control = len(_CONTROL_RE.findall(text))
    stats.c1 = len(_C1_RE.findall(text))
    return stats


def clean_text(text: str, *, strip_controls: bool = True, normalize: bool = True) -> tuple[str, TextStats]:
    """Remove characters that are never meaningful in converted output.

    Tabs and newlines are preserved because they carry structure. Normalisation
    uses NFC, which is a safe, reversible-for-our-purposes default; the counter
    records how much changed so the caller can report it.
    """
    before = len(text)
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_controls:
        out = _CONTROL_RE.sub("", out)
    stats = measure(out)
    if normalize:
        normalized = unicodedata.normalize("NFC", out)
        stats.normalized = before - len(normalized) if len(normalized) != before else 0
        out = normalized
    stats.total = len(out)
    return out, stats


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def ref(
    adapter: str,
    locator: Locator | None = None,
    *,
    origin: Origin = Origin.OBSERVED,
    confidence: Confidence = Confidence.HIGH,
    note: str | None = None,
) -> SourceRef:
    """Build provenance for a node extracted by ``adapter``."""
    return SourceRef(
        adapter=adapter,
        origin=origin,
        confidence=confidence,
        locator=locator or Locator(),
        note=note,
    )


def paragraph(
    text: str,
    node_id: str,
    source: SourceRef,
    **kwargs,
) -> Paragraph:
    return Paragraph(id=node_id, text=text, sources=[source], **kwargs)


def char_locator(offset: int, length: int) -> Locator:
    return Locator(kind="char", char_offset=offset, char_length=length)


def join_blocks(blocks: Iterable) -> str:
    """Join text for display purposes (Markdown renderer, plain output)."""
    out: list[str] = []
    for b in blocks:
        t = getattr(b, "text", None)
        if t:
            out.append(t)
    return "\n\n".join(out)
