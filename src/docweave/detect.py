"""Format detection by content signature.

Design rules from the spec:

* Never trust the extension. It is only used as a tiebreaker and a mismatch is
  reported, not silently obeyed.
* Be cheap. We read a bounded prefix, plus at most one extra read for ZIP
  containers so we can look at ``[Content_Types].xml``.
* Be honest. When the content does not match a known signature we say so
  instead of guessing from a familiar-looking extension.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docweave.exceptions import (
    EmptyFileError,
    EncryptedDocumentError,
    FileTooLargeError,
    MalformedDocumentError,
    UnsupportedFormatError,
)

#: Canonical format identifiers used throughout the engine.
FMT_PDF = "pdf"
FMT_DOCX = "docx"
FMT_DOC = "doc"
FMT_PPTX = "pptx"
FMT_XLSX = "xlsx"
FMT_XLS = "xls"
FMT_EPUB = "epub"
FMT_HTML = "html"
FMT_XHTML = "xhtml"
FMT_XML = "xml"
FMT_JSON = "json"
FMT_CSV = "csv"
FMT_TSV = "tsv"
FMT_MARKDOWN = "markdown"
FMT_TEXT = "text"
FMT_IMAGE = "image"
FMT_RTF = "rtf"
FMT_ZIP = "zip"
FMT_AUDIO = "audio"
FMT_MSG = "msg"
FMT_RSS = "rss"
FMT_ATOM = "atom"
FMT_IPYNB = "ipynb"
FMT_UNKNOWN = "unknown"

#: Formats the base install can actually convert without optional dependencies.
NATIVE_FORMATS = (
    FMT_HTML,
    FMT_XHTML,
    FMT_XML,
    FMT_JSON,
    FMT_CSV,
    FMT_TSV,
    FMT_MARKDOWN,
    FMT_TEXT,
)

PDF = FMT_PDF
OOXML_FORMATS = (FMT_DOCX, FMT_PPTX, FMT_XLSX)

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"RIFF", "audio/wav"),  # WAV starts with RIFF
    (b"ID3", "audio/mpeg"),  # MP3 often starts with ID3
    (b"\xff\xfb", "audio/mpeg"),  # MP3 frame sync
    (b"\xff\xf3", "audio/mpeg"),  # MP3 frame sync
    (b"\xff\xf2", "audio/mpeg"),  # MP3 frame sync
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
)

_AUDIO_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", "audio/wav"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
    (b"\xff\xf2", "audio/mpeg"),
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"\x00\x00\x00", "audio/mp4"),  # M4A/MP4 often starts with size atom
)

_CONTAINER_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "application/zip"),
    (b"PK\x05\x06", "application/zip"),
    (b"PK\x07\x08", "application/zip"),
)

_RSS_NS = (b'xmlns="http://purl.org/rss/1.0/"', b'xmlns="http://www.w3.org/2005/Atom"', b"rss", b"feed")

_CONTROL_CHARS = bytes(range(0, 9)) + bytes((11, 12)) + bytes(range(14, 32))
_XML_DECL_RE = re.compile(rb"^\s*<\?xml", re.IGNORECASE)
_HTML_HINT_RE = re.compile(
    rb"<\s*(!doctype\s+html|html|head|body|title|meta|div|p|table|span|a)\b", re.IGNORECASE
)
_MARKDOWN_HINT_RE = re.compile(
    rb"(^|\n)\s{0,3}(#{1,6}\s|```|>\s|[-*+]\s|\d+\.\s|\[[^\]]+\]\([^)]+\)|\s*\|.*\|\s*$)"
)
_RTF_RE = re.compile(rb"^\s*\{\\rtf", re.IGNORECASE)


@dataclass
class Detection:
    """Outcome of format detection."""

    format: str
    confidence: float
    method: str
    declared_extension: str | None = None
    media_type: str | None = None
    encrypted: bool = False
    #: Set when the content is a known-but-unsupported container (OLE2, RTF...).
    container_format: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def extension_mismatch(self) -> bool:
        """True when the extension implies a different format than the content.

        Compared through the extension map, so ``.md`` and ``.markdown`` both match
        the ``markdown`` format instead of looking like a mismatch.
        """
        if not self.declared_extension:
            return False
        expected = EXTENSION_MAP.get(self.declared_extension.lower())
        if expected is None:
            return False
        return expected != self.format


#: Extension -> expected format. Used only for mismatch reporting.
EXTENSION_MAP: dict[str, str] = {
    "pdf": FMT_PDF,
    "docx": FMT_DOCX,
    "doc": FMT_DOC,
    "pptx": FMT_PPTX,
    "ppt": "ppt",
    "xlsx": FMT_XLSX,
    "xls": FMT_XLS,
    "epub": FMT_EPUB,
    "html": FMT_HTML,
    "htm": FMT_HTML,
    "xhtml": FMT_XHTML,
    "xml": FMT_XML,
    "json": FMT_JSON,
    "jsonl": FMT_JSON,
    "ndjson": FMT_JSON,
    "csv": FMT_CSV,
    "tsv": FMT_TSV,
    "tab": FMT_TSV,
    "md": FMT_MARKDOWN,
    "markdown": FMT_MARKDOWN,
    "txt": FMT_TEXT,
    "text": FMT_TEXT,
    "log": FMT_TEXT,
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "bmp": "image",
    "webp": "image",
    "rtf": FMT_RTF,
    "zip": FMT_ZIP,
    "mp3": FMT_AUDIO,
    "wav": FMT_AUDIO,
    "m4a": FMT_AUDIO,
    "flac": FMT_AUDIO,
    "ogg": FMT_AUDIO,
    "aac": FMT_AUDIO,
    "msg": FMT_MSG,
    "rss": FMT_RSS,
    "atom": FMT_ATOM,
    "ipynb": FMT_IPYNB,
}


@dataclass
class InspectionResult:
    """What ``docweave inspect`` reports: identity plus cheap facts."""

    name: str
    byte_size: int
    sha256: str
    detection: Detection
    readable: bool
    error: str | None = None
    #: Populated for containers: entry counts etc.
    structure: dict[str, object] = field(default_factory=dict)


class _Reader:
    """Bounded, streaming file access with a size guard."""

    def __init__(self, path: Path, max_bytes: int, sniff: int = 8192) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.sniff = sniff
        self._zipfile: zipfile.ZipFile | None = None

    def head(self, size: int | None = None) -> bytes:
        size = size or self.sniff
        with self.path.open("rb") as fh:
            return fh.read(size)

    def iter_prefix_lines(self, max_lines: int = 20) -> list[bytes]:
        lines: list[bytes] = []
        with self.path.open("rb") as fh:
            for raw in fh:
                lines.append(raw[:4096])
                if len(lines) >= max_lines:
                    break
        return lines

    def read_all(self) -> bytes:
        return self.path.read_bytes()

    def open_zip(self) -> zipfile.ZipFile | None:
        if self._zipfile is None:
            try:
                self._zipfile = zipfile.ZipFile(self.path)
            except (zipfile.BadZipFile, OSError):
                return None
        return self._zipfile

    def close(self) -> None:
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None


def _zip_ooxml_format(reader: _Reader) -> tuple[str | None, list[str]]:
    """Identify an OOXML package by its content types / parts."""
    zf = reader.open_zip()
    if zf is None:
        return None, ["zip container could not be opened"]
    names = set(zf.namelist())
    notes: list[str] = []
    if "word/document.xml" in names:
        return FMT_DOCX, notes
    if "ppt/presentation.xml" in names:
        return FMT_PPTX, notes
    if "xl/workbook.xml" in names:
        return FMT_XLSX, notes
    if "mimetype" in names:
        try:
            mt = zf.read("mimetype").decode("ascii", "replace").strip()
        except (KeyError, zipfile.BadZipFile, OSError):
            mt = ""
        if mt == "application/epub+zip":
            return FMT_EPUB, notes
    if "metadata.json" in names and any(n.endswith(".ipynb") for n in names):
        # IPython notebook exported as zip (rare) or check for notebook.json
        pass
    # Check for .ipynb structure: typically has a single .ipynb file or notebook.json
    for name in names:
        if name.endswith(".ipynb"):
            try:
                import json
                content = zf.read(name).decode("utf-8")
                data = json.loads(content)
                if "cells" in data and "metadata" in data and "nbformat" in data:
                    return FMT_IPYNB, notes
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                pass
    if any(n.startswith("EncryptedPackage") for n in names):
        return "mso_encrypted", notes
    if any(n.startswith("Encrypted") for n in names):
        return "ooxml_encrypted", notes
    return None, notes


def _classify_text(prefix: bytes, path: Path, sniff: int) -> tuple[str, float, str, list[str]]:
    """Classify text-ish content. Order matters: most specific first."""
    notes: list[str] = []

    if _XML_DECL_RE.match(prefix):
        return FMT_XML, 0.9, "signature", notes
    if _HTML_HINT_RE.search(prefix):
        return FMT_HTML, 0.75, "signature", notes

    sample = prefix
    stripped = sample.lstrip()
    if stripped[:1] in (b"{", b"["):
        # Try a strict parse of the sniffed prefix; fall back to a lenient scan.
        try:
            json.loads(sample.decode("utf-8"))
            return FMT_JSON, 1.0, "signature", notes
        except (UnicodeDecodeError, json.JSONDecodeError):
            # A complete JSON document is always under the sniff window unless
            # deeply nested; if it parses when we ignore the tail, accept it.
            decoder = json.JSONDecoder()
            text = sample.decode("utf-8", "replace")
            try:
                _, end = decoder.raw_decode(text)
                if text[end:].strip() == "":
                    return FMT_JSON, 0.9, "content", notes
            except ValueError:
                pass
        # A ``{``/``[`` opener that does not parse is far more likely to be a
        # broken JSON file than prose. Classifying it as JSON lets the adapter
        # report the precise syntax error instead of quietly emitting the
        # source as a paragraph of text.
        notes.append("content starts with a JSON container but did not parse")
        return FMT_JSON, 0.6, "content", notes

    # Markdown before delimited text: a Markdown table would otherwise be
    # misread as a CSV file, losing the document's heading structure.
    if _MARKDOWN_HINT_RE.search(sample):
        return FMT_MARKDOWN, 0.7, "content", notes

    lines = [ln for ln in sample.split(b"\n") if ln.strip()][:10]
    if len(lines) >= 2:
        fmt = _sniff_delimiter(lines)
        if fmt:
            return fmt, 0.7, "content", notes

    return FMT_TEXT, 0.5, "fallback", notes


def _sniff_delimiter(lines: list[bytes]) -> str | None:
    """Guess the delimiter from field-count behaviour.

    A strict "every line has the same count" rule fails on exactly the files that
    most need a table (ragged exports, trailing notes). We require instead that
    the delimiter appears in most lines and that the typical row has at least two
    fields, which separates tabular data from prose containing commas.
    """
    scored: list[tuple[int, str]] = []
    for delim, fmt in ((b"\t", FMT_TSV), (b",", FMT_CSV)):
        counts = [ln.count(delim) for ln in lines]
        if counts[0] == 0:
            continue
        present = [c for c in counts if c > 0]
        if len(present) < max(2, len(lines) * 0.6):
            continue
        typical = sorted(present)[len(present) // 2]
        if typical < 1 or typical > 64:
            continue
        scored.append((typical, fmt))
    if not scored:
        return None
    # More fields wins; tab wins ties, because TSV is rarer in prose than commas.
    scored.sort(key=lambda item: (item[0], item[1] == FMT_TSV))
    return scored[-1][1]


def detect_format(path: str | Path, max_file_size_mb: int = 256, sniff: int = 8192) -> Detection:
    """Detect the real format of ``path`` from its content.

    Raises :class:`InputError` subclasses when the file is unusable.
    """
    p = Path(path)
    if not p.exists():
        raise UnsupportedFormatError(f"No such file: {p.name}")
    if p.is_dir():
        raise UnsupportedFormatError(f"{p.name} is a directory, not a document")

    size = p.stat().st_size
    if size == 0:
        raise EmptyFileError(f"{p.name} is empty (0 bytes)")
    limit = max_file_size_mb * 1024 * 1024
    if size > limit:
        raise FileTooLargeError(size, limit)

    declared = p.suffix.lstrip(".").lower() or None
    reader = _Reader(p, limit, sniff)
    try:
        prefix = reader.head()
        fmt, confidence, method, notes, media_type, encrypted, container = _detect_from_bytes(
            reader, prefix, size
        )
    finally:
        reader.close()

    return Detection(
        format=fmt,
        confidence=confidence,
        method=method,
        declared_extension=declared,
        media_type=media_type,
        encrypted=encrypted,
        container_format=container,
        notes=notes,
    )


def _detect_from_bytes(
    reader: _Reader, prefix: bytes, size: int
) -> tuple[str, float, str, list[str], str | None, bool, str | None]:
    notes: list[str] = []

    if prefix.startswith(b"%PDF-"):
        encrypted = b"/Encrypt" in reader.head(min(size, 1 << 20))
        if encrypted:
            notes.append("PDF trailer advertises /Encrypt")
        return FMT_PDF, 1.0, "signature", notes, "application/pdf", encrypted, None

    if prefix.startswith(_ZIP_MAGIC):
        fmt, zip_notes = _zip_ooxml_format(reader)
        notes.extend(zip_notes)
        if fmt == "mso_encrypted":
            raise EncryptedDocumentError(
                "This Office file is encrypted at the container level (EncryptedPackage). "
                "Decrypt it in Office before converting."
            )
        if fmt == "ooxml_encrypted":
            raise EncryptedDocumentError(
                "This Office file contains an encrypted package. Decrypt it in Office first."
            )
        if fmt in (FMT_DOCX, FMT_PPTX, FMT_XLSX):
            return fmt, 0.95, "container", notes, _OOXML_MEDIA[fmt], False, None
        if fmt == FMT_EPUB:
            return FMT_EPUB, 0.9, "container", notes, "application/epub+zip", False, None
        if fmt == FMT_IPYNB:
            return fmt, 0.95, "container", notes, "application/x-ipynb+json", False, None
        notes.append("zip container without a recognised document part")
        return FMT_UNKNOWN, 0.4, "container", notes, None, False, "zip"

    if prefix.startswith(_OLE_MAGIC):
        return _classify_ole(reader), 0.9, "signature", notes, None, False, "ole2"

    for magic, media in _AUDIO_MAGIC:
        if prefix.startswith(magic):
            return FMT_AUDIO, 1.0, "signature", notes, media, False, None

    for magic, media in _IMAGE_MAGIC:
        if prefix.startswith(magic):
            return FMT_IMAGE, 1.0, "signature", notes, media, False, None

    if _RTF_RE.match(prefix):
        return FMT_RTF, 0.95, "signature", notes, "application/rtf", False, None

    # Check for RSS/Atom in the first few KB of XML
    if _XML_DECL_RE.match(prefix):
        lower = prefix.lower()
        if b"rss" in lower[:2048] or b"feed" in lower[:2048] or b"atom" in lower[:2048]:
            if b"<rss" in lower or b'<feed xmlns="http://www.w3.org/2005/Atom"' in lower:
                return FMT_ATOM, 0.9, "signature", notes, "application/atom+xml", False, None
            if b"<rss" in lower:
                return FMT_RSS, 0.9, "signature", notes, "application/rss+xml", False, None

    fmt, confidence, method, notes = _classify_text(prefix, reader.path, reader.sniff)
    return fmt, confidence, method, notes, _TEXT_MEDIA.get(fmt), False, None


def _classify_ole(reader: _Reader) -> str:
    """Legacy OLE2 compound files: .doc/.xls/.ppt, or an encrypted OOXML shell."""
    head = reader.read_all()[:1 << 20]
    if b"EncryptedPackage" in head or b"EncryptionInfo" in head:
        return "mso_encrypted"
    # Check for Outlook MSG (has __properties_version1.0 or __substg1.0 streams)
    if b"__properties_version1.0" in head or b"__substg1.0" in head or b"__nameid_version1.0" in head:
        return FMT_MSG
    if b"Workbook" in head or b"Book" in head:
        return FMT_XLS
    if b"WordDocument" in head:
        return FMT_DOC
    if b"PowerPoint Document" in head:
        return "ppt"
    return "ole2"


_OOXML_MEDIA = {
    FMT_DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FMT_PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    FMT_XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_TEXT_MEDIA = {
    FMT_HTML: "text/html",
    FMT_XHTML: "application/xhtml+xml",
    FMT_XML: "application/xml",
    FMT_JSON: "application/json",
    FMT_CSV: "text/csv",
    FMT_TSV: "text/tab-separated-values",
    FMT_MARKDOWN: "text/markdown",
    FMT_TEXT: "text/plain",
}


def inspect(path: str | Path, max_file_size_mb: int = 256) -> InspectionResult:
    """Cheap inspection used by ``docweave inspect``; never parses the document body."""
    p = Path(path)
    name = p.name
    digest = ""
    size = 0
    structure: dict[str, object] = {}

    if p.exists() and p.is_file():
        size = p.stat().st_size
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()

    try:
        detection = detect_format(p, max_file_size_mb=max_file_size_mb)
    except EncryptedDocumentError:
        raise
    except (EmptyFileError, FileTooLargeError, UnsupportedFormatError) as exc:
        return InspectionResult(
            name=name,
            byte_size=size,
            sha256=digest,
            detection=Detection(
                format=FMT_UNKNOWN, confidence=0.0, method="failed", declared_extension=p.suffix.lstrip(".").lower() or None
            ),
            readable=False,
            error=str(exc),
        )

    structure = _describe_structure(p, detection.format)
    return InspectionResult(
        name=name,
        byte_size=size,
        sha256=digest,
        detection=detection,
        readable=True,
        structure=structure,
    )


def _describe_structure(path: Path, fmt: str) -> dict[str, object]:
    """Shallow, cheap structural facts. Never loads a full document object."""
    if fmt in (FMT_DOCX, FMT_PPTX, FMT_XLSX, FMT_EPUB, FMT_IPYNB):
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
        except (zipfile.BadZipFile, OSError) as exc:
            raise MalformedDocumentError(f"Archive structure is unreadable: {exc}") from exc
        return {
            "archive_entries": len(names),
            "media_parts": sum(1 for n in names if n.startswith(("word/media/", "ppt/media/", "xl/media/"))),
            "has_macros": any(n.endswith("vbaProject.bin") for n in names),
            "encrypted_parts": any(n.startswith("Encrypted") for n in names),
        }
    if fmt == FMT_PDF:
        return {"pdf_version": _pdf_version(path)}
    if fmt in (FMT_CSV, FMT_TSV):
        return {"delimiter": "\\t" if fmt == FMT_TSV else ","}
    if fmt == FMT_ZIP:
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            return {
                "archive_entries": len(names),
                "total_uncompressed": sum(zf.getinfo(n).file_size for n in names),
            }
        except (zipfile.BadZipFile, OSError):
            return {"error": "invalid zip"}
    if fmt == FMT_AUDIO:
        # Could add duration estimation from file size/bitrate heuristic
        return {"format": "audio"}
    if fmt == FMT_MSG:
        return {"format": "outlook_msg"}
    if fmt in (FMT_RSS, FMT_ATOM):
        return {"format": "feed"}
    return {}


def _pdf_version(path: Path) -> str | None:
    try:
        head = path.open("rb").read(16)
    except OSError:
        return None
    if head.startswith(b"%PDF-"):
        return head[5:8].decode("ascii", "replace").strip()
    return None


def available_optional_dependency(package: str) -> bool:
    """Check for an optional dependency without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False
