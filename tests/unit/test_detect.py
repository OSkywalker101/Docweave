"""Format detection: signatures beat extensions, and lies are reported."""

from __future__ import annotations

import pytest

from docweave.detect import (
    EXTENSION_MAP,
    FMT_CSV,
    FMT_DOCX,
    FMT_HTML,
    FMT_JSON,
    FMT_MARKDOWN,
    FMT_PDF,
    FMT_TEXT,
    FMT_TSV,
    FMT_XLSX,
    FMT_XML,
    detect_format,
    inspect,
)
from docweave.exceptions import (
    EmptyFileError,
    EncryptedDocumentError,
    FileTooLargeError,
    UnsupportedFormatError,
)


class TestSignatures:
    def test_pdf_detected_by_magic(self, fx):
        detection = detect_format(fx("simple.pdf"))
        assert detection.format == FMT_PDF
        assert detection.confidence == 1.0
        assert detection.method == "signature"

    def test_docx_detected_from_zip_parts(self, fx):
        assert detect_format(fx("sample.docx")).format == FMT_DOCX

    def test_xlsx_detected_from_zip_parts(self, fx):
        assert detect_format(fx("sample.xlsx")).format == FMT_XLSX

    def test_html_detected_without_extension(self, scratch):
        path = scratch / "page.dat"
        path.write_text("<!DOCTYPE html><html><body><p>hi</p></body></html>")
        assert detect_format(path).format == FMT_HTML

    def test_xml_declaration(self, fx):
        assert detect_format(fx("sample.xml")).format == FMT_XML

    def test_json_parsed_not_guessed(self, fx):
        assert detect_format(fx("sample.json")).format == FMT_JSON

    def test_csv_detected_from_field_counts(self, fx):
        assert detect_format(fx("sample.csv")).format == FMT_CSV

    def test_tsv_prefers_tab(self, scratch):
        path = scratch / "a.csv"
        path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        assert detect_format(path).format == FMT_TSV

    def test_ragged_csv_still_detected_as_csv(self, fx):
        # The strict "all rows equal" rule would classify this as prose.
        assert detect_format(fx("ragged.csv")).format == FMT_CSV

    def test_markdown_beats_pipe_table(self, scratch):
        path = scratch / "notes.txt"
        path.write_text("# Title\n\ntext\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n")
        assert detect_format(path).format == FMT_MARKDOWN

    def test_plain_text_fallback(self, scratch):
        path = scratch / "notes.txt"
        path.write_text("just some prose without structure at all")
        assert detect_format(path).format == FMT_TEXT


class TestExtensionsAreNotTrusted:
    def test_csv_named_pdf_is_detected_as_csv(self, fx):
        detection = detect_format(fx("liar.pdf"))
        assert detection.format == FMT_CSV
        assert detection.declared_extension == "pdf"
        assert detection.extension_mismatch is True

    def test_markdown_extension_is_not_a_mismatch(self, fx):
        detection = detect_format(fx("sample.md"))
        assert detection.format == FMT_MARKDOWN
        assert detection.extension_mismatch is False

    def test_unknown_extension_is_not_a_mismatch(self, scratch):
        path = scratch / "data.qqq"
        path.write_text("name,qty\napple,3\n")
        detection = detect_format(path)
        assert detection.format == FMT_CSV
        assert detection.extension_mismatch is False

    def test_extension_map_covers_documented_formats(self):
        assert EXTENSION_MAP["pdf"] == FMT_PDF
        assert EXTENSION_MAP["docx"] == FMT_DOCX
        assert EXTENSION_MAP["xlsx"] == FMT_XLSX


class TestRejections:
    def test_empty_file(self, fx):
        with pytest.raises(EmptyFileError):
            detect_format(fx("empty.bin"))

    def test_directory_is_not_a_document(self, scratch):
        with pytest.raises(UnsupportedFormatError):
            detect_format(scratch)

    def test_missing_file(self, scratch):
        with pytest.raises(UnsupportedFormatError):
            detect_format(scratch / "nope.pdf")

    def test_size_limit(self, scratch):
        path = scratch / "big.txt"
        # 1.5 MB of prose: above the 1 MB floor that max_file_size_mb enforces.
        path.write_text("lorem ipsum dolor sit amet " * 45_000)
        assert path.stat().st_size > 1024 * 1024
        with pytest.raises(FileTooLargeError) as exc:
            detect_format(path, max_file_size_mb=1)
        assert exc.value.limit == 1024 * 1024

    def test_encrypted_container_is_detected(self, scratch):
        import zipfile

        path = scratch / "secret.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("EncryptedPackage", b"\x00" * 32)
        with pytest.raises(EncryptedDocumentError):
            detect_format(path)


class TestInspection:
    def test_inspect_reports_structure_without_parsing(self, fx):
        result = inspect(fx("sample.xlsx"))
        assert result.detection.format == FMT_XLSX
        assert result.structure["archive_entries"] > 0
        assert "has_macros" in result.structure
        assert len(result.sha256) == 64

    def test_inspect_pdf_version(self, fx):
        assert inspect(fx("simple.pdf")).structure["pdf_version"] == "1.7"

    def test_inspect_empty_file_is_unreadable(self, fx):
        result = inspect(fx("empty.bin"))
        assert result.readable is False
        assert result.detection.format == "unknown"

    def test_encrypted_pdf_is_flagged(self, fx):
        result = inspect(fx("encrypted.pdf"))
        assert result.detection.encrypted is True


class TestMalformedContainers:
    def test_broken_zip_container(self, fx):
        # Signatures are right, structure is not: reported as zip with low confidence
        # (previously 'unknown', now 'zip' with notes about failure)
        result = detect_format(fx("notzip.docx"))
        assert result.format in {"unknown", "zip"}
        if result.format == "zip":
            assert result.confidence < 0.5
            assert any("could not be opened" in n for n in result.notes)

    def test_ole2_container(self, scratch):
        path = scratch / "legacy.doc"
        path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
        assert detect_format(path).format in {"doc", "ole2"}

    def test_rtf_is_identified_but_unsupported(self, scratch):
        path = scratch / "legacy.rtf"
        path.write_bytes(b"{\\rtf1\\ansi hello}")
        detection = detect_format(path)
        assert detection.format == "rtf"
        assert detection.confidence > 0.9
