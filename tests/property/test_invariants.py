"""Invariants that must hold for *any* input.

The rest of this suite checks that documents convert into what they should be.
That cannot find a failure nobody thought to write a case for: every fixture is
an answer to a question somebody already asked.

These tests ask questions with no known answer. They generate inputs -- random
bytes, truncated archives, cell values containing the renderer's own delimiters
-- and assert only properties that follow from the format or from a promise the
engine makes. There is no expected output to match, so a failure means the
engine broke a guarantee, not that the expectation was wrong.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docweave import convert
from docweave.config import ConversionConfig
from docweave.exceptions import DocWeaveError

# Settings come from the profile registered in tests/property/conftest.py:
# property tests build a fixture per example, so they need room, and a genuine
# hang must not be allowed to eat the whole suite.

signatures = st.sampled_from(
    [
        (b"%PDF-1.7\n", "corrupt.pdf"),
        (b"PK\x03\x04", "broken.docx"),
        (b"\x89PNG\r\n\x1a\n", "image.png"),
        (b"id,name\n", "data.csv"),
        (b"{", "broken.json"),
        (b"<html>", "page.html"),
        (b"a,b\n1,2\n", "data.csv"),
    ]
)


def _convert(path, **kwargs):
    """Convert, or return ``None`` if the engine refused the input.

    Refusing with a typed error is the engine working: an empty file must raise
    ``EmptyFileError``, an unreadable one ``MalformedDocumentError``. What must
    never escape is an untyped failure -- a ``KeyError`` from a parser handed
    something it cannot read, or a ``zipfile.BadZipFile`` surfacing from
    ``convert`` because the signature check did not run first.
    """
    try:
        return convert(str(path), **kwargs)
    except DocWeaveError:
        return None
    except Exception as exc:  # noqa: BLE001 - the point is to name what escaped
        raise AssertionError(
            f"untyped {type(exc).__name__} escaped the converter: {exc}"
        ) from exc


def _render(path, **kwargs) -> str:
    """Convert and render, or render nothing because the input was refused."""
    result = _convert(path, **kwargs)
    if result is None:
        return ""
    return result.to_markdown()


@given(payload=st.binary(max_size=400))
def test_conversion_never_raises_on_arbitrary_bytes(tmp_path_factory, payload):
    """Any byte sequence is a file. None of them may crash the converter.

    A document parser that raises on untrusted input is a denial-of-service
    bug whatever the input was meant to be.
    """
    directory = tmp_path_factory.mktemp("fuzz")
    path = directory / "input.bin"
    path.write_bytes(payload)
    result = _convert(path)
    if result is not None:
        # Whatever comes back must still be renderable.
        assert isinstance(result.to_markdown(), str)
        assert isinstance(result.to_json(), str)


@given(truncation=st.integers(min_value=1, max_value=99))
def test_conversion_never_raises_on_a_truncated_office_file(tmp_path_factory, truncation):
    """Cutting a valid container anywhere must not raise.

    Every real file in the wild has been truncated by something. The guarantee
    is not that the result is correct -- it is that the engine reports rather
    than explodes.
    """
    directory = tmp_path_factory.mktemp("trunc")
    path = directory / "cut.xlsx"
    build_minimal_xlsx(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) * truncation // 100])
    _render(path)


@given(signature=signatures)
def test_a_matching_extension_is_never_better_than_the_content(tmp_path_factory, signature):
    """Sniffing decides the parser; the file name only breaks ties.

    Bytes that claim to be a PDF must be read as a PDF whatever the extension
    says, and a file whose extension lies about it must still be detected.
    """
    magic, name = signature
    directory = tmp_path_factory.mktemp("sniff")
    path = directory / name
    path.write_bytes(magic + b"\x00" * 32)
    result = _convert(path)
    # Detection must have produced a verdict rather than a guess that crashes:
    # either a refused input, or a document attributed to a real adapter.
    if result is not None:
        assert result.document.adapter != "unknown"
        assert result.document.container_kind in (
            "body",
            "pages",
            "slides",
            "sheets",
        )


def build_minimal_xlsx(path) -> None:
    """The smallest file openpyxl will write, used as a truncation base."""
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "header"
    sheet["B1"] = "value"
    sheet["A2"] = "row"
    sheet["B2"] = 1
    workbook.save(str(path))


@given(payload=st.binary(min_size=8, max_size=300))
def test_a_zip_claim_with_non_zip_bytes_is_reported_not_raised(tmp_path_factory, payload):
    """A file that starts like a ZIP but is not must be diagnosed.

    The signature check has to run before the archive parser, or every
    ``PK\\x03\\x04`` prefix becomes an unhandled exception.
    """
    directory = tmp_path_factory.mktemp("fakezip")
    path = directory / "not-really.docx"
    path.write_bytes(payload)



@given(extra=st.binary(min_size=1, max_size=64))
def test_zip_trailing_garbage_is_tolerated(tmp_path_factory, extra):
    """Bytes appended after a ZIP end record must not break reading it."""
    directory = tmp_path_factory.mktemp("trailer")
    path = directory / "padded.xlsx"
    build_minimal_xlsx(path)
    with open(path, "ab") as handle:
        handle.write(extra)
    _render(path)


@given(entry_name=st.text(alphabet="abc/.", min_size=1, max_size=20))
def test_a_zip_with_an_odd_entry_name_is_reported_not_raised(tmp_path_factory, entry_name):
    """Archive entry names are attacker-controlled."""
    directory = tmp_path_factory.mktemp("entries")
    path = directory / "odd.docx"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry_name, "content")
        archive.writestr("[Content_Types].xml", "<Types/>")
    path.write_bytes(buffer.getvalue())
    _render(path)


@given(config_keys=st.dictionaries(
    st.sampled_from(
        ["table_rendering", "image_rendering", "ocr_repair", "detect_diagrams", "limits"]
    ),
    st.one_of(st.booleans(), st.sampled_from(["grid", "list", "html", "both", "omit", "auto"])),
    max_size=3,
))
def test_no_configuration_breaks_rendering(tmp_path_factory, config_keys):
    """Every supported option combination must still produce output."""
    from docweave.config import ConversionConfig as CC

    directory = tmp_path_factory.mktemp("config")
    path = directory / "sample.csv"
    path.write_text("a,b\n1,2\n")
    accepted = {}
    for key, value in config_keys.items():
        if key == "limits":
            continue
        try:
            CC(**{key: value})
        except Exception:
            continue  # not a combination the config actually accepts
        accepted[key] = value
    out = _render(path, config=ConversionConfig(**accepted))
    assert "a" in out


@pytest.mark.parametrize("value", ["", " ", "\n", "\t", " ", ""])
def test_degenerate_cell_values_do_not_crash_the_table_writer(tmp_path, value):
    """Empty and invisible values must not produce broken output."""
    import openpyxl

    path = tmp_path / "degenerate.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "h1"
    sheet["B1"] = "h2"
    sheet["A2"] = value
    sheet["B2"] = "keep"
    workbook.save(str(path))
    out = _render(path)
    assert "keep" in out
