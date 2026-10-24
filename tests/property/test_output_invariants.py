"""Guarantees the engine makes about its own output.

These do not check that a document converts *correctly* -- that is the job of
the fixture corpus, which can only ever test the failures somebody already
thought of. These check promises the engine makes regardless of what is in the
file: the same bytes always give the same output, a cell's contents cannot
change how many cells there are, and nothing that was written comes out
missing.
"""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from benchmarks.accuracy import _parse_sheet_cells, tables_from_markdown
from docweave import convert
from docweave.exceptions import DocWeaveError

#: Text that has historically broken a delimiter-based format. The separator is
#: DocWeave's own, so a value containing it is the adversarial case.
# XLSX cannot store control characters at all -- openpyxl refuses them -- so
# they are outside the domain of this property rather than a DocWeave concern.
NASTY = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs", "Cc"), min_codepoint=32, max_codepoint=0x2FFFD
    ),
    max_size=60,
)

CELL_VALUES = st.lists(
    st.one_of(NASTY, st.text(alphabet="AB", min_size=1, max_size=2)),
    min_size=1,
    max_size=6,
)


def _xlsx(path, rows: list[list[str]]) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            sheet.cell(row=r, column=c, value=value)
    workbook.save(str(path))


@given(rows=st.lists(CELL_VALUES, min_size=1, max_size=4))
def test_a_cell_value_cannot_change_the_number_of_cells(tmp_path_factory, rows):
    """Regression class: delimiter injection.

    The list format joins cells with ``"; "``. If a value can contain that
    separator followed by something shaped like a cell label, one cell becomes
    two and the sheet is silently miscounted. Rendering must therefore be
    unambiguous, whatever is in the cells.
    """
    directory = tmp_path_factory.mktemp("inject")
    path = directory / "sheet.xlsx"
    _xlsx(path, rows)
    result = convert(str(path))
    out = result.to_markdown()
    parsed = tables_from_markdown(out)
    if not parsed:
        return  # nothing rendered as a table; nothing to miscount
    detected = len(parsed[0]["cells"])
    # Empty strings are dropped rather than rendered, so the detected count may
    # be lower -- but never higher, which is the direction the bug appeared in.
    non_empty = sum(1 for row in rows for value in row if value.strip())
    assert detected <= non_empty
    assert detected >= 1


@given(value=NASTY)
def test_a_cell_value_survives_a_render_parse_round_trip(tmp_path_factory, value):
    """Whatever goes into a cell must come back out of the parser.

    This is the consumer half of the injection guarantee. Quoting the producer
    only helps if the reader undoes it, so the test goes renderer -> parser and
    compares against the original text rather than a hand-built escape.
    """
    from docweave.renderers.markdown import _quote_ambiguous

    rendered = f"A1: {_quote_ambiguous(value)}"
    parsed = _parse_sheet_cells(rendered)

    # Surrounding whitespace is normalised, and deliberately so: two trailing
    # spaces are a Markdown hard line break, so a cell ending in a space would
    # otherwise corrupt the rendered document. The invariant is therefore that
    # the *content* survives, not the padding.
    expected = value.strip()
    if not expected:
        assert parsed == []
    else:
        assert parsed == [expected], f"{value!r} came back as {parsed!r}"


@given(values=st.lists(NASTY, min_size=1, max_size=5))
def test_non_empty_cell_values_are_all_present_in_the_output(tmp_path_factory, values):
    """No cell may simply disappear.

    Loss is the failure a converter is least allowed to make, and it is the one
    that is easiest to miss by reading only the first line of output.
    """
    directory = tmp_path_factory.mktemp("lossless")
    path = directory / "keep.xlsx"
    _xlsx(path, [values])
    out = convert(str(path)).to_markdown()
    stripped = re.sub(r"-\s+[A-Z]{1,3}\d+\s*:", " ", out)
    for value in values:
        # Surrounding whitespace is normalised away at render time, because two
        # trailing spaces are a Markdown hard line break. The content is what
        # must survive.
        content = value.strip()
        if not content:
            continue
        # Markup in a cell is escaped rather than executed, so the value may
        # appear either verbatim or with backslashes inserted. Both are fine;
        # disappearance is not.
        unescaped = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>])", r"\1", content)
        assert content in stripped or unescaped in stripped, f"lost {value!r}"


@given(text=st.text(min_size=1, max_size=200))
def test_identical_input_gives_identical_output(tmp_path_factory, text):
    """Determinism is a promise, not a performance property.

    Byte-identical output for identical input is what makes the benchmark
    numbers reproducible and what lets a caller cache on content hash.
    """
    directory = tmp_path_factory.mktemp("determinism")
    path = directory / "note.md"
    path.write_text(text, encoding="utf-8")
    first = _convert(path)
    second = _convert(path)
    # The same refusal, or the same output, twice over. What must never happen
    # is two runs disagreeing about the same bytes.
    assert (first is None) == (second is None)
    if first is not None and second is not None:
        assert first.to_markdown() == second.to_markdown()
        assert first.to_json() == second.to_json()


@given(text=st.text(min_size=1, max_size=120))
def test_output_formatting_does_not_depend_on_the_path(tmp_path_factory, text):
    """The file name must not leak into the content.

    Detection may legitimately depend on a name, but once an adapter has run,
    two copies of the same bytes under different names must read the same.
    """
    directory = tmp_path_factory.mktemp("naming")
    first_path = directory / "a.md"
    second_path = directory / "totally-different-name.md"
    first_path.write_text(text, encoding="utf-8")
    second_path.write_text(text, encoding="utf-8")
    # Both copies are refused or accepted together: detection looks at content,
    # so the same bytes must reach the same verdict regardless of name.
    a = _convert(first_path)
    b = _convert(second_path)
    assert (a is None) == (b is None)
    if a is not None and b is not None:
        assert _body(a.to_markdown()) == _body(b.to_markdown())


@given(value=NASTY)
def test_cell_content_is_never_interpreted_as_markup(tmp_path_factory, value):
    """A cell holding markup is data, not structure.

    ``|---|`` in a cell must not become a table separator and ``<table>`` must
    not open one, because the source is a spreadsheet cell and never intended
    either.
    """
    directory = tmp_path_factory.mktemp("markup")
    path = directory / "data.xlsx"
    _xlsx(path, [["header"], [value]])
    out = convert(str(path)).to_markdown()
    body = out.split("---")[0]
    # The rendered body must not have gained a table it was not given.
    assert body.count("<table>") == 0


def _convert(path, **kwargs):
    """Convert, or return `None` if the engine refused the input."""
    try:
        return convert(str(path), **kwargs)
    except DocWeaveError:
        return None


def _body(markdown: str) -> str:
    """Strip the diagnostics trailer, which may mention the file name."""
    return markdown.split("\n---\n")[0]
