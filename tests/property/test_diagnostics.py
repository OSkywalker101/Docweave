"""A diagnostic must point somewhere real.

The engine's contract is that it tells you *where* a problem is, not just that
one exists. A diagnostic scoped to page 7 of a three-page document, or to a
sheet that was never read, is worse than no diagnostic: it sends a reader to a
location with nothing in it and costs them the time to discover the report is
lying.

This suite generates documents and checks that every locator and scope the
engine emits resolves against the document it came from.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docweave import convert
from docweave.models.diagnostics import Category, Diagnostic, Severity
from tests.property.test_invariants import _convert

CELL_REF = re.compile(r"^[A-Z]{1,3}[0-9]+$")


def _cells(document) -> set[str]:
    references: set[str] = set()
    for sheet in document.sheets:
        for row in sheet.table.rows:
            for cell in row:
                locator = cell.locator
                if locator is not None and locator.cell:
                    references.add(locator.cell)
    return references


def _sheets(document) -> set[str]:
    return {sheet.name for sheet in document.sheets}


def _check_diagnostics(result) -> None:
    document = result.document
    # The canonical registry: body nodes, container blocks, and a spreadsheet's
    # table. Anything a diagnostic is allowed to point at lives in here.
    known = {node.id for node in document.iter_nodes()}
    # Not every container carries an id; only contribute the ones that do.
    known |= {
        item.id
        for group in (document.pages, document.slides, document.sheets)
        for item in group
        if getattr(item, "id", None)
    }

    for diagnostic in result.diagnostics.diagnostics:
        # Every diagnostic must be able to say where it is, even if only
        # "document". This is what location() falls back to.
        described = diagnostic.location()
        assert described, f"{diagnostic.code} has no location at all"

        scope_kind = diagnostic.scope_kind
        index = diagnostic.scope_index
        name = diagnostic.scope_name

        # 1-based convention first: it is the more fundamental fault, and a
        # "page 0" caught by the range check reports the wrong problem.
        if index is not None:
            assert index >= 1, (
                f"{diagnostic.code} uses a {index}-based index as if it were "
                f"0-based: {described}"
            )

        if scope_kind == "page" and index is not None:
            assert 1 <= index <= len(document.pages), (
                f"{diagnostic.code} points at page {index} "
                f"of {len(document.pages)}: {described}"
            )
        if scope_kind == "slide" and index is not None:
            assert 1 <= index <= len(document.slides), (
                f"{diagnostic.code} points at slide {index} "
                f"of {len(document.slides)}: {described}"
            )
        if scope_kind == "sheet" and name:
            assert name in _sheets(document), (
                f"{diagnostic.code} names sheet {name!r}, "
                f"which is not in the document: {described}"
            )
        if scope_kind is not None:
            assert scope_kind in ("page", "slide", "sheet", "document", "body"), (
                f"{diagnostic.code} has unknown scope_kind {scope_kind!r}"
            )

        # A node reference must resolve, or the reader is sent nowhere.
        if diagnostic.node_id is not None:
            assert diagnostic.node_id in known, (
                f"{diagnostic.code} references node {diagnostic.node_id!r} "
                "which does not exist"
            )


def _check_locators(result) -> None:
    """Cell references must be well formed and belong to a sheet that exists."""
    document = result.document
    sheets = _sheets(document)

    for sheet in document.sheets:
        for row in sheet.table.rows:
            for cell in row:
                locator = cell.locator
                if locator is None or not locator.cell:
                    continue
                assert CELL_REF.match(locator.cell), (
                    f"cell reference {locator.cell!r} is not an A1 reference"
                )
                if locator.sheet is not None:
                    assert locator.sheet in sheets, (
                        f"cell {locator.cell!r} claims sheet "
                        f"{locator.sheet!r}, which was not read"
                    )

    for node in document.body:
        locator = getattr(node, "locator", None)
        if locator is None:
            continue
        if locator.sheet is not None:
            assert locator.sheet in sheets, (
                f"node locator names sheet {locator.sheet!r}, not in the document"
            )
        if locator.kind == "page" and locator.page is not None:
            assert 1 <= locator.page <= max(len(document.pages), 1)
        if locator.char_offset is not None:
            assert locator.char_offset >= 0


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------

SHEET_VALUES = st.lists(
    st.one_of(
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs", "Cc"), min_codepoint=32, max_codepoint=0x2FFFD
            ),
            max_size=24,
        ),
        st.integers(min_value=-10000, max_value=10000),
        st.booleans(),
    ),
    min_size=0,
    max_size=8,
)


@given(rows=st.lists(SHEET_VALUES, min_size=1, max_size=5), names=st.lists(st.text(alphabet="Sheet", min_size=1, max_size=6), min_size=1, max_size=3))
def test_spreadsheet_diagnostics_point_at_real_locations(tmp_path_factory, rows, names):
    """Blank sheets, formulas and truncated values still get true locations."""
    import openpyxl

    directory = tmp_path_factory.mktemp("diag_xlsx")
    path = directory / "sheet.xlsx"
    workbook = openpyxl.Workbook()
    for index, title in enumerate(dict.fromkeys(names)):
        sheet = workbook.active if index == 0 else workbook.create_sheet(title)
        sheet.title = title
        for r, row in enumerate(rows, start=1):
            for c, value in enumerate(row, start=1):
                if value is not None and value != "":
                    sheet.cell(row=r, column=c, value=value)
    workbook.save(str(path))

    result = _convert(path)
    if result is not None:
        _check_diagnostics(result)
        _check_locators(result)


@given(text=st.text(alphabet=st.characters(blacklist_categories=("Cs",), min_codepoint=32, max_codepoint=0x2FFFD), max_size=300))
def test_markdown_diagnostics_point_at_real_locations(tmp_path_factory, text):
    directory = tmp_path_factory.mktemp("diag_md")
    path = directory / "note.md"
    path.write_text(text, encoding="utf-8")
    result = _convert(path)
    if result is not None:
        _check_diagnostics(result)
        _check_locators(result)


@given(text=st.text(alphabet=st.characters(blacklist_categories=("Cs",), min_codepoint=32, max_codepoint=0x2FFFD), max_size=300))
def test_html_diagnostics_point_at_real_locations(tmp_path_factory, text):
    directory = tmp_path_factory.mktemp("diag_html")
    path = directory / "page.html"
    path.write_text(text, encoding="utf-8")
    result = _convert(path)
    if result is not None:
        _check_diagnostics(result)
        _check_locators(result)


@given(depth=st.integers(min_value=0, max_value=4))
def test_nested_html_diagnostics_point_at_real_locations(tmp_path_factory, depth):
    """Nesting is where a css_path or node id is most likely to be invented."""
    directory = tmp_path_factory.mktemp("diag_nested")
    path = directory / "deep.html"
    inner = "<p>leaf</p>"
    for _ in range(depth):
        inner = f"<div class='w'>{inner}</div>"
    path.write_text(f"<html><body>{inner}</body></html>", encoding="utf-8")
    result = _convert(path)
    if result is not None:
        _check_diagnostics(result)
        _check_locators(result)


@given(payload=st.binary(max_size=200))
def test_unreadable_documents_never_emit_a_diagnostic_for_a_place_that_does_not_exist(
    tmp_path_factory, payload
):
    """A refusal may still carry diagnostics; they must still be honest."""
    directory = tmp_path_factory.mktemp("diag_junk")
    path = directory / "input.bin"
    path.write_bytes(payload)
    result = _convert(path)
    if result is not None:
        _check_diagnostics(result)


# --------------------------------------------------------------------------
# The checker itself.
#
# A property that never fails proves nothing unless it is shown to be able to
# fail. These plant specific lies in a real conversion and require the checker
# to catch each one, so the suite cannot quietly decay into a loop over an
# empty list of diagnostics.
# --------------------------------------------------------------------------


def _lie(**kwargs) -> Diagnostic:
    return Diagnostic(
        code="fabricated",
        category=Category.STRUCTURAL,
        severity=Severity.WARNING,
        message="planted",
        **kwargs,
    )


def test_the_checker_sees_real_diagnostics(fx):
    """If this ever finds nothing, the property tests are vacuous."""
    result = convert(fx("sample.xlsx"))
    assert len(result.diagnostics.diagnostics) > 0
    _check_diagnostics(result)


def test_the_checker_catches_a_page_that_does_not_exist(fx):
    result = convert(fx("sample.xlsx"))
    result.diagnostics.diagnostics.append(_lie(scope_kind="page", scope_index=99))
    with pytest.raises(AssertionError, match="page 99"):
        _check_diagnostics(result)


def test_the_checker_catches_a_sheet_that_was_never_read(fx):
    result = convert(fx("sample.xlsx"))
    result.diagnostics.diagnostics.append(_lie(scope_kind="sheet", scope_name="Ghost"))
    with pytest.raises(AssertionError, match="Ghost"):
        _check_diagnostics(result)


def test_the_checker_catches_a_node_that_does_not_exist(fx):
    result = convert(fx("sample.xlsx"))
    result.diagnostics.diagnostics.append(_lie(node_id="d.does-not-exist"))
    with pytest.raises(AssertionError, match="does not exist"):
        _check_diagnostics(result)


def test_the_checker_catches_a_zero_based_index(fx):
    """Page and slide numbers are 1-based in a human-facing report."""
    result = convert(fx("simple.pdf"))
    result.diagnostics.diagnostics.append(_lie(scope_kind="page", scope_index=0))
    with pytest.raises(AssertionError, match="0-based"):
        _check_diagnostics(result)
