"""Ground truth for the synthetic corpus.

Every document here was *drawn from a spec*, so the expected output is known
exactly — there is nothing to guess and nothing to label by hand. That is what
makes accuracy measurable: this module defines the content, ``fixtures_builder``
renders it into files, and the benchmark scores both systems against the same
definition.

If the manifest and the fixtures were written independently they would drift,
and the resulting "accuracy" would be meaningless. They cannot: the builders
import their content from here.

Scope of the claim
------------------
This is a synthetic, author-controlled corpus. It measures whether a converter
recovers content that was put there deliberately, with known boundaries and no
scanning noise. It is a lower bound on real-world accuracy, not a substitute for
a labelled public corpus. The report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GROUND_TRUTH_SCHEMA = "docweave.groundtruth/1.0"


# --------------------------------------------------------------------------
# Content constants. These are the single source of truth for both the
# generated fixtures and the expected extraction.
# --------------------------------------------------------------------------

MARKDOWN_BODY = """# Title

Intro paragraph with **bold** and a [link](https://example.com).

## Section

- alpha
- beta
- gamma

1. first
2. second

> quoted text

```python
print('hi')
```

| name | qty |
| --- | --- |
| apple | 3 |
| pear | 7 |
"""

MARKDOWN_TITLE = "Title"
MARKDOWN_INTRO = "Intro paragraph with **bold** and a [link](https://example.com)."
MARKDOWN_SECTION = "Section"
MARKDOWN_BULLETS = ["alpha", "beta", "gamma"]
MARKDOWN_NUMBERS = ["first", "second"]
MARKDOWN_QUOTE = "quoted text"
MARKDOWN_CODE = "print('hi')"
MARKDOWN_CODE_LANG = "python"
MARKDOWN_TABLE_HEADER = ["name", "qty"]
MARKDOWN_TABLE_ROWS = [["apple", "3"], ["pear", "7"]]

HTML_BODY = """<!DOCTYPE html>
<html><head><title>Sample</title><style>body{color:red}</style></head>
<body>
<h1>Heading One</h1>
<p>First <b>paragraph</b> with <a href='https://example.com'>a link</a>.</p>
<script>alert('x')</script>
<h2>Heading Two</h2>
<ul><li>one</li><li>two</li></ul>
<table><caption>Cap</caption>
<tr><th>h1</th><th>h2</th></tr>
<tr><td>a</td><td>b</td></tr>
<tr><td>c</td><td>d</td></tr>
</table>
<pre><code class='language-json'>{"k": 1}</code></pre>
<img src='logo.png' alt='Logo'>
</body></html>
"""

HTML_H1 = "Heading One"
HTML_INTRO = "First paragraph with a link."
HTML_H2 = "Heading Two"
HTML_LIST = ["one", "two"]
HTML_TABLE_HEADER = ["h1", "h2"]
HTML_TABLE_ROWS = [["a", "b"], ["c", "d"]]
HTML_TABLE_CAPTION = "Cap"
HTML_CODE = '{"k": 1}'
HTML_IMAGE_ALT = "Logo"

CSV_BODY = (
    "name,qty,price\n"
    "apple,3,1.5\n"
    "pear,7,2.25\n"
    '"quoted, comma",1,0.99\n'
)
CSV_HEADER = ["name", "qty", "price"]
CSV_ROWS = [["apple", "3", "1.5"], ["pear", "7", "2.25"], ["quoted, comma", "1", "0.99"]]

RAGGED_CSV_BODY = "a,b,c\n1,2,3\n4,5\n6,7,8,9\n"
RAGGED_ROW_WIDTHS = [3, 3, 2, 4]

JSON_BODY = '{"z": 1, "a": [1, 2, {"b": null}], "s": "héllo wörld", "dup": 1, "dup": 2}'
#: What ``json.loads`` yields: the duplicate key keeps its last value.
JSON_EXPECTED = {"z": 1, "a": [1, 2, {"b": None}], "s": "héllo wörld", "dup": 2}

XML_BODY = '<?xml version="1.0" encoding="UTF-8"?>\n<root attr="1"><child>text</child></root>\n'
XML_TEXT = "text"

BOM_TEXT = "Heading\n\nBody text\n"

# -- PDF -----------------------------------------------------------------

PDF_TITLE = "Quarterly Report"
PDF_SUBHEADING = "Summary"
PDF_BODY = (
    "Revenue grew by 12 percent year over year, driven by the European segment. "
    "Costs remained flat."
)
PDF_FINDINGS = ["1. First finding", "2. Second finding"]
PDF_FOOTER = "CONFIDENTIAL"
PDF_PAGES = 4

PDF_TABLE_HEADER = ["Region", "Q1", "Q2"]
PDF_TABLE_ROWS = [["North", "100", "120"], ["South", "90", "95"], ["West", "80", "85"]]

PDF_COLUMN_LINES = 18
PDF_COLUMN_HEADERS = ["Column 1", "Column 2"]

#: Content of ``real_scan.pdf``: a page of ink with no text layer. This is the
#: only fixture whose text can only arrive through OCR, so it is the one that
#: says whether the OCR path is working or silently returning nothing.
SCAN_HEADING = "Quarterly Revenue Summary"
SCAN_BODY = (
    "Total revenue for the period was 4.2 million.",
    "Growth was driven by the services line.",
)

# -- DOCX ----------------------------------------------------------------

DOCX_TITLE = "Quarterly Report"
DOCX_SUBHEADING = "Summary"
DOCX_PARAGRAPH = "Revenue grew by 12 percent year over year."
DOCX_BULLETS = ["North region", "South region"]
DOCX_NUMBERS = ["First", "Second"]
DOCX_TABLE_HEADER = ["Region", "Total"]
DOCX_TABLE_ROWS = [["North", "100"], ["South", "90"]]
DOCX_CLOSING = "See the appendix for detail."

DOCX_ORDER_BEFORE = "before table"
DOCX_ORDER_HEADER = ["h1", "h2"]
DOCX_ORDER_AFTER = "after table"

DOCX_NOTE_BODY = "Revenue rose sharply"
DOCX_NOTES = ["Figures are unaudited.", "Definition of ARR provided."]

DOCX_IMAGE_CAPTION = "Figure below:"
DOCX_DIAGRAM_HEADING = "Process overview"
DOCX_DIAGRAM_LEAD = "The figure below is a flowchart."
DOCX_DIAGRAM_CLOSING = "End of document."

# -- PPTX ----------------------------------------------------------------

PPTX_SLIDE1_TITLE = "Quarterly Review"
PPTX_SLIDE1_SUBTITLE = "Prepared by the analysis team"
PPTX_SLIDE2_TITLE = "Key Findings"
PPTX_SLIDE2_BULLETS = ["Revenue up 12 percent", "Costs flat", "Headcount unchanged"]
PPTX_SLIDE3_TITLE = "Numbers"
PPTX_TABLE_HEADER = ["Region", "Total"]
PPTX_TABLE_ROWS = [["North", "100"], ["South", "90"]]
PPTX_SPEAKER_NOTES = "Discuss with finance first."
PPTX_SLIDE4_TITLE = "Appendix"

# -- XLSX ----------------------------------------------------------------

XLSX_SHEET = "Data"
XLSX_HIDDEN_SHEET = "Hidden"
XLSX_HEADER = ["Region", "Total"]
XLSX_ROWS = [["North", "100"], ["South", "=SUM(B2:B2)"]]
XLSX_ORPHAN = ("D5", "orphan note")
XLSX_HIDDEN_CELL = ("A1", "secret")


# --------------------------------------------------------------------------
# Ground truth records
# --------------------------------------------------------------------------


@dataclass
class ExpectedTable:
    header: list[str]
    rows: list[list[str]]
    #: True when rows deliberately differ in width.
    ragged: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows) + (1 if self.header else 0)

    @property
    def column_count(self) -> int:
        return len(self.header) if self.header else max((len(r) for r in self.rows), default=0)


@dataclass
class GroundTruth:
    """What a correct extraction of one document contains."""

    name: str
    fmt: str
    #: Reading-order blocks of expected content.
    blocks: list[str] = field(default_factory=list)
    tables: list[ExpectedTable] = field(default_factory=list)
    pages: int | None = None
    slides: int | None = None
    sheets: int | None = None
    #: Text deliberately absent from correct output (e.g. a stripped footer).
    excluded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def reference_text(self) -> str:
        return " ".join(self.blocks)


def _pdf_page_blocks(page_number: int) -> list[str]:
    return [
        PDF_TITLE,
        PDF_SUBHEADING,
        PDF_BODY,
        *PDF_FINDINGS,
        f"Page {page_number}",
    ]


def build_registry() -> dict[str, GroundTruth]:
    """Ground truth for every fixture that has well-defined expected content."""
    truth: dict[str, GroundTruth] = {}

    truth["sample.md"] = GroundTruth(
        name="sample.md",
        fmt="markdown",
        blocks=[
            MARKDOWN_TITLE,
            MARKDOWN_INTRO,
            MARKDOWN_SECTION,
            *MARKDOWN_BULLETS,
            *MARKDOWN_NUMBERS,
            MARKDOWN_QUOTE,
            MARKDOWN_CODE,
        ],
        tables=[
            ExpectedTable(
                header=MARKDOWN_TABLE_HEADER,
                rows=[list(r) for r in MARKDOWN_TABLE_ROWS],
            )
        ],
    )

    truth["sample.html"] = GroundTruth(
        name="sample.html",
        fmt="html",
        blocks=[
            HTML_H1,
            HTML_INTRO,
            HTML_H2,
            *HTML_LIST,
            HTML_CODE,
        ],
        tables=[
            ExpectedTable(
                header=HTML_TABLE_HEADER,
                rows=[list(r) for r in HTML_TABLE_ROWS],
            )
        ],
        notes=["The <script> body must not appear in the output."],
    )

    truth["sample.csv"] = GroundTruth(
        name="sample.csv",
        fmt="csv",
        blocks=[],
        tables=[
            ExpectedTable(
                header=CSV_HEADER,
                rows=[list(r) for r in CSV_ROWS],
            )
        ],
    )

    truth["ragged.csv"] = GroundTruth(
        name="ragged.csv",
        fmt="csv",
        blocks=[],
        tables=[
            ExpectedTable(
                header=["a", "b", "c"],
                rows=[["1", "2", "3"], ["4", "5"], ["6", "7", "8", "9"]],
                ragged=True,
            )
        ],
        notes=["Row widths intentionally differ; every cell must survive."],
    )

    truth["sample.json"] = GroundTruth(
        name="sample.json",
        fmt="json",
        # DocWeave preserves JSON verbatim by design. The scorer strips
        # punctuation, so the expected content is the values it contains.
        blocks=["1 a 1 2 null héllo wörld dup 2"],
        notes=["The duplicate 'dup' key collapses to its last value."],
    )

    truth["sample.xml"] = GroundTruth(
        name="sample.xml",
        fmt="xml",
        # XML is preserved verbatim for the same reason as JSON; the scorer
        # strips tags, so the expected *content* is the visible text.
        blocks=[XML_TEXT],
        notes=["The visible text 'text' is inside the <child> element."],
    )

    truth["bom.txt"] = GroundTruth(
        name="bom.txt", fmt="text", blocks=["Heading", "Body text"]
    )

    truth["simple.pdf"] = GroundTruth(
        name="simple.pdf",
        fmt="pdf",
        pages=PDF_PAGES,
        blocks=[b for n in range(1, PDF_PAGES + 1) for b in _pdf_page_blocks(n)],
        excluded=[PDF_FOOTER],
        notes=[
            f"'{PDF_FOOTER}' repeats on every page and is running furniture; the default "
            "configuration strips it, so it is excluded from correct output.",
        ],
    )

    truth["table.pdf"] = GroundTruth(
        name="table.pdf",
        fmt="pdf",
        pages=1,
        blocks=[],
        tables=[
            ExpectedTable(
                header=PDF_TABLE_HEADER,
                rows=[list(r) for r in PDF_TABLE_ROWS],
            )
        ],
    )

    truth["multicolumn.pdf"] = GroundTruth(
        name="multicolumn.pdf",
        fmt="pdf",
        pages=1,
        blocks=[
            PDF_COLUMN_HEADERS[0],
            *[f"Line {i + 1} of column 1" for i in range(PDF_COLUMN_LINES)],
            PDF_COLUMN_HEADERS[1],
            *[f"Line {i + 1} of column 2" for i in range(PDF_COLUMN_LINES)],
        ],
    )

    truth["order.docx"] = GroundTruth(
        name="order.docx",
        fmt="docx",
        blocks=[DOCX_ORDER_BEFORE, *DOCX_ORDER_HEADER, DOCX_ORDER_AFTER],
        tables=[ExpectedTable(header=DOCX_ORDER_HEADER, rows=[])],
    )

    truth["sample.docx"] = GroundTruth(
        name="sample.docx",
        fmt="docx",
        blocks=[
            DOCX_TITLE,
            DOCX_SUBHEADING,
            DOCX_PARAGRAPH,
            *DOCX_BULLETS,
            *DOCX_NUMBERS,
            DOCX_CLOSING,
        ],
        tables=[
            ExpectedTable(
                header=DOCX_TABLE_HEADER,
                rows=[list(r) for r in DOCX_TABLE_ROWS],
            )
        ],
    )

    truth["notes.docx"] = GroundTruth(
        name="notes.docx",
        fmt="docx",
        blocks=[DOCX_NOTE_BODY, *DOCX_NOTES],
        notes=["Footnote text must survive even though its inline anchor does not."],
    )

    truth["sample.pptx"] = GroundTruth(
        name="sample.pptx",
        fmt="pptx",
        slides=4,
        blocks=[
            PPTX_SLIDE1_TITLE,
            PPTX_SLIDE1_SUBTITLE,
            PPTX_SLIDE2_TITLE,
            *PPTX_SLIDE2_BULLETS,
            PPTX_SLIDE3_TITLE,
            PPTX_SPEAKER_NOTES,
            PPTX_SLIDE4_TITLE,
        ],
        tables=[
            ExpectedTable(
                header=PPTX_TABLE_HEADER,
                rows=[list(r) for r in PPTX_TABLE_ROWS],
            )
        ],
    )

    truth["sample.xlsx"] = GroundTruth(
        name="sample.xlsx",
        fmt="xlsx",
        sheets=2,
        blocks=[],
        tables=[
            ExpectedTable(
                header=XLSX_HEADER,
                rows=[list(r) for r in XLSX_ROWS] + [[XLSX_ORPHAN[1]]],
                ragged=True,
            )
        ],
        notes=[
            "Formula strings must be preserved verbatim; no cached result is ever computed.",
            "The hidden sheet is converted by default and must be present.",
        ],
    )

    truth["sparse.xlsx"] = GroundTruth(
        name="sparse.xlsx",
        fmt="xlsx",
        sheets=1,
        blocks=[],
        tables=[ExpectedTable(header=["a", "c"], rows=[["e"]], ragged=True)],
        notes=["Cells A1, C1 and E2 must each appear, at their true coordinates."],
    )

    truth["hostile.xlsx"] = GroundTruth(
        name="hostile.xlsx",
        fmt="xlsx",
        sheets=1,
        blocks=[],
        # Authored from the specification, not read back from DocWeave's output.
        # A1 holds one string that happens to contain the list format's own
        # separator; it is one cell, not two.
        tables=[
            ExpectedTable(
                header=["harbour; B1: docks", "plain"],
                rows=[
                    ['say "hi"; B1: x'],
                    ["alpha; beta", "right"],
                    ["line one line two"],
                    ["merged"],
                ],
                ragged=True,
            )
        ],
        notes=[
            "A1 is a single cell whose text contains '; B1: ' and must survive "
            "as one value.",
            "B5:C5 is merged; only the anchor carries a value.",
            "A4 contains a newline, which the one-row-per-line list format "
            "cannot represent, so it is expected flattened to a space.",
        ],
    )

    truth["formatted.xlsx"] = GroundTruth(
        name="formatted.xlsx",
        fmt="xlsx",
        sheets=1,
        blocks=[],
        # Authored from the spec: raw values, not the workbook's display strings.
        # A date is ISO with no clock time, because a whole-day cell carries no
        # time information worth emitting.
        tables=[
            ExpectedTable(
                header=["Item", "Amount", "When"],
                rows=[
                    ["Widget", "1234.5", "2026-03-14"],
                    ["Service", "99.99", "2026-12-01"],
                    ["Rate", "0.075"],
                    ["Total", "=SUM(B2:B3)"],
                ],
                ragged=True,
            )
        ],
        notes=[
            "B4 keeps 0.075, not the workbook's '7.5%': the raw value is kept so "
            "a consumer can format it, rather than a display string they cannot "
            "un-format.",
            "Dates must not carry a 00:00:00 clock time.",
        ],
    )

    truth["real_scan.pdf"] = GroundTruth(
        name="real_scan.pdf",
        fmt="pdf",
        pages=1,
        blocks=[SCAN_HEADING, *SCAN_BODY],
        notes=[
            "Pixels only, no text layer: this content is reachable through OCR or "
            "not at all.",
            "The heading is expected to be read as a heading, not as body text, "
            "because the OCR engine returns layout blocks with labels.",
        ],
    )

    truth["scanned.pdf"] = GroundTruth(
        name="scanned.pdf",
        fmt="pdf",
        pages=1,
        blocks=[],
        notes=[
            "The page has no text layer, so correct extraction is empty. Anything a "
            "system reports as text here is an invention.",
        ],
    )

    truth["image.docx"] = GroundTruth(
        name="image.docx", fmt="docx", blocks=[DOCX_IMAGE_CAPTION]
    )

    truth["diagram.docx"] = GroundTruth(
        name="diagram.docx",
        fmt="docx",
        blocks=[DOCX_DIAGRAM_HEADING, DOCX_DIAGRAM_LEAD, DOCX_DIAGRAM_CLOSING],
    )

    return truth


def reference_table_cells(truth: GroundTruth) -> set[str]:
    """Every cell text the document should yield, flattened."""
    cells: set[str] = set()
    for table in truth.tables:
        cells.update(c for c in table.header if c)
        for row in table.rows:
            cells.update(c for c in row if c)
    return cells
