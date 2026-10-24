"""Synthetic fixture builders.

Tests must not download anything and must not depend on external corpora, so
every fixture is generated here with the same libraries the adapters read.
Generated documents are committed under ``tests/fixtures`` for manual inspection
and used as the seed for the benchmark corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts ``tests/`` on sys.path, where an unrelated
# installed package called ``tests`` would shadow the local one. Put the repo
# root first so ``tests`` always resolves to this project.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests import ground_truth as GT  # noqa: E402

# Content is NOT written here. It lives in ``tests/ground_truth.py`` and both
# the fixtures and the benchmark import it, so a fixture and its expected
# extraction can never drift apart. A benchmark whose ground truth is a second
# hand-written copy of the content is worse than no benchmark.


# --------------------------------------------------------------------------
# text family
# --------------------------------------------------------------------------


def write_text(path: Path, body: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(body.encode(encoding))
    return path


def make_markdown(path: Path) -> Path:
    return write_text(path, GT.MARKDOWN_BODY)


def make_html(path: Path) -> Path:
    return write_text(path, GT.HTML_BODY)


def make_csv(path: Path) -> Path:
    return write_text(path, GT.CSV_BODY)


def make_ragged_csv(path: Path) -> Path:
    return write_text(path, GT.RAGGED_CSV_BODY)


def make_json(path: Path) -> Path:
    return write_text(path, GT.JSON_BODY)


def make_xml(path: Path) -> Path:
    return write_text(path, GT.XML_BODY)


def make_bad_xml(path: Path) -> Path:
    return write_text(path, "<root><unclosed></root>")


def make_bom_text(path: Path) -> Path:
    path.write_bytes(b"\xef\xbb\xbf" + GT.BOM_TEXT.encode("utf-8"))
    return path


def make_binary_pretending_to_be_csv(path: Path) -> Path:
    path.write_bytes(b"col1,col2\n" + bytes(range(0, 32)) * 4 + b"\n")
    return path


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def make_pdf_simple(path: Path) -> Path:
    """A four page text PDF with a heading hierarchy and repeated page furniture."""
    import pymupdf

    doc = pymupdf.open()
    for page_number in range(1, 5):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 90), GT.PDF_TITLE, fontsize=22, fontname="hebo")
        page.insert_text((72, 130), GT.PDF_SUBHEADING, fontsize=16, fontname="hebo")
        page.insert_text((72, 170), GT.PDF_BODY, fontsize=11, fontname="helv")
        page.insert_text((72, 220), GT.PDF_FINDINGS[0], fontsize=11, fontname="helv")
        page.insert_text((72, 240), GT.PDF_FINDINGS[1], fontsize=11, fontname="helv")
        # Same text in the bottom band on every page: a running footer.
        page.insert_text((72, 790), GT.PDF_FOOTER, fontsize=8, fontname="helv")
        # Differs per page, so it must NOT be treated as running furniture.
        page.insert_text((500, 30), f"Page {page_number}", fontsize=8, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def make_pdf_multicolumn(path: Path) -> Path:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for column, x in enumerate((60, 320)):
        page.insert_text((x, 100), GT.PDF_COLUMN_HEADERS[column], fontsize=14, fontname="hebo")
        for i in range(18):
            page.insert_text(
                (x, 130 + i * 18), f"Line {i + 1} of column {column + 1}", fontsize=9, fontname="helv"
            )
    doc.save(str(path))
    doc.close()
    return path


def make_pdf_table(path: Path) -> Path:
    """A page with a ruled table, drawn with real lines."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=400)
    rows, cols = 4, 3
    x0, y0, w, h = 60, 80, 420, 24
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * h), (x0 + w, y0 + r * h), width=0.8)
    for c in range(cols + 1):
        page.draw_line((x0 + c * (w / cols), y0), (x0 + c * (w / cols), y0 + rows * h), width=0.8)
    labels = [GT.PDF_TABLE_HEADER, *[list(r) for r in GT.PDF_TABLE_ROWS]]
    for r, row in enumerate(labels):
        for c, value in enumerate(row):
            page.insert_text(
                (x0 + c * (w / cols) + 6, y0 + r * h + 16), value, fontsize=10, fontname="helv"
            )
    doc.save(str(path))
    doc.close()
    return path


def make_pdf_scanned_looking(path: Path) -> Path:
    """A page with no text layer but a big raster image covering the page."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Generate a noisy raster so the page has image area but no glyphs.
    import random

    random.seed(7)
    width, height = 60, 84
    pixels = bytearray()
    for _ in range(width * height):
        value = random.randint(0, 255)
        pixels.extend((value, value, value))
    image_bytes = _png(width, height, bytes(pixels))
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=image_bytes)
    doc.save(str(path))
    doc.close()
    return path


def make_pdf_empty_page(path: Path) -> Path:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=300, height=300)
    doc.new_page(width=300, height=300)
    doc.save(str(path))
    doc.close()
    return path


def make_pdf_encrypted(path: Path) -> Path:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret", fontsize=12)
    doc.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
        permissions=pymupdf.PDF_PERM_PRINT,
    )
    doc.close()
    return path


def make_pdf_corrupt(path: Path) -> Path:
    """A file that starts with the PDF signature but is structurally broken."""
    good = make_pdf_simple(path.with_suffix(".tmp.pdf"))
    data = good.read_bytes()
    good.unlink()
    path.write_bytes(data[: len(data) // 2])
    return path


def _png(width: int, height: int, gray: bytes) -> bytes:
    """Minimal 8-bit grayscale PNG encoder (no Pillow dependency in tests)."""
    import struct
    import zlib

    raw = bytearray()
    stride = width
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(gray[y * stride : (y + 1) * stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------


def _mark_header_row(table, row_index: int = 0) -> None:
    """Set w:tblHeader so the adapter can recognise a repeating header row."""
    from docx.oxml.ns import qn

    row = table.rows[row_index]
    tr_pr = row._tr.get_or_add_trPr()
    header = tr_pr.makeelement(qn("w:tblHeader"), {})
    tr_pr.append(header)


def make_docx(path: Path) -> Path:
    import docx

    document = docx.Document()
    document.core_properties.title = GT.DOCX_TITLE
    document.core_properties.author = "DocWeave Tests"

    document.add_heading(GT.DOCX_TITLE, level=1)
    document.add_heading(GT.DOCX_SUBHEADING, level=2)
    document.add_paragraph(GT.DOCX_PARAGRAPH)

    bullets = document.add_paragraph(GT.DOCX_BULLETS[0], style="List Bullet")
    del bullets
    document.add_paragraph(GT.DOCX_BULLETS[1], style="List Bullet")
    document.add_paragraph(GT.DOCX_NUMBERS[0], style="List Number")
    document.add_paragraph(GT.DOCX_NUMBERS[1], style="List Number")

    table = document.add_table(rows=3, cols=2)
    table.style = "Table Grid"
    rows = [GT.DOCX_TABLE_HEADER, *[list(r) for r in GT.DOCX_TABLE_ROWS]]
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    _mark_header_row(table)

    document.add_paragraph(GT.DOCX_CLOSING)
    document.save(str(path))
    return path


def make_docx_with_order(path: Path) -> Path:
    """Paragraphs and tables interleaved, to prove body order is preserved."""
    import docx

    document = docx.Document()
    document.add_paragraph(GT.DOCX_ORDER_BEFORE)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = GT.DOCX_ORDER_HEADER[0]
    table.cell(0, 1).text = GT.DOCX_ORDER_HEADER[1]
    document.add_paragraph(GT.DOCX_ORDER_AFTER)
    document.save(str(path))
    return path


def make_docx_with_footnotes(path: Path) -> Path:
    """A DOCX with a real footnote part, injected into the package directly."""
    import zipfile

    import docx
    from docx.oxml.ns import qn

    document = docx.Document()
    document.add_paragraph("Revenue rose.")
    document.save(str(path))

    footnotes_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b'<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
        b'<w:footnote w:type="continuationSeparator" w:id="0"><w:p/></w:footnote>'
        b"<w:footnote w:id=\"2\"><w:p><w:r><w:t xml:space=\"preserve\">Figures are unaudited."
        b"</w:t></w:r></w:p></w:footnote>"
        b'<w:footnote w:id="1"><w:p><w:r><w:t xml:space="preserve">Definition of ARR provided.'
        b"</w:t></w:r></w:p></w:footnote>"
        b"</w:footnotes>"
    )

    document_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body>"
        b"<w:p><w:r><w:t>Revenue rose</w:t></w:r>"
        b'<w:r><w:footnoteReference w:id="2"/></w:r>'
        b"<w:r><w:t xml:space=\"preserve\"> sharply</w:t></w:r>"
        b'<w:r><w:footnoteReference w:id="1"/></w:r>'
        b"</w:p>"
        b"<w:sectPr/>"
        b"</w:body></w:document>"
    )

    content_types = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        b'<Default Extension="xml" ContentType="application/xml"/>'
        b'<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
        b'officedocument.wordprocessingml.document.main+xml"/>'
        b'<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-'
        b'officedocument.wordprocessingml.footnotes+xml"/>'
        b"</Types>"
    )

    rels = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        b'relationships/officeDocument" Target="word/document.xml"/>'
        b"</Relationships>"
    )

    doc_rels = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        b'<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        b'relationships/footnotes" Target="footnotes.xml"/>'
        b"</Relationships>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/footnotes.xml", footnotes_xml)
    del qn
    return path


def make_docx_with_image(path: Path) -> Path:
    import docx

    document = docx.Document()
    document.add_paragraph(GT.DOCX_IMAGE_CAPTION)
    document.add_picture(_io_bytes(_png(4, 4, bytes(range(16, 32)))))
    document.save(str(path))
    return path


def make_pptx_with_image(path: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Chart page"
    slide.shapes.add_picture(
        _io_bytes(_png(4, 4, bytes(range(64)))), Inches(1), Inches(2), Inches(2), Inches(2)
    )
    presentation.save(str(path))
    return path


def _io_bytes(data: bytes):
    import io

    return io.BytesIO(data)


def make_docx_with_diagram(path: Path, diagram_dir: Path | None = None) -> Path:
    """A DOCX whose body image is an actual diagram, for engine integration."""
    import docx

    source = (diagram_dir or Path(__file__).parent / "fixtures" / "diagrams") / "flowchart.png"
    if not source.exists():
        from tests.diagram_corpus import flowchart, render, write_png

        source = write_png(render(flowchart()), source)

    document = docx.Document()
    document.add_heading(GT.DOCX_DIAGRAM_HEADING, level=1)
    document.add_paragraph(GT.DOCX_DIAGRAM_LEAD)
    document.add_picture(str(source))
    document.add_paragraph(GT.DOCX_DIAGRAM_CLOSING)
    document.save(str(path))
    return path


def make_broken_docx(path: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
    return path


# --------------------------------------------------------------------------
# PPTX
# --------------------------------------------------------------------------


def make_pptx(path: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    presentation = Presentation()
    title_layout = presentation.slide_layouts[0]
    bullet_layout = presentation.slide_layouts[1]

    slide = presentation.slides.add_slide(title_layout)
    slide.shapes.title.text = GT.PPTX_SLIDE1_TITLE
    slide.placeholders[1].text = GT.PPTX_SLIDE1_SUBTITLE

    slide = presentation.slides.add_slide(bullet_layout)
    slide.shapes.title.text = GT.PPTX_SLIDE2_TITLE
    body = slide.placeholders[1].text_frame
    body.text = GT.PPTX_SLIDE2_BULLETS[0]
    for text in GT.PPTX_SLIDE2_BULLETS[1:]:
        para = body.add_paragraph()
        para.text = text
        para.level = 1

    slide = presentation.slides.add_slide(bullet_layout)
    slide.shapes.title.text = GT.PPTX_SLIDE3_TITLE
    table = slide.shapes.add_table(
        3, 2, Inches(1), Inches(2), Inches(6), Inches(2)
    ).table
    for r, row in enumerate([GT.PPTX_TABLE_HEADER, *[list(x) for x in GT.PPTX_TABLE_ROWS]]):
        table.cell(r, 0).text = row[0]
        table.cell(r, 1).text = row[1]

    slide.notes_slide.notes_text_frame.text = GT.PPTX_SPEAKER_NOTES

    fourth = presentation.slides.add_slide(bullet_layout)
    fourth.shapes.title.text = GT.PPTX_SLIDE4_TITLE
    del fourth

    presentation.save(str(path))
    del Pt
    return path


def make_empty_pptx(path: Path) -> Path:
    from pptx import Presentation

    Presentation().save(str(path))
    return path


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------


def make_xlsx(path: Path) -> Path:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = GT.XLSX_SHEET
    sheet["A1"] = GT.XLSX_HEADER[0]
    sheet["B1"] = GT.XLSX_HEADER[1]
    sheet["A2"] = GT.XLSX_ROWS[0][0]
    sheet["B2"] = int(GT.XLSX_ROWS[0][1])
    sheet["A3"] = GT.XLSX_ROWS[1][0]
    sheet["B3"] = GT.XLSX_ROWS[1][1]
    sheet[GT.XLSX_ORPHAN[0]] = GT.XLSX_ORPHAN[1]
    sheet.merge_cells("A6:B6")

    hidden = workbook.create_sheet(GT.XLSX_HIDDEN_SHEET)
    hidden[GT.XLSX_HIDDEN_CELL[0]] = GT.XLSX_HIDDEN_CELL[1]
    hidden.sheet_state = "hidden"

    workbook.save(str(path))
    return path


def make_xlsx_sparse(path: Path) -> Path:
    """Sparse rows: only some cells carry values, and coordinates must survive."""
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sparse"
    sheet["A1"] = "a"
    sheet["C1"] = "c"
    sheet["E2"] = "e"
    workbook.save(str(path))
    return path


def make_xlsx_hostile(path: Path) -> Path:
    """Values and structures chosen to break a faithful extractor.

    Every case here was found by probing the engine, not by imagining a
    document that might exist. If the corpus contains only shapes the engine
    already handles, a perfect score measures nothing.
    """
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Hostile"
    # A value containing the list format's own separator followed by a cell
    # label. Unquoted this renders identically to a two-cell row.
    sheet["A1"] = "harbour; B1: docks"
    sheet["B1"] = "plain"
    # A value containing double quotes as well.
    sheet["A2"] = 'say "hi"; B1: x'
    # A semicolon that cannot be mistaken for a separator.
    sheet["A3"] = "alpha; beta"
    sheet["D3"] = "right"
    # A value with an embedded newline: newlines cannot be represented in a
    # one-line-per-row list format.
    sheet["A4"] = "line one\nline two"
    # Merged range: the anchor carries the value and the covered cell has none.
    sheet["B5"] = "merged"
    sheet.merge_cells("B5:C5")
    workbook.save(str(path))
    return path


def make_xlsx_formatted(path: Path) -> Path:
    """Number formats, dates and a real formula cache.

    The other spreadsheet fixtures all use plain strings, so they never test
    what happens to a cell whose value is a date, a currency amount, a
    percentage, or a formula whose result the writer actually computed. Those
    are exactly the cells where a converter either loses information or starts
    inventing it.
    """
    from datetime import date

    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Report"
    sheet["A1"] = "Item"
    sheet["B1"] = "Amount"
    sheet["C1"] = "When"
    sheet["A2"] = "Widget"
    sheet["B2"] = 1234.5
    sheet["B2"].number_format = "#,##0.00"
    sheet["C2"] = date(2026, 3, 14)
    sheet["C2"].number_format = "yyyy-mm-dd"
    sheet["A3"] = "Service"
    sheet["B3"] = 99.99
    sheet["B3"].number_format = "$#,##0.00"
    sheet["C3"] = date(2026, 12, 1)
    sheet["C3"].number_format = "yyyy-mm-dd"
    sheet["A4"] = "Rate"
    sheet["B4"] = 0.075
    sheet["B4"].number_format = "0.0%"
    # A formula whose cached value openpyxl cannot compute is preserved but
    # unevaluated; this one is here for the cached-result path.
    sheet["A5"] = "Total"
    sheet["B5"] = "=SUM(B2:B3)"
    workbook.save(str(path))
    return path


# --------------------------------------------------------------------------
# misc / hostile inputs
# --------------------------------------------------------------------------


def make_pdf_real_scan(path: Path) -> Path:
    """A page that is genuinely pixels: a heading and two sentences, no text layer.

    ``scanned.pdf`` is blank, so it can prove the engine reports an empty page
    but never that it reads one. This is the fixture that checks the second
    half, and it has known content in the ground truth so a regression in the
    OCR path shows up as lost text rather than as a silently shorter report.

    The text is drawn onto a page, that page is rasterised, and the result is
    written back as a new document holding only the image. Writing the text
    straight into the file would leave a text layer behind, and the file would
    pass without any OCR running at all -- which is the trap this fixture
    exists to avoid.
    """
    import pymupdf

    source = pymupdf.open()
    page = source.new_page(width=595, height=842)
    page.draw_rect(
        pymupdf.Rect(60, 80, 535, 180), color=(0, 0, 0), fill=(0.92, 0.92, 0.92)
    )
    page.insert_text((72, 120), GT.SCAN_HEADING, fontsize=22, fontname="helv")
    page.insert_text((72, 150), GT.SCAN_BODY[0], fontsize=11, fontname="helv")
    page.insert_text((72, 168), GT.SCAN_BODY[1], fontsize=11, fontname="helv")

    pixmap = page.get_pixmap(dpi=150)
    rasterised = pymupdf.open()
    scanned_page = rasterised.new_page(width=595, height=842)
    scanned_page.insert_image(scanned_page.rect, pixmap=pixmap)
    rasterised.save(str(path))
    rasterised.close()
    source.close()
    return path


def make_pdf_control_chars(path: Path) -> Path:
    """A text layer carrying control and zero-width characters.

    Added after a real 37-page PDF put ``\\x08`` between a contents entry and
    its page number, and ``U+00AD``/``U+200B`` through the body. Every
    synthetic fixture before this one used plain ASCII, so nothing caught it.
    The characters must come out as a space, because here they are the
    separator PyMuPDF drops when it concatenates spans.
    """
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=420, height=240)
    page.insert_text((40, 90), "Contents", fontsize=16, fontname="helv")
    page.insert_text((40, 130), "Introduction\x08 1", fontsize=11, fontname="helv")
    page.insert_text((40, 160), "hyper­feminine", fontsize=11, fontname="helv")
    page.insert_text((40, 190), "zerowidth", fontsize=11, fontname="helv")
    document.save(str(path))
    document.close()
    return path


def make_empty_file(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def make_not_really_zip(path: Path) -> Path:
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    return path


def make_fake_pdf_extension(path: Path) -> Path:
    """A CSV named .pdf: detection must follow content, not the extension."""
    return write_text(path, "name,qty\napple,3\n")


def make_large_csv(path: Path, rows: int = 5000) -> Path:
    lines = ["i,value"]
    lines.extend(f"{i},{i * 3}" for i in range(rows))
    return write_text(path, "\n".join(lines) + "\n")


def build_all(directory: Path) -> dict[str, Path]:
    """Build every fixture into ``directory`` and return a name -> path map."""
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    builders = {
        "sample.md": make_markdown,
        "sample.html": make_html,
        "sample.csv": make_csv,
        "ragged.csv": make_ragged_csv,
        "sample.json": make_json,
        "sample.xml": make_xml,
        "bad.xml": make_bad_xml,
        "bom.txt": make_bom_text,
        "binary.csv": make_binary_pretending_to_be_csv,
        "simple.pdf": make_pdf_simple,
        "multicolumn.pdf": make_pdf_multicolumn,
        "table.pdf": make_pdf_table,
        "scanned.pdf": make_pdf_scanned_looking,
        "empty_pages.pdf": make_pdf_empty_page,
        "encrypted.pdf": make_pdf_encrypted,
        "corrupt.pdf": make_pdf_corrupt,
        "real_scan.pdf": make_pdf_real_scan,
        "control_chars.pdf": make_pdf_control_chars,
        "sample.docx": make_docx,
        "order.docx": make_docx_with_order,
        "image.docx": make_docx_with_image,
        "diagram.docx": make_docx_with_diagram,
        "notes.docx": make_docx_with_footnotes,
        "image.pptx": make_pptx_with_image,
        "broken.docx": make_broken_docx,
        "sample.pptx": make_pptx,
        "empty.pptx": make_empty_pptx,
        "sample.xlsx": make_xlsx,
        "sparse.xlsx": make_xlsx_sparse,
        "hostile.xlsx": make_xlsx_hostile,
        "formatted.xlsx": make_xlsx_formatted,
        "empty.bin": make_empty_file,
        "notzip.docx": make_not_really_zip,
        "liar.pdf": make_fake_pdf_extension,
        "large.csv": make_large_csv,
    }
    for name, builder in builders.items():
        out[name] = builder(directory / name)
    return out


if __name__ == "__main__":  # pragma: no cover - manual regeneration
    target = Path(__file__).parent / "fixtures"
    built = build_all(target)
    for name in sorted(built):
        print(name, built[name].stat().st_size)
