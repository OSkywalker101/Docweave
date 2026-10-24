"""OCR repair: only touch what was measured to be broken, never lose content."""

from __future__ import annotations

import numpy as np
import pytest

from docweave import ConversionConfig, convert
from docweave.models import ImageBlock, Page, PageSignals, Paragraph
from docweave.models.diagnostics import Code, Severity
from docweave.repair import (
    IndraEngine,
    OcrRepairer,
    TesseractEngine,
    default_engine,
    select_pages,
)


class FakeEngine:
    """A deterministic stand-in so repair logic is testable without Tesseract."""

    name = "fake"

    def __init__(self, text="REPAIRED PAGE TEXT", confidence=0.91, available=True):
        self.text = text
        self.confidence = confidence
        self._available = available
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def read(self, page):
        self.calls += 1
        return self.text, self.confidence


def _document(pages):
    from docweave.models import Document

    return Document(adapter="test", container_kind="pages", pages=pages)


def _scanned_page(index=1, chars=0, images=1, ratio=0.95):
    page = Page(
        index=index,
        width=595,
        height=842,
        signals=PageSignals(char_count=chars, image_count=images, media_area_ratio=ratio),
        images=[ImageBlock(id=f"i{index}", sources=[])],
    )
    return page


def _render(page_number):
    return np.full((10, 10, 3), 255, dtype=np.uint8)


class TestPageSelection:
    def test_scanned_pages_are_selected(self):
        document = _document([_scanned_page(1), _scanned_page(2)])
        assert select_pages(document, ConversionConfig()) == [1, 2]

    def test_text_pages_are_untouched(self):
        page = _scanned_page(chars=800)
        assert select_pages(_document([page]), ConversionConfig()) == []

    def test_textless_pages_without_images_are_left_alone(self):
        """A blank page has nothing to OCR; OCR would only hallucinate."""
        page = Page(index=1, signals=PageSignals(char_count=0, image_count=0))
        assert select_pages(_document([page]), ConversionConfig()) == []

    def test_threshold_is_configurable(self):
        document = _document([_scanned_page(chars=100)])
        assert select_pages(document, ConversionConfig(scanned_max_chars=10)) == []
        assert select_pages(document, ConversionConfig(scanned_max_chars=200)) == [1]


class TestRepair:
    def test_repaired_page_gets_text(self):
        document = _document([_scanned_page()])
        repairer = OcrRepairer(engine=FakeEngine())
        report = repairer.repair(document, _render)
        assert report.repaired_pages == [1]
        assert "REPAIRED PAGE TEXT" in document.pages[0].blocks[0].text

    def test_repaired_content_is_marked_as_ocr(self):
        document = _document([_scanned_page()])
        OcrRepairer(engine=FakeEngine()).repair(document, _render)
        source = document.pages[0].blocks[0].sources[0]
        assert source.adapter == "ocr:fake"
        assert source.origin.value == "inferred"
        assert "never exact" in source.note

    def test_repair_is_disclosed(self):
        document = _document([_scanned_page()])
        report = OcrRepairer(engine=FakeEngine()).repair(document, _render)
        assert any(d.code == Code.OCR_REPAIRED for d in report.diagnostics)
        repair_diag = next(
            d for d in report.diagnostics if "repaired by OCR" in d.message
        )
        assert repair_diag.evidence["engine"] == "fake"
        assert repair_diag.evidence["before_chars"] == 0

    def test_missing_engine_is_reported_not_silently_skipped(self):
        document = _document([_scanned_page()])
        report = OcrRepairer(engine=FakeEngine(available=False)).repair(document, _render)
        assert report.repaired_pages == []
        assert report.skipped_pages == [1]
        diagnostic = next(d for d in report.diagnostics if d.code == Code.OCR_UNAVAILABLE)
        assert diagnostic.severity is Severity.WARNING
        assert "Tesseract" in diagnostic.hint

    def test_empty_ocr_result_is_rejected(self):
        document = _document([_scanned_page()])
        report = OcrRepairer(engine=FakeEngine(text="   ")).repair(document, _render)
        assert report.repaired_pages == []
        assert report.rejected_pages[0]["reason"] == "OCR returned no text"

    def test_ocr_that_loses_content_is_rejected(self):
        """The original extraction must never be replaced by less text."""
        # Selected for repair (12 chars is under the threshold) but the OCR
        # result is smaller still, so the original has to win.
        page = _scanned_page(chars=12)
        page.blocks.append(Paragraph(id="p1.b0", text="ORIGINAL TWELVE CHARS"))
        document = _document([page])
        report = OcrRepairer(engine=FakeEngine(text="tiny")).repair(document, _render)
        assert report.repaired_pages == []
        assert "less text" in report.rejected_pages[0]["reason"]
        # The original block survives untouched.
        assert document.pages[0].blocks[0].text == "ORIGINAL TWELVE CHARS"

    def test_good_pages_are_never_rasterised(self):
        document = _document([_scanned_page(), _scanned_page(index=2, chars=900)])
        engine = FakeEngine()
        report = OcrRepairer(engine=engine).repair(document, _render)
        assert engine.calls == 1
        assert report.repaired_pages == [1]

    def test_clean_document_does_nothing(self):
        document = _document([_scanned_page(chars=900)])
        engine = FakeEngine()
        report = OcrRepairer(engine=engine).repair(document, _render)
        assert engine.calls == 0
        assert report.repaired_pages == []

    def test_unrenderable_page_is_skipped(self):
        document = _document([_scanned_page()])
        report = OcrRepairer(engine=FakeEngine()).repair(document, lambda n: None)
        assert report.skipped_pages == [1]
        assert report.repaired_pages == []

    def test_original_provenance_is_preserved(self):
        page = _scanned_page(chars=12)
        page.blocks.append(
            Paragraph(
                id="p1.b0",
                text="original",
                sources=[{"adapter": "pymupdf", "origin": "observed", "confidence": "high"}],
            )
        )
        document = _document([page])
        OcrRepairer(engine=FakeEngine()).repair(document, _render)
        adapters = {s.adapter for s in document.pages[0].blocks[0].sources}
        assert "ocr:fake" in adapters and "pymupdf" in adapters

    def test_repair_is_off_by_default(self):
        assert ConversionConfig().ocr_repair is False

    def test_metadata_records_the_repair(self):
        document = _document([_scanned_page()])
        OcrRepairer(engine=FakeEngine()).repair(document, _render)
        # The repairer only touches the document; the converter records metadata.
        assert "ocr_repaired_pages" in document.metadata


class TestTesseractEngine:
    def test_availability_is_a_bool(self):
        assert isinstance(TesseractEngine().available(), bool)

    def test_engine_name_is_stable(self):
        assert TesseractEngine().name == "tesseract"


class TestIndraEngine:
    """The scan path, with content that can only arrive through OCR."""

    def test_indra_is_the_preferred_engine_when_installed(self):
        from docweave.repair import default_engine

        engine = default_engine()
        if IndraEngine().available():
            assert engine.name == "indra"
        else:
            assert engine.name == "tesseract"

    def test_a_page_of_ink_is_read(self, fx):
        """The half of OCR that ``scanned.pdf`` cannot prove.

        That fixture is blank, so it only ever shows the engine reporting an
        empty page. This one has known words drawn as pixels.
        """
        result = convert(fx("real_scan.pdf"), ConversionConfig(ocr_repair=True))
        text = result.document.text_length()
        if not default_engine().available():
            pytest.skip("no OCR engine installed")
        assert text > 40, f"OCR recovered almost nothing: {text} chars"

    def test_the_scanned_text_is_recovered(self, fx):
        """Indra returns layout labels; this asserts the text, not the structure."""
        if not default_engine().available():
            pytest.skip("no OCR engine installed")
        out = convert(fx("real_scan.pdf"), ConversionConfig(ocr_repair=True)).to_markdown()
        assert "Quarterly Revenue Summary" in out
        assert "4.2 million" in out

    def test_layout_labels_become_node_kinds(self, fx):
        """The reason to prefer a document OCR engine over a line recogniser.

        Tesseract returns a flat list of lines, so a heading arrives as another
        line of body text. Indra returns labelled layout blocks, and the whole
        point of this project is that structure survives conversion -- a flat
        blob of page text throws away exactly what was recovered.
        """
        if not default_engine().available():
            pytest.skip("no OCR engine installed")
        out = convert(fx("real_scan.pdf"), ConversionConfig(ocr_repair=True)).to_markdown()
        assert "# Quarterly Revenue Summary" in out, "heading lost its kind"

    def test_a_slab_of_blocks_becomes_nodes_not_one_paragraph(self, fx):
        """One engine call may produce several labelled blocks."""
        from docweave.repair import _nodes_from_blocks

        blocks = [
            {"label": "Section-Header", "text": "Findings"},
            {"label": "List-Group", "text": "- one\n- two"},
            {"label": "Caption", "text": "Table 1: revenue"},
            {"label": "", "text": ""},
        ]
        nodes = _nodes_from_blocks(blocks, 1, 0.9, "indra")
        kinds = [type(n).__name__ for n in nodes]
        assert kinds == ["Heading", "ListBlock", "Paragraph"], kinds
        assert [item.text for item in nodes[1].items] == ["one", "two"]

    def test_bullets_are_stripped_once_not_twice(self):
        from docweave.repair import _strip_bullet

        assert _strip_bullet("- Revenue grew") == "Revenue grew"
        assert _strip_bullet("1. First") == "First"
        assert _strip_bullet("plain text") == "plain text"

    def test_unknown_labels_become_text_not_guesses(self):
        """An unrecognised label must not be invented into a structure."""
        from docweave.repair import _nodes_from_blocks

        nodes = _nodes_from_blocks(
            [{"label": "Something-New", "text": "body"}], 1, 0.9, "indra"
        )
        assert [type(n).__name__ for n in nodes] == ["Paragraph"]

    def test_the_fixture_really_has_no_text_layer(self, fx):
        """Guards the guard.

        A scan fixture built by writing text into the PDF has a text layer and
        passes every OCR assertion without any OCR running. This fails if that
        ever happens again.
        """
        import pymupdf

        with pymupdf.open(str(fx("real_scan.pdf"))) as document:
            assert document[0].get_text().strip() == ""
        assert convert(fx("real_scan.pdf")).document.text_length() == 0

    def test_ocr_never_invents_text_for_a_blank_page(self, fx):
        result = convert(fx("scanned.pdf"), ConversionConfig(ocr_repair=True))
        assert result.document.text_length() == 0


class TestIntegration:
    def test_convert_with_repair_enabled_but_engine_missing(self, fx, monkeypatch):
        """End-to-end: the option exists, and a missing engine degrades cleanly.

        The engine is forced unavailable rather than relying on Tesseract being
        absent. This test originally passed only because the binary was missing
        on the machine that wrote it -- the moment a real install appeared it
        took the OCR-succeeded branch instead and failed. A test whose outcome
        depends on what is or is not installed is not a test.
        """
        monkeypatch.setattr(TesseractEngine, "available", lambda self: False)
        # Both backends, not just Tesseract: the default engine is whatever is
        # installed, so patching one of them would leave the other in charge and
        # the test would pass or fail depending on the machine.
        monkeypatch.setattr(IndraEngine, "available", lambda self: False)
        result = convert(fx("scanned.pdf"), ConversionConfig(ocr_repair=True))
        codes = {d.code for d in result.diagnostics.diagnostics}
        assert Code.OCR_UNAVAILABLE in codes
        # Degrading must not mean silently returning nothing: the page is still
        # reported as empty and why.
        assert result.document.text_length() == 0
        assert Code.NO_CONTENT in codes

    def test_repair_leaves_a_clean_document_untouched(self, fx):
        result = convert(fx("simple.pdf"), ConversionConfig(ocr_repair=True))
        assert "ocr" not in result.document.metadata
