# Dependency and license inventory

All runtime dependencies are permissively licensed. No dependency requires a GPU
or downloads model weights.

## Base install

| Package | License | Why |
|---|---|---|
| pydantic | MIT | Typed, validating IR models; strict `extra="forbid"` gives us a real schema contract for the JSON output |
| click | BSD-3-Clause | CLI framework; stable and far less dependency-heavy than alternatives |
| rich | MIT | Terminal tables and colour for the diagnostics report |

## Format adapters

| Package | Extra | License | Why it is here |
|---|---|---|---|
| beautifulsoup4 | base | MIT | HTML parsing with the safe pure-Python backend |
| defusedxml | base | PSF | XML parsing that refuses entity expansion and external entities |
| PyMuPDF (AGPL-3.0) | `pdf` | AGPL-3.0 ⚠ | Fastest permissively-usable PDF text+geometry extractor. **AGPL has copyleft obligations for network use** — see the note below |
| python-docx | `office` | MIT | DOCX object model; heading styles, lists, tables |
| python-pptx | `office` | MIT | PPTX object model; slides, shapes, charts, notes |
| openpyxl | `office` | MIT | XLSX reading with formula strings preserved |
| pytesseract | `ocr` | Apache-2.0 | Fallback OCR wrapper (requires a Tesseract binary) |

### ⚠ OCR engines — read this before shipping

The optional OCR pass uses **Indra OCR** by default and falls back to
Tesseract. They are not interchangeable legally, and the difference matters.

| | Indra OCR | Tesseract |
|---|---|---|
| Package | `indra-ocr` | `pytesseract` + binary |
| Extra | `ocr` | `ocr` |
| Code licence | Apache-2.0 | Apache-2.0 |
| **Model weights** | **modified OpenRAIL-M** ⚠ | Apache-2.0 |
| Runtime | ONNX Runtime, no PyTorch | native binary |
| Model size | ~18 MB, downloaded from Hugging Face on first run | none |
| Text score (OmniParseBench) | 41.3 | 19.3 |
| Layout blocks + reading order | yes | no (flat lines) |

**The weight licence is the constraint that decides whether you can use this.**
Indra's weights are under a modified OpenRAIL-M: free for research, personal
use, and companies under $20M revenue/funding, and **not usable competitively
with Datalab's own API**. DocWeave's code is MIT and the `indra-ocr` code is
Apache-2.0, but anyone shipping a build that downloads those weights inherits
their terms. Set `INDRA_OCR_MODEL_DIR` to your own copy to avoid the download,
and read the licence yourself before shipping — this file is a pointer, not
advice.

Tesseract is kept rather than dropped for exactly this reason: it has no model
licence question, needs no network on first run, and is a single binary.

### Why Indra rather than Tesseract

Not only accuracy. Indra returns **layout blocks with labels** (headings, lists,
tables, captions, footnotes) and a reading order. Tesseract returns a flat list
of lines, which throws away the structure this project is built to preserve. The
repair pass maps those labels into IR node kinds, so a scanned page yields real
headings and lists rather than one undifferentiated paragraph.

**The split is deliberate and measured.** Page OCR uses Indra. Diagram labels
use Tesseract, because a label is a 63x12 crop and Indra's line detector returns
nothing for it at 4x, 8x, 12x or 16x upscale, with and without padding, while
Tesseract in single-line mode (`--psm 7`) reads the same crop immediately. A page
pipeline pointed at one word does not become a line recogniser.

### Model limits, inherited

Known limits of the model, worth stating rather than discovering: no handwriting,
weak on degraded scans, poor on complex tables with merged cells, and only Latin,
Cyrillic, Greek, Devanagari, Bengali and Chinese are read on images. Other
scripts work only through a digital PDF's own text layer.

Not only accuracy. Indra returns **layout blocks with labels** (headings, lists,
tables, captions, footnotes) and a reading order. Tesseract returns a flat list
of lines, which throws away the structure this project is built to preserve. On
the `real_scan.pdf` fixture, Indra recovers the title *as a heading*; Tesseract
can only return it as another line of text.

Known limits, inherited from the model and worth stating: no handwriting, weak
on degraded scans, poor on complex tables with merged cells, and only Latin,
Cyrillic, Greek, Devanagari, Bengali and Chinese are read on images.

### ⚠ Diagram engine

| Package | Extra | License | Why |
|---|---|---|---|
| opencv-python | `diagrams` | Apache-2.0 | Contour analysis, Hough line detection, distance transform. All deterministic geometry — no models, no downloads. |
| numpy | `diagrams` | BSD-3-Clause | Array backend for image data |

`opencv-python-headless` is a drop-in substitute for servers and containers, and
avoids pulling in GUI libraries.

### ⚠ PyMuPDF licensing

PyMuPDF is **AGPL-3.0**. That is a deliberate trade: it is dramatically faster
than the MIT/permissive alternatives (pypdf, pdfminer.six) and is the only
practical way to get per-span geometry fast enough for column detection and
table recovery.

What that means for you:

- Using DocWeave as a **library or CLI on your own machine** is fine.
- If you modify PyMuPDF itself, AGPL §13 obligations apply.
- If you **serve** DocWeave over a network, AGPL requires you to offer the
  corresponding source of your combined work.

Before shipping DocWeave inside a hosted service, either (a) comply with AGPL,
or (b) swap the PDF adapter for `pypdf` + `pdfminer.six` and accept the
performance loss. The `Adapter` interface is the seam for exactly this
decision — `PdfAdapter` is one class and can be replaced without touching the
IR, the validator, or any renderer.

This is documented rather than buried because it is a real constraint on
downstream use.

## Optional

| Package | Extra | License | Notes |
|---|---|---|---|
| fastapi / uvicorn | `api` | MIT / BSD-3 | Optional HTTP layer; not installed by default |
| markitdown | `bench` | MIT | Comparison baseline for benchmarks |
| rapidfuzz | `bench` | MIT | Fast string distance for accuracy metrics |
| pytest, pytest-asyncio, pytest-timeout | `dev` | MIT | Test tooling |

## External binaries

| Binary | Required for | Install |
|---|---|---|
| Tesseract | the optional OCR pass only | system package manager |

Nothing else in the project shells out to an executable.

## Deliberately excluded

| Package | Why not |
|---|---|
| torch / transformers | Hundreds of MB for no benefit; the diagram engine is pure geometry |
| paddleocr / easyocr | Heavy ML stacks; Tesseract is enough for the optional OCR pass |
| lxml | Faster, but the C parser has its own attack surface; html.parser is adequate and safer for untrusted input |
| networkx, pandas | Not needed; the IR and the diagram graph handle structure explicitly |
| scikit-image / scipy | OpenCV covers every operation the engine needs |

The diagram engine deliberately avoids ML. A learned shape classifier would be
faster to build and impossible to audit: when it gets a shape wrong there is no
way to say *why*. Geometric classification is slower and every decision carries
the measurements that produced it.

## Auditing

```bash
pip install pip-audit
pip-audit
```

Re-run before every release and record the result in the release notes.