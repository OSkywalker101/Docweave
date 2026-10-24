# DocWeave

A local-first document conversion engine that keeps structural fidelity, refuses
to fake confidence, and tells you what it could not read.

```bash
pip install docweave                 # base: PDF, DOCX, PPTX, XLSX, HTML, CSV/TSV, JSON, XML, MD, TXT
pip install 'docweave[all]'          # + every document parser
pip install 'docweave[diagrams]'     # + OpenCV, for diagram understanding
docweave convert report.pdf -o report.md
```

No API keys. No telemetry. No network calls during conversion. No GPU.

---

## What makes it different

Most converters tell you a file parsed. That is not the same as telling you the
content came through. DocWeave separates the two.

| | Behaviour |
|---|---|
| **Detection** | Format comes from file signatures, never the extension. A CSV named `report.pdf` is reported as a CSV, with the mismatch disclosed. |
| **Provenance** | Every node records its page number + bounding box, slide id + shape id, worksheet + A1 cell, or character offset. |
| **Honesty about content** | Content read directly, content reconstructed from geometry, and content we could not recover are three different things in the output. |
| **Diagnostics** | Every conversion emits a classified report: confirmed structural defects, heuristic suspicions, unverified content, and things needing a human. |
| **Determinism** | Same input + same config ⇒ byte-identical Markdown, JSON and JSONL. No timestamps in the output unless you ask for them. |
| **Cost** | No OCR, no vision model, no embeddings on the default path. A scanned page is *reported*, not silently returned as empty. |
| **Diagrams** | Reconstructs nodes, edges and directions from a diagram image — structure and meaning, not just the text drawn on it. |

### Scanned PDFs are reported, not faked

```
$ docweave validate scan.pdf
0/1 converted
error scan.pdf (2 diagnostic(s))
    ERROR no_content_extracted document: No text content was extracted from this document.
    WARN heuristic_likely_scanned page 1: Page 1 of 1 has almost no text (0 chars) but
         1 image(s) covering 99% of the page. OCR was not run, so pixel text is missing
         from the output.
```

A converter that returns an empty `.md` for a scanned PDF has failed you quietly.
This one exits non-zero and explains itself.

---

## Install

Requires Python 3.10+. The base install supports the most common native
document workflows and stays small.

```bash
pip install docweave                 # pydantic, click, rich, beautifulsoup4, defusedxml

pip install 'docweave[pdf]'          # + PyMuPDF        -> .pdf
pip install 'docweave[office]'       # + python-docx, python-pptx, openpyxl
pip install 'docweave[all]'          # both
pip install 'docweave[ocr]'          # + pytesseract (needs Tesseract on PATH)
pip install 'docweave[api]'          # + FastAPI/uvicorn (optional HTTP layer)
pip install 'docweave[dev]'          # + pytest
```

Run `docweave capabilities` at any time to see exactly what is installed, what
each adapter can do, and — importantly — what it cannot do.

### External binaries

Only Tesseract, and only for the optional OCR pass. Everything else is pure
Python or a wheel. No model downloads happen at any point.

---

## CLI

```bash
docweave convert report.pdf -o report.md
docweave convert report.pdf --format json -o report.json
docweave convert report.pdf --extract-assets --assets-dir ./assets
docweave convert ./corpus --recursive -o ./out --concurrency 4
docweave convert report.pdf --tables html          # keep complex table structure
docweave convert book.xlsx --tables list           # A1-referenced cells, no fake grid
docweave inspect report.pdf                         # format + integrity, no parsing
docweave validate ./corpus --recursive              # diagnostics only, no output files
docweave capabilities                               # adapters and their limits
```

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | success |
| 1 | unexpected internal error |
| 2 | invalid input (unsupported format, encrypted, malformed, missing) |
| 3 | a configured limit or deadline was reached |
| 4 | a required optional dependency is missing |
| 5 | completed, but something failed or produced errors |

Add `--json-output` to any command for machine-readable stdout.

---

## Python API

No subprocess, no server. Designed to sit inside a RAG pipeline.

```python
from docweave import convert, ConversionConfig

result = convert("report.pdf")

result.to_markdown()          # human/LLM facing
result.to_json()              # full structure + provenance + diagnostics
result.to_jsonl()             # one record per node, for chunking
result.to_report()            # counts + diagnostics only

for d in result.diagnostics.sorted():
    print(d.to_line())        # WARN heuristic_likely_scanned page 1: ...
```

Batching, with per-file error isolation:

```python
from docweave import Converter

report = Converter().convert_many(paths, concurrency=8)
print(report.summary())
# {'total': 120, 'succeeded': 117, 'failed': 3, ...}

for item in report.failed:
    print(item.path.name, item.reason)
```

Configuration is one object, serialisable to JSON:

```python
config = ConversionConfig(
    extract_assets=True,
    assets_dir="./assets",
    table_rendering="html",
    include_headers_footers=False,
)
result = convert("report.pdf", config)
```

---

## Supported formats

| Format | Read | Preserved structure |
|---|---|---|
| PDF | ✅ | pages, bounding boxes, reading order, tables, links, images, running header/footer detection |
| DOCX | ✅ | exact body order, heading hierarchy, lists, tables, header rows, metadata |
| PPTX | ✅ | slides, shapes, tables, charts, speaker notes, images |
| XLSX | ✅ | worksheets, cell coordinates, formulas **and** cached results separately, merged ranges, hidden sheets |
| HTML / XHTML | ✅ | headings, lists, tables with spans, code, images, links |
| CSV / TSV | ✅ | rows, columns, header detection, ragged-row reporting |
| JSON / XML | ✅ | lossless re-emission, duplicate-key detection |
| Markdown | ✅ | headings, lists, fenced code, pipe tables, quotes (round-trips) |
| Plain text | ✅ | encoding detection, block structure |
| EPUB, legacy .doc/.xls, RTF | ❌ | reported as unsupported, with a reason |

---

## Diagram understanding

The goal is the diagram's **structure and meaning**, not a text dump of the
shapes drawn on it.

```bash
docweave diagram architecture.png --format mermaid  -o architecture.mmd
docweave diagram flowchart.png    --format json     -o flowchart.json
docweave diagram er.png           --format plantuml -o schema.puml
docweave diagram sequence.png     --format markdown -o sequence.md
```

From Python:

```python
from docweave.diagram import analyze_file, render

result = analyze_file("architecture.png")
result.graph.kind            # DiagramKind.ARCHITECTURE
result.graph.nodes[0].kind   # NodeKind.PROCESS
result.graph.edges[0].arrowhead          # "head" | "tail" | "both" | None
result.graph.edges[0].direction_inferred # True when no arrowhead was found

print(render(result.graph, "mermaid"))
```

Diagrams can also be found *inside* documents:

```bash
docweave convert spec.pdf --detect-diagrams --extract-assets
```

### How it works, and why

Everything is **deterministic geometry**. No machine learning, so the same image
always produces the same graph and every decision can be traced to a
measurement.

1. **Shapes** — contour geometry (vertices, angles, circularity, fill ratio)
   classifies rectangles, rounded rectangles, diamonds, ellipses,
   parallelograms, cylinders and triangles. The contour *hierarchy* matters:
   where a connector touches a box, `findContours` returns one outer contour
   wrapping the whole drawing, and treating that as a shape would turn every
   diagram into a blob.
2. **Text** — a morphological gradient locates label regions. This always runs.
   *Reading* them is a separate, pluggable step.
3. **Connectors** — lines are found on ink with labels removed, then filtered to
   those whose endpoints both land on a shape.
4. **Arrowheads** — an arrowhead is fused into the line, so contours cannot see
   it. What distinguishes it is thickness: the distance transform isolates it.
5. **Edges** — published *only* when a line terminates on two distinct shapes. A
   glyph stroke cannot do that, so precision does not depend on tuning.
6. **Semantics** — diamond → decision, cylinder → data store, ellipse →
   terminator. Start/end and the diagram kind are inferred from layout, and
   marked as inferred.

### Measured accuracy

`python tests/diagram_accuracy.py` scores the engine against the synthetic
corpus, where every image ships the graph that was drawn. On this machine:

```
diagrams: 11
node recall     : 1.000      node precision : 1.000
edge recall     : 0.879      edge precision : 1.000
kind accuracy   : 1.000      direction proven: 0.787
```

These are the real numbers, including the misses. Every diagram is now read as
the right kind, every drawn node is recovered, and both node and edge precision
are 1.000 — nothing is invented to reach those scores:

- **Kind accuracy is 1.000 because the corpus was challenged, not because the
  rules were tuned to it.** It was 1.000 on seven diagrams with two rules drawn
  from those same seven. Adding UML class, state machine, grouped subsystem and
  network topology dropped it to 0.727 immediately — the clearest evidence that
  the first 1.000 was circular — and each of the four was then fixed on measured
  evidence before the number returned.

Fixing those fixtures closed five more real defects:

- **Precision was not precision.** The harness computed it as
  `detected / expected`, an unbounded over-generation ratio that reported
  **1.202**. It now matches detections to drawn geometry and reports
  `matched / detected`, with a test asserting every metric stays in `[0, 1]`.
- **Compartmented boxes were exploded.** A UML class with dividers was detected
  as three separate nodes joined by two invented edges. Stacked contours that
  share a left and right edge are now merged into one node, and the dividers
  are recorded as internal structure rather than published as relationships.
- **`UML_CLASS` was never returned.** The kind existed in the IR and no code
  path produced it. It is now identified by measured evidence - boxes drawn
  with internal dividers - rather than by a heuristic on arrowheads.
- **A ring of four connectors invented a node.** `state_machine` draws a cycle
  of arrows around a blank that `findContours` closed into a rounded rectangle,
  a phantom node and a phantom edge. A rounded contour pinned to three or more
  other shapes on distinct sides is now recognised as the enclosure of a
  connector ring and dropped.
- **A dashed connector was mistaken for a text label.** The dash row of a
  dashed link is a run of equal-height blobs, which the text detector read as a
  word and then erased before the line detector ran — the `network_topology`
  `router`→`mail` link. A region whose blobs all share one height is now read
  as a rule, not text.

### Still wrong

Two connectors are still missed and are left failing, each for a different
reason. Both are pinned by a strict `xfail` test so the day they start passing
is a deliberate decision, not a silent drift:

- `sequence` recovers 2 of 3 edges. The message and its reply run a few pixels
  apart and are now correctly separated as two strokes, but the reply carries
  no detectable arrowhead, so the two hops are indistinguishable and are merged
  back into one undirected relationship. Telling them apart needs direction
  evidence that the drawing does not provide.
- `ambiguous` recovers 0 of 1 edges. Its only connector is a ~20 px hop between
  boxes drawn nearly touching — below the connector length floor, and not
  recoverable by Hough even at half that floor without regressing other
  diagrams. It is the deliberate "reconstruction must degrade" fixture.

Edge recall of 0.879 is the honest headline weakness, and it is the two cases
above; several other diagrams are also flagged `degraded` because with no OCR
engine installed their labels are located but left unread. Node and edge
precision both stay at 1.000, so nothing is invented to close the gap.

### What it will not do

- **Read arbitrary illustration.** The detector works on outlined primitives on a
  plain background: rectangles, diamonds, ellipses, parallelograms, cylinders
  and arrows, joined by lines. A hand-drawn diagram with filled pastel nodes,
  drop shadows and a decorative background is outside it — and is reported as
  `diagram_nothing_detected` with exit code 5, never as a clean reconstruction.
  "Nothing found" and "understood" must not look alike.
- **Invent a relationship.** No arrowhead and no layout signal means the
  direction is marked `inferred`, not asserted.
- **Pretend OCR is exact.** Without Tesseract, text regions are located and
  reported as unread; the graph is unaffected. With it, labels are read and
  tagged `reader: tesseract`.
- **Force Mermaid to fit.** Containment, groups and lifeline verticals have no
  Mermaid equivalent; the renderer emits a `%%` note and says what it did.
- **Claim perfect representation.** `markdown` and `json` are lossless;
  `mermaid` and `plantuml` are projections that state their own limits.

---

## Measured performance

`python benchmarks/compare_markitdown.py --corpus tests/fixtures` runs both
systems on identical inputs in the same process and reports latency, memory and
accuracy.

### How the ground truth works

There is no hand-labelled corpus here, and there does not need to be. Every
fixture is **rendered from a declarative spec** in `tests/ground_truth.py`, which
also defines what a correct extraction must contain. The builders import their
content from that module, so a fixture and its expected extraction cannot drift
apart — which is what makes the numbers below reproducible rather than folklore.

### Results

22 scored documents (12 of them containing tables), Python 3.10 / Windows:

| metric | DocWeave | MarkItDown | |
|---|---:|---:|---|
| character error rate ↓ | **0.211** | 0.749 | docweave |
| word recall ↑ | **0.943** | 0.913 | docweave |
| word precision ↑ | **0.843** | 0.800 | docweave |
| word F1 ↑ | **0.866** | 0.831 | docweave |
| reading order ↑ | **0.821** | 0.589 | docweave |
| table cell recall ↑ | **1.000** | 0.938 | docweave |
| table cell precision ↑ | **1.000** | 0.858 | docweave |
| median latency | **26.1 ms** | 43.6 ms | docweave |

DocWeave leads on every metric measured here. **That is a claim about this
corpus, not about the world**, and the recall and F1 figures moved *down* from
the previous run for a reason worth being explicit about: the corpus gained
`real_scan.pdf`, a page of ink with no text layer, and OCR is opt-in. So the
default path scores 0.000 on it — **and so does MarkItDown's**. Neither system
invents text for a page it cannot read, which is the behaviour this project
claims as its design. Run with `--ocr-repair` and DocWeave recovers that page;
the number above is the default path, because that is what most callers get.

Word precision went from 0.828 to 0.843 earlier because two pieces of *engine
scaffolding* were being charged to the converter as if it had invented them —
the language token on a code fence (```` ```xml ````), and the `- A1:` prefix of
a list-rendered spreadsheet row. Both are renderer formatting, now stripped by
context rather than by blanket rule, with the same filter applied to MarkItDown,
which simply does not emit them.

On a scanned PDF with no text layer, DocWeave emits an empty document plus a
diagnostic naming the page and the remedy; MarkItDown emits nothing at all.
Neither invents text, which is the correct outcome, and DocWeave additionally
tells you why.

### A retraction

An earlier version of this file reported `table_cell_recall` of **0.394 vs
0.507** and named tables as "where Docweave loses on its own stated strength".
That was wrong, and the cause was the benchmark rather than the product:

1. Documents with no table were averaged in as `0.000` instead of excluded —
   nine of nineteen files, which alone buried the result.
2. The spreadsheet extractor read only the first cell of each line
   (`- A1: Region; B1: Total`), so columns `B` onward never matched anything.
3. Markup was stripped *before* tables were extracted, so every table rendered as
   HTML fallback scored zero cells.

Fixing those three — plus two infinite loops in the extractor, one of them
introduced by fix 2 and found only because `sample.docx` happened to contain a
bullet list — took the same code from 0.394 to 1.000. The lesson is recorded in
the code rather than this file: `tables_from_markdown` now enforces its
progress invariant structurally, so no branch can spin again.

A metric that improves 0.6 points when the harness, not the engine, changes is
not evidence about the engine. It is evidence the harness needed fixing first.

### Then the corpus was attacked

A perfect score is only worth something if the corpus tried to break it. So
`hostile.xlsx` was added: a cell whose text contains the list format's own
separator (`harbour; B1: docks`), a value containing double quotes, a value with
an embedded newline, a harmless semicolon, and a merged range.

That found a **real format-injection bug**. DocWeave renders spreadsheet rows as
`- A1: …; B1: …`, so a cell containing the literal text `harbour; B1: docks` was
indistinguishable from a two-cell row — silently changing the cell count for
anyone reading the output. Such values are now quoted (CSV-style, `"` doubled),
and the consumer side unquotes them. On that document:

| | DocWeave | MarkItDown |
|---|---:|---:|
| table cell recall | **1.00** | 0.86 |
| table cell precision | **1.00** | 0.60 |
| header match | **1.00** | 0.00 |
| CER ↓ | **0.276** | 0.961 |

`formatted.xlsx` was added the same way, because every other spreadsheet fixture
used plain strings and so never tested what happens to a date, a currency amount
or a percentage. That found a second defect: `str(datetime)` emitted
`2026-03-14 00:00:00` for a whole-day cell. Zero clock times are now dropped.

The merged range was already handled correctly, with a diagnostic explaining
that only the anchor cell carries a value. The embedded newline is genuinely
unrepresentable in a one-row-per-line format and is flattened to a space; the
ground truth says so rather than pretending otherwise.

#### Finding bugs nobody thought to look for

Every fixture in this project answers a question that was asked first — and
several exist *because* probing the engine found a bug. That makes the corpus
excellent at catching known failures and structurally blind to everything else.

`tests/property/` closes that gap. It generates inputs — random bytes, truncated
archives, cell values containing the renderer's own delimiters — and asserts
only promises the engine makes regardless of content. There is no expected
output to match, so a failure means a guarantee broke, not that a fixture was
wrong.

What it checks:

- **Typed refusal, never a crash** - any byte sequence is a file, and none may
  crash the converter.
- **Determinism** - the same bytes give the same output, which is what makes the
  numbers above reproducible and lets a caller cache on hash.
- **A cell cannot change the cell count** - the delimiter-injection class of bug,
  stated as a property rather than as one test case.
- **Nothing written disappears** - every non-empty cell value appears in the
  output, verbatim or unescaped.
- **Cell content is never structure** - markup inside a spreadsheet cell is data.
- **A diagnostic points somewhere real** - page and slide numbers are 1-based and
  in range, named sheets were actually read, node references resolve, cell
  locators are well-formed A1 references. A report that points at page 7 of a
  three-page document costs the reader more than saying nothing.

Five further tests plant deliberate lies in a real conversion - page 99, a sheet
called `Ghost`, a node id that does not exist, a 0-based index - and require the
checker to catch each one, so the suite cannot quietly decay into a loop over an
empty list.

The same approach now covers the diagram engine, which is the weaker half of the
project. `tests/property/test_diagram_invariants.py` generates random graphs,
renders them, and asserts that detected nodes equal drawn nodes. It found a
general phantom-node defect no fixture could reach: `findContours` returns an
outer contour for a whole connected component - two boxes and the line joining
them - that classified as a diamond and was published as a node, with the real
connectors then attributed to it. Fixed, and covered by the property rather than
by a case.

```bash
pytest tests/property -q
```

It found two real defects in one run, neither reachable from a hand-written
case:

- **A cell containing only a double quote was silently dropped.** The reader
  took a lone `"` as the start of a quoted value and consumed to end of line.
- **Pipe escaping was applied to the list-table format**, where cells are
  separated by `; ` and there is no pipe to escape — so a cell holding `0|`
  rendered as `0\|`.

It also caught the tests being wrong twice, which is the failure mode a property
suite has to be trusted not to hide: a `.md` file whose content is `[` is
sniffed as JSON and refused. That is content-first detection working correctly,
so the test was corrected rather than the engine.


## Gates

Four checks run in CI and can be run locally:

`ash
pytest tests -q                     # 487 tests
ruff check .                        # lint (configured in pyproject.toml)
python tools/check_types.py         # mypy, against a recorded baseline
python tools/check_metrics.py --report benchmarks/reports/markitdown_comparison.json \
                               --diagram benchmarks/reports/diagram_accuracy.json
`

CI installs from `requirements-dev.lock` — the known-good versions the published
numbers were produced on — and keys its pip cache on that file, so the
environment a red build reproduces is the one this page was measured in. The
pyproject ranges stay permissive for library consumers; the lockfile is what the
project's own automation installs.

The last two are ratchets rather than zero-error checks, and deliberately so.

**Types.** The codebase carries a recorded set of mypy findings, almost all of
them pydantic union variance: Document.body is declared list[AnyNode] and the
adapters append Paragraph to it, which is correct at runtime and
inexpressible without a covariant union. Suppressing whole modules would also
hide a real mistake in a line nobody was looking at, so the findings are
recorded in 	ools/mypy-baseline.txt and CI fails on *new* ones only. Turning
it on immediately found four real defects, three of them wrong annotations on
public API:

- convert_many(progress: callable | None) annotated with the **builtin
  function**, not a type.
- nalyze_asset() documented to return DiagramResult, annotated object.
- should_analyze(data: bytes, ...) annotated non-optional while its body
  explicitly guards if not data.
- ody redefined with a second annotation in the same scope.

**Metrics.** Every number in this file comes from two harnesses, and until now
nothing ran them automatically. enchmarks/baseline.json records the current
values and CI fails when one regresses, prints what improved, and requires the
baseline to be regenerated deliberately. Latency is excluded: it depends on the
machine, so a threshold would either fail constantly or be too loose to mean
anything.

Both ratchets were verified to fail on a planted regression before being
trusted. A guard that cannot fail is not a guard.

## What these numbers do not say

- **This is a synthetic corpus.** Known boundaries, no scanning noise, no font
  substitution. It is a lower bound on real-world accuracy, not a substitute for
  a labelled public corpus. Twelve of the twenty-one documents contain a table;
  the table metrics average over those twelve only, and say so in the report.
- **The corpus is not independent of the engine.** Several fixtures were written
  *because* probing this engine found a defect in it. That makes them excellent
  at catching known failure modes and structurally unable to catch unknown ones.
  This is the central limitation of every number on this page.
- **A perfect table score means the corpus has not found the next bug yet.**
  Formula *evaluation* is not implemented (formulas are preserved verbatim with a
  diagnostic), merged-cell semantics beyond the anchor are untested, and no
  pivot-style or nested layout is in the corpus. A real spreadsheet corpus would
  almost certainly score lower.
- **Two diagram connectors still fail** and are documented rather than tuned
  until green: `sequence`'s reply hop has no arrowhead to tell it from the
  message, and `ambiguous`'s only connector is a ~20 px hop below the length
  floor. Diagram edge recall of 0.879 is that engine's real weakness.
- **CER alone is misleading.** A system that drops half a document and emits
  equal garbage scores the same as a clean transcription. That is why word
  recall and word precision are reported alongside it: recall catches omission,
  precision catches invention.
- **Markup is not content.** HTML tags, Markdown syntax, image placeholders and
  engine explanatory prose are stripped before comparing text. Applied
  identically to both systems; a no-op for any that do not emit it.

---

```
source ──▶ detect ──▶ adapter ──▶ canonical IR ──▶ validate ──▶ render
           (cheap)   (lazy)       (typed,          (explainable   (markdown /
                        imports)     provenance)     rules)         json / jsonl)
```

**`detect.py`** — signature-first format detection. ZIP containers are identified
by their parts (`word/document.xml`), text formats by structure. Encryption and
size limits are checked before any parsing.

**`adapters/`** — one module per format family behind a single `Adapter`
interface. Each declares its formats, required packages, capabilities,
limitations, cost, and whether it streams. Imports are lazy: converting a `.md`
file does not load PyMuPDF, python-docx or openpyxl (there is a test for this).

**`models/ir.py`** — the canonical intermediate representation. Pages, slides
and worksheets are real containers; Markdown is a *rendering* of the IR, never
the storage format. Node ids are derived from structural position, so they are
stable across runs.

**`validation.py`** — turns extraction signals into classified diagnostics.
Single owner per diagnostic code, so a problem is reported once rather than once
per layer that noticed it.

**`renderers/`** — Markdown, versioned JSON envelope, JSONL records, and a
diagnostics report.

Data flows one way. There is no global mutable state, no circular import, and no
hidden cache.

---

## Known limitations

Stated plainly, because a converter that overstates its capability is worse than
one that does less.

- **No OCR on the default path.** Scanned pages produce a diagnostic, not text.
  The optional `--ocr-repair` pass repairs *only* pages measurement says are
  broken, and rejects any result that would lose content relative to what was
  already extracted.
- **Diagrams are reconstructed, not recognised perfectly.** Accuracy is measured
  (above) and the misses are real: some connectors are missed, and direction
  without an arrowhead is inferred rather than proven.
- **PDF heading levels are inferred** from relative font size, not semantics.
  Such nodes are marked `origin: inferred` in the JSON output.
- **PDF table recovery is geometry-based** and can mis-merge dense layouts.
  Complex tables fall back to HTML rather than being silently flattened.
- **Equations are not converted to LaTeX.** They are preserved as image
  references, and the omission is reported.
- **SmartArt, text boxes and OLE objects in DOCX** are detected and reported as
  unsupported, not silently dropped.
- **DOCX/PPTX footnotes and endnotes** are appended as text; inline anchors are
  not reconstructed.
- **Chart values** come from the cached workbook, not from reading the graphic.
- **Spreadsheet formulas are never evaluated.** If the file has no cached
  result, the cell says so instead of inventing a number.
- **Text-format detection between CSV, Markdown and prose is heuristic.** When
  the content disagrees with the extension, the mismatch is reported.

---

## Development

```bash
git clone <repo> && cd docweave
pip install -e '.[dev,all]'
python tests/fixtures_builder.py     # regenerate synthetic fixtures
python -m pytest tests -q
```

Tests never download anything and never need an API key. Optional-dependency
tests skip cleanly when the package is absent.

---

## License

MIT. Dependencies are permissively licensed; run a license audit before
vendoring. See `docs/dependencies.md`.