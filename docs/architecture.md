# Architecture

This document explains *why* the code is shaped the way it is. For usage, see
the [README](../README.md).

## The core claim

A parser that exits without raising has not, in general, successfully extracted a
document. It has only failed to fail.

Three distinct states exist for every piece of content, and the IR models all
three explicitly:

| State | `Origin` | Meaning |
|---|---|---|
| Observed | `observed` | Read directly from the source bytes / object model. |
| Reconstructed | `reconstructed` | Rebuilt from structure or geometry — table cells from ruling lines, reading order from bounding boxes. |
| Inferred | `inferred` | Produced by a heuristic — heading level from font size, list membership from a marker glyph. |
| Absent | `absent` | Known to exist in the source but not recovered. |

A node's `origin` reports the **weakest** source it has. If one source observed
the text and another inferred it, the node is `inferred`. Downstream consumers
must be able to act on the weakest claim, not the strongest.

Confidence is an explicit band (`high`/`medium`/`low`/`none`), never a number.
The model cannot express a calibrated probability, so it does not pretend to.

## Pipeline

```
bytes ─▶ detect ─▶ adapter ─▶ IR ─▶ validate ─▶ render
```

Each stage is a pure function of the previous one. Nothing is cached between
runs, nothing is global, and the same input always produces the same output.

### 1. Detection (`detect.py`)

Extensions are treated as a *hint*, never as truth. Resolution order:

1. Binary magic (`%PDF-`, OLE2 CFB, image signatures, RTF).
2. ZIP container inspection — OOXML packages are identified by the parts they
   actually contain (`word/document.xml` vs `ppt/presentation.xml` vs
   `xl/workbook.xml`), not by their extension.
3. Structural text analysis: XML declaration, HTML tag hints, strict JSON parse,
   Markdown block markers, delimiter field-count statistics.
4. Plain text fallback.

When the detected format disagrees with the extension, the conversion proceeds
with the content's answer and emits `input_extension_mismatch`. Detection also
reports its own `confidence` and `method` (`signature` vs `content` vs
`fallback`), so a guessed verdict is visible as a guessed verdict.

The delimiter heuristic deserves a note: a naive "all rows have the same field
count" rule classifies exactly the files that most need a table — ragged
exports, trailing notes — as prose. DocWeave instead requires the delimiter to
appear in most lines with a typical row of at least one field, which separates
tabular data from paragraphs that merely contain commas.

### 2. Adapters (`adapters/`)

```python
class Adapter(ABC):
    formats: tuple[str, ...]
    requires: dict[str, str]        # import name -> pip package
    capabilities: tuple[str, ...]   # what it does
    limitations: tuple[str, ...]    # what it does not
    cost: str                       # cheap | moderate | expensive
    streaming: bool
```

Capabilities and limitations are class attributes, which means
`docweave capabilities` can report the truth about the build without importing
anything heavy — it uses `importlib.util.find_spec`.

**Lazy imports are a correctness requirement, not an optimisation.** Loading
PyMuPDF, python-docx and openpyxl to convert a Markdown file costs seconds and
hundreds of megabytes for nothing. `tests/integration/test_api.py` asserts that
none of those modules appear in `sys.modules` after a text conversion.

**Diagnostics have exactly one owner.** Adapters report adapter-specific facts
(unsupported structures, decode damage, limit hits, delimiter corrections,
parser repair). The validator reports judgements about the result (empty pages,
ragged tables, scanned pages, unverified content). When two layers notice the
same problem, the report would contain a duplicate; instead the adapter emits a
*signal* and the validator emits the *diagnostic*.

### 3. The intermediate representation (`models/ir.py`)

Pydantic models, `extra="forbid"`, versioned via `schema_version`.

Design decisions worth defending:

- **Format-specific containers are first-class.** `Document` holds `pages`,
  `slides` or `sheets` — not a forced page model with slides shoehorned into it.
  A spreadsheet keeps its worksheet identity; a presentation keeps its slide
  identity. `Document.body` is used only by formats that genuinely have a flat
  flow (text, HTML, CSV, JSON, XML).
- **Traversal is unified.** `iter_nodes()` yields reading order across whichever
  container the format uses, so validators and renderers never special-case the
  container kind.
- **Ids are structural, not random.** `p3.b2.t1.r0c2` is derived from position,
  so ids are stable across runs and diffable across versions. Any randomness
  would break the determinism guarantee that downstream caches rely on.
- **`Cell.reference` is a computed field.** Spreadsheet addresses appear
  consistently in every output format. A `model_validator` drops it on input so
  the model round-trips its own JSON.
- **`Table.sparse` marks legitimately ragged data.** A worksheet whose rows have
  different widths is not defective; each cell carries its own coordinates. The
  validator would otherwise emit a false positive on every real spreadsheet.

### 4. Validation (`validation.py`)

Four categories, because they imply different actions:

- `structural_problem` — something is definitely wrong; a defect is confirmed.
- `heuristic_warning` — a rule fired; possibly a false positive.
- `unverified_content` — content exists in the source that we could not verify.
- `user_intervention_required` — a human must act (install something, raise a
  limit, supply a password).

Severity (`info`/`warning`/`error`) is orthogonal and drives the exit code.
Warnings and unverified content do **not** fail a conversion; a confirmed error
does. A document that is 99% recovered with two unverified cells is a success
with caveats, not a failure.

Thresholds are configuration, not magic numbers, and the scanned-page heuristic
has a master switch (`detect_scanned_pages`) so it can be turned off without
sentinel values.

The module deliberately never claims that the absence of a diagnostic proves the
content is correct. It cannot, and a validator that implies otherwise is worse
than one that admits the limit.

### 5. Rendering (`renderers/`)

**Markdown** is the lossy target and is explicit about it:

- Ragged tables fall back to HTML (which preserves spans) rather than being
  silently flattened into a misaligned grid.
- Tables with no header get an empty header row, because promoting data into a
  header position would reclassify content.
- Sparse sheets render as `A1: value; B2: value` lists. Padding a sparse row
  into a grid puts values under the wrong columns.
- Images that were not extracted render as visible placeholders.
- Nodes sourced from Markdown carry `meta.verbatim` so their text is not escaped
  a second time and littered with literal backslashes.
- Escaping is deliberately conservative: only `` ` ``, `*`, `[` and `]` are
  escaped. Over-escaping mangles prose (`--` becomes `\-\-`).

**JSON** is the versioned contract: `schema_version`, tool identity, source
identity including SHA-256, the IR, and the diagnostics envelope. It is
byte-identical across runs. The only non-reproducible field, timing, is emitted
solely on explicit request.

**JSONL** is one record per node with container identity, provenance grade, and
page/bbox or slide or sheet fields — the shape a chunker usually wants.

## Determinism

Guaranteed for identical input and identical configuration:

- No timestamps, no clock, no RNG in any output path.
- Adapter instances are cached per process, but hold no per-document state.
- Asset filenames derive from content hash and structural id.
- Diagnostics are sorted (severity, scope, code, message) before emission.
- Dictionaries in JSON preserve insertion order and are never re-sorted.

`tests/unit/test_renderers.py::TestDeterminism` asserts byte-equality for
Markdown, JSON and JSONL across every supported format. This is a real
guaranteee, not an aspiration: an earlier version leaked `conversion_ms` into
document metadata and the test caught it.

## Security posture

Every input is treated as untrusted.

- XML is parsed with `defusedxml`; entity expansion and external entities are
  refused (there is a test with a billion-laughs payload).
- HTML uses BeautifulSoup with the pure-Python parser: no script execution, no
  external entity resolution.
- Archive containers are bounded by file size and inspected by part name.
- Spreadsheet formulas are never evaluated and macros are never executed.
- No network access occurs during conversion. No subprocess is spawned.
- Resource budgets: file size, page count, cell count, CSV rows, total asset
  bytes, and a cooperative deadline checked between pages and rows.
- The CLI exposes no service and binds nothing.

Known risks are documented in [SECURITY.md](../SECURITY.md).

---

## Diagram engine (`diagram/`)

The diagram engine reconstructs **structure and meaning**. Its pipeline is
`shapes → text regions → connectors → arrowheads → edges → semantics`, and every
stage is deterministic geometry. No machine learning is involved, so results are
reproducible and each classification can be traced to the measurement that
produced it.

### The hard parts, and why they are hard

**Contour hierarchy is not optional.** In a line drawing where connectors touch
the boxes they join, `findContours(RETR_TREE)` returns one outer contour wrapping
the entire connected component. Emitting it as a shape turns every diagram into
a single blob. The tree is walked instead, and a contour is kept as a *frame*
only when it fills its bounding box, holds a real share of its area in
substantial children, and contains at least two of them.

**Classification is ordered by distinctiveness.** Each test is chosen so it
cannot fire on its neighbour:

| Test | Cannot fire on |
|---|---|
| inscribed (`fill ≤ 0.62`) + 4 vertices | a drawn rectangle (fills ~0.98 of its box) |
| inscribed + ≥7 vertices + circularity ≥0.70 | a rounded rectangle (fills ~0.98) |
| `fill ≥ 0.90` + 4 vertices | a diamond (fills 0.50) |
| 60/120° corners + flat top | a rectangle (corners are 90°) |
| square-ish corners, flat top and bottom, `fill` 0.68–0.90 | a parallelogram (60/120°) |

**A stroked outline yields two contours.** Every drawn box produces an outer and
an inner contour overlapping almost completely. Collapsing them is not an
optimisation: without it every node count silently doubles.

**Arrowheads are not contours.** They are fused into the connector and the shape
they touch. What distinguishes them is *thickness*: the distance transform of the
ink peaks around 1–2 on a hairline and 4 or more inside a filled head.
Thresholding it isolates the heads and erases the lines.

**Text detection must not eat arrowheads.** A morphological gradient finds both,
because both are ink that is not a straight line. The discriminator is glyph
count: a label is two or more connected blobs, a solid arrowhead is one. Before
that rule existed, arrowheads were erased as "text" before the arrowhead
detector ran, and most edge directions were wrong.

**Precision comes from the edge rule, not from tuning.** An edge is published
only when a line terminates on two *distinct* shapes. A glyph stroke, a box
outline and a stray mark cannot satisfy that, so false-positive edges do not
depend on a threshold. Segments are grouped by the pair of shapes they join
*before* direction is decided, because the arrowhead often sits on a different
fragment from the one Hough returned first.

**Direction is proven or admitted.** One arrowhead settles it, two mean
bidirectional, neither means the edge falls back to reading order and is marked
`direction_inferred`; Mermaid and PlantUML output carry a `%%` note saying so.

### Structure before semantics

Text *regions* are always located. Reading them is a separate, pluggable step:
`TesseractLabelReader` imports `pytesseract` on first use, `NullLabelReader`
reports that it cannot read. The graph is identical either way, which is why a
missing OCR engine costs labels rather than structure.

### Measured, not asserted

`tests/diagram_corpus.py` draws each diagram from a declarative spec and keeps
the spec, so the reconstructed graph can be compared against the graph that was
actually drawn. `tests/diagram_accuracy.py` reports recall and precision
separately, because overshooting — inventing structure — is as much a failure as
missing it.

Current run over the eleven-diagram corpus: node recall 1.000 / precision
1.000, edge recall 0.879 / precision 1.000, direction proven 0.787, diagram-kind
accuracy 1.000. Nothing is invented to reach those: a ring of connectors that
`findContours` closes into a phantom rounded rectangle is dropped, and a dashed
link whose equal-height dashes read as a "word" is no longer erased as text.
Two connectors are still missed and are pinned by strict `xfail` tests — the
deliberately ambiguous fixture contributes zero edges, which is the correct
outcome because overlapping boxes leave no unambiguous connector, and a
sequence reply hop carries no arrowhead to separate it from the message.

### Integration

`diagram/integrate.py` runs the engine over images extracted from a PDF, DOCX or
PPTX when `detect_diagrams` is enabled. Opt-in — it costs OpenCV plus a
reconstruction pass per image — and gated by a cheap size check so icons never
reach the engine.

---

## OCR repair (`repair.py`)

The only stage that redoes work the first pass already decided was missing.

Three rules govern it:

1. **Only repair what was measured to be broken.** Pages are selected from the
   same signals the validator reports, so the repair set and the diagnostic
   report can never disagree. A document with a full text layer is never
   rasterised.
2. **Never overwrite good content.** A repair that produces fewer characters than
   the existing extraction is rejected, and the reason is recorded. Original
   provenance is preserved on repaired nodes alongside the OCR source.
3. **Disclose the repair.** Every repaired page emits a diagnostic naming the
   engine, its confidence, and the before/after character counts.

The engine is imported lazily and its absence is a reported diagnostic, not a
silent skip.

---

## Benchmarking and the ground truth

`tests/ground_truth.py` declares what a correct extraction of each synthetic
document contains; `tests/fixtures_builder.py` imports its content from there.
That direction matters. If the manifest and the fixtures were written
separately they would drift, and every number below would be meaningless.

`benchmarks/accuracy.py` scores against it, and the metric choice is the second
half of the honesty story:

- **CER** is reported, but never alone. A converter that omits half a document
  and emits equal garbage can match a clean transcription on CER, so word recall
  (catches omission) and word precision (catches invention) always sit beside it.
- **Markup is not content.** HTML tags, Markdown syntax, image placeholders and
  the engine's own explanatory prose are stripped before comparing text. This is
  applied identically to both systems and is a no-op for any system that does
  not emit it — it removes *scaffolding*, not *DocWeave's output*.
- **Running furniture** is excluded from ground truth, because correct
  configuration strips it.
- **Lossless formats** (JSON, XML) are scored on their content rather than their
  markup, because the scorer strips tags.

Five bugs were found by building this, which is the argument for building it:

1. The table extractor looped forever when an HTML `<tr>` was not on the current
   line but a lookahead matched.
2. The DOCX adapter emitted every embedded image **twice** — once as an image
   part, once as an inline drawing.
3. The XML adapter emitted the document's visible text *and* the verbatim markup
   containing it, duplicating content for any downstream chunker.
4. The same extractor looped a second time once the spreadsheet branch was fixed,
   because a bullet list matched the cell-reference line pattern and the branch
   consumed nothing while `continue`ing.
5. Markup was stripped *before* tables were extracted, so every table rendered as
   an HTML fallback scored zero cells, and only the first cell of a spreadsheet
   row was parsed at all.

Bugs 1, 4 and 5 share a shape: two passes over the same text disagreeing about
order. The invariant is now enforced structurally in `tables_from_markdown` —
each iteration records where it started and advances if it made no progress —
so a future branch cannot reintroduce the spin.

Current results over 20 scored documents are in the README. An earlier run
reported `table_cell_recall` as 0.394 against MarkItDown's 0.507 and called
tables "where Docweave loses on its own stated strength". That was a harness
fault: documents without tables were averaged in as `0.000`, spreadsheet rows
lost every column after the first, and HTML-fallback tables were scored before
their tags were stripped. Fixed, the same code scores 1.000. The README records
the retraction rather than quietly replacing the number.

## Attacking the corpus

A perfect score means nothing until something tries to break it, so the corpus
gained `hostile.xlsx`: a cell containing the list format's own separator
(`harbour; B1: docks`), a value with embedded double quotes, a value with an
embedded newline, and a merged range.

The separator case was a genuine engine defect. `_table_list` joins cells with
`"; "`, so a cell whose text contains `; B1: ` rendered byte-identically to a
two-cell row — a consumer reading it silently counts the wrong number of cells.
Ambiguous values are now quoted CSV-style with `"` doubled, and the benchmark's
reader unquotes them.

The rule that came out of it: **an encoding must be unambiguous, and the
unambiguity has to be proven by a consumer, not just the producer.** A test now
renders a value through the real renderer and parses it back, so the two sides
cannot drift apart.

The merged range turned out to be handled already, with a diagnostic stating
that only the anchor cell carries a value. Probing is still worth it for the
cases that turn out to be fine — that is recorded rather than quietly dropped.

## Diagram classification: evidence, not verdicts

The same approach found three defects in the diagram engine, none of which the
headline metric was pointing at:

1. **Dash detection did not exist.** `EdgeStyle.DASHED` and the `dashed_edges`
   evidence field were both present, and `dashed_edges` fed the
   ER-versus-architecture decision, but no code ever assigned it. The field was
   permanently zero, which made that branch test "no arrowheads" rather than
   "undashed". Detection now samples ink along each segment and counts gaps.
2. **Every edge shared one style.** The style was read from `segment`, a leaked
   loop variable from the grouping pass, so all edges inherited the last
   segment's flag. A test asserting that a mixed diagram reports more than one
   style would have caught it; the per-edge test now does.
3. **Classification over-named.** `architecture` was returned for any directed
   flow of plain boxes, and a flowchart required both a diamond *and* a
   terminator — so `read input -> transform`, which has neither, was called a
   generic box-and-arrow diagram.

The general lesson, and the reason the rules are written the way they are:

- **A kind is claimed on positive evidence.** Architecture requires a topology
  (fan-out hub or explicit boundary). Arrows alone are not evidence of one.
- **"I cannot tell" is a valid output.** A diagram whose nodes are largely
  unattached is reported as `unknown` with the reason in the evidence, because
  naming it anyway would be a guess presented as a reading.
- **Dump the evidence, not the verdict.** All three bugs were visible in the
  classifier's evidence dict minutes before they were visible in its output.

Kind accuracy on the seven-diagram corpus reached 1.000, and that was circular:
two of the rules above were derived from those same seven diagrams. The corpus
now carries eleven, adding UML class, state machine, grouped subsystem and
network topology specifically to break the classifier. Kind accuracy fell to
0.727 immediately; each of the four was then fixed on measured evidence, and the
number returned to 1.000 honestly. A corpus that cannot make your own number go
down is not testing anything — so the number was allowed to go down first.

The expansion also exposed a measurement defect worth recording: **precision was
defined as `detected / expected`**, an unbounded over-generation ratio that
reported 1.202 — a "precision" above a perfect score. It is now
`matched / detected`, with detections matched to drawn geometry by endpoint
overlap. A test asserts every metric stays within `[0, 1]`, because the failure
mode here was silent: a number that cannot exist was simply reported.

And one structural defect: a box drawn with internal dividers was detected as
one contour per compartment, so a UML class became three nodes joined by two
invented edges. Stacked contours sharing a left and right edge are now merged,
the dividers are recorded as internal structure, and `UML_CLASS` — present in
the IR but never returned by any code path — is identified by that measured
evidence.

`state_machine` was the hardest of the four and is now fixed. Its four arrows
joining the states in a cycle formed a closed ring that `findContours` read as a
rounded rectangle, and that phantom absorbed the connectors. There are no
children to disqualify it and no slack to measure, so the discriminator is the
one thing that is actually known about it: the rounded contour is *pinned to
three or more other shapes on distinct sides* — it is the enclosure of a
connector ring, not a drawn box. That contour is dropped, and the ring's
connectors bind to the states instead. `node precision` rose to 1.000 and edge
precision with it.

`grouped_subsystem` taught the frame rule. A subsystem box is very often drawn
with a **dashed** stroke, and a dashed outline never closes, so `findContours`
cannot see it at all — the boundary was simply invisible. Frames are now rebuilt
from the line detector instead: join collinear runs, pair parallel ones into a
rectangle, and keep only boxes that *contain* something. The strokes must
actually be dashed; without that check every solid rectangle is detected a second
time, which regressed the UML fixtures until it was added.

`network_topology` exposed a subtler one: its `router`→`mail` link is dashed, and
a dashed run is a row of equal-height blobs — which the text detector read as a
word and *erased* before the line detector ran. A detected text region whose
blobs all share one height is now read as a rule, not text. That recovered the
fifth connector and took edge recall from 0.792 to 0.879.

## What every number here has in common

All accuracy figures in this project come from a corpus the project wrote itself.
That is not a disclaimer bolted on afterwards — it is the structural limitation,
and it is worth stating plainly:

- **The fixtures are rendered from a spec**, so they have known boundaries, no
  scanning noise, and no font substitution. Real documents have all three.
- **Several fixtures exist because probing this engine found a defect in it.**
  That makes the corpus excellent at catching *known* failure modes and
  structurally incapable of catching unknown ones. `hostile.xlsx` was written
  after finding the separator-injection bug; `formatted.xlsx` after finding the
  `00:00:00` date bug. Both are genuine findings and both make the corpus
  better — and neither is evidence about a bug we did not think to look for.
- **The diagram corpus grew from seven to eleven specifically to break the
  classifier.** Kind accuracy fell from 1.000 to 0.727 the moment it did. A
  corpus that cannot make your own number go down is not testing anything.

The honest summary: the numbers are reproducible and internally consistent, and
they are a lower bound. A labelled public corpus would be the only way to turn
them into an upper claim, and building one is the highest-value thing this
project has not done.

## Property-based testing

### For diagrams

The diagram corpus is eleven hand-written fixtures, and two classification rules
were derived from those same eleven — when it grew from seven to eleven, kind
accuracy fell from 1.000 to 0.727 immediately. That is the clearest sign the
corpus was flattering itself.

`tests/property/test_diagram_invariants.py` removes the flattering. It generates
random graphs, renders them through the same renderer the corpus uses, and
asserts only what follows from the fact that the graph was drawn:

- **Detected nodes equal drawn nodes.** This kills the phantom-node class.
- **Detected edges equal drawn edges**, with no self-loops on an acyclic input.
- **Analysis is deterministic** on the same image.
- **A diagram with no connectors is `unknown`**, never given a kind.

The generator is independent evidence in a way the corpus cannot be: it never
looks at the engine, so it cannot encode the engine's blind spots.

### What it found: the component wrapper

Every generated two-node diagram came back with **three**. The cause is general
and was not in any fixture:

`findContours` with `RETR_TREE` returns an outer contour for a whole connected
component — two boxes plus the line joining them. That contour was classified as
a **diamond**, and `_is_primitive` whitelists diamonds past the fill test, so it
was published as a phantom `decision` node. Worse, the connectors were then
attributed to the phantom instead of to the shapes they actually join.

Two rules fix it, both about not publishing a contour that is not a drawn thing:

1. **A frame must have slack.** `FRAME_MIN_SLACK` requires visible padding
   between a frame and its members. A contour hugging its children is the
   boundary of a cluster, not a drawn boundary.
2. **A contour that encloses other real shapes is never a node.** It is either a
   frame (emitted as a container) or a component boundary (emitted as nothing).

### What it did not fix

Two connectors remain missing and are pinned by strict `xfail` tests, each for a
different reason, each left failing rather than special-cased:

- **`sequence`'s reply hop** carries no arrowhead, so the message and its reply
  are indistinguishable and merge back into one undirected relationship. They
  are now correctly detected as two separate strokes, but direction evidence the
  drawing does not supply is what is needed to publish them as two edges.
- **`ambiguous`'s only connector** is a ~20 px hop between boxes drawn nearly
  touching — below the connector length floor, and not recoverable by Hough even
  at half that floor without regressing other diagrams. It is the deliberate
  "reconstruction must degrade" fixture, and zero edges there is the intended
  outcome.

The generator also avoids routing connectors through unrelated nodes. That is a
property of well-formed diagrams, not a convenience: a line drawn across a box
splits its contour in two and makes the ground truth ambiguous. The engine's
weakness there — a box crossed by a line becomes two contours — is recorded
here instead of being tested around.

### For documents

A corpus can only ever test the failures somebody already thought of. Every
fixture in this project answers a question that was asked first — and several
of them exist *because* probing the engine found a bug, which makes them good
regression tests and blind to everything else.

`tests/property/` closes that gap by generating inputs and asserting only
properties that follow from the format or from a promise the engine makes.
There is no expected output to match, so a failure means a guarantee broke
rather than an expectation being wrong:

- **Typed refusal, never a crash.** Any byte sequence is a file. Conversion may
  fail — an empty file *must* raise `EmptyFileError` — but no `KeyError` or
  `zipfile.BadZipFile` may escape from a parser handed something unreadable.
- **Determinism.** The same bytes give the same output, always. This is what
  makes the numbers above reproducible and what lets a caller cache on hash.
- **A cell cannot change the cell count.** The delimiter-injection class of
  bug, stated as a property rather than as one test case.
- **Nothing written disappears.** Every non-empty cell value appears in the
  output, verbatim or unescaped.
- **Cell content is never structure.** Markup inside a spreadsheet cell is data.
- **A diagnostic points somewhere real.** Every page index is 1-based and inside
  the page count, every sheet named was actually read, every node reference
  resolves, every cell locator is a well-formed A1 reference on a sheet that
  exists. A diagnostic pointing at page 7 of a three-page document sends the
  reader somewhere with nothing in it and costs them the time to find out the
  report is lying.

Two real defects came out of this within one run, neither of which any fixture
would have caught:

1. **A cell containing only a double quote was silently dropped.** The reader
   treated a lone `"` as the start of a quoted value and consumed to end of line,
   losing the cell. A quote now only opens a quoted value if a partner follows.
2. **Pipe escaping was applied to the list table format**, where cells are
   separated by `; ` and there is no pipe to escape. A cell holding `0|`
   rendered as `0\|`, which no consumer reads back as the value it was.

A third "failure" was the tests being wrong rather than the engine, and it is
worth recording because the instinct is usually to "fix" the engine: a `.md`
file whose content is `[` is detected as JSON and rejected. That is
content-first sniffing working as designed, so the test was changed to expect
the refusal.

The general rule the property suite enforces is that **whitespace normalisation
is a requirement, not a shortcut**: two trailing spaces are a Markdown hard line
break, so a cell ending in a space *must* be trimmed or the rendered document is
corrupted. Invariants are written against the real contract rather than the
idealised one.

### Proving a checker can fail

The diagnostic checker is itself a piece of logic, and a green check proves
nothing unless the checker is shown to be able to fail. Five tests therefore
plant specific lies in a real conversion — a page 99, a sheet called `Ghost`, a
node id that does not exist, a 0-based index — and require the checker to catch
each one. One of them asserts the fixture *does* produce diagnostics, so the
property tests cannot quietly decay into a loop over an empty list.

Without that, "29 property tests pass" and "the property tests check nothing"
are indistinguishable, and the second is the failure mode that property suites
are most prone to.

## Tests that depended on the machine

Installing a real OCR engine exposed two tests that had been passing for the
wrong reason:

1. `test_convert_with_repair_enabled_but_engine_missing` asserted that a repair
   request either reported the engine unavailable or produced text. With no
   engine installed the first branch was always taken. The moment a real one
   appeared, OCR ran, found a blank synthetic page, emitted
   `ocr_returned_no_text`, and the test failed.
2. `test_diagnostics_are_reported` asserted the CLI reports unread labels. With
   no engine installed they always were unread. With one installed they were
   read, and there was nothing to report.

Both are now environment-independent: the first forces the engine unavailable
via `monkeypatch`, the second accepts *either* outcome — labels read, or the
unread ones reported — and asserts only that the CLI accounts for every label
it finds.

The lesson generalises past OCR: **a test whose result depends on what happens
to be installed is not a test, it is a snapshot of one machine.** It fails
loudly on someone else's setup and silently stops testing anything on yours.
The two skipped tests in this suite were, until now, the only ones that could
have caught this — which is an argument for keeping environment-gated tests
visible rather than quietly deleting them.

---

## What is deliberately not here

- **Accuracy claims.** The comparison harness measures latency, memory and
  output volume against MarkItDown, and states in its own output that accuracy
  is *not measured* because no ground truth exists for the corpus. Producing that
  number anyway — from an LLM, from a fuzzy match against MarkItDown's own
  output, from anything circular — would be the single most damaging thing this
  project could do.
- **HTTP API.** The core is useful without it, and adding a server before the
  engine is solid is the wrong order.
- **Diagram ground truth beyond the synthetic corpus.** Every diagram image
  ships the graph that was drawn, which is what makes the accuracy numbers above
  real. Real-world diagrams are not yet scored, because that needs a human
  annotation pass this project has not done.
- **A vision model.** The diagram engine is geometry only. Where it cannot
  establish a relationship, it reports the absence rather than guessing with a
  language model.