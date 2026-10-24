# Build an Intelligent, Self Correcting, Open Source Document Conversion Engine

## 1. Your Role

Act as a senior software architect, document processing engineer, Python systems engineer, and open source maintainer.

Design and implement a production quality, open source document conversion engine that improves upon Microsoft MarkItDown.

The project should prioritize:

1. Extraction accuracy and structural fidelity.
2. Low latency and low memory consumption.
3. Zero mandatory API calls and zero per document costs.
4. Automatic detection and repair of extraction failures.
5. Modular architecture with replaceable extraction engines.
6. Reproducible benchmarks proving improvements over existing tools.

Do not merely produce an architecture document, a proof of concept, or a thin wrapper around existing libraries. Build a working repository with tests, benchmarks, documentation, and a usable interface.

Work incrementally. Inspect the repository and environment before making architectural decisions. If the repository is empty, initialize a clean project. If existing code is present, preserve useful functionality.

## 2. Product Vision

Build a local first document conversion engine that converts heterogeneous files into accurate, structured, machine readable outputs suitable for LLMs, RAG pipelines, search engines, analytics and downstream automation.

Working name: DocWeave. Treat this as a temporary name that can be changed later.

The central innovation is a confidence guided, self correcting extraction pipeline.

Unlike a conventional converter that extracts content once and returns it, the engine should:

1. Analyze the source document.
2. Select the fastest appropriate extraction method.
3. Detect potential extraction failures.
4. Apply targeted fallback methods only where necessary.
5. Validate the repaired output.
6. Preserve references to the original document and relevant page, slide, sheet or cell.
7. Report unresolved uncertainties rather than silently pretending the conversion is correct.

The default path must remain fast. Do not invoke an LLM, OCR engine or vision model for every file.

## 3. Core Design Principles

### A. Local first

The complete core workflow must operate offline after installation and dependency setup.

No mandatory cloud services, API keys, telemetry, paid endpoints or external inference calls.

Optional integrations may exist, but they must be disabled by default and isolated from the core engine.

### B. Fast by default

Use deterministic parsers before machine learning.

Avoid loading OCR models, vision models or embedding models unless a particular operation requires them.

Use lazy imports, reusable processing resources, bounded concurrency and streaming where practical.

### C. Correctness over false confidence

Do not equate a successful parser execution with a successful extraction.

A document can produce valid Markdown while losing critical information.

Every extraction stage should distinguish between:
- Successfully extracted content.
- Content inferred or reconstructed.
- Content that could not be extracted.
- Content whose correctness remains uncertain.

Never silently invent missing content.

### D. Preserve structure

Markdown is a useful output format, but it is not a complete representation of every document.

Use a canonical intermediate representation that can preserve structure, provenance, tables, equations, images, layout information and spreadsheet relationships before rendering Markdown.

### E. Evidence driven engineering

Do not claim the tool is more accurate, faster or more efficient than MarkItDown without reproducible measurements.

Benchmark both systems on identical input files under equivalent conditions.

## 4. Supported Input Formats

Implement format detection using file signatures and parser validation rather than trusting extensions alone.

Prioritize the following formats.

### PDF
- Digitally generated PDFs.
- Scanned PDFs.
- Mixed scanned and digitally generated PDFs.
- Multi column documents.
- PDFs containing tables, images, footnotes, headers, footers and equations.
- PDFs with unusual reading orders and rotated pages.

### Microsoft Office
- DOCX.
- PPTX.
- XLSX.

### Other formats
- HTML and XHTML.
- Plain text.
- CSV and TSV.
- JSON.
- XML.
- Markdown.
- EPUB, if practical.

Implement the first production milestone around PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON and plain text.

Support additional formats through adapters rather than coupling them to the core pipeline.

Reject unsupported or malformed files with informative errors. Do not assume every file with a familiar extension is valid.

## 5. Architecture

Implement the following logical components.

### 5.1 Input and format detector

Responsibilities:
- Identify actual file format.
- Check file readability and structural integrity.
- Extract basic metadata.
- Detect encrypted or password protected files.
- Apply configurable file size and processing limits.
- Determine which processing adapters are available.

The detector should be inexpensive and should not parse the entire document unnecessarily.

### 5.2 Document analyzer

Before extraction, estimate document characteristics.

Examples:
- Whether PDF pages contain native text.
- Whether a page is probably scanned.
- Whether a document has multiple columns.
- Whether tables or equations are likely present.
- Whether the file contains embedded images.
- Whether spreadsheets contain formulas, merged cells or hidden sheets.

Use cheap signals first.

Do not run expensive computer vision models just to decide whether to use a normal parser.

### 5.3 Extraction adapter system

Create a common adapter interface.

Each adapter should define:
- Supported input formats.
- Required dependencies.
- Extraction method.
- Capabilities and limitations.
- Resource requirements.
- Structured output.
- Diagnostics and warnings.
- Processing duration.

Adapters should be independently testable and replaceable.

Use established libraries where appropriate. Evaluate PyMuPDF, pypdf, pdfplumber, python-docx, python-pptx, openpyxl, BeautifulSoup and other suitable open source tools.

Do not build every parser from scratch.

Do not assume that combining several parsers automatically improves quality. Select them based on measurable strengths.

### 5.4 Canonical intermediate representation

Create a typed document representation.

It should support:
- Documents.
- Pages.
- Sections.
- Paragraphs.
- Headings.
- Lists.
- Tables, rows, columns and cells.
- Images and image references.
- Charts.
- Equations.
- Hyperlinks.
- Footnotes and endnotes.
- Headers and footers.
- Code blocks.
- Metadata.
- Source coordinates and provenance.
- Extraction confidence and warnings.

Represent source locations using the appropriate coordinates for the format, such as page bounding boxes, slide identifiers, worksheet names, cell references or character offsets.

Preserve the distinction between observed source content and reconstructed content.

Do not force every format into a page based model. A spreadsheet should retain its worksheet and cell structure, and a presentation should retain slide structure.

Use Pydantic or equivalent typed models where appropriate.

Define a versioned schema so downstream integrations can rely on stable output contracts.

### 5.5 Confidence guided routing

Develop a lightweight routing engine that selects extraction methods and decides when repair is necessary.

Use multiple signals rather than a single arbitrary confidence score.

Potential signals include:
- Amount of extracted text relative to page dimensions.
- Character distribution and replacement characters.
- Reading order anomalies.
- Geometric overlap.
- Missing or inconsistent page content.
- Table structure consistency.
- Agreement between independent extraction methods.
- Presence of image content with little corresponding extracted text.
- Parser warnings and failures.
- Differences between source object counts and extracted object counts.

Build an explainable rule based baseline first.

Each decision should include a reason, such as:

"Page 7 contains substantial image content but almost no extractable text. OCR was selected."

Do not represent heuristic confidence as a statistically calibrated probability.

Make thresholds configurable and evaluate them on a labeled validation dataset.

### 5.6 Targeted repair engine

Repair only the affected parts of a document.

Examples:

- If one PDF page is scanned, OCR that page rather than rerunning the whole document.
- If a table has malformed rows, reconstruct the table without discarding the surrounding text.
- If columns appear interleaved, attempt geometric reading order correction.
- If native text extraction fails on a page, retry with a different parser.
- If a spreadsheet contains formulas, preserve formulas and values separately when available.
- If a document contains embedded images, retain references to them even if visual interpretation is unavailable.

Each repair operation must preserve the original extraction for comparison and record its own provenance.

Never replace good content with an unverified fallback merely because the fallback completed successfully.

### 5.7 Validation and diagnostics engine

After extraction and repair, perform inexpensive validation checks.

Check:
- Whether all expected pages or slides were processed.
- Whether content appears to be missing.
- Whether text is suspiciously empty.
- Whether tables have inconsistent dimensions.
- Whether reading order is implausible.
- Whether embedded assets have been accounted for.
- Whether extraction introduced malformed Unicode.
- Whether output structure is valid.
- Whether repair improved the relevant validation signal.

Produce structured warnings and a machine readable report.

The report must distinguish between:
1. Confirmed structural problems.
2. Heuristic warnings.
3. Unverified content.
4. Failures requiring user intervention.

Do not claim that every omission can be detected automatically.

## 6. Format Specific Requirements

### 6.1 PDF processing

Use native text extraction as the default.

For each page:
1. Extract text and available geometry.
2. Inspect text coverage and basic structural signals.
3. Detect likely scanned content or unusual layout.
4. Apply targeted fallback methods if needed.
5. Reconstruct reading order using geometric information when necessary.
6. Preserve tables, image references, links and source coordinates where supported.
7. Record page specific warnings.

Handle mixed documents in which some pages contain native text and others contain scanned images.

For OCR, investigate Tesseract as a lightweight optional dependency. Evaluate alternative engines only where they provide measurable benefits.

OCR must be optional and must not be loaded when unnecessary.

### 6.2 Tables

Treat tables as structured data, not merely formatted strings.

Preserve:
- Row and column boundaries.
- Header relationships.
- Empty cells.
- Cell text.
- Merged cells where representable.
- Nested tables where feasible.
- Numeric values and units.
- Source location.

Use Markdown tables for simple tables.

For complex tables, use an appropriate structured representation and provide a readable Markdown rendering.

Do not silently flatten a table when doing so would destroy meaningful relationships.

### 6.3 DOCX

Preserve:
- Heading hierarchy.
- Paragraph order.
- Numbered and bulleted lists.
- Tables.
- Hyperlinks.
- Images.
- Footnotes and endnotes when supported.
- Headers and footers.
- Relevant document metadata.

Document unsupported structures explicitly.

### 6.4 PPTX

Preserve:
- Slide order.
- Titles and text boxes.
- Reading order.
- Speaker notes where available.
- Tables.
- Images.
- Chart data where accessible.
- Hyperlinks.
- Relevant shape and grouping relationships.

Represent each slide as an identifiable structural unit.

### 6.5 XLSX and CSV

Spreadsheet extraction must be designed around structured data rather than forcing everything into a single Markdown document.

Preserve:
- Workbook and worksheet names.
- Cell coordinates.
- Cell values.
- Formula strings.
- Cached formula results when available.
- Number formats and units where useful.
- Merged ranges.
- Hidden rows and columns where relevant.
- Multiple tables on the same sheet.
- Empty cells that are structurally significant.

Never execute spreadsheet formulas or macros.

Clearly distinguish a formula from its cached result. Do not claim to calculate formula results when the engine has not actually done so.

Provide both:
- A readable Markdown representation.
- A structured JSON representation.

### 6.6 Images, charts and diagrams

Do not assume a text converter can understand an image.

First extract embedded images and available metadata.

Where vector data or chart objects expose meaningful labels and values, extract them directly.

Where image interpretation is needed, provide an optional OCR or vision adapter.

The default core must not depend on a remote vision API.

Preserve image assets in an output directory and reference them from the converted document.

For charts, preserve underlying data where accessible. If the data cannot be recovered, report that limitation rather than fabricating values.

### 6.7 Equations

Preserve existing mathematical markup when available.

Support LaTeX output when equations can be extracted reliably.

If mathematical recognition is unavailable or uncertain, preserve the original equation image or an asset reference and report the limitation.

Do not introduce an expensive vision model into the normal extraction path.

## 7. Output Formats

Implement at least three output modes.

### Markdown
Human readable content optimized for general LLM ingestion.

### Structured JSON
Machine readable representation of document structure, source references, assets and diagnostics.

### JSONL
Optional record oriented output for large documents and downstream pipelines.

Also provide:
- Asset extraction into a directory.
- A diagnostics report.
- Stable identifiers for structural elements.
- An option to include or exclude headers and footers.
- Configurable handling of tables and images.

Markdown must not be treated as the canonical storage format.

Support deterministic output where possible, so identical inputs and configuration produce equivalent outputs.

## 8. CLI and Python API

Create a polished command line interface.

Illustrative commands:

    docweave convert report.pdf -o output.md

    docweave convert report.pdf --format json -o output.json

    docweave convert report.pdf --assets-dir ./assets

    docweave inspect report.pdf

    docweave validate report.pdf

    docweave benchmark ./corpus

    docweave capabilities

The commands are illustrative. Design the final CLI consistently and document it.

Support:
- Single file conversion.
- Batch conversion.
- Recursive directory processing.
- Configurable concurrency.
- JSON configuration.
- Clear exit codes.
- Progress reporting.
- Cancellation.
- Per file error handling.
- Optional machine readable CLI output.

Create a Python API suitable for integration into RAG pipelines.

The API should expose document conversion, inspection, diagnostics and structured output without requiring subprocess invocation.

Do not make the Python API depend on a running web server.

## 9. Optional HTTP API

After the core engine and CLI work, implement an optional local HTTP API using FastAPI.

Include:
- Health and capability endpoints.
- File conversion.
- Structured diagnostics.
- Request limits.
- Safe temporary file handling.
- Configurable concurrency.
- Clear error responses.

Do not expose a publicly accessible service by default.

Avoid loading all available parsers and models at server startup.

## 10. Performance Requirements

Optimize for ordinary consumer laptops and CPU only environments.

Use the following as initial engineering targets, not guaranteed performance claims:

- Ordinary digitally generated PDFs should avoid OCR.
- Simple documents should not require an ML model.
- The common path should minimize allocations and unnecessary passes.
- Optional dependencies should be loaded lazily.
- Large documents should support bounded memory usage where practical.
- Batch processing should reuse initialized resources.
- One difficult page should not trigger a full document reprocessing pass.
- The user should be able to disable every optional expensive operation.

Measure:
- End to end latency.
- Time per page.
- Throughput.
- Peak resident memory.
- CPU utilization.
- Optional OCR overhead.
- Repair frequency.
- Accuracy improvements from fallback processing.

Set hardware and dataset conditions explicitly in benchmark reports.

Do not promise a fixed speedup before measurement.

## 11. Free and Open Source Dependencies

Prefer permissively licensed dependencies where practical.

Before including any library, inspect its license, maintenance status, platform compatibility and transitive dependencies.

Provide:
- A minimal installation.
- Optional dependency groups.
- A documented offline configuration.
- A dependency and license inventory.
- A clear explanation of which features require external executables or model downloads.

Avoid heavyweight ML dependencies in the base installation.

The base installation should support the most common native document conversion workflows.

OCR and advanced layout support should be installable separately.

Do not require a GPU.

## 12. Benchmarking Against MarkItDown

A credible benchmark suite is essential to this project.

Create a reproducible corpus covering:
- Normal text PDFs.
- Scanned PDFs.
- Mixed PDFs.
- Multi column documents.
- PDFs with complex tables.
- DOCX with tables and lists.
- PPTX with multiple layouts.
- XLSX with formulas and multiple sheets.
- Documents with images.
- Malformed or partially corrupted files.
- Unicode and multilingual content.

Use public or synthetic documents with appropriate licenses and redistribution permissions.

Create a manually verified subset with ground truth.

Compare against Microsoft MarkItDown using the same inputs and equivalent output requirements.

Measure:
- Character and word extraction accuracy.
- Normalized edit distance or character error rate.
- Table structure accuracy.
- Reading order accuracy.
- Page and slide coverage.
- Formula preservation.
- Structural element preservation.
- Failure detection precision and recall.
- False positive repair rate.
- Latency and peak memory.

Do not rely on a single aggregate score. Report results by format and document difficulty.

Ensure that normalization does not hide missing content or structural damage.

For outputs that are not directly comparable, explain the comparison methodology and limitations.

Include baseline results from the unmodified parser, the complete pipeline and each optional repair stage.

Document the benchmark environment and provide commands to reproduce the measurements.

Never fabricate benchmark results.

## 13. Testing Requirements

Implement unit, integration and regression tests.

Include tests for:
- Every supported file format.
- Invalid file contents.
- Unsupported formats.
- Corrupted documents.
- Empty files.
- Password protected files.
- Large inputs.
- Unicode and multilingual text.
- Complex tables.
- Scanned pages.
- Mixed native and scanned PDFs.
- Missing optional dependencies.
- Fallback failures.
- Interrupted processing.
- Temporary file cleanup.
- Output schema validation.
- Deterministic conversion.
- Batch processing with individual file failures.

Create small synthetic test fixtures wherever possible.

Tests must run without API keys or paid services.

Tests requiring OCR should be marked clearly and should skip gracefully when the optional dependency is unavailable.

Use property based testing where it adds value, particularly for structured outputs and malformed inputs.

## 14. Security and Robustness

Treat every input file as untrusted.

Implement:
- File size limits.
- Processing timeouts where feasible.
- Safe temporary file handling.
- Path traversal protection.
- Output path validation.
- Archive extraction limits where relevant.
- Resource limits for pathological documents.
- Safe handling of malformed XML and embedded objects.
- No macro execution.
- No automatic network access during conversion.
- No execution of document supplied commands or formulas.

Avoid shell command construction from user controlled values.

Provide clear error messages without exposing sensitive filesystem information.

Document known parser vulnerabilities and dependency update procedures where relevant.

## 15. Repository Structure

Design a maintainable repository, adapting the exact structure to the implementation.

An illustrative structure:

    docweave/
        pyproject.toml
        README.md
        LICENSE
        CONTRIBUTING.md
        SECURITY.md
        docs/
        src/
            docweave/
                cli/
                core/
                models/
                analyzers/
                routing/
                repair/
                validation/
                renderers/
                adapters/
                assets/
                api/
        tests/
            unit/
            integration/
            regression/
            fixtures/
        benchmarks/
            corpus_manifest.json
            scripts/
            reports/
        examples/

Separate interfaces, implementations, configuration and rendering.

Avoid giant modules, circular imports, hidden global state and hardcoded machine specific paths.

Choose an appropriate package manager and document setup for Windows, Linux and macOS.

## 16. Development Phases

Implement the project in the following sequence.

### Phase 1: Foundation
- Inspect the repository and environment.
- Establish the package structure.
- Define the intermediate representation.
- Implement format detection.
- Build the adapter interface.
- Add the CLI and Python API.
- Implement native extraction for the initial supported formats.
- Add basic diagnostics and tests.

Deliverable: a usable native document converter.

### Phase 2: Extraction Validation
- Add page and element coverage checks.
- Detect suspicious extraction results.
- Add structured warnings.
- Build a small verified test corpus.
- Implement baseline comparisons.

Deliverable: a converter that can identify common extraction problems.

### Phase 3: Targeted Repair
- Add optional OCR.
- Implement mixed page routing.
- Improve reading order.
- Improve table reconstruction.
- Add alternative parser fallbacks.
- Record repair provenance.

Deliverable: an adaptive conversion pipeline.

### Phase 4: Structural Fidelity
- Improve DOCX and PPTX relationships.
- Improve spreadsheet preservation.
- Add equation and image asset handling.
- Extend the JSON schema.
- Add format specific regression tests.

Deliverable: more faithful structured outputs.

### Phase 5: Performance and Benchmarking
- Profile latency and memory.
- Optimize hot paths.
- Add batch processing.
- Compare against MarkItDown.
- Tune routing thresholds using held out validation data.

Deliverable: reproducible evidence of performance and quality.

### Phase 6: Release
- Finalize documentation.
- Add examples.
- Add continuous integration.
- Validate installation on supported platforms.
- Audit dependency licenses.
- Publish release instructions.

Deliverable: a maintainable open source project ready for external users.

Do not implement all phases simultaneously. Keep each phase functional and testable.

## 17. Engineering Rules

1. Inspect the existing codebase before changing it.
2. Make the smallest architectural decisions that preserve future extensibility.
3. Implement working code rather than stubs or pseudocode.
4. Never claim a feature is complete until it has been implemented and tested.
5. Never hide unsupported features behind success messages.
6. Do not add dependencies without explaining why they are needed.
7. Avoid unnecessary abstraction and premature optimization.
8. Keep default installation lightweight.
9. Never silently discard source content.
10. Preserve source provenance whenever possible.
11. Add regression tests for every fixed extraction failure.
12. Keep all benchmark claims reproducible.
13. Never upload user documents externally by default.
14. Avoid adding a web UI until the core engine is useful.
15. Maintain clear separation between core functionality and optional integrations.

When a design decision is uncertain, select a simple working baseline, document the tradeoff and proceed. Ask me only when a decision materially affects scope, compatibility or product direction.

## 18. Definition of Done

The initial release is complete only when:

- The project installs cleanly in a documented environment.
- The CLI converts supported document formats.
- The Python API works independently of the CLI.
- Markdown and structured JSON outputs are valid.
- Extraction diagnostics identify at least a defined set of common failures.
- Optional fallbacks work when their dependencies are installed.
- The base installation works without API keys or a GPU.
- Automated tests cover core workflows.
- A reproducible benchmark suite exists.
- A comparison against MarkItDown can be run locally.
- No benchmark claims are fabricated.
- Known limitations are documented.
- The README contains installation, usage, examples and architecture documentation.

## 19. Your First Task

Start by inspecting the repository, available runtime, installed Python version and existing dependencies.

Then provide a concise initial assessment covering:
1. What already exists.
2. What must be created.
3. The recommended architecture.
4. The smallest viable milestone.
5. Important technical risks.
6. The first implementation steps.

After this assessment, proceed to implement Phase 1 rather than stopping at planning.

Run the relevant tests and fix actual failures.

At the end of each milestone, report:
- Files created or changed.
- Features implemented.
- Tests executed and their actual results.
- Known limitations.
- The next milestone.

The final goal is a genuinely useful, fast, local first document conversion engine that improves upon MarkItDown through structural fidelity, intelligent fallbacks and verifiable extraction quality, while remaining free to use and practical on ordinary hardware.