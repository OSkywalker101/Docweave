# Security Policy

## Reporting a vulnerability

Open a private security advisory on the repository. Please include a minimal
reproducing input (or a generator for it), the observed behaviour, and the
expected behaviour. Do not include real customer documents.

## Threat model

DocWeave converts untrusted files. The realistic attacker is someone who can get
you to convert a file they control. What they gain is limited: the engine does
not execute document content, does not reach the network, and does not evaluate
formulas.

### What the engine does *not* do

- Execute macros, VBA, or embedded OLE objects.
- Evaluate spreadsheet formulas or macros.
- Execute JavaScript from HTML.
- Resolve external XML entities or perform entity expansion.
- Follow external workbook links.
- Make any network request during conversion.
- Spawn a subprocess.
- Load an ML model or download model weights.

### Defences in place

| Risk | Mitigation |
|---|---|
| XML entity expansion (billion laughs) | `defusedxml`; payloads are refused |
| XXE / external entity file read | `defusedxml` |
| HTML script execution | BeautifulSoup with the pure-Python parser; scripts are never run |
| Zip bombs / oversized archives | File size limit checked before the archive is opened |
| Path traversal via embedded assets | Assets are only written inside the configured `--assets-dir` |
| Formula injection into downstream tools | Cell values are emitted as data with explicit `formula`/`value` fields; nothing is concatenated into a formula |
| Markdown injection from document text | Structural characters are escaped; pipes inside cells are escaped |
| Resource exhaustion (page/cell/row counts) | Explicit, configurable budgets |
| Runaway conversion | Cooperative deadline checked between pages and rows |
| Encrypted files | Detected and refused with a clear message; no bypass attempted |

## Known limitations of these defences

- **The deadline is cooperative.** It is checked between units of work, so a
  single pathological page or row can overrun it. It is not a hard kill, and it
  is not a sandbox.
- **No process isolation.** DocWeave runs in your process with your privileges.
  For untrusted input from untrusted parties, run it in a container or sandbox
  with CPU, memory and wall-clock limits.
- **Parser bugs are possible.** Third-party parsers are the largest attack
  surface. Their advisories are the ones to watch.
- **Markdown output is not sanitised for every consumer.** It is intended for
  LLM ingestion and Markdown viewers. If you render it in a browser, treat it as
  untrusted HTML-adjacent content.

## Dependency updates

Parser libraries are where security fixes will matter most. A reasonable cadence:

1. Watch advisories for PyMuPDF, python-docx, python-pptx, openpyxl and
   BeautifulSoup.
2. Run the test suite after every bump. The test corpus is synthetic and
   generated on demand, so there is no external fixture to re-download.
3. Bump the floor in `pyproject.toml` and record why in the commit message.

## Out of scope

- Vulnerabilities in upstream parsers that only manifest through exotic
  documents already covered by an upstream advisory.
- Denial of service via unbounded work when the user has raised the limits
  deliberately.
- Anything requiring an API key, a network call, or a GPU. Those code paths do
  not exist in the core engine.