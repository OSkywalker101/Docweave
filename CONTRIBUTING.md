# Contributing

## Setup

```bash
git clone <repo> && cd docweave
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e '.[dev,all]'
python -m pytest tests -q
```

Tests need no API key, no network access and no GPU. Optional-dependency tests
skip cleanly when the package is absent.

## Ground rules

These exist because they were learned the hard way, and violating them is the
fastest way to make this project worse than the tools it replaces.

1. **Never report success without checking.** If a conversion "works", there
   must be a test asserting the extracted *content*, not just the absence of an
   exception. The most damaging converter bug is one that silently returns an
   empty document.
2. **Never fabricate a measurement.** No benchmark number appears in this repo
   that was not produced by a command someone can re-run. If you cannot measure
   it, do not publish it.
3. **One owner per diagnostic code.** If two layers can notice the same problem,
   one emits a *signal* and the other emits the *diagnostic*. Duplicated
   diagnostics train users to ignore them.
4. **Never silently discard content.** If the engine cannot read something, the
   output says so. An omission that is reported is a bug report; an omission that
   is hidden is a data-loss incident.
5. **Preserve provenance.** Every node needs at least one `SourceRef`. A node
   without provenance cannot be traced back to a page, slide or cell.
6. **Keep the default path cheap.** Lazy imports are a correctness requirement.
   `test_heavy_parsers_are_not_imported_by_the_core` will fail your PR if you
   add a top-level import of a heavy parser.
7. **Keep determinism.** Same input + same config ⇒ byte-identical output. No
   timestamps, no RNG, no iteration-order dependence.

## Adding a format

1. Write `src/docweave/adapters/<name>_adapter.py` with an `Adapter` subclass.
   Declare `formats`, `requires`, `capabilities` and `limitations`. The
   limitations field is mandatory and must be honest — `docweave capabilities`
   prints it to users.
2. Register it in `build_registry()`. If the format needs its own delimiter or
   variant, pass that in the factory lambda (see `DelimitedAdapter`).
3. Add fixtures to `tests/fixtures_builder.py`. Fixtures must be *synthetic and
   generated*, never downloaded.
4. Add tests that assert extracted content, not just "did not raise".

## Adding a diagnostic

1. Add the code to `Code` in `models/diagnostics.py`.
2. Decide the category honestly. Ask: can a user act on this without a human? Is
   it confirmed or merely suspected? Should it fail the conversion?
3. Emit it from the correct layer — adapter-specific facts from the adapter,
   judgements about the result from the validator.
4. Include `evidence` (measured numbers) so downstream systems can filter
   programmatically rather than parsing prose.

## Commit and PR

- One logical change per PR.
- Run `python -m pytest tests -q` before pushing. Do not push with a red suite.
- Describe what changed and, for behaviour changes, what a user will notice.
- If you fixed an extraction bug, add a regression test that fails without the
  fix. This is not optional.

## License

Contributions are accepted under the MIT License.