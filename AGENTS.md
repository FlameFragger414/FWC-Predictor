# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
`fwc-predictor` is a single-product, pure-Python (stdlib-only, requires Python >= 3.10) command-line Monte Carlo simulator for FIFA World Cup 2026 probabilities. There are no third-party dependencies, no databases, and no long-running services. See `README.md` for the full command reference and `docs/MODEL_FRAMEWORK.md` for the statistical framework.

### Run / test / lint
- Run: `python3 -m fwc_predictor --sims 50000 --seed 20260622` (a full 50k-sim run finishes in ~11s). Use a small `--sims` value for quick smoke tests.
- Test: `python3 -m unittest discover -s tests` (fully offline; no network needed).
- Lint/build: none configured. `.gitignore` references ruff/mypy caches, but no linter is set up and there is no build step.

### Non-obvious caveats
- `pip install -e .` currently FAILS: `pyproject.toml` has no `[build-system]` table and setuptools cannot auto-discover the package given the flat repo layout. The `fwc-predictor` console script entry point is therefore unavailable. Always run the tool via the module form `python3 -m fwc_predictor` instead. (Do not "fix" the packaging unless explicitly asked.)
- The first run (or any `--refresh`) fetches Elo/results from `eloratings.net` and the third-place table from `en.wikipedia.org`, caching them under `.cache/` (gitignored). Outbound internet is required for that first fetch; subsequent runs work fully offline from the cache.
- `outputs/` is tracked in git and is overwritten on every run. Running the tool will show many modified `outputs/*.csv` files in `git status`; revert them with `git checkout -- outputs/` unless you intend to commit refreshed predictions.
- The "completed group results fixed through" date depends on the live/cached `latest.tsv`, so regenerated outputs can differ from committed ones purely due to newer source data.
