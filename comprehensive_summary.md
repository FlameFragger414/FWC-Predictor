# Comprehensive Repository Summary

## Project at a Glance

**FWC Predictor** is a Python package for reproducible FIFA World Cup tournament forecasting. It combines public EloRatings data, recent international results, optional manually curated team features, group-stage state, Poisson match simulation, extra-time and penalty shootout logic, and Monte Carlo tournament simulation to estimate every qualified team's probability of reaching each tournament stage and winning the competition.

The repository is intentionally lightweight: it has no declared third-party runtime dependencies, uses the Python standard library for data loading and simulation, and includes a small `unittest` suite that validates tournament invariants and output generation.

## Repository Scaffold

```text
.
├── README.md
├── comprehensive_summary.md
├── data/
│   ├── groups_2026.csv
│   └── team_feature_overrides.csv
├── docs/
│   └── MODEL_FRAMEWORK.md
├── fwc_predictor/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── data.py
│   ├── features.py
│   └── simulation.py
├── outputs/
│   ├── feature_importance.csv
│   ├── group_stage.csv
│   ├── knockout_matchups.csv
│   ├── match_probabilities.csv
│   ├── probabilities.csv
│   ├── probabilities.md
│   ├── probability_intervals.csv
│   ├── route_summary.csv
│   ├── sanity_checks.csv
│   ├── simulation_metadata.json
│   └── team_diagnostics.csv
├── pyproject.toml
└── tests/
    └── test_simulation.py
```

## Primary User Workflow

1. Run the simulator from the repository root:

   ```bash
   python -m fwc_predictor --sims 50000 --seed 20260622
   ```

2. Optionally refresh cached web inputs:

   ```bash
   python -m fwc_predictor --sims 50000 --refresh
   ```

3. Inspect generated artifacts under `outputs/`.
4. Run the automated checks:

   ```bash
   python -m unittest discover -s tests
   ```

## Package Metadata

### `pyproject.toml`

- Defines the package name as `fwc-predictor`.
- Sets version `0.1.0`.
- Requires Python `>=3.10`.
- Declares no third-party dependencies.
- Exposes a console script named `fwc-predictor` that invokes `fwc_predictor.cli:main`.

### `fwc_predictor/__init__.py`

- Defines package metadata and exports `__version__`.

### `fwc_predictor/__main__.py`

- Enables module execution via `python -m fwc_predictor` by calling the CLI entrypoint.

## Core Application Modules

### `fwc_predictor/cli.py`

The CLI module wires together all major pipeline steps:

- Parses command-line arguments for simulation count, RNG seed, refresh behavior, input paths, cache directory, and output directory.
- Loads team group assignments from `data/groups_2026.csv`.
- Loads optional feature overrides from `data/team_feature_overrides.csv`.
- Loads Elo ratings and latest match results, using cached files unless refresh is requested.
- Extracts completed World Cup group-stage matches from the latest results feed.
- Loads the 2026 third-place assignment table.
- Builds per-team model objects.
- Runs the tournament simulation.
- Writes all output files.
- Prints a concise completion summary.

### `fwc_predictor/data.py`

This module owns data access, parsing, and normalized domain records.

Key dataclasses:

- `Team`: group, slot, EloRatings country code, and display name.
- `MatchResult`: date, teams, goals, tournament code, and venue code.

Key responsibilities:

- Locating the project root.
- Downloading and caching remote text inputs.
- Loading the 2026 group structure.
- Loading optional numeric feature overrides.
- Loading Elo ratings from EloRatings `World.tsv`.
- Loading recent international results from EloRatings `latest.tsv`.
- Filtering completed World Cup group-stage results.
- Creating all group-stage pairings.
- Downloading and parsing the official-style third-place assignment table from Wikipedia's raw template markup.
- Utility parsing for signed numeric values.

Important external default sources:

- `https://www.eloratings.net/World.tsv`
- `https://www.eloratings.net/latest.tsv`
- `https://www.eloratings.net/en.teams.tsv`
- Wikipedia raw template for the 2026 FIFA World Cup third-place table.

### `fwc_predictor/features.py`

This module transforms raw team, rating, result, and optional feature data into model-ready team strength profiles.

Key dataclass:

- `TeamModel`: contains identity fields, Elo, FIFA rank, effective rating, attack rating, defense rating, penalty rating, uncertainty, component contributions, and diagnostics.

Main model-building steps:

1. Compute recent-form features from match history:
   - points per match over 5, 10, and 20 matches
   - goal difference
   - goals for
   - goals against
   - clean-sheet rate
   - performance against strong opponents
2. Standardize optional feature override columns across teams.
3. Combine Elo, recent form, host-region boost, optional squad/tactical/country features, and injury/suspension penalty into an effective rating.
4. Derive attack, defense, and penalty ratings.
5. Increase uncertainty for missing optional features or limited recent-match data.
6. Provide human-readable factor and risk summaries for output reporting.

Host-region codes are currently `CA`, `MX`, and `US`.

Optional feature families include squad quality, predicted XI strength, market value, club level, league strength, age balance, injuries/suspensions, tactical dimensions, penalty skill, climate fit, population, GDP, participation, domestic league strength, historical World Cup index, and football investment.

### `fwc_predictor/simulation.py`

This is the simulation engine and output writer.

Core constants:

- Tournament stages: round of 32, round of 16, quarter-final, semi-final, final, champion.
- Expected slot counts per stage: 32, 16, 8, 4, 2, and 1.
- Group stat keys: points, goal difference, goals for, and goals against.

Key dataclasses:

- `Standing`: mutable group-table row with played, points, goals for, goals against, and goal difference.
- `SimulationResult`: aggregate simulation outputs including stage counts, group finish counts, third-place qualifier counts, group stat sums, matchup counts, current standings, simulation count, and completed-results date.

Main simulation behavior:

1. Samples latent effective, attack, and defense ratings for every team each simulation.
2. Starts from locked completed group-stage results when available.
3. Simulates remaining group matches with a Poisson expected-goals model.
4. Sorts group standings by points, goal difference, goals for, latent strength, then random tiebreak.
5. Advances group winners, runners-up, and the eight best third-place teams.
6. Applies the Round-of-32 third-place assignment table, with a deterministic fallback if needed.
7. Simulates knockouts through 90 minutes, extra time, and penalties.
8. Aggregates stage advancement, matchup frequencies, group statistics, and validation checks.

Output responsibilities include writing:

- team stage probabilities
- group-stage projections
- Wilson confidence intervals
- markdown probability tables
- knockout matchup probabilities
- route summaries
- match-level expected goals and probabilities
- team diagnostics
- feature importance
- sanity checks
- metadata JSON

## Data Files

### `data/groups_2026.csv`

Defines the 48-team tournament field as 12 groups of four teams. Each row contains:

- `group`
- `slot`
- `elo_code`
- `team`

The file is the canonical local source for tournament membership, group assignment, bracket seeding slots, and EloRatings country-code joins.

### `data/team_feature_overrides.csv`

A header-only optional feature template keyed by `elo_code`. The model accepts numeric values for squad, tactical, injury, country, and investment features. Missing values are treated conservatively rather than replaced with fabricated team-specific data.

## Documentation

### `README.md`

The README explains the model purpose, quick-start commands, generated outputs, live data sources, optional richer feature inputs, high-level model formulas, tournament simulation steps, and where to find the full framework notes.

### `docs/MODEL_FRAMEWORK.md`

The framework document provides the deeper statistical and operational design, including:

- prediction objectives
- feature architecture
- player and lineup data recommendations
- tactical and match-level features
- country-level priors
- data cleaning rules
- missing-data treatment
- match-model assumptions
- tournament simulation details
- feature-importance strategy
- overfitting avoidance
- backtesting recommendations
- validation and calibration plans
- limitations
- suggested roadmap for production upgrades

## Generated Output Artifacts

The `outputs/` directory contains a generated simulation snapshot. Important files include:

- `probabilities.csv`: team-level stage probabilities, factor/risk explanations, and title intervals.
- `probabilities.md`: readable markdown probability table.
- `group_stage.csv`: projected group points, goals, finish probabilities, and best-third advancement probability.
- `probability_intervals.csv`: Wilson intervals for team-stage probabilities.
- `knockout_matchups.csv`: possible opponent probabilities by team and round.
- `route_summary.csv`: knockout route and path difficulty information.
- `match_probabilities.csv`: expected goals and match outcome probabilities for group pairings.
- `team_diagnostics.csv`: ratings, uncertainty, form, and model diagnostics.
- `feature_importance.csv`: feature-block contribution summary.
- `sanity_checks.csv`: invariant checks for stage totals, finish totals, and probability mass.
- `simulation_metadata.json`: metadata about the simulation run and validation status.

These files are generated artifacts rather than handwritten source code, but they are useful examples of the package's expected output contract.

## Test Suite

### `tests/test_simulation.py`

The test suite uses `unittest` and synthetic team models to verify core tournament mechanics. It checks that:

- Wilson confidence intervals contain the observed probability.
- Round-of-32 construction yields 16 matches and 32 unique teams.
- Full simulation invariants hold for stage slot totals and validation checks.
- Output writing creates required files.
- Champion probabilities sum to one.
- Sanity checks all pass.
- The 72 group-match probability rows are produced.
- Match-level 90-minute win/draw/loss probabilities sum to one.

## Architectural Strengths

- **Minimal dependency footprint:** standard-library implementation makes setup simple.
- **Reproducibility:** simulation count, seed, inputs, cache, and output directory are explicit.
- **Data-quality awareness:** missing optional feature data widens uncertainty rather than inventing precision.
- **Tournament-rule coverage:** implements 48-team groups, best third-place teams, Round-of-32 allocation, and full knockout progression.
- **Validation outputs:** generated sanity checks make bracket and probability-mass errors easier to detect.
- **Extensibility:** optional feature override columns provide a clear path to richer squad, tactical, and country-level inputs.

## Current Limitations and Risks

- The project currently relies on public Elo and recent-result feeds unless richer override data is supplied.
- Optional feature weights are hand-coded priors rather than fitted coefficients.
- The Poisson model is simple and does not include bivariate score correlation, game-state effects, cards, referee effects, or venue-specific effects.
- The fallback third-place assignment table is deterministic and should only be used when the exact external table is unavailable.
- Generated outputs reflect the input cache and seed from the prior run and should be regenerated when inputs change.
- The group file appears to represent a specific projected or assumed 2026 field; users should verify it against official qualification and draw data before production use.

## Suggested Maintenance Checklist

When updating this repository:

1. Confirm `data/groups_2026.csv` matches the intended tournament field and draw.
2. Regenerate outputs with a documented seed and simulation count.
3. Run `python -m unittest discover -s tests`.
4. Inspect `outputs/sanity_checks.csv` for failures.
5. Review `outputs/simulation_metadata.json` for run metadata and validation status.
6. Keep `docs/MODEL_FRAMEWORK.md` aligned with model changes.
7. Add tests for any modified bracket, scoring, data-loading, or output-writing behavior.

## Recommended Next Enhancements

- Add typed fixtures for historical tournaments and perform backtests.
- Add calibration metrics such as Brier score and log loss on historical match outcomes.
- Replace hand-tuned feature weights with fitted parameters using time-based validation.
- Add optional support for licensed xG, squad, injury, rest, travel, and venue data.
- Add CI configuration that runs the `unittest` suite on every pull request.
- Add schema documentation or data dictionaries for every generated output file.
- Add a CLI option to write run manifests containing source cache checksums.
