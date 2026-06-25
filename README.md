# FWC Predictor

FWC Predictor is an interactive FIFA World Cup forecasting app. It combines World
Football Elo, recent international form, optional squad/tactical inputs, FIFA-style
2026 tournament rules, and Monte Carlo simulation into a Streamlit + Plotly GUI.

Predictions are probabilistic estimates, not guaranteed outcomes.

## Screenshots

Place final product screenshots in `docs/screenshots/` after deployment:

- `docs/screenshots/dashboard.png` - title probabilities and 3D strength view.
- `docs/screenshots/groups.png` - group qualification and finish distributions.
- `docs/screenshots/match_predictor.png` - match xG and probability surface.

## Installation

```bash
python3 -m pip install -e ".[dev]"
```

## Start the GUI

```bash
python -m fwc_predictor
```

Equivalent explicit command:

```bash
python -m fwc_predictor gui
```

The GUI includes tabs for Dashboard, Team rankings, Group predictions, Knockout
bracket, Match predictor, Team comparison, Simulation settings, Model explanation,
and Data quality and warnings. The sidebar controls simulation count, seed, data
refresh, dark mode, and temporary team scenario adjustments.

## CLI usage

```bash
python -m fwc_predictor simulate --sims 50000 --seed 20260622
python -m fwc_predictor simulate --sims 50000 --refresh
```

Legacy flags still work:

```bash
python -m fwc_predictor --sims 50000
```

## Visualisations

The app uses Plotly for:

- title probability rankings and uncertainty intervals
- stage advancement bars
- group qualification and finishing-position charts
- expected-goals comparisons
- team strength radar charts
- likely knockout opponent heatmaps
- tournament stage probability heatmaps
- rating/form trend charts
- 3D strength/form/title scatter
- 3D attack/defence/progress scatter
- 3D knockout probability surface
- 3D team-cluster view

## Model summary

Model constants live in `fwc_predictor/config.py`. Each team starts from Elo, then
receives shrunk recent-form, host-region, optional squad/tactical/country, injury,
and temporary scenario components. The simulator samples latent uncertainty once
per tournament run, simulates group scores with a Poisson expected-goals model, and
simulates knockout matches through regulation, extra time, and penalties.

Group ordering uses points, goal difference, goals scored, head-to-head criteria
for tied teams, then drawing of lots. Best third-place ranking uses points, goal
difference, goals scored, then drawing of lots. The Round-of-32 third-place table
is loaded from the official-style source when available, with a warning if fallback
mapping is used.

## Data sources

Default inputs:

- `data/groups_2026.csv` - projected 48-team field and group slots.
- `data/team_feature_overrides.csv` - optional numeric feature template.
- `https://www.eloratings.net/World.tsv` - Elo ratings.
- `https://www.eloratings.net/latest.tsv` - recent results.
- Wikipedia raw 2026 third-place table template - bracket assignment source.

If live/cached sources are unavailable, the app can fall back to the checked-in
diagnostic snapshot and clearly reports that warning.

## Output files

CLI runs write:

- `probabilities.csv`
- `group_stage.csv`
- `probability_intervals.csv`
- `knockout_matchups.csv`
- `route_summary.csv`
- `match_probabilities.csv`
- `team_diagnostics.csv`
- `feature_importance.csv`
- `sanity_checks.csv`
- `data_quality_warnings.csv`
- `performance_metrics.json`
- `simulation_metadata.json`
- `probabilities.md`

## Development

```bash
python3 -m pip install -e ".[dev]"
python3 -m ruff check .
python3 -m mypy fwc_predictor
python3 -m pytest
coverage run -m pytest && coverage report
```

Pre-commit hooks are configured in `.pre-commit-config.yaml`, and GitHub Actions
runs Ruff, mypy, and tests with coverage.

## Limitations

- The checked-in group file is a projected field; verify it against the official
  draw before production use.
- Public defaults do not include licensed xG, player value, injury, or lineup data.
- The Poisson model is intentionally transparent and does not model game state,
  cards, rest, travel, referee effects, or correlated low-score outcomes.
- Knockout football has high variance, especially because extra time and penalties
  create heavy-tailed outcomes.

See `docs/MODEL_FRAMEWORK.md` for deeper model and validation notes.
