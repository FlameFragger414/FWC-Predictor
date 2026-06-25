# Comprehensive Repository Summary

## Project at a Glance

FWC Predictor is now a GUI-first FIFA World Cup forecasting application. It uses
World Football Elo, recent form, optional squad/tactical/country overrides, FIFA-style
2026 tournament rules, Poisson match simulation, extra time, penalties, and Monte Carlo
uncertainty to estimate team stage and title probabilities.

Running `python -m fwc_predictor` launches the Streamlit interface. Running
`python -m fwc_predictor simulate --sims 50000` keeps the reproducible CLI export mode.

## Main Components

- `fwc_predictor/config.py`: documented model constants and feature weights.
- `fwc_predictor/data.py`: source loading and parsing.
- `fwc_predictor/validation.py`: group, feature, completed-match, and data-quality checks.
- `fwc_predictor/features.py`: team strength, attack, defence, penalty, and uncertainty models.
- `fwc_predictor/simulation.py`: tournament engine, tiebreaks, bracket, and output writer.
- `fwc_predictor/pipeline.py`: shared GUI/CLI orchestration, fallback handling, warnings, metrics.
- `fwc_predictor/charts.py`: Plotly dashboard, heatmap, radar, and 3D chart builders.
- `fwc_predictor/gui.py`: Streamlit app with tabs, controls, exports, warnings, and dark mode.

## GUI Features

Tabs include Dashboard, Team rankings, Group predictions, Knockout bracket, Match
predictor, Team comparison, Simulation settings, Model explanation, and Data quality
and warnings. Users can change simulation count, seed, refresh behavior, dark mode,
and temporary team Elo scenario adjustments.

## Visualisations

The app includes title rankings, stage bars, group qualification charts, finishing
distributions, expected-goals charts, strength radars, uncertainty intervals,
knockout opponent heatmaps, probability heatmaps, rating/form trends, and 3D Plotly
views for strength/form/title probability, attack/defence/progress, probability
surface, and team clusters.

## Model and Rules

Model parameters live in `config.py`. Group ranking now uses points, goal difference,
goals scored, head-to-head criteria for tied teams, and drawing of lots. Best third
ranking uses points, goal difference, goals scored, and drawing of lots. Completed
matches are validated for unknown teams, cross-group matches, duplicates, and invalid
scores.

## Outputs

CLI mode writes probabilities, group projections, intervals, knockout matchups,
route summaries, match probabilities, diagnostics, feature importance, sanity checks,
data-quality warnings, performance metrics, metadata, and markdown summaries.

## Tooling

`pyproject.toml` declares runtime and development dependencies. The project includes
pytest, Ruff, mypy, coverage, pre-commit, and GitHub Actions.

## Current Limitations

The checked-in group file is a projected field, optional rich features are mostly a
schema template, and public defaults do not include licensed xG, injury, lineup, rest,
travel, referee, or game-state data. Outputs are probabilistic forecasts, not guaranteed
outcomes.
