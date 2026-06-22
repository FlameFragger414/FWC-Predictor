# FWC Predictor

This is a reproducible FIFA World Cup prediction framework. It estimates each team's tournament probabilities with:

- current World Football Elo ratings
- recent form from international results
- fixed completed 2026 World Cup group-stage results, when available
- optional player, tactical, injury, country, and squad-depth feature overrides
- Poisson score simulation for group matches
- extra-time and penalty shootout simulation for knockouts
- exact 2026 Round-of-32 third-place assignment table
- Monte Carlo tournament simulation with uncertainty in team strength

The current generated output uses 50,000 simulations and fixes completed World Cup group results through `2026-06-21`, the latest completed result available in the cached EloRatings feed at run time.

## Quick Start

```powershell
python -m fwc_predictor --sims 50000 --seed 20260622
```

Outputs are written to:

- `outputs/probabilities.csv`: all 48 teams, stage probabilities, champion confidence intervals, helping factors, risks
- `outputs/group_stage.csv`: expected group points/goals, finish-position probabilities, and best-third probabilities
- `outputs/probability_intervals.csv`: Wilson confidence intervals for every team-stage probability
- `outputs/probabilities.md`: readable stage-probability table
- `outputs/knockout_matchups.csv`: probability of each possible knockout opponent by round
- `outputs/route_summary.csv`: team-by-team knockout path probabilities and opponent ratings
- `outputs/match_probabilities.csv`: expected goals and 90-minute/knockout win probabilities for every group pairing
- `outputs/team_diagnostics.csv`: form, rating, uncertainty, and feature-component diagnostics
- `outputs/feature_importance.csv`: feature-block contribution summary
- `outputs/sanity_checks.csv`: invariant checks for bracket totals, monotonic stage counts, group finishes, and title probability mass
- `outputs/simulation_metadata.json`: run metadata and validation status

Use `--refresh` to pull fresh source files:

```powershell
python -m fwc_predictor --sims 50000 --refresh
```

Run the automated checks:

```powershell
python -m unittest discover -s tests
```

## Data Sources

Default live inputs:

- `https://www.eloratings.net/World.tsv`: current Elo ratings
- `https://www.eloratings.net/latest.tsv`: recent international match results
- `https://en.wikipedia.org/w/index.php?title=Template:2026_FIFA_World_Cup_third-place_table&action=raw`: machine-readable third-place assignment table derived from the FIFA regulations
- `data/groups_2026.csv`: 2026 group draw and EloRatings country codes

Recommended richer feature inputs:

- FIFA rankings and squads: FIFA official rankings, squad lists, suspensions
- xG and tactical data: StatsBomb, Opta, Wyscout, SkillCorner, FBref where licensed
- player values and clubs: Transfermarkt, CIES Football Observatory, club minutes from domestic league feeds
- market and league strength: ClubElo, UEFA coefficient, Opta Power Rankings, domestic league Elo
- injuries: team medical reports, credible news wires, official federation updates
- country factors: World Bank, UN population, FIFA participation reports, national federation data

The file `data/team_feature_overrides.csv` is intentionally header-only. Fill it with numeric feature values by `elo_code` when you have licensed or manually curated data. Missing columns are mean-imputed and increase model uncertainty rather than forcing fake precision.

## Model Summary

Each team starts with a latent strength prior:

```text
effective_rating =
    Elo
  + recent_form_component
  + host_region_component
  + optional_squad_tactical_country_component
  - injury_suspension_component
  + simulation_noise
```

The match model converts latent rating differences into expected goals:

```text
lambda_team_a = base_goal_rate * exp(rating_delta + attack_vs_defense_delta)
lambda_team_b = base_goal_rate * exp(-rating_delta + attack_vs_defense_delta)
```

Group matches may draw. Knockout matches simulate 90 minutes, then extra time, then a penalty shootout where penalty skill and goalkeeper quality can be supplied as optional features.

The tournament simulation:

1. Loads completed World Cup group matches and locks those results.
2. Simulates all unplayed group matches.
3. Applies FIFA-style group ordering: points, goal difference, goals scored, then stochastic/rating tiebreak.
4. Selects the top two from each group and the eight best third-place teams.
5. Uses the exact Round-of-32 third-place allocation table.
6. Simulates every knockout match through the final.
7. Repeats the process at least 50,000 times.

See `docs/MODEL_FRAMEWORK.md` for the full statistical framework, data-cleaning plan, validation plan, missing-data treatment, and limitations.
