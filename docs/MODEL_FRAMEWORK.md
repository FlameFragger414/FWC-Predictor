# FIFA World Cup Prediction Framework

## Objective

Estimate each country's probability of reaching every tournament stage and winning the World Cup. The model is designed to be data-rich when high-quality inputs are available, but it remains runnable with public Elo and results data.

The current implementation produces:

- Round of 32 probability
- Round of 16 probability
- Quarter-final probability
- Semi-final probability
- Final probability
- Title probability
- 95% simulation confidence interval for title probability
- expected final group points, goals for, goals against, and goal difference
- probabilities of finishing 1st, 2nd, 3rd, and 4th in the group
- probability of advancing through top two versus as a best third-place team
- stage exit probabilities
- conditional title probabilities after reaching key stages
- weighted knockout path difficulty
- match-level expected goals and 90-minute win/draw/loss probabilities
- strongest positive factors
- biggest risks
- possible knockout opponent probabilities by round
- automated sanity checks for bracket totals and probability mass
- interactive Streamlit dashboards, match predictors, exports, and data-quality warnings
- 3D Plotly views for team strength/form/title probability, attack/defence/progress, and knockout probability surfaces

## Feature Architecture

### Team Strength

Core features:

- current Elo rating
- Elo global rank
- recent points per match over 5, 10, and 20 matches
- goals for, goals against, and goal difference over 5, 10, and 20 matches
- clean-sheet rate
- points per match against strong opponents, defined by current opponent Elo >= 1800
- current World Cup group points and goal difference for in-tournament updates

Expected predictive value:

- Elo is usually the strongest single pre-match feature.
- Recent form is useful, but must be shrunk because international schedules are sparse and opponent strength varies.
- Goals and clean sheets are useful when opponent-adjusted. Raw goal totals can overrate teams with easy qualification paths.

### Player And Lineup Data

Optional columns in `data/team_feature_overrides.csv`:

- `squad_rating`
- `predicted_xi_rating`
- `squad_value_eur_m`
- `club_level_index`
- `league_strength`
- `age_balance`
- `injuries_suspensions_index`
- `minutes_recent`
- `attack_quality`
- `midfield_quality`
- `defensive_quality`
- `goalkeeper_quality`
- `squad_depth`
- `penalty_skill`

Recommended construction:

- Predicted XI quality: weighted mean of player ratings or market values for likely starters.
- Squad depth: value or rating of players 12 to 23, with extra weight for injury-prone positions.
- Club level: minutes weighted by club Elo, league strength, and competition tier.
- Age profile: penalize extreme old/young squads; reward peak-age minute share.
- Injuries/suspensions: expected minutes lost times player importance.

### Tactical And Match-Level Data

Optional columns:

- `formation_stability`
- `pressing_intensity`
- `possession_style`
- `defensive_compactness`
- `set_piece_strength`
- `counterattack_quality`
- `travel_fatigue`
- `climate_fit`

Recommended construction:

- Pressing: PPDA, high turnovers, counterpress recoveries.
- Possession: possession share adjusted for opponent quality.
- Compactness: defensive line height, box entries allowed, central progression allowed.
- Set pieces: xG for and against from corners/free kicks.
- Counterattack: direct attacks, transition xG, speed of attack.
- Travel/rest/climate: venue-to-base travel, rest days, temperature/humidity, altitude, and time-zone change.

### Country-Level Background

Optional columns:

- `population_m`
- `gdp_per_capita_usd`
- `participation_rate`
- `domestic_league_strength`
- `historical_wc_index`
- `football_investment_index`

These should be low-weight prior features. They help forecast long-run talent production, but they are much less informative than current squad quality and Elo once teams have qualified.

## Data Cleaning And Structure

Use one canonical team key. This project uses EloRatings country codes in `data/groups_2026.csv`.

Recommended tables:

- `teams`: team code, country name, confederation, group, host flag
- `matches`: date, competition, venue, home/away/neutral, teams, goals, xG, cards, penalties
- `ratings`: date, team, Elo, FIFA rank, other power ratings
- `players`: player, team, position, club, age, market value, minutes, injury status
- `lineups`: match, team, formation, starters, substitutes, minutes
- `venues`: stadium, city, country, altitude, climate, travel distance
- `country_factors`: population, GDP, participation, federation investment

Cleaning rules:

- Normalize country names to codes before joining.
- Convert all dates to UTC and keep local kickoff time separately.
- Treat penalty shootout wins as draws in 90/120-minute modelling.
- Separate pre-match features from post-match outcomes to avoid leakage.
- Opponent-adjust all form, xG, and tactical features.
- Cap extreme player values and log-transform monetary variables.
- Use rolling windows that end before the match being predicted.

## Missing Data

The model never fills missing values with invented team-specific facts.

Current treatment:

- Missing optional numeric features are mean-imputed after standardization, giving zero contribution.
- Teams with few optional features receive higher latent-rating uncertainty.
- Missing recent-match history receives conservative neutral form and extra uncertainty.

Recommended production treatment:

- Use hierarchical imputation by confederation, Elo tier, and competition level.
- Include missingness indicators for important feature families.
- Run sensitivity scenarios for uncertain injuries and predicted lineups.
- Keep a data-quality score per team and widen intervals when data quality is low.

## Match Model

The default implementation uses an expected-goals Poisson model:

```text
goals_team_a ~ Poisson(lambda_a)
goals_team_b ~ Poisson(lambda_b)
```

The goal rates depend on:

- effective rating difference
- attacking quality of the scoring team
- defensive and goalkeeper quality of the opponent
- host-region advantage
- team-level uncertainty sampled each simulation

For knockouts:

1. Simulate 90 minutes.
2. If tied, simulate extra time at one-third of normal match intensity.
3. If tied, simulate penalties using penalty and goalkeeper ratings.

Production upgrade:

- Fit a bivariate Poisson or Dixon-Coles adjustment for low-score correlation.
- Estimate parameters on international matches with time decay.
- Include tournament-stage effects, referee/card rates, and game-state effects.

## Tournament Simulation

The simulation respects the 2026 48-team format:

- 12 groups of four
- top two in each group advance
- eight best third-place teams advance
- exact Round-of-32 allocation table for third-place teams
- fixed knockout bracket

Each simulation samples team strength once, then simulates all remaining matches. This captures both match randomness and uncertainty about true team quality. Group ordering follows points, goal difference, goals scored, head-to-head criteria for tied teams, and drawing of lots when unresolved. Best third-place ordering uses points, goal difference, goals scored, and drawing of lots rather than hidden Elo ordering.

## Application Architecture

The package is split into:

- `config`: documented model parameters and feature weights
- `data`: source loading and parsing
- `validation`: tournament-field, feature-file, completed-match, and warning checks
- `features`: team model construction
- `simulation`: tournament engine and output schemas
- `pipeline`: shared GUI/CLI orchestration, offline fallback, warnings, and performance metrics
- `charts`: testable Plotly figure builders
- `gui`: Streamlit application

Running `python -m fwc_predictor` starts the GUI. Running `python -m fwc_predictor simulate --sims 50000` writes reproducible CLI outputs.

## Feature Importance

The output `feature_importance.csv` reports mean absolute Elo-point contribution by feature block. In the current public-data run, Elo dominates because player/tactical/country overrides are not populated.

For a production model, use three feature-importance layers:

- global permutation importance on historical match log loss
- SHAP or grouped coefficient analysis for the match model
- tournament sensitivity: rerun simulations while perturbing one feature block

The most predictive features in previous international football models are usually:

1. Elo or market-implied team strength
2. squad quality and player availability
3. opponent-adjusted xG difference
4. recent opponent-adjusted form
5. goalkeeper and defensive quality
6. set-piece differential
7. path difficulty and bracket position
8. rest/travel/host effects

## Avoiding Overfitting

Use:

- time-based train/validation/test splits
- nested cross-validation for hyperparameters
- strong regularization for sparse international features
- grouped feature weights instead of dozens of fragile one-off coefficients
- opponent-adjusted rolling features only
- calibration checks by probability bin
- log loss and Brier score, not just accuracy
- conservative priors for country-level variables

Avoid:

- using post-draw tournament odds as both feature and target
- tuning to a single World Cup
- treating friendlies and knockout World Cup matches as equally informative
- unshrunk recent-form streaks
- raw goals against weak opponents without opponent adjustment

## Backtesting

Recommended backtest:

1. Collect all international matches from at least 2006 onward.
2. Freeze a feature snapshot before each historical World Cup.
3. Reconstruct each tournament's actual groups, bracket rules, and host effects.
4. Simulate each historical tournament 50,000 times using only pre-tournament data.
5. Score:
   - match-level log loss
   - match-level Brier score
   - calibration by probability bucket
   - stage-probability Brier score
   - champion probability assigned to the actual winner
6. Compare against baselines:
   - Elo only
   - FIFA ranking only
   - betting-market implied probabilities, if legally available
   - equal-probability naive model
7. Run ablations:
   - remove squad features
   - remove xG features
   - remove recent form
   - remove country priors
   - remove host/rest/travel effects

For 2026 specifically, update the backtest so the expanded 48-team format is tested with simulated synthetic 48-team draws or continental tournaments, because no previous World Cup used exactly this structure.

## Limitations

- Public default run has no licensed xG, player-value, injury, or tactical feed populated.
- EloRatings are excellent but not perfect; they can lag sudden tactical or lineup changes.
- International sample sizes are small and uneven across confederations.
- Knockout football has high variance; penalties create heavy tails.
- Group-stage incentives can change when teams have already qualified.
- The model infers remaining group fixtures from the draw rather than venue-specific scheduling.
- Climate, travel, and rest are only represented when optional data is supplied.
- Current results depend on the source feed refresh time.

The model should be interpreted as a probabilistic decision aid, not as a deterministic ranking.
