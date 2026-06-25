from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class ModelParameters:
    """Documented model constants used by feature and match simulation code."""

    baseline_elo: float = 1800.0
    host_region_codes: FrozenSet[str] = frozenset({"CA", "MX", "US"})
    host_advantage_elo: float = 42.0
    base_goal_rate: float = 1.29
    rating_goal_scale: float = 410.0
    attack_defense_goal_scale: float = 950.0
    rating_goal_weight: float = 0.48
    attack_defense_goal_weight: float = 0.42
    min_expected_goals: float = 0.08
    max_expected_goals: float = 5.5
    min_poisson_lambda: float = 0.03
    max_poisson_lambda: float = 7.5
    extra_time_minutes_factor: float = 1.0 / 3.0
    penalty_logit_scale: float = 280.0
    latent_attack_defense_shock_share: float = 0.55
    base_rating_uncertainty: float = 30.0
    missing_feature_uncertainty: float = 18.0
    sparse_form_uncertainty: float = 10.0
    required_optional_feature_count: float = 18.0
    strong_opponent_elo: float = 1800.0
    injury_weight: float = -30.0
    optional_rating_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "squad_rating": 42.0,
            "predicted_xi_rating": 34.0,
            "squad_value_eur_m": 16.0,
            "club_level_index": 18.0,
            "league_strength": 10.0,
            "age_balance": 8.0,
            "minutes_recent": 8.0,
            "formation_stability": 8.0,
            "pressing_intensity": 7.0,
            "possession_style": 5.0,
            "defensive_compactness": 7.0,
            "set_piece_strength": 9.0,
            "counterattack_quality": 8.0,
            "climate_fit": 5.0,
            "population_m": 2.0,
            "gdp_per_capita_usd": 3.0,
            "participation_rate": 5.0,
            "domestic_league_strength": 7.0,
            "historical_wc_index": 8.0,
            "football_investment_index": 6.0,
        }
    )


DEFAULT_MODEL_PARAMETERS = ModelParameters()
