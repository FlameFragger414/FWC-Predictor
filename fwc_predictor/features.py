from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

from .config import DEFAULT_MODEL_PARAMETERS, ModelParameters
from .data import MatchResult, Team


@dataclass
class TeamModel:
    code: str
    name: str
    group: str
    slot: str
    elo: float
    fifa_rank: Optional[float]
    effective_rating: float
    attack_rating: float
    defense_rating: float
    penalty_rating: float
    uncertainty: float
    components: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_form_features(
    teams: Dict[str, Team],
    matches: Iterable[MatchResult],
    ratings: Dict[str, Dict[str, float]],
    parameters: ModelParameters = DEFAULT_MODEL_PARAMETERS,
) -> Dict[str, Dict[str, float]]:
    by_team: Dict[str, List[Dict[str, float]]] = {code: [] for code in teams}
    for match in sorted(matches, key=lambda m: m.date, reverse=True):
        if match.team1 in by_team:
            opp_rating = ratings.get(match.team2, {}).get("elo", 1500.0)
            by_team[match.team1].append(
                {
                    "points": _points_for(match.goals1, match.goals2),
                    "gf": match.goals1,
                    "ga": match.goals2,
                    "gd": match.goals1 - match.goals2,
                    "clean": 1.0 if match.goals2 == 0 else 0.0,
                    "strong_points": _points_for(match.goals1, match.goals2)
                    if opp_rating >= parameters.strong_opponent_elo
                    else math.nan,
                }
            )
        if match.team2 in by_team:
            opp_rating = ratings.get(match.team1, {}).get("elo", 1500.0)
            by_team[match.team2].append(
                {
                    "points": _points_for(match.goals2, match.goals1),
                    "gf": match.goals2,
                    "ga": match.goals1,
                    "gd": match.goals2 - match.goals1,
                    "clean": 1.0 if match.goals1 == 0 else 0.0,
                    "strong_points": _points_for(match.goals2, match.goals1)
                    if opp_rating >= parameters.strong_opponent_elo
                    else math.nan,
                }
            )

    result: Dict[str, Dict[str, float]] = {}
    for code, rows in by_team.items():
        features: Dict[str, float] = {}
        for n in (5, 10, 20):
            window = rows[:n]
            if not window:
                features[f"matches_{n}"] = 0.0
                features[f"ppg_{n}"] = 1.0
                features[f"gd_{n}"] = 0.0
                features[f"gf_{n}"] = 1.0
                features[f"ga_{n}"] = 1.0
                features[f"clean_sheet_{n}"] = 0.0
                features[f"strong_ppg_{n}"] = 1.0
                continue
            features[f"matches_{n}"] = float(len(window))
            features[f"ppg_{n}"] = mean(r["points"] for r in window)
            features[f"gd_{n}"] = mean(r["gd"] for r in window)
            features[f"gf_{n}"] = mean(r["gf"] for r in window)
            features[f"ga_{n}"] = mean(r["ga"] for r in window)
            features[f"clean_sheet_{n}"] = mean(r["clean"] for r in window)
            strong = [r["strong_points"] for r in window if not math.isnan(r["strong_points"])]
            features[f"strong_ppg_{n}"] = mean(strong) if strong else 1.0
        result[code] = features
    return result


def _standardize_optional_features(
    teams: Dict[str, Team],
    feature_overrides: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, Dict[str, float]]:
    columns: set[str] = set()
    for values in feature_overrides.values():
        columns.update(values)
    stats: Dict[str, tuple[float, float]] = {}
    for column in columns:
        vals: List[float] = []
        for code, values in feature_overrides.items():
            value = values.get(column)
            if code in teams and value is not None:
                vals.append(float(value))
        if not vals:
            continue
        mu = mean(vals)
        sd = pstdev(vals) or 1.0
        stats[column] = (mu, sd)

    standardized: Dict[str, Dict[str, float]] = {code: {} for code in teams}
    for code in teams:
        raw = feature_overrides.get(code, {})
        for column, (mu, sd) in stats.items():
            value = raw.get(column)
            standardized[code][column] = 0.0 if value is None else (value - mu) / sd
    return standardized


def build_team_models(
    teams: Dict[str, Team],
    ratings: Dict[str, Dict[str, float]],
    matches: Iterable[MatchResult],
    feature_overrides: Dict[str, Dict[str, Optional[float]]],
    parameters: ModelParameters = DEFAULT_MODEL_PARAMETERS,
    scenario_adjustments: Optional[Dict[str, float]] = None,
) -> Dict[str, TeamModel]:
    form = compute_form_features(teams, matches, ratings, parameters)
    optional_z = _standardize_optional_features(teams, feature_overrides)

    models: Dict[str, TeamModel] = {}
    scenario_adjustments = scenario_adjustments or {}

    for code, team in teams.items():
        elo = ratings.get(code, {}).get("elo", 1500.0)
        rank = feature_overrides.get(code, {}).get("fifa_rank")
        f = form.get(code, {})
        components: Dict[str, float] = {"elo": elo - parameters.baseline_elo}

        form_component = (
            10.0 * _clamp(f.get("ppg_5", 1.0) - 1.45, -1.2, 1.2)
            + 8.0 * _clamp(f.get("ppg_10", 1.0) - 1.45, -1.1, 1.1)
            + 5.0 * _clamp(f.get("ppg_20", 1.0) - 1.45, -1.0, 1.0)
            + 8.0 * _clamp(f.get("gd_10", 0.0), -1.5, 1.5)
            + 5.0 * _clamp(f.get("strong_ppg_10", 1.0) - 1.1, -1.0, 1.2)
        )
        components["recent_form"] = form_component

        host_component = (
            parameters.host_advantage_elo
            if code in parameters.host_region_codes
            else 0.0
        )
        components["host_advantage"] = host_component

        optional_component = 0.0
        for column, weight in parameters.optional_rating_weights.items():
            optional_component += weight * optional_z.get(code, {}).get(column, 0.0)
        injury_raw = feature_overrides.get(code, {}).get("injuries_suspensions_index")
        injury_component = (
            parameters.injury_weight * injury_raw if injury_raw is not None else 0.0
        )
        scenario_component = scenario_adjustments.get(code, 0.0)
        components["squad_tactical_country"] = optional_component
        components["injury_suspension"] = injury_component
        components["scenario_adjustment"] = scenario_component

        effective = (
            elo
            + form_component
            + host_component
            + optional_component
            + injury_component
            + scenario_component
        )

        attack_boost = (
            9.0 * _clamp(f.get("gf_10", 1.0) - 1.35, -1.0, 1.5)
            + 14.0 * optional_z.get(code, {}).get("attack_quality", 0.0)
            + 8.0 * optional_z.get(code, {}).get("midfield_quality", 0.0)
        )
        defense_boost = (
            9.0 * _clamp(1.15 - f.get("ga_10", 1.0), -1.3, 1.2)
            + 8.0 * _clamp(f.get("clean_sheet_10", 0.0) - 0.28, -0.3, 0.6)
            + 14.0 * optional_z.get(code, {}).get("defensive_quality", 0.0)
            + 9.0 * optional_z.get(code, {}).get("goalkeeper_quality", 0.0)
        )
        penalty_boost = (
            8.0 * optional_z.get(code, {}).get("penalty_skill", 0.0)
            + 4.0 * optional_z.get(code, {}).get("goalkeeper_quality", 0.0)
            + 0.04 * (effective - 1800.0)
        )

        known_optional = sum(
            1
            for value in feature_overrides.get(code, {}).values()
            if value is not None
        )
        missing_share = 1.0 - min(1.0, known_optional / parameters.required_optional_feature_count)
        form_penalty = parameters.sparse_form_uncertainty if f.get("matches_10", 0.0) < 10 else 0.0
        uncertainty = (
            parameters.base_rating_uncertainty
            + parameters.missing_feature_uncertainty * missing_share
            + form_penalty
        )

        diagnostics = dict(f)
        diagnostics.update(
            {
                "elo": elo,
                "effective_rating": effective,
                "attack_rating": effective + attack_boost,
                "defense_rating": effective + defense_boost,
                "penalty_rating": effective + penalty_boost,
                "rating_uncertainty": uncertainty,
                "known_optional_features": float(known_optional),
            }
        )

        models[code] = TeamModel(
            code=code,
            name=team.name,
            group=team.group,
            slot=team.slot,
            elo=elo,
            fifa_rank=rank,
            effective_rating=effective,
            attack_rating=effective + attack_boost,
            defense_rating=effective + defense_boost,
            penalty_rating=effective + penalty_boost,
            uncertainty=uncertainty,
            components=components,
            diagnostics=diagnostics,
        )
    return models


def strongest_factors(model: TeamModel, current_points: float, current_gd: float) -> str:
    factors: List[tuple[float, str]] = []
    if model.elo >= 1950:
        factors.append((model.elo - 1900, "elite Elo baseline"))
    elif model.elo >= 1850:
        factors.append((model.elo - 1800, "strong Elo baseline"))
    if model.components.get("recent_form", 0.0) > 8:
        factors.append((model.components["recent_form"], "positive recent form"))
    if model.components.get("host_advantage", 0.0) > 0:
        factors.append((model.components["host_advantage"], "host-region advantage"))
    if current_points >= 4:
        factors.append((22 + 3 * current_points, "banked group-stage points"))
    if current_gd >= 2:
        factors.append((12 + 2 * current_gd, "healthy current goal difference"))
    if model.attack_rating - model.effective_rating > 10:
        factors.append((model.attack_rating - model.effective_rating, "attacking output"))
    if model.defense_rating - model.effective_rating > 10:
        factors.append((model.defense_rating - model.effective_rating, "defensive profile"))
    if not factors:
        factors.append((1.0, "viable qualification route"))
    return "; ".join(label for _, label in sorted(factors, reverse=True)[:3])


def biggest_risks(model: TeamModel, current_points: float, current_gd: float) -> str:
    risks: List[tuple[float, str]] = []
    if model.elo < 1650:
        risks.append((1700 - model.elo, "low Elo baseline"))
    if model.components.get("recent_form", 0.0) < -8:
        risks.append((-model.components["recent_form"], "weak recent form"))
    if current_points <= 1:
        risks.append((20 - 4 * current_points, "limited group-stage margin"))
    if current_gd <= -2:
        risks.append((12 - 2 * current_gd, "negative current goal difference"))
    if model.diagnostics.get("ga_10", 1.0) > 1.4:
        risks.append((10 * model.diagnostics["ga_10"], "defensive concession rate"))
    if model.diagnostics.get("known_optional_features", 0.0) < 3:
        risks.append((8.0, "player/tactical inputs not populated"))
    if not risks:
        risks.append((1.0, "normal knockout variance"))
    return "; ".join(label for _, label in sorted(risks, reverse=True)[:3])

