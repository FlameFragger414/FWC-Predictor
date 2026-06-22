from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .data import MatchResult, Team, all_group_pairs
from .features import TeamModel, biggest_risks, strongest_factors


STAGES = [
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "champion",
]

STAGE_SLOT_COUNTS = {
    "round_of_32": 32,
    "round_of_16": 16,
    "quarter_final": 8,
    "semi_final": 4,
    "final": 2,
    "champion": 1,
}

GROUP_STAT_KEYS = ("points", "goal_difference", "goals_for", "goals_against")


@dataclass
class Standing:
    code: str
    played: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def add(self, gf: int, ga: int) -> None:
        self.played += 1
        self.gf += gf
        self.ga += ga
        if gf > ga:
            self.points += 3
        elif gf == ga:
            self.points += 1


@dataclass
class SimulationResult:
    stage_counts: Dict[str, Dict[str, int]]
    group_finish_counts: Dict[str, Dict[int, int]]
    third_qualifier_counts: Dict[str, int]
    group_stat_sums: Dict[str, Dict[str, float]]
    matchup_counts: Dict[Tuple[str, str, str], int]
    current_standings: Dict[str, Standing]
    simulations: int
    as_of: Optional[str]


def poisson(rng: random.Random, lam: float) -> int:
    lam = max(0.03, min(7.5, lam))
    limit = math.exp(-lam)
    k = 0
    product = 1.0
    while product > limit:
        k += 1
        product *= rng.random()
    return k - 1


def expected_goals(
    first: str,
    second: str,
    models: Dict[str, TeamModel],
    latent_effective: Dict[str, float],
    latent_attack: Dict[str, float],
    latent_defense: Dict[str, float],
) -> Tuple[float, float]:
    base = 1.29
    rating_delta = (latent_effective[first] - latent_effective[second]) / 410.0
    attack_delta = (latent_attack[first] - latent_defense[second]) / 950.0
    defense_delta = (latent_attack[second] - latent_defense[first]) / 950.0
    lam1 = base * math.exp(0.48 * rating_delta + 0.42 * attack_delta)
    lam2 = base * math.exp(-0.48 * rating_delta + 0.42 * defense_delta)
    return max(0.08, min(5.5, lam1)), max(0.08, min(5.5, lam2))


def poisson_pmf(lam: float, max_goals: int = 10) -> List[float]:
    probabilities = [math.exp(-lam)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * lam / goals)
    total = sum(probabilities)
    if total:
        probabilities = [value / total for value in probabilities]
    return probabilities


def match_probability_summary(
    first: str,
    second: str,
    models: Dict[str, TeamModel],
) -> Dict[str, float]:
    latent_effective = {
        first: models[first].effective_rating,
        second: models[second].effective_rating,
    }
    latent_attack = {
        first: models[first].attack_rating,
        second: models[second].attack_rating,
    }
    latent_defense = {
        first: models[first].defense_rating,
        second: models[second].defense_rating,
    }
    lam1, lam2 = expected_goals(
        first, second, models, latent_effective, latent_attack, latent_defense
    )
    pmf1 = poisson_pmf(lam1)
    pmf2 = poisson_pmf(lam2)
    p_first_win = 0.0
    p_draw = 0.0
    p_second_win = 0.0
    for g1, p1 in enumerate(pmf1):
        for g2, p2 in enumerate(pmf2):
            probability = p1 * p2
            if g1 > g2:
                p_first_win += probability
            elif g1 == g2:
                p_draw += probability
            else:
                p_second_win += probability

    et_pmf1 = poisson_pmf(lam1 / 3.0)
    et_pmf2 = poisson_pmf(lam2 / 3.0)
    p_et_first = 0.0
    p_et_draw = 0.0
    for g1, p1 in enumerate(et_pmf1):
        for g2, p2 in enumerate(et_pmf2):
            probability = p1 * p2
            if g1 > g2:
                p_et_first += probability
            elif g1 == g2:
                p_et_draw += probability
    penalty_delta = (models[first].penalty_rating - models[second].penalty_rating) / 280.0
    p_penalty_first = 1.0 / (1.0 + math.exp(-penalty_delta))
    p_first_advance = p_first_win + p_draw * (
        p_et_first + p_et_draw * p_penalty_first
    )

    return {
        "expected_goals_team1": lam1,
        "expected_goals_team2": lam2,
        "prob_team1_win_90": p_first_win,
        "prob_draw_90": p_draw,
        "prob_team2_win_90": p_second_win,
        "prob_team1_advance_knockout": p_first_advance,
        "prob_team2_advance_knockout": 1.0 - p_first_advance,
    }


def simulate_score(
    rng: random.Random,
    first: str,
    second: str,
    models: Dict[str, TeamModel],
    latent_effective: Dict[str, float],
    latent_attack: Dict[str, float],
    latent_defense: Dict[str, float],
    minutes_factor: float = 1.0,
) -> Tuple[int, int]:
    lam1, lam2 = expected_goals(
        first, second, models, latent_effective, latent_attack, latent_defense
    )
    return poisson(rng, lam1 * minutes_factor), poisson(rng, lam2 * minutes_factor)


def simulate_knockout_winner(
    rng: random.Random,
    first: str,
    second: str,
    models: Dict[str, TeamModel],
    latent_effective: Dict[str, float],
    latent_attack: Dict[str, float],
    latent_defense: Dict[str, float],
) -> str:
    g1, g2 = simulate_score(
        rng, first, second, models, latent_effective, latent_attack, latent_defense
    )
    if g1 > g2:
        return first
    if g2 > g1:
        return second

    et1, et2 = simulate_score(
        rng,
        first,
        second,
        models,
        latent_effective,
        latent_attack,
        latent_defense,
        minutes_factor=1.0 / 3.0,
    )
    if et1 > et2:
        return first
    if et2 > et1:
        return second

    penalty_delta = (models[first].penalty_rating - models[second].penalty_rating) / 280.0
    p_first = 1.0 / (1.0 + math.exp(-penalty_delta))
    return first if rng.random() < p_first else second


def initial_standings(
    teams: Dict[str, Team],
    completed_matches: Iterable[MatchResult],
) -> Tuple[Dict[str, Standing], set[Tuple[str, str]]]:
    standings = {code: Standing(code=code) for code in teams}
    completed_pairs: set[Tuple[str, str]] = set()
    for match in completed_matches:
        standings[match.team1].add(match.goals1, match.goals2)
        standings[match.team2].add(match.goals2, match.goals1)
        completed_pairs.add(tuple(sorted((match.team1, match.team2))))
    return standings, completed_pairs


def sorted_group(
    codes: List[str],
    standings: Dict[str, Standing],
    latent_effective: Dict[str, float],
    rng: random.Random,
) -> List[str]:
    return sorted(
        codes,
        key=lambda code: (
            standings[code].points,
            standings[code].gd,
            standings[code].gf,
            latent_effective[code],
            rng.random(),
        ),
        reverse=True,
    )


def fallback_third_assignment(advancing_groups: Iterable[str]) -> Dict[str, str]:
    """Deterministic fallback if the external FIFA Annex C table is unavailable."""
    eligible = {
        "A": ["C", "E", "F", "H", "I"],
        "B": ["E", "F", "G", "I", "J"],
        "D": ["B", "E", "F", "I", "J"],
        "E": ["A", "B", "C", "D", "F"],
        "G": ["A", "E", "H", "I", "J"],
        "I": ["C", "D", "F", "G", "H"],
        "K": ["D", "E", "I", "J", "L"],
        "L": ["E", "H", "I", "J", "K"],
    }
    remaining = set(advancing_groups)
    output: Dict[str, str] = {}
    for winner_group in ["A", "B", "D", "E", "G", "I", "K", "L"]:
        choices = [group for group in eligible[winner_group] if group in remaining]
        if not choices:
            choices = sorted(remaining)
        output[winner_group] = choices[0]
        remaining.remove(choices[0])
    return output


def round_of_32_matches(
    placements: Dict[str, List[str]],
    third_qualifiers: Dict[str, str],
    third_assignments: Dict[str, Dict[str, str]],
) -> Dict[int, Tuple[str, str]]:
    third_groups = "".join(sorted(third_qualifiers))
    third_map = third_assignments.get(third_groups) or fallback_third_assignment(
        third_qualifiers
    )
    matches: Dict[int, Tuple[str, str]] = {
        73: (placements["A"][1], placements["B"][1]),
        75: (placements["F"][0], placements["C"][1]),
        76: (placements["C"][0], placements["F"][1]),
        78: (placements["E"][1], placements["I"][1]),
        83: (placements["K"][1], placements["L"][1]),
        84: (placements["H"][0], placements["J"][1]),
        86: (placements["J"][0], placements["H"][1]),
        88: (placements["D"][1], placements["G"][1]),
    }
    match_numbers = {
        "A": 79,
        "B": 85,
        "D": 81,
        "E": 74,
        "G": 82,
        "I": 77,
        "K": 87,
        "L": 80,
    }
    for winner_group, match_number in match_numbers.items():
        third_group = third_map[winner_group]
        matches[match_number] = (
            placements[winner_group][0],
            third_qualifiers[third_group],
        )
    return matches


def record_matchup(
    matchup_counts: Dict[Tuple[str, str, str], int],
    first: str,
    second: str,
    stage: str,
) -> None:
    matchup_counts[(first, second, stage)] = matchup_counts.get((first, second, stage), 0) + 1
    matchup_counts[(second, first, stage)] = matchup_counts.get((second, first, stage), 0) + 1


def simulate_tournament(
    teams: Dict[str, Team],
    models: Dict[str, TeamModel],
    completed_matches: List[MatchResult],
    third_assignments: Dict[str, Dict[str, str]],
    simulations: int = 50_000,
    seed: int = 20260622,
) -> SimulationResult:
    rng = random.Random(seed)
    group_pairs = all_group_pairs(teams)
    by_group: Dict[str, List[str]] = {}
    for code, team in teams.items():
        by_group.setdefault(team.group, []).append(code)
    for group in by_group:
        by_group[group].sort(key=lambda code: teams[code].slot)

    base_standings, completed_pairs = initial_standings(teams, completed_matches)
    stage_counts = {code: {stage: 0 for stage in STAGES} for code in teams}
    group_finish_counts = {code: {1: 0, 2: 0, 3: 0, 4: 0} for code in teams}
    third_qualifier_counts = {code: 0 for code in teams}
    group_stat_sums = {
        code: {key: 0.0 for key in GROUP_STAT_KEYS} for code in teams
    }
    matchup_counts: Dict[Tuple[str, str, str], int] = {}
    as_of = max((m.date.isoformat() for m in completed_matches), default=None)

    for _ in range(simulations):
        latent_effective: Dict[str, float] = {}
        latent_attack: Dict[str, float] = {}
        latent_defense: Dict[str, float] = {}
        for code, model in models.items():
            shock = rng.gauss(0.0, model.uncertainty)
            latent_effective[code] = model.effective_rating + shock
            latent_attack[code] = model.attack_rating + shock * 0.55
            latent_defense[code] = model.defense_rating + shock * 0.55

        standings = {
            code: Standing(
                code=standing.code,
                played=standing.played,
                points=standing.points,
                gf=standing.gf,
                ga=standing.ga,
            )
            for code, standing in base_standings.items()
        }

        for group, pairs in group_pairs.items():
            for first, second in pairs:
                if tuple(sorted((first, second))) in completed_pairs:
                    continue
                g1, g2 = simulate_score(
                    rng,
                    first,
                    second,
                    models,
                    latent_effective,
                    latent_attack,
                    latent_defense,
                )
                standings[first].add(g1, g2)
                standings[second].add(g2, g1)

        placements: Dict[str, List[str]] = {}
        third_rank_rows: List[Tuple[int, int, int, float, float, str, str]] = []
        for group, codes in by_group.items():
            ordered = sorted_group(codes, standings, latent_effective, rng)
            placements[group] = ordered
            for code in codes:
                group_stat_sums[code]["points"] += standings[code].points
                group_stat_sums[code]["goal_difference"] += standings[code].gd
                group_stat_sums[code]["goals_for"] += standings[code].gf
                group_stat_sums[code]["goals_against"] += standings[code].ga
            for idx, code in enumerate(ordered, start=1):
                group_finish_counts[code][idx] += 1
            third = ordered[2]
            third_rank_rows.append(
                (
                    standings[third].points,
                    standings[third].gd,
                    standings[third].gf,
                    latent_effective[third],
                    rng.random(),
                    group,
                    third,
                )
            )

        third_rank_rows.sort(reverse=True)
        third_qualifiers = {group: code for *_, group, code in third_rank_rows[:8]}
        for code in third_qualifiers.values():
            third_qualifier_counts[code] += 1

        qualified = set()
        for group, ordered in placements.items():
            qualified.update(ordered[:2])
        qualified.update(third_qualifiers.values())
        for code in qualified:
            stage_counts[code]["round_of_32"] += 1

        r32 = round_of_32_matches(placements, third_qualifiers, third_assignments)
        winners: Dict[int, str] = {}
        for match_number in range(73, 89):
            first, second = r32[match_number]
            record_matchup(matchup_counts, first, second, "round_of_32")
            winners[match_number] = simulate_knockout_winner(
                rng,
                first,
                second,
                models,
                latent_effective,
                latent_attack,
                latent_defense,
            )
            stage_counts[winners[match_number]]["round_of_16"] += 1

        r16_pairs = {
            89: (winners[73], winners[75]),
            90: (winners[74], winners[77]),
            91: (winners[76], winners[78]),
            92: (winners[79], winners[80]),
            93: (winners[83], winners[84]),
            94: (winners[81], winners[82]),
            95: (winners[86], winners[88]),
            96: (winners[85], winners[87]),
        }
        for match_number in range(89, 97):
            first, second = r16_pairs[match_number]
            record_matchup(matchup_counts, first, second, "round_of_16")
            winners[match_number] = simulate_knockout_winner(
                rng,
                first,
                second,
                models,
                latent_effective,
                latent_attack,
                latent_defense,
            )
            stage_counts[winners[match_number]]["quarter_final"] += 1

        qf_pairs = {
            97: (winners[89], winners[90]),
            98: (winners[93], winners[94]),
            99: (winners[91], winners[92]),
            100: (winners[95], winners[96]),
        }
        for match_number in range(97, 101):
            first, second = qf_pairs[match_number]
            record_matchup(matchup_counts, first, second, "quarter_final")
            winners[match_number] = simulate_knockout_winner(
                rng,
                first,
                second,
                models,
                latent_effective,
                latent_attack,
                latent_defense,
            )
            stage_counts[winners[match_number]]["semi_final"] += 1

        sf_pairs = {
            101: (winners[97], winners[98]),
            102: (winners[99], winners[100]),
        }
        for match_number in (101, 102):
            first, second = sf_pairs[match_number]
            record_matchup(matchup_counts, first, second, "semi_final")
            winners[match_number] = simulate_knockout_winner(
                rng,
                first,
                second,
                models,
                latent_effective,
                latent_attack,
                latent_defense,
            )
            stage_counts[winners[match_number]]["final"] += 1

        first, second = winners[101], winners[102]
        record_matchup(matchup_counts, first, second, "final")
        champion = simulate_knockout_winner(
            rng,
            first,
            second,
            models,
            latent_effective,
            latent_attack,
            latent_defense,
        )
        stage_counts[champion]["champion"] += 1

    return SimulationResult(
        stage_counts=stage_counts,
        group_finish_counts=group_finish_counts,
        third_qualifier_counts=third_qualifier_counts,
        group_stat_sums=group_stat_sums,
        matchup_counts=matchup_counts,
        current_standings=base_standings,
        simulations=simulations,
        as_of=as_of,
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def stage_probability(result: SimulationResult, code: str, stage: str) -> float:
    return result.stage_counts[code][stage] / result.simulations


def top_stage_opponents(
    team_code: str,
    stage: str,
    teams: Dict[str, Team],
    result: SimulationResult,
    limit: int = 3,
) -> str:
    rows: List[Tuple[int, str]] = []
    for (team, opponent, matchup_stage), count in result.matchup_counts.items():
        if team == team_code and matchup_stage == stage:
            rows.append((count, opponent))
    rows.sort(reverse=True)
    parts = []
    for count, opponent in rows[:limit]:
        parts.append(f"{teams[opponent].name} {count / result.simulations:.1%}")
    return "; ".join(parts)


def path_difficulty_index(
    team_code: str,
    models: Dict[str, TeamModel],
    result: SimulationResult,
) -> Optional[float]:
    stage_weights = {
        "round_of_32": 0.35,
        "round_of_16": 0.55,
        "quarter_final": 0.75,
        "semi_final": 0.95,
        "final": 1.15,
    }
    weighted_rating = 0.0
    weighted_probability = 0.0
    for (team, opponent, stage), count in result.matchup_counts.items():
        if team != team_code:
            continue
        weight = stage_weights.get(stage, 1.0)
        probability = count / result.simulations
        weighted_rating += probability * weight * models[opponent].effective_rating
        weighted_probability += probability * weight
    if weighted_probability == 0:
        return None
    return weighted_rating / weighted_probability


def validation_checks(
    teams: Dict[str, Team],
    result: SimulationResult,
) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []

    def add(name: str, passed: bool, value: object, expected: object) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": str(value),
                "expected": str(expected),
            }
        )

    for stage, slots in STAGE_SLOT_COUNTS.items():
        total = sum(result.stage_counts[code][stage] for code in teams)
        add(
            f"stage_total_{stage}",
            total == slots * result.simulations,
            total,
            slots * result.simulations,
        )

    for code, team in teams.items():
        finish_total = sum(result.group_finish_counts[code].values())
        add(
            f"{team.name}_group_finish_total",
            finish_total == result.simulations,
            finish_total,
            result.simulations,
        )
        monotonic = all(
            result.stage_counts[code][STAGES[i]]
            >= result.stage_counts[code][STAGES[i + 1]]
            for i in range(len(STAGES) - 1)
        )
        add(f"{team.name}_stage_monotonic", monotonic, monotonic, True)

    champion_sum = sum(
        result.stage_counts[code]["champion"] / result.simulations for code in teams
    )
    add("champion_probability_sum", abs(champion_sum - 1.0) < 1e-12, champion_sum, 1.0)

    third_total = sum(result.third_qualifier_counts.values())
    add(
        "best_third_total",
        third_total == 8 * result.simulations,
        third_total,
        8 * result.simulations,
    )

    return checks


def write_outputs(
    output_dir: Path,
    teams: Dict[str, Team],
    models: Dict[str, TeamModel],
    result: SimulationResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities_path = output_dir / "probabilities.csv"
    with probabilities_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "team",
            "group",
            "elo",
            "effective_rating",
            "current_group_points",
            "current_group_goal_difference",
            "expected_group_points",
            "expected_group_goal_difference",
            "expected_group_goals_for",
            "expected_group_goals_against",
            "prob_finish_1st",
            "prob_finish_2nd",
            "prob_finish_3rd",
            "prob_finish_4th",
            "prob_top2",
            "prob_advance_as_third",
            "prob_round_of_32",
            "prob_round_of_16",
            "prob_quarter_final",
            "prob_semi_final",
            "prob_final",
            "prob_champion",
            "prob_exit_group",
            "prob_exit_round_of_32",
            "prob_exit_round_of_16",
            "prob_exit_quarter_final",
            "prob_exit_semi_final",
            "prob_runner_up",
            "prob_win_given_round_of_32",
            "prob_win_given_quarter_final",
            "champion_ci_low",
            "champion_ci_high",
            "path_difficulty_index",
            "most_likely_round_of_32_opponents",
            "most_likely_round_of_16_opponents",
            "strongest_factors_helping",
            "biggest_weaknesses_or_risks",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rows = []
        for code, team in teams.items():
            model = models[code]
            standing = result.current_standings[code]
            r32 = stage_probability(result, code, "round_of_32")
            r16 = stage_probability(result, code, "round_of_16")
            qf = stage_probability(result, code, "quarter_final")
            sf = stage_probability(result, code, "semi_final")
            final = stage_probability(result, code, "final")
            champion = stage_probability(result, code, "champion")
            champion_low, champion_high = wilson_interval(
                result.stage_counts[code]["champion"], result.simulations
            )
            finish = {
                place: result.group_finish_counts[code][place] / result.simulations
                for place in (1, 2, 3, 4)
            }
            path_difficulty = path_difficulty_index(code, models, result)
            row = {
                "team": team.name,
                "group": team.group,
                "elo": round(model.elo, 1),
                "effective_rating": round(model.effective_rating, 1),
                "current_group_points": standing.points,
                "current_group_goal_difference": standing.gd,
                "expected_group_points": result.group_stat_sums[code]["points"]
                / result.simulations,
                "expected_group_goal_difference": result.group_stat_sums[code][
                    "goal_difference"
                ]
                / result.simulations,
                "expected_group_goals_for": result.group_stat_sums[code]["goals_for"]
                / result.simulations,
                "expected_group_goals_against": result.group_stat_sums[code][
                    "goals_against"
                ]
                / result.simulations,
                "prob_finish_1st": finish[1],
                "prob_finish_2nd": finish[2],
                "prob_finish_3rd": finish[3],
                "prob_finish_4th": finish[4],
                "prob_top2": finish[1] + finish[2],
                "prob_advance_as_third": result.third_qualifier_counts[code]
                / result.simulations,
                "prob_round_of_32": r32,
                "prob_round_of_16": r16,
                "prob_quarter_final": qf,
                "prob_semi_final": sf,
                "prob_final": final,
                "prob_champion": champion,
                "prob_exit_group": 1.0 - r32,
                "prob_exit_round_of_32": r32 - r16,
                "prob_exit_round_of_16": r16 - qf,
                "prob_exit_quarter_final": qf - sf,
                "prob_exit_semi_final": sf - final,
                "prob_runner_up": final - champion,
                "prob_win_given_round_of_32": champion / r32 if r32 else 0.0,
                "prob_win_given_quarter_final": champion / qf if qf else 0.0,
                "champion_ci_low": champion_low,
                "champion_ci_high": champion_high,
                "path_difficulty_index": ""
                if path_difficulty is None
                else round(path_difficulty, 1),
                "most_likely_round_of_32_opponents": top_stage_opponents(
                    code, "round_of_32", teams, result
                ),
                "most_likely_round_of_16_opponents": top_stage_opponents(
                    code, "round_of_16", teams, result
                ),
                "strongest_factors_helping": strongest_factors(
                    model, standing.points, standing.gd
                ),
                "biggest_weaknesses_or_risks": biggest_risks(
                    model, standing.points, standing.gd
                ),
            }
            rows.append(row)
        rows.sort(key=lambda row: row["prob_champion"], reverse=True)
        for row in rows:
            writer.writerow(row)

    group_stage_path = output_dir / "group_stage.csv"
    with group_stage_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "team",
            "group",
            "expected_points",
            "expected_goal_difference",
            "expected_goals_for",
            "expected_goals_against",
            "prob_finish_1st",
            "prob_finish_2nd",
            "prob_finish_3rd",
            "prob_finish_4th",
            "prob_top2",
            "prob_best_third",
            "prob_eliminated_from_3rd",
            "prob_eliminated_from_4th",
            "current_points",
            "current_goal_difference",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rows = []
        for code, team in teams.items():
            finish = {
                place: result.group_finish_counts[code][place] / result.simulations
                for place in (1, 2, 3, 4)
            }
            best_third = result.third_qualifier_counts[code] / result.simulations
            standing = result.current_standings[code]
            rows.append(
                {
                    "team": team.name,
                    "group": team.group,
                    "expected_points": result.group_stat_sums[code]["points"]
                    / result.simulations,
                    "expected_goal_difference": result.group_stat_sums[code][
                        "goal_difference"
                    ]
                    / result.simulations,
                    "expected_goals_for": result.group_stat_sums[code]["goals_for"]
                    / result.simulations,
                    "expected_goals_against": result.group_stat_sums[code][
                        "goals_against"
                    ]
                    / result.simulations,
                    "prob_finish_1st": finish[1],
                    "prob_finish_2nd": finish[2],
                    "prob_finish_3rd": finish[3],
                    "prob_finish_4th": finish[4],
                    "prob_top2": finish[1] + finish[2],
                    "prob_best_third": best_third,
                    "prob_eliminated_from_3rd": max(0.0, finish[3] - best_third),
                    "prob_eliminated_from_4th": finish[4],
                    "current_points": standing.points,
                    "current_goal_difference": standing.gd,
                }
            )
        rows.sort(key=lambda row: (row["group"], -row["expected_points"]))
        writer.writerows(rows)

    diagnostics_path = output_dir / "team_diagnostics.csv"
    diagnostic_keys = sorted(
        {key for model in models.values() for key in model.diagnostics}
    )
    component_keys = sorted({key for model in models.values() for key in model.components})
    with diagnostics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["team", "group"] + diagnostic_keys + [
            f"component_{key}" for key in component_keys
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for code, team in sorted(teams.items(), key=lambda item: item[1].name):
            model = models[code]
            row = {"team": team.name, "group": team.group}
            row.update({key: model.diagnostics.get(key, "") for key in diagnostic_keys})
            row.update(
                {
                    f"component_{key}": model.components.get(key, "")
                    for key in component_keys
                }
            )
            writer.writerow(row)

    matchups_path = output_dir / "knockout_matchups.csv"
    by_pair: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (team, opponent, stage), count in result.matchup_counts.items():
        by_pair.setdefault((team, opponent), {})[stage] = count / result.simulations
    with matchups_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["team", "opponent"] + [
            "round_of_32",
            "round_of_16",
            "quarter_final",
            "semi_final",
            "final",
            "any_knockout",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (team, opponent), stages in sorted(
            by_pair.items(), key=lambda item: (teams[item[0][0]].name, -sum(item[1].values()))
        ):
            row = {
                "team": teams[team].name,
                "opponent": teams[opponent].name,
                "round_of_32": stages.get("round_of_32", 0.0),
                "round_of_16": stages.get("round_of_16", 0.0),
                "quarter_final": stages.get("quarter_final", 0.0),
                "semi_final": stages.get("semi_final", 0.0),
                "final": stages.get("final", 0.0),
                "any_knockout": sum(stages.values()),
            }
            writer.writerow(row)

    route_summary_path = output_dir / "route_summary.csv"
    with route_summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "team",
            "stage",
            "opponent",
            "probability",
            "opponent_effective_rating",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rows = []
        for (team, opponent, stage), count in result.matchup_counts.items():
            if count == 0:
                continue
            rows.append(
                {
                    "team": teams[team].name,
                    "stage": stage,
                    "opponent": teams[opponent].name,
                    "probability": count / result.simulations,
                    "opponent_effective_rating": round(
                        models[opponent].effective_rating, 1
                    ),
                }
            )
        rows.sort(
            key=lambda row: (
                row["team"],
                row["stage"],
                -row["probability"],
                row["opponent"],
            )
        )
        writer.writerows(rows)

    match_probabilities_path = output_dir / "match_probabilities.csv"
    with match_probabilities_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "group",
            "team1",
            "team2",
            "expected_goals_team1",
            "expected_goals_team2",
            "prob_team1_win_90",
            "prob_draw_90",
            "prob_team2_win_90",
            "prob_team1_advance_knockout",
            "prob_team2_advance_knockout",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rows = []
        for group, pairs in all_group_pairs(teams).items():
            for first, second in pairs:
                row = {
                    "group": group,
                    "team1": teams[first].name,
                    "team2": teams[second].name,
                }
                row.update(match_probability_summary(first, second, models))
                rows.append(row)
        rows.sort(key=lambda row: (row["group"], row["team1"], row["team2"]))
        writer.writerows(rows)

    intervals_path = output_dir / "probability_intervals.csv"
    with intervals_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["team", "stage", "probability", "ci_low", "ci_high"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        rows = []
        for code, team in teams.items():
            for stage in STAGES:
                count = result.stage_counts[code][stage]
                low, high = wilson_interval(count, result.simulations)
                rows.append(
                    {
                        "team": team.name,
                        "stage": stage,
                        "probability": count / result.simulations,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        rows.sort(key=lambda row: (row["stage"], -row["probability"], row["team"]))
        writer.writerows(rows)

    importance_path = output_dir / "feature_importance.csv"
    with importance_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature_block", "mean_abs_elo_points", "interpretation"]
        )
        writer.writeheader()
        for key in component_keys:
            values = [abs(model.components.get(key, 0.0)) for model in models.values()]
            interpretation = {
                "elo": "baseline team strength from World Football Elo",
                "recent_form": "last 5/10/20 match form, goal difference, strong-opponent form",
                "host_advantage": "co-host location and support effect",
                "squad_tactical_country": "optional player, tactical, and country-level feature block",
                "injury_suspension": "optional injury and suspension penalty",
            }.get(key, "model feature contribution")
            writer.writerow(
                {
                    "feature_block": key,
                    "mean_abs_elo_points": sum(values) / len(values),
                    "interpretation": interpretation,
                }
            )

    sanity_path = output_dir / "sanity_checks.csv"
    checks = validation_checks(teams, result)
    with sanity_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["check", "status", "value", "expected"]
        )
        writer.writeheader()
        writer.writerows(checks)

    metadata_path = output_dir / "simulation_metadata.json"
    metadata = {
        "simulations": result.simulations,
        "completed_group_results_fixed_through": result.as_of,
        "teams": len(teams),
        "validation_status": "PASS"
        if all(check["status"] == "PASS" for check in checks)
        else "FAIL",
        "stage_slot_counts": STAGE_SLOT_COUNTS,
        "output_files": [
            "probabilities.csv",
            "group_stage.csv",
            "probability_intervals.csv",
            "knockout_matchups.csv",
            "route_summary.csv",
            "match_probabilities.csv",
            "team_diagnostics.csv",
            "feature_importance.csv",
            "sanity_checks.csv",
            "probabilities.md",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    markdown_path = output_dir / "probabilities.md"
    with probabilities_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("# FIFA World Cup 2026 Predictor Output\n\n")
        f.write(f"Simulations: {result.simulations:,}\n\n")
        if result.as_of:
            f.write(f"Completed group results fixed through: {result.as_of}\n\n")
        f.write("| Team | Group | R32 | R16 | QF | SF | Final | Win |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {team} | {group} | {r32:.1%} | {r16:.1%} | {qf:.1%} | "
                "{sf:.1%} | {final:.1%} | {win:.1%} |\n".format(
                    team=row["team"],
                    group=row["group"],
                    r32=float(row["prob_round_of_32"]),
                    r16=float(row["prob_round_of_16"]),
                    qf=float(row["prob_quarter_final"]),
                    sf=float(row["prob_semi_final"]),
                    final=float(row["prob_final"]),
                    win=float(row["prob_champion"]),
                )
            )
