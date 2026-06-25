from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path

from fwc_predictor.data import Team, load_groups, project_root
from fwc_predictor.features import TeamModel
from fwc_predictor.simulation import (
    STAGE_SLOT_COUNTS,
    MatchRecord,
    Standing,
    match_probability_summary,
    round_of_32_matches,
    simulate_tournament,
    sorted_group,
    validation_checks,
    wilson_interval,
    write_outputs,
)


def make_model(team: Team, index: int) -> TeamModel:
    rating = 1700.0 + index * 4.0
    return TeamModel(
        code=team.code,
        name=team.name,
        group=team.group,
        slot=team.slot,
        elo=rating,
        fifa_rank=None,
        effective_rating=rating,
        attack_rating=rating,
        defense_rating=rating,
        penalty_rating=rating,
        uncertainty=20.0,
        components={"elo": rating - 1800.0},
        diagnostics={
            "elo": rating,
            "effective_rating": rating,
            "attack_rating": rating,
            "defense_rating": rating,
            "penalty_rating": rating,
            "rating_uncertainty": 20.0,
            "known_optional_features": 0.0,
        },
    )


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.teams = load_groups(project_root() / "data" / "groups_2026.csv")
        self.models = {
            code: make_model(team, idx)
            for idx, (code, team) in enumerate(sorted(self.teams.items()))
        }

    def test_wilson_interval_contains_observed_probability(self) -> None:
        low, high = wilson_interval(23, 100)
        self.assertLessEqual(low, 0.23)
        self.assertGreaterEqual(high, 0.23)

    def test_round_of_32_has_16_matches_and_32_unique_teams(self) -> None:
        placements = {
            group: [f"{group}{place}" for place in range(1, 5)]
            for group in "ABCDEFGHIJKL"
        }
        third_qualifiers = {group: f"{group}3" for group in "ABCDEFGH"}
        third_assignments = {
            "ABCDEFGH": {
                "A": "A",
                "B": "B",
                "D": "C",
                "E": "D",
                "G": "E",
                "I": "F",
                "K": "G",
                "L": "H",
            }
        }
        matches = round_of_32_matches(
            placements, third_qualifiers, third_assignments
        )
        entrants = [team for match in matches.values() for team in match]
        self.assertEqual(len(matches), 16)
        self.assertEqual(len(entrants), 32)
        self.assertEqual(len(set(entrants)), 32)

    def test_simulation_invariants_hold(self) -> None:
        result = simulate_tournament(
            teams=self.teams,
            models=self.models,
            completed_matches=[],
            third_assignments={},
            simulations=250,
            seed=99,
        )
        checks = validation_checks(self.teams, result)
        failing = [check for check in checks if check["status"] != "PASS"]
        self.assertEqual(failing, [])

        for stage, slots in STAGE_SLOT_COUNTS.items():
            total = sum(
                result.stage_counts[code][stage] for code in self.teams
            )
            self.assertEqual(total, slots * result.simulations)

    def test_seeded_simulation_is_deterministic(self) -> None:
        first = simulate_tournament(
            teams=self.teams,
            models=self.models,
            completed_matches=[],
            third_assignments={},
            simulations=120,
            seed=123,
        )
        second = simulate_tournament(
            teams=self.teams,
            models=self.models,
            completed_matches=[],
            third_assignments={},
            simulations=120,
            seed=123,
        )
        self.assertEqual(first.stage_counts, second.stage_counts)
        self.assertEqual(first.group_finish_counts, second.group_finish_counts)

    def test_group_tiebreak_uses_head_to_head_before_lots(self) -> None:
        codes = ["A", "B", "C", "D"]
        standings = {
            code: Standing(code=code, played=3, points=6, gf=4, ga=2)
            for code in codes
        }
        records = [
            MatchRecord("X", "A", "B", 2, 0),
            MatchRecord("X", "A", "C", 0, 1),
            MatchRecord("X", "B", "C", 3, 0),
        ]
        ordered = sorted_group(codes, standings, records, random.Random(2))
        self.assertEqual(ordered[:3], ["B", "A", "C"])

    def test_equal_strength_match_is_symmetric(self) -> None:
        team_a = Team("A", "A1", "AA", "Alpha")
        team_b = Team("A", "A2", "BB", "Beta")
        model_a = make_model(team_a, 0)
        model_b = make_model(team_b, 0)
        models = {"AA": model_a, "BB": model_b}
        summary = match_probability_summary("AA", "BB", models)
        self.assertAlmostEqual(
            summary["prob_team1_win_90"],
            summary["prob_team2_win_90"],
            places=12,
        )
        self.assertAlmostEqual(summary["prob_team1_advance_knockout"], 0.5, places=12)

    def test_extreme_ratings_stay_bounded(self) -> None:
        team_a = Team("A", "A1", "AA", "Alpha")
        team_b = Team("A", "A2", "BB", "Beta")
        model_a = make_model(team_a, 0)
        model_b = make_model(team_b, 0)
        model_a.effective_rating = 2600
        model_a.attack_rating = 2700
        model_b.effective_rating = 900
        model_b.defense_rating = 850
        summary = match_probability_summary("AA", "BB", {"AA": model_a, "BB": model_b})
        self.assertGreaterEqual(summary["expected_goals_team1"], 0.08)
        self.assertLessEqual(summary["expected_goals_team1"], 5.5)
        self.assertGreaterEqual(summary["prob_team1_advance_knockout"], 0.0)
        self.assertLessEqual(summary["prob_team1_advance_knockout"], 1.0)

    def test_output_files_are_written_and_probability_sum_is_one(self) -> None:
        result = simulate_tournament(
            teams=self.teams,
            models=self.models,
            completed_matches=[],
            third_assignments={},
            simulations=200,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_outputs(output_dir, self.teams, self.models, result)
            required = [
                "probabilities.csv",
                "group_stage.csv",
                "probability_intervals.csv",
                "knockout_matchups.csv",
                "route_summary.csv",
                "match_probabilities.csv",
                "team_diagnostics.csv",
                "feature_importance.csv",
                "sanity_checks.csv",
                "simulation_metadata.json",
            ]
            for name in required:
                self.assertTrue((output_dir / name).exists(), name)

            with (output_dir / "probabilities.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            champion_sum = sum(float(row["prob_champion"]) for row in rows)
            self.assertAlmostEqual(champion_sum, 1.0)

            with (output_dir / "sanity_checks.csv").open(encoding="utf-8") as f:
                sanity_rows = list(csv.DictReader(f))
            self.assertTrue(sanity_rows)
            self.assertTrue(all(row["status"] == "PASS" for row in sanity_rows))

            with (output_dir / "match_probabilities.csv").open(encoding="utf-8") as f:
                match_rows = list(csv.DictReader(f))
            self.assertEqual(len(match_rows), 72)
            for row in match_rows:
                probability_sum = (
                    float(row["prob_team1_win_90"])
                    + float(row["prob_draw_90"])
                    + float(row["prob_team2_win_90"])
                )
                self.assertAlmostEqual(probability_sum, 1.0)


if __name__ == "__main__":
    unittest.main()
