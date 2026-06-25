from __future__ import annotations

import unittest

from fwc_predictor.charts import (
    attack_defense_progress_3d,
    expected_goals_chart,
    form_rating_trend_chart,
    group_dataframe,
    group_finish_distribution_chart,
    group_qualification_chart,
    knockout_path_heatmap,
    match_probabilities_dataframe,
    matchup_dataframe,
    probabilities_dataframe,
    probability_heatmap,
    probability_surface_3d,
    stage_advancement_chart,
    strength_title_3d_scatter,
    team_cluster_3d,
    team_strength_comparison_chart,
    title_probability_chart,
    uncertainty_interval_chart,
)
from fwc_predictor.data import load_groups, project_root
from fwc_predictor.simulation import simulate_tournament
from tests.test_simulation import make_model


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.teams = load_groups(project_root() / "data" / "groups_2026.csv")
        cls.models = {
            code: make_model(team, idx)
            for idx, (code, team) in enumerate(sorted(cls.teams.items()))
        }
        cls.result = simulate_tournament(
            teams=cls.teams,
            models=cls.models,
            completed_matches=[],
            third_assignments={},
            simulations=80,
            seed=5,
        )
        cls.probabilities = probabilities_dataframe(cls.teams, cls.models, cls.result)
        cls.groups = group_dataframe(cls.teams, cls.result)
        cls.matchups = matchup_dataframe(cls.teams, cls.result)
        cls.matches = match_probabilities_dataframe(cls.teams, cls.models)

    def assert_has_traces(self, figure: object) -> None:
        self.assertGreater(len(figure.data), 0)

    def test_all_dashboard_charts_receive_valid_data(self) -> None:
        team = str(self.probabilities.iloc[0]["team"])
        group = str(self.groups.iloc[0]["group"])
        selected = self.probabilities["team"].head(3).tolist()
        charts = [
            title_probability_chart(self.probabilities),
            uncertainty_interval_chart(self.probabilities),
            stage_advancement_chart(self.probabilities, selected),
            form_rating_trend_chart(self.probabilities, selected),
            group_qualification_chart(self.groups, group),
            group_finish_distribution_chart(self.groups, group),
            knockout_path_heatmap(self.matchups, team),
            expected_goals_chart(self.matches, team),
            team_strength_comparison_chart(self.probabilities, selected),
            probability_heatmap(self.probabilities),
            strength_title_3d_scatter(self.probabilities),
            attack_defense_progress_3d(self.probabilities),
            probability_surface_3d(),
            team_cluster_3d(self.probabilities),
        ]
        for chart in charts:
            self.assert_has_traces(chart)


if __name__ == "__main__":
    unittest.main()
