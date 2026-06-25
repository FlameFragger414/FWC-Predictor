from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import Team, all_group_pairs
from .features import TeamModel
from .simulation import STAGES, SimulationResult, match_probability_summary, stage_probability
from .validation import DataQualityWarning

STAGE_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarter_final": "Quarter-final",
    "semi_final": "Semi-final",
    "final": "Final",
    "champion": "Champion",
}


def probabilities_dataframe(
    teams: Dict[str, Team],
    models: Dict[str, TeamModel],
    result: SimulationResult,
) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    for code, team in teams.items():
        row: dict[str, object] = {
            "code": code,
            "team": team.name,
            "group": team.group,
            "elo": models[code].elo,
            "effective_rating": models[code].effective_rating,
            "attack_rating": models[code].attack_rating,
            "defense_rating": models[code].defense_rating,
            "penalty_rating": models[code].penalty_rating,
            "uncertainty": models[code].uncertainty,
            "recent_form": models[code].components.get("recent_form", 0.0),
        }
        for stage in STAGES:
            row[f"prob_{stage}"] = stage_probability(result, code, stage)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("prob_champion", ascending=False)


def group_dataframe(
    teams: Dict[str, Team],
    result: SimulationResult,
) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    for code, team in teams.items():
        finish = {
            place: result.group_finish_counts[code][place] / result.simulations
            for place in (1, 2, 3, 4)
        }
        rows.append(
            {
                "code": code,
                "team": team.name,
                "group": team.group,
                "expected_points": result.group_stat_sums[code]["points"]
                / result.simulations,
                "expected_goal_difference": result.group_stat_sums[code]["goal_difference"]
                / result.simulations,
                "prob_top2": finish[1] + finish[2],
                "prob_best_third": result.third_qualifier_counts[code] / result.simulations,
                "prob_finish_1st": finish[1],
                "prob_finish_2nd": finish[2],
                "prob_finish_3rd": finish[3],
                "prob_finish_4th": finish[4],
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "expected_points"], ascending=[True, False])


def matchup_dataframe(
    teams: Dict[str, Team],
    result: SimulationResult,
) -> pd.DataFrame:
    rows = [
        {
            "team": teams[team].name,
            "opponent": teams[opponent].name,
            "stage": STAGE_LABELS.get(stage, stage),
            "probability": count / result.simulations,
        }
        for (team, opponent, stage), count in result.matchup_counts.items()
    ]
    return pd.DataFrame(rows)


def match_probabilities_dataframe(
    teams: Dict[str, Team],
    models: Dict[str, TeamModel],
) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    for group, pairs in all_group_pairs(teams).items():
        for first, second in pairs:
            summary = match_probability_summary(first, second, models)
            rows.append(
                {
                    "group": group,
                    "team1": teams[first].name,
                    "team2": teams[second].name,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def warnings_dataframe(warnings: Iterable[DataQualityWarning]) -> pd.DataFrame:
    return pd.DataFrame([warning.__dict__ for warning in warnings])


def title_probability_chart(probabilities: pd.DataFrame, limit: int = 16) -> go.Figure:
    data = probabilities.head(limit).sort_values("prob_champion")
    fig = px.bar(
        data,
        x="prob_champion",
        y="team",
        color="group",
        orientation="h",
        title="Title probability leaders",
        labels={"prob_champion": "Title probability", "team": ""},
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def stage_advancement_chart(probabilities: pd.DataFrame, team_names: List[str]) -> go.Figure:
    data = probabilities[probabilities["team"].isin(team_names)]
    long = data.melt(
        id_vars=["team"],
        value_vars=[f"prob_{stage}" for stage in STAGES],
        var_name="stage",
        value_name="probability",
    )
    long["stage"] = long["stage"].str.replace("prob_", "", regex=False).map(STAGE_LABELS)
    fig = px.bar(
        long,
        x="stage",
        y="probability",
        color="team",
        barmode="group",
        title="Stage advancement probabilities",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def group_qualification_chart(groups: pd.DataFrame, group: str) -> go.Figure:
    data = groups[groups["group"] == group]
    fig = px.bar(
        data,
        x="team",
        y=["prob_top2", "prob_best_third"],
        title=f"Group {group} qualification routes",
        labels={"value": "Probability", "variable": "Route"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def group_finish_distribution_chart(groups: pd.DataFrame, group: str) -> go.Figure:
    data = groups[groups["group"] == group]
    long = data.melt(
        id_vars=["team"],
        value_vars=[
            "prob_finish_1st",
            "prob_finish_2nd",
            "prob_finish_3rd",
            "prob_finish_4th",
        ],
        var_name="finish",
        value_name="probability",
    )
    long["finish"] = long["finish"].str.extract(r"(\d)").astype(str) + " place"
    fig = px.bar(
        long,
        x="team",
        y="probability",
        color="finish",
        title=f"Group {group} finishing-position distribution",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def expected_goals_chart(matches: pd.DataFrame, team_name: str) -> go.Figure:
    data = matches[(matches["team1"] == team_name) | (matches["team2"] == team_name)].copy()
    data["opponent"] = np.where(data["team1"] == team_name, data["team2"], data["team1"])
    data["xg_for"] = np.where(
        data["team1"] == team_name,
        data["expected_goals_team1"],
        data["expected_goals_team2"],
    )
    data["xg_against"] = np.where(
        data["team1"] == team_name,
        data["expected_goals_team2"],
        data["expected_goals_team1"],
    )
    fig = px.bar(
        data,
        x="opponent",
        y=["xg_for", "xg_against"],
        barmode="group",
        title=f"Expected goals profile: {team_name}",
    )
    return fig


def team_strength_comparison_chart(probabilities: pd.DataFrame, teams: List[str]) -> go.Figure:
    data = probabilities[probabilities["team"].isin(teams)]
    categories = ["effective_rating", "attack_rating", "defense_rating", "penalty_rating"]
    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[column] for column in categories],
                theta=["Overall", "Attack", "Defence", "Penalties"],
                fill="toself",
                name=str(row["team"]),
            )
        )
    fig.update_layout(title="Team strength comparison", polar={"radialaxis": {"visible": True}})
    return fig


def uncertainty_interval_chart(probabilities: pd.DataFrame, limit: int = 16) -> go.Figure:
    data = probabilities.head(limit)
    z = 1.96
    n = max(1, len(probabilities))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["team"],
            y=data["prob_champion"],
            error_y={
                "type": "data",
                "array": z
                * np.sqrt(data["prob_champion"] * (1 - data["prob_champion"]) / n),
                "visible": True,
            },
            mode="markers",
            name="Title probability",
        )
    )
    fig.update_layout(title="Title uncertainty intervals", yaxis_tickformat=".0%")
    return fig


def knockout_path_heatmap(matchups: pd.DataFrame, team_name: str) -> go.Figure:
    data = matchups[matchups["team"] == team_name]
    pivot = data.pivot_table(
        values="probability",
        index="opponent",
        columns="stage",
        aggfunc="sum",
        fill_value=0.0,
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"Likely knockout opponents: {team_name}",
        labels={"color": "Probability"},
    )
    return fig


def probability_heatmap(probabilities: pd.DataFrame) -> go.Figure:
    stage_columns = [f"prob_{stage}" for stage in STAGES]
    data = probabilities.set_index("team")[stage_columns]
    data.columns = [STAGE_LABELS[stage] for stage in STAGES]
    fig = px.imshow(
        data,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Tournament stage probability heatmap",
        labels={"color": "Probability"},
    )
    return fig


def form_rating_trend_chart(probabilities: pd.DataFrame, teams: List[str]) -> go.Figure:
    data = probabilities[probabilities["team"].isin(teams)]
    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=["Elo", "Recent form adjusted", "Effective"],
                y=[
                    row["elo"],
                    row["elo"] + row["recent_form"],
                    row["effective_rating"],
                ],
                mode="lines+markers",
                name=str(row["team"]),
            )
        )
    fig.update_layout(title="Rating and form trend")
    return fig


def strength_title_3d_scatter(probabilities: pd.DataFrame) -> go.Figure:
    fig = px.scatter_3d(
        probabilities,
        x="effective_rating",
        y="recent_form",
        z="prob_champion",
        color="group",
        hover_name="team",
        size="prob_round_of_32",
        title="3D strength, form, and title probability",
    )
    fig.update_layout(scene={"zaxis": {"tickformat": ".0%"}})
    return fig


def attack_defense_progress_3d(probabilities: pd.DataFrame) -> go.Figure:
    fig = px.scatter_3d(
        probabilities,
        x="attack_rating",
        y="defense_rating",
        z="prob_quarter_final",
        color="prob_champion",
        hover_name="team",
        title="3D attack, defence, and tournament progress",
        color_continuous_scale="Plasma",
    )
    fig.update_layout(scene={"zaxis": {"tickformat": ".0%"}})
    return fig


def probability_surface_3d() -> go.Figure:
    rating_diffs = np.linspace(-350, 350, 50)
    penalty_diffs = np.linspace(-160, 160, 50)
    x, y = np.meshgrid(rating_diffs, penalty_diffs)
    regulation_edge = 1 / (1 + np.exp(-x / 210))
    penalty_edge = 1 / (1 + np.exp(-y / 280))
    z = 0.78 * regulation_edge + 0.22 * penalty_edge
    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Viridis",
                hovertemplate=(
                    "Rating diff %{x:.0f}<br>Penalty diff %{y:.0f}<br>"
                    "Advance %{z:.1%}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="3D knockout advancement surface",
        scene={
            "xaxis_title": "Rating difference",
            "yaxis_title": "Penalty-rating difference",
            "zaxis_title": "Advance probability",
        },
    )
    return fig


def team_cluster_3d(probabilities: pd.DataFrame) -> go.Figure:
    data = probabilities.copy()
    data["progress_index"] = (
        data["prob_round_of_16"]
        + data["prob_quarter_final"]
        + data["prob_semi_final"]
        + data["prob_final"]
        + data["prob_champion"]
    )
    fig = px.scatter_3d(
        data,
        x="attack_rating",
        y="defense_rating",
        z="progress_index",
        color="group",
        hover_name="team",
        symbol="group",
        title="3D team clusters by attack, defence, and progress",
    )
    return fig
