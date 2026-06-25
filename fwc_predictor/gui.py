from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from .charts import (
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
    warnings_dataframe,
)
from .data import project_root
from .pipeline import SimulationPipelineResult, run_simulation_pipeline
from .simulation import match_probability_summary

ACCENT = "#28d6a3"


@st.cache_data(show_spinner=False)
def cached_simulation(
    simulations: int,
    seed: int,
    refresh: bool,
    scenario_items: tuple[tuple[str, float], ...],
) -> SimulationPipelineResult:
    return run_simulation_pipeline(
        simulations=simulations,
        seed=seed,
        refresh=refresh,
        scenario_adjustments=dict(scenario_items),
        write_files=False,
    )


def launch_streamlit() -> None:
    from streamlit.web import cli as stcli

    app_path = Path(__file__).with_name("gui.py")
    sys.argv = ["streamlit", "run", str(app_path), "--server.headless=true"]
    raise SystemExit(stcli.main())


def _style(dark_mode: bool) -> None:
    background = "#0e1117" if dark_mode else "#f7f9fc"
    card = "#161b22" if dark_mode else "#ffffff"
    text = "#f4f7fb" if dark_mode else "#17202a"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {background}; color: {text}; }}
        div[data-testid="stMetric"] {{
            background: {card};
            border: 1px solid rgba(127,127,127,.18);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 12px 32px rgba(0,0,0,.08);
        }}
        .hero {{
            padding: 28px 30px;
            border-radius: 24px;
            background: linear-gradient(135deg, #10233f 0%, #0b7b74 55%, {ACCENT} 100%);
            color: white;
            margin-bottom: 18px;
        }}
        .hero h1 {{ margin-bottom: 6px; }}
        .warning-card {{
            border-radius: 16px;
            padding: 12px 14px;
            background: {card};
            border-left: 5px solid {ACCENT};
            margin-bottom: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _download_dataframe(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _metric_row(probabilities: pd.DataFrame, elapsed_seconds: float, simulations: int) -> None:
    leader = probabilities.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Title favorite", leader["team"], f"{leader['prob_champion']:.1%}")
    col2.metric("Simulation count", f"{simulations:,}")
    col3.metric("Model runtime", f"{elapsed_seconds:.2f}s")
    col4.metric("Teams", f"{len(probabilities)}")


def app() -> None:
    st.set_page_config(
        page_title="FWC Predictor",
        page_icon="FWC",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.header("Simulation settings")
    simulations = st.sidebar.slider(
        "Monte Carlo simulations",
        min_value=250,
        max_value=50_000,
        value=2_000,
        step=250,
        help="Higher values reduce Monte Carlo noise but take longer.",
    )
    seed = st.sidebar.number_input("Random seed", value=20260622, step=1)
    refresh = st.sidebar.toggle("Refresh live data", value=False)
    dark_mode = st.sidebar.toggle("Dark mode", value=True)
    _style(dark_mode)

    root = project_root()
    teams_preview = pd.read_csv(root / "data" / "groups_2026.csv")
    scenario_team = st.sidebar.selectbox(
        "Scenario team",
        ["No adjustment"] + teams_preview["team"].tolist(),
        help="Apply a temporary Elo-point adjustment without editing data files.",
    )
    scenario_delta = st.sidebar.slider("Scenario adjustment", -120, 120, 0, 5)
    name_to_code = dict(zip(teams_preview["team"], teams_preview["elo_code"], strict=True))
    scenario: Dict[str, float] = {}
    if scenario_team != "No adjustment" and scenario_delta:
        scenario[name_to_code[scenario_team]] = float(scenario_delta)
    run = st.sidebar.button("Run new simulation", type="primary", use_container_width=True)
    if run:
        cached_simulation.clear()

    st.markdown(
        """
        <div class="hero">
          <h1>FWC Predictor</h1>
          <p>Interactive FIFA World Cup probabilities, knockout paths, and model diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Running tournament simulations and validating outputs..."):
        pipeline = cached_simulation(
            simulations,
            int(seed),
            refresh,
            tuple(sorted(scenario.items())),
        )

    probabilities = probabilities_dataframe(pipeline.teams, pipeline.models, pipeline.result)
    groups = group_dataframe(pipeline.teams, pipeline.result)
    matchups = matchup_dataframe(pipeline.teams, pipeline.result)
    matches = match_probabilities_dataframe(pipeline.teams, pipeline.models)
    team_names = probabilities["team"].tolist()
    default_compare = team_names[:2]

    _metric_row(probabilities, pipeline.elapsed_seconds, simulations)

    tabs = st.tabs(
        [
            "Dashboard",
            "Team rankings",
            "Group predictions",
            "Knockout bracket",
            "Match predictor",
            "Team comparison",
            "Simulation settings",
            "Model explanation",
            "Data quality and warnings",
        ]
    )

    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        col1.plotly_chart(title_probability_chart(probabilities), use_container_width=True)
        col2.plotly_chart(uncertainty_interval_chart(probabilities), use_container_width=True)
        st.plotly_chart(strength_title_3d_scatter(probabilities), use_container_width=True)
        st.plotly_chart(probability_heatmap(probabilities), use_container_width=True)

    with tabs[1]:
        selected = st.multiselect("Teams to inspect", team_names, default=team_names[:6])
        st.plotly_chart(stage_advancement_chart(probabilities, selected), use_container_width=True)
        st.plotly_chart(
            form_rating_trend_chart(probabilities, selected[:5]),
            use_container_width=True,
        )
        _download_dataframe("Export team probabilities", probabilities, "fwc_probabilities.csv")

    with tabs[2]:
        group = st.selectbox("Group", sorted(groups["group"].unique()))
        col1, col2 = st.columns(2)
        col1.plotly_chart(group_qualification_chart(groups, group), use_container_width=True)
        col2.plotly_chart(group_finish_distribution_chart(groups, group), use_container_width=True)
        _download_dataframe("Export group projections", groups, "fwc_group_predictions.csv")

    with tabs[3]:
        team = st.selectbox("Team path", team_names, index=0)
        st.plotly_chart(knockout_path_heatmap(matchups, team), use_container_width=True)
        st.plotly_chart(attack_defense_progress_3d(probabilities), use_container_width=True)
        _download_dataframe("Export likely knockout opponents", matchups, "fwc_knockout_paths.csv")

    with tabs[4]:
        col1, col2 = st.columns(2)
        first = col1.selectbox("Team A", team_names, index=0)
        second = col2.selectbox("Team B", team_names, index=1)
        code_by_name = {team.name: code for code, team in pipeline.teams.items()}
        if first == second:
            st.info("Pick two different teams to simulate a match.")
        else:
            summary = match_probability_summary(
                code_by_name[first], code_by_name[second], pipeline.models
            )
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{first} win in 90", f"{summary['prob_team1_win_90']:.1%}")
            m2.metric("Draw in 90", f"{summary['prob_draw_90']:.1%}")
            m3.metric(f"{second} win in 90", f"{summary['prob_team2_win_90']:.1%}")
            st.plotly_chart(expected_goals_chart(matches, first), use_container_width=True)
            st.plotly_chart(probability_surface_3d(), use_container_width=True)

    with tabs[5]:
        compare = st.multiselect("Compare teams", team_names, default=default_compare)
        if compare:
            st.plotly_chart(
                team_strength_comparison_chart(probabilities, compare),
                use_container_width=True,
            )
            st.plotly_chart(team_cluster_3d(probabilities), use_container_width=True)
        else:
            st.info("Select at least one team to compare strengths.")

    with tabs[6]:
        st.subheader("Current run")
        st.write(
            {
                "simulations": simulations,
                "seed": int(seed),
                "refresh": refresh,
                "scenario": scenario,
                "completed_matches_locked": len(pipeline.completed_matches),
                "used_snapshot_fallback": pipeline.used_snapshot_fallback,
            }
        )
        st.caption(
            "Use the sidebar to change simulation count, seed, refresh behavior, "
            "and scenario adjustments."
        )

    with tabs[7]:
        st.markdown(
            """
            ### Model summary
            The model starts from World Football Elo, adds shrunk recent-form signals,
            applies host-region and optional squad/tactical/country adjustments, samples
            latent uncertainty, then simulates scores with a Poisson expected-goals model.

            Knockouts simulate regulation, extra time, and penalties. Group ordering uses
            points, goal difference, goals scored, head-to-head criteria for tied teams,
            and drawing of lots when unresolved. Predictions are probabilistic and not
            guaranteed outcomes.
            """
        )

    with tabs[8]:
        warning_rows = warnings_dataframe(pipeline.warnings)
        if warning_rows.empty:
            st.success("No data quality warnings for this run.")
        else:
            for warning in pipeline.warnings:
                st.markdown(
                    (
                        f"<div class='warning-card'><b>{warning.severity.upper()} "
                        f"&middot; {warning.area}</b><br>{warning.message}</div>"
                    ),
                    unsafe_allow_html=True,
                )
            _download_dataframe("Export warnings", warning_rows, "fwc_data_warnings.csv")


if __name__ == "__main__":
    app()
