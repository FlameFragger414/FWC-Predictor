from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import DEFAULT_MODEL_PARAMETERS, ModelParameters
from .data import (
    MatchResult,
    Team,
    load_feature_overrides,
    load_groups,
    load_latest_results,
    load_third_place_assignments,
    load_world_ratings,
    project_root,
    world_cup_results,
)
from .features import TeamModel, build_team_models
from .simulation import SimulationResult, simulate_tournament, write_outputs
from .validation import (
    DataQualityWarning,
    build_data_quality_warnings,
    validate_completed_matches,
    validate_feature_file,
    validate_groups,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationPipelineResult:
    teams: Dict[str, Team]
    models: Dict[str, TeamModel]
    result: SimulationResult
    warnings: List[DataQualityWarning]
    elapsed_seconds: float
    used_snapshot_fallback: bool
    output_dir: Optional[Path] = None
    completed_matches: List[MatchResult] = field(default_factory=list)


def load_snapshot_ratings(root: Path) -> Dict[str, Dict[str, float]]:
    path = root / "outputs" / "team_diagnostics.csv"
    if not path.exists():
        return {}
    ratings: Dict[str, Dict[str, float]] = {}
    name_to_code = {
        team.name: code for code, team in load_groups(root / "data" / "groups_2026.csv").items()
    }
    with path.open(newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f), start=1):
            code = name_to_code.get(row.get("team", ""))
            if not code:
                continue
            try:
                ratings[code] = {
                    "global_rank": float(index),
                    "elo": float(row["elo"]),
                }
            except (KeyError, ValueError):
                continue
    return ratings


def load_model_inputs(
    groups_path: Path,
    features_path: Path,
    cache_dir: Path,
    refresh: bool = False,
    allow_snapshot_fallback: bool = True,
) -> tuple[
    Dict[str, Team],
    Dict[str, Dict[str, float | None]],
    Dict[str, Dict[str, float]],
    List[MatchResult],
    Dict[str, Dict[str, str]],
    bool,
    List[DataQualityWarning],
]:
    teams = load_groups(groups_path)
    validate_groups(teams)
    validate_feature_file(features_path, teams)
    feature_overrides = load_feature_overrides(features_path)
    root = project_root()
    used_snapshot_fallback = False
    warnings: List[DataQualityWarning] = []

    try:
        ratings = load_world_ratings(cache_dir, refresh=refresh)
    except Exception as exc:  # pragma: no cover - network behavior is environment-specific
        if not allow_snapshot_fallback:
            raise
        LOGGER.warning("Falling back to checked-in ratings snapshot: %s", exc)
        ratings = load_snapshot_ratings(root)
        used_snapshot_fallback = True

    try:
        latest_results = load_latest_results(cache_dir, refresh=refresh)
    except Exception as exc:  # pragma: no cover - network behavior is environment-specific
        if not allow_snapshot_fallback:
            raise
        LOGGER.warning("Recent results unavailable; using neutral form fallback: %s", exc)
        latest_results = []
        used_snapshot_fallback = True

    try:
        third_assignments = load_third_place_assignments(cache_dir, refresh=refresh)
    except Exception as exc:  # pragma: no cover - network behavior is environment-specific
        LOGGER.warning("Third-place assignment table unavailable; using built-in fallback: %s", exc)
        third_assignments = {}
        warnings.append(
            DataQualityWarning(
                "warning",
                "bracket",
                "Official third-place assignment table was unavailable; "
                "deterministic fallback mapping is active.",
            )
        )

    warnings.extend(
        build_data_quality_warnings(
            teams,
            ratings,
            latest_results,
            feature_overrides,
            used_snapshot_fallback,
        )
    )
    return (
        teams,
        feature_overrides,
        ratings,
        latest_results,
        third_assignments,
        used_snapshot_fallback,
        warnings,
    )


def run_simulation_pipeline(
    simulations: int,
    seed: int,
    groups_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    refresh: bool = False,
    parameters: ModelParameters = DEFAULT_MODEL_PARAMETERS,
    scenario_adjustments: Optional[Dict[str, float]] = None,
    write_files: bool = True,
) -> SimulationPipelineResult:
    root = project_root()
    groups_path = groups_path or root / "data" / "groups_2026.csv"
    features_path = features_path or root / "data" / "team_feature_overrides.csv"
    cache_dir = cache_dir or root / ".cache"
    output_dir = output_dir or root / "outputs"
    started = time.perf_counter()
    (
        teams,
        feature_overrides,
        ratings,
        latest_results,
        third_assignments,
        used_snapshot_fallback,
        warnings,
    ) = load_model_inputs(groups_path, features_path, cache_dir, refresh)
    completed_wc = validate_completed_matches(world_cup_results(latest_results, teams), teams)
    models = build_team_models(
        teams,
        ratings,
        latest_results,
        feature_overrides,
        parameters=parameters,
        scenario_adjustments=scenario_adjustments,
    )
    result = simulate_tournament(
        teams=teams,
        models=models,
        completed_matches=completed_wc,
        third_assignments=third_assignments,
        simulations=simulations,
        seed=seed,
        parameters=parameters,
    )
    if write_files:
        write_outputs(output_dir, teams, models, result)
        metadata_path = output_dir / "simulation_metadata.json"
        with (output_dir / "data_quality_warnings.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["severity", "area", "message"])
            writer.writeheader()
            writer.writerows([warning.__dict__ for warning in warnings])
        (output_dir / "performance_metrics.json").write_text(
            json.dumps(
                {
                    "simulations": simulations,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "simulations_per_second": round(
                        simulations / max(time.perf_counter() - started, 1e-9), 2
                    ),
                    "used_snapshot_fallback": used_snapshot_fallback,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["output_files"] = sorted(
                set(metadata.get("output_files", []))
                | {"data_quality_warnings.csv", "performance_metrics.json"}
            )
            metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    return SimulationPipelineResult(
        teams=teams,
        models=models,
        result=result,
        warnings=warnings,
        elapsed_seconds=elapsed,
        used_snapshot_fallback=used_snapshot_fallback,
        output_dir=output_dir if write_files else None,
        completed_matches=completed_wc,
    )
