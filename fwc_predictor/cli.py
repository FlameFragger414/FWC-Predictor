from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    load_feature_overrides,
    load_groups,
    load_latest_results,
    load_third_place_assignments,
    load_world_ratings,
    project_root,
    world_cup_results,
)
from .features import build_team_models
from .simulation import simulate_tournament, write_outputs


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run FIFA World Cup Monte Carlo tournament simulations."
    )
    parser.add_argument("--sims", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--refresh", action="store_true", help="refresh cached web data")
    parser.add_argument("--groups", type=Path, default=root / "data" / "groups_2026.csv")
    parser.add_argument(
        "--features",
        type=Path,
        default=root / "data" / "team_feature_overrides.csv",
    )
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teams = load_groups(args.groups)
    feature_overrides = load_feature_overrides(args.features)
    ratings = load_world_ratings(args.cache_dir, refresh=args.refresh)
    latest_results = load_latest_results(args.cache_dir, refresh=args.refresh)
    completed_wc = world_cup_results(latest_results, teams)
    third_assignments = load_third_place_assignments(
        args.cache_dir, refresh=args.refresh
    )
    models = build_team_models(teams, ratings, latest_results, feature_overrides)
    result = simulate_tournament(
        teams=teams,
        models=models,
        completed_matches=completed_wc,
        third_assignments=third_assignments,
        simulations=args.sims,
        seed=args.seed,
    )
    write_outputs(args.output_dir, teams, models, result)
    print(f"Ran {args.sims:,} simulations.")
    if result.as_of:
        print(f"Fixed completed World Cup group results through {result.as_of}.")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

