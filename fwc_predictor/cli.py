from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .data import project_root
from .gui import launch_streamlit
from .pipeline import run_simulation_pipeline


def _add_simulation_options(parser: argparse.ArgumentParser) -> None:
    root = project_root()
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the FWC Predictor GUI or run tournament simulations."
    )
    subparsers = parser.add_subparsers(dest="command")
    gui_parser = subparsers.add_parser("gui", help="launch the graphical interface")
    gui_parser.set_defaults(command="gui")
    simulate_parser = subparsers.add_parser("simulate", help="run CLI simulation outputs")
    _add_simulation_options(simulate_parser)
    simulate_parser.set_defaults(command="simulate")

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return argparse.Namespace(command="gui")
    if argv[0].startswith("-"):
        legacy_parser = argparse.ArgumentParser(
            description="Run FIFA World Cup Monte Carlo tournament simulations."
        )
        _add_simulation_options(legacy_parser)
        args = legacy_parser.parse_args(argv)
        args.command = "simulate"
        return args
    return parser.parse_args(argv)


def run_cli_simulation(args: argparse.Namespace) -> None:
    pipeline = run_simulation_pipeline(
        simulations=args.sims,
        seed=args.seed,
        groups_path=args.groups,
        features_path=args.features,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        refresh=args.refresh,
    )
    print(f"Ran {args.sims:,} simulations.")
    if pipeline.result.as_of:
        print(f"Fixed completed World Cup group results through {pipeline.result.as_of}.")
    print(f"Model runtime: {pipeline.elapsed_seconds:.2f}s.")
    for warning in pipeline.warnings:
        print(f"{warning.severity.upper()}: {warning.area}: {warning.message}")
    print(f"Wrote outputs to {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "gui":
        launch_streamlit()
    elif args.command == "simulate":
        run_cli_simulation(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

