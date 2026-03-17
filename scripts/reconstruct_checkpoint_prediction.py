#!/usr/bin/env python3
"""Reconstruct one missed checkpoint prediction from a stored session snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime_dependencies():
    """Import project modules after ensuring the repository root is importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.checkpoint_reconstruction import reconstruct_checkpoint_prediction

    return reconstruct_checkpoint_prediction


reconstruct_checkpoint_prediction = _load_runtime_dependencies()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for checkpoint reconstruction."""
    parser = argparse.ArgumentParser(
        description="Reconstruct a missed checkpoint prediction from a stored session snapshot."
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, for example 2026.")
    parser.add_argument(
        "--race-name",
        required=True,
        help="Race name, for example 'Australian Grand Prix'.",
    )
    parser.add_argument(
        "--checkpoint-session",
        required=True,
        help="Checkpoint session code, for example FP1 or FP3.",
    )
    parser.add_argument(
        "--weather",
        help="Optional race-weather label. Defaults to the first saved weather for the race.",
    )
    parser.add_argument(
        "--qualifying-n-simulations",
        type=int,
        help="Override qualifying simulation count.",
    )
    parser.add_argument(
        "--race-n-simulations",
        type=int,
        help="Override race simulation count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rewriting an existing checkpoint artifact with the reconstructed payload.",
    )
    return parser


def main() -> int:
    """Run the checkpoint reconstruction CLI."""
    args = build_parser().parse_args()
    summary = reconstruct_checkpoint_prediction(
        year=int(args.year),
        race_name=str(args.race_name).strip(),
        checkpoint_session=str(args.checkpoint_session).strip(),
        weather=args.weather,
        overwrite=bool(args.overwrite),
        qualifying_n_simulations=args.qualifying_n_simulations,
        race_n_simulations=args.race_n_simulations,
    )

    print(f"Reconstructed: {summary.race_name} {summary.year} {summary.checkpoint_session}")
    print(f"Weekend type: {'sprint' if summary.is_sprint else 'normal'}")
    print(f"Weather: {summary.weather}")
    print(f"Targets: {', '.join(summary.target_keys)}")
    print(f"Prediction file: {summary.prediction_path}")
    print(f"Accuracy snapshots written: {summary.snapshot_records_written}")
    print(f"Actuals source: {summary.actuals_source}")
    print(f"Information cutoff: {summary.information_cutoff_at or 'unavailable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
