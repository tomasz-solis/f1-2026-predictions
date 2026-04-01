#!/usr/bin/env python3
"""Rebuild local checkpoint forecast files from historical 2026 session data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.accuracy_targets import TARGET_SPRINT_QUALIFYING  # noqa: E402
from src.utils.historical_replay import run_historical_checkpoint_replay  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the historical replay helper."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay preseason testing and completed 2026 weekends into local checkpoint "
            "forecast files without leaking future sessions into earlier predictions."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Season year to replay.",
    )
    parser.add_argument(
        "--source-processed-dir",
        default="data/processed",
        help="Processed source directory used to seed the sidecar replay root.",
    )
    parser.add_argument(
        "--output-root",
        default="data/historical_replay",
        help="Sidecar output root for replayed processed data, predictions, and reports.",
    )
    parser.add_argument(
        "--weather",
        default="dry",
        help="Weather label used for race and sprint-race predictions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output root before rebuilding it.",
    )
    parser.add_argument(
        "--score-sprint-qualifying",
        action="store_true",
        help="Include sprint qualifying in accuracy scoring instead of saving it as unscored.",
    )
    return parser


def main() -> int:
    """Run the historical replay and print the key output paths."""
    args = build_parser().parse_args()
    excluded_targets = set()
    if not args.score_sprint_qualifying:
        excluded_targets.add(TARGET_SPRINT_QUALIFYING)

    summary = run_historical_checkpoint_replay(
        year=int(args.year),
        source_processed_dir=str(args.source_processed_dir),
        output_root=str(args.output_root),
        weather=str(args.weather).strip().lower(),
        overwrite=bool(args.overwrite),
        excluded_scoring_targets=excluded_targets,
    )

    logger.info("Replay output root: %s", summary.output_root)
    logger.info("Processed replay state: %s", summary.processed_data_dir)
    logger.info("Checkpoint files generated: %s", len(summary.checkpoints))
    logger.info("Summary report: %s", Path(summary.output_root) / "reports" / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
