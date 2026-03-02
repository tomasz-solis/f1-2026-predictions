"""Run one or more background session-automation cycles."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.session_automation import (  # noqa: E402
    ensure_session_automation_config,
    run_session_automation_cycle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for session automation worker execution."""
    parser = argparse.ArgumentParser(description="Run scheduled session automation.")
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now(UTC).year,
        help="Season year to process (default: current UTC year).",
    )
    parser.add_argument(
        "--weather",
        choices=["dry", "rain", "mixed"],
        default=None,
        help="Weather assumption for automated predictions (default: schedule config value).",
    )
    parser.add_argument(
        "--disable-auto-predict",
        action="store_true",
        help="Skip prediction generation; only refresh updates and reconcile actuals.",
    )
    parser.add_argument(
        "--force-recheck",
        action="store_true",
        help="Re-check boundaries and completed races even if they were seen before.",
    )
    parser.add_argument(
        "--no-reconcile-actuals",
        action="store_true",
        help="Skip post-race actuals reconciliation and accuracy snapshot writes.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=0,
        help="When > 0, run continuously with this sleep interval.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Event polling lookback window (stored in schedule config).",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=2,
        help="Event polling lookahead window (stored in schedule config).",
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Persist schedule as disabled and exit after one no-op cycle.",
    )
    return parser


def _run_once(args: argparse.Namespace) -> int:
    """Run a single automation cycle and log structured summary output."""
    config = ensure_session_automation_config(
        args.year,
        enabled=not bool(args.disabled),
        auto_predict=not bool(args.disable_auto_predict),
        weather=args.weather,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
    )
    logger.info(
        "Session automation schedule: year=%s enabled=%s auto_predict=%s weather=%s",
        config.year,
        config.enabled,
        config.auto_predict,
        config.weather,
    )

    summary = run_session_automation_cycle(
        year=args.year,
        weather=args.weather,
        auto_predict=False if args.disable_auto_predict else None,
        force_recheck=bool(args.force_recheck),
        reconcile_actuals=not bool(args.no_reconcile_actuals),
    )
    logger.info("Session automation cycle summary: %s", summary.to_dict())
    return 0


def main() -> int:
    """Program entrypoint for the session automation worker script."""
    parser = _build_parser()
    args = parser.parse_args()

    interval_seconds = max(0, int(args.interval_seconds))
    if interval_seconds == 0:
        return _run_once(args)

    logger.info("Starting continuous mode with interval=%ss", interval_seconds)
    while True:
        exit_code = _run_once(args)
        if exit_code != 0:
            return exit_code
        time.sleep(interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
