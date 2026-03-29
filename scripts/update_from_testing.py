"""
Update Car Directionality from Testing/Practice Sessions

Manual/explicit updater for testing and weekend practice telemetry.
This script only runs when called directly.

Testing pace is useful for car-characteristic baselines, but it is not a clean
proxy for true team order under race-weekend conditions.

Examples:
    # Preview only (default mode)
    python scripts/update_from_testing.py "Pre-Season Testing"
    # Persist changes to car_characteristics JSON
    python scripts/update_from_testing.py "Australian Grand Prix" --sessions FP1 FP2 FP3 --apply
    # Explicit preview mode (equivalent to default)
    python scripts/update_from_testing.py "Pre-Season Testing" "Australian Grand Prix" --dry-run
    python scripts/update_from_testing.py "Testing 1" --session-aggregation laps_weighted
    python scripts/update_from_testing.py "Testing 1" --run-profile balanced
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.testing_updater import update_from_testing_sessions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update team directionality from testing/practice sessions"
    )
    parser.add_argument(
        "events",
        nargs="+",
        help="Event names to scan (e.g. 'Pre-Season Testing', 'Australian Grand Prix')",
    )
    parser.add_argument("--year", type=int, default=2026, help="Season year")
    parser.add_argument(
        "--characteristics-year",
        type=int,
        default=None,
        help="Characteristics file year (defaults to --year)",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional specific session names (default: auto candidate list)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "fastf1", "f1timing"],
        default="auto",
        help="FastF1 backend for testing sessions (default: auto with fallback)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw/.fastf1_cache_testing",
        help=("FastF1 cache directory for this run. Relative paths are created under data/raw."),
    )
    parser.add_argument(
        "--force-renew-cache",
        action="store_true",
        help="Force FastF1 to renew cache entries",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument(
        "--new-weight",
        type=float,
        default=0.7,
        help="Weight of newly extracted metrics in directionality blend (0-1)",
    )
    parser.add_argument(
        "--directionality-scale",
        type=float,
        default=0.10,
        help="Scale factor to convert 0-1 relative metrics into centered directionality deltas",
    )
    parser.add_argument(
        "--session-aggregation",
        choices=["mean", "median", "laps_weighted"],
        default="laps_weighted",
        help=(
            "How to combine multiple sessions (e.g., Day 1/2/3). "
            "laps_weighted uses valid lap counts per team as weights."
        ),
    )
    parser.add_argument(
        "--run-profile",
        choices=["balanced", "all", "short_run", "long_run"],
        default="balanced",
        help=(
            "Which run types to trust inside each session. "
            "balanced blends short and long stint representatives."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Persist updates to characteristics file. By default the script runs in dry-run mode."
        ),
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview updates without writing characteristics file (default mode)",
    )

    args = parser.parse_args()
    dry_run = args.dry_run or not args.apply

    if dry_run:
        logger.info("Running in dry-run mode. Use --apply to write changes to disk.")

    try:
        summary = update_from_testing_sessions(
            year=args.year,
            events=args.events,
            data_dir=args.data_dir,
            sessions=args.sessions,
            characteristics_year=args.characteristics_year,
            testing_backend=args.backend,
            cache_dir=args.cache_dir,
            force_renew_cache=args.force_renew_cache,
            new_weight=args.new_weight,
            directionality_scale=args.directionality_scale,
            session_aggregation=args.session_aggregation,
            run_profile=args.run_profile,
            dry_run=dry_run,
        )
    except Exception as exc:
        logger.error(f"Testing directionality update failed: {exc}")
        raise SystemExit(1) from exc

    logger.info("=" * 70)
    logger.info("Testing directionality update complete")
    logger.info("=" * 70)
    logger.info(f"Year: {summary['year']}")
    logger.info(f"Characteristics year: {summary['characteristics_year']}")
    logger.info(f"Events: {', '.join(summary['events'])}")
    logger.info(f"Sessions loaded: {len(summary['loaded_sessions'])}")
    logger.info(f"Teams updated: {len(summary['updated_teams'])}")
    logger.info(f"Backend: {summary['testing_backend']}")
    logger.info(f"Cache dir: {summary['cache_dir']}")
    logger.info(f"Force renew cache: {summary['force_renew_cache']}")
    logger.info(f"Session aggregation: {summary['session_aggregation']}")
    logger.info(f"Run profile: {summary['run_profile']}")
    logger.info(f"Profiles captured: {', '.join(summary['profiles_captured'])}")
    logger.info(f"Dry run: {summary['dry_run']}")
    logger.info(f"Target file: {summary['characteristics_file']}")

    if summary["loaded_sessions"]:
        logger.info("Loaded sessions:")
        for session_id in summary["loaded_sessions"]:
            logger.info(f"  - {session_id}")

    if summary["updated_teams"]:
        logger.info("Updated teams:")
        for team in summary["updated_teams"]:
            logger.info(f"  - {team}")


if __name__ == "__main__":
    main()
