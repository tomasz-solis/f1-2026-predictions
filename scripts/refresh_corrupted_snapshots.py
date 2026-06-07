#!/usr/bin/env python
"""Re-extract car-characteristics snapshots that were corrupted by the old short-run
lap-selection logic, using the fixed extractor (pace-based short-run selection,
absolute pace floor, clamped teammate deltas).

Background: the snapshot extractor previously classified short-run laps by *stint
length* and accepted a stint's minimum lap even when a stint contained only slow
cooldown/in-out laps. This stored physically impossible "representative" laps (e.g.
McLaren Monaco FP2 = 104.3s vs a real ~74s) and saturating teammate deltas (±29s),
which the qualifying blend then trusted at ~78%. The fix lives in
``src/systems/testing_updater_metrics.py`` and ``src/systems/testing_updater.py``.

This script re-runs ``backfill_session_snapshot_history`` (which uses the fixed
extractor) for the affected sessions and re-versions the snapshot artifacts.

Usage (from repo root, with .env.local loaded so DB writes target Supabase):
    python scripts/refresh_corrupted_snapshots.py                 # dry-run (no writes)
    python scripts/refresh_corrupted_snapshots.py --write         # persist corrected snapshots
    python scripts/refresh_corrupted_snapshots.py --write --only "Monaco Grand Prix"
"""

from __future__ import annotations

import argparse
import logging

from src.systems.testing_updater import backfill_session_snapshot_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("refresh_corrupted_snapshots")

# (event_name, [session codes]) flagged by the cross-session corruption scan.
CORRUPTED_SESSIONS: list[tuple[str, list[str]]] = [
    ("Monaco Grand Prix", ["FP2", "FP3"]),
    ("Canadian Grand Prix", ["FP1"]),
    ("Miami Grand Prix", ["FP1"]),
    ("Australian Grand Prix", ["FP2"]),
    ("Testing 1", ["Testing 1 Day 1"]),
    ("Testing 2", ["Testing 2 Day 2", "Testing 2 Day 3"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--write", action="store_true", help="Persist (default is dry-run).")
    parser.add_argument(
        "--only", default=None, help="Restrict to a single event name (exact match)."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="FastF1 cache dir. Defaults to the project testing cache.",
    )
    args = parser.parse_args()

    targets = CORRUPTED_SESSIONS
    if args.only:
        targets = [(e, s) for e, s in targets if e == args.only]
        if not targets:
            raise SystemExit(f"No matching event for --only {args.only!r}")

    dry_run = not args.write
    logger.info("Mode: %s", "DRY-RUN (no writes)" if dry_run else "WRITE")

    for event_name, sessions in targets:
        kwargs: dict = dict(
            year=args.year,
            events=[event_name],
            sessions=sessions,
            characteristics_year=args.year,
            dry_run=dry_run,
        )
        if args.cache_dir:
            kwargs["cache_dir"] = args.cache_dir
        try:
            result = backfill_session_snapshot_history(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface per-event failures, continue
            logger.error("FAILED %s %s: %s: %s", event_name, sessions, type(exc).__name__, exc)
            continue
        logger.info(
            "%s %s -> loaded=%s written=%s keys=%s",
            event_name,
            sessions,
            result.get("loaded_sessions"),
            result.get("snapshots_written"),
            result.get("snapshot_keys"),
        )


if __name__ == "__main__":
    main()
