#!/usr/bin/env python3
"""Rebuild Team Comparison snapshots from their production artifact keys.

The migration reads the currently stored snapshot keys from Supabase, re-extracts
those exact sessions with the current telemetry code, and writes new artifact
versions only when ``--apply`` is supplied. Existing versions are retained.

Render usage (run both commands in the same cron-service Shell so the FastF1
cache populated by the dry-run is reused by the apply run):

    python scripts/backfill_team_comparison_snapshots.py --year 2026
    python scripts/backfill_team_comparison_snapshots.py --year 2026 --apply
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.persistence.artifact_store import ArtifactStore  # noqa: E402
from src.persistence.config import (  # noqa: E402
    get_storage_mode,
    should_read_db_first,
    should_write_to_db,
)

logger = logging.getLogger("backfill_team_comparison_snapshots")

_SNAPSHOT_ARTIFACT_TYPE = "car_characteristics_snapshot"
_MAIN_CACHE_DIR = "data/raw/.fastf1_cache"
_TESTING_CACHE_DIR = "data/raw/.fastf1_cache_testing"
_RAW_METRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("slow_corner_performance", "slow_corner_seconds"),
    ("medium_corner_performance", "medium_corner_seconds"),
    ("fast_corner_performance", "fast_corner_seconds"),
    ("braking_performance", "braking_pct"),
)


@dataclass(frozen=True)
class SnapshotJob:
    """Describe the sessions to rebuild for one event."""

    event_name: str
    sessions: tuple[str, ...]
    cache_dir: str


@dataclass(frozen=True)
class SnapshotAudit:
    """Summarize validation of the latest production snapshot versions."""

    latest_key_count: int
    source_counts: dict[str, int]
    stale_sources: tuple[tuple[str, str], ...]
    missing_raw_metrics: tuple[tuple[str, str, str, str, str], ...]

    @property
    def passed(self) -> bool:
        """Return whether every migrated key uses a valid latest payload."""
        return not self.stale_sources and not self.missing_raw_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Season year to rebuild.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write new snapshot versions. Without this flag the command is a dry-run.",
    )
    parser.add_argument(
        "--force-renew-cache",
        action="store_true",
        help="Force FastF1 to renew cached session data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=600,
        help="Maximum artifact rows to inspect when discovering production keys.",
    )
    parser.add_argument("--worker-event", help=argparse.SUPPRESS)
    parser.add_argument("--worker-session", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache-dir", help=argparse.SUPPRESS)
    return parser.parse_args()


def _production_snapshot_keys(
    store: ArtifactStore,
    *,
    year: int,
    limit: int,
) -> tuple[str, ...]:
    """Return unique production snapshot keys for one season."""
    rows = store.list_artifacts(
        _SNAPSHOT_ARTIFACT_TYPE,
        key_prefix=f"{int(year)}::",
        limit=max(1, int(limit)),
    )
    return tuple(
        sorted(
            {
                str(row.get("artifact_key", "")).strip()
                for row in rows
                if isinstance(row, dict) and str(row.get("artifact_key", "")).strip()
            }
        )
    )


def _build_snapshot_jobs(
    artifact_keys: tuple[str, ...],
    *,
    year: int,
) -> tuple[SnapshotJob, ...]:
    """Group artifact keys into explicit event/session backfill jobs."""
    sessions_by_event: dict[str, set[str]] = defaultdict(set)
    year_token = str(int(year))

    for artifact_key in artifact_keys:
        parts = str(artifact_key).split("::", 2)
        if len(parts) != 3:
            raise ValueError(f"Unexpected snapshot artifact key: {artifact_key}")
        key_year, event_name, stored_session_name = parts
        if key_year != year_token:
            continue

        session_argument = stored_session_name
        if event_name.lower().startswith("testing"):
            event_prefix = f"{event_name} "
            if session_argument.startswith(event_prefix):
                session_argument = session_argument[len(event_prefix) :]
        sessions_by_event[event_name].add(session_argument)

    jobs: list[SnapshotJob] = []
    for event_name in sorted(sessions_by_event):
        is_testing = event_name.lower().startswith("testing")
        jobs.append(
            SnapshotJob(
                event_name=event_name,
                sessions=tuple(sorted(sessions_by_event[event_name])),
                cache_dir=_TESTING_CACHE_DIR if is_testing else _MAIN_CACHE_DIR,
            )
        )
    return tuple(jobs)


def _latest_snapshot_payloads(
    store: ArtifactStore,
    *,
    year: int,
    limit: int,
) -> dict[str, dict[str, Any]]:
    """Return the newest listed payload for each production snapshot key."""
    rows = store.list_artifacts(
        _SNAPSHOT_ARTIFACT_TYPE,
        key_prefix=f"{int(year)}::",
        limit=max(1, int(limit)),
    )
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        artifact_key = str(row.get("artifact_key", "")).strip()
        payload = row.get("data")
        if artifact_key and isinstance(payload, dict):
            # ArtifactStore lists DB rows newest-first.
            latest.setdefault(artifact_key, payload)
    return latest


def _audit_latest_snapshots(
    latest: dict[str, dict[str, Any]],
    *,
    migrated_keys: tuple[str, ...],
) -> SnapshotAudit:
    """Check source provenance and raw-metric completeness for migrated keys."""
    source_counts: Counter[str] = Counter()
    stale_sources: list[tuple[str, str]] = []
    missing_raw_metrics: list[tuple[str, str, str, str, str]] = []

    for artifact_key in migrated_keys:
        payload = latest.get(artifact_key)
        if not isinstance(payload, dict):
            stale_sources.append((artifact_key, "missing_payload"))
            continue

        source = str(payload.get("source", "unknown"))
        source_counts[source] += 1
        if source != "snapshot_history_backfill":
            stale_sources.append((artifact_key, source))

        teams_payload = payload.get("teams")
        if not isinstance(teams_payload, dict):
            continue
        for team_name, team_payload in teams_payload.items():
            if not isinstance(team_payload, dict):
                continue
            profiles = team_payload.get("profiles")
            if not isinstance(profiles, dict):
                continue
            for profile_name, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                for normalized_key, raw_key in _RAW_METRIC_PAIRS:
                    if normalized_key in profile and raw_key not in profile:
                        missing_raw_metrics.append(
                            (
                                artifact_key,
                                str(team_name),
                                str(profile_name),
                                normalized_key,
                                raw_key,
                            )
                        )

    return SnapshotAudit(
        latest_key_count=len(latest),
        source_counts=dict(source_counts),
        stale_sources=tuple(stale_sources),
        missing_raw_metrics=tuple(missing_raw_metrics),
    )


def _run_single_session_worker(args: argparse.Namespace) -> int:
    """Rebuild one session in an isolated process to cap FastF1 memory usage."""
    if not args.worker_event or not args.worker_session or not args.worker_cache_dir:
        logger.error("Incomplete snapshot worker arguments")
        return 2

    # This is intentionally imported only inside the short-lived worker. The
    # orchestrator remains lightweight while each FastF1 session is loaded, and
    # all telemetry memory is returned to the OS when the worker exits.
    from src.systems.testing_updater import backfill_session_snapshot_history

    summary = backfill_session_snapshot_history(
        year=args.year,
        characteristics_year=args.year,
        events=[args.worker_event],
        sessions=[args.worker_session],
        testing_backend="auto",
        cache_dir=args.worker_cache_dir,
        force_renew_cache=bool(args.force_renew_cache),
        run_profile="balanced",
        dry_run=not args.apply,
    )
    loaded_sessions = list(summary.get("loaded_sessions", []))
    snapshots_written = int(summary.get("snapshots_written", 0))
    logger.info(
        "Worker %s %s: loaded=%s written=%s",
        args.worker_event,
        args.worker_session,
        len(loaded_sessions),
        snapshots_written,
    )
    if len(loaded_sessions) != 1:
        logger.error("Worker did not load its requested session: %s", loaded_sessions)
        return 1
    if args.apply and snapshots_written != 1:
        logger.error("Worker wrote %s snapshots; expected 1", snapshots_written)
        return 1
    return 0


def _worker_command(
    args: argparse.Namespace,
    *,
    job: SnapshotJob,
    session: str,
) -> list[str]:
    """Build the isolated worker command for one event session."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--year",
        str(args.year),
        "--worker-event",
        job.event_name,
        "--worker-session",
        session,
        "--worker-cache-dir",
        job.cache_dir,
    ]
    if args.apply:
        command.append("--apply")
    if args.force_renew_cache:
        command.append("--force-renew-cache")
    return command


def _run_backfill(args: argparse.Namespace) -> int:
    if args.worker_event or args.worker_session or args.worker_cache_dir:
        return _run_single_session_worker(args)

    if not should_read_db_first() or not should_write_to_db():
        logger.error(
            "DB-backed storage is required (mode=%s, db_read=%s, db_write=%s)",
            get_storage_mode(),
            should_read_db_first(),
            should_write_to_db(),
        )
        return 2

    store = ArtifactStore("data")
    artifact_keys = _production_snapshot_keys(store, year=args.year, limit=args.limit)
    if not artifact_keys:
        logger.error("No production snapshot keys found for %s", args.year)
        return 2

    jobs = _build_snapshot_jobs(artifact_keys, year=args.year)
    logger.info(
        "Mode=%s year=%s production_keys=%s events=%s",
        "APPLY" if args.apply else "DRY-RUN",
        args.year,
        len(artifact_keys),
        len(jobs),
    )

    total_loaded = 0
    total_written = 0
    for job in jobs:
        logger.info("Rebuilding %s: %s", job.event_name, ", ".join(job.sessions))
        for session in job.sessions:
            logger.info("Starting isolated worker: %s %s", job.event_name, session)
            completed = subprocess.run(
                _worker_command(args, job=job, session=session),
                check=False,
            )
            if completed.returncode != 0:
                logger.error(
                    "Worker failed for %s %s with exit code %s",
                    job.event_name,
                    session,
                    completed.returncode,
                )
                return 1
            total_loaded += 1
            if args.apply:
                total_written += 1

    logger.info("Sessions loaded=%s snapshots written=%s", total_loaded, total_written)
    if not args.apply:
        logger.info("Dry-run passed; no artifacts were written")
        return 0

    latest = _latest_snapshot_payloads(store, year=args.year, limit=args.limit)
    audit = _audit_latest_snapshots(latest, migrated_keys=artifact_keys)
    logger.info(
        "Audit latest_keys=%s sources=%s stale_sources=%s missing_raw_metrics=%s",
        audit.latest_key_count,
        audit.source_counts,
        len(audit.stale_sources),
        len(audit.missing_raw_metrics),
    )
    for stale_violation in audit.stale_sources[:20]:
        logger.error("Stale source: %s", stale_violation)
    for raw_metric_violation in audit.missing_raw_metrics[:20]:
        logger.error("Missing raw metric: %s", raw_metric_violation)
    if not audit.passed:
        logger.error("Backfill completed, but the production audit failed")
        return 1

    logger.info("Backfill and production audit passed")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(_run_backfill(_parse_args()))


if __name__ == "__main__":
    main()
