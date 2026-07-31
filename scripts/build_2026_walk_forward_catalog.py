#!/usr/bin/env python3
"""Build the immutable 2026 walk-forward event catalog from real cached data only.

This script never invents an event, session, grid, or timestamp. Every row is built
from FastF1's local session cache (``data/raw/.fastf1_cache``); an event that cannot
be fully resolved (missing qualifying or race classification, a session that fails to
load) is *skipped* and the reason is recorded in the sibling ``*_report.json`` -- it is
never fabricated or interpolated.

Only the main Grand Prix qualifying (``Q``) and race (``R``) sessions are scored, per
the walk-forward contract used by ``run_challenger_walk_forward``. Sprint-format
weekends are still included (their ``session_kind`` is recorded as ``sprint``) but the
sprint qualifying/sprint race sessions themselves are not part of this catalog.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fastf1  # noqa: E402

from src.data.actual_results_fetcher import (  # noqa: E402
    fetch_actual_session_results,
    fetch_official_starting_grid,
)

YEAR = 2026
FASTF1_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"
CATALOG_PATH = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "event_catalog.json"
REPORT_PATH = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "event_catalog_report.json"
# Excluded explicitly: the weekend live/in-progress on the run date (2026-07-19).
# Excluding it here (rather than relying solely on the natural "no race data yet"
# fail-closed path) makes the exclusion reason explicit in the report.
EXPLICITLY_EXCLUDED_EVENTS = {"Belgian Grand Prix": "in_progress_weekend_excluded_by_run_date"}

_MAIN_PRACTICE_SESSIONS = ("Practice 1", "Practice 2", "Practice 3")
_CHECKPOINT_FOR_PRACTICE_SESSION = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    ts = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _slugify(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("'", "")


def detect_session_kind(event_format: str) -> str:
    """Return ``"sprint"`` or ``"main"`` from a FastF1 ``EventFormat`` string.

    2026 sprint weekends report formats such as ``sprint_qualifying``; every other
    known format (``conventional``, ``sprint_shootout`` from earlier seasons, etc.)
    that does not contain ``"sprint"`` is treated as the standard FP1-FP2-FP3-Q-R
    weekend. This is a pure function so the walk-forward catalog's session_kind
    labelling is unit-testable without a live FastF1 session load.
    """
    return "sprint" if "sprint" in str(event_format).strip().lower() else "main"


def _session_is_dry(year: int, race_name: str, session_name: str) -> bool | None:
    """Return whether a session's weather log shows any rainfall, or None if unknown."""
    try:
        session = fastf1.get_session(year, race_name, session_name)
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception:  # noqa: BLE001 - fail closed to "unknown", never invent weather
        return None
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0 or "Rainfall" not in weather.columns:
        return None
    return not bool(weather["Rainfall"].any())


def _build_event_row(row: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    race_name = str(row["EventName"]).strip()
    round_number = int(row["RoundNumber"])
    event_id = f"{YEAR}_{round_number:02d}_{_slugify(race_name)}"
    diagnostics: dict[str, Any] = {
        "event_id": event_id,
        "race_name": race_name,
        "round_number": round_number,
    }

    if race_name in EXPLICITLY_EXCLUDED_EVENTS:
        diagnostics["status"] = "excluded"
        diagnostics["reason"] = EXPLICITLY_EXCLUDED_EVENTS[race_name]
        return None, diagnostics

    event_format = str(row.get("EventFormat", "")).strip().lower()
    session_kind = detect_session_kind(event_format)
    is_sprint = session_kind == "sprint"
    diagnostics["event_format"] = event_format
    diagnostics["session_kind"] = session_kind

    try:
        event = fastf1.get_event(YEAR, race_name)
    except Exception as exc:  # noqa: BLE001
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = f"schedule_lookup_failed: {exc}"
        return None, diagnostics

    event_start_at = _utc(event.get_session_date("Practice 1", utc=True))
    try:
        qualifying_start_at = _utc(event.get_session_date("Qualifying", utc=True))
    except Exception as exc:  # noqa: BLE001
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = f"qualifying_session_date_unavailable: {exc}"
        return None, diagnostics
    if event_start_at is None or qualifying_start_at is None:
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = "missing_session_dates"
        return None, diagnostics

    actual_qualifying_grid = fetch_actual_session_results(YEAR, race_name, "Q")
    if not actual_qualifying_grid:
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = "missing_or_incomplete_qualifying_results"
        return None, diagnostics

    actual_race_finish_order = fetch_actual_session_results(YEAR, race_name, "R")
    if not actual_race_finish_order:
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = "missing_or_incomplete_race_results"
        return None, diagnostics

    actual_starting_grid = fetch_official_starting_grid(
        YEAR,
        race_name,
        session_name="R",
        qualifying_classification=actual_qualifying_grid,
    )
    if not actual_starting_grid:
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = "missing_official_starting_grid"
        return None, diagnostics

    q_dry = _session_is_dry(YEAR, race_name, "Q")
    r_dry = _session_is_dry(YEAR, race_name, "R")
    if q_dry is None or r_dry is None:
        diagnostics["status"] = "skipped"
        diagnostics["reason"] = "weather_data_unavailable"
        return None, diagnostics
    is_dry = bool(q_dry and r_dry)
    diagnostics["is_dry"] = is_dry
    diagnostics["qualifying_dry"] = q_dry
    diagnostics["race_dry"] = r_dry

    # The event schedule's own session ordering is the cutoff source: a practice
    # checkpoint's information becomes available once the *next* scheduled session on
    # the weekend begins (still strictly inside the pre-qualifying window). This
    # avoids depending on the dashboard's live-clock checkpoint reconstruction, which
    # is designed for "now" rather than a fixed historical replay.
    weekend_sessions: list[tuple[str, datetime]] = []
    for session_label in ("Practice 1", "Practice 2", "Practice 3", "Qualifying"):
        if is_sprint and session_label in ("Practice 2", "Practice 3"):
            continue
        try:
            session_start = _utc(event.get_session_date(session_label, utc=True))
        except Exception:  # noqa: BLE001
            continue
        if session_start is not None:
            weekend_sessions.append((session_label, session_start))
    weekend_sessions.sort(key=lambda row: row[1])

    checkpoint_payloads: dict[str, Any] = {
        "PRE": {
            "information_cutoff_at": _iso(event_start_at),
            "sessions_available": [],
        }
    }
    cumulative_sessions: list[str] = []
    for index, (session_label, _) in enumerate(weekend_sessions):
        if session_label not in _CHECKPOINT_FOR_PRACTICE_SESSION:
            continue
        checkpoint = _CHECKPOINT_FOR_PRACTICE_SESSION[session_label]
        next_session_start = weekend_sessions[index + 1][1]
        cutoff_dt = next_session_start - timedelta(seconds=1)
        if not (event_start_at <= cutoff_dt < qualifying_start_at):
            continue
        cumulative_sessions = [*cumulative_sessions, session_label]
        checkpoint_payloads[checkpoint] = {
            "information_cutoff_at": _iso(cutoff_dt),
            "sessions_available": list(cumulative_sessions),
        }

    diagnostics["status"] = "included"
    diagnostics["checkpoints"] = sorted(checkpoint_payloads)
    event_row = {
        "event_id": event_id,
        "race_name": race_name,
        "round_number": round_number,
        "event_start_at": _iso(event_start_at),
        "qualifying_start_at": _iso(qualifying_start_at),
        "session_kind": "sprint" if is_sprint else "main",
        "is_dry": is_dry,
        "input_snapshot_ids": [f"fastf1_cache_2026::{_slugify(race_name)}"],
        "actual_qualifying_grid": actual_qualifying_grid,
        "actual_race_finish_order": actual_race_finish_order,
        "actual_starting_grid": actual_starting_grid,
        "checkpoint_payloads": checkpoint_payloads,
        "fastf1_cache_dir": str(FASTF1_CACHE_DIR),
    }
    return event_row, diagnostics


def main() -> int:
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
    schedule = schedule.sort_values("RoundNumber")

    events: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for _, row in schedule.iterrows():
        race_name = str(row["EventName"]).strip()
        round_number = int(row["RoundNumber"])
        cache_dir = FASTF1_CACHE_DIR / str(YEAR)
        event_cache_dirs = [
            child
            for child in (cache_dir.iterdir() if cache_dir.is_dir() else [])
            if race_name.split(" Grand Prix")[0].replace(" ", "_") in child.name
        ]
        has_local_cache = any(
            any(session_dir.glob("*.ff1pkl"))
            for event_dir in event_cache_dirs
            for session_dir in event_dir.iterdir()
        )
        if race_name not in EXPLICITLY_EXCLUDED_EVENTS and not has_local_cache:
            diagnostics_rows.append(
                {
                    "event_id": f"{YEAR}_{round_number:02d}_{_slugify(race_name)}",
                    "race_name": race_name,
                    "round_number": round_number,
                    "status": "skipped",
                    "reason": "no_local_fastf1_cache_directory",
                }
            )
            continue
        event_row, diagnostics = _build_event_row(row)
        diagnostics_rows.append(diagnostics)
        if event_row is not None:
            events.append(event_row)

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog_payload = {
        "artifact_type": "walk_forward_event_catalog",
        "schema_version": 1,
        "year": YEAR,
        "generated_at": _iso(datetime.now(UTC)),
        "source": "fastf1_local_cache",
        "cache_dir": str(FASTF1_CACHE_DIR),
        "events": events,
    }
    CATALOG_PATH.write_text(
        json.dumps(catalog_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_payload = {
        "generated_at": _iso(datetime.now(UTC)),
        "year": YEAR,
        "included_count": len(events),
        "candidate_count": len(diagnostics_rows),
        "rows": diagnostics_rows,
    }
    REPORT_PATH.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"catalog: {CATALOG_PATH} ({len(events)} events)")
    print(f"report:  {REPORT_PATH}")
    for diag in diagnostics_rows:
        print(f"  {diag['event_id']}: {diag['status']} ({diag.get('reason', 'ok')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
