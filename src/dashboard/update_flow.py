"""Dashboard update flows for race learning and practice capture."""

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fastf1
import streamlit as st

_PRACTICE_UPDATE_STATE_FILE = Path("data/systems/practice_characteristics_state.json")
_EVENT_BOUNDARY_STATE_FILE = Path("data/systems/event_boundary_refresh_state.json")
_CONVENTIONAL_BOUNDARY_SESSIONS = ("FP1", "FP2", "FP3", "Q", "R")
_SPRINT_BOUNDARY_SESSIONS = ("FP1", "SQ", "Sprint", "Q", "R")

logger = logging.getLogger(__name__)


def _coerce_utc_datetime(value) -> datetime | None:
    """Normalize FastF1 datetime-like values to UTC-aware datetime."""
    if value is None:
        return None

    candidate = value
    if hasattr(candidate, "to_pydatetime"):
        try:
            candidate = candidate.to_pydatetime()
        except Exception:
            return None

    if not isinstance(candidate, datetime):
        return None

    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=UTC)

    return candidate.astimezone(UTC)


def _load_event_boundary_state() -> dict:
    """Load persisted event-boundary refresh state."""
    if not _EVENT_BOUNDARY_STATE_FILE.exists():
        return {"races": {}}

    try:
        with open(_EVENT_BOUNDARY_STATE_FILE) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"races": {}}

    if not isinstance(state, dict):
        return {"races": {}}

    races = state.get("races")
    if not isinstance(races, dict):
        return {"races": {}}

    return {"races": races}


def _save_event_boundary_state(state: dict) -> None:
    """Persist event-boundary refresh state."""
    _EVENT_BOUNDARY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _EVENT_BOUNDARY_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(_EVENT_BOUNDARY_STATE_FILE)


def _build_event_boundary_snapshot(
    year: int,
    race_name: str,
    is_sprint: bool,
    now_utc: datetime | None = None,
) -> dict:
    """Build current schedule-boundary snapshot for a race weekend."""
    from src.utils.session_detector import SessionDetector

    sessions = _SPRINT_BOUNDARY_SESSIONS if is_sprint else _CONVENTIONAL_BOUNDARY_SESSIONS
    now = now_utc or datetime.now(UTC)

    try:
        event = fastf1.get_event(year, race_name)
    except Exception as exc:
        logger.debug(f"Could not load FastF1 event for boundary refresh check: {exc}")
        return {
            "weekend_type": "sprint" if is_sprint else "conventional",
            "session_order": list(sessions),
            "session_schedule": {},
            "elapsed_sessions": [],
            "latest_elapsed_session": None,
            "has_schedule_data": False,
        }

    session_schedule: dict[str, str] = {}
    elapsed_sessions: list[str] = []
    latest_elapsed: str | None = None

    for session_name in sessions:
        try:
            raw_session_date = event.get_session_date(session_name)
        except Exception:
            raw_session_date = None

        session_date = _coerce_utc_datetime(raw_session_date)
        session_schedule[session_name] = session_date.isoformat() if session_date else ""
        if session_date is None:
            continue

        duration_hours = SessionDetector.SESSION_DURATIONS.get(session_name, 2.0)
        session_end = session_date + timedelta(hours=duration_hours)
        if now >= session_end:
            elapsed_sessions.append(session_name)
            latest_elapsed = session_name

    has_schedule_data = any(session_schedule.values())
    return {
        "weekend_type": "sprint" if is_sprint else "conventional",
        "session_order": list(sessions),
        "session_schedule": session_schedule,
        "elapsed_sessions": elapsed_sessions,
        "latest_elapsed_session": latest_elapsed,
        "has_schedule_data": has_schedule_data,
    }


def _boundary_signature(snapshot: dict) -> str:
    """Build deterministic signature for boundary-sensitive cache keys."""
    payload = {
        "weekend_type": snapshot.get("weekend_type", "conventional"),
        "session_order": snapshot.get("session_order", []),
        "session_schedule": snapshot.get("session_schedule", {}),
        "elapsed_sessions": snapshot.get("elapsed_sessions", []),
        "latest_elapsed_session": snapshot.get("latest_elapsed_session"),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def detect_event_boundary_refresh_if_needed(
    year: int,
    race_name: str,
    is_sprint: bool,
    now_utc: datetime | None = None,
) -> dict:
    """
    Detect whether a new schedule/session boundary was crossed since the last run.

    Returns:
        dict with `refresh_needed`, `reason`, `new_sessions`, and latest-boundary details.
    """
    snapshot = _build_event_boundary_snapshot(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        now_utc=now_utc,
    )

    if not snapshot["has_schedule_data"]:
        return {
            "refresh_needed": False,
            "reason": "schedule_unavailable",
            "new_sessions": [],
            "boundary_signature": "",
            "latest_elapsed_session": None,
            "previous_latest_elapsed_session": None,
        }

    race_key = f"{year}::{race_name}"
    state = _load_event_boundary_state()
    previous_raw = state["races"].get(race_key)
    previous = previous_raw if isinstance(previous_raw, dict) else None

    current_elapsed = snapshot["elapsed_sessions"]
    current_latest = snapshot["latest_elapsed_session"]
    refresh_needed = False
    reason = "no_change"
    new_sessions: list[str] = []
    previous_latest: str | None = None

    if previous:
        previous_elapsed_raw = previous.get("elapsed_sessions", [])
        previous_elapsed = (
            [str(session) for session in previous_elapsed_raw]
            if isinstance(previous_elapsed_raw, list)
            else []
        )
        previous_latest = previous.get("latest_elapsed_session")
        previous_schedule = previous.get("session_schedule", {})
        previous_weekend_type = previous.get("weekend_type")

        if previous_weekend_type != snapshot["weekend_type"]:
            refresh_needed = True
            reason = "weekend_type_changed"
        elif previous_schedule != snapshot["session_schedule"]:
            refresh_needed = True
            reason = "schedule_changed"
        else:
            new_sessions = [
                session for session in current_elapsed if session not in previous_elapsed
            ]
            if new_sessions or previous_latest != current_latest:
                refresh_needed = True
                reason = "session_boundary_delta"
    else:
        if current_latest is not None:
            refresh_needed = True
            reason = "first_seen_after_boundary"
            new_sessions = list(current_elapsed)

    state["races"][race_key] = {
        **snapshot,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    try:
        _save_event_boundary_state(state)
    except OSError as exc:
        logger.warning(f"Could not persist event-boundary refresh state: {exc}")

    return {
        "refresh_needed": refresh_needed,
        "reason": reason,
        "new_sessions": new_sessions,
        "boundary_signature": _boundary_signature(snapshot),
        "latest_elapsed_session": current_latest,
        "previous_latest_elapsed_session": previous_latest,
    }


def auto_update_if_needed(force_recheck: bool = False, year: int = 2026) -> None:
    """
    Check for and apply updates from completed races.
    Also refreshes predictor if characteristic files were manually updated.

    Args:
        force_recheck: If True, clears learned races cache to force re-check
        year: Season year to evaluate for newly completed races
    """
    from src.utils.auto_updater import auto_update_from_races, needs_update

    if force_recheck:
        try:
            needs_update_flag, new_races = needs_update(year=year, force_recheck=True)
        except TypeError:
            try:
                # Backward-compatible fallback for patched or older callables without year kwargs.
                needs_update_flag, new_races = needs_update(force_recheck=True)
            except TypeError:
                needs_update_flag, new_races = needs_update()
    else:
        try:
            needs_update_flag, new_races = needs_update(year=year)
        except TypeError:
            needs_update_flag, new_races = needs_update()

    if needs_update_flag:
        st.info(f"Found {len(new_races)} new race(s) to learn from. Updating characteristics...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(message)

        try:
            updated_count = auto_update_from_races(
                progress_callback=progress_callback,
                races_to_update=new_races,
                year=year,
            )
        except TypeError:
            try:
                # Backward-compatible fallback for patched callables without year kwargs.
                updated_count = auto_update_from_races(
                    progress_callback=progress_callback,
                    races_to_update=new_races,
                )
            except TypeError:
                updated_count = auto_update_from_races(progress_callback)

        progress_bar.empty()
        status_text.empty()

        if updated_count == len(new_races):
            st.success(f"Learned from {updated_count} race(s). Predictions now use updated data.")
            st.cache_resource.clear()
            st.cache_data.clear()
        else:
            raise RuntimeError(
                f"Race refresh incomplete: updated {updated_count} of {len(new_races)} new races."
            )


def _load_practice_update_state() -> dict:
    """Load persisted state for practice characteristic updates."""
    if not _PRACTICE_UPDATE_STATE_FILE.exists():
        return {"races": {}}

    try:
        with open(_PRACTICE_UPDATE_STATE_FILE) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"races": {}}

    if not isinstance(state, dict):
        return {"races": {}}

    races = state.get("races")
    if not isinstance(races, dict):
        return {"races": {}}

    return {"races": races}


def _save_practice_update_state(state: dict) -> None:
    """Persist state for practice characteristic updates."""
    _PRACTICE_UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRACTICE_UPDATE_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(_PRACTICE_UPDATE_STATE_FILE)


def auto_update_practice_characteristics_if_needed(
    year: int,
    race_name: str,
    is_sprint: bool,
    force_recheck: bool = False,
) -> dict:
    """
    Update car characteristics from completed free-practice sessions (FP1/FP2/FP3).

    This is conservative and only runs when new FP sessions are completed for a race.

    Args:
        year: Season year
        race_name: Name of the race
        is_sprint: Whether this is a sprint weekend
        force_recheck: If True, ignores cached state and re-checks session completion
    """
    from src.systems.testing_updater import update_from_testing_sessions
    from src.utils import config_loader
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    completed = detector.get_completed_sessions(year, race_name, is_sprint)
    completed_fp_sessions = [session for session in completed if session.startswith("FP")]

    if not completed_fp_sessions:
        return {"updated": False, "completed_fp_sessions": []}

    session_order = {"FP1": 1, "FP2": 2, "FP3": 3}
    completed_fp_sessions = sorted(
        set(completed_fp_sessions), key=lambda s: session_order.get(s, 99)
    )

    race_key = f"{year}::{race_name}"
    state = _load_practice_update_state()
    processed_sessions = set(state["races"].get(race_key, {}).get("sessions", []))
    sessions_to_update = (
        completed_fp_sessions
        if force_recheck
        else [session for session in completed_fp_sessions if session not in processed_sessions]
    )
    if not sessions_to_update:
        return {"updated": False, "completed_fp_sessions": completed_fp_sessions}

    practice_new_weight = config_loader.get("baseline_predictor.practice_capture.new_weight", 0.35)
    practice_directionality_scale = config_loader.get(
        "baseline_predictor.practice_capture.directionality_scale", 0.08
    )
    practice_session_aggregation = config_loader.get(
        "baseline_predictor.practice_capture.session_aggregation", "laps_weighted"
    )
    practice_run_profile = config_loader.get(
        "baseline_predictor.practice_capture.run_profile", "balanced"
    )

    summary = update_from_testing_sessions(
        year=year,
        characteristics_year=year,
        events=[race_name],
        sessions=sessions_to_update,
        testing_backend="auto",
        cache_dir="data/raw/.fastf1_cache_testing",
        force_renew_cache=False,
        # Lower weight than pre-season testing to avoid abrupt directionality swings.
        new_weight=practice_new_weight,
        directionality_scale=practice_directionality_scale,
        session_aggregation=practice_session_aggregation,
        run_profile=practice_run_profile,
        dry_run=False,
    )
    updated_teams = summary.get("updated_teams", []) if isinstance(summary, dict) else []

    state["races"][race_key] = {
        "sessions": completed_fp_sessions,
        "updated_at": datetime.now().isoformat(),
        "teams_updated": len(updated_teams),
    }
    _save_practice_update_state(state)

    return {
        "updated": True,
        "completed_fp_sessions": completed_fp_sessions,
        "teams_updated": len(updated_teams),
    }
