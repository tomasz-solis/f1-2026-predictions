"""Dashboard update flows for race learning and practice capture."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import fastf1
import streamlit as st

from src.persistence.config import should_read_db_first, should_write_to_db, should_write_to_file
from src.persistence.runtime_state_store import RuntimeStateStore
from src.utils.operational_observability import record_alert, record_counter

_PRACTICE_UPDATE_STATE_FILE = Path("data/systems/practice_characteristics_state.json")
_EVENT_BOUNDARY_STATE_FILE = Path("data/systems/event_boundary_refresh_state.json")
_CONVENTIONAL_BOUNDARY_SESSIONS = ("FP1", "FP2", "FP3", "Q", "R")
_SPRINT_BOUNDARY_SESSIONS = ("FP1", "SQ", "Sprint", "Q", "R")
_STATE_NAMESPACE_EVENT_BOUNDARY = "event_boundary_refresh"
_STATE_NAMESPACE_PRACTICE = "practice_characteristics"
_PRACTICE_BACKLOG_LOCK_TTL_SECONDS = 900

logger = logging.getLogger(__name__)


def _get_runtime_state_store() -> RuntimeStateStore:
    return RuntimeStateStore()


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
    if should_read_db_first():
        db_races = _get_runtime_state_store().load_namespace(_STATE_NAMESPACE_EVENT_BOUNDARY)
        if db_races:
            return {"races": db_races}
        if should_write_to_db():
            return {"races": {}}

    if not _EVENT_BOUNDARY_STATE_FILE.exists():
        return {"races": {}}

    try:
        with open(_EVENT_BOUNDARY_STATE_FILE) as f:
            loaded_state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"races": {}}

    if not isinstance(loaded_state, dict):
        return {"races": {}}

    file_races = loaded_state.get("races")
    if not isinstance(file_races, dict):
        return {"races": {}}

    return {"races": file_races}


def _save_event_boundary_state(state: dict) -> None:
    """Persist event-boundary refresh state."""
    races = state.get("races", {})
    if isinstance(races, dict) and should_write_to_db():
        db_payload = {str(key): value for key, value in races.items() if isinstance(value, dict)}
        _get_runtime_state_store().upsert_many(_STATE_NAMESPACE_EVENT_BOUNDARY, db_payload)

    if not should_write_to_file():
        return

    _EVENT_BOUNDARY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _EVENT_BOUNDARY_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(_EVENT_BOUNDARY_STATE_FILE)


def _build_event_boundary_snapshot(
    year: int,
    race_name: str,
    is_sprint: bool,
    session_detector=None,
    now_utc: datetime | None = None,
) -> dict:
    """Build current boundary snapshot using FastF1 data availability."""
    from src.utils.session_detector import SessionDetector

    now = now_utc or datetime.now(UTC)
    sessions = _SPRINT_BOUNDARY_SESSIONS if is_sprint else _CONVENTIONAL_BOUNDARY_SESSIONS
    detector = session_detector or SessionDetector()

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
    session_completion: dict[str, bool] = {}
    elapsed_sessions: list[str] = []
    latest_elapsed: str | None = None

    for session_name in sessions:
        try:
            raw_session_date = event.get_session_date(session_name)
        except Exception:
            raw_session_date = None

        session_date = _coerce_utc_datetime(raw_session_date)
        session_schedule[session_name] = session_date.isoformat() if session_date else ""

        if session_date is not None and session_date > now:
            session_completion[session_name] = False
            continue

        try:
            if hasattr(detector, "get_session_completion_state"):
                session_state = detector.get_session_completion_state(year, race_name, session_name)
                session_is_completed = session_state == "completed"
            else:
                session_is_completed = bool(
                    detector.is_session_completed(year, race_name, session_name)
                )
        except Exception as exc:
            logger.debug(
                "Could not determine completion state for %s %s %s: %s",
                year,
                race_name,
                session_name,
                exc,
            )
            session_is_completed = False

        session_completion[session_name] = session_is_completed
        if session_is_completed:
            elapsed_sessions.append(session_name)
            latest_elapsed = session_name

    has_schedule_data = any(session_schedule.values()) or any(session_completion.values())
    return {
        "weekend_type": "sprint" if is_sprint else "conventional",
        "session_order": list(sessions),
        "session_schedule": session_schedule,
        "session_completion": session_completion,
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
        "session_completion": snapshot.get("session_completion", {}),
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
    session_detector=None,
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
        session_detector=session_detector,
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
        previous_completion = previous.get("session_completion", {})

        if previous_weekend_type != snapshot["weekend_type"]:
            refresh_needed = True
            reason = "weekend_type_changed"
        elif previous_schedule != snapshot["session_schedule"]:
            refresh_needed = True
            reason = "schedule_changed"
        elif previous_completion != snapshot["session_completion"]:
            refresh_needed = True
            reason = "session_data_changed"
            new_sessions = [
                session for session in current_elapsed if session not in previous_elapsed
            ]
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
    except Exception as exc:
        logger.warning(f"Could not persist event-boundary refresh state: {exc}")
        record_counter(
            "event_boundary_state_persist_failure_total",
            labels={"year": year, "race_name": race_name},
        )
        record_alert(
            "event_boundary_state_persist_failure",
            f"Could not persist event-boundary refresh state for {race_name} {year}: {exc}",
            labels={"year": year, "race_name": race_name},
        )
        if should_write_to_db():
            raise RuntimeError(
                f"Supabase event-boundary state persistence failed for {race_name} {year}: {exc}"
            ) from exc

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

    needs_update_flag, new_races = needs_update(year=year, force_recheck=force_recheck)

    if needs_update_flag:
        st.info(f"Found {len(new_races)} new race(s) to learn from. Updating characteristics...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(message)

        updated_count = auto_update_from_races(
            progress_callback=progress_callback,
            races_to_update=new_races,
            year=year,
        )

        progress_bar.empty()
        status_text.empty()

        if updated_count == len(new_races):
            st.success(f"Learned from {updated_count} race(s). Predictions now use updated data.")
            st.cache_resource.clear()
            st.cache_data.clear()
        elif updated_count > 0:
            st.warning(
                "Race refresh partially completed: "
                f"updated {updated_count} of {len(new_races)} race(s). "
                "Continuing with available updates; remaining races will retry automatically."
            )
            logger.warning(
                "Race refresh incomplete for %s: updated %s/%s race(s).",
                year,
                updated_count,
                len(new_races),
            )
            st.cache_resource.clear()
            st.cache_data.clear()
        else:
            st.warning(
                "Race refresh did not apply any new updates. "
                "Continuing with existing model state; failed races will retry automatically."
            )
            logger.warning(
                "Race refresh skipped all pending races for %s (0/%s updated).",
                year,
                len(new_races),
            )


def _load_practice_update_state() -> dict:
    """Load persisted state for practice characteristic updates."""
    if should_read_db_first():
        db_races = _get_runtime_state_store().load_namespace(_STATE_NAMESPACE_PRACTICE)
        if db_races:
            return {"races": db_races}
        if should_write_to_db():
            return {"races": {}}

    if not _PRACTICE_UPDATE_STATE_FILE.exists():
        return {"races": {}}

    try:
        with open(_PRACTICE_UPDATE_STATE_FILE) as f:
            loaded_state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"races": {}}

    if not isinstance(loaded_state, dict):
        return {"races": {}}

    file_races = loaded_state.get("races")
    if not isinstance(file_races, dict):
        return {"races": {}}

    return {"races": file_races}


def _save_practice_update_state(state: dict) -> None:
    """Persist state for practice characteristic updates."""
    races = state.get("races", {})
    if isinstance(races, dict) and should_write_to_db():
        db_payload = {str(key): value for key, value in races.items() if isinstance(value, dict)}
        _get_runtime_state_store().upsert_many(_STATE_NAMESPACE_PRACTICE, db_payload)

    if not should_write_to_file():
        return

    _PRACTICE_UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRACTICE_UPDATE_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(_PRACTICE_UPDATE_STATE_FILE)


def _is_competitive_race_event(event_name: str, event_format: str) -> bool:
    """Return True for race weekends and False for testing placeholders."""
    normalized_name = str(event_name).strip().lower()
    normalized_format = str(event_format).strip().lower()
    if not normalized_name:
        return False
    if "testing" in normalized_name or "testing" in normalized_format:
        return False
    return True


def _event_is_sprint(event_format: str) -> bool:
    """Infer sprint weekend from FastF1 EventFormat."""
    return "sprint" in str(event_format).strip().lower()


def _iter_candidate_practice_events(
    year: int,
    focus_race_name: str,
    focus_is_sprint: bool,
) -> list[tuple[str, bool]]:
    """
    Return ordered race events to check for pending practice deltas.

    Includes all completed raceweekends in the season and always includes the
    currently selected race as a fallback/current-weekend guard.
    """
    candidates: list[tuple[str, bool]] = []
    now_utc = datetime.now(UTC)

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:
        logger.debug("Could not load schedule for backlog practice updates: %s", exc)
        return [(focus_race_name, focus_is_sprint)]

    for _, event in schedule.iterrows():
        event_name = str(event.get("EventName", "")).strip()
        event_format = str(event.get("EventFormat", "")).strip()
        if not _is_competitive_race_event(event_name, event_format):
            continue

        event_date = _coerce_utc_datetime(event.get("EventDate"))
        if event_date is not None and event_date > now_utc:
            continue

        candidates.append((event_name, _event_is_sprint(event_format)))

    if not any(name == focus_race_name for name, _ in candidates):
        candidates.append((focus_race_name, focus_is_sprint))

    return candidates


def auto_update_practice_characteristics_if_needed(
    year: int,
    race_name: str,
    is_sprint: bool,
    force_recheck: bool = False,
    session_detector=None,
) -> dict:
    """
    Update car characteristics from completed free-practice sessions (FP1/FP2/FP3).

    This is conservative and only runs when new FP sessions are completed for a race.

    Args:
        year: Season year
        race_name: Name of the race
        is_sprint: Whether this is a sprint weekend
        force_recheck: If True, ignores cached state and re-checks session completion
        session_detector: Optional pre-built detector instance for call-level memoization
    """
    from src.systems.testing_updater import update_from_testing_sessions
    from src.utils import config_loader
    from src.utils.session_detector import SessionDetector

    detector = session_detector or SessionDetector()
    session_order = {"FP1": 1, "FP2": 2, "FP3": 3}
    expected_fp_sessions_by_weekend = {
        False: {"FP1", "FP2", "FP3"},
        True: {"FP1"},
    }
    race_key = f"{year}::{race_name}"
    state = _load_practice_update_state()
    completed_by_race: dict[str, list[str]] = {}
    pending_updates: list[tuple[str, list[str], list[str]]] = []

    for event_name, event_is_sprint in _iter_candidate_practice_events(
        year=year,
        focus_race_name=race_name,
        focus_is_sprint=is_sprint,
    ):
        event_key = f"{year}::{event_name}"
        processed_sessions = {
            str(session) for session in state["races"].get(event_key, {}).get("sessions", [])
        }
        expected_sessions = expected_fp_sessions_by_weekend[event_is_sprint]
        if not force_recheck and expected_sessions.issubset(processed_sessions):
            completed_by_race[event_name] = sorted(
                expected_sessions,
                key=lambda s: session_order.get(s, 99),
            )
            continue

        completed = detector.get_completed_sessions(year, event_name, event_is_sprint)
        completed_fp_sessions = sorted(
            {session for session in completed if session.startswith("FP")},
            key=lambda s: session_order.get(s, 99),
        )
        if not completed_fp_sessions:
            completed_by_race[event_name] = []
            continue

        completed_by_race[event_name] = completed_fp_sessions
        sessions_to_update = (
            completed_fp_sessions
            if force_recheck
            else [session for session in completed_fp_sessions if session not in processed_sessions]
        )
        if sessions_to_update:
            pending_updates.append((event_name, completed_fp_sessions, sessions_to_update))

    focus_completed_sessions = completed_by_race.get(race_name, [])
    if not pending_updates:
        return {"updated": False, "completed_fp_sessions": focus_completed_sessions}

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

    all_updated_events: list[str] = []
    all_updated_teams: set[str] = set()
    focus_updated_teams: set[str] = set()
    retried_events: list[str] = []
    state_store = _get_runtime_state_store()
    for event_name, full_completed_sessions, sessions_to_update in pending_updates:
        lock_owner = uuid4().hex
        lock_key = f"practice_backlog::{year}::{event_name}"
        lock_acquired = True
        if should_write_to_db():
            lock_acquired = state_store.acquire_lock(
                lock_key,
                lock_owner,
                ttl_seconds=_PRACTICE_BACKLOG_LOCK_TTL_SECONDS,
            )
            if not lock_acquired:
                retried_events.append(event_name)
                record_counter(
                    "practice_backlog_retry_total",
                    labels={"year": year, "race_name": event_name},
                )
                record_alert(
                    "practice_backlog_retry",
                    (
                        f"Practice backlog update deferred because another worker holds lock "
                        f"for {event_name} {year}."
                    ),
                    labels={"year": year, "race_name": event_name},
                )
                continue

        try:
            summary = update_from_testing_sessions(
                year=year,
                characteristics_year=year,
                events=[event_name],
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
        finally:
            if should_write_to_db() and lock_acquired:
                try:
                    state_store.release_lock(lock_key, lock_owner)
                except Exception as exc:
                    logger.warning("Could not release practice backlog lock %s: %s", lock_key, exc)
        updated_teams = summary.get("updated_teams", []) if isinstance(summary, dict) else []
        normalized_updated = {str(team) for team in updated_teams}
        all_updated_teams.update(normalized_updated)
        if event_name == race_name:
            focus_updated_teams.update(normalized_updated)
        all_updated_events.append(event_name)

        state["races"][f"{year}::{event_name}"] = {
            "sessions": full_completed_sessions,
            "updated_at": datetime.now(UTC).isoformat(),
            "teams_updated": len(normalized_updated),
        }
        _save_practice_update_state(state)

    if not all_updated_events:
        return {
            "updated": False,
            "completed_fp_sessions": focus_completed_sessions,
            "retried_events": retried_events,
        }

    if race_key not in state["races"] and race_name in completed_by_race:
        state["races"][race_key] = {
            "sessions": focus_completed_sessions,
            "updated_at": datetime.now(UTC).isoformat(),
            "teams_updated": 0,
        }
        _save_practice_update_state(state)

    return {
        "updated": True,
        "completed_fp_sessions": focus_completed_sessions,
        "teams_updated": len(focus_updated_teams),
        "updated_events": all_updated_events,
        "total_teams_updated": len(all_updated_teams),
        "retried_events": retried_events,
    }
