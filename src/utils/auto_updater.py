"""
Automatic Race Data Updater

Checks for completed 2026 races and automatically updates team/driver characteristics.
Called by the dashboard before predictions; manual scripts remain available.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd

from src.persistence.config import should_read_db_first, should_write_to_db, should_write_to_file
from src.persistence.runtime_state_store import RuntimeStateStore
from src.utils.weekend import should_skip_schedule_event

logger = logging.getLogger(__name__)
_LEARNING_STATE_FILE = Path("data/learning_state.json")
_LEARNING_STATE_NAMESPACE = "race_learning"


def _is_competitive_race_event(event: pd.Series, *, year: int) -> bool:
    """Return True only for proper race weekends (exclude testing/non-race placeholders)."""
    event_name = str(event.get("EventName", "")).strip()
    if should_skip_schedule_event(year, event_name):
        return False

    # Guardrail 1: EventFormat metadata (when available).
    event_format = str(event.get("EventFormat", "")).strip().lower()
    if "testing" in event_format:
        return False

    # Guardrail 2: testing events are usually round 0.
    round_number = event.get("RoundNumber")
    if pd.notna(round_number):
        try:
            if int(round_number) <= 0:
                return False
        except (TypeError, ValueError):
            pass

    return True


def _default_learning_state(year: int) -> dict:
    return {
        "season": year,
        "races_completed": 0,
        "history": [],
        "method_performance": {},
        "last_updated": None,
    }


def _get_runtime_state_store() -> RuntimeStateStore:
    return RuntimeStateStore()


def _load_learning_state(year: int) -> dict:
    """Load learning state with DB-first semantics when configured."""
    if should_read_db_first():
        try:
            db_record = _get_runtime_state_store().get_record(
                _LEARNING_STATE_NAMESPACE,
                str(year),
            )
            if isinstance(db_record, dict):
                return db_record
            if should_write_to_db():
                # In write-capable DB modes, treat missing DB state as empty season state.
                return _default_learning_state(year)
        except Exception as exc:
            logger.warning(
                "Could not load race-learning state from DB for season %s: %s",
                year,
                exc,
            )
            if should_write_to_db():
                # Fail-open to empty state; do not trust ephemeral local files in stateless mode.
                return _default_learning_state(year)

    if not _LEARNING_STATE_FILE.exists():
        return _default_learning_state(year)

    try:
        with open(_LEARNING_STATE_FILE) as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Learning state at %s is invalid (%s). Rebuilding state.",
            _LEARNING_STATE_FILE,
            exc,
        )
        return _default_learning_state(year)

    if not isinstance(loaded, dict):
        logger.warning(
            "Learning state at %s is not an object. Rebuilding state.",
            _LEARNING_STATE_FILE,
        )
        return _default_learning_state(year)

    return loaded


def _save_learning_state(year: int, state: dict) -> None:
    """Persist learning state to configured backends."""
    if should_write_to_db():
        _get_runtime_state_store().upsert_record(_LEARNING_STATE_NAMESPACE, str(year), state)

    if not should_write_to_file():
        return

    _LEARNING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _LEARNING_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(_LEARNING_STATE_FILE)


def get_completed_races(year: int = 2026) -> list[str]:
    """Get list of completed races for the given year."""
    try:
        # Create cache directory if missing.
        import os

        cache_dir = Path(os.getenv("F1_CACHE_DIR", "data/raw/.fastf1_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        fastf1.Cache.enable_cache(str(cache_dir))
        schedule = fastf1.get_event_schedule(year)

        completed = []
        now = datetime.now(UTC)

        for _, event in schedule.iterrows():
            if not _is_competitive_race_event(event, year=year):
                continue

            # Check if race has happened (date in the past)
            if "EventDate" in event and pd.notna(event["EventDate"]):
                event_date = pd.Timestamp(event["EventDate"])
                if event_date.tzinfo is None:
                    event_date = event_date.tz_localize("UTC")
                else:
                    event_date = event_date.tz_convert("UTC")

                if event_date.to_pydatetime() < now:
                    race_name = event["EventName"]
                    # Try to load session metadata to confirm data is available.
                    try:
                        session = fastf1.get_session(year, race_name, "R")
                        if session is None:
                            continue
                        session.load(laps=False, telemetry=False, weather=False, messages=False)
                        results = getattr(session, "results", None)
                        if results is None:
                            continue
                        try:
                            if len(results) == 0:
                                continue
                        except TypeError:
                            pass
                        completed.append(race_name)
                    except (
                        ValueError,
                        KeyError,
                        AttributeError,
                        TypeError,
                        FileNotFoundError,
                        RuntimeError,
                    ) as e:
                        logger.debug(f"Race {race_name} not available yet: {e}")
                        continue  # Race not available yet

        return completed

    except Exception as e:
        logger.warning(f"Could not check for completed races: {e}")
        return []


def get_learned_races(year: int = 2026) -> list[str]:
    """Get races already learned for a specific season year."""
    state = _load_learning_state(year)
    history = state.get("history", [])
    if not isinstance(history, list):
        return []
    state_season = state.get("season")

    learned: list[str] = []
    for record in history:
        if not isinstance(record, dict) or "race" not in record:
            continue
        raw_record_year = record.get("year", state_season)
        try:
            record_year = int(raw_record_year) if raw_record_year is not None else None
        except (TypeError, ValueError):
            record_year = None
        if record_year is None:
            # Backward compatibility for legacy records without year:
            # treat them as belonging to default season only.
            if year == 2026:
                learned.append(record["race"])
        elif record_year == year:
            learned.append(record["race"])
    return learned


def needs_update(year: int = 2026, force_recheck: bool = False) -> tuple[bool, list[str]]:
    """
    Check if there are new races to learn from.

    Args:
        force_recheck: If True, re-check all completed races regardless of learned state
    """
    completed = get_completed_races(year=year)

    if force_recheck:
        # Force re-check: treat all completed races as potentially new
        logger.info(f"Force recheck enabled: found {len(completed)} completed race(s)")
        return len(completed) > 0, completed

    learned = get_learned_races(year=year)
    new_races = [race for race in completed if race not in learned]

    return len(new_races) > 0, new_races


def auto_update_from_races(
    progress_callback=None,
    races_to_update: list[str] | None = None,
    year: int = 2026,
) -> int:
    """Automatically update characteristics from completed races.

    Args:
        progress_callback: Optional callback receiving (current, total, message).
        races_to_update: Explicit race list to update. When provided, this exact
            list is used and `needs_update()` is not recomputed.
    """
    if races_to_update is None:
        needs_update_flag, new_races = needs_update(year=year)
        if not needs_update_flag:
            logger.info("All completed races have already been learned from.")
            return 0
    else:
        # Preserve caller order while removing duplicates.
        new_races = list(dict.fromkeys(races_to_update))
        if not new_races:
            logger.info("No races provided for explicit update.")
            return 0

    logger.info(f"Found {len(new_races)} new race(s) to learn from: {new_races}")

    # Import here to avoid circular dependency
    from src.systems.updater import update_from_race

    updated_count = 0

    for i, race_name in enumerate(new_races):
        try:
            if progress_callback:
                progress_callback(i + 1, len(new_races), f"Learning from {race_name}...")

            logger.info(f"Updating from {race_name} ({i + 1}/{len(new_races)})...")

            # Update from race (loads results, updates teams & drivers)
            update_from_race(year, race_name)

            # Mark as learned
            mark_race_as_learned(race_name, year=year)

            updated_count += 1
            logger.info(f"  Learned from {race_name}")

        except Exception as e:
            logger.warning(f"  Could not update from {race_name}: {e}")
            # Continue with other races even if one fails

    if updated_count > 0:
        logger.info(f"Updated from {updated_count} race(s).")

    return updated_count


def mark_race_as_learned(race_name: str, year: int = 2026) -> None:
    """Mark a race as learned in the learning state for a given season."""
    state = _load_learning_state(year)
    if not isinstance(state, dict):
        state = _default_learning_state(year)

    def _record_year_for_dedupe(record: dict) -> int | None:
        raw_year = record.get("year")
        if raw_year is not None:
            try:
                return int(raw_year)
            except (TypeError, ValueError):
                return None
        # Legacy records without explicit year are treated as default-season history.
        # This mirrors get_learned_races() behavior and avoids cross-season collisions.
        return 2026

    # Add to history if not already there
    if "history" not in state:
        state["history"] = []

    # Check if already marked
    if not any(
        r.get("race") == race_name and _record_year_for_dedupe(r) == year for r in state["history"]
    ):
        state["history"].append(
            {
                "race": race_name,
                "year": year,
                "date": datetime.now().isoformat(),
                "method": "auto_update",
            }
        )
        state["races_completed"] = state.get("races_completed", 0) + 1

    state["season"] = year
    state["last_updated"] = datetime.now().isoformat()
    _save_learning_state(year, state)
