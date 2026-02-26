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

logger = logging.getLogger(__name__)


def _is_competitive_race_event(event: pd.Series) -> bool:
    """Return True only for proper race weekends (exclude testing/non-race placeholders)."""
    event_name = str(event.get("EventName", "")).strip()
    if not event_name:
        return False

    # Guardrail 1: explicit testing labels in event name.
    if "testing" in event_name.lower():
        return False

    # Guardrail 2: EventFormat metadata (when available).
    event_format = str(event.get("EventFormat", "")).strip().lower()
    if "testing" in event_format:
        return False

    # Guardrail 3: testing events are usually round 0.
    round_number = event.get("RoundNumber")
    if pd.notna(round_number):
        try:
            if int(round_number) <= 0:
                return False
        except (TypeError, ValueError):
            pass

    return True


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
            if not _is_competitive_race_event(event):
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
    learning_file = Path("data/learning_state.json")

    if not learning_file.exists():
        return []

    try:
        with open(learning_file) as f:
            state = json.load(f)
            history = state.get("history", [])
            state_season = state.get("season")
            learned: list[str] = []
            for record in history:
                if "race" not in record:
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
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Could not load learning state: {e}")
        return []


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
    learning_file = Path("data/learning_state.json")
    learning_file.parent.mkdir(parents=True, exist_ok=True)

    default_state = {
        "season": year,
        "races_completed": 0,
        "history": [],
        "method_performance": {},
    }

    if learning_file.exists():
        try:
            with open(learning_file) as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError("Learning state root must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                f"Learning state at {learning_file} is invalid ({exc}). Rebuilding state file."
            )
            state = default_state
    else:
        state = default_state

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

    with open(learning_file, "w") as f:
        json.dump(state, f, indent=2)
