"""
Weekend Type Utilities

Uses FastF1 EventFormat and avoids hardcoded sprint lists.
Falls back to local track characteristics data if FastF1 schedule is unavailable.

EventFormat values:
- 'sprint', 'sprint_qualifying', 'sprint_shootout' → sprint weekend
- 'conventional' → normal weekend

Usage:
    from src.utils.weekend import get_weekend_type, is_sprint_weekend

    weekend_type = get_weekend_type(2025, 'Chinese Grand Prix')  # 'sprint' or 'conventional'

    if is_sprint_weekend(2025, race_name):
        session = 'Sprint Qualifying'
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import fastf1

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_schedule_rows(year: int) -> tuple[tuple[str, str], ...]:
    """
    Load (EventName, EventFormat) rows from FastF1.

    Falls back to local track characteristics data if FastF1 is unavailable
    or returns an empty schedule (common in offline/test environments).
    """
    rows: list[tuple[str, str]] = []

    try:
        schedule = fastf1.get_event_schedule(year)
        if "EventName" in schedule.columns and "EventFormat" in schedule.columns:
            for _, event in schedule.iterrows():
                event_name = str(event.get("EventName", "")).strip()
                event_format = str(event.get("EventFormat", "")).strip().lower()
                if event_name:
                    rows.append((event_name, event_format))
    except Exception as exc:
        logger.warning(f"Could not load FastF1 schedule for {year}: {exc}")

    if rows:
        return tuple(rows)

    fallback_file = (
        Path("data/processed/track_characteristics") / f"{year}_track_characteristics.json"
    )
    if not fallback_file.exists():
        return tuple()

    try:
        with open(fallback_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not load fallback schedule from {fallback_file}: {exc}")
        return tuple()

    tracks = data.get("tracks", {})
    for race_name, track_data in tracks.items():
        if not race_name:
            continue
        has_sprint = bool(isinstance(track_data, dict) and track_data.get("has_sprint", False))
        rows.append((race_name, "sprint" if has_sprint else "conventional"))

    if rows:
        logger.info(f"Using local fallback schedule from {fallback_file} for {year}.")

    return tuple(rows)


def refresh_schedule_cache() -> None:
    """Clear cached schedule rows so next lookup fetches fresh FastF1 metadata."""
    _get_schedule_rows.cache_clear()


def get_schedule_rows(year: int) -> tuple[tuple[str, str], ...]:
    """Return cached `(EventName, EventFormat)` rows for a season."""
    return _get_schedule_rows(year)


def _find_event_format(year: int, race_name: str) -> str | None:
    """Return EventFormat string for race_name, or None if not found."""
    race_name_lower = race_name.lower()
    for event_name, event_format in _get_schedule_rows(year):
        if event_name == race_name or event_name.lower() == race_name_lower:
            return event_format
    return None


def get_weekend_type(year: int, race_name: str) -> Literal["sprint", "conventional"]:
    """Get weekend type from FastF1 EventFormat. Raises ValueError if race not found."""
    event_format = _find_event_format(year, race_name)
    if event_format is None:
        # Retry once with a fresh schedule snapshot to avoid stale in-process cache.
        refresh_schedule_cache()
        event_format = _find_event_format(year, race_name)

    if event_format is None:
        available_races = [event_name for event_name, _ in _get_schedule_rows(year)]
        raise ValueError(
            f"Race '{race_name}' not found in {year} schedule. Available races: {available_races}"
        )

    # Check if sprint weekend
    # Possible sprint formats: 'sprint', 'sprint_qualifying', 'sprint_shootout'
    if "sprint" in event_format:
        return "sprint"
    else:
        return "conventional"


def is_sprint_weekend(year: int, race_name: str) -> bool:
    """Check if weekend has sprint format. Returns bool."""
    try:
        return get_weekend_type(year, race_name) == "sprint"
    except ValueError as e:
        logger.warning(
            f"Could not determine weekend type for {race_name} ({year}): {e}. "
            f"Defaulting to conventional."
        )
        return False


def get_event_format(year: int, race_name: str) -> str:
    """Get exact EventFormat string from FastF1 schedule."""
    event_format = _find_event_format(year, race_name)
    if event_format is None:
        raise ValueError(f"Race '{race_name}' not found in {year} schedule")

    return event_format


def get_all_sprint_races(year: int) -> list[str]:
    """Get all sprint race names for a season from FastF1 schedule."""
    return [
        event_name
        for event_name, event_format in _get_schedule_rows(year)
        if "sprint" in event_format
    ]


def get_all_conventional_races(year: int) -> list[str]:
    """Get all conventional (non-sprint) race names for a season."""
    return [
        event_name
        for event_name, event_format in _get_schedule_rows(year)
        if "sprint" not in event_format
    ]


def get_best_qualifying_session(year: int, race_name: str) -> str:
    """Get best session for qualifying (Sprint Qualifying for sprint weekends, FP3 otherwise)."""
    weekend_type = get_weekend_type(year, race_name)

    if weekend_type == "sprint":
        return "Sprint Qualifying"
    else:
        return "FP3"
