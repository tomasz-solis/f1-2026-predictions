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
def _load_fallback_schedule_rows(year: int) -> tuple[tuple[str, str], ...]:
    """Load fallback `(EventName, EventFormat)` rows from local track data."""
    rows: list[tuple[str, str]] = []
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

    return tuple(rows)


def _merge_schedule_rows(
    primary_rows: tuple[tuple[str, str], ...],
    fallback_rows: tuple[tuple[str, str], ...],
    *,
    year: int,
) -> tuple[tuple[str, str], ...]:
    """Append fallback races that are missing from the primary schedule snapshot."""
    merged = list(primary_rows)
    seen_names = {event_name.lower() for event_name, _ in primary_rows}
    supplemented: list[str] = []

    for race_name, event_format in fallback_rows:
        normalized_name = race_name.lower()
        if normalized_name in seen_names:
            continue
        merged.append((race_name, event_format))
        seen_names.add(normalized_name)
        supplemented.append(race_name)

    if supplemented:
        logger.info(
            "Supplemented %s schedule with local fallback races: %s",
            year,
            supplemented,
        )

    return tuple(merged)


@lru_cache(maxsize=8)
def _get_schedule_rows(year: int) -> tuple[tuple[str, str], ...]:
    """
    Load `(EventName, EventFormat)` rows from FastF1 and local fallback data.

    FastF1 remains the primary source. Local track characteristics only fill in
    races that are missing from the current FastF1 schedule snapshot.
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

    fallback_rows = _load_fallback_schedule_rows(year)
    if rows and fallback_rows:
        return _merge_schedule_rows(tuple(rows), fallback_rows, year=year)

    if rows:
        return tuple(rows)

    if fallback_rows:
        logger.info("Using local fallback schedule for %s because FastF1 returned no rows.", year)
        return fallback_rows

    return tuple()


def refresh_schedule_cache() -> None:
    """Clear cached schedule rows so next lookup fetches fresh FastF1 metadata."""
    _get_schedule_rows.cache_clear()
    _load_fallback_schedule_rows.cache_clear()


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
    """Return True for sprint weekends and raise when the race cannot be resolved."""
    return get_weekend_type(year, race_name) == "sprint"


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
