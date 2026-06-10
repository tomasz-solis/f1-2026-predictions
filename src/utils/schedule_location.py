"""Resolve a race's physical ``Location`` from the FastF1 schedule (cached, best-effort).

Used to feed the circuit registry an authoritative venue so a migrating GP name (e.g. the
Spanish Grand Prix moving from Barcelona to Madrid) resolves by where the race is actually
held. Any failure to load the schedule returns ``None`` so callers fall back to name/year
resolution rather than breaking.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _location_map(year: int) -> dict[str, str]:
    """Return a normalized ``event-name -> location`` map for one season (cached)."""
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(int(year), include_testing=False)
    except Exception as exc:  # noqa: BLE001 - best-effort; any failure means "no hint"
        logger.debug("Could not load FastF1 schedule for %s to resolve location: %s", year, exc)
        return {}

    mapping: dict[str, str] = {}
    try:
        for _, event in schedule.iterrows():
            name = str(event.get("EventName", "")).strip().lower()
            location = str(event.get("Location", "")).strip()
            if name and location:
                mapping[name] = location
    except Exception as exc:  # noqa: BLE001 - tolerate unexpected schedule shapes
        logger.debug("Could not parse FastF1 schedule for %s: %s", year, exc)
        return {}
    return mapping


def location_for_race(year: int | None, race_name: str | None) -> str | None:
    """Return the schedule ``Location`` for a race, or ``None`` when unavailable."""
    if year is None or not race_name:
        return None
    return _location_map(int(year)).get(str(race_name).strip().lower()) or None
