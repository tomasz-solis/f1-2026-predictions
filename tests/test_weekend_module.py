"""Focused tests for weekend-type utilities and fallback schedule behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils import weekend


def test_get_schedule_rows_uses_track_fallback_when_fastf1_empty(patcher, tmp_path):
    patcher.chdir(tmp_path)
    weekend.refresh_schedule_cache()

    fallback_file = Path("data/processed/track_characteristics/2027_track_characteristics.json")
    fallback_file.parent.mkdir(parents=True, exist_ok=True)
    fallback_file.write_text(
        json.dumps(
            {
                "tracks": {
                    "Chinese Grand Prix": {"has_sprint": True},
                    "Bahrain Grand Prix": {"has_sprint": False},
                }
            }
        )
    )

    patcher.setattr(
        weekend.fastf1,
        "get_event_schedule",
        lambda year: pd.DataFrame(columns=["EventName", "EventFormat"]),
    )

    rows = weekend._get_schedule_rows(2027)

    assert ("Chinese Grand Prix", "sprint") in rows
    assert ("Bahrain Grand Prix", "conventional") in rows


def test_refresh_schedule_cache_forces_new_fastf1_fetch(patcher):
    weekend.refresh_schedule_cache()

    schedules = [
        pd.DataFrame(
            {"EventName": ["Chinese Grand Prix"], "EventFormat": ["sprint"]},
        ),
        pd.DataFrame(
            {"EventName": ["Chinese Grand Prix"], "EventFormat": ["conventional"]},
        ),
    ]

    call_count = {"n": 0}

    def _get_event_schedule(year: int):
        current = schedules[min(call_count["n"], len(schedules) - 1)]
        call_count["n"] += 1
        return current

    patcher.setattr(weekend.fastf1, "get_event_schedule", _get_event_schedule)

    first = weekend.is_sprint_weekend(2026, "Chinese Grand Prix")
    weekend.refresh_schedule_cache()
    second = weekend.is_sprint_weekend(2026, "Chinese Grand Prix")

    assert first is True
    assert second is False
    assert call_count["n"] >= 2


def test_get_event_format_and_all_conventional_races(patcher):
    weekend.refresh_schedule_cache()
    patcher.setattr(
        weekend.fastf1,
        "get_event_schedule",
        lambda year: pd.DataFrame(
            {
                "EventName": ["Chinese Grand Prix", "Bahrain Grand Prix"],
                "EventFormat": ["sprint_shootout", "conventional"],
            }
        ),
    )

    assert weekend.get_event_format(2026, "Chinese Grand Prix") == "sprint_shootout"
    assert weekend.get_all_conventional_races(2026) == ["Bahrain Grand Prix"]


def test_is_sprint_weekend_returns_false_when_lookup_raises(patcher):
    patcher.setattr(
        weekend,
        "get_weekend_type",
        lambda year, race_name: (_ for _ in ()).throw(ValueError("missing race")),
    )

    assert weekend.is_sprint_weekend(2026, "Missing Race") is False
