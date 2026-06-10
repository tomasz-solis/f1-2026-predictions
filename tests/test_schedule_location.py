"""Tests for the cached schedule-location helper (best-effort, graceful)."""

import fastf1
import pandas as pd

from src.utils import schedule_location


def test_location_for_race_none_inputs():
    assert schedule_location.location_for_race(None, "Barcelona Grand Prix") is None
    assert schedule_location.location_for_race(2026, None) is None


def test_location_for_race_reads_schedule(monkeypatch):
    schedule_location._location_map.cache_clear()
    fake = pd.DataFrame(
        [
            {"EventName": "Barcelona Grand Prix", "Location": "Barcelona"},
            {"EventName": "Spanish Grand Prix", "Location": "Madrid"},
        ]
    )
    monkeypatch.setattr(fastf1, "get_event_schedule", lambda year, include_testing=False: fake)

    assert schedule_location.location_for_race(2026, "Barcelona Grand Prix") == "Barcelona"
    assert schedule_location.location_for_race(2026, "Spanish Grand Prix") == "Madrid"
    assert schedule_location.location_for_race(2026, "Unknown Grand Prix") is None
    schedule_location._location_map.cache_clear()


def test_location_for_race_graceful_on_schedule_failure(monkeypatch):
    schedule_location._location_map.cache_clear()

    def _boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(fastf1, "get_event_schedule", _boom)
    assert schedule_location.location_for_race(2099, "Barcelona Grand Prix") is None
    schedule_location._location_map.cache_clear()
