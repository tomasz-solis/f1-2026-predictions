"""Tests for the team-performance helper script."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def _load_script_module():
    """Load the team-performance script as a module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "calculate_team_performance.py"
    spec = importlib.util.spec_from_file_location("calculate_team_performance_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLaps(pd.DataFrame):
    """Small DataFrame subclass that mimics the FastF1 laps helpers used by the script."""

    @property
    def _constructor(self):
        """Preserve the subclass across pandas slicing operations."""
        return _FakeLaps

    def pick_accurate(self):
        """Return the same frame for tests that do not model lap filtering."""
        return self

    def pick_quicklaps(self):
        """Return the same frame for tests that do not model lap filtering."""
        return self


class _FakeSession:
    """Minimal FastF1 session stub for team-performance tests."""

    def __init__(self, date: datetime, laps: _FakeLaps) -> None:
        self.date = date
        self.laps = laps
        self.load_calls = 0

    def load(self, laps: bool = True, telemetry: bool = False) -> None:
        """Record that the script attempted to load the session."""
        _ = laps
        _ = telemetry
        self.load_calls += 1


def _sample_laps() -> _FakeLaps:
    """Build one small lap dataset with enough rows for two teams."""
    return _FakeLaps(
        {
            "Team": (["McLaren"] * 10) + (["Ferrari"] * 10),
            "LapTime": [pd.Timedelta(seconds=90)] * 10 + [pd.Timedelta(seconds=91)] * 10,
        }
    )


def _sample_laps_with_metrics() -> _FakeLaps:
    """Build lap data with sector, speed, and stint fields for metric extraction."""
    mclaren_laps = {
        "Team": ["McLaren"] * 10,
        "Driver": ["NOR"] * 5 + ["PIA"] * 5,
        "LapTime": [pd.Timedelta(seconds=90 + (lap_idx * 0.2)) for lap_idx in range(10)],
        "Sector1Time": [pd.Timedelta(seconds=30.0)] * 10,
        "Sector2Time": [pd.Timedelta(seconds=29.5)] * 10,
        "Sector3Time": [pd.Timedelta(seconds=30.5)] * 10,
        "SpeedST": [320.0] * 10,
        "SpeedFL": [318.0] * 10,
        "Stint": [1] * 5 + [2] * 5,
        "Compound": ["MEDIUM"] * 10,
        "LapNumber": list(range(1, 6)) + list(range(10, 15)),
    }
    ferrari_laps = {
        "Team": ["Ferrari"] * 10,
        "Driver": ["LEC"] * 5 + ["HAM"] * 5,
        "LapTime": [pd.Timedelta(seconds=91 + (lap_idx * 0.2)) for lap_idx in range(10)],
        "Sector1Time": [pd.Timedelta(seconds=30.5)] * 10,
        "Sector2Time": [pd.Timedelta(seconds=30.1)] * 10,
        "Sector3Time": [pd.Timedelta(seconds=30.7)] * 10,
        "SpeedST": [315.0] * 10,
        "SpeedFL": [314.0] * 10,
        "Stint": [1] * 5 + [2] * 5,
        "Compound": ["MEDIUM"] * 10,
        "LapNumber": list(range(1, 6)) + list(range(10, 15)),
    }
    return _FakeLaps(
        {column: mclaren_laps[column] + ferrari_laps[column] for column in mclaren_laps}
    )


def test_coerce_utc_timestamp_handles_naive_and_aware_values():
    """Timezone normalization should accept both naive and aware inputs."""
    module = _load_script_module()

    naive = datetime(2024, 3, 2, 12, 0, 0)
    aware = datetime(2024, 3, 2, 13, 0, 0, tzinfo=timezone(timedelta(hours=1)))

    assert module._coerce_utc_timestamp(naive) == pd.Timestamp(naive, tz=UTC)
    assert module._coerce_utc_timestamp(aware) == pd.Timestamp(
        datetime(2024, 3, 2, 12, 0, tzinfo=UTC)
    )


def test_calculate_team_performance_loads_completed_race_with_naive_session_date(monkeypatch):
    """Naive FastF1 session dates should not crash the completed-race check."""
    module = _load_script_module()
    schedule = pd.DataFrame(
        [
            {
                "EventName": "Test Grand Prix",
                "EventFormat": "conventional",
            }
        ]
    )
    session = _FakeSession(
        date=datetime(2024, 3, 2, 12, 0, 0),
        laps=_sample_laps(),
    )

    monkeypatch.setattr(module.ff1, "get_event_schedule", lambda year: schedule)
    monkeypatch.setattr(module.ff1, "get_session", lambda year, race_name, session_name: session)

    module.calculate_team_performance_from_races(2024)

    assert session.load_calls == 1


def test_calculate_team_performance_emits_normalized_team_shape_metrics(monkeypatch):
    """Race-derived team payloads should include normalized side metrics when available."""
    module = _load_script_module()
    schedule = pd.DataFrame(
        [
            {
                "EventName": "Test Grand Prix 1",
                "EventFormat": "conventional",
            },
            {
                "EventName": "Test Grand Prix 2",
                "EventFormat": "conventional",
            },
            {
                "EventName": "Test Grand Prix 3",
                "EventFormat": "conventional",
            },
        ]
    )

    def _make_session() -> _FakeSession:
        return _FakeSession(
            date=datetime(2024, 3, 2, 12, 0, 0),
            laps=_sample_laps_with_metrics(),
        )

    monkeypatch.setattr(module.ff1, "get_event_schedule", lambda year: schedule)
    monkeypatch.setattr(
        module.ff1, "get_session", lambda year, race_name, session_name: _make_session()
    )

    payload = module.calculate_team_performance_from_races(2024)

    assert (
        payload["McLaren"]["normalized_overall_pace"]
        > payload["Ferrari"]["normalized_overall_pace"]
    )
    assert payload["McLaren"]["normalized_top_speed"] > payload["Ferrari"]["normalized_top_speed"]
    assert "normalized_slow_corner_performance" in payload["McLaren"]
    assert "normalized_tire_deg_performance" in payload["McLaren"]


def test_calculate_team_performance_respects_max_races(monkeypatch):
    """Early-season label generation should stop once the requested race count is reached."""
    module = _load_script_module()
    schedule = pd.DataFrame(
        [
            {"EventName": "Test Grand Prix 1", "EventFormat": "conventional"},
            {"EventName": "Test Grand Prix 2", "EventFormat": "conventional"},
            {"EventName": "Test Grand Prix 3", "EventFormat": "conventional"},
        ]
    )
    seen_races: list[str] = []

    def _get_session(year, race_name, session_name):
        seen_races.append(race_name)
        return _FakeSession(
            date=datetime(2024, 3, 2, 12, 0, 0),
            laps=_sample_laps(),
        )

    monkeypatch.setattr(module.ff1, "get_event_schedule", lambda year: schedule)
    monkeypatch.setattr(module.ff1, "get_session", _get_session)

    module.calculate_team_performance_from_races(2024, max_races=2)

    assert seen_races == ["Test Grand Prix 1", "Test Grand Prix 2"]
