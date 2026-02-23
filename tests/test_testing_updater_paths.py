from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.systems import testing_updater as tu


class _Event(dict):
    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    def get_session(self, day_number: int):
        return self._session


def test_resolve_testing_backends_invalid_value():
    with pytest.raises(ValueError, match="Invalid testing backend"):
        tu._resolve_testing_backends("invalid")


def test_resolve_testing_cache_dir_supports_absolute_paths(tmp_path):
    absolute = tmp_path / "cache"
    assert tu._resolve_testing_cache_dir(str(absolute)) == absolute


def test_coerce_utc_datetime_handles_timezone_aware_values():
    value = pd.Timestamp("2026-02-10T12:00:00+01:00")
    coerced = tu._coerce_utc_datetime(value)

    assert coerced is not None
    assert coerced.tzinfo is not None
    assert coerced.utcoffset() == timedelta(0)


def test_get_testing_event_with_backends_records_errors_and_falls_back(patcher):
    def _mock_get_testing_event(year, test_number, backend=None):
        if backend == "f1timing":
            raise RuntimeError("backend unavailable")
        return {"EventName": "Testing"}

    patcher.setattr(tu.fastf1, "get_testing_event", _mock_get_testing_event)
    errors = []

    event = tu._get_testing_event_with_backends(
        year=2026,
        test_number=1,
        testing_backends=("f1timing", "fastf1"),
        error_messages=errors,
    )

    assert event == {"EventName": "Testing"}
    assert errors and "backend=f1timing" in errors[0]


def test_load_testing_session_with_backends_handles_failed_backend(patcher):
    bad_session = SimpleNamespace(laps=None)
    bad_session.load = lambda **kwargs: None

    good_session = SimpleNamespace(
        laps=pd.DataFrame({"LapTime": [pd.to_timedelta("0:01:30")]}),
    )
    good_session.load = lambda **kwargs: None

    def _mock_get_testing_event(year, test_number, backend=None):
        if backend == "f1timing":
            return _Event(bad_session, Session1="Day 1")
        return _Event(good_session, Session1="Day 1")

    patcher.setattr(tu.fastf1, "get_testing_event", _mock_get_testing_event)

    errors = []
    loaded = tu._load_testing_session_with_backends(
        year=2026,
        test_number=1,
        day_number=1,
        testing_backends=("f1timing", "fastf1"),
        error_messages=errors,
    )

    assert loaded is good_session
    assert errors and "backend=f1timing" in errors[0]


def test_load_sessions_for_non_testing_event_collects_errors(patcher):
    session = SimpleNamespace(laps=pd.DataFrame({"LapTime": [pd.to_timedelta("0:01:30")]}))
    session.load = lambda **kwargs: None

    def _mock_get_session(year, event_name, session_name):
        if session_name == "FP1":
            raise RuntimeError("missing")
        return session

    patcher.setattr(tu.fastf1, "get_session", _mock_get_session)

    errors = []
    loaded = tu._load_sessions_for_event(
        year=2026,
        event_name="Bahrain Grand Prix",
        session_candidates=["FP1", "FP2"],
        error_messages=errors,
    )

    assert len(loaded) == 1
    assert loaded[0][0] == "FP2"
    assert errors and "Bahrain Grand Prix::FP1" in errors[0]


def test_load_sessions_for_testing_event_skips_future_sessions(patcher):
    future_event = {
        "Session1DateUtc": datetime.now(UTC) + timedelta(hours=2),
    }

    patcher.setattr(tu, "_get_testing_event_with_backends", lambda **kwargs: future_event)
    load_session = MagicMock(return_value=SimpleNamespace())
    patcher.setattr(tu, "_load_testing_session_with_backends", load_session)

    errors = []
    loaded = tu._load_sessions_for_event(
        year=2026,
        event_name="Testing 1",
        session_candidates=["Day 1"],
        testing_backends=("fastf1",),
        error_messages=errors,
    )

    assert loaded == []
    load_session.assert_not_called()
    assert any("session has not started yet" in msg for msg in errors)


def test_filter_valid_laps_branches():
    empty = pd.DataFrame()
    assert tu._filter_valid_laps(empty).empty

    missing_laptime = pd.DataFrame({"Team": ["Ferrari"]})
    assert tu._filter_valid_laps(missing_laptime).empty

    laps = pd.DataFrame(
        {
            "LapTime": [pd.to_timedelta("0:01:30"), pd.to_timedelta("0:01:31"), pd.NaT],
            "IsAccurate": [True, False, None],
        }
    )
    filtered = tu._filter_valid_laps(laps)
    assert len(filtered) == 1


def test_classify_run_laps_without_stint_uses_quantiles():
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 8,
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(8)],
            "PitOutTime": [pd.NaT] * 8,
            "PitInTime": [pd.NaT] * 8,
        }
    )

    short_laps, long_laps = tu._classify_run_laps(laps)
    assert not short_laps.empty
    assert not long_laps.empty


def test_select_program_aware_laps_invalid_profile_raises():
    laps = pd.DataFrame({"LapTime": [pd.to_timedelta("0:01:30")]})

    with pytest.raises(ValueError, match="Invalid run_profile"):
        tu._select_program_aware_laps(laps, run_profile="invalid")


def test_count_team_selected_laps_handles_laps_errors_and_invalid_profile(patcher):
    class BrokenSession:
        @property
        def laps(self):
            raise RuntimeError("no laps")

    assert tu._count_team_selected_laps(BrokenSession(), {"Ferrari"}) == {}

    session = SimpleNamespace(
        laps=pd.DataFrame(
            {
                "Team": ["Ferrari"],
                "LapTime": [pd.to_timedelta("0:01:30")],
            }
        )
    )
    patcher.setattr(tu, "_canonicalize_team_name", lambda raw_team, known_teams: "Ferrari")

    with pytest.raises(ValueError, match="Invalid run_profile"):
        tu._count_team_selected_laps(session, {"Ferrari"}, run_profile="invalid")


def test_metric_helpers_and_payload_extraction():
    assert tu._median_timedelta_seconds(pd.Series([], dtype=object)) is None
    assert tu._median_lap_seconds(pd.DataFrame()) is None
    assert tu._normalize_lower_better({"A": 1.0, "B": 1.0}) == {"A": 0.5, "B": 0.5}

    laps = pd.DataFrame(
        {
            "LapTime": [pd.to_timedelta("0:01:30"), pd.to_timedelta("0:01:32")],
            "Sector1Time": [pd.to_timedelta("0:00:30"), pd.to_timedelta("0:00:31")],
            "Sector2Time": [pd.to_timedelta("0:00:30"), pd.to_timedelta("0:00:31")],
            "Sector3Time": [pd.to_timedelta("0:00:30"), pd.to_timedelta("0:00:30")],
            "SpeedST": [330, 331],
        }
    )
    payload = tu._extract_team_payload(laps)

    assert "sector_times" in payload
    assert "speed_profile" in payload
    assert "consistency" in payload


def test_collect_session_metrics_unavailable_paths():
    diagnostics = []

    class BrokenSession:
        @property
        def laps(self):
            raise RuntimeError("fail")

    perf, tire = tu._collect_session_metrics(
        session=BrokenSession(),
        session_key="FP1",
        known_teams={"Ferrari"},
        diagnostics=diagnostics,
    )

    assert perf == {}
    assert tire == {}
    assert any("laps unavailable" in item for item in diagnostics)

    no_team_session = SimpleNamespace(laps=pd.DataFrame({"LapTime": [pd.to_timedelta("0:01:30")]}))
    diagnostics = []
    perf, tire = tu._collect_session_metrics(
        session=no_team_session,
        session_key="FP2",
        known_teams={"Ferrari"},
        diagnostics=diagnostics,
    )
    assert perf == {}
    assert tire == {}
    assert any("missing Team column" in item for item in diagnostics)


def test_collect_session_metrics_with_data(patcher):
    laps = pd.DataFrame(
        {
            "Team": ["Ferrari"] * 8,
            "Driver": ["LEC"] * 8,
            "Stint": [1, 1, 1, 1, 1, 1, 1, 1],
            "Compound": ["C3"] * 8,
            "LapNumber": list(range(2, 10)),
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(8)],
            "PitOutTime": [pd.NaT] * 8,
            "PitInTime": [pd.NaT] * 8,
        }
    )
    session = SimpleNamespace(laps=laps)

    patcher.setattr(tu, "_canonicalize_team_name", lambda raw_team, known_teams: "Ferrari")
    patcher.setattr(
        tu,
        "extract_all_teams_performance",
        lambda payload, session_name: {"Ferrari": {"top_speed": 0.6}},
    )

    diagnostics = []
    perf, tire = tu._collect_session_metrics(
        session=session,
        session_key="FP1",
        known_teams={"Ferrari"},
        run_profile="long_run",
        diagnostics=diagnostics,
    )

    assert "Ferrari" in perf
    assert "overall_pace" in perf["Ferrari"]
    assert diagnostics and "profile=long_run" in diagnostics[0]


def _write_characteristics(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_update_from_testing_sessions_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="At least one event name"):
        tu.update_from_testing_sessions(year=2026, events=[])

    with pytest.raises(FileNotFoundError, match="Characteristics file not found"):
        tu.update_from_testing_sessions(
            year=2026, events=["Testing 1"], data_dir=str(tmp_path / "missing")
        )


def test_update_from_testing_sessions_unusable_discovered_sessions(tmp_path, patcher):
    _write_characteristics(
        tmp_path,
        {
            "teams": {
                "Ferrari": {
                    "directionality": {
                        "max_speed": 0.0,
                        "slow_corner_speed": 0.0,
                        "medium_corner_speed": 0.0,
                        "high_corner_speed": 0.0,
                    }
                }
            }
        },
    )

    session = SimpleNamespace(
        laps=pd.DataFrame({"Team": ["Ferrari"], "LapTime": [pd.to_timedelta("0:01:30")]})
    )

    patcher.setattr(tu, "_load_sessions_for_event", lambda **kwargs: [("FP1", session)])
    patcher.setattr(tu, "_collect_session_metrics", lambda **kwargs: ({}, {}))
    patcher.setattr(tu.fastf1.Cache, "enable_cache", lambda path, force_renew=False: None)

    with pytest.raises(ValueError, match="too little completed running"):
        tu.update_from_testing_sessions(
            year=2026,
            events=["Bahrain Grand Prix"],
            data_dir=str(tmp_path / "processed"),
            dry_run=True,
        )


def test_update_from_testing_sessions_raises_when_no_teams_matched(tmp_path, patcher):
    _write_characteristics(
        tmp_path,
        {
            "teams": {
                "Ferrari": {
                    "directionality": {
                        "max_speed": 0.0,
                        "slow_corner_speed": 0.0,
                        "medium_corner_speed": 0.0,
                        "high_corner_speed": 0.0,
                    }
                }
            }
        },
    )

    session = SimpleNamespace(
        laps=pd.DataFrame({"Team": ["Unknown"], "LapTime": [pd.to_timedelta("0:01:30")]})
    )

    patcher.setattr(tu, "_load_sessions_for_event", lambda **kwargs: [("FP1", session)])
    patcher.setattr(
        tu, "_collect_session_metrics", lambda **kwargs: ({"Unknown": {"overall_pace": 0.7}}, {})
    )
    patcher.setattr(
        tu, "_count_team_selected_laps", lambda session, known_teams, run_profile: {"Unknown": 5.0}
    )
    patcher.setattr(tu.fastf1.Cache, "enable_cache", lambda path, force_renew=False: None)

    with pytest.raises(ValueError, match="no teams were matched"):
        tu.update_from_testing_sessions(
            year=2026,
            events=["Bahrain Grand Prix"],
            data_dir=str(tmp_path / "processed"),
            dry_run=True,
        )


def test_update_from_testing_sessions_writes_file_when_not_dry_run(tmp_path, patcher):
    original_last_updated = "2026-01-01T00:00:00"
    _write_characteristics(
        tmp_path,
        {
            "version": 2,
            "last_updated": original_last_updated,
            "teams": {
                "Ferrari": {
                    "directionality": {
                        "max_speed": 0.0,
                        "slow_corner_speed": 0.0,
                        "medium_corner_speed": 0.0,
                        "high_corner_speed": 0.0,
                    },
                    "testing_characteristics": {},
                    "compound_characteristics": {},
                }
            },
        },
    )

    session = SimpleNamespace(
        laps=pd.DataFrame({"Team": ["Ferrari"], "LapTime": [pd.to_timedelta("0:01:30")]})
    )

    patcher.setattr(tu, "_load_sessions_for_event", lambda **kwargs: [("FP1", session)])
    patcher.setattr(
        tu, "_collect_session_metrics", lambda **kwargs: ({"Ferrari": {"overall_pace": 0.7}}, {})
    )
    patcher.setattr(
        tu, "_count_team_selected_laps", lambda session, known_teams, run_profile: {"Ferrari": 10.0}
    )
    patcher.setattr(tu, "extract_compound_metrics", lambda team_laps, canonical_team, race_name: {})
    patcher.setattr(tu.fastf1.Cache, "enable_cache", lambda path, force_renew=False: None)

    atomic_write = MagicMock()
    patcher.setattr(tu, "atomic_json_write", atomic_write)

    summary = tu.update_from_testing_sessions(
        year=2026,
        events=["Bahrain Grand Prix"],
        data_dir=str(tmp_path / "processed"),
        dry_run=False,
    )

    assert summary["updated_teams"] == ["Ferrari"]
    atomic_write.assert_called_once()
    written_payload = atomic_write.call_args.args[1]
    assert written_payload["version"] == 3
    assert written_payload["last_updated"] != original_last_updated
