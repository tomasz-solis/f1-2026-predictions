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
    good_session.load = MagicMock(return_value=None)

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
    good_session.load.assert_called_once_with(
        laps=True,
        telemetry=True,
        weather=False,
        messages=False,
    )


def test_load_sessions_for_non_testing_event_collects_errors(patcher):
    session = SimpleNamespace(laps=pd.DataFrame({"LapTime": [pd.to_timedelta("0:01:30")]}))
    session.load = MagicMock(return_value=None)

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
    session.load.assert_called_once_with(
        laps=True,
        telemetry=True,
        weather=False,
        messages=False,
    )


def test_load_sessions_for_non_testing_event_skips_incomplete_laps(patcher):
    class BrokenSession:
        def load(self, **_kwargs):
            return None

        @property
        def laps(self):
            raise RuntimeError("DataNotLoadedError")

    good_session = SimpleNamespace(laps=pd.DataFrame({"LapTime": [pd.to_timedelta("0:01:30")]}))
    good_session.load = lambda **kwargs: None

    def _mock_get_session(year, event_name, session_name):
        if session_name == "FP1":
            return BrokenSession()
        return good_session

    patcher.setattr(tu.fastf1, "get_session", _mock_get_session)

    errors = []
    loaded = tu._load_sessions_for_event(
        year=2026,
        event_name="Australian Grand Prix",
        session_candidates=["FP1", "FP2"],
        error_messages=errors,
    )

    assert len(loaded) == 1
    assert loaded[0][0] == "FP2"
    assert errors and "Australian Grand Prix::FP1" in errors[0]


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


def test_extract_session_driver_deltas_tracks_teammate_form():
    session = SimpleNamespace(
        laps=pd.DataFrame(
            {
                "Team": ["Mercedes"] * 8,
                "Driver": ["RUS"] * 4 + ["ANT"] * 4,
                "LapTime": [
                    pd.to_timedelta("0:01:31.000"),
                    pd.to_timedelta("0:01:31.100"),
                    pd.to_timedelta("0:01:31.200"),
                    pd.to_timedelta("0:01:31.100"),
                    pd.to_timedelta("0:01:31.500"),
                    pd.to_timedelta("0:01:31.600"),
                    pd.to_timedelta("0:01:31.700"),
                    pd.to_timedelta("0:01:31.600"),
                ],
                "Stint": [1] * 8,
                "Compound": ["SOFT"] * 8,
            }
        )
    )

    deltas = tu._extract_session_driver_deltas(
        session=session,
        known_teams={"Mercedes"},
        run_profile="balanced",
    )

    assert deltas["Mercedes"]["RUS"] == pytest.approx(-0.25)
    assert deltas["Mercedes"]["ANT"] == pytest.approx(0.25)


def test_metric_helpers_and_payload_extraction():
    assert tu._median_timedelta_seconds(pd.Series([], dtype=object)) is None
    assert tu._median_lap_seconds(pd.DataFrame()) is None
    assert tu._normalize_lower_better({"A": 1.0, "B": 1.0}) == {"A": 0.5, "B": 0.5}
    assert tu._normalize_lower_better({"A": 90.0, "B": 91.0, "C": 120.0}) == {
        "A": 1.0,
        "B": 0.5,
        "C": 0.0,
    }

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
        tu,
        "_extract_session_driver_deltas",
        lambda **kwargs: {"Ferrari": {"LEC": -0.11, "HAM": 0.11}},
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
    assert summary["snapshots_written"] == 1

    snapshot_path = (
        tmp_path / "car_characteristics_snapshot" / "2026" / "bahrain_grand_prix" / "fp1.json"
    )
    snapshot_payload = json.loads(snapshot_path.read_text())
    assert snapshot_payload["event_name"] == "Bahrain Grand Prix"
    assert snapshot_payload["session_name"] == "FP1"
    assert snapshot_payload["teams"]["Ferrari"]["profiles"]["balanced"]["overall_pace"] == 0.7
    assert snapshot_payload["teams"]["Ferrari"]["driver_deltas_seconds"]["balanced"][
        "LEC"
    ] == pytest.approx(-0.11)


def test_update_from_testing_sessions_persists_to_artifact_store_when_db_enabled(tmp_path, patcher):
    _write_characteristics(
        tmp_path,
        {
            "version": 2,
            "teams": {
                "Ferrari": {
                    "directionality": {
                        "max_speed": 0.0,
                        "slow_corner_speed": 0.0,
                        "medium_corner_speed": 0.0,
                        "high_corner_speed": 0.0,
                    },
                    "testing_characteristics": {},
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
        tu,
        "_extract_session_driver_deltas",
        lambda **kwargs: {"Ferrari": {"LEC": -0.11, "HAM": 0.11}},
    )
    patcher.setattr(
        tu, "_count_team_selected_laps", lambda session, known_teams, run_profile: {"Ferrari": 10.0}
    )
    patcher.setattr(tu, "extract_compound_metrics", lambda team_laps, canonical_team, race_name: {})
    patcher.setattr(tu.fastf1.Cache, "enable_cache", lambda path, force_renew=False: None)
    patcher.setattr(tu, "should_write_to_db", lambda: True)

    atomic_write = MagicMock()
    patcher.setattr(tu, "atomic_json_write", atomic_write)

    store_instance = MagicMock()
    store_instance.get_latest_version.return_value = 5
    store_instance.save_artifact.return_value = {"version": 6}
    artifact_store_ctor = MagicMock(return_value=store_instance)
    patcher.setattr(tu, "ArtifactStore", artifact_store_ctor)

    summary = tu.update_from_testing_sessions(
        year=2026,
        events=["Bahrain Grand Prix"],
        data_dir=str(tmp_path / "processed"),
        dry_run=False,
    )

    assert summary["updated_teams"] == ["Ferrari"]
    atomic_write.assert_called_once()
    written_payload = atomic_write.call_args.args[1]
    assert written_payload["version"] == 6
    store_instance.get_latest_version.assert_called_once_with(
        "car_characteristics", "2026::car_characteristics"
    )
    assert store_instance.save_artifact.call_count == 2
    first_call = store_instance.save_artifact.call_args_list[0].kwargs
    assert first_call["artifact_type"] == "car_characteristics"
    assert first_call["artifact_key"] == "2026::car_characteristics"
    assert first_call["version"] == 6

    second_call = store_instance.save_artifact.call_args_list[1].kwargs
    assert second_call["artifact_type"] == "car_characteristics_snapshot"
    assert second_call["artifact_key"] == "2026::Bahrain Grand Prix::FP1"
    # Snapshots auto-increment (version=None) so a re-extraction supersedes the latest
    # version the reader (load_artifact version="latest") picks up.
    assert second_call["version"] is None


def test_backfill_session_snapshot_history_writes_only_snapshots(tmp_path, patcher):
    original_payload = {
        "version": 7,
        "last_updated": "2026-03-03T23:33:59",
        "teams": {
            "Ferrari": {
                "directionality": {
                    "max_speed": 0.0,
                    "slow_corner_speed": 0.0,
                    "medium_corner_speed": 0.0,
                    "high_corner_speed": 0.0,
                },
                "testing_characteristics": {},
            }
        },
    }
    characteristics_file = _write_characteristics(tmp_path, original_payload)

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
    patcher.setattr(tu.fastf1.Cache, "enable_cache", lambda path, force_renew=False: None)

    summary = tu.backfill_session_snapshot_history(
        year=2026,
        events=["Bahrain Grand Prix"],
        data_dir=str(tmp_path / "processed"),
        dry_run=False,
    )

    assert summary["snapshots_written"] == 1
    reloaded_payload = json.loads(characteristics_file.read_text())
    assert reloaded_payload == original_payload

    snapshot_path = (
        tmp_path / "car_characteristics_snapshot" / "2026" / "bahrain_grand_prix" / "fp1.json"
    )
    assert snapshot_path.exists()


def test_season_snapshot_plan_discovers_cached_events_in_calendar_order(tmp_path, patcher):
    testing_cache = tmp_path / "cache-testing"
    race_cache = tmp_path / "cache-race"

    testing_session_dirs = [
        testing_cache / "2026" / "2026-02-13_Pre-Season_Testing" / "2026-02-11_Day_1",
        testing_cache / "2026" / "2026-02-20_Pre-Season_Testing" / "2026-02-18_Day_1",
    ]
    race_session_dirs = [
        race_cache / "2026" / "2026-03-08_Australian_Grand_Prix" / "2026-03-06_Practice_1",
        race_cache / "2026" / "2026-03-08_Australian_Grand_Prix" / "2026-03-07_Qualifying",
        race_cache / "2026" / "2026-03-15_Chinese_Grand_Prix" / "2026-03-13_Practice_1",
        race_cache / "2026" / "2026-03-15_Chinese_Grand_Prix" / "2026-03-13_Sprint_Qualifying",
        race_cache / "2026" / "2026-03-15_Chinese_Grand_Prix" / "2026-03-14_Sprint",
        race_cache / "2026" / "2026-03-15_Chinese_Grand_Prix" / "2026-03-14_Qualifying",
        race_cache / "2026" / "2026-03-15_Chinese_Grand_Prix" / "2026-03-15_Race",
    ]
    empty_placeholder_dir = (
        race_cache / "2026" / "2026-03-29_Japanese_Grand_Prix" / "2026-03-27_Practice_1"
    )

    for directory in [*testing_session_dirs, *race_session_dirs, empty_placeholder_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for directory in [*testing_session_dirs, *race_session_dirs]:
        (directory / "session_info.ff1pkl").write_text("cached")

    patcher.setattr(tu, "_DEFAULT_TESTING_CACHE_DIR", testing_cache)
    patcher.setattr(tu, "_DEFAULT_RACE_CACHE_DIR", race_cache)

    plan = tu._season_snapshot_plan(2026)

    assert [entry["event_name"] for entry in plan[:4]] == [
        "Testing 1",
        "Testing 2",
        "Australian Grand Prix",
        "Chinese Grand Prix",
    ]
    assert plan[2]["sessions"] == ["FP1", "Q"]
    assert plan[3]["sessions"] == ["FP1", "SQ", "Sprint", "Q", "R"]
    assert all(entry["event_name"] != "Japanese Grand Prix" for entry in plan)


def test_backfill_season_snapshot_history_builds_testing_and_race_plan(tmp_path, patcher):
    _write_characteristics(
        tmp_path,
        {
            "version": 7,
            "teams": {
                "Ferrari": {
                    "directionality": {
                        "max_speed": 0.0,
                        "slow_corner_speed": 0.0,
                        "medium_corner_speed": 0.0,
                        "high_corner_speed": 0.0,
                    },
                    "testing_characteristics": {},
                }
            },
        },
    )

    captured_calls = []

    def _mock_backfill(**kwargs):
        captured_calls.append(kwargs)
        return {
            "loaded_sessions": [
                f"{kwargs['events'][0]}::{session}" for session in kwargs["sessions"]
            ],
            "snapshot_keys": [
                f"2026::{kwargs['events'][0]}::{session}" for session in kwargs["sessions"]
            ],
        }

    patcher.setattr(tu, "backfill_session_snapshot_history", _mock_backfill)
    patcher.setattr(
        tu,
        "_season_snapshot_plan",
        lambda year: [
            {
                "event_name": "Testing 1",
                "sessions": ["Day 1", "Day 2", "Day 3"],
                "cache_dirs": [str(tmp_path / "cache-a")],
            },
            {
                "event_name": "Testing 2",
                "sessions": ["Day 1", "Day 2", "Day 3"],
                "cache_dirs": [str(tmp_path / "cache-a")],
            },
            {
                "event_name": "Australian Grand Prix",
                "sessions": ["FP1", "FP2", "FP3", "Q", "R"],
                "cache_dirs": [str(tmp_path / "cache-a")],
            },
            {
                "event_name": "Chinese Grand Prix",
                "sessions": ["FP1", "SQ", "Sprint", "Q", "R"],
                "cache_dirs": [str(tmp_path / "cache-a")],
            },
        ],
    )

    summary = tu.backfill_season_snapshot_history(
        year=2026,
        data_dir=str(tmp_path / "processed"),
        dry_run=False,
    )

    assert "Testing 1" in summary["events_processed"]
    assert "Testing 2" in summary["events_processed"]
    assert "Australian Grand Prix" in summary["events_processed"]
    assert "Chinese Grand Prix" in summary["events_processed"]
    australian_calls = [
        call for call in captured_calls if call["events"] == ["Australian Grand Prix"]
    ]
    chinese_calls = [call for call in captured_calls if call["events"] == ["Chinese Grand Prix"]]
    assert australian_calls[0]["sessions"] == ["FP1", "FP2", "FP3", "Q", "R"]
    assert chinese_calls[0]["sessions"] == ["FP1", "SQ", "Sprint", "Q", "R"]


def test_replay_season_characteristics_from_cache_resets_then_replays_in_order(tmp_path, patcher):
    characteristics_file = _write_characteristics(
        tmp_path,
        {
            "version": 4,
            "directionality_last_updated": "2026-03-03T23:33:59.825095",
            "directionality_meta": {"events": ["Testing 1"]},
            "teams": {
                "Ferrari": {
                    "overall_performance": 0.72,
                    "directionality": {"max_speed": 0.03},
                    "testing_characteristics": {"overall_pace": 0.81},
                    "testing_characteristics_profiles": {"balanced": {"overall_pace": 0.81}},
                    "compound_characteristics": {"SOFT": {"pace": 0.8}},
                }
            },
        },
    )

    update_calls: list[tuple[list[str], list[str], str]] = []
    persisted_state: dict[str, object] = {}

    def _mock_update(**kwargs):
        payload = json.loads(characteristics_file.read_text())
        ferrari = payload["teams"]["Ferrari"]
        assert "directionality" not in ferrari
        assert "testing_characteristics" not in ferrari
        assert "testing_characteristics_profiles" not in ferrari
        assert "compound_characteristics" not in ferrari
        update_calls.append(
            (list(kwargs["events"]), list(kwargs["sessions"]), str(kwargs["cache_dir"]))
        )
        session_name = kwargs["sessions"][0]
        event_name = kwargs["events"][0]
        return {
            "loaded_sessions": [f"{event_name}::{session_name}"],
            "snapshot_keys": [f"2026::{event_name}::{session_name}"],
            "updated_teams": ["Ferrari"],
        }

    patcher.setattr(
        tu,
        "_season_snapshot_plan",
        lambda year: [
            {
                "event_name": "Testing 1",
                "sessions": ["Day 1", "Day 2"],
                "cache_dirs": [str(tmp_path / "cache-testing")],
            },
            {
                "event_name": "Australian Grand Prix",
                "sessions": ["FP1", "Q"],
                "cache_dirs": [str(tmp_path / "cache-race")],
            },
        ],
    )
    patcher.setattr(tu, "update_from_testing_sessions", _mock_update)
    patcher.setattr(
        tu,
        "_write_practice_replay_state",
        lambda year, event_sessions, event_team_counts: persisted_state.update(
            {
                "year": year,
                "event_sessions": event_sessions,
                "event_team_counts": event_team_counts,
            }
        ),
    )

    summary = tu.replay_season_characteristics_from_cache(
        year=2026,
        data_dir=str(tmp_path / "processed"),
        dry_run=False,
    )

    assert update_calls == [
        (["Testing 1"], ["Day 1"], str(tmp_path / "cache-testing")),
        (["Testing 1"], ["Day 2"], str(tmp_path / "cache-testing")),
        (["Australian Grand Prix"], ["FP1"], str(tmp_path / "cache-race")),
        (["Australian Grand Prix"], ["Q"], str(tmp_path / "cache-race")),
    ]
    assert persisted_state == {
        "year": 2026,
        "event_sessions": {
            "Testing 1": ["Day 1", "Day 2"],
            "Australian Grand Prix": ["FP1", "Q"],
        },
        "event_team_counts": {"Testing 1": 1, "Australian Grand Prix": 1},
    }
    assert summary["events_processed"] == ["Testing 1", "Australian Grand Prix"]
    assert summary["sessions_applied"] == [
        "Testing 1::Day 1",
        "Testing 1::Day 2",
        "Australian Grand Prix::FP1",
        "Australian Grand Prix::Q",
    ]
    assert Path(summary["backup_path"]).exists()


def test_replay_season_characteristics_from_cache_restores_backup_after_failure(tmp_path, patcher):
    original_payload = {
        "version": 4,
        "directionality_last_updated": "2026-03-03T23:33:59.825095",
        "directionality_meta": {"events": ["Testing 1"]},
        "teams": {
            "Ferrari": {
                "overall_performance": 0.72,
                "directionality": {"max_speed": 0.03},
                "testing_characteristics": {"overall_pace": 0.81},
                "testing_characteristics_profiles": {"balanced": {"overall_pace": 0.81}},
                "compound_characteristics": {"SOFT": {"pace": 0.8}},
            }
        },
    }
    characteristics_file = _write_characteristics(tmp_path, original_payload)

    patcher.setattr(
        tu,
        "_season_snapshot_plan",
        lambda year: [
            {
                "event_name": "Australian Grand Prix",
                "sessions": ["FP1", "Q"],
                "cache_dirs": [str(tmp_path / "cache-race")],
            }
        ],
    )

    call_count = {"value": 0}

    def _mock_update(**kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "loaded_sessions": ["Australian Grand Prix::FP1"],
                "snapshot_keys": ["2026::Australian Grand Prix::FP1"],
                "updated_teams": ["Ferrari"],
            }
        raise RuntimeError("replay step failed")

    patcher.setattr(tu, "update_from_testing_sessions", _mock_update)

    with pytest.raises(RuntimeError, match="replay step failed"):
        tu.replay_season_characteristics_from_cache(
            year=2026,
            data_dir=str(tmp_path / "processed"),
            dry_run=False,
        )

    restored_payload = json.loads(characteristics_file.read_text())
    assert restored_payload == original_payload
