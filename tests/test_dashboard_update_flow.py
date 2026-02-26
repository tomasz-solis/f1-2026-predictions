"""Tests for dashboard race/practice auto-update orchestration."""

import json
from types import SimpleNamespace

import pytest

from src.dashboard import update_flow


class _ProgressBar:
    def __init__(self):
        self.values: list[float] = []
        self.was_cleared = False

    def progress(self, value: float) -> None:
        self.values.append(value)

    def empty(self) -> None:
        self.was_cleared = True


class _StatusText:
    def __init__(self):
        self.messages: list[str] = []
        self.was_cleared = False

    def text(self, message: str) -> None:
        self.messages.append(message)

    def empty(self) -> None:
        self.was_cleared = True


def _stub_streamlit(patcher):
    calls: list[tuple[str, str]] = []
    progress_bar = _ProgressBar()
    status_text = _StatusText()
    cache_calls: list[str] = []

    patcher.setattr(update_flow.st, "info", lambda msg: calls.append(("info", str(msg))))
    patcher.setattr(update_flow.st, "success", lambda msg: calls.append(("success", str(msg))))
    patcher.setattr(update_flow.st, "warning", lambda msg: calls.append(("warning", str(msg))))
    patcher.setattr(update_flow.st, "progress", lambda _initial=0: progress_bar)
    patcher.setattr(update_flow.st, "empty", lambda: status_text)
    patcher.setattr(
        update_flow.st,
        "cache_resource",
        SimpleNamespace(clear=lambda: cache_calls.append("resource")),
    )
    patcher.setattr(
        update_flow.st,
        "cache_data",
        SimpleNamespace(clear=lambda: cache_calls.append("data")),
    )

    return calls, progress_bar, status_text, cache_calls


def test_auto_update_if_needed_skips_when_no_new_races(patcher):
    calls, progress_bar, status_text, cache_calls = _stub_streamlit(patcher)

    patcher.setattr("src.utils.auto_updater.needs_update", lambda: (False, []))
    patcher.setattr(
        "src.utils.auto_updater.auto_update_from_races",
        lambda _callback=None: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    update_flow.auto_update_if_needed()

    assert calls == []
    assert progress_bar.values == []
    assert status_text.messages == []
    assert cache_calls == []


def test_auto_update_if_needed_runs_update_and_clears_cache(patcher):
    calls, progress_bar, status_text, cache_calls = _stub_streamlit(patcher)

    patcher.setattr(
        "src.utils.auto_updater.needs_update",
        lambda: (True, ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]),
    )

    def _auto_update(progress_callback):
        progress_callback(1, 2, "Learning race 1")
        progress_callback(2, 2, "Learning race 2")
        return 2

    patcher.setattr("src.utils.auto_updater.auto_update_from_races", _auto_update)

    update_flow.auto_update_if_needed()

    assert ("info", "Found 2 new race(s) to learn from. Updating characteristics...") in calls
    assert ("success", "Learned from 2 race(s). Predictions now use updated data.") in calls
    assert progress_bar.values == [0.5, 1.0]
    assert status_text.messages == ["Learning race 1", "Learning race 2"]
    assert progress_bar.was_cleared is True
    assert status_text.was_cleared is True
    assert cache_calls == ["resource", "data"]


def test_auto_update_if_needed_raises_when_update_is_incomplete(patcher):
    calls, _progress_bar, _status_text, cache_calls = _stub_streamlit(patcher)

    patcher.setattr("src.utils.auto_updater.needs_update", lambda: (True, ["Bahrain Grand Prix"]))
    patcher.setattr("src.utils.auto_updater.auto_update_from_races", lambda _callback: 0)

    with pytest.raises(RuntimeError, match="Race refresh incomplete"):
        update_flow.auto_update_if_needed()

    assert ("info", "Found 1 new race(s) to learn from. Updating characteristics...") in calls
    assert cache_calls == []


def test_auto_update_if_needed_force_recheck_passes_explicit_race_list(patcher):
    _stub_streamlit(patcher)
    seen_force_recheck: list[bool] = []
    captured_races: list[str] = []

    def _needs_update(force_recheck=False):
        seen_force_recheck.append(force_recheck)
        return True, ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]

    def _auto_update_from_races(progress_callback=None, races_to_update=None):
        del progress_callback
        captured_races.extend(races_to_update or [])
        return len(races_to_update or [])

    patcher.setattr("src.utils.auto_updater.needs_update", _needs_update)
    patcher.setattr("src.utils.auto_updater.auto_update_from_races", _auto_update_from_races)

    update_flow.auto_update_if_needed(force_recheck=True)

    assert seen_force_recheck == [True]
    assert captured_races == ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]


def test_auto_update_if_needed_passes_year_to_updater_dependencies(patcher):
    _stub_streamlit(patcher)
    seen_years: list[int] = []
    seen_update_years: list[int] = []

    def _needs_update(year=2026, force_recheck=False):
        del force_recheck
        seen_years.append(year)
        return True, ["Bahrain Grand Prix"]

    def _auto_update_from_races(progress_callback=None, races_to_update=None, year=2026):
        del progress_callback, races_to_update
        seen_update_years.append(year)
        return 1

    patcher.setattr("src.utils.auto_updater.needs_update", _needs_update)
    patcher.setattr("src.utils.auto_updater.auto_update_from_races", _auto_update_from_races)

    update_flow.auto_update_if_needed(year=2027)

    assert seen_years == [2027]
    assert seen_update_years == [2027]


def test_load_practice_update_state_handles_invalid_json(patcher, tmp_path):
    state_file = tmp_path / "practice_state.json"
    state_file.write_text("{invalid json")
    patcher.setattr(update_flow, "_PRACTICE_UPDATE_STATE_FILE", state_file)

    assert update_flow._load_practice_update_state() == {"races": {}}


def test_auto_update_practice_characteristics_no_completed_fp(patcher, tmp_path):
    state_file = tmp_path / "practice_state.json"
    patcher.setattr(update_flow, "_PRACTICE_UPDATE_STATE_FILE", state_file)

    class _Detector:
        def get_completed_sessions(self, year: int, race_name: str, is_sprint: bool):
            return []

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)

    result = update_flow.auto_update_practice_characteristics_if_needed(
        year=2026,
        race_name="Bahrain Grand Prix",
        is_sprint=False,
    )

    assert result == {"updated": False, "completed_fp_sessions": []}


def test_auto_update_practice_characteristics_skips_if_sessions_already_processed(
    patcher,
    tmp_path,
):
    state_file = tmp_path / "practice_state.json"
    state_file.write_text(
        json.dumps(
            {
                "races": {
                    "2026::Bahrain Grand Prix": {
                        "sessions": ["FP1", "FP2"],
                    }
                }
            }
        )
    )
    patcher.setattr(update_flow, "_PRACTICE_UPDATE_STATE_FILE", state_file)

    class _Detector:
        def get_completed_sessions(self, year: int, race_name: str, is_sprint: bool):
            return ["FP2", "FP1"]

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)
    patcher.setattr(
        "src.systems.testing_updater.update_from_testing_sessions",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not update")),
    )

    result = update_flow.auto_update_practice_characteristics_if_needed(
        year=2026,
        race_name="Bahrain Grand Prix",
        is_sprint=False,
    )

    assert result == {"updated": False, "completed_fp_sessions": ["FP1", "FP2"]}


def test_auto_update_practice_characteristics_updates_state(patcher, tmp_path):
    state_file = tmp_path / "practice_state.json"
    patcher.setattr(update_flow, "_PRACTICE_UPDATE_STATE_FILE", state_file)

    class _Detector:
        def get_completed_sessions(self, year: int, race_name: str, is_sprint: bool):
            return ["FP2", "FP1"]

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)

    config_values = {
        "baseline_predictor.practice_capture.new_weight": 0.4,
        "baseline_predictor.practice_capture.directionality_scale": 0.09,
        "baseline_predictor.practice_capture.session_aggregation": "laps_weighted",
        "baseline_predictor.practice_capture.run_profile": "balanced",
    }
    patcher.setattr(
        "src.utils.config_loader.get",
        lambda key, default=None: config_values.get(key, default),
    )

    captured_kwargs: dict = {}

    def _update_from_testing_sessions(**kwargs):
        captured_kwargs.update(kwargs)
        return {"updated_teams": ["Ferrari", "McLaren", "Mercedes"]}

    patcher.setattr(
        "src.systems.testing_updater.update_from_testing_sessions",
        _update_from_testing_sessions,
    )

    result = update_flow.auto_update_practice_characteristics_if_needed(
        year=2026,
        race_name="Bahrain Grand Prix",
        is_sprint=False,
    )

    assert result["updated"] is True
    assert result["completed_fp_sessions"] == ["FP1", "FP2"]
    assert result["teams_updated"] == 3
    assert captured_kwargs["events"] == ["Bahrain Grand Prix"]
    assert captured_kwargs["sessions"] == ["FP1", "FP2"]
    assert captured_kwargs["new_weight"] == 0.4
    assert captured_kwargs["directionality_scale"] == 0.09

    persisted = json.loads(state_file.read_text())
    race_state = persisted["races"]["2026::Bahrain Grand Prix"]
    assert race_state["sessions"] == ["FP1", "FP2"]
    assert race_state["teams_updated"] == 3
