"""Focused tests for the historical replay helper rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.accuracy_targets import TARGET_GRAND_PRIX_RACE, TARGET_SPRINT_QUALIFYING
from src.utils.historical_replay import (
    HistoricalReplaySummary,
    ReplayConfigOverride,
    _apply_session_update,
    _reset_replay_artifacts,
    _resolve_race_section_for_replay,
    apply_target_scoring_policy,
    checkpoint_sequence_for_weekend,
    session_is_available_at_checkpoint,
)


def test_checkpoint_sequence_for_weekend_matches_requested_outputs():
    """Normal and sprint weekends should expose the expected replay checkpoints."""
    assert checkpoint_sequence_for_weekend(False) == ("PRE", "FP1", "FP2", "FP3")
    assert checkpoint_sequence_for_weekend(True) == ("PRE", "FP1", "SQ")


def test_session_is_available_at_checkpoint_respects_weekend_order():
    """Historical replay should only surface sessions that had already happened."""
    assert session_is_available_at_checkpoint("PRE", "Q") is False
    assert session_is_available_at_checkpoint("FP1", "SQ") is False
    assert session_is_available_at_checkpoint("SQ", "SQ") is True
    assert session_is_available_at_checkpoint("SPRINT", "SQ") is True
    assert session_is_available_at_checkpoint("Q", "R") is False


def test_apply_target_scoring_policy_marks_excluded_targets_unscored():
    """Excluded targets should stay in the payload but drop out of scoring."""
    adjusted = apply_target_scoring_policy(
        {
            TARGET_SPRINT_QUALIFYING: {
                "target_session": "SQ",
                "eligible_at_save": True,
            },
            TARGET_GRAND_PRIX_RACE: {
                "target_session": "R",
                "eligible_at_save": True,
            },
        },
        excluded_scoring_targets={TARGET_SPRINT_QUALIFYING},
    )

    assert adjusted[TARGET_SPRINT_QUALIFYING]["eligible_at_save"] is False
    assert adjusted[TARGET_GRAND_PRIX_RACE]["eligible_at_save"] is True


def test_replay_config_override_blocks_saved_actual_inference():
    """Replay config should be able to disable prediction-log feedback cleanly."""

    class _BaseConfig:
        def get(self, key, default=None):
            if key == "baseline_predictor.current_season_form.infer_from_saved_actuals":
                return True
            if key == "other.key":
                return "base-value"
            return default

    config = ReplayConfigOverride(
        base_config=_BaseConfig(),
        overrides={"baseline_predictor.current_season_form.infer_from_saved_actuals": False},
    )

    assert config.get("baseline_predictor.current_season_form.infer_from_saved_actuals") is False
    assert config.get("other.key") == "base-value"


def test_historical_replay_summary_tracks_driver_update_trace_report() -> None:
    """Replay summaries should point at the separate driver update trace report."""
    summary = HistoricalReplaySummary(
        year=2026,
        output_root="data/historical_replay",
        processed_data_dir="data/historical_replay/processed",
        excluded_scoring_targets=[],
        driver_update_trace_path="data/historical_replay/reports/driver_update_trace.json",
    )

    assert summary.to_dict()["driver_update_trace_path"].endswith("driver_update_trace.json")


def test_reset_replay_artifacts_reports_missing_preseason_baseline(tmp_path: Path) -> None:
    """A missing flat pre-season baseline should raise an actionable error."""
    processed_dir = tmp_path / "processed"
    (processed_dir / "car_characteristics").mkdir(parents=True)
    (processed_dir / "car_characteristics" / "2026_car_characteristics.json").write_text("{}")
    # No flat driver_characteristics.json is present.

    with pytest.raises(FileNotFoundError, match="pre-season driver baseline"):
        _reset_replay_artifacts(processed_dir, year=2026)


def _grid(*drivers: str) -> list[dict]:
    return [
        {"driver": driver, "team": "McLaren", "position": position}
        for position, driver in enumerate(drivers, start=1)
    ]


def _race_section_kwargs(**overrides):
    kwargs = {
        "predictor": None,
        "year": 2026,
        "race_name": "Belgian Grand Prix",
        "checkpoint_session": "FP2",
        "target_session": "R",
        "qualifying_grid": _grid("NOR", "VER", "PIA"),
        "qualifying_grid_source": "ACTUAL",
        "grid_session_name": "Q",
        "weather": "dry",
        "input_confidence": 1.0,
        "actual_cache": {},
    }
    kwargs.update(overrides)
    return kwargs


def test_replayed_race_starts_from_the_actual_grid_not_the_qualifying_classification(monkeypatch):
    """A ten-place penalty must move the driver in the replay, not just in the record."""
    penalised_grid = _grid("VER", "PIA", "NOR")
    monkeypatch.setattr(
        "src.data.actual_results_fetcher.fetch_actual_starting_grid",
        lambda year, race_name: penalised_grid,
    )
    seen: dict[str, list[dict]] = {}

    class _Predictor:
        def predict_race(self, *, qualifying_grid, **_kwargs):
            seen["grid"] = qualifying_grid
            return {"finish_order": []}

    _resolve_race_section_for_replay(**_race_section_kwargs(predictor=_Predictor()))

    assert [row["driver"] for row in seen["grid"]] == ["VER", "PIA", "NOR"]


def test_a_replay_before_qualifying_keeps_predicting_the_grid(monkeypatch):
    """Before qualifying the penalty is not known, so substituting it would leak."""

    def _fail(year, race_name):
        raise AssertionError("must not read the actual grid before qualifying")

    monkeypatch.setattr("src.data.actual_results_fetcher.fetch_actual_starting_grid", _fail)
    seen: dict[str, list[dict]] = {}

    class _Predictor:
        def predict_race(self, *, qualifying_grid, **_kwargs):
            seen["grid"] = qualifying_grid
            return {"finish_order": []}

    _resolve_race_section_for_replay(
        **_race_section_kwargs(predictor=_Predictor(), qualifying_grid_source="PREDICTED")
    )

    assert [row["driver"] for row in seen["grid"]] == ["NOR", "VER", "PIA"]


def test_an_unusable_practice_session_is_skipped_not_fatal(monkeypatch, tmp_path):
    """FastF1 publishes 2026 Barcelona FP1 with no team names; the weekend still replays."""

    def _no_usable_telemetry(**_kwargs):
        raise ValueError("no usable team telemetry could be extracted yet")

    monkeypatch.setattr(
        "src.utils.historical_replay.update_from_testing_sessions", _no_usable_telemetry
    )

    applied = _apply_session_update(
        year=2026,
        event_name="Barcelona Grand Prix",
        session_name="FP1",
        cache_dirs=["data/raw/.fastf1_cache"],
        processed_dir=tmp_path,
    )

    assert applied is False


def test_an_unusable_competitive_session_still_fails_closed(monkeypatch, tmp_path):
    """A scored result built on missing data is not a replay."""

    def _no_usable_telemetry(**_kwargs):
        raise ValueError("no usable team telemetry could be extracted yet")

    monkeypatch.setattr(
        "src.utils.historical_replay.update_from_testing_sessions", _no_usable_telemetry
    )

    with pytest.raises(ValueError, match="Could not replay"):
        _apply_session_update(
            year=2026,
            event_name="Barcelona Grand Prix",
            session_name="Q",
            cache_dirs=["data/raw/.fastf1_cache"],
            processed_dir=tmp_path,
        )


def test_a_testing_day_still_fails_closed(monkeypatch, tmp_path):
    """Testing seeds the season, so a missing day must not pass silently."""

    def _no_usable_telemetry(**_kwargs):
        raise ValueError("no usable team telemetry could be extracted yet")

    monkeypatch.setattr(
        "src.utils.historical_replay.update_from_testing_sessions", _no_usable_telemetry
    )

    with pytest.raises(ValueError, match="Could not replay"):
        _apply_session_update(
            year=2026,
            event_name="Pre-Season Testing",
            session_name="Day 1",
            cache_dirs=["data/raw/.fastf1_cache"],
            processed_dir=tmp_path,
        )
