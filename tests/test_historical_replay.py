"""Focused tests for the historical replay helper rules."""

from __future__ import annotations

from src.utils.accuracy_targets import TARGET_GRAND_PRIX_RACE, TARGET_SPRINT_QUALIFYING
from src.utils.historical_replay import (
    HistoricalReplaySummary,
    ReplayConfigOverride,
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
