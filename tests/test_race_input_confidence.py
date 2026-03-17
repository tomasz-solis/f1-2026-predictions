"""Tests for race-input confidence calibration helpers."""

import pytest

from src.utils.race_input_confidence import cap_predicted_main_race_input_confidence


class _ConfigStub:
    """Small config stub that mimics the project's dot-path lookup API."""

    def __init__(self, values: dict[str, object]):
        """Store a flat mapping of config keys to test values."""
        self.values = values

    def get(self, key: str, default=None):
        """Return the configured value for a dot-path key."""
        return self.values.get(key, default)


def test_cap_predicted_main_race_input_confidence_limits_sprint_predicted_main_grid():
    """Sprint-weekend Sunday race runs should cap confidence before main qualifying is real."""
    capped = cap_predicted_main_race_input_confidence(
        0.8,
        qualifying_result={"qualifying_stage": "main"},
        grid_source="PREDICTED",
        is_sprint_weekend=True,
    )

    assert capped == pytest.approx(0.55)


def test_cap_predicted_main_race_input_confidence_uses_checkpoint_override():
    """Checkpoint-specific caps should allow SQ and Sprint to calibrate separately."""
    capped = cap_predicted_main_race_input_confidence(
        0.8,
        qualifying_result={"qualifying_stage": "main"},
        grid_source="PREDICTED",
        is_sprint_weekend=True,
        boundary_session_name="SQ",
        config=_ConfigStub(
            {
                "baseline_predictor.race.main_race_predicted_grid_sprint_confidence_cap": 0.55,
                "baseline_predictor.race.main_race_predicted_grid_sprint_confidence_caps_by_checkpoint": {
                    "SQ": 0.5,
                    "SPRINT": 0.55,
                },
            }
        ),
    )

    assert capped == pytest.approx(0.5)


def test_cap_predicted_main_race_input_confidence_leaves_actual_grid_untouched():
    """Actual qualifying results should keep their full confidence."""
    unchanged = cap_predicted_main_race_input_confidence(
        0.8,
        qualifying_result={"qualifying_stage": "main"},
        grid_source="ACTUAL",
        is_sprint_weekend=True,
    )

    assert unchanged == pytest.approx(0.8)


def test_cap_predicted_main_race_input_confidence_skips_normal_weekends():
    """Normal weekends should not inherit sprint-specific skepticism."""
    unchanged = cap_predicted_main_race_input_confidence(
        0.8,
        qualifying_result={"qualifying_stage": "main"},
        grid_source="PREDICTED",
        is_sprint_weekend=False,
    )

    assert unchanged == pytest.approx(0.8)
