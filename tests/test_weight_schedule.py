from __future__ import annotations

import pytest

from src.systems.weight_schedule import (
    calculate_blended_performance,
    get_recommended_schedule,
    get_schedule_weights,
)


def test_get_schedule_weights_validates_inputs():
    with pytest.raises(ValueError, match="Unknown schedule"):
        get_schedule_weights(1, schedule="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="race_number must be >= 1"):
        get_schedule_weights(0, schedule="extreme")


def test_get_schedule_weights_interpolates_between_checkpoints():
    weights = get_schedule_weights(race_number=7, schedule="conservative")
    assert weights == pytest.approx({"baseline": 0.2625, "testing": 0.0875, "current": 0.65})


def test_get_schedule_weights_clamps_after_last_checkpoint():
    weights = get_schedule_weights(race_number=99, schedule="insane")
    assert weights == {"baseline": 0.0, "testing": 0.0, "current": 1.0}


def test_get_schedule_weights_supports_rapid_adaptive_profile():
    weights_r2 = get_schedule_weights(race_number=2, schedule="rapid_adaptive")
    weights_r3 = get_schedule_weights(race_number=3, schedule="rapid_adaptive")

    assert weights_r2 == pytest.approx({"baseline": 0.20, "testing": 0.10, "current": 0.70})
    assert weights_r3 == pytest.approx({"baseline": 0.08, "testing": 0.05, "current": 0.87})


def test_calculate_blended_performance_uses_schedule_weights():
    blended = calculate_blended_performance(
        baseline_score=0.8,
        testing_modifier=0.1,
        current_score=0.6,
        race_number=1,
        schedule="extreme",
    )
    assert blended == pytest.approx(0.56)


def test_get_recommended_schedule_switches_with_regulation_flag():
    assert get_recommended_schedule(is_regulation_change=True) == "extreme"
    assert get_recommended_schedule(is_regulation_change=False) == "moderate"
