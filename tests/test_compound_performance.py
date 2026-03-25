from __future__ import annotations

import pytest

from src.data.compound_performance import (
    get_compound_performance_modifier,
    get_compound_tire_deg_factor,
    get_team_compound_advantage,
    should_use_compound_adjustments,
)


def test_get_compound_performance_modifier_handles_empty_inputs():
    assert get_compound_performance_modifier({}, "SOFT") == 0.0
    assert get_compound_performance_modifier({"SOFT": {}}, "MEDIUM") == 0.0


def test_get_compound_performance_modifier_respects_weights_and_clips():
    payload = {
        "SOFT": {
            "pace_performance": 1.0,
            "tire_deg_performance": 1.0,
        }
    }
    modifier = get_compound_performance_modifier(
        payload,
        "soft",
        metric_weights={"pace_performance": 10.0, "tire_deg_performance": 0.0},
    )
    assert modifier == pytest.approx(0.05)


def test_get_compound_tire_deg_factor_defaults_and_inversion():
    payload = {"MEDIUM": {"tire_deg_performance": 0.8}}

    assert get_compound_tire_deg_factor(payload, "MEDIUM") == pytest.approx(0.2)
    assert get_compound_tire_deg_factor(payload, "HARD") == 0.5
    assert get_compound_tire_deg_factor({}, "SOFT") == 0.5


def test_should_use_compound_adjustments_checks_sample_depth():
    shallow = {
        "SOFT": {"laps_sampled": 4},
        "MEDIUM": {"laps_sampled": 2},
    }
    rich = {
        "SOFT": {"laps_sampled": 6},
        "MEDIUM": {"laps_sampled": 6},
    }

    assert should_use_compound_adjustments(shallow, min_laps_threshold=10) is False
    assert should_use_compound_adjustments(rich, min_laps_threshold=10) is True


def test_get_team_compound_advantage_averages_compound_modifiers():
    payload = {
        "SOFT": {"pace_performance": 0.7, "tire_deg_performance": 0.5},
        "HARD": {"pace_performance": 0.4, "tire_deg_performance": 0.5},
    }
    advantage = get_team_compound_advantage(payload, ["SOFT", "HARD"])

    assert advantage == pytest.approx(0.0035)
