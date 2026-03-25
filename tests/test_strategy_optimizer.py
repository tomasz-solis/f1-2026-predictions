"""Unit tests for strategy optimization helpers."""

from __future__ import annotations

import pytest

from src.simulation.strategy_optimizer import (
    calculate_pit_timing_bias_laps,
    calculate_undercut_window,
    evaluate_overcut,
)


def test_calculate_undercut_window_returns_reasonable_bounds():
    """Undercut window should stay ordered and within race boundaries."""
    earliest, latest = calculate_undercut_window(
        race_distance=60,
        car_ahead_tire_age=24,
        pit_loss_time=21.0,
        fresh_tire_advantage=0.30,
    )

    assert earliest >= 5
    assert latest <= 57
    assert earliest <= latest


def test_evaluate_overcut_prefers_hard_overtaking_tracks():
    """Overcut score should be higher when overtaking is difficult."""
    monaco_score = evaluate_overcut(track_overtaking=0.95, tire_age=18)
    monza_score = evaluate_overcut(track_overtaking=0.20, tire_age=18)

    assert monaco_score > monza_score


def test_pit_timing_bias_monaco_vs_monza():
    """Monaco should bias later pit timing vs Monza for same grid position."""
    monaco_bias = calculate_pit_timing_bias_laps(
        track_overtaking=0.95,
        grid_position=5,
        race_distance=60,
    )
    monza_bias = calculate_pit_timing_bias_laps(
        track_overtaking=0.20,
        grid_position=5,
        race_distance=60,
    )

    assert monaco_bias > 0
    assert monza_bias < 0
    assert monaco_bias > monza_bias


@pytest.mark.parametrize("grid_pos", [1, 6, 12, 20])
def test_pit_timing_bias_is_bounded(grid_pos: int):
    """Bias should always stay within configured safety range."""
    bias = calculate_pit_timing_bias_laps(
        track_overtaking=0.8,
        grid_position=grid_pos,
        race_distance=60,
        strategy_signal=1.5,
    )

    assert -4.0 <= bias <= 4.0
