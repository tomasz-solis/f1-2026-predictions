"""Unit tests for traffic and dirty-air model utilities."""

from __future__ import annotations

import pytest

from src.simulation.traffic_model import calculate_dirty_air_penalty, get_track_downforce_level


def test_dirty_air_penalty_zero_outside_window():
    """Dirty-air penalty should not apply beyond the configured gap window."""
    penalty = calculate_dirty_air_penalty(
        gap_to_car_ahead_s=2.0,
        track_downforce_level=1.0,
        dirty_air_window_s=1.8,
    )
    assert penalty == 0.0


def test_dirty_air_penalty_monaco_close_following():
    """At Monaco and ~1.0s gap, penalty should reflect the steeper gap falloff."""
    penalty = calculate_dirty_air_penalty(
        gap_to_car_ahead_s=1.0,
        track_downforce_level=1.0,
        dirty_air_window_s=1.8,
    )
    assert penalty == pytest.approx(0.015, abs=0.005)


def test_monaco_penalty_is_about_double_monza():
    """High-downforce Monaco should be roughly 2x Monza at same following gap."""
    monaco_penalty = calculate_dirty_air_penalty(
        gap_to_car_ahead_s=1.0,
        track_downforce_level=1.0,
    )
    monza_penalty = calculate_dirty_air_penalty(
        gap_to_car_ahead_s=1.0,
        track_downforce_level=0.30,
    )

    assert monaco_penalty >= monza_penalty * 1.8
    assert monaco_penalty <= monza_penalty * 2.2


def test_get_track_downforce_level_fallbacks():
    """Known tracks should map directly, unknown tracks should fallback safely."""
    assert get_track_downforce_level("Monaco Grand Prix") == 1.0
    assert get_track_downforce_level("Miami Grand Prix") == pytest.approx(0.72)
    assert get_track_downforce_level("Unknown Grand Prix", track_overtaking=0.66) == pytest.approx(
        0.66
    )
