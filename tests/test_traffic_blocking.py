"""The track's observed overtaking rate caps how often a pass can succeed."""

import numpy as np

from src.utils.lap_by_lap_simulator import _expand_overtake_cfg, _get_traffic_overtake_effect


def _states(gap_s: float, field_size: int = 22) -> dict[str, dict]:
    """Two cars in a battle, padded to a real field: the cap divides by following pairs."""
    states = {
        "AHEAD": {"cumulative_time": 100.0, "base_pace": 92.0, "has_dnf": False, "position": 5},
        "CHASER": {
            "cumulative_time": 100.0 + gap_s,
            "base_pace": 88.0,
            "has_dnf": False,
            "position": 6,
        },
    }
    for index in range(field_size - 2):
        states[f"P{index:02d}"] = {
            "cumulative_time": 200.0 + index,
            "base_pace": 92.0,
            "has_dnf": False,
            "position": 7 + index,
        }
    return states


def _info(states: dict[str, dict]) -> dict[str, dict]:
    common = {"overtaking_skill": 0.5, "defensive_skill": 0.5, "skill": 0.5, "race_advantage": 0.0}
    return {driver: dict(common) for driver in states}


def _params(avg_changes_per_lap: float | None, track_overtaking: float) -> dict:
    params = {
        "track_overtaking": track_overtaking,
        "overtake_model": _expand_overtake_cfg({}),
        "track_name": "Test",
    }
    if avg_changes_per_lap is not None:
        params["overtaking_avg_changes_per_lap"] = avg_changes_per_lap
    return params


def _pass_rate(
    avg_changes_per_lap: float | None, track_overtaking: float, runs: int = 400
) -> float:
    """Share of laps on which a much quicker chaser completes a pass."""
    passes = 0
    states = _states(0.4)
    for seed in range(runs):
        effect = _get_traffic_overtake_effect(
            driver="CHASER",
            driver_states=states,
            driver_info_map=_info(states),
            driver_ahead_map={"CHASER": "AHEAD"},
            race_params=_params(avg_changes_per_lap, track_overtaking),
            rng=np.random.default_rng(seed),
        )
        passes += effect < 0.0
    return passes / runs


def test_a_low_overtaking_track_lets_far_fewer_passes_through_than_a_high_one():
    """Monaco (1.12 changes/lap) against Spa (5.15) for the same 4 s/lap advantage."""
    monaco = _pass_rate(1.12, 0.95)
    spa = _pass_rate(5.15, 0.36)

    assert monaco < 0.10
    assert spa > monaco * 2


def test_a_track_without_observed_change_data_keeps_the_previous_ceiling():
    """Missing data must fall back, not guess a rate."""
    assert _pass_rate(None, 0.36) > 0.5


def test_the_cap_is_the_observed_rate_divided_by_the_following_pairs():
    """A 22-car field has 21 following pairs, so Monza's 3.9 changes/lap caps near 0.19."""
    assert 0.12 < _pass_rate(3.9, 0.55) < 0.26
