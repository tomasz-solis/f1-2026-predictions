"""Unit tests for lap-by-lap race simulator behavior."""

import numpy as np
import pytest

from src.utils.lap_by_lap_simulator import (
    _get_traffic_overtake_effect,
    _resolve_base_chaos_std,
    simulate_race_lap_by_lap,
)


def _base_race_params() -> dict:
    return {
        "fuel": {
            "initial_load_kg": 100.0,
            "effect_per_lap": 0.0,
            "burn_rate_kg_per_lap": 1.5,
        },
        "lap_time": {
            "reference_base": 90.0,
            "team_pace_penalty_range": 1.0,
            "skill_improvement_max": 0.2,
            "bounds": [70.0, 120.0],
        },
        "team_strength_compression": 1.0,
        "race_advantage_lap_impact": 0.0,
        "start_grid_gap_seconds": 0.4,
        "base_chaos": {"dry": 0.0, "wet": 0.0},
        "lap1_chaos": {
            "front_row": 0.0,
            "upper_midfield": 0.0,
            "midfield": 0.0,
            "back_field": 0.0,
        },
        "pit_stops": {
            "loss_duration": 22.0,
            "overtake_loss_range": [0.0, 0.0],
        },
        "sc_probability": 0.0,
        "safety_car_luck_range": 0.0,
        "teammate_variance_std": 0.0,
        "teammate_setup_offset_ratio": 0.0,
        "teammate_variance_lap_ratio": 0.0,
        "track_overtaking": 0.5,
        "overtake_model": {
            "dirty_air_window_s": 1.8,
            "dirty_air_penalty_base": 0.0,
            "dirty_air_penalty_track_scale": 0.0,
            "pass_window_s": 1.2,
            "pass_threshold_base": 0.1,
            "pass_threshold_track_scale": 0.0,
            "pass_probability_base": 0.0,
            "pass_probability_scale": 0.0,
            "pass_time_bonus_range": [0.1, 0.1],
            "pace_diff_scale": 0.5,
            "skill_scale": 0.2,
            "race_adv_scale": 0.2,
            "track_ease_scale": 0.2,
        },
    }


def _strategy() -> dict:
    return {
        "num_stops": 0,
        "pit_laps": [],
        "compound_sequence": ["MEDIUM"],
        "stint_lengths": [60],
    }


def test_mixed_weather_chaos_is_interpolated_between_dry_and_wet():
    """Mixed weather should use blended chaos instead of full wet value."""
    race_params = _base_race_params()
    race_params["base_chaos"] = {"dry": 0.20, "wet": 0.60}
    race_params["mixed_weather_chaos_blend"] = 0.50

    dry_std = _resolve_base_chaos_std(race_params, "dry")
    mixed_std = _resolve_base_chaos_std(race_params, "mixed")
    wet_std = _resolve_base_chaos_std(race_params, "rain")

    assert dry_std == pytest.approx(0.20)
    assert mixed_std == pytest.approx(0.40)
    assert wet_std == pytest.approx(0.60)
    assert dry_std < mixed_std < wet_std


def test_grid_gap_keeps_front_car_ahead_in_short_race():
    """One lap should still respect starting order when pace gap is modest."""
    race_params = _base_race_params()
    race_params["start_grid_gap_seconds"] = 0.8
    race_params["track_overtaking"] = 0.95
    race_params["overtake_model"]["pass_probability_base"] = 0.0

    driver_info_map = {
        "A": {
            "grid_pos": 1,
            "dnf_probability": 0.0,
            "team_strength": 0.55,
            "team_strength_by_compound": {"MEDIUM": 0.55},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
        "B": {
            "grid_pos": 2,
            "dnf_probability": 0.0,
            "team_strength": 0.65,
            "team_strength_by_compound": {"MEDIUM": 0.65},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
    }

    strategies = {"A": _strategy(), "B": _strategy()}
    rng = np.random.default_rng(seed=42)
    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=race_params,
        race_distance=1,
        weather="dry",
        rng=rng,
    )

    assert result["finish_order"] == ["A", "B"]


def test_fast_car_can_pass_on_easy_overtaking_track():
    """A clearly faster driver from P2 should pass with easy overtaking settings."""
    race_params = _base_race_params()
    race_params["start_grid_gap_seconds"] = 0.1
    race_params["track_overtaking"] = 0.1
    race_params["overtake_model"]["pass_threshold_base"] = -1.0
    race_params["overtake_model"]["pass_probability_base"] = 1.0
    race_params["overtake_model"]["pass_probability_scale"] = 0.0

    driver_info_map = {
        "A": {
            "grid_pos": 1,
            "dnf_probability": 0.0,
            "team_strength": 0.30,
            "team_strength_by_compound": {"MEDIUM": 0.30},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.45,
            "race_advantage": -0.02,
            "overtaking_skill": 0.45,
        },
        "B": {
            "grid_pos": 2,
            "dnf_probability": 0.0,
            "team_strength": 0.85,
            "team_strength_by_compound": {"MEDIUM": 0.85},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.65,
            "race_advantage": 0.05,
            "overtaking_skill": 0.75,
        },
    }

    strategies = {"A": _strategy(), "B": _strategy()}
    rng = np.random.default_rng(seed=7)
    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=race_params,
        race_distance=8,
        weather="dry",
        rng=rng,
    )

    assert result["finish_order"][0] == "B"


def test_persistent_teammate_setup_offset_can_break_identical_teammates():
    """Persistent teammate offsets should not wash out like per-lap white noise."""
    race_params = _base_race_params()
    race_params["start_grid_gap_seconds"] = 0.0
    race_params["teammate_variance_std"] = 0.20
    race_params["teammate_setup_offset_ratio"] = 1.0
    race_params["teammate_variance_lap_ratio"] = 0.0
    race_params["track_overtaking"] = 0.95
    race_params["overtake_model"]["pass_probability_base"] = 0.0

    driver_info_map = {
        "A": {
            "grid_pos": 1,
            "team": "Teammates",
            "dnf_probability": 0.0,
            "team_strength": 0.60,
            "team_strength_by_compound": {"MEDIUM": 0.60},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
        "B": {
            "grid_pos": 2,
            "team": "Teammates",
            "dnf_probability": 0.0,
            "team_strength": 0.60,
            "team_strength_by_compound": {"MEDIUM": 0.60},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
    }

    strategies = {"A": _strategy(), "B": _strategy()}
    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=race_params,
        race_distance=12,
        weather="dry",
        rng=np.random.default_rng(seed=3),
    )

    assert result["finish_order"] == ["B", "A"]


def test_strong_defender_reduces_overtake_success():
    """Defender skill should make passes materially harder in close pace scenarios."""
    race_params = _base_race_params()
    race_params["track_overtaking"] = 0.15
    race_params["overtake_model"]["pass_threshold_base"] = 0.28
    race_params["overtake_model"]["pass_threshold_track_scale"] = 0.0
    race_params["overtake_model"]["pass_probability_base"] = 1.0
    race_params["overtake_model"]["pass_probability_scale"] = 0.0
    race_params["overtake_model"]["pass_time_bonus_range"] = [0.15, 0.15]
    race_params["overtake_model"]["skill_scale"] = 0.25
    race_params["overtake_model"]["defense_scale"] = 0.45
    race_params["overtake_model"]["track_ease_scale"] = 0.20
    race_params["overtake_model"]["dirty_air_penalty_base"] = 0.0
    race_params["overtake_model"]["dirty_air_penalty_track_scale"] = 0.0

    driver_states = {
        "A": {
            "position": 12,
            "cumulative_time": 100.0,
            "base_pace": 90.0,
            "has_dnf": False,
        },
        "B": {
            "position": 13,
            "cumulative_time": 100.6,
            "base_pace": 89.95,
            "has_dnf": False,
        },
    }
    driver_ahead_map = {"B": "A"}

    weak_map = {
        "A": {"skill": 0.55, "defensive_skill": 0.20, "overtaking_skill": 0.50},
        "B": {"skill": 0.60, "race_advantage": 0.0, "overtaking_skill": 0.90},
    }
    strong_map = {
        "A": {"skill": 0.55, "defensive_skill": 0.95, "overtaking_skill": 0.50},
        "B": {"skill": 0.60, "race_advantage": 0.0, "overtaking_skill": 0.90},
    }

    rng_weak = np.random.default_rng(seed=1)
    weak_effect = _get_traffic_overtake_effect(
        driver="B",
        driver_states=driver_states,
        driver_info_map=weak_map,
        driver_ahead_map=driver_ahead_map,
        race_params=race_params,
        rng=rng_weak,
    )

    rng_strong = np.random.default_rng(seed=1)
    strong_effect = _get_traffic_overtake_effect(
        driver="B",
        driver_states=driver_states,
        driver_info_map=strong_map,
        driver_ahead_map=driver_ahead_map,
        race_params=race_params,
        rng=rng_strong,
    )

    # Lower effect is better for attacker (negative = pass gain).
    assert weak_effect < strong_effect


def test_dirty_air_penalty_is_stronger_on_monaco_than_monza():
    """Dirty-air loss should be materially higher at Monaco than Monza for same gap."""
    race_params = _base_race_params()
    race_params["overtake_model"]["dirty_air_penalty_base"] = 0.05
    race_params["overtake_model"]["dirty_air_penalty_track_scale"] = 0.12
    race_params["overtake_model"]["pass_threshold_base"] = 10.0
    race_params["overtake_model"]["pass_threshold_track_scale"] = 0.0
    race_params["overtake_model"]["pass_probability_base"] = 0.0
    race_params["overtake_model"]["pass_probability_scale"] = 0.0

    driver_states = {
        "A": {"position": 4, "cumulative_time": 100.0, "base_pace": 90.0, "has_dnf": False},
        "B": {"position": 5, "cumulative_time": 101.0, "base_pace": 90.0, "has_dnf": False},
    }
    driver_ahead_map = {"B": "A"}
    driver_info_map = {
        "A": {"skill": 0.6, "defensive_skill": 0.8, "overtaking_skill": 0.5},
        "B": {"skill": 0.6, "race_advantage": 0.0, "overtaking_skill": 0.0},
    }

    race_params["track_name"] = "Monaco Grand Prix"
    monaco_effect = _get_traffic_overtake_effect(
        driver="B",
        driver_states=driver_states,
        driver_info_map=driver_info_map,
        driver_ahead_map=driver_ahead_map,
        race_params=race_params,
        rng=np.random.default_rng(1),
    )

    race_params["track_name"] = "Italian Grand Prix"
    monza_effect = _get_traffic_overtake_effect(
        driver="B",
        driver_states=driver_states,
        driver_info_map=driver_info_map,
        driver_ahead_map=driver_ahead_map,
        race_params=race_params,
        rng=np.random.default_rng(1),
    )

    assert monaco_effect > monza_effect
    assert monaco_effect >= monza_effect * 1.8
    assert monaco_effect == pytest.approx(0.04, abs=0.01)


def test_dirty_air_penalty_not_applied_beyond_gap_window():
    """Following outside 1.8s should produce no dirty-air lap-time penalty."""
    race_params = _base_race_params()
    race_params["overtake_model"]["dirty_air_penalty_base"] = 0.05
    race_params["overtake_model"]["dirty_air_penalty_track_scale"] = 0.12
    race_params["overtake_model"]["pass_threshold_base"] = 10.0
    race_params["overtake_model"]["pass_threshold_track_scale"] = 0.0
    race_params["track_name"] = "Monaco Grand Prix"

    driver_states = {
        "A": {"position": 4, "cumulative_time": 100.0, "base_pace": 90.0, "has_dnf": False},
        "B": {"position": 5, "cumulative_time": 102.0, "base_pace": 90.0, "has_dnf": False},
    }
    driver_ahead_map = {"B": "A"}
    driver_info_map = {
        "A": {"skill": 0.6, "defensive_skill": 0.8, "overtaking_skill": 0.5},
        "B": {"skill": 0.6, "race_advantage": 0.0, "overtaking_skill": 0.0},
    }

    effect = _get_traffic_overtake_effect(
        driver="B",
        driver_states=driver_states,
        driver_info_map=driver_info_map,
        driver_ahead_map=driver_ahead_map,
        race_params=race_params,
        rng=np.random.default_rng(1),
    )

    assert effect == 0.0


def test_front_overtakes_are_harder_than_backfield_overtakes():
    """Overtakes near the front should be lower-yield than backfield passes."""
    race_params = _base_race_params()
    race_params["track_overtaking"] = 0.10
    race_params["overtake_model"]["pass_threshold_base"] = -1.0
    race_params["overtake_model"]["pass_threshold_track_scale"] = 0.0
    race_params["overtake_model"]["pass_probability_base"] = 1.5
    race_params["overtake_model"]["pass_probability_scale"] = 0.0
    race_params["overtake_model"]["pass_time_bonus_range"] = [0.20, 0.20]
    race_params["overtake_model"]["dirty_air_penalty_base"] = 0.0
    race_params["overtake_model"]["dirty_air_penalty_track_scale"] = 0.0

    front_states = {
        "A": {"position": 2, "cumulative_time": 100.0, "base_pace": 90.0, "has_dnf": False},
        "B": {"position": 3, "cumulative_time": 100.5, "base_pace": 89.9, "has_dnf": False},
    }
    back_states = {
        "C": {"position": 18, "cumulative_time": 100.0, "base_pace": 90.0, "has_dnf": False},
        "D": {"position": 19, "cumulative_time": 100.5, "base_pace": 89.9, "has_dnf": False},
    }
    front_ahead_map = {"B": "A"}
    back_ahead_map = {"D": "C"}

    info_map = {
        "A": {"skill": 0.6, "defensive_skill": 0.8, "overtaking_skill": 0.5},
        "B": {"skill": 0.7, "race_advantage": 0.05, "overtaking_skill": 0.9},
        "C": {"skill": 0.6, "defensive_skill": 0.8, "overtaking_skill": 0.5},
        "D": {"skill": 0.7, "race_advantage": 0.05, "overtaking_skill": 0.9},
    }

    front_effects = []
    back_effects = []
    for seed in range(100):
        front_effects.append(
            _get_traffic_overtake_effect(
                driver="B",
                driver_states=front_states,
                driver_info_map=info_map,
                driver_ahead_map=front_ahead_map,
                race_params=race_params,
                rng=np.random.default_rng(seed),
            )
        )
        back_effects.append(
            _get_traffic_overtake_effect(
                driver="D",
                driver_states=back_states,
                driver_info_map=info_map,
                driver_ahead_map=back_ahead_map,
                race_params=race_params,
                rng=np.random.default_rng(seed),
            )
        )

    # Lower effect is better for attacker (negative = pass gain).
    assert np.mean(back_effects) < np.mean(front_effects)


def test_elite_skill_lap_bonus_improves_finishing_position():
    """Elite skill bonus should create additional pace for top-tier drivers."""
    race_params = _base_race_params()
    race_params["lap_time"]["skill_improvement_max"] = 0.0
    race_params["lap_time"]["elite_skill_threshold"] = 0.90
    race_params["lap_time"]["elite_skill_lap_bonus_max"] = 0.10
    race_params["lap_time"]["elite_skill_exponent"] = 1.0
    race_params["start_grid_gap_seconds"] = 0.0
    race_params["track_overtaking"] = 0.1
    race_params["overtake_model"]["pass_threshold_base"] = -1.0
    race_params["overtake_model"]["pass_probability_base"] = 1.0
    race_params["overtake_model"]["pass_probability_scale"] = 0.0

    driver_info_map = {
        "A": {
            "grid_pos": 1,
            "dnf_probability": 0.0,
            "team_strength": 0.60,
            "team_strength_by_compound": {"MEDIUM": 0.60},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.89,  # below elite threshold
            "race_advantage": 0.0,
            "overtaking_skill": 0.70,
            "defensive_skill": 0.50,
        },
        "B": {
            "grid_pos": 2,
            "dnf_probability": 0.0,
            "team_strength": 0.60,
            "team_strength_by_compound": {"MEDIUM": 0.60},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.99,  # elite bonus active
            "race_advantage": 0.0,
            "overtaking_skill": 0.80,
            "defensive_skill": 0.50,
        },
    }

    strategies = {"A": _strategy(), "B": _strategy()}
    rng = np.random.default_rng(seed=11)
    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=race_params,
        race_distance=8,
        weather="dry",
        rng=rng,
    )

    assert result["finish_order"][0] == "B"
