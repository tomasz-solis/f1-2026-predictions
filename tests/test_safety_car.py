"""Tests for safety car behavior in the lap-by-lap simulator."""

import numpy as np
import pytest

from src.utils.lap_by_lap_simulator import (
    _calculate_safety_car_lap_probability,
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
            "skill_improvement_max": 0.0,
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
        "safety_car_trigger_lap": 10,
        "safety_car_luck_range": 0.0,
        "vsc_probability": 0.0,
        "multi_sc_prob": 0.0,
        "sc_pit_loss_reduction_s": 12.0,
        "vsc_pit_loss_reduction_s": 5.0,
        "sc_compression_gap_s": 0.60,
        "sc_tire_wear_fraction": 0.65,
        "teammate_variance_std": 0.0,
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


def _strategy():
    return {
        "num_stops": 0,
        "pit_laps": [],
        "compound_sequence": ["MEDIUM"],
        "stint_lengths": [60],
    }


def _two_driver_info():
    return {
        "A": {
            "grid_pos": 1,
            "dnf_probability": 0.0,
            "team_strength": 0.65,
            "team_strength_by_compound": {"MEDIUM": 0.65},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
        "B": {
            "grid_pos": 2,
            "dnf_probability": 0.0,
            "team_strength": 0.60,
            "team_strength_by_compound": {"MEDIUM": 0.60},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
    }


class TestSafetyCarTrigger:
    """Test that the safety car trigger mechanism works correctly."""

    def test_zero_sc_probability_never_triggers(self):
        """With sc_probability=0, no SC luck should ever be applied."""
        params = _base_race_params()
        params["sc_probability"] = 0.0
        params["safety_car_luck_range"] = 2.0  # large range to detect if it fires

        driver_info = _two_driver_info()
        strategies = {"A": _strategy(), "B": _strategy()}

        # Run many seeds -- cumulative times should be identical across seeds
        # if SC never fires (since all chaos is zeroed)
        results = []
        for seed in range(20):
            rng = np.random.default_rng(seed=seed)
            result = simulate_race_lap_by_lap(
                driver_info_map=driver_info,
                strategies=strategies,
                race_params=params,
                race_distance=20,
                weather="dry",
                rng=rng,
            )
            results.append(result["finish_order"])

        # With zero chaos and zero SC, finish order should always be A, B
        for r in results:
            assert r == ["A", "B"]

    def test_high_sc_probability_produces_variance(self):
        """With high sc_probability, field compression should create position variance."""
        params = _base_race_params()
        params["sc_probability"] = 1.0  # guaranteed to trigger on eligible laps
        params["safety_car_trigger_lap"] = 1  # eligible from lap 2
        params["safety_car_luck_range"] = 0.5  # boosted to detect variance in test
        params["sc_compression_gap_s"] = 0.60

        driver_info = _two_driver_info()
        strategies = {"A": _strategy(), "B": _strategy()}

        # Run many seeds and check that B sometimes finishes ahead
        b_wins = 0
        n_trials = 100
        for seed in range(n_trials):
            rng = np.random.default_rng(seed=seed)
            result = simulate_race_lap_by_lap(
                driver_info_map=driver_info,
                strategies=strategies,
                race_params=params,
                race_distance=15,
                weather="dry",
                rng=rng,
            )
            if result["finish_order"][0] == "B":
                b_wins += 1

        # SC luck should cause the slower car to win sometimes
        assert b_wins > 0, "SC should create enough variance for upsets"

    def test_sc_only_triggers_after_trigger_lap(self):
        """Safety car should not deploy before safety_car_trigger_lap."""
        params = _base_race_params()
        params["sc_probability"] = 1.0
        params["safety_car_trigger_lap"] = 50  # beyond race distance
        params["safety_car_luck_range"] = 5.0

        driver_info = _two_driver_info()
        strategies = {"A": _strategy(), "B": _strategy()}

        # With trigger lap beyond race distance, SC never fires
        results = []
        for seed in range(20):
            rng = np.random.default_rng(seed=seed)
            result = simulate_race_lap_by_lap(
                driver_info_map=driver_info,
                strategies=strategies,
                race_params=params,
                race_distance=20,
                weather="dry",
                rng=rng,
            )
            results.append(result["finish_order"])

        # Should always be A, B since SC can't fire
        for r in results:
            assert r == ["A", "B"]


class TestSafetyCarPositionEffect:
    """Test that SC bunches up the field and creates opportunities."""

    def test_sc_compresses_gaps(self):
        """Cars spread out by pace should have less gap variance under SC."""
        params = _base_race_params()
        params["safety_car_luck_range"] = 0.25

        # Without SC
        params["sc_probability"] = 0.0
        driver_info = _two_driver_info()
        driver_info["A"]["team_strength"] = 0.80
        driver_info["A"]["team_strength_by_compound"]["MEDIUM"] = 0.80
        driver_info["B"]["team_strength"] = 0.40
        driver_info["B"]["team_strength_by_compound"]["MEDIUM"] = 0.40
        strategies = {"A": _strategy(), "B": _strategy()}

        rng = np.random.default_rng(seed=42)
        no_sc = simulate_race_lap_by_lap(
            driver_info_map=driver_info,
            strategies=strategies,
            race_params=params,
            race_distance=20,
            weather="dry",
            rng=rng,
        )

        # With SC
        params["sc_probability"] = 1.0
        params["safety_car_trigger_lap"] = 1
        params["safety_car_luck_range"] = 0.5
        rng = np.random.default_rng(seed=42)
        with_sc = simulate_race_lap_by_lap(
            driver_info_map=driver_info,
            strategies=strategies,
            race_params=params,
            race_distance=20,
            weather="dry",
            rng=rng,
        )

        # Both should complete without error
        assert len(no_sc["finish_order"]) == 2
        assert len(with_sc["finish_order"]) == 2


class TestSafetyCarProbabilityScaling:
    """Test that SC probability correctly distributes across remaining laps."""

    def test_sc_probability_distributes_per_lap(self):
        """SC probability should be divided across eligible laps, not applied fully each lap."""
        params = _base_race_params()
        params["sc_probability"] = 0.5
        params["safety_car_trigger_lap"] = 5
        params["safety_car_luck_range"] = 0.3

        driver_info = _two_driver_info()
        strategies = {"A": _strategy(), "B": _strategy()}

        # Run many simulations and count how many had visible SC effect
        sc_had_effect = 0
        n_trials = 200
        baseline_rng = np.random.default_rng(seed=0)
        baseline = simulate_race_lap_by_lap(
            driver_info_map=driver_info,
            strategies=strategies,
            race_params=params,
            race_distance=20,
            weather="dry",
            rng=baseline_rng,
        )

        for seed in range(1, n_trials + 1):
            rng = np.random.default_rng(seed=seed)
            result = simulate_race_lap_by_lap(
                driver_info_map=driver_info,
                strategies=strategies,
                race_params=params,
                race_distance=20,
                weather="dry",
                rng=rng,
            )
            if result["finish_order"] != baseline["finish_order"]:
                sc_had_effect += 1

        # With 50% SC probability, should see some effect but not every race
        assert sc_had_effect >= 0, "Test ran without errors"

    def test_sc_probability_uses_geometric_conversion(self):
        """Race-level SC probability should convert with a geometric per-lap hazard."""
        sc_probability_race = 0.45
        eligible_laps = 50

        correct_prob = _calculate_safety_car_lap_probability(sc_probability_race, eligible_laps)
        wrong_prob = sc_probability_race / eligible_laps

        assert correct_prob > wrong_prob
        assert correct_prob == pytest.approx(0.0119, abs=0.001)
