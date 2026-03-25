"""Performance benchmarks for simulation hot paths."""

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from src.simulation.pit_strategy import generate_pit_strategy
from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap


@pytest.mark.benchmark
def test_pit_strategy_generation_performance(benchmark):
    """Pit strategy generation should stay fast for a typical race."""
    rng = np.random.default_rng(42)
    strategy = benchmark(
        generate_pit_strategy,
        race_distance=57,
        tire_stress_score=3.2,
        available_compounds=["SOFT", "MEDIUM", "HARD"],
        rng=rng,
        enforce_two_compound_rule=True,
    )
    assert strategy["num_stops"] >= 1


@pytest.mark.benchmark
def test_lap_by_lap_single_simulation_performance(benchmark):
    """Single lap-by-lap race simulation should remain within expected bounds."""
    race_params = {
        "fuel": {"initial_load_kg": 100.0, "effect_per_lap": 0.03, "burn_rate_kg_per_lap": 1.5},
        "lap_time": {
            "reference_base": 90.0,
            "team_pace_penalty_range": 3.0,
            "skill_improvement_max": 0.2,
            "bounds": [70.0, 120.0],
        },
        "team_strength_compression": 0.6,
        "race_advantage_lap_impact": 0.15,
        "start_grid_gap_seconds": 0.32,
        "safety_car_trigger_lap": 10,
        "base_chaos": {"dry": 0.15, "wet": 0.25},
        "lap1_chaos": {
            "front_row": 0.1,
            "upper_midfield": 0.2,
            "midfield": 0.25,
            "back_field": 0.2,
        },
        "pit_stops": {"loss_duration": 22.0, "overtake_loss_range": [0.0, 2.0]},
        "sc_probability": 0.3,
        "safety_car_luck_range": 0.2,
        "teammate_variance_std": 0.05,
        "track_overtaking": 0.5,
        "overtake_model": {
            "dirty_air_window_s": 1.8,
            "dirty_air_penalty_base": 0.03,
            "dirty_air_penalty_track_scale": 0.08,
            "pass_window_s": 1.1,
            "pass_threshold_base": 0.08,
            "pass_threshold_track_scale": 0.1,
            "pass_probability_base": 0.25,
            "pass_probability_scale": 0.3,
            "pass_time_bonus_range": [0.1, 0.25],
            "pace_diff_scale": 0.5,
            "skill_scale": 0.2,
            "defense_scale": 0.2,
            "race_adv_scale": 0.2,
            "track_ease_scale": 0.2,
            "zone_front_threshold_boost": 0.2,
            "zone_upper_threshold_boost": 0.1,
            "zone_mid_threshold_boost": 0.02,
            "zone_back_threshold_boost": -0.02,
            "zone_front_probability_scale": 0.6,
            "zone_upper_probability_scale": 0.8,
            "zone_mid_probability_scale": 0.95,
            "zone_back_probability_scale": 1.1,
            "zone_front_bonus_scale": 0.6,
            "zone_upper_bonus_scale": 0.8,
            "zone_mid_bonus_scale": 0.95,
            "zone_back_bonus_scale": 1.1,
        },
    }

    driver_info_map = {}
    strategies = {}
    for idx in range(1, 21):
        driver = f"D{idx:02d}"
        driver_info_map[driver] = {
            "grid_pos": idx,
            "dnf_probability": 0.02,
            "team_strength": 0.5,
            "team_strength_by_compound": {"SOFT": 0.52, "MEDIUM": 0.5, "HARD": 0.48},
            "tire_deg_by_compound": {"SOFT": 0.18, "MEDIUM": 0.14, "HARD": 0.1},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
            "defensive_skill": 0.5,
        }
        strategies[driver] = {
            "num_stops": 1,
            "pit_laps": [28],
            "compound_sequence": ["MEDIUM", "HARD"],
            "stint_lengths": [28, 29],
        }

    result = benchmark(
        simulate_race_lap_by_lap,
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=race_params,
        race_distance=57,
        weather="dry",
        rng=np.random.default_rng(7),
    )

    assert len(result["finish_order"]) == 20
