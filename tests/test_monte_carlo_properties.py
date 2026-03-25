"""Property-based checks for Monte Carlo simulation invariants."""

import numpy as np
import pytest

from src.simulation.pit_strategy import generate_pit_strategy
from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - depends on optional test dependency.
    pytest.skip("hypothesis is not installed", allow_module_level=True)


@settings(max_examples=40, deadline=None)
@given(
    race_distance=st.integers(min_value=15, max_value=70),
    tire_stress=st.floats(min_value=2.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_pit_strategy_always_valid(race_distance: int, tire_stress: float):
    """Generated pit strategies satisfy structural invariants."""
    rng = np.random.default_rng(42)
    strategy = generate_pit_strategy(
        race_distance=race_distance,
        tire_stress_score=tire_stress,
        available_compounds=["SOFT", "MEDIUM", "HARD"],
        rng=rng,
        enforce_two_compound_rule=True,
    )

    assert 1 <= strategy["num_stops"] <= 3
    assert all(1 <= lap < race_distance for lap in strategy["pit_laps"])
    assert len(strategy["compound_sequence"]) == strategy["num_stops"] + 1
    assert sum(strategy["stint_lengths"]) == race_distance
    assert len(set(strategy["compound_sequence"])) >= 2


@settings(max_examples=25, deadline=None)
@given(drivers=st.integers(min_value=2, max_value=22))
def test_race_simulation_finish_order_is_unique(drivers: int):
    """Race simulation returns each driver exactly once in finish order."""
    race_params = {
        "fuel": {"initial_load_kg": 100.0, "effect_per_lap": 0.0, "burn_rate_kg_per_lap": 1.5},
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
        "lap1_chaos": {"front_row": 0.0, "upper_midfield": 0.0, "midfield": 0.0, "back_field": 0.0},
        "pit_stops": {"loss_duration": 22.0, "overtake_loss_range": [0.0, 0.0]},
        "sc_probability": 0.0,
        "safety_car_luck_range": 0.0,
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

    driver_info_map = {}
    strategies = {}
    for idx in range(1, drivers + 1):
        driver = f"D{idx:02d}"
        driver_info_map[driver] = {
            "grid_pos": idx,
            "dnf_probability": 0.0,
            "team_strength": 0.5,
            "team_strength_by_compound": {"MEDIUM": 0.5},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
            "defensive_skill": 0.5,
        }
        strategies[driver] = {
            "num_stops": 0,
            "pit_laps": [],
            "compound_sequence": ["MEDIUM"],
            "stint_lengths": [20],
        }

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=race_params,
        race_distance=20,
        weather="dry",
        rng=np.random.default_rng(7),
    )

    finish_order = result["finish_order"]
    expected = set(driver_info_map.keys())
    assert len(finish_order) == drivers
    assert set(finish_order) == expected
