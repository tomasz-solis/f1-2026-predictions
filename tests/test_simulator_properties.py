"""Additional property-style tests for the lap-by-lap simulator.

The repo already had a small Hypothesis layer. These tests widen that coverage
to the structural guarantees we rely on when the race model evolves.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.simulation.pit_strategy import generate_pit_strategy
from src.utils.lap_by_lap_simulator import aggregate_simulation_results, simulate_race_lap_by_lap

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - depends on optional test dependency.
    pytest.skip("hypothesis is not installed", allow_module_level=True)


def _race_params() -> dict:
    """Return a compact race-parameter set suitable for simulator invariants."""
    return {
        "fuel": {"initial_load_kg": 100.0, "effect_per_lap": 0.0, "burn_rate_kg_per_lap": 1.5},
        "lap_time": {
            "reference_base": 90.0,
            "team_pace_penalty_range": 1.2,
            "skill_improvement_max": 0.15,
            "bounds": [70.0, 120.0],
        },
        "team_strength_compression": 1.0,
        "race_advantage_lap_impact": 0.0,
        "start_grid_gap_seconds": 0.35,
        "base_chaos": {"dry": 0.0, "wet": 0.05},
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
            "pass_probability_base": 0.35,
            "pass_probability_scale": 0.0,
            "pass_time_bonus_range": [0.1, 0.1],
            "pace_diff_scale": 0.5,
            "skill_scale": 0.2,
            "defense_scale": 0.2,
            "race_adv_scale": 0.0,
            "track_ease_scale": 0.0,
        },
    }


def _build_driver_info_map(
    drivers: int,
    *,
    identical: bool = False,
    dnf_probability: float = 0.0,
) -> dict[str, dict[str, float | int | str | dict[str, float]]]:
    """Create a valid driver map for simulator tests."""
    driver_info_map: dict[str, dict[str, float | int | str | dict[str, float]]] = {}
    for idx in range(1, drivers + 1):
        driver = f"D{idx:02d}"
        team_name = f"Team {(idx - 1) // 2 + 1}"
        if identical:
            strength = 0.5
            skill = 0.5
        else:
            strength = max(0.25, 0.9 - ((idx - 1) * 0.025))
            skill = max(0.35, 0.8 - ((idx - 1) * 0.01))

        driver_info_map[driver] = {
            "driver": driver,
            "team": team_name,
            "grid_pos": idx,
            "dnf_probability": dnf_probability,
            "team_strength": strength,
            "team_strength_by_compound": {"SOFT": strength, "MEDIUM": strength, "HARD": strength},
            "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.08, "HARD": 0.06},
            "skill": skill,
            "race_advantage": 0.0,
            "overtaking_skill": skill,
            "defensive_skill": skill,
        }
    return driver_info_map


def _build_strategies(
    driver_info_map: dict[str, dict[str, float | int | str | dict[str, float]]],
    *,
    race_distance: int,
    weather: str = "dry",
) -> dict[str, dict[str, int | list[int] | list[str]]]:
    """Create legal strategies for every driver in the field."""
    strategies: dict[str, dict[str, int | list[int] | list[str]]] = {}
    for index, driver in enumerate(driver_info_map, start=1):
        rng = np.random.default_rng(100 + index)
        strategies[driver] = generate_pit_strategy(
            race_distance=race_distance,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"]
            if weather == "dry"
            else ["INTERMEDIATE", "WET"],
            rng=rng,
            enforce_two_compound_rule=(weather == "dry"),
            grid_position=index,
            track_overtaking=0.5,
        )
    return strategies


def _build_uniform_strategies(
    driver_info_map: dict[str, dict[str, float | int | str | dict[str, float]]],
    *,
    race_distance: int,
) -> dict[str, dict[str, int | list[int] | list[str]]]:
    """Create one common dry strategy so pace, not strategy variance, drives the test."""
    midpoint = race_distance // 2
    return {
        driver: {
            "num_stops": 1,
            "pit_laps": [midpoint],
            "compound_sequence": ["MEDIUM", "HARD"],
            "stint_lengths": [midpoint, race_distance - midpoint],
        }
        for driver in driver_info_map
    }


@settings(max_examples=20, deadline=None)
@given(drivers=st.integers(min_value=2, max_value=22))
def test_race_result_is_partition_of_input_field(drivers: int):
    """Every driver must end up either in the finish order or in the DNF list."""
    driver_info_map = _build_driver_info_map(drivers, dnf_probability=0.25)
    strategies = _build_strategies(driver_info_map, race_distance=24)

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=_race_params(),
        race_distance=24,
        weather="dry",
        rng=np.random.default_rng(7),
    )

    finish_set = set(result["finish_order"])
    dnf_set = set(result["dnf_drivers"])
    assert finish_set == set(driver_info_map)
    assert dnf_set.issubset(finish_set)
    assert len(result["finish_order"]) == drivers


@settings(max_examples=20, deadline=None)
@given(drivers=st.integers(min_value=2, max_value=22))
def test_finish_positions_are_unique_and_contiguous(drivers: int):
    """Simulation output should never duplicate or skip positions."""
    driver_info_map = _build_driver_info_map(drivers)
    strategies = _build_strategies(driver_info_map, race_distance=22)

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=_race_params(),
        race_distance=22,
        weather="dry",
        rng=np.random.default_rng(11),
    )

    assert result["finish_order"] == list(dict.fromkeys(result["finish_order"]))
    assert Counter(result["finish_order"]).most_common(1)[0][1] == 1
    assert len(result["finish_order"]) == drivers


@settings(max_examples=30, deadline=None)
@given(
    race_distance=st.integers(min_value=18, max_value=70),
    tire_stress=st.floats(min_value=2.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_generated_dry_strategies_always_use_at_least_two_compounds(
    race_distance: int,
    tire_stress: float,
):
    """Dry-race strategy generator should respect the FIA two-compound rule."""
    strategy = generate_pit_strategy(
        race_distance=race_distance,
        tire_stress_score=tire_stress,
        available_compounds=["SOFT", "MEDIUM", "HARD"],
        rng=np.random.default_rng(23),
        enforce_two_compound_rule=True,
    )

    assert strategy["num_stops"] >= 1
    assert len(set(strategy["compound_sequence"])) >= 2
    assert sum(strategy["stint_lengths"]) == race_distance


def test_every_dry_strategy_used_in_simulation_has_a_pit_stop():
    """A dry-race run should not quietly generate zero-stop strategies."""
    driver_info_map = _build_driver_info_map(12)
    strategies = _build_strategies(driver_info_map, race_distance=30)

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=_race_params(),
        race_distance=30,
        weather="dry",
        rng=np.random.default_rng(19),
    )

    assert result["strategies_used"]
    assert all(int(strategy["num_stops"]) >= 1 for strategy in result["strategies_used"].values())


def test_identical_drivers_finish_with_similar_average_positions():
    """Symmetric cars and drivers should not drift into large persistent gaps."""
    driver_info_map = _build_driver_info_map(8, identical=True)
    strategies = _build_uniform_strategies(driver_info_map, race_distance=20)
    finish_positions: dict[str, list[int]] = {driver: [] for driver in driver_info_map}

    for seed in range(120):
        result = simulate_race_lap_by_lap(
            driver_info_map=driver_info_map,
            strategies=strategies,
            race_params=_race_params(),
            race_distance=20,
            weather="dry",
            rng=np.random.default_rng(seed),
        )
        for position, driver in enumerate(result["finish_order"], start=1):
            finish_positions[driver].append(position)

    assert abs(np.mean(finish_positions["D01"]) - np.mean(finish_positions["D02"])) <= 2.0


def test_pole_sitter_finishes_top_five_more_than_half_the_time():
    """Pole should still matter in aggregate even with race-day noise."""
    driver_info_map = _build_driver_info_map(12)
    strategies = _build_uniform_strategies(driver_info_map, race_distance=26)
    top_five_finishes = 0

    for seed in range(100):
        result = simulate_race_lap_by_lap(
            driver_info_map=driver_info_map,
            strategies=strategies,
            race_params=_race_params(),
            race_distance=26,
            weather="dry",
            rng=np.random.default_rng(seed),
        )
        pole_position = result["finish_order"].index("D01") + 1
        if pole_position <= 5:
            top_five_finishes += 1

    assert top_five_finishes >= 50


def test_aggregate_results_probability_outputs_stay_bounded():
    """Aggregated simulator outputs should remain valid probability tables."""
    driver_info_map = _build_driver_info_map(10, dnf_probability=0.1)
    strategies = _build_strategies(driver_info_map, race_distance=25)
    results = [
        simulate_race_lap_by_lap(
            driver_info_map=driver_info_map,
            strategies=strategies,
            race_params=_race_params(),
            race_distance=25,
            weather="dry",
            rng=np.random.default_rng(seed),
        )
        for seed in range(20)
    ]

    aggregated = aggregate_simulation_results(results)

    assert aggregated["median_positions"]
    assert all(1 <= position <= 10 for position in aggregated["median_positions"].values())
    assert all(0.0 <= rate <= 1.0 for rate in aggregated["dnf_rates"].values())
    assert (
        pytest.approx(sum(aggregated["compound_strategy_distribution"].values()), rel=1e-6) == 1.0
    )


def test_wet_strategy_generation_can_legally_repeat_one_compound():
    """Wet strategies should be allowed to stay on one wet-weather compound."""
    strategy = generate_pit_strategy(
        race_distance=24,
        tire_stress_score=3.8,
        available_compounds=["INTERMEDIATE", "WET"],
        rng=np.random.default_rng(8),
        enforce_two_compound_rule=False,
    )

    assert len(strategy["compound_sequence"]) == strategy["num_stops"] + 1
    assert all(compound in {"INTERMEDIATE", "WET"} for compound in strategy["compound_sequence"])
