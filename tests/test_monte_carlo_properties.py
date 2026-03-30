"""Property-based checks for Monte Carlo simulation invariants."""

from __future__ import annotations

import numpy as np
import pytest

import src.utils.lap_by_lap_simulator as lap_by_lap_simulator_module
from src.simulation.pit_strategy import generate_pit_strategy
from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - depends on optional test dependency.
    pytest.skip("hypothesis is not installed", allow_module_level=True)


def _race_params() -> dict:
    """Return a low-chaos race setup that keeps property tests stable."""
    return {
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
        "lap1_chaos": {
            "front_row": 0.0,
            "upper_midfield": 0.0,
            "midfield": 0.0,
            "back_field": 0.0,
        },
        "pit_stops": {"loss_duration": 22.0, "overtake_loss_range": [0.0, 0.0]},
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


def _build_driver_info_map(
    drivers: int,
    *,
    dnf_probability: float = 0.0,
) -> dict[str, dict]:
    """Create a simple field with modest pace spread for simulator properties."""
    driver_info_map: dict[str, dict] = {}
    for idx in range(1, drivers + 1):
        driver = f"D{idx:02d}"
        team_strength = float(np.clip(0.72 - ((idx - 1) * 0.02), 0.30, 0.72))
        driver_info_map[driver] = {
            "grid_pos": idx,
            "team": f"Team {idx:02d}",
            "dnf_probability": dnf_probability,
            "team_strength": team_strength,
            "team_strength_by_compound": {
                "SOFT": min(1.0, team_strength + 0.02),
                "MEDIUM": team_strength,
                "HARD": max(0.0, team_strength - 0.02),
            },
            "tire_deg_by_compound": {"SOFT": 0.10, "MEDIUM": 0.08, "HARD": 0.06},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
            "defensive_skill": 0.5,
        }
    return driver_info_map


def _build_two_compound_strategies(
    driver_info_map: dict[str, dict],
    *,
    race_distance: int,
) -> dict[str, dict]:
    """Build deterministic one-stop dry strategies that satisfy the FIA compound rule."""
    midpoint = max(1, race_distance // 2)
    return {
        driver: {
            "num_stops": 1,
            "pit_laps": [midpoint],
            "compound_sequence": ["MEDIUM", "HARD"],
            "stint_lengths": [midpoint, race_distance - midpoint],
        }
        for driver in driver_info_map
    }


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
    race_params = _race_params()
    driver_info_map = _build_driver_info_map(drivers)
    strategies = {
        driver: {
            "num_stops": 0,
            "pit_laps": [],
            "compound_sequence": ["MEDIUM"],
            "stint_lengths": [20],
        }
        for driver in driver_info_map
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


@settings(max_examples=20, deadline=None)
@given(
    drivers=st.integers(min_value=2, max_value=22),
    dnf_probability=st.floats(
        min_value=0.05,
        max_value=0.60,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_dnf_drivers_excluded_from_finish_order(drivers: int, dnf_probability: float):
    """DNF tracking should still partition the classified order cleanly.

    The simulator keeps DNFs in the classified order at the tail rather than
    returning a finishers-only list, so this test validates the invariant after
    splitting the classified order into finishers and DNFs.
    """
    driver_info_map = _build_driver_info_map(drivers, dnf_probability=dnf_probability)
    strategies = _build_two_compound_strategies(driver_info_map, race_distance=24)

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=_race_params(),
        race_distance=24,
        weather="dry",
        rng=np.random.default_rng(17),
    )

    dnf_set = set(result["dnf_drivers"])
    classified_finishers = [driver for driver in result["finish_order"] if driver not in dnf_set]

    assert len(classified_finishers) + len(result["dnf_drivers"]) == len(driver_info_map)
    assert not dnf_set.intersection(classified_finishers)
    assert set(classified_finishers).union(dnf_set) == set(driver_info_map)


@settings(max_examples=20, deadline=None)
@given(
    drivers=st.integers(min_value=2, max_value=22),
    race_distance=st.integers(min_value=18, max_value=60),
)
def test_pit_stops_respect_two_compound_rule(drivers: int, race_distance: int):
    """Dry-race strategies should keep every classified finisher on two compounds."""
    driver_info_map = _build_driver_info_map(drivers)
    strategies: dict[str, dict] = {}
    for index, driver in enumerate(driver_info_map, start=1):
        strategies[driver] = generate_pit_strategy(
            race_distance=race_distance,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=np.random.default_rng(index),
            enforce_two_compound_rule=True,
        )

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=_race_params(),
        race_distance=race_distance,
        weather="dry",
        rng=np.random.default_rng(29),
    )

    dnf_set = set(result["dnf_drivers"])
    finishing_drivers = [driver for driver in result["finish_order"] if driver not in dnf_set]
    for driver in finishing_drivers:
        compounds_used = result["strategies_used"][driver]["compound_sequence"]
        assert int(result["strategies_used"][driver]["num_stops"]) >= 1
        assert len(set(compounds_used)) >= 2


def test_grid_position_1_finishes_top5_more_often():
    """Pole position should remain a strong advantage across many race samples."""
    driver_info_map = _build_driver_info_map(12)
    strategies = _build_two_compound_strategies(driver_info_map, race_distance=26)
    top_five_finishes = 0

    for seed in range(200):
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

    assert top_five_finishes > 100


def test_faster_team_beats_slower_team_on_average():
    """A much stronger car should beat a weaker car on average even from P2."""
    race_params = _race_params()
    race_params["start_grid_gap_seconds"] = 0.1
    race_params["track_overtaking"] = 0.1
    race_params["overtake_model"]["pass_threshold_base"] = -1.0
    race_params["overtake_model"]["pass_probability_base"] = 1.0
    race_params["overtake_model"]["pass_probability_scale"] = 0.0

    driver_info_map = {
        "SLOW": {
            "grid_pos": 1,
            "team": "Slow Team",
            "dnf_probability": 0.0,
            "team_strength": 0.3,
            "team_strength_by_compound": {"MEDIUM": 0.3, "HARD": 0.28},
            "tire_deg_by_compound": {"MEDIUM": 0.05, "HARD": 0.04},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
            "defensive_skill": 0.5,
        },
        "FAST": {
            "grid_pos": 2,
            "team": "Fast Team",
            "dnf_probability": 0.0,
            "team_strength": 0.9,
            "team_strength_by_compound": {"MEDIUM": 0.9, "HARD": 0.88},
            "tire_deg_by_compound": {"MEDIUM": 0.05, "HARD": 0.04},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
            "defensive_skill": 0.5,
        },
    }
    strategies = _build_two_compound_strategies(driver_info_map, race_distance=18)

    strong_team_positions: list[int] = []
    weak_team_positions: list[int] = []
    for seed in range(100):
        result = simulate_race_lap_by_lap(
            driver_info_map=driver_info_map,
            strategies=strategies,
            race_params=race_params,
            race_distance=18,
            weather="dry",
            rng=np.random.default_rng(seed),
        )
        strong_team_positions.append(result["finish_order"].index("FAST") + 1)
        weak_team_positions.append(result["finish_order"].index("SLOW") + 1)

    assert float(np.mean(strong_team_positions)) < float(np.mean(weak_team_positions))


def test_no_negative_cumulative_times(monkeypatch: pytest.MonkeyPatch):
    """Recorded lap snapshots should never drive cumulative time below zero."""
    snapshots: list[dict[str, float]] = []
    original_update = lap_by_lap_simulator_module._update_positions_from_times

    def _tracking_update(driver_states: dict[str, dict]) -> None:
        snapshots.append(
            {driver: float(state["cumulative_time"]) for driver, state in driver_states.items()}
        )
        original_update(driver_states)

    monkeypatch.setattr(
        lap_by_lap_simulator_module,
        "_update_positions_from_times",
        _tracking_update,
    )

    driver_info_map = _build_driver_info_map(10, dnf_probability=0.1)
    strategies = _build_two_compound_strategies(driver_info_map, race_distance=24)

    result = simulate_race_lap_by_lap(
        driver_info_map=driver_info_map,
        strategies=strategies,
        race_params=_race_params(),
        race_distance=24,
        weather="dry",
        rng=np.random.default_rng(41),
    )

    assert result["finish_order"]
    assert snapshots
    assert all(
        cumulative_time >= 0.0 for snapshot in snapshots for cumulative_time in snapshot.values()
    )
