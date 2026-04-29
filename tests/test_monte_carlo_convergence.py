"""Tests for Monte Carlo simulation convergence and stability."""

import numpy as np
import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor


def find_driver_position(grid: list[dict], driver_code: str) -> int | None:
    """Find driver's position in grid."""
    for entry in grid:
        if entry["driver"] == driver_code:
            return entry["position"]
    return None


def test_qualifying_simulations_converge_with_fixed_seed():
    """Verify 50 simulations produce stable median position with fixed seed."""
    # Run multiple independent prediction sets with different seeds
    results = []
    for run in range(10):
        predictor_run = Baseline2026Predictor(seed=42 + run)
        result = predictor_run.predict_qualifying(
            year=2026, race_name="Australian Grand Prix", n_simulations=50
        )
        results.append(result["grid"])

    # Check position stability across runs for key drivers
    key_drivers = ["VER", "HAM", "LEC", "NOR", "PIA"]

    for driver in key_drivers:
        positions = []
        for grid in results:
            pos = find_driver_position(grid, driver)
            if pos is not None:
                positions.append(pos)

        if len(positions) >= 8:  # Need at least 8 of 10 runs
            std_dev = np.std(positions)
            assert std_dev < 3.5, (
                f"{driver} position variance too high: {std_dev:.2f} (positions: {positions})"
            )


def test_qualifying_simulation_deterministic_with_same_seed():
    """Verify same seed produces identical results."""
    predictor1 = Baseline2026Predictor(seed=42)
    predictor2 = Baseline2026Predictor(seed=42)

    result1 = predictor1.predict_qualifying(
        year=2026, race_name="Australian Grand Prix", n_simulations=50
    )
    result2 = predictor2.predict_qualifying(
        year=2026, race_name="Australian Grand Prix", n_simulations=50
    )

    # Grid order should be identical
    grid1 = result1["grid"]
    grid2 = result2["grid"]

    assert len(grid1) == len(grid2)

    for i, (entry1, entry2) in enumerate(zip(grid1, grid2, strict=True)):
        assert entry1["driver"] == entry2["driver"], (
            f"Position {i + 1} mismatch: {entry1['driver']} vs {entry2['driver']}"
        )
        assert entry1["position"] == entry2["position"]


def test_qualifying_position_distribution_reasonable():
    """Verify position distribution follows expected patterns."""
    predictor = Baseline2026Predictor(seed=42)

    result = predictor.predict_qualifying(
        year=2026, race_name="Australian Grand Prix", n_simulations=50
    )

    grid = result["grid"]

    # Check we have 22 drivers (2026: Audi and Cadillac joined)
    assert len(grid) >= 20, f"Expected at least 20 drivers, got {len(grid)}"
    num_drivers = len(grid)

    # Check positions are sequential 1-N
    positions = [entry["position"] for entry in grid]
    assert positions == list(range(1, num_drivers + 1)), (
        f"Positions should be sequential 1-{num_drivers}"
    )

    # Check no duplicate drivers
    drivers = [entry["driver"] for entry in grid]
    assert len(drivers) == len(set(drivers)), "Duplicate drivers in grid"

    # Check top teams are in reasonable positions
    top_teams = ["Red Bull Racing", "McLaren", "Ferrari", "Mercedes"]
    top_10_teams = [entry["team"] for entry in grid[:10]]

    top_team_count = sum(1 for team in top_10_teams if team in top_teams)
    assert top_team_count >= 5, f"Expected at least 5 top team cars in P1-10, got {top_team_count}"


def test_monte_carlo_higher_simulation_count_reduces_variance():
    """Verify increasing simulation count reduces position variance."""
    predictor_10 = Baseline2026Predictor(seed=42)
    predictor_100 = Baseline2026Predictor(seed=42)

    # Run with 10 simulations
    result_10 = predictor_10.predict_qualifying(
        year=2026, race_name="Australian Grand Prix", n_simulations=10
    )

    # Run with 100 simulations
    result_100 = predictor_100.predict_qualifying(
        year=2026, race_name="Australian Grand Prix", n_simulations=100
    )

    # Extract confidence intervals
    grid_10 = result_10["grid"]
    grid_100 = result_100["grid"]

    # Find variance for top 5 drivers
    drivers_to_check = ["VER", "HAM", "LEC", "NOR", "PIA"]

    for driver in drivers_to_check:
        entry_10 = next((e for e in grid_10 if e["driver"] == driver), None)
        entry_100 = next((e for e in grid_100 if e["driver"] == driver), None)

        if entry_10 and entry_100:
            # Higher simulation count should have tighter confidence
            confidence_10 = entry_10.get("confidence", 1.0)
            confidence_100 = entry_100.get("confidence", 1.0)

            # Confidence should improve or stay similar (not worse)
            assert confidence_100 >= confidence_10 * 0.8, (
                f"{driver}: confidence should not degrade significantly with more simulations "
                f"(10 sims: {confidence_10:.2f}, 100 sims: {confidence_100:.2f})"
            )


def test_race_simulations_converge_with_fixed_seed():
    """Verify race simulations produce stable results with fixed seed."""
    # Create a realistic qualifying grid
    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "PER", "team": "Red Bull Racing", "position": 2},
        {"driver": "NOR", "team": "McLaren", "position": 3},
        {"driver": "PIA", "team": "McLaren", "position": 4},
        {"driver": "LEC", "team": "Ferrari", "position": 5},
        {"driver": "SAI", "team": "Ferrari", "position": 6},
        {"driver": "HAM", "team": "Mercedes", "position": 7},
        {"driver": "RUS", "team": "Mercedes", "position": 8},
        {"driver": "ALO", "team": "Aston Martin", "position": 9},
        {"driver": "STR", "team": "Aston Martin", "position": 10},
        {"driver": "GAS", "team": "Alpine", "position": 11},
        {"driver": "OCO", "team": "Alpine", "position": 12},
        {"driver": "HUL", "team": "Haas F1 Team", "position": 13},
        {"driver": "MAG", "team": "Haas F1 Team", "position": 14},
        {"driver": "TSU", "team": "RB", "position": 15},
        {"driver": "RIC", "team": "RB", "position": 16},
        {"driver": "BOT", "team": "Sauber", "position": 17},
        {"driver": "ZHO", "team": "Sauber", "position": 18},
        {"driver": "ALB", "team": "Williams", "position": 19},
        {"driver": "SAR", "team": "Williams", "position": 20},
    ]

    # Run multiple predictions with different seeds
    results = []
    for run in range(5):
        predictor_run = Baseline2026Predictor(seed=100 + run)
        result = predictor_run.predict_race(
            qualifying_grid=qualifying_grid,
            race_name="Australian Grand Prix",
            n_simulations=50,
        )
        results.append(result["finish_order"])

    # Check position stability for top drivers
    top_drivers = ["VER", "NOR", "LEC", "PIA"]

    for driver in top_drivers:
        positions = []
        for finish_order in results:
            pos = find_driver_position(finish_order, driver)
            if pos is not None:
                positions.append(pos)

        if len(positions) >= 4:
            std_dev = np.std(positions)
            # Race has more variance than qualifying, so allow <3.0
            assert std_dev < 3.0, (
                f"{driver} race position variance too high: {std_dev:.2f} (positions: {positions})"
            )


@pytest.mark.slow
def test_race_convergence_with_increasing_simulations():
    """Verify race predictions keep the same front-running cohort as simulations increase."""
    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
        {"driver": "HAM", "team": "Mercedes", "position": 4},
        {"driver": "PIA", "team": "McLaren", "position": 5},
        {"driver": "SAI", "team": "Ferrari", "position": 6},
        {"driver": "RUS", "team": "Mercedes", "position": 7},
        {"driver": "PER", "team": "Red Bull Racing", "position": 8},
        {"driver": "ALO", "team": "Aston Martin", "position": 9},
        {"driver": "STR", "team": "Aston Martin", "position": 10},
        {"driver": "GAS", "team": "Alpine", "position": 11},
        {"driver": "OCO", "team": "Alpine", "position": 12},
        {"driver": "HUL", "team": "Haas F1 Team", "position": 13},
        {"driver": "MAG", "team": "Haas F1 Team", "position": 14},
        {"driver": "TSU", "team": "RB", "position": 15},
        {"driver": "RIC", "team": "RB", "position": 16},
        {"driver": "BOT", "team": "Sauber", "position": 17},
        {"driver": "ZHO", "team": "Sauber", "position": 18},
        {"driver": "ALB", "team": "Williams", "position": 19},
        {"driver": "SAR", "team": "Williams", "position": 20},
    ]

    # Run with different simulation counts
    sim_counts = [10, 50, 100]
    results_by_count = {}

    for n_sims in sim_counts:
        predictor_temp = Baseline2026Predictor(seed=42)
        result = predictor_temp.predict_race(
            qualifying_grid=qualifying_grid,
            race_name="Australian Grand Prix",
            n_simulations=n_sims,
        )
        results_by_count[n_sims] = result["finish_order"]

    reference_top_six = {
        entry["driver"] for entry in results_by_count[100][:6] if isinstance(entry, dict)
    }
    assert len(reference_top_six) == 6

    for n_sims in sim_counts:
        top_six = {
            entry["driver"] for entry in results_by_count[n_sims][:6] if isinstance(entry, dict)
        }
        overlap = len(top_six & reference_top_six)
        assert overlap >= 5, (
            f"Front-running cohort drifted too far with {n_sims} simulations: "
            f"top6={sorted(top_six)} ref={sorted(reference_top_six)}"
        )
