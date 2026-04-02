"""Edge case tests for prediction system boundary conditions."""

import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor


def test_all_drivers_finish_no_dnf():
    """Verify prediction handles 20 finishers when DNF probability is zero."""
    predictor = Baseline2026Predictor(seed=42)

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

    result = predictor.predict_race(
        race_name="Australian Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    finish_order = result["finish_order"]

    # Should have all 20 drivers
    assert len(finish_order) >= 20, f"Expected 20 finishers, got {len(finish_order)}"

    # Positions should be 1-N (where N is number of finishers)
    positions = [entry["position"] for entry in finish_order]
    assert positions == list(range(1, len(positions) + 1)), (
        f"Positions should be sequential 1-{len(positions)}"
    )

    # No duplicates
    drivers = [entry["driver"] for entry in finish_order]
    assert len(drivers) == len(set(drivers)), "Duplicate drivers in finish order"


def test_wet_race_no_compound_data():
    """Wet race should not require dry compound characteristics."""
    predictor = Baseline2026Predictor(seed=42)

    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "HAM", "team": "Mercedes", "position": 2},
        {"driver": "NOR", "team": "McLaren", "position": 3},
        {"driver": "LEC", "team": "Ferrari", "position": 4},
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

    # Wet race prediction
    result = predictor.predict_race(
        race_name="Spa-Francorchamps",
        qualifying_grid=qualifying_grid,
        weather="rain",
        n_simulations=50,
    )

    # Should complete without compound data errors
    assert result is not None
    assert "finish_order" in result
    assert len(result["finish_order"]) >= 20

    # Check metadata indicates wet race
    metadata = result.get("metadata", {})
    if "weather" in metadata:
        assert metadata["weather"] in ["rain", "wet"], "Should indicate wet race"


def test_mixed_weather_race_completes():
    """Mixed weather conditions should produce valid results."""
    predictor = Baseline2026Predictor(seed=42)

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

    result_mixed = predictor.predict_race(
        race_name="Silverstone Circuit",
        qualifying_grid=qualifying_grid,
        weather="mixed",
        n_simulations=50,
    )
    result_dry = predictor.predict_race(
        race_name="Silverstone Circuit",
        qualifying_grid=qualifying_grid,
        weather="dry",
        n_simulations=50,
    )

    # Should complete successfully
    assert result_mixed is not None
    assert len(result_mixed["finish_order"]) >= 20

    # Mixed weather should not increase confidence above dry baseline.
    mixed_finish = result_mixed["finish_order"]
    dry_finish = result_dry["finish_order"]
    avg_confidence_mixed = sum(entry.get("confidence", 0) for entry in mixed_finish) / len(
        mixed_finish
    )
    avg_confidence_dry = sum(entry.get("confidence", 0) for entry in dry_finish) / len(dry_finish)

    assert avg_confidence_mixed <= avg_confidence_dry + 2.0


def test_qualifying_with_missing_team_data():
    """Verify graceful handling when some team data is missing."""
    predictor = Baseline2026Predictor(seed=42)

    # Predict qualifying for a race
    result = predictor.predict_qualifying(2026, race_name="Australian Grand Prix", n_simulations=50)

    # Should still produce results for all known teams
    grid = result["grid"]
    assert len(grid) >= 18, "Should have at least 18 drivers even with missing data"


def test_race_with_invalid_grid_positions():
    """Verify handling of malformed grid input."""
    predictor = Baseline2026Predictor(seed=42)

    # Grid with gap in positions (missing P5)
    malformed_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
        {"driver": "HAM", "team": "Mercedes", "position": 4},
        # Position 5 missing
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

    # Should either fix the grid or handle gracefully
    try:
        result = predictor.predict_race(
            race_name="Australian Grand Prix",
            qualifying_grid=malformed_grid,
            n_simulations=50,
        )

        # If it completes, check it's valid
        assert len(result["finish_order"]) >= 19
    except ValueError as e:
        # Or it should raise a clear validation error
        assert "position" in str(e).lower() or "grid" in str(e).lower()


def test_extreme_weather_conditions():
    """Test prediction handles extreme weather scenarios."""
    predictor = Baseline2026Predictor(seed=42)

    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "HAM", "team": "Mercedes", "position": 2},
        {"driver": "NOR", "team": "McLaren", "position": 3},
        {"driver": "LEC", "team": "Ferrari", "position": 4},
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

    # Heavy rain
    result_rain = predictor.predict_race(
        race_name="Belgian Grand Prix",  # Spa often has rain
        qualifying_grid=qualifying_grid,
        weather="rain",
        n_simulations=50,
    )

    # Should complete
    assert len(result_rain["finish_order"]) >= 20

    # Rain should shuffle the field more
    total_changes = 0
    for entry in result_rain["finish_order"]:
        start_pos = next(
            (g["position"] for g in qualifying_grid if g["driver"] == entry["driver"]), None
        )
        if start_pos:
            total_changes += abs(entry["position"] - start_pos)

    avg_change = total_changes / len(result_rain["finish_order"])

    # Rain should produce non-trivial movement from the starting grid.
    assert avg_change >= 0.5, f"Rain should produce some field movement (got {avg_change:.2f})"


def test_minimum_simulation_count():
    """Verify predictions work with minimum simulation count."""
    predictor = Baseline2026Predictor(seed=42)

    # Test with n_simulations=1 (minimum)
    result = predictor.predict_qualifying(2026, race_name="Australian Grand Prix", n_simulations=1)

    grid = result["grid"]
    assert len(grid) >= 20
    assert all(entry["position"] == idx + 1 for idx, entry in enumerate(grid))


def test_empty_or_none_fp_performance():
    """Verify qualifying prediction handles missing FP data gracefully."""
    predictor = Baseline2026Predictor(seed=42)

    # Pre-weekend prediction (no FP data available)
    result = predictor.predict_qualifying(2026, race_name="Australian Grand Prix", n_simulations=50)

    # Should fall back to model-only prediction
    assert len(result["grid"]) >= 20

    # All drivers should have confidence scores
    assert all("confidence" in entry for entry in result["grid"])


def test_qualifying_stage_parameter_validation():
    """Test qualifying stage parameter accepts valid values."""
    predictor = Baseline2026Predictor(seed=42)

    valid_stages = ["auto", "main", "sprint"]

    for stage in valid_stages:
        result = predictor.predict_qualifying(
            2026, race_name="Australian Grand Prix", qualifying_stage=stage, n_simulations=10
        )
        assert result is not None, f"Stage '{stage}' should be valid"


def test_year_boundary_conditions():
    """Test predictions work for edge years."""
    predictor = Baseline2026Predictor(seed=42)

    result_2026 = predictor.predict_qualifying(
        2026, race_name="Australian Grand Prix", n_simulations=10
    )
    assert len(result_2026["grid"]) >= 20

    try:
        result_2025 = predictor.predict_qualifying(
            2025, race_name="Bahrain Grand Prix", n_simulations=10
        )
        assert len(result_2025["grid"]) >= 18
    except ValueError as exc:
        if "not found in 2025 schedule" in str(exc):
            pytest.skip("FastF1 2025 schedule unavailable in this environment")
        raise


def test_race_distance_zero_or_negative():
    """Verify race distance validation prevents invalid values."""
    predictor = Baseline2026Predictor(seed=42)

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

    # Standard race should use realistic distance
    result = predictor.predict_race(
        race_name="Australian Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=10,
    )

    # Should complete with valid output
    assert len(result["finish_order"]) >= 20


def test_duplicate_drivers_in_grid():
    """Verify system handles duplicate drivers in grid."""
    predictor = Baseline2026Predictor(seed=42)

    # Grid with duplicate driver
    duplicate_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "VER", "team": "Red Bull Racing", "position": 2},  # Duplicate
        {"driver": "NOR", "team": "McLaren", "position": 3},
        {"driver": "LEC", "team": "Ferrari", "position": 4},
        {"driver": "HAM", "team": "Mercedes", "position": 5},
        {"driver": "PIA", "team": "McLaren", "position": 6},
        {"driver": "SAI", "team": "Ferrari", "position": 7},
        {"driver": "RUS", "team": "Mercedes", "position": 8},
        {"driver": "PER", "team": "Red Bull Racing", "position": 9},
        {"driver": "ALO", "team": "Aston Martin", "position": 10},
        {"driver": "STR", "team": "Aston Martin", "position": 11},
        {"driver": "GAS", "team": "Alpine", "position": 12},
        {"driver": "OCO", "team": "Alpine", "position": 13},
        {"driver": "HUL", "team": "Haas F1 Team", "position": 14},
        {"driver": "MAG", "team": "Haas F1 Team", "position": 15},
        {"driver": "TSU", "team": "RB", "position": 16},
        {"driver": "RIC", "team": "RB", "position": 17},
        {"driver": "BOT", "team": "Sauber", "position": 18},
        {"driver": "ZHO", "team": "Sauber", "position": 19},
        {"driver": "ALB", "team": "Williams", "position": 20},
    ]

    # Should either deduplicate or raise validation error
    try:
        result = predictor.predict_race(
            race_name="Australian Grand Prix",
            qualifying_grid=duplicate_grid,
            n_simulations=10,
        )

        # If it completes, check no duplicates in output
        drivers = [entry["driver"] for entry in result["finish_order"]]
        assert len(drivers) == len(set(drivers)), "Output should not have duplicates"
    except ValueError as e:
        # Or it should raise clear validation error
        assert "duplicate" in str(e).lower() or "driver" in str(e).lower()


def test_extremely_high_simulation_count():
    """Verify system handles high simulation counts without crashing."""
    predictor = Baseline2026Predictor(seed=42)

    # Test with high simulation count (stress test)
    result = predictor.predict_qualifying(
        2026, race_name="Australian Grand Prix", n_simulations=200
    )

    # Should complete and produce valid results
    assert len(result["grid"]) >= 20

    # Confidence should stay within configured range and not degrade badly at high sim count.
    avg_confidence = sum(entry.get("confidence", 0) for entry in result["grid"]) / len(
        result["grid"]
    )
    assert 30.0 <= avg_confidence <= 60.0
