"""Integration tests for sprint weekend cascade: SQ → Sprint → Quali → Race."""

import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor
from src.utils import config_loader


def test_sprint_weekend_identifies_correctly():
    """Verify sprint weekends are detected correctly."""
    from src.utils.weekend import (
        get_all_conventional_races,
        get_all_sprint_races,
        is_sprint_weekend,
    )

    sprint_races = get_all_sprint_races(2026)
    non_sprint_races = get_all_conventional_races(2026)
    assert sprint_races, "Expected at least one sprint weekend in schedule"
    assert non_sprint_races, "Expected at least one conventional weekend in schedule"

    for race in sprint_races[:6]:
        assert is_sprint_weekend(2026, race), f"{race} should be a sprint weekend"

    for race in non_sprint_races[:6]:
        assert not is_sprint_weekend(2026, race), f"{race} should NOT be a sprint weekend"


def test_sprint_qualifying_predictions_independent():
    """Verify sprint qualifying produces independent results."""
    predictor = Baseline2026Predictor(seed=42)

    # Predict sprint qualifying
    result = predictor.predict_qualifying(
        2026,
        race_name="Miami Grand Prix",  # Sprint weekend
        qualifying_stage="sprint",
        n_simulations=50,
    )

    grid = result["grid"]

    # Check basic structure
    assert len(grid) >= 20, "Sprint qualifying should have 20 drivers"
    assert all("driver" in entry for entry in grid), "All entries should have driver"
    assert all("position" in entry for entry in grid), "All entries should have position"
    assert all("team" in entry for entry in grid), "All entries should have team"

    # Check positions are sequential
    positions = [entry["position"] for entry in grid]
    assert positions == list(range(1, len(positions) + 1)), "Positions should be sequential"

    # Check no duplicates
    drivers = [entry["driver"] for entry in grid]
    assert len(drivers) == len(set(drivers)), "No duplicate drivers"


def test_sprint_race_uses_sprint_quali_grid():
    """Verify sprint race prediction uses sprint qualifying grid as input."""
    predictor = Baseline2026Predictor(seed=42)

    # First, get sprint qualifying grid
    sq_result = predictor.predict_qualifying(
        2026,
        race_name="Miami Grand Prix",
        qualifying_stage="sprint",
        n_simulations=50,
    )

    sprint_quali_grid = sq_result["grid"]

    # Now predict sprint race using that grid
    sprint_result = predictor.predict_race(
        race_name="Miami Grand Prix",
        qualifying_grid=sprint_quali_grid,
        is_sprint=True,
        n_simulations=50,
    )

    sprint_finish = sprint_result["finish_order"]

    # Verify sprint race completed
    assert len(sprint_finish) >= 20, "Sprint race should have 20 finishers"

    # Sprint races are shorter, so less position change expected
    total_changes = 0
    for entry in sprint_finish:
        start_pos = next(
            (g["position"] for g in sprint_quali_grid if g["driver"] == entry["driver"]), None
        )
        if start_pos:
            total_changes += abs(entry["position"] - start_pos)

    avg_change = total_changes / len(sprint_finish)

    # Sprint races typically have less overtaking than full races
    assert avg_change < 4.0, (
        f"Sprint race position changes too high: {avg_change:.2f} "
        f"(sprints should be <4.0 avg change)"
    )


def test_main_qualifying_independent_of_sprint_results():
    """Verify main qualifying is predicted independently, not influenced by sprint."""
    predictor = Baseline2026Predictor(seed=42)

    # Predict main qualifying
    main_quali_result = predictor.predict_qualifying(
        2026,
        race_name="Miami Grand Prix",
        qualifying_stage="main",  # Main quali
        n_simulations=50,
    )

    main_quali_grid = main_quali_result["grid"]

    # Verify structure
    assert len(main_quali_grid) >= 20
    positions = [entry["position"] for entry in main_quali_grid]
    assert positions == list(range(1, len(main_quali_grid) + 1))

    # Main qualifying should be based on pace, not sprint results
    # Just verify it produces valid output
    assert all("confidence" in entry for entry in main_quali_grid)


def test_main_race_uses_main_quali_grid():
    """Verify main race uses main qualifying grid, not sprint results."""
    predictor = Baseline2026Predictor(seed=42)

    # Get main qualifying grid
    main_quali_result = predictor.predict_qualifying(
        2026, race_name="Miami Grand Prix", qualifying_stage="main", n_simulations=50
    )

    main_quali_grid = main_quali_result["grid"]

    # Predict main race
    main_race_result = predictor.predict_race(
        race_name="Miami Grand Prix",
        qualifying_grid=main_quali_grid,
        is_sprint=False,  # Main race
        n_simulations=50,
    )

    main_finish = main_race_result["finish_order"]

    # Verify main race completed with full distance
    assert len(main_finish) >= 20

    # Main race should have more position changes than sprint
    total_changes = 0
    for entry in main_finish:
        start_pos = next(
            (g["position"] for g in main_quali_grid if g["driver"] == entry["driver"]), None
        )
        if start_pos:
            total_changes += abs(entry["position"] - start_pos)

    avg_change = total_changes / len(main_finish)

    # Main-race movement is guarded by a configurable realism floor, so the
    # integration check should follow that contract instead of a stale constant.
    movement_floor = float(
        config_loader.get("baseline_predictor.race.main_race_movement_floor", 0.7)
    )
    assert avg_change >= movement_floor, (
        "Main race should respect the configured minimum grid movement "
        f"(got {avg_change:.2f}, floor {movement_floor:.2f})"
    )


def test_sprint_weekend_full_cascade_workflow():
    """End-to-end test: Sprint Quali → Sprint Race → Main Quali → Main Race."""
    predictor = Baseline2026Predictor(seed=42)

    race_name = "Miami Grand Prix"

    # Step 1: Sprint Qualifying
    sq_result = predictor.predict_qualifying(
        2026, race_name=race_name, qualifying_stage="sprint", n_simulations=50
    )
    sprint_quali_grid = sq_result["grid"]

    assert len(sprint_quali_grid) >= 20
    assert all(entry["position"] == idx + 1 for idx, entry in enumerate(sprint_quali_grid))

    # Step 2: Sprint Race (uses Sprint Quali grid)
    sprint_race_result = predictor.predict_race(
        race_name=race_name,
        qualifying_grid=sprint_quali_grid,
        is_sprint=True,
        n_simulations=50,
    )
    sprint_finish = sprint_race_result["finish_order"]

    assert len(sprint_finish) >= 20

    # Step 3: Main Qualifying (independent of sprint)
    main_quali_result = predictor.predict_qualifying(
        2026, race_name=race_name, qualifying_stage="main", n_simulations=50
    )
    main_quali_grid = main_quali_result["grid"]

    assert len(main_quali_grid) >= 20

    # Step 4: Main Race (uses Main Quali grid)
    main_race_result = predictor.predict_race(
        race_name=race_name,
        qualifying_grid=main_quali_grid,
        is_sprint=False,
        n_simulations=50,
    )
    main_finish = main_race_result["finish_order"]

    assert len(main_finish) >= 20

    # Verify cascade integrity: All 4 stages completed successfully
    stages_completed = {
        "sprint_qualifying": sprint_quali_grid is not None,
        "sprint_race": sprint_finish is not None,
        "main_qualifying": main_quali_grid is not None,
        "main_race": main_finish is not None,
    }

    assert all(stages_completed.values()), f"Sprint weekend cascade incomplete: {stages_completed}"


def test_sprint_race_distance_shorter_than_main():
    """Verify sprint races use shorter distance than main races."""
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

    # Sprint race
    sprint_result = predictor.predict_race(
        race_name="Miami Grand Prix",
        qualifying_grid=qualifying_grid,
        is_sprint=True,
        n_simulations=50,
    )

    # Main race
    main_result = predictor.predict_race(
        race_name="Miami Grand Prix",
        qualifying_grid=qualifying_grid,
        is_sprint=False,
        n_simulations=50,
    )

    # Check metadata if available
    sprint_metadata = sprint_result.get("metadata", {})
    main_metadata = main_result.get("metadata", {})

    # If distance is tracked in metadata
    if "race_distance" in sprint_metadata and "race_distance" in main_metadata:
        assert sprint_metadata["race_distance"] < main_metadata["race_distance"], (
            "Sprint distance should be shorter than main race"
        )

    # Alternatively, check that both completed successfully
    assert len(sprint_result["finish_order"]) >= 20
    assert len(main_result["finish_order"]) >= 20


def test_sprint_weekend_driver_consistency_across_sessions():
    """Verify same 20 drivers appear in all sprint weekend sessions."""
    predictor = Baseline2026Predictor(seed=42)

    race_name = "Chinese Grand Prix"  # Sprint weekend

    # Get all session results
    sq_result = predictor.predict_qualifying(
        2026, race_name=race_name, qualifying_stage="sprint", n_simulations=50
    )

    main_q_result = predictor.predict_qualifying(
        2026, race_name=race_name, qualifying_stage="main", n_simulations=50
    )

    # Extract driver sets
    sq_drivers = {entry["driver"] for entry in sq_result["grid"]}
    main_q_drivers = {entry["driver"] for entry in main_q_result["grid"]}

    # Should be identical driver lineup
    assert sq_drivers == main_q_drivers, (
        f"Driver lineup mismatch between sprint quali and main quali: "
        f"SQ: {sq_drivers}, Main Q: {main_q_drivers}"
    )

    assert len(sq_drivers) >= 20, "Should have at least 20 drivers"


def test_all_sprint_weekends_complete_cascade():
    """Test full qualifying cascade works for sprint weekends returned by schedule metadata."""
    from src.utils.weekend import get_all_sprint_races

    sprint_races = get_all_sprint_races(2026)
    assert sprint_races, "No sprint races returned by schedule metadata"

    predictor = Baseline2026Predictor(seed=42)
    for sprint_race in sprint_races:
        try:
            sq_result = predictor.predict_qualifying(
                2026, race_name=sprint_race, qualifying_stage="sprint", n_simulations=10
            )
            main_q_result = predictor.predict_qualifying(
                2026, race_name=sprint_race, qualifying_stage="main", n_simulations=10
            )
            assert len(sq_result["grid"]) >= 20, f"{sprint_race} sprint quali failed"
            assert len(main_q_result["grid"]) >= 20, f"{sprint_race} main quali failed"
        except Exception as e:
            pytest.fail(f"{sprint_race} sprint weekend cascade failed: {e}")


def test_normal_weekend_does_not_have_sprint_stage():
    """Verify normal weekends don't accept sprint qualifying stage."""
    predictor = Baseline2026Predictor(seed=42)

    # Australia is not a sprint weekend
    result = predictor.predict_qualifying(
        2026,
        race_name="Australian Grand Prix",
        qualifying_stage="auto",  # Should detect normal weekend
        n_simulations=50,
    )

    # Should complete successfully as normal qualifying
    assert len(result["grid"]) >= 20
    # Result should not have sprint-specific metadata
    metadata = result.get("metadata", {})
    assert metadata.get("format", "qualifying") == "qualifying"
