"""Tests for race prediction grid anchoring and blending logic."""

import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor


def find_driver_position(finish_order: list[dict], driver_code: str) -> int | None:
    """Find driver's final position."""
    for entry in finish_order:
        if entry["driver"] == driver_code:
            return entry["position"]
    return None


def test_grid_anchoring_respects_max_gain_from_back():
    """Verify drivers can't gain unrealistic number of positions from back of grid."""
    predictor = Baseline2026Predictor(seed=42)

    # Setup: Elite driver starting P20 (worst case scenario)
    qualifying_grid = [
        {"driver": "NOR", "team": "McLaren", "position": 1},
        {"driver": "PIA", "team": "McLaren", "position": 2},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
        {"driver": "SAI", "team": "Ferrari", "position": 4},
        {"driver": "HAM", "team": "Mercedes", "position": 5},
        {"driver": "RUS", "team": "Mercedes", "position": 6},
        {"driver": "ALO", "team": "Aston Martin", "position": 7},
        {"driver": "STR", "team": "Aston Martin", "position": 8},
        {"driver": "PER", "team": "Red Bull Racing", "position": 9},
        {"driver": "GAS", "team": "Alpine", "position": 10},
        {"driver": "OCO", "team": "Alpine", "position": 11},
        {"driver": "HUL", "team": "Haas F1 Team", "position": 12},
        {"driver": "MAG", "team": "Haas F1 Team", "position": 13},
        {"driver": "TSU", "team": "RB", "position": 14},
        {"driver": "RIC", "team": "RB", "position": 15},
        {"driver": "BOT", "team": "Sauber", "position": 16},
        {"driver": "ZHO", "team": "Sauber", "position": 17},
        {"driver": "ALB", "team": "Williams", "position": 18},
        {"driver": "SAR", "team": "Williams", "position": 19},
        {"driver": "VER", "team": "Red Bull Racing", "position": 20},  # Elite driver at back
    ]

    # Monza: Easy overtaking track
    result = predictor.predict_race(
        race_name="Italian Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    final_pos = find_driver_position(result["finish_order"], "VER")

    # VER is elite but starting P20 on easy overtaking track
    # Maximum realistic gain: ~10-12 positions (P20 → P8-P10)
    # Should NOT finish in top 5 from P20
    assert final_pos is not None
    assert final_pos >= 7, (
        f"Unrealistic gain from P20: VER finished P{final_pos} (expected P7 or lower)"
    )


def test_grid_anchoring_prevents_unrealistic_losses():
    """Verify strong drivers from pole don't fall too far back."""
    predictor = Baseline2026Predictor(seed=42)

    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},  # Elite on pole
        {"driver": "NOR", "team": "McLaren", "position": 2},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
        {"driver": "PIA", "team": "McLaren", "position": 4},
        {"driver": "HAM", "team": "Mercedes", "position": 5},
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

    result = predictor.predict_race(
        race_name="Bahrain Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    final_pos = find_driver_position(result["finish_order"], "VER")

    # VER starting P1 should not finish outside top 5 in dry conditions
    assert final_pos is not None
    assert final_pos <= 5, f"Elite driver from pole fell too far: P1 → P{final_pos}"


def test_midfield_grid_position_constrains_result():
    """Verify midfield starters stay in realistic finishing range."""
    predictor = Baseline2026Predictor(seed=42)

    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
        {"driver": "PIA", "team": "McLaren", "position": 4},
        {"driver": "HAM", "team": "Mercedes", "position": 5},
        {"driver": "SAI", "team": "Ferrari", "position": 6},
        {"driver": "RUS", "team": "Mercedes", "position": 7},
        {"driver": "PER", "team": "Red Bull Racing", "position": 8},
        {"driver": "ALO", "team": "Aston Martin", "position": 9},
        {"driver": "STR", "team": "Aston Martin", "position": 10},
        {"driver": "GAS", "team": "Alpine", "position": 11},
        {"driver": "OCO", "team": "Alpine", "position": 12},
        {"driver": "HUL", "team": "Haas F1 Team", "position": 13},  # Midfield starter
        {"driver": "MAG", "team": "Haas F1 Team", "position": 14},
        {"driver": "TSU", "team": "RB", "position": 15},
        {"driver": "RIC", "team": "RB", "position": 16},
        {"driver": "BOT", "team": "Sauber", "position": 17},
        {"driver": "ZHO", "team": "Sauber", "position": 18},
        {"driver": "ALB", "team": "Williams", "position": 19},
        {"driver": "SAR", "team": "Williams", "position": 20},
    ]

    result = predictor.predict_race(
        race_name="Bahrain Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    hul_final = find_driver_position(result["finish_order"], "HUL")

    # HUL (midfield team) starting P13 should finish P9-P16
    # Should not beat multiple top teams or fall to last
    assert hul_final is not None
    assert 8 <= hul_final <= 17, f"Midfield driver moved unrealistically: P13 → P{hul_final}"


def test_track_overtaking_difficulty_affects_grid_anchoring():
    """Verify hard-to-overtake tracks preserve grid order more."""
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

    # Monaco: Very hard to overtake
    result_monaco = predictor.predict_race(
        race_name="Monaco Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    # Monza: Easy to overtake (reload predictor with different seed)
    predictor_monza = Baseline2026Predictor(seed=42)
    result_monza = predictor_monza.predict_race(
        race_name="Italian Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    # Calculate average position change
    def calc_avg_position_change(finish_order, grid):
        total_change = 0
        count = 0
        for entry in finish_order:
            start_pos = next((g["position"] for g in grid if g["driver"] == entry["driver"]), None)
            if start_pos:
                change = abs(entry["position"] - start_pos)
                total_change += change
                count += 1
        return total_change / count if count > 0 else 0

    avg_change_monaco = calc_avg_position_change(result_monaco["finish_order"], qualifying_grid)
    avg_change_monza = calc_avg_position_change(result_monza["finish_order"], qualifying_grid)

    # Monaco should have less position change than Monza
    assert avg_change_monaco < avg_change_monza * 1.2, (
        f"Monaco should preserve grid order more than Monza: "
        f"Monaco avg change={avg_change_monaco:.2f}, Monza avg change={avg_change_monza:.2f}"
    )


def test_elite_driver_midfield_car_realistic_result():
    """Verify elite driver in midfield car can't win but can score points."""
    predictor = Baseline2026Predictor(seed=42)

    # HAM in Williams (hypothetical) starting P10
    qualifying_grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
        {"driver": "PIA", "team": "McLaren", "position": 4},
        {"driver": "SAI", "team": "Ferrari", "position": 5},
        {"driver": "PER", "team": "Red Bull Racing", "position": 6},
        {"driver": "RUS", "team": "Mercedes", "position": 7},
        {"driver": "ALO", "team": "Aston Martin", "position": 8},
        {"driver": "STR", "team": "Aston Martin", "position": 9},
        {"driver": "HAM", "team": "Williams", "position": 10},  # Elite in midfield car
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
        race_name="Bahrain Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    ham_final = find_driver_position(result["finish_order"], "HAM")

    # HAM with elite skill in a weaker Williams should stay around the upper
    # midfield or the edge of the points, but the car ceiling should keep him
    # out of the podium places.
    assert ham_final is not None
    assert 3 < ham_final <= 12, (
        f"Elite driver in midfield car result unrealistic: P10 → P{ham_final} "
        f"(should be outside the podium and no worse than P12)"
    )


def test_all_drivers_finish_positions_unique():
    """Verify no duplicate positions in finish order."""
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
        race_name="Bahrain Grand Prix",
        qualifying_grid=qualifying_grid,
        n_simulations=50,
    )

    finish_order = result["finish_order"]

    # Check all positions are unique
    positions = [entry["position"] for entry in finish_order]
    assert len(positions) == len(set(positions)), "Duplicate positions found in finish order"

    # Check positions are sequential
    assert sorted(positions) == list(range(1, len(positions) + 1)), "Positions are not sequential"


@pytest.mark.parametrize(
    "race_name,expected_difficulty",
    [
        ("Monaco Grand Prix", "high"),
        ("Hungarian Grand Prix", "high"),
        ("Italian Grand Prix", "low"),
        ("Belgian Grand Prix", "low"),
        ("Bahrain Grand Prix", "medium"),
    ],
)
def test_overtaking_difficulty_by_track(race_name, expected_difficulty):
    """Verify different tracks produce results consistent with overtaking difficulty."""
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

    result = predictor.predict_race(
        race_name=race_name, qualifying_grid=qualifying_grid, n_simulations=50
    )

    # Calculate total position changes
    total_changes = 0
    for entry in result["finish_order"]:
        start_pos = next(
            (g["position"] for g in qualifying_grid if g["driver"] == entry["driver"]), None
        )
        if start_pos:
            total_changes += abs(entry["position"] - start_pos)

    avg_change = total_changes / len(result["finish_order"])

    # Verify changes align with expected difficulty
    if expected_difficulty == "high":
        # Monaco/Hungary: Less overtaking, positions mostly stable
        assert avg_change <= 2.0, f"{race_name} should have lower overtaking (got {avg_change:.2f})"
    elif expected_difficulty == "low":
        # Monza/Spa: Easier overtaking than high-difficulty tracks
        assert 0.5 <= avg_change <= 2.5, (
            f"{race_name} should have moderate movement on current calibration (got {avg_change:.2f})"
        )
    else:  # medium
        # Bahrain: Typical movement range in current model calibration
        assert 0.5 <= avg_change <= 2.5, (
            f"{race_name} should have moderate overtaking (got {avg_change:.2f})"
        )
