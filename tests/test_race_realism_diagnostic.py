"""Diagnostic realism checks without stdout side effects."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.predictors.baseline_2026 import Baseline2026Predictor


@pytest.fixture(scope="module")
def quali_and_race():
    predictor = Baseline2026Predictor(seed=42)
    qualifying = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=80)
    race = predictor.predict_race(
        qualifying["grid"],
        weather="dry",
        race_name="Australian Grand Prix",
        n_simulations=80,
    )
    return qualifying, race


def test_diagnostic_grid_to_race_correlation_is_finite(quali_and_race):
    qualifying, race = quali_and_race

    grid_positions = {entry["driver"]: entry["position"] for entry in qualifying["grid"]}
    race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}
    drivers = sorted(grid_positions.keys())

    correlation, _ = spearmanr(
        [grid_positions[driver] for driver in drivers],
        [race_positions[driver] for driver in drivers],
    )
    assert correlation >= 0.60, (
        f"Grid-to-race Spearman correlation is {correlation:.3f}. "
        "Expected >= 0.60 so qualifying still matters."
    )


def test_diagnostic_mean_position_change_is_bounded(quali_and_race):
    qualifying, race = quali_and_race

    grid_positions = {entry["driver"]: entry["median_position"] for entry in qualifying["grid"]}
    race_positions = {entry["driver"]: entry["median_position"] for entry in race["finish_order"]}
    position_changes = [abs(grid_positions[d] - race_positions[d]) for d in grid_positions]
    mean_change = float(np.mean(position_changes))

    assert 0.5 <= mean_change <= 4.0, (
        f"Mean position change is {mean_change:.2f}. "
        "Expected 0.5-4.0 positions for a believable race spread."
    )


def test_diagnostic_pole_sitter_podium_probability_is_valid(quali_and_race):
    qualifying, race = quali_and_race

    pole_driver = qualifying["grid"][0]["driver"]
    pole_finish = next(entry for entry in race["finish_order"] if entry["driver"] == pole_driver)
    podium_probability = float(pole_finish["podium_probability"])
    assert podium_probability >= 35.0, (
        f"Pole sitter top-3 probability is {podium_probability:.1f}%. "
        "Expected at least 35% so pole still carries a real edge."
    )


def test_diagnostic_top5_podium_share_is_valid(quali_and_race):
    qualifying, race = quali_and_race

    top5_drivers = {entry["driver"] for entry in qualifying["grid"][:5]}
    total_podium_probability = 0.0
    top5_podium_probability = 0.0
    for entry in race["finish_order"]:
        probability = float(entry["podium_probability"])
        total_podium_probability += probability
        if entry["driver"] in top5_drivers:
            top5_podium_probability += probability

    top5_fraction = (
        (top5_podium_probability / total_podium_probability) * 100.0
        if total_podium_probability > 0
        else 0.0
    )
    assert top5_fraction >= 55.0, (
        f"Top-5 qualifiers hold only {top5_fraction:.1f}% of podium probability. "
        "Expected at least 55% so the front of the grid still matters."
    )


def test_diagnostic_top_grid_falloff_frequency_is_bounded(quali_and_race):
    qualifying, race = quali_and_race

    top3_drivers = [entry["driver"] for entry in qualifying["grid"][:3]]
    falloff_count = 0
    for driver in top3_drivers:
        entry = next(result for result in race["finish_order"] if result["driver"] == driver)
        if int(entry["p95"]) >= 10:
            falloff_count += 1

    assert falloff_count <= 1, (
        f"{falloff_count} of 3 top-grid drivers have P95 >= 10. "
        "Expected at most 1 in a realistic dry race."
    )


def test_diagnostic_podium_probability_ordering_has_limited_inversions(quali_and_race):
    qualifying, race = quali_and_race
    del qualifying

    top8 = sorted(race["finish_order"], key=lambda item: item["position"])[:8]
    inversions = 0
    for index in range(1, len(top8)):
        previous_probability = float(top8[index - 1]["podium_probability"])
        current_probability = float(top8[index]["podium_probability"])
        if current_probability > previous_probability + 5.0:
            inversions += 1

    assert inversions <= 2, (
        f"Found {inversions} podium-probability inversions in top-8. "
        "Expected at most 2 for coherent probability ordering."
    )
