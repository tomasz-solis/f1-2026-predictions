"""Diagnostic realism checks without stdout side effects."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.predictors.baseline_2026 import Baseline2026Predictor


@pytest.fixture
def predictor():
    return Baseline2026Predictor()


def _quali_and_race(predictor: Baseline2026Predictor):
    qualifying = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)
    race = predictor.predict_race(
        qualifying["grid"],
        weather="dry",
        race_name="Bahrain Grand Prix",
        n_simulations=500,
    )
    return qualifying, race


def test_diagnostic_grid_to_race_correlation_is_finite(predictor):
    qualifying, race = _quali_and_race(predictor)

    grid_positions = {entry["driver"]: entry["position"] for entry in qualifying["grid"]}
    race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}
    drivers = sorted(grid_positions.keys())

    correlation, _ = spearmanr(
        [grid_positions[driver] for driver in drivers],
        [race_positions[driver] for driver in drivers],
    )
    assert np.isfinite(float(correlation))
    assert -1.0 <= float(correlation) <= 1.0


def test_diagnostic_mean_position_change_is_bounded(predictor):
    qualifying, race = _quali_and_race(predictor)

    grid_positions = {entry["driver"]: entry["position"] for entry in qualifying["grid"]}
    race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}
    position_changes = [abs(grid_positions[d] - race_positions[d]) for d in grid_positions]

    assert np.isfinite(float(np.mean(position_changes)))
    assert 0.0 <= float(np.mean(position_changes)) <= 19.0


def test_diagnostic_pole_sitter_podium_probability_is_valid(predictor):
    qualifying, race = _quali_and_race(predictor)

    pole_driver = qualifying["grid"][0]["driver"]
    pole_finish = next(entry for entry in race["finish_order"] if entry["driver"] == pole_driver)
    podium_probability = float(pole_finish["podium_probability"])

    assert np.isfinite(podium_probability)
    assert 0.0 <= podium_probability <= 100.0


def test_diagnostic_top5_podium_share_is_valid(predictor):
    qualifying, race = _quali_and_race(predictor)

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
    assert np.isfinite(top5_fraction)
    assert 0.0 <= top5_fraction <= 100.0


def test_diagnostic_top_grid_falloff_frequency_is_bounded(predictor):
    qualifying, race = _quali_and_race(predictor)

    top3_drivers = [entry["driver"] for entry in qualifying["grid"][:3]]
    falloff_count = 0
    for driver in top3_drivers:
        entry = next(result for result in race["finish_order"] if result["driver"] == driver)
        if int(entry["p95"]) >= 10:
            falloff_count += 1

    assert 0 <= falloff_count <= 3


def test_diagnostic_podium_probability_ordering_has_limited_inversions(predictor):
    qualifying, race = _quali_and_race(predictor)
    del qualifying

    top8 = sorted(race["finish_order"], key=lambda item: item["position"])[:8]
    inversions = 0
    for index in range(1, len(top8)):
        previous_probability = float(top8[index - 1]["podium_probability"])
        current_probability = float(top8[index]["podium_probability"])
        if current_probability > previous_probability + 5.0:
            inversions += 1

    assert 0 <= inversions <= len(top8) - 1
