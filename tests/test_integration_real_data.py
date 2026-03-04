"""Integration test that exercises qualifying prediction with real FastF1 session data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor
from src.utils.lineups import get_lineups

# Skip module when FastF1 is unavailable.
pytest.importorskip("fastf1")


@pytest.mark.integration
@pytest.mark.slow
def test_prediction_with_real_2024_bahrain_data():
    """Run a real-data qualifying prediction and verify lineup/grid consistency."""
    if os.getenv("RUN_REAL_DATA_TESTS") != "1":
        pytest.skip("Set RUN_REAL_DATA_TESTS=1 to enable real FastF1 integration tests")

    predictor = Baseline2026Predictor(data_dir=Path("data/processed"), seed=42)
    predictor.load_data()

    result = predictor.predict_qualifying(
        year=2024,
        race_name="Bahrain Grand Prix",
        n_simulations=50,
    )

    expected_lineups = get_lineups(2024, "Bahrain Grand Prix")
    expected_grid_size = sum(len(drivers) for drivers in expected_lineups.values())

    assert "grid" in result
    assert len(result["grid"]) == expected_grid_size
    assert all("driver" in entry for entry in result["grid"])
    positions = [entry["position"] for entry in result["grid"]]
    assert len(positions) == len(set(positions))
