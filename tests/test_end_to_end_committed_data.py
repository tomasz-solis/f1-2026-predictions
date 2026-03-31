"""End-to-end prediction checks that use only committed local artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor

_DATA_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def predictor() -> Baseline2026Predictor:
    """Build the predictor from committed artifacts only."""
    if not (_DATA_DIR / "car_characteristics" / "2026_car_characteristics.json").exists():
        pytest.skip("Committed car characteristics not present")
    if not (_DATA_DIR / "driver_characteristics.json").exists():
        pytest.skip("Committed driver characteristics not present")
    return Baseline2026Predictor(seed=42)


def test_qualifying_produces_full_grid(predictor: Baseline2026Predictor) -> None:
    """Qualifying should produce 22 unique drivers ranked from P1 to P22."""
    result = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=30)
    grid = result["grid"]

    assert len(grid) == 22
    assert sorted(entry["position"] for entry in grid) == list(range(1, 23))
    assert len({entry["driver"] for entry in grid}) == 22


def test_race_produces_full_finish_order(predictor: Baseline2026Predictor) -> None:
    """Race prediction should produce a full ordered result with core fields present."""
    qualifying = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=30)
    race = predictor.predict_race(
        qualifying["grid"],
        weather="dry",
        race_name="Australian Grand Prix",
        n_simulations=30,
    )
    finish_order = race["finish_order"]

    assert len(finish_order) == 22
    assert sorted(entry["position"] for entry in finish_order) == list(range(1, 23))

    for entry in finish_order:
        assert "driver" in entry
        assert "team" in entry
        assert "podium_probability" in entry
        assert "dnf_probability" in entry
        assert 0 <= entry["podium_probability"] <= 100
        assert 0 <= entry["dnf_probability"] <= 1.0


def test_sprint_weekend_produces_all_sessions(predictor: Baseline2026Predictor) -> None:
    """Sprint weekends should produce valid sprint qualifying and sprint race outputs."""
    sprint_qualifying = predictor.predict_qualifying(
        2026,
        "Chinese Grand Prix",
        n_simulations=30,
        qualifying_stage="sprint",
    )
    assert len(sprint_qualifying["grid"]) == 22

    sprint_race = predictor.predict_sprint_race(
        sprint_qualifying["grid"],
        weather="dry",
        race_name="Chinese Grand Prix",
        n_simulations=30,
    )
    assert len(sprint_race["finish_order"]) == 22


def test_different_weather_changes_predictions(predictor: Baseline2026Predictor) -> None:
    """Dry and wet race predictions should not collapse to the same ordering."""
    qualifying = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=30)
    dry_race = predictor.predict_race(
        qualifying["grid"],
        weather="dry",
        race_name="Australian Grand Prix",
        n_simulations=30,
    )
    wet_race = predictor.predict_race(
        qualifying["grid"],
        weather="rain",
        race_name="Australian Grand Prix",
        n_simulations=30,
    )

    dry_order = [
        entry["driver"]
        for entry in sorted(dry_race["finish_order"], key=lambda row: row["position"])
    ]
    wet_order = [
        entry["driver"]
        for entry in sorted(wet_race["finish_order"], key=lambda row: row["position"])
    ]

    differences = sum(
        1
        for dry_driver, wet_driver in zip(dry_order, wet_order, strict=True)
        if dry_driver != wet_driver
    )
    assert differences >= 2, "Wet and dry predictions are too similar to trust weather sensitivity"
