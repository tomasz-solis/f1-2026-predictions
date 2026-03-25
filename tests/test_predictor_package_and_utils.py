"""Tests for predictor package exports and small utility helpers."""

from src.predictors import Baseline2026Predictor
from src.predictors.baseline_2026 import Baseline2026Predictor as CanonicalPredictor
from src.utils.driver_numbers import (
    get_all_drivers_2026,
    get_driver_from_abbreviation,
    get_driver_number,
    get_team_drivers_2026,
)
from src.utils.performance_tracker import PerformanceTracker


def test_predictor_package_exports_canonical_predictor():
    """The predictor package should expose the canonical baseline entry point."""
    assert Baseline2026Predictor is CanonicalPredictor


def test_driver_number_utilities():
    assert get_driver_number("Lando Norris") == 4
    assert get_driver_number("Max Verstappen", use_champion_number=True) == 1
    assert get_driver_number("Unknown Driver") is None

    assert get_driver_from_abbreviation("nor") == "Lando Norris"
    assert get_driver_from_abbreviation("zzz") is None

    all_drivers = get_all_drivers_2026()
    assert "Lewis Hamilton" in all_drivers
    assert len(all_drivers) >= 20

    assert get_team_drivers_2026("MCLAREN") == ["Lando Norris", "Oscar Piastri"]
    assert get_team_drivers_2026("sauber") == ["Nico Hulkenberg", "Gabriel Bortoleto"]
    assert get_team_drivers_2026("unknown") == []


def test_performance_tracker_records_and_exports():
    tracker = PerformanceTracker()

    assert tracker.get_average("mae") is None

    tracker.record("mae", 2.5)
    tracker.add_result("mae", 3.5)
    tracker.record("winner_accuracy", 1.0)

    assert tracker.get_average("mae") == 3.0
    exported = tracker.to_dict()

    assert exported["mae"]["values"] == [2.5, 3.5]
    assert exported["mae"]["avg"] == 3.0
    assert exported["winner_accuracy"]["avg"] == 1.0
