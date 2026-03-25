"""Tests for predictor wrappers and small utility helpers."""

from src.predictors import qualifying as qualifying_wrapper
from src.predictors import race as race_wrapper
from src.utils.driver_numbers import (
    get_all_drivers_2026,
    get_driver_from_abbreviation,
    get_driver_number,
    get_team_drivers_2026,
)
from src.utils.performance_tracker import PerformanceTracker


def test_qualifying_predictor_wrapper_delegates(patcher):
    class _FakeBaseline:
        def __init__(self, data_dir: str = "data/processed"):
            self.data_dir = data_dir

        def predict_qualifying(self, year: int, race_name: str):
            return {"grid": [{"driver": "NOR"}], "year": year, "race": race_name}

    patcher.setattr(qualifying_wrapper, "Baseline2026Predictor", _FakeBaseline)

    predictor = qualifying_wrapper.QualifyingPredictor(data_dir="tmp")
    out = predictor.predict(
        year=2026,
        race_name="Bahrain Grand Prix",
        method="blend",
        blend_weight=0.7,
        verbose=True,
    )

    assert out["year"] == 2026
    assert out["race"] == "Bahrain Grand Prix"
    assert predictor.driver_ranker is None


def test_race_predictor_wrapper_delegates(patcher):
    class _FakeBaseline:
        def __init__(self, data_dir: str = "data/processed"):
            self.data_dir = data_dir

        def predict_race(self, qualifying_grid, weather: str, race_name: str, n_simulations: int):
            return {
                "finish_order": qualifying_grid,
                "weather": weather,
                "race": race_name,
                "n_simulations": n_simulations,
            }

    patcher.setattr(race_wrapper, "Baseline2026Predictor", _FakeBaseline)

    predictor = race_wrapper.RacePredictor(data_dir="tmp")
    grid = [{"position": 1, "driver": "NOR", "team": "McLaren"}]
    out = predictor.predict(
        year=2026,
        race_name="Bahrain Grand Prix",
        qualifying_grid=grid,
        fp2_pace={"McLaren": 0.1},
        weather_forecast="rain",
        verbose=True,
        n_simulations=80,
    )

    assert out["finish_order"] == grid
    assert out["weather"] == "rain"
    assert out["race"] == "Bahrain Grand Prix"
    assert out["n_simulations"] == 80


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
