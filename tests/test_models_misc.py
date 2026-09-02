"""Tests for model utility modules (car/regulations/priors)."""

import json
from pathlib import Path

import pytest

from src.models.bayesian import DriverPrior
from src.models.car import Car
from src.models.priors_factory import PriorsFactory
from src.models.regulations import apply_2026_regulations


def test_car_update_and_scoring_by_track():
    car = Car("McLaren", 2026)
    car.update_from_testing(
        {
            "slow_corner_performance": 0.62,
            "medium_corner_performance": 0.67,
            "fast_corner_performance": 0.64,
            "top_speed": 340.0,
            "consistency": 0.9,
            "tire_deg_slope": 0.55,
        }
    )

    assert round(car.characteristics.straight_line, 3) == 0.8
    assert car._calculate_base_score(0.0) == 5.0
    assert car._calculate_base_score(1.0) == 18.0


def test_car_update_no_data_is_noop():
    car = Car("Ferrari", 2026)
    baseline = car.characteristics.slow_corner
    car.update_from_testing({})
    assert car.characteristics.slow_corner == baseline


def test_apply_2026_regulations_adjusts_mu_and_sigma():
    priors = {
        "44": DriverPrior("44", "HAM", "Ferrari", "top", mu=14.0, sigma=2.0),
        "63": DriverPrior("63", "RUS", "Mercedes", "top", mu=13.5, sigma=2.1),
        "27": DriverPrior("27", "HUL", "Audi", "midfield", mu=10.0, sigma=2.5),
        "18": DriverPrior("18", "STR", "Aston Martin", "midfield", mu=11.0, sigma=2.2),
        "12": DriverPrior("12", "CAD", "Cadillac F1", "backmarker", mu=9.0, sigma=2.4),
    }

    adjusted = apply_2026_regulations(priors)

    assert adjusted["44"].mu == pytest.approx(15.0)
    assert adjusted["63"].mu == pytest.approx(14.5)
    assert adjusted["27"].mu == pytest.approx(11.0)
    assert adjusted["18"].mu == pytest.approx(10.7)
    assert adjusted["12"].mu == pytest.approx(7.5)
    assert adjusted["44"].sigma == 3.0
    assert priors["44"].mu == 14.0


def test_priors_factory_prefers_season_scoped_driver_characteristics(tmp_path):
    processed_dir = Path(tmp_path)
    driver_dir = processed_dir / "driver_characteristics"
    car_dir = processed_dir / "car_characteristics"
    driver_dir.mkdir(parents=True, exist_ok=True)
    car_dir.mkdir(parents=True, exist_ok=True)

    season_payload = {
        "drivers": {
            "NOR": {
                "racecraft": {"skill_score": 0.91},
                "pace": {"quali_pace": 0.80},
            }
        }
    }
    legacy_payload = {
        "drivers": {
            "NOR": {
                "racecraft": {"skill_score": 0.20},
                "pace": {"quali_pace": 0.20},
            }
        }
    }
    (driver_dir / "2026_driver_characteristics.json").write_text(json.dumps(season_payload))
    (processed_dir / "driver_characteristics.json").write_text(json.dumps(legacy_payload))
    (car_dir / "2026_car_characteristics.json").write_text(json.dumps({"teams": {}}))

    factory = PriorsFactory(data_dir=processed_dir, season_year=2026)
    factory.load_data()

    assert factory.drivers["NOR"]["racecraft"]["skill_score"] == 0.91
