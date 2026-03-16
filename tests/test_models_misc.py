"""Tests for model utility modules (car/regulations/scoring)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.bayesian import DriverPrior
from src.models.car import Car
from src.models.priors_factory import PriorsFactory
from src.models.regulations import apply_2026_regulations
from src.models.scoring import (
    AbsoluteDifferenceScoring,
    PerformanceScoringMethod,
    QuantileScoring,
    RankingScoring,
    ZScoreScoring,
)


def _sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_number": "1",
                "slow_corner_speed": 120.0,
                "medium_corner_speed": 160.0,
                "high_corner_speed": 210.0,
                "avg_speed_full_throttle": 305.0,
                "pct_full_throttle": 52.0,
            },
            {
                "driver_number": "2",
                "slow_corner_speed": 118.0,
                "medium_corner_speed": 158.0,
                "high_corner_speed": 208.0,
                "avg_speed_full_throttle": 302.0,
                "pct_full_throttle": 50.0,
            },
            {
                "driver_number": "3",
                "slow_corner_speed": 116.0,
                "medium_corner_speed": 155.0,
                "high_corner_speed": 205.0,
                "avg_speed_full_throttle": 299.0,
                "pct_full_throttle": 48.0,
            },
        ]
    )


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
    assert 5.0 <= car.get_performance_score("balanced") <= 18.0
    assert 5.0 <= car.get_performance_score("monaco") <= 18.0
    assert 5.0 <= car.get_performance_score("monza") <= 18.0
    assert 5.0 <= car.get_performance_score("silverstone") <= 18.0
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
        "77": DriverPrior("77", "BOT", "Kick Sauber", "midfield", mu=10.0, sigma=2.5),
        "18": DriverPrior("18", "STR", "Aston Martin", "Customer", mu=11.0, sigma=2.2),
    }

    adjusted = apply_2026_regulations(priors)

    assert adjusted["44"].mu == 15.5
    assert adjusted["63"].mu == 15.0
    assert adjusted["77"].mu == 8.0
    assert adjusted["18"].mu == 10.5
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


def test_performance_scoring_base_class_raises():
    with pytest.raises(NotImplementedError):
        PerformanceScoringMethod().score_drivers(_sample_features())


def test_absolute_difference_scoring_outputs_centered_values():
    scores = AbsoluteDifferenceScoring().score_drivers(_sample_features())
    first = scores[scores["driver_number"] == "1"].iloc[0]
    third = scores[scores["driver_number"] == "3"].iloc[0]

    assert first["slow_corner_score"] > 0
    assert third["slow_corner_score"] < 0


def test_ranking_scoring_assigns_rank_one_to_best():
    scores = RankingScoring().score_drivers(_sample_features())
    first = scores[scores["driver_number"] == "1"].iloc[0]
    third = scores[scores["driver_number"] == "3"].iloc[0]

    assert first["slow_corner_score"] == 1
    assert third["slow_corner_score"] == 3


def test_quantile_scoring_assigns_tiers():
    scores = QuantileScoring().score_drivers(_sample_features())
    first = scores[scores["driver_number"] == "1"].iloc[0]
    middle = scores[scores["driver_number"] == "2"].iloc[0]
    third = scores[scores["driver_number"] == "3"].iloc[0]

    assert first["slow_corner_score"] == 3
    assert middle["slow_corner_score"] == 2
    assert third["slow_corner_score"] == 1


def test_z_score_scoring_handles_zero_std():
    features = _sample_features()
    features["pct_full_throttle"] = 50.0

    scores = ZScoreScoring().score_drivers(features)

    assert "slow_corner_score" in scores.columns
    assert (scores["throttle_usage_score"] == 0).all()
