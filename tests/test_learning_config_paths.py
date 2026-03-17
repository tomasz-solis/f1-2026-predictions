from __future__ import annotations

from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.race.prediction_mixin import BaselineRacePredictionMixin


def test_qualifying_learning_uses_top_level_learning_config():
    captured: dict[str, float | int | str | list[str]] = {}

    class _CalibrationStub:
        def get_combined_position_adjustment(
            self,
            *,
            team,
            driver,
            teammates,
            session,
            min_samples,
            driver_error_scale,
            teammate_gap_scale,
            max_adjustment,
        ):
            captured.update(
                {
                    "team": team,
                    "driver": driver,
                    "teammates": teammates,
                    "session": session,
                    "min_samples": min_samples,
                    "driver_error_scale": driver_error_scale,
                    "teammate_gap_scale": teammate_gap_scale,
                    "max_adjustment": max_adjustment,
                }
            )
            return 0.0

    class _Config:
        def get(self, key, default=None):
            overrides = {
                "learning.min_samples": 3,
                "learning.driver_error_scale": 0.21,
                "learning.teammate_gap_scale": 0.07,
                "learning.max_adjustment": 1.9,
            }
            return overrides.get(key, default)

    class _Predictor(BaselineQualifyingMixin):
        def __init__(self):
            self.calibration_system = _CalibrationStub()
            self.config = _Config()

    predictor = _Predictor()
    assert (
        predictor._get_learned_position_adjustment(
            team="Red Bull Racing",
            driver="VER",
            teammates=["VER", "HAD"],
            session="qualifying",
        )
        == 0.0
    )
    assert captured == {
        "team": "Red Bull Racing",
        "driver": "VER",
        "teammates": ["VER", "HAD"],
        "session": "qualifying",
        "min_samples": 3,
        "driver_error_scale": 0.21,
        "teammate_gap_scale": 0.07,
        "max_adjustment": 1.9,
    }


def test_race_learning_uses_top_level_learning_config():
    captured: dict[str, float | int | str | list[str]] = {}

    class _CalibrationStub:
        def get_combined_position_adjustment(
            self,
            *,
            team,
            driver,
            teammates,
            session,
            min_samples,
            driver_error_scale,
            teammate_gap_scale,
            max_adjustment,
        ):
            captured.update(
                {
                    "team": team,
                    "driver": driver,
                    "teammates": teammates,
                    "session": session,
                    "min_samples": min_samples,
                    "driver_error_scale": driver_error_scale,
                    "teammate_gap_scale": teammate_gap_scale,
                    "max_adjustment": max_adjustment,
                }
            )
            return 0.0

    class _Config:
        def get(self, key, default=None):
            overrides = {
                "learning.min_samples": 4,
                "learning.driver_error_scale": 0.11,
                "learning.teammate_gap_scale": 0.05,
                "learning.max_adjustment": 1.7,
            }
            return overrides.get(key, default)

    class _Predictor(BaselineRacePredictionMixin):
        def __init__(self):
            self.calibration_system = _CalibrationStub()
            self.config = _Config()

    predictor = _Predictor()
    assert (
        predictor._get_learned_position_adjustment(
            team="Ferrari",
            driver="LEC",
            teammates=["LEC", "HAM"],
            session="race",
        )
        == 0.0
    )
    assert captured == {
        "team": "Ferrari",
        "driver": "LEC",
        "teammates": ["LEC", "HAM"],
        "session": "race",
        "min_samples": 4,
        "driver_error_scale": 0.11,
        "teammate_gap_scale": 0.05,
        "max_adjustment": 1.7,
    }
