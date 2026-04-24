"""Tests for qualifying and race residual model artifacts."""

from __future__ import annotations

import pandas as pd

from src.models.qualifying_residual_model import (
    CAT_FEATURE_COLUMNS as QUAL_CAT_COLUMNS,
)
from src.models.qualifying_residual_model import (
    NUMERIC_FEATURE_COLUMNS as QUAL_NUMERIC_COLUMNS,
)
from src.models.qualifying_residual_model import (
    apply_qualifying_residual_model,
    fit_qualifying_residual_model,
)
from src.models.race_residual_model import (
    CAT_FEATURE_COLUMNS as RACE_CAT_COLUMNS,
)
from src.models.race_residual_model import (
    NUMERIC_FEATURE_COLUMNS as RACE_NUMERIC_COLUMNS,
)
from src.models.race_residual_model import (
    apply_race_residual_model,
    fit_race_residual_model,
)
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.race.prediction_mixin import BaselineRacePredictionMixin


class _DictConfig:
    """Tiny config stand-in with dotted-key access."""

    def __init__(self, values: dict[str, object]):
        self.values = values

    def get(self, key: str, default=None):
        """Return a configured value or a default."""
        return self.values.get(key, default)


class _QualifyingResidualLoader(BaselineQualifyingMixin):
    """Minimal object for testing qualifying residual loader guardrails."""

    def __init__(self, *, allow_with_testing_seed: bool):
        self.config = _DictConfig(
            {
                "baseline_predictor.qualifying.qualifying_residual_model.enabled": True,
                "baseline_predictor.qualifying.qualifying_residual_model.allow_with_testing_seed": (
                    allow_with_testing_seed
                ),
            }
        )
        self._qualifying_residual_model_cache = object()

    def _uses_testing_model_team_seed(self) -> bool:
        """Pretend the active team payload came from the testing seed model."""
        return True


class _RaceResidualLoader(BaselineRacePredictionMixin):
    """Minimal object for testing race residual loader guardrails."""

    def __init__(self, *, allow_with_testing_seed: bool):
        self.config = _DictConfig(
            {
                "baseline_predictor.race.race_residual_model.enabled": True,
                "baseline_predictor.race.race_residual_model.allow_with_testing_seed": (
                    allow_with_testing_seed
                ),
            }
        )
        self._race_residual_model_cache = object()

    def _uses_testing_model_team_seed(self) -> bool:
        """Pretend the active team payload came from the testing seed model."""
        return True


def _qualifying_dataset() -> pd.DataFrame:
    """Return a tiny but valid qualifying residual training dataset."""
    rows: list[dict[str, object]] = []
    for season_year, baseline_position, residual in (
        (2022, 1.0, -0.5),
        (2022, 2.0, 0.4),
        (2023, 1.0, -0.3),
        (2023, 2.0, 0.2),
        (2024, 1.0, -0.2),
        (2024, 2.0, 0.1),
    ):
        row: dict[str, object] = {
            "season_year": season_year,
            "target_residual_positions": residual,
        }
        for column in QUAL_NUMERIC_COLUMNS:
            row[column] = baseline_position if column == "baseline_position" else 0.5
        for column in QUAL_CAT_COLUMNS:
            row[column] = "normal" if column == "weekend_format" else "dry"
        row["experience_tier"] = "established"
        row["data_source_mode"] = "practice_backed"
        rows.append(row)
    return pd.DataFrame(rows)


def _race_dataset() -> pd.DataFrame:
    """Return a tiny but valid race residual training dataset."""
    rows: list[dict[str, object]] = []
    for season_year, grid_position, gain in (
        (2022, 1.0, 0.0),
        (2022, 5.0, 1.0),
        (2023, 2.0, 0.5),
        (2023, 6.0, 1.5),
        (2024, 3.0, 0.25),
        (2024, 7.0, 1.25),
    ):
        row: dict[str, object] = {
            "season_year": season_year,
            "target_positions_gained": gain,
        }
        for column in RACE_NUMERIC_COLUMNS:
            row[column] = grid_position if column == "grid_position" else 0.5
        for column in RACE_CAT_COLUMNS:
            row[column] = "normal" if column == "weekend_format" else "dry"
        row["grid_source_mode"] = "predicted"
        row["data_regime"] = "practice_backed"
        rows.append(row)
    return pd.DataFrame(rows)


def test_fit_qualifying_residual_model_predicts_clipped_adjustments():
    """Qualifying model inference should stay bounded by the configured clip."""
    dataset = _qualifying_dataset()
    model = fit_qualifying_residual_model(dataset, clip_positions=0.6)

    feature_frame = dataset.loc[:, list(QUAL_NUMERIC_COLUMNS) + list(QUAL_CAT_COLUMNS)].copy()
    driver_rows = [{"driver": "NOR"}, {"driver": "VER"}, {"driver": "PIA"}]
    repeated_frame = pd.concat(
        [feature_frame.iloc[[0]], feature_frame.iloc[[1]], feature_frame.iloc[[0]]]
    )
    adjustments = apply_qualifying_residual_model(
        model=model,
        feature_frame=repeated_frame.reset_index(drop=True),
        all_drivers=driver_rows,
    )

    assert set(adjustments) == {"NOR", "VER", "PIA"}
    assert all(abs(value) <= 0.6 for value in adjustments.values())
    assert all("qualifying_residual_adjustment" in row for row in driver_rows)


def test_qualifying_residual_loader_skips_testing_seed_by_default():
    """Testing-model team seeds should not stack qualifying residuals by default."""
    loader = _QualifyingResidualLoader(allow_with_testing_seed=False)

    assert loader._load_qualifying_residual_model() is None


def test_qualifying_residual_loader_allows_testing_seed_when_opted_in():
    """The explicit opt-in should preserve residual experiments for ablations."""
    loader = _QualifyingResidualLoader(allow_with_testing_seed=True)

    assert loader._load_qualifying_residual_model() is loader._qualifying_residual_model_cache


def test_fit_race_residual_model_applies_bounded_advantage_updates():
    """Race residual adjustments should clip and update race advantage in place."""
    dataset = _race_dataset()
    model = fit_race_residual_model(dataset, clip_positions_gained=1.0)

    feature_frame = dataset.loc[:, list(RACE_NUMERIC_COLUMNS) + list(RACE_CAT_COLUMNS)].copy()
    feature_frame = feature_frame.iloc[[0, 1]].copy().reset_index(drop=True)
    feature_frame["driver"] = ["NOR", "VER"]
    driver_info_map = {
        "NOR": {"race_advantage": 0.0},
        "VER": {"race_advantage": 0.1},
    }
    adjustments = apply_race_residual_model(
        model=model,
        feature_frame=feature_frame,
        driver_info_map=driver_info_map,
        positions_to_race_advantage_scale=0.05,
    )

    assert set(adjustments) == {"NOR", "VER"}
    assert all(abs(value) <= 1.0 for value in adjustments.values())
    assert driver_info_map["NOR"]["race_advantage"] != 0.0
    assert "race_residual_adjustment" in driver_info_map["VER"]


def test_race_residual_loader_skips_testing_seed_by_default():
    """Testing-model team seeds should not stack race residuals by default."""
    loader = _RaceResidualLoader(allow_with_testing_seed=False)

    assert loader._load_race_residual_model() is None


def test_race_residual_loader_allows_testing_seed_when_opted_in():
    """The explicit opt-in should preserve race residual experiments for ablations."""
    loader = _RaceResidualLoader(allow_with_testing_seed=True)

    assert loader._load_race_residual_model() is loader._race_residual_model_cache
