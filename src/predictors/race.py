"""
Legacy compatibility wrapper for race predictions.

Older scripts import `RacePredictor` from this module; the active
implementation now delegates to `Baseline2026Predictor`.
"""

from __future__ import annotations

import inspect
from typing import Any

from src.predictors.baseline_2026 import Baseline2026Predictor
from src.types.prediction_types import QualifyingGridEntry


class RacePredictor:
    """Compat layer that preserves legacy constructor and predict signature."""

    def __init__(
        self,
        driver_chars: Any = None,
        driver_chars_path: str | None = None,
        performance_tracker: Any = None,
        data_dir: str = "data/processed",
    ):
        self.driver_chars = driver_chars
        self.driver_chars_path = driver_chars_path
        self.performance_tracker = performance_tracker
        self._predictor = Baseline2026Predictor(data_dir=data_dir)

    def predict(
        self,
        year: int,
        race_name: str,
        qualifying_grid: list[QualifyingGridEntry],
        fp2_pace: Any = None,
        weather_forecast: str = "dry",
        verbose: bool = False,
        n_simulations: int = 50,
    ) -> dict[str, Any]:
        # `fp2_pace` and `verbose` are legacy wrapper parameters.
        _ = (fp2_pace, verbose)
        predict_race = self._predictor.predict_race
        try:
            signature = inspect.signature(predict_race)
            parameters = signature.parameters
        except (TypeError, ValueError):
            parameters = {}

        supports_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {
            "qualifying_grid": qualifying_grid,
            "weather": weather_forecast,
            "race_name": race_name,
            "n_simulations": n_simulations,
        }
        if supports_var_kwargs or "year" in parameters:
            kwargs["year"] = year

        return predict_race(**kwargs)
