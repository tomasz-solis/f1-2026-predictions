"""Compatibility wrapper for older qualifying prediction imports."""

from __future__ import annotations

from typing import Any

from src.predictors.baseline_2026 import Baseline2026Predictor


class QualifyingPredictor:
    """Keep the old qualifying predictor interface working."""

    def __init__(
        self,
        driver_ranker: Any = None,
        performance_tracker: Any = None,
        data_dir: str = "data/processed",
    ):
        self.driver_ranker = driver_ranker
        self.performance_tracker = performance_tracker
        self._predictor = Baseline2026Predictor(data_dir=data_dir)

    def predict(
        self,
        year: int,
        race_name: str,
        method: str = "blend",
        blend_weight: float = 0.7,
        verbose: bool = False,
    ) -> dict[str, Any]:
        # `method`, `blend_weight`, and `verbose` are kept for compatibility.
        _ = (method, blend_weight, verbose)
        return self._predictor.predict_qualifying(year=year, race_name=race_name)
