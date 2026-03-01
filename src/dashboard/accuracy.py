"""Prediction-accuracy data pipeline for the dashboard.

Centralises metric computation, season-level aggregation, and
trend analysis so that the accuracy page and any future API
consumers can share the same logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------


@dataclass
class RaceAccuracySnapshot:
    """Accuracy metrics for a single race prediction."""

    race_name: str
    session_name: str
    qualifying: dict[str, float] | None = None
    race: dict[str, Any] | None = None


@dataclass
class SeasonAccuracySummary:
    """Aggregated accuracy metrics across a full season."""

    year: int
    n_predictions: int = 0
    qualifying_aggregate: dict[str, Any] = field(default_factory=dict)
    race_aggregate: dict[str, Any] = field(default_factory=dict)
    per_race: list[RaceAccuracySnapshot] = field(default_factory=list)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


class AccuracyPipeline:
    """Load, compute, and structure accuracy data for a given season.

    Usage::

        pipeline = AccuracyPipeline(year=2026)
        summary = pipeline.build_summary()
    """

    def __init__(self, year: int = 2026):
        self.year = year
        self._predictions: list[dict[str, Any]] = []
        self._metrics_calculator: Any = None

    # -- lazy imports to avoid heavyweight deps at module level ----------

    def _ensure_deps(self) -> None:
        """Lazily import PredictionLogger and PredictionMetrics."""
        if self._metrics_calculator is not None:
            return

        from src.utils.prediction_logger import PredictionLogger
        from src.utils.prediction_metrics import PredictionMetrics

        logger_inst = PredictionLogger()
        self._predictions = logger_inst.get_all_predictions(self.year)
        self._metrics_calculator = PredictionMetrics()

    # -- public API ------------------------------------------------------

    def build_summary(self) -> SeasonAccuracySummary:
        """Return a :class:`SeasonAccuracySummary` for the configured year."""
        self._ensure_deps()

        summary = SeasonAccuracySummary(year=self.year)

        if not self._predictions:
            return summary

        predictions_with_actuals = [
            p
            for p in self._predictions
            if p.get("actuals") and (p["actuals"].get("qualifying") or p["actuals"].get("race"))
        ]

        summary.n_predictions = len(self._predictions)

        for pred in predictions_with_actuals:
            metrics = self._metrics_calculator.calculate_all_metrics(pred)
            if metrics is None:
                continue
            snapshot = RaceAccuracySnapshot(
                race_name=metrics["metadata"]["race_name"],
                session_name=metrics["metadata"]["session_name"],
                qualifying=metrics.get("qualifying"),
                race=metrics.get("race"),
            )
            summary.per_race.append(snapshot)

        if predictions_with_actuals:
            agg = self._metrics_calculator.aggregate_metrics(predictions_with_actuals)
            summary.qualifying_aggregate = agg.get("qualifying", {})
            summary.race_aggregate = agg.get("race", {})

        return summary

    @property
    def all_predictions(self) -> list[dict[str, Any]]:
        """Raw prediction dicts (useful for the saved-predictions list)."""
        self._ensure_deps()
        return self._predictions

    @property
    def has_actuals(self) -> bool:
        """True when at least one prediction has actual results attached."""
        self._ensure_deps()
        return any(
            p.get("actuals") and (p["actuals"].get("qualifying") or p["actuals"].get("race"))
            for p in self._predictions
        )
