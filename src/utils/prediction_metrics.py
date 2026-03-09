"""Calculates accuracy metrics for saved prediction payloads."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.utils.accuracy_targets import (
    explicit_target_actuals,
    explicit_target_predictions,
    legacy_target_keys_for_prediction,
    sanitize_actual_rows,
    sanitize_prediction_rows,
    synthesize_legacy_actuals,
    synthesize_legacy_targets,
)
from src.utils.driver_name_mapper import DriverNameMapper

logger = logging.getLogger(__name__)


class PredictionMetrics:
    """Calculate accuracy metrics for qualifying, race, and target-specific views."""

    @staticmethod
    def _normalize_result_lists(
        predicted: list[dict[str, Any]],
        actual: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Normalize two result lists for driver-level comparison."""
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)
        return predicted_norm, actual_norm

    @staticmethod
    def position_accuracy(predicted: list[dict[str, Any]], actual: list[dict[str, Any]]) -> float:
        """Calculate exact position accuracy as a percentage."""
        if not predicted or not actual:
            return 0.0

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        actual_positions = {row["driver"]: row["position"] for row in actual_norm}

        missing_from_actual = [
            row["driver"] for row in predicted_norm if row["driver"] not in actual_positions
        ]
        if missing_from_actual:
            logger.warning(
                "Drivers in prediction but not in actuals (possible substitution): %s",
                missing_from_actual,
            )

        correct = sum(
            1
            for row in predicted_norm
            if row["driver"] in actual_positions
            and actual_positions[row["driver"]] == row["position"]
        )
        return (correct / len(predicted_norm)) * 100

    @staticmethod
    def mean_absolute_error(predicted: list[dict[str, Any]], actual: list[dict[str, Any]]) -> float:
        """Calculate mean absolute position error."""
        if not predicted or not actual:
            return float("inf")

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        actual_positions = {row["driver"]: row["position"] for row in actual_norm}
        errors = [
            abs(row["position"] - actual_positions[row["driver"]])
            for row in predicted_norm
            if row["driver"] in actual_positions
        ]
        if not errors:
            return float("inf")
        return float(np.mean(errors))

    @staticmethod
    def within_n_positions(
        predicted: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        n: int = 1,
    ) -> float:
        """Calculate the share of predictions within ``n`` positions."""
        if not predicted or not actual:
            return 0.0

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        actual_positions = {row["driver"]: row["position"] for row in actual_norm}
        within_n = sum(
            1
            for row in predicted_norm
            if row["driver"] in actual_positions
            and abs(row["position"] - actual_positions[row["driver"]]) <= n
        )
        return (within_n / len(predicted_norm)) * 100

    @staticmethod
    def correlation_coefficient(
        predicted: list[dict[str, Any]], actual: list[dict[str, Any]]
    ) -> float:
        """Calculate Spearman rank correlation between predicted and actual positions."""
        if not predicted or not actual:
            return 0.0

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        actual_positions = {row["driver"]: row["position"] for row in actual_norm}

        predicted_positions: list[int] = []
        actual_positions_list: list[int] = []
        for row in predicted_norm:
            if row["driver"] in actual_positions:
                predicted_positions.append(row["position"])
                actual_positions_list.append(actual_positions[row["driver"]])

        if len(predicted_positions) < 2:
            return 0.0

        correlation, _ = spearmanr(predicted_positions, actual_positions_list)
        return float(correlation) if not np.isnan(correlation) else 0.0

    @staticmethod
    def podium_accuracy(
        predicted: list[dict[str, Any]], actual: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate podium hit rate for a race prediction."""
        if not predicted or not actual:
            return {"correct_drivers": 0, "correct_positions": 0, "accuracy": 0.0}

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        predicted_podium = {row["driver"]: row["position"] for row in predicted_norm[:3]}
        actual_podium = {row["driver"]: row["position"] for row in actual_norm[:3]}

        correct_drivers = sum(1 for driver in predicted_podium if driver in actual_podium)
        correct_positions = sum(
            1
            for driver, position in predicted_podium.items()
            if driver in actual_podium and actual_podium[driver] == position
        )

        return {
            "correct_drivers": correct_drivers,
            "correct_positions": correct_positions,
            "accuracy": (correct_drivers / 3) * 100,
        }

    @staticmethod
    def winner_accuracy(predicted: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
        """Return True when the predicted and actual winners match."""
        if not predicted or not actual:
            return False

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        return predicted_norm[0]["driver"] == actual_norm[0]["driver"]

    @staticmethod
    def top_k_overlap(
        predicted: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        *,
        k: int,
    ) -> dict[str, float]:
        """Return overlap hits and percentage for the top ``k`` entries."""
        if not predicted or not actual or k <= 0:
            return {"hits": 0.0, "pct": 0.0}

        predicted_norm, actual_norm = PredictionMetrics._normalize_result_lists(predicted, actual)
        predicted_top = {row["driver"] for row in predicted_norm[:k]}
        actual_top = {row["driver"] for row in actual_norm[:k]}
        field_size = min(k, len(actual_norm))
        if field_size <= 0:
            return {"hits": 0.0, "pct": 0.0}
        hits = float(len(predicted_top & actual_top))
        return {"hits": hits, "pct": (hits / field_size) * 100}

    @staticmethod
    def calculate_target_metrics(
        predicted_order: list[dict[str, Any]],
        actual_order: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate the target-aware metric set for one predicted order."""
        top_3 = PredictionMetrics.top_k_overlap(predicted_order, actual_order, k=3)
        top_10 = PredictionMetrics.top_k_overlap(predicted_order, actual_order, k=10)
        return {
            "field_size": len(actual_order),
            "overall_mae": PredictionMetrics.mean_absolute_error(predicted_order, actual_order),
            "top_3_hits": top_3["hits"],
            "top_3_pct": top_3["pct"],
            "top_10_hits": top_10["hits"],
            "top_10_pct": top_10["pct"],
            "exact_accuracy": PredictionMetrics.position_accuracy(predicted_order, actual_order),
            "within_1": PredictionMetrics.within_n_positions(predicted_order, actual_order, 1),
            "within_3": PredictionMetrics.within_n_positions(predicted_order, actual_order, 3),
            "correlation": PredictionMetrics.correlation_coefficient(predicted_order, actual_order),
        }

    @staticmethod
    def calculate_prediction_target_metrics(
        prediction_data: dict[str, Any],
        *,
        is_sprint: bool,
    ) -> dict[str, dict[str, Any]]:
        """Calculate metrics for every explicit or synthesized target in a prediction."""
        target_predictions = explicit_target_predictions(prediction_data)
        if not target_predictions:
            target_predictions = synthesize_legacy_targets(prediction_data, is_sprint=is_sprint)

        target_actuals = explicit_target_actuals(prediction_data)
        if not target_actuals:
            target_actuals = synthesize_legacy_actuals(prediction_data, is_sprint=is_sprint)

        metrics: dict[str, dict[str, Any]] = {}
        for target_key, target_payload in target_predictions.items():
            predicted_order = sanitize_prediction_rows(target_payload.get("predicted_order"))
            actual_order = sanitize_actual_rows(target_actuals.get(target_key))
            if not predicted_order or not actual_order:
                continue
            metrics[target_key] = PredictionMetrics.calculate_target_metrics(
                predicted_order,
                actual_order,
            )
        return metrics

    @staticmethod
    def _aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Aggregate metric rows into mean and standard deviation by field."""
        if not metric_rows:
            return {}

        aggregate: dict[str, dict[str, float]] = {}
        keys = sorted({key for row in metric_rows for key in row})
        for key in keys:
            values = [
                float(row[key])
                for row in metric_rows
                if isinstance(row.get(key), int | float) and np.isfinite(float(row[key]))
            ]
            if not values:
                continue
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        return aggregate

    @staticmethod
    def aggregate_target_metrics(
        all_predictions: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate target-aware metrics across multiple saved predictions."""
        grouped_metrics: dict[str, list[dict[str, Any]]] = {}
        for prediction in all_predictions:
            metadata = prediction.get("metadata", {})
            weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
            is_sprint = weekend_format == "sprint"
            checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
            if weekend_format not in {"normal", "sprint"}:
                qualifying_target, race_target = legacy_target_keys_for_prediction(
                    checkpoint_session,
                    is_sprint=False,
                )
                is_sprint = (
                    qualifying_target is not None
                    and race_target is not None
                    and checkpoint_session
                    in {
                        "SQ",
                        "SPRINT",
                    }
                )
            target_metrics = PredictionMetrics.calculate_prediction_target_metrics(
                prediction,
                is_sprint=is_sprint,
            )
            for target_key, metrics in target_metrics.items():
                grouped_metrics.setdefault(target_key, []).append(metrics)

        return {
            target_key: PredictionMetrics._aggregate_metric_rows(metric_rows)
            for target_key, metric_rows in grouped_metrics.items()
            if metric_rows
        }

    @staticmethod
    def calculate_all_metrics(prediction_data: dict[str, Any]) -> dict[str, Any] | None:
        """Calculate legacy qualifying and race metrics for compatibility."""
        actuals = prediction_data.get("actuals")
        if not isinstance(actuals, dict):
            return None

        qualifying_actual = sanitize_actual_rows(actuals.get("qualifying"))
        race_actual = sanitize_actual_rows(actuals.get("race"))
        if not qualifying_actual and not race_actual:
            return None

        metrics: dict[str, Any] = {"metadata": prediction_data.get("metadata", {})}

        qualifying_prediction = sanitize_prediction_rows(
            (prediction_data.get("qualifying") or {}).get("predicted_grid")
        )
        if qualifying_prediction and qualifying_actual:
            metrics["qualifying"] = {
                "exact_accuracy": PredictionMetrics.position_accuracy(
                    qualifying_prediction,
                    qualifying_actual,
                ),
                "mae": PredictionMetrics.mean_absolute_error(
                    qualifying_prediction,
                    qualifying_actual,
                ),
                "within_1": PredictionMetrics.within_n_positions(
                    qualifying_prediction,
                    qualifying_actual,
                    1,
                ),
                "within_3": PredictionMetrics.within_n_positions(
                    qualifying_prediction,
                    qualifying_actual,
                    3,
                ),
                "correlation": PredictionMetrics.correlation_coefficient(
                    qualifying_prediction,
                    qualifying_actual,
                ),
            }

        race_prediction = sanitize_prediction_rows(
            (prediction_data.get("race") or {}).get("predicted_results")
        )
        if race_prediction and race_actual:
            metrics["race"] = {
                "exact_accuracy": PredictionMetrics.position_accuracy(race_prediction, race_actual),
                "mae": PredictionMetrics.mean_absolute_error(race_prediction, race_actual),
                "within_1": PredictionMetrics.within_n_positions(race_prediction, race_actual, 1),
                "within_3": PredictionMetrics.within_n_positions(race_prediction, race_actual, 3),
                "correlation": PredictionMetrics.correlation_coefficient(
                    race_prediction,
                    race_actual,
                ),
                "podium": PredictionMetrics.podium_accuracy(race_prediction, race_actual),
                "winner_correct": PredictionMetrics.winner_accuracy(race_prediction, race_actual),
            }

        return metrics if len(metrics) > 1 else None

    @staticmethod
    def aggregate_metrics(all_predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate legacy qualifying and race metrics for compatibility."""
        all_metrics = []
        for prediction in all_predictions:
            metrics = PredictionMetrics.calculate_all_metrics(prediction)
            if metrics is not None:
                all_metrics.append(metrics)

        if not all_metrics:
            return {"error": "No predictions with actuals found"}

        aggregated: dict[str, Any] = {}
        qualifying_metrics = [
            metrics["qualifying"] for metrics in all_metrics if "qualifying" in metrics
        ]
        race_metrics = [metrics["race"] for metrics in all_metrics if "race" in metrics]

        if qualifying_metrics:
            aggregated["qualifying"] = PredictionMetrics._aggregate_metric_rows(qualifying_metrics)

        if race_metrics:
            race_aggregate = PredictionMetrics._aggregate_metric_rows(race_metrics)
            if "winner_correct" in race_aggregate:
                winner_aggregate = race_aggregate.pop("winner_correct")
                race_aggregate["winner_accuracy"] = {
                    "mean": winner_aggregate["mean"],
                    "std": winner_aggregate["std"],
                    "percentage": winner_aggregate["mean"] * 100,
                }
            aggregated["race"] = race_aggregate

        return aggregated
