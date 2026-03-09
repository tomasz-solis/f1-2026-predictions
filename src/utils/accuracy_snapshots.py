"""Helpers for building persisted accuracy-snapshot artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.utils.accuracy_targets import (
    explicit_target_predictions,
    legacy_target_keys_for_prediction,
    synthesize_legacy_targets,
    target_label,
    weekend_format_name,
)


def accuracy_snapshot_artifact_key(
    *,
    year: int,
    race_name: str,
    checkpoint_session: str,
    target_key: str,
) -> str:
    """Return the canonical artifact key for one checkpoint-target snapshot."""
    return f"{int(year)}::{race_name}::{str(checkpoint_session).strip().upper()}::{target_key}"


def build_accuracy_snapshot_records(
    *,
    prediction_data: dict[str, Any],
    is_sprint: bool,
    metrics_calculator: Any,
    generated_by: str,
) -> list[dict[str, Any]]:
    """Build persisted snapshot payloads for every scored target in a prediction."""
    metadata = prediction_data.get("metadata", {})
    target_predictions = explicit_target_predictions(prediction_data)
    if not target_predictions:
        target_predictions = synthesize_legacy_targets(prediction_data, is_sprint=is_sprint)

    target_metrics = calculate_target_metric_map(
        metrics_calculator=metrics_calculator,
        prediction_data=prediction_data,
        is_sprint=is_sprint,
    )
    if not target_predictions or not target_metrics:
        return []

    checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
    race_name = str(metadata.get("race_name", "")).strip()
    year = int(metadata.get("year", 0) or 0)
    weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    if weekend_format not in {"normal", "sprint"}:
        weekend_format = weekend_format_name(is_sprint)

    records: list[dict[str, Any]] = []
    for target_key, metrics in target_metrics.items():
        target_payload = target_predictions.get(target_key)
        if not isinstance(target_payload, dict):
            continue
        records.append(
            {
                "artifact_key": accuracy_snapshot_artifact_key(
                    year=year,
                    race_name=race_name,
                    checkpoint_session=checkpoint_session,
                    target_key=target_key,
                ),
                "data": {
                    "metadata": {
                        "year": year,
                        "race_name": race_name,
                        "checkpoint_session": checkpoint_session,
                        "weekend_format": weekend_format,
                        "target_key": target_key,
                        "target_label": target_label(target_key),
                        "target_session": target_payload.get("target_session"),
                        "predicted_at": metadata.get("predicted_at"),
                        "generated_at": datetime.now(UTC).isoformat(),
                        "source_run_id": metadata.get("run_id"),
                        "eligible": bool(target_payload.get("eligible_at_save", True)),
                        "generated_by": generated_by,
                    },
                    "metrics": metrics,
                },
            }
        )
    return records


def calculate_target_metric_map(
    *,
    metrics_calculator: Any,
    prediction_data: dict[str, Any],
    is_sprint: bool,
) -> dict[str, dict[str, Any]]:
    """Return target metrics with a fallback for older metric stubs."""
    calculate_target_metrics = getattr(
        metrics_calculator,
        "calculate_prediction_target_metrics",
        None,
    )
    if callable(calculate_target_metrics):
        return calculate_target_metrics(prediction_data, is_sprint=is_sprint)

    calculate_all_metrics = getattr(metrics_calculator, "calculate_all_metrics", None)
    if not callable(calculate_all_metrics):
        return {}

    legacy_metrics = calculate_all_metrics(prediction_data) or {}
    checkpoint_session = (
        str((prediction_data.get("metadata") or {}).get("session_name", "")).strip().upper()
    )
    qualifying_target, race_target = legacy_target_keys_for_prediction(
        checkpoint_session,
        is_sprint=is_sprint,
    )
    metrics: dict[str, dict[str, Any]] = {}
    if qualifying_target and isinstance(legacy_metrics.get("qualifying"), dict):
        metrics[qualifying_target] = legacy_metrics["qualifying"]
    if race_target and isinstance(legacy_metrics.get("race"), dict):
        metrics[race_target] = legacy_metrics["race"]
    return metrics
