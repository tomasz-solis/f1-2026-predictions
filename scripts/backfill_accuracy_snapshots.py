#!/usr/bin/env python3
"""Backfill missing accuracy-snapshot artifacts from stored prediction truth."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime_dependencies() -> tuple[Any, ...]:
    """Import project modules after making the repository root importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.persistence.artifact_store import ArtifactStore
    from src.utils.accuracy_snapshots import build_accuracy_snapshot_records
    from src.utils.accuracy_targets import explicit_target_predictions
    from src.utils.prediction_logger import PredictionLogger
    from src.utils.prediction_metrics import PredictionMetrics
    from src.utils.weekend import is_sprint_weekend

    return (
        ArtifactStore,
        build_accuracy_snapshot_records,
        explicit_target_predictions,
        PredictionLogger,
        PredictionMetrics,
        is_sprint_weekend,
    )


(
    ArtifactStore,
    build_accuracy_snapshot_records,
    explicit_target_predictions,
    PredictionLogger,
    PredictionMetrics,
    is_sprint_weekend,
) = _load_runtime_dependencies()

for logger_name in ("requests_cache", "urllib3", "fastf1", "fastf1.req", "req"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def prediction_is_sprint_weekend(prediction: dict[str, Any]) -> bool:
    """Infer weekend format from saved metadata with a schedule fallback."""
    metadata = prediction.get("metadata", {})
    weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    if weekend_format == "sprint":
        return True
    if weekend_format == "normal":
        return False

    target_predictions = explicit_target_predictions(prediction)
    if any("sprint" in target_key for target_key in target_predictions):
        return True

    checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
    if checkpoint_session in {"SQ", "SPRINT"}:
        return True
    if checkpoint_session in {"FP2", "FP3"}:
        return False

    year = int(metadata.get("year", 0) or 0)
    race_name = str(metadata.get("race_name", "")).strip()
    if not year or not race_name:
        return False
    try:
        return bool(is_sprint_weekend(year, race_name))
    except Exception:
        return False


def backfill_accuracy_snapshots(
    *,
    year: int,
    race_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Write missing accuracy snapshots for one season or one race."""
    prediction_logger = PredictionLogger()
    artifact_store = ArtifactStore()
    metrics = PredictionMetrics()

    counters = {
        "predictions_seen": 0,
        "predictions_scored": 0,
        "predictions_without_metrics": 0,
        "snapshots_existing": 0,
        "snapshots_written": 0,
    }
    target_race_name = str(race_name).strip() if race_name else None

    for prediction in prediction_logger.get_all_predictions(year):
        metadata = prediction.get("metadata", {})
        prediction_race_name = str(metadata.get("race_name", "")).strip()
        checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
        if target_race_name and prediction_race_name != target_race_name:
            continue
        if not prediction_race_name or not checkpoint_session:
            continue

        counters["predictions_seen"] += 1
        snapshot_records = build_accuracy_snapshot_records(
            prediction_data=prediction,
            is_sprint=prediction_is_sprint_weekend(prediction),
            metrics_calculator=metrics,
            generated_by="scripts/backfill_accuracy_snapshots.py",
        )
        if not snapshot_records:
            counters["predictions_without_metrics"] += 1
            print(
                f"SKIP {prediction_race_name} {checkpoint_session}: "
                "no stored actuals for any tracked target"
            )
            continue

        counters["predictions_scored"] += 1
        for record in snapshot_records:
            artifact_key = str(record["artifact_key"])
            if artifact_store.load_artifact("accuracy_snapshot", artifact_key) is not None:
                counters["snapshots_existing"] += 1
                print(f"KEEP {artifact_key}")
                continue

            if dry_run:
                counters["snapshots_written"] += 1
                print(f"PLAN {artifact_key}")
                continue

            artifact_store.save_artifact(
                artifact_type="accuracy_snapshot",
                artifact_key=artifact_key,
                data=record["data"],
                version=1,
                run_id=metadata.get("run_id"),
            )
            counters["snapshots_written"] += 1
            print(f"WRITE {artifact_key}")

    return counters


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill missing accuracy snapshots from stored predictions."
    )
    parser.add_argument("--year", type=int, default=2026, help="Season year to backfill.")
    parser.add_argument(
        "--race-name",
        help="Optional race name filter, for example 'Bahrain Grand Prix'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned writes without saving artifacts.",
    )
    return parser


def main() -> int:
    """Run the backfill CLI."""
    args = build_parser().parse_args()
    counters = backfill_accuracy_snapshots(
        year=args.year,
        race_name=args.race_name,
        dry_run=bool(args.dry_run),
    )

    action_label = "planned" if args.dry_run else "written"
    print("")
    print(f"Predictions scanned: {counters['predictions_seen']}")
    print(f"Predictions with scoreable targets: {counters['predictions_scored']}")
    print(f"Predictions without stored actuals: {counters['predictions_without_metrics']}")
    print(f"Existing snapshots kept: {counters['snapshots_existing']}")
    print(f"Snapshots {action_label}: {counters['snapshots_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
