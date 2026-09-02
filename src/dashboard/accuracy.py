"""Prediction-accuracy pipeline for target-aware dashboard views."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.accuracy_snapshots import accuracy_snapshot_artifact_key, calculate_target_metric_map
from src.utils.accuracy_targets import (
    CHECKPOINT_ORDER,
    PRIMARY_TARGET_KEYS,
    explicit_target_actuals,
    explicit_target_predictions,
    synthesize_legacy_actuals,
    synthesize_legacy_targets,
    target_checkpoint_index,
    target_label,
    weekend_format_name,
)
from src.utils.weekend import get_schedule_rows

logger = logging.getLogger(__name__)


@dataclass
class CheckpointAccuracyPoint:
    """Season-average metric values for one checkpoint in one target view."""

    target_key: str
    weekend_format: str
    checkpoint_session: str
    checkpoint_index: int
    metrics: dict[str, float]
    race_count: int


@dataclass
class CheckpointStatusPoint:
    """Aggregate save-status counts for one checkpoint in one target view."""

    target_key: str
    weekend_format: str
    checkpoint_session: str
    checkpoint_index: int
    scored_count: int = 0
    pending_count: int = 0
    excluded_count: int = 0


@dataclass
class SeasonTrendPoint:
    """One race-level point for a target checkpoint trend line."""

    target_key: str
    weekend_format: str
    race_name: str
    race_order: int
    checkpoint_session: str
    checkpoint_index: int
    metrics: dict[str, float]


@dataclass
class TargetAccuracySummary:
    """Aggregated accuracy summary for one canonical target."""

    target_key: str
    label: str
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)
    checkpoint_progression: list[CheckpointAccuracyPoint] = field(default_factory=list)
    checkpoint_status: list[CheckpointStatusPoint] = field(default_factory=list)
    season_trend: list[SeasonTrendPoint] = field(default_factory=list)
    n_scored_predictions: int = 0


@dataclass
class RaceAccuracySnapshot:
    """Compatibility container for per-race accuracy details."""

    race_name: str
    session_name: str
    qualifying: dict[str, float] | None = None
    race: dict[str, Any] | None = None
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SeasonAccuracySummary:
    """Aggregated accuracy metrics across a full season."""

    year: int
    n_predictions: int = 0
    qualifying_aggregate: dict[str, Any] = field(default_factory=dict)
    race_aggregate: dict[str, Any] = field(default_factory=dict)
    per_race: list[RaceAccuracySnapshot] = field(default_factory=list)
    targets: dict[str, TargetAccuracySummary] = field(default_factory=dict)
    n_excluded_predictions: int = 0
    n_excluded_targets: int = 0


@dataclass
class _TargetAccuracyRecord:
    """Internal normalized target record used to build dashboard summaries."""

    race_name: str
    checkpoint_session: str
    weekend_format: str
    target_key: str
    metrics: dict[str, float]
    predicted_at: str


@dataclass
class _TargetStatusRecord:
    """Internal per-target checkpoint status used for chart annotations."""

    race_name: str
    checkpoint_session: str
    weekend_format: str
    target_key: str
    status: str


class AccuracyPipeline:
    """Load, compute, and structure target-aware accuracy data for one season."""

    def __init__(self, year: int = 2026, *, reconcile_actuals_on_load: bool = False):
        """Create a pipeline for one season."""
        self.year = int(year)
        self.reconcile_actuals_on_load = bool(reconcile_actuals_on_load)
        self._predictions: list[dict[str, Any]] = []
        self._prediction_logger: Any = None
        self._metrics_calculator: Any = None
        self._artifact_store: Any = None
        self._actuals_reconciled = 0
        self._snapshots_written = 0
        self._initialized = False
        self._prediction_status_rows: list[dict[str, Any]] = []
        self._excluded_predictions: list[dict[str, Any]] = []
        self._excluded_target_keys: set[str] = set()
        self._schedule_rounds: dict[str, int] | None = None

    def _ensure_deps(self) -> None:
        """Lazily initialize storage, logger, and metrics dependencies."""
        if self._initialized:
            return

        from src.persistence.artifact_store import ArtifactStore
        from src.utils.prediction_logger import PredictionLogger
        from src.utils.prediction_metrics import PredictionMetrics

        logger_inst = PredictionLogger()
        self._prediction_logger = logger_inst
        self._metrics_calculator = PredictionMetrics()
        self._artifact_store = ArtifactStore(data_root="data")
        self._actuals_reconciled = 0
        self._snapshots_written = 0
        if self.reconcile_actuals_on_load:
            self._actuals_reconciled = self._run_actuals_reconciliation(logger_inst)
        self._predictions = logger_inst.get_all_predictions(self.year)
        self._initialized = True

    def reconcile_actuals(self) -> int:
        """Fetch actuals, rebuild snapshot artifacts, and refresh the in-memory season state."""
        self._ensure_deps()
        if self._prediction_logger is None:
            return 0

        self._actuals_reconciled = self._run_actuals_reconciliation(self._prediction_logger)
        self._predictions = self._prediction_logger.get_all_predictions(self.year)
        self._snapshots_written = self._sync_snapshot_artifacts()
        self._clear_cached_summary_state()
        return self._actuals_reconciled

    def build_summary(self) -> SeasonAccuracySummary:
        """Return a target-aware season summary for the configured year."""
        self._ensure_deps()

        summary = SeasonAccuracySummary(year=self.year, n_predictions=len(self._predictions))
        if not self._predictions:
            self._prediction_status_rows = []
            self._excluded_predictions = []
            self._excluded_target_keys = set()
            return summary

        snapshot_map = self._load_snapshot_map()
        records, target_status_records, status_rows, excluded_target_count = (
            self._collect_target_records(snapshot_map)
        )
        self._prediction_status_rows = status_rows
        self._excluded_predictions = [
            row for row in status_rows if int(row.get("scored_target_count", 0)) <= 0
        ]
        self._excluded_target_keys = {
            str(row["artifact_key"])
            for row in status_rows
            if int(row.get("excluded_target_count", 0)) > 0
        }

        summary.n_excluded_predictions = len(self._excluded_predictions)
        summary.n_excluded_targets = excluded_target_count
        summary.targets = self._build_target_summaries(records, target_status_records)
        summary.qualifying_aggregate = summary.targets.get(
            "main_qualifying", TargetAccuracySummary("main_qualifying", "")
        ).aggregate
        summary.race_aggregate = summary.targets.get(
            "grand_prix_race", TargetAccuracySummary("grand_prix_race", "")
        ).aggregate
        summary.per_race = self._build_per_race_snapshots(records)
        return summary

    @property
    def all_predictions(self) -> list[dict[str, Any]]:
        """Return all saved predictions for the configured season."""
        self._ensure_deps()
        return self._predictions

    @property
    def predictions_with_actuals(self) -> list[dict[str, Any]]:
        """Return predictions that already have at least one scored target."""
        self._ensure_deps()
        return [
            prediction
            for prediction in self._predictions
            if any(
                (
                    explicit_target_actuals(prediction)
                    or synthesize_legacy_actuals(
                        prediction,
                        is_sprint=self._prediction_is_sprint(prediction),
                    )
                ).values()
            )
        ]

    @property
    def has_actuals(self) -> bool:
        """Return True when at least one saved prediction has actuals attached."""
        return bool(self.predictions_with_actuals)

    @property
    def actuals_reconciled(self) -> int:
        """Return the number of predictions reconciled during page load."""
        self._ensure_deps()
        return self._actuals_reconciled

    @property
    def snapshots_written(self) -> int:
        """Return the number of snapshot artifacts written during the last refresh."""
        self._ensure_deps()
        return self._snapshots_written

    @property
    def prediction_status_rows(self) -> list[dict[str, Any]]:
        """Return user-facing status rows for the saved-prediction list."""
        self._ensure_deps()
        if not self._prediction_status_rows:
            self.build_summary()
        return self._prediction_status_rows

    def _clear_cached_summary_state(self) -> None:
        """Reset cached derived summary state after underlying data changes."""
        self._prediction_status_rows = []
        self._excluded_predictions = []
        self._excluded_target_keys = set()

    def _schedule_round_map(self) -> dict[str, int]:
        """Return normalized race names keyed to their calendar round number."""
        if self._schedule_rounds is not None:
            return self._schedule_rounds

        try:
            schedule_rows = get_schedule_rows(self.year)
        except Exception as exc:
            logger.warning("Could not load %s schedule round order: %s", self.year, exc)
            self._schedule_rounds = {}
            return self._schedule_rounds

        round_map: dict[str, int] = {}
        for round_number, schedule_row in enumerate(schedule_rows, start=1):
            if not isinstance(schedule_row, tuple) or not schedule_row:
                continue
            race_name = str(schedule_row[0]).strip()
            if race_name:
                round_map[_normalize_race_name(race_name)] = round_number
        self._schedule_rounds = round_map
        return self._schedule_rounds

    def _run_actuals_reconciliation(self, logger_inst: Any) -> int:
        """Reconcile stored predictions with completed-session actuals when requested."""
        try:
            return int(logger_inst.reconcile_completed_prediction_actuals(self.year))
        except Exception as exc:
            logger.warning(
                "Could not reconcile completed prediction actuals for %s: %s", self.year, exc
            )
            return 0

    def _sync_snapshot_artifacts(self) -> int:
        """Persist fresh accuracy snapshots so dashboard cards and charts stay in sync."""
        if self._artifact_store is None or self._metrics_calculator is None:
            return 0

        from src.utils.accuracy_snapshots import build_accuracy_snapshot_records

        snapshots_written = 0
        for prediction in self._predictions:
            metadata = prediction.get("metadata", {})
            snapshot_records = build_accuracy_snapshot_records(
                prediction_data=prediction,
                is_sprint=self._prediction_is_sprint(prediction),
                metrics_calculator=self._metrics_calculator,
                generated_by="dashboard_accuracy_refresh",
            )
            for record in snapshot_records:
                try:
                    self._artifact_store.save_artifact(
                        artifact_type="accuracy_snapshot",
                        artifact_key=record["artifact_key"],
                        data=record["data"],
                        version=1,
                        run_id=metadata.get("run_id"),
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not save accuracy snapshot %s for %s: %s",
                        record.get("artifact_key"),
                        metadata.get("race_name"),
                        exc,
                    )
                    continue
                snapshots_written += 1
        return snapshots_written

    def _load_snapshot_map(self) -> dict[str, dict[str, Any]]:
        """Load persisted accuracy snapshots keyed by artifact key."""
        if self._artifact_store is None:
            return {}
        try:
            snapshot_rows = self._artifact_store.list_artifacts(
                "accuracy_snapshot",
                key_prefix=f"{self.year}::",
                limit=8192,
            )
        except Exception as exc:
            logger.warning("Could not list accuracy snapshots for %s: %s", self.year, exc)
            return {}

        snapshot_map: dict[str, dict[str, Any]] = {}
        for row in snapshot_rows:
            artifact_key = str(row.get("artifact_key", "")).strip()
            payload = row.get("data")
            if artifact_key and isinstance(payload, dict):
                snapshot_map[artifact_key] = payload
        return snapshot_map

    def _collect_target_records(
        self,
        snapshot_map: dict[str, dict[str, Any]],
    ) -> tuple[list[_TargetAccuracyRecord], list[_TargetStatusRecord], list[dict[str, Any]], int]:
        """Collect normalized target records and status rows from saved predictions."""
        records: list[_TargetAccuracyRecord] = []
        target_status_records: list[_TargetStatusRecord] = []
        status_rows: list[dict[str, Any]] = []
        excluded_targets = 0
        schedule_rounds = self._schedule_round_map()
        fallback_race_order: dict[str, int] = {}

        for prediction in self._predictions:
            metadata = prediction.get("metadata", {})
            race_name = str(metadata.get("race_name", "")).strip()
            checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
            predicted_at = str(metadata.get("predicted_at", "")).strip()
            if not race_name or not checkpoint_session:
                continue

            normalized_race_name = _normalize_race_name(race_name)
            fallback_race_order.setdefault(normalized_race_name, len(fallback_race_order))
            round_number = schedule_rounds.get(normalized_race_name)
            is_sprint = self._prediction_is_sprint(prediction)
            weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
            if weekend_format not in {"normal", "sprint"}:
                weekend_format = weekend_format_name(is_sprint)

            target_predictions = explicit_target_predictions(prediction)
            if not target_predictions:
                target_predictions = synthesize_legacy_targets(prediction, is_sprint=is_sprint)
            target_actuals = explicit_target_actuals(prediction)
            if not target_actuals:
                target_actuals = synthesize_legacy_actuals(prediction, is_sprint=is_sprint)

            scored_target_count = 0
            pending_target_count = 0
            excluded_target_count = 0
            target_labels: list[str] = []

            fallback_metrics = calculate_target_metric_map(
                metrics_calculator=self._metrics_calculator,
                prediction_data=prediction,
                is_sprint=is_sprint,
            )
            checkpoint_records: list[_TargetAccuracyRecord] = []
            for target_key, target_payload in target_predictions.items():
                target_labels.append(target_label(target_key))
                if not bool(target_payload.get("eligible_at_save", True)):
                    target_status_records.append(
                        _TargetStatusRecord(
                            race_name=race_name,
                            checkpoint_session=checkpoint_session,
                            weekend_format=weekend_format,
                            target_key=target_key,
                            status="excluded",
                        )
                    )
                    excluded_target_count += 1
                    excluded_targets += 1
                    continue

                artifact_key = accuracy_snapshot_artifact_key(
                    year=self.year,
                    race_name=race_name,
                    checkpoint_session=checkpoint_session,
                    target_key=target_key,
                )
                snapshot_payload = snapshot_map.get(artifact_key, {})
                metrics = {}
                if isinstance(snapshot_payload, dict):
                    metrics_payload = snapshot_payload.get("metrics")
                    if isinstance(metrics_payload, dict):
                        metrics = metrics_payload
                if not metrics:
                    metrics = fallback_metrics.get(target_key, {})

                if metrics:
                    checkpoint_records.append(
                        _TargetAccuracyRecord(
                            race_name=race_name,
                            checkpoint_session=checkpoint_session,
                            weekend_format=weekend_format,
                            target_key=target_key,
                            metrics={
                                key: float(value)
                                for key, value in metrics.items()
                                if isinstance(value, int | float) and np.isfinite(float(value))
                            },
                            predicted_at=predicted_at,
                        )
                    )
                    scored_target_count += 1
                    target_status_records.append(
                        _TargetStatusRecord(
                            race_name=race_name,
                            checkpoint_session=checkpoint_session,
                            weekend_format=weekend_format,
                            target_key=target_key,
                            status="scored",
                        )
                    )
                    continue

                if target_actuals.get(target_key):
                    target_status_records.append(
                        _TargetStatusRecord(
                            race_name=race_name,
                            checkpoint_session=checkpoint_session,
                            weekend_format=weekend_format,
                            target_key=target_key,
                            status="excluded",
                        )
                    )
                    excluded_target_count += 1
                    excluded_targets += 1
                else:
                    target_status_records.append(
                        _TargetStatusRecord(
                            race_name=race_name,
                            checkpoint_session=checkpoint_session,
                            weekend_format=weekend_format,
                            target_key=target_key,
                            status="pending",
                        )
                    )
                    pending_target_count += 1

            records.extend(checkpoint_records)
            status_rows.append(
                {
                    "artifact_key": f"{self.year}::{race_name}::{checkpoint_session}",
                    "race_name": race_name,
                    "checkpoint_session": checkpoint_session,
                    "weekend_format": weekend_format,
                    "round_number": round_number,
                    "_race_order": fallback_race_order[normalized_race_name],
                    "target_labels": sorted(set(target_labels)),
                    "scored_target_count": scored_target_count,
                    "pending_target_count": pending_target_count,
                    "excluded_target_count": excluded_target_count,
                    "status_text": self._status_text(
                        scored_target_count=scored_target_count,
                        pending_target_count=pending_target_count,
                        excluded_target_count=excluded_target_count,
                    ),
                }
            )

        status_rows.sort(
            key=lambda row: (
                row.get("round_number") is None,
                int(row.get("round_number") or row.get("_race_order", 0)),
                CHECKPOINT_ORDER.get(str(row.get("checkpoint_session", "")).upper(), 99),
                str(row.get("checkpoint_session", "")),
            )
        )
        for row in status_rows:
            row.pop("_race_order", None)
        return records, target_status_records, status_rows, excluded_targets

    def _build_target_summaries(
        self,
        records: list[_TargetAccuracyRecord],
        target_status_records: list[_TargetStatusRecord],
    ) -> dict[str, TargetAccuracySummary]:
        """Build target summaries from normalized score records."""
        schedule_rounds = self._schedule_round_map()
        fallback_race_order: dict[str, int] = {}
        for record in records:
            normalized_race_name = _normalize_race_name(record.race_name)
            fallback_race_order.setdefault(normalized_race_name, len(fallback_race_order))

        schedule_offset = len(schedule_rounds) + 1
        race_order: dict[str, int] = {}
        for race_name_normalized, fallback_index in fallback_race_order.items():
            race_order[race_name_normalized] = schedule_rounds.get(
                race_name_normalized,
                schedule_offset + fallback_index if schedule_rounds else fallback_index,
            )

        grouped: dict[str, list[_TargetAccuracyRecord]] = {}
        for record in records:
            grouped.setdefault(record.target_key, []).append(record)

        grouped_status: dict[str, list[_TargetStatusRecord]] = {}
        for status_record in target_status_records:
            grouped_status.setdefault(status_record.target_key, []).append(status_record)

        summaries: dict[str, TargetAccuracySummary] = {}
        for target_key, target_records in grouped.items():
            aggregate = self._aggregate_metric_rows([record.metrics for record in target_records])
            checkpoint_progression = self._build_checkpoint_progression(target_key, target_records)
            checkpoint_status = self._build_checkpoint_status(
                target_key,
                grouped_status.get(target_key, []),
            )
            season_trend = [
                SeasonTrendPoint(
                    target_key=target_key,
                    weekend_format=record.weekend_format,
                    race_name=record.race_name,
                    race_order=race_order[_normalize_race_name(record.race_name)],
                    checkpoint_session=record.checkpoint_session,
                    checkpoint_index=target_checkpoint_index(
                        target_key,
                        record.weekend_format,
                        record.checkpoint_session,
                    ),
                    metrics=record.metrics,
                )
                for record in sorted(
                    target_records,
                    key=lambda item: (
                        race_order[_normalize_race_name(item.race_name)],
                        target_checkpoint_index(
                            target_key,
                            item.weekend_format,
                            item.checkpoint_session,
                        ),
                    ),
                )
            ]
            summaries[target_key] = TargetAccuracySummary(
                target_key=target_key,
                label=target_label(target_key),
                aggregate=aggregate,
                checkpoint_progression=checkpoint_progression,
                checkpoint_status=checkpoint_status,
                season_trend=season_trend,
                n_scored_predictions=len(target_records),
            )

        return summaries

    def _build_checkpoint_progression(
        self,
        target_key: str,
        records: list[_TargetAccuracyRecord],
    ) -> list[CheckpointAccuracyPoint]:
        """Aggregate one target into weekend-progression points."""
        grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
        for record in records:
            grouped.setdefault(
                (record.weekend_format, record.checkpoint_session),
                [],
            ).append(record.metrics)

        points: list[CheckpointAccuracyPoint] = []
        for (weekend_format, checkpoint_session), metric_rows in grouped.items():
            points.append(
                CheckpointAccuracyPoint(
                    target_key=target_key,
                    weekend_format=weekend_format,
                    checkpoint_session=checkpoint_session,
                    checkpoint_index=target_checkpoint_index(
                        target_key,
                        weekend_format,
                        checkpoint_session,
                    ),
                    metrics={
                        key: values["mean"]
                        for key, values in self._aggregate_metric_rows(metric_rows).items()
                    },
                    race_count=len(metric_rows),
                )
            )

        return sorted(
            points,
            key=lambda point: (point.weekend_format, point.checkpoint_index),
        )

    def _build_checkpoint_status(
        self,
        target_key: str,
        status_records: list[_TargetStatusRecord],
    ) -> list[CheckpointStatusPoint]:
        """Aggregate one target into checkpoint save-status counts."""
        grouped: dict[tuple[str, str], CheckpointStatusPoint] = {}
        for record in status_records:
            key = (record.weekend_format, record.checkpoint_session)
            status_point = grouped.setdefault(
                key,
                CheckpointStatusPoint(
                    target_key=target_key,
                    weekend_format=record.weekend_format,
                    checkpoint_session=record.checkpoint_session,
                    checkpoint_index=target_checkpoint_index(
                        target_key,
                        record.weekend_format,
                        record.checkpoint_session,
                    ),
                ),
            )
            if record.status == "scored":
                status_point.scored_count += 1
            elif record.status == "pending":
                status_point.pending_count += 1
            elif record.status == "excluded":
                status_point.excluded_count += 1

        return sorted(
            grouped.values(),
            key=lambda point: (point.weekend_format, point.checkpoint_index),
        )

    def _build_per_race_snapshots(
        self,
        records: list[_TargetAccuracyRecord],
    ) -> list[RaceAccuracySnapshot]:
        """Build compatibility per-race snapshots from target records."""
        grouped: dict[tuple[str, str], RaceAccuracySnapshot] = {}
        for record in records:
            key = (record.race_name, record.checkpoint_session)
            snapshot = grouped.setdefault(
                key,
                RaceAccuracySnapshot(
                    race_name=record.race_name,
                    session_name=record.checkpoint_session,
                ),
            )
            snapshot.targets[record.target_key] = record.metrics
            if record.target_key == "main_qualifying":
                snapshot.qualifying = record.metrics
            if record.target_key == "grand_prix_race":
                snapshot.race = record.metrics
        schedule_rounds = self._schedule_round_map()
        fallback_race_order: dict[str, int] = {}
        for snapshot in grouped.values():
            normalized_race_name = _normalize_race_name(snapshot.race_name)
            fallback_race_order.setdefault(normalized_race_name, len(fallback_race_order))
        schedule_offset = len(schedule_rounds) + 1
        return sorted(
            grouped.values(),
            key=lambda snapshot: (
                schedule_rounds.get(
                    _normalize_race_name(snapshot.race_name),
                    schedule_offset + fallback_race_order[_normalize_race_name(snapshot.race_name)]
                    if schedule_rounds
                    else fallback_race_order[_normalize_race_name(snapshot.race_name)],
                ),
                CHECKPOINT_ORDER.get(snapshot.session_name, 99),
                snapshot.session_name,
            ),
        )

    @staticmethod
    def _aggregate_metric_rows(metric_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        """Aggregate metric rows into mean and standard deviation by metric name."""
        aggregate: dict[str, dict[str, float]] = {}
        if not metric_rows:
            return aggregate
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
    def _status_text(
        *,
        scored_target_count: int,
        pending_target_count: int,
        excluded_target_count: int,
    ) -> str:
        """Build a compact status line for one saved checkpoint."""
        parts: list[str] = []
        if scored_target_count > 0:
            parts.append(f"{scored_target_count} scored")
        if pending_target_count > 0:
            parts.append(f"{pending_target_count} pending")
        if excluded_target_count > 0:
            parts.append(f"{excluded_target_count} excluded")
        return ", ".join(parts) if parts else "No scoreable targets"

    @staticmethod
    def _prediction_is_sprint(prediction: dict[str, Any]) -> bool:
        """Infer weekend format from saved metadata with a legacy fallback."""
        metadata = prediction.get("metadata", {})
        weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
        if weekend_format in {"normal", "sprint"}:
            return weekend_format == "sprint"

        target_predictions = explicit_target_predictions(prediction)
        if any("sprint" in target_key for target_key in target_predictions):
            return True

        checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
        if checkpoint_session in {"SQ", "SPRINT"}:
            return True
        if checkpoint_session in {"FP2", "FP3"}:
            return False
        qualifying_target = str(metadata.get("top_level_qualifying_target", "")).strip()
        race_target = str(metadata.get("top_level_race_target", "")).strip()
        if qualifying_target or race_target:
            return (
                qualifying_target not in PRIMARY_TARGET_KEYS
                or race_target not in PRIMARY_TARGET_KEYS
            )
        return False


def _normalize_race_name(race_name: str) -> str:
    """Normalize one race name for case-insensitive schedule lookups."""
    without_accents = unicodedata.normalize("NFKD", str(race_name)).encode(
        "ascii",
        "ignore",
    )
    return " ".join(without_accents.decode("ascii").split()).lower()
