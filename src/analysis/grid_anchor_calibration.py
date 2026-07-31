"""Offline calibration of source-specific post-simulation grid anchors.

This module is research-only.  It returns a deterministic, config-compatible
mapping but never writes configuration or activates a runtime challenger.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

GRID_ANCHOR_CALIBRATION_SCHEMA_VERSION = 2
ALLOWED_GRID_SOURCE_DETAILS = (
    "predicted_joint",
    "predicted_marginal_fallback",
    "actual_qualifying",
    "actual_starting_grid",
)
_REQUIRED_FIELDS = (
    "event_id",
    "event_at",
    "grid_source_detail",
    "simulated_position",
    "grid_position",
    "actual_position",
)
_DRIVER_ID_FIELDS = ("driver_id", "driver_code", "driver")


@dataclass(frozen=True)
class _ReplayRow:
    event_id: str
    event_at: str
    grid_source_detail: str
    driver_identity: str | None
    simulated_position: float
    grid_position: float
    actual_position: float


def _normalise_timestamp(value: Any, *, field_name: str) -> tuple[str, datetime]:
    """Return one required timezone-aware timestamp in canonical UTC form."""

    candidate = value
    if isinstance(candidate, str):
        text = candidate.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            candidate = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if (
        not isinstance(candidate, datetime)
        or candidate.tzinfo is None
        or candidate.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must include a timezone")
    utc_value = candidate.astimezone(UTC)
    return utc_value.isoformat().replace("+00:00", "Z"), utc_value


def _driver_identity(row: Mapping[str, Any], *, row_index: int) -> str | None:
    identities: list[str] = []
    for field_name in _DRIVER_ID_FIELDS:
        if field_name not in row:
            continue
        value = str(row[field_name]).strip() if row[field_name] is not None else ""
        if not value:
            raise ValueError(f"row {row_index} {field_name} must not be blank")
        identities.append(value)
    if not identities:
        return None
    if len(set(identities)) != 1:
        raise ValueError(f"row {row_index} contains conflicting driver identifiers")
    # Field aliases describe the same identity.  Omitting the alias name prevents
    # duplicate observations from bypassing validation by switching columns.
    return identities[0]


def _finite_position(value: Any, *, field_name: str, row_index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"row {row_index} {field_name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index} {field_name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"row {row_index} {field_name} must be a finite positive number")
    return result


def _normalise_rows(
    rows: Sequence[Mapping[str, Any]],
    allowed_sources: tuple[str, ...],
    *,
    cutoff: datetime,
) -> tuple[list[_ReplayRow], dict[str, str]]:
    normalised: list[_ReplayRow] = []
    allowed = set(allowed_sources)
    event_timestamps: dict[str, str] = {}
    exact_rows: dict[_ReplayRow, int] = {}
    driver_rows: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(rows):
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} is missing required fields: {', '.join(missing)}")
        event_id = str(row["event_id"]).strip()
        source = str(row["grid_source_detail"]).strip()
        if not event_id:
            raise ValueError(f"row {index} event_id must not be blank")
        if source not in allowed:
            raise ValueError(f"row {index} grid_source_detail is not allowed: {source!r}")
        event_at, event_datetime = _normalise_timestamp(
            row["event_at"], field_name=f"row {index} event_at"
        )
        if event_datetime >= cutoff:
            raise ValueError(f"row {index} event_at must be strictly before cutoff_at")
        previous_event_at = event_timestamps.setdefault(event_id, event_at)
        if previous_event_at != event_at:
            raise ValueError(f"event {event_id!r} has inconsistent event_at timestamps")

        driver_identity = _driver_identity(row, row_index=index)
        replay_row = _ReplayRow(
            event_id=event_id,
            event_at=event_at,
            grid_source_detail=source,
            driver_identity=driver_identity,
            simulated_position=_finite_position(
                row["simulated_position"], field_name="simulated_position", row_index=index
            ),
            grid_position=_finite_position(
                row["grid_position"], field_name="grid_position", row_index=index
            ),
            actual_position=_finite_position(
                row["actual_position"], field_name="actual_position", row_index=index
            ),
        )
        if replay_row in exact_rows:
            raise ValueError(f"row {index} duplicates replay row {exact_rows[replay_row]}")
        exact_rows[replay_row] = index

        if driver_identity is not None:
            driver_key = (event_id, source, driver_identity)
            if driver_key in driver_rows:
                raise ValueError(
                    f"row {index} duplicates driver/source observation from row "
                    f"{driver_rows[driver_key]}"
                )
            driver_rows[driver_key] = index
        normalised.append(replay_row)

    normalised.sort(
        key=lambda replay_row: (
            replay_row.event_at,
            replay_row.event_id,
            replay_row.grid_source_detail,
            replay_row.driver_identity or "",
            replay_row.simulated_position,
            replay_row.grid_position,
            replay_row.actual_position,
        )
    )
    return normalised, event_timestamps


def _mean_event_mae(rows: Sequence[_ReplayRow], weight: float) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        anchored = ((1.0 - weight) * row.simulated_position) + (weight * row.grid_position)
        by_event[row.event_id].append(abs(anchored - row.actual_position))
    event_maes = [
        math.fsum(by_event[event_id]) / len(by_event[event_id]) for event_id in sorted(by_event)
    ]
    return math.fsum(event_maes) / len(event_maes)


def _candidate_weights(rows: Sequence[_ReplayRow]) -> list[float]:
    """Return all in-range breakpoints of the convex absolute-error loss."""
    candidates = {0.0, 1.0}
    for row in rows:
        delta = row.grid_position - row.simulated_position
        if delta == 0.0:
            continue
        breakpoint = (row.actual_position - row.simulated_position) / delta
        if 0.0 <= breakpoint <= 1.0:
            candidates.add(float(breakpoint))
    return sorted(candidates)


def _input_digest(rows: Sequence[_ReplayRow], *, cutoff_at: str) -> str:
    payload = {
        "cutoff_at": cutoff_at,
        "rows": [
            {
                "actual_position": row.actual_position,
                "driver_identity": row.driver_identity,
                "event_at": row.event_at,
                "event_id": row.event_id,
                "grid_position": row.grid_position,
                "grid_source_detail": row.grid_source_detail,
                "simulated_position": row.simulated_position,
            }
            for row in rows
        ],
    }
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_event_ids(values: Sequence[str], *, field_name: str) -> set[str]:
    if isinstance(values, str | bytes):
        raise ValueError(f"{field_name} must be a sequence of event IDs")
    event_ids: set[str] = set()
    for index, value in enumerate(values):
        event_id = str(value).strip()
        if not event_id:
            raise ValueError(f"{field_name}[{index}] must not be blank")
        if event_id in event_ids:
            raise ValueError(f"{field_name} must not contain duplicate event IDs")
        event_ids.add(event_id)
    return event_ids


def validate_grid_anchor_event_separation(
    calibration_training_event_ids: Sequence[str],
    *,
    promotion_event_ids: Sequence[str] = (),
    evaluation_event_ids: Sequence[str] = (),
) -> None:
    """Reject promotion or evaluation events used to fit grid anchors."""

    training = _normalise_event_ids(
        calibration_training_event_ids,
        field_name="calibration_training_event_ids",
    )
    promotion = _normalise_event_ids(promotion_event_ids, field_name="promotion_event_ids")
    evaluation = _normalise_event_ids(evaluation_event_ids, field_name="evaluation_event_ids")
    promotion_overlap = sorted(training.intersection(promotion))
    evaluation_overlap = sorted(training.intersection(evaluation))
    if promotion_overlap or evaluation_overlap:
        overlap_details: list[str] = []
        if promotion_overlap:
            overlap_details.append(f"promotion={promotion_overlap}")
        if evaluation_overlap:
            overlap_details.append(f"evaluation={evaluation_overlap}")
        raise ValueError(
            "grid-anchor calibration training events overlap held-out events: "
            + ", ".join(overlap_details)
        )


def fit_source_specific_grid_anchors(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoff_at: datetime | str,
    min_events: int = 8,
    allowed_source_details: Sequence[str] = ALLOWED_GRID_SOURCE_DETAILS,
    objective_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Fit cutoff-bound post-simulation anchors using equal-total-weight events.

    For weight ``w``, the evaluated position is
    ``(1-w)*simulated_position + w*grid_position``.  The objective is the mean
    of each event's driver-level MAE, so every weekend has equal total weight.
    The absolute-error objective is convex and piecewise linear; evaluating its
    exact breakpoints finds a global minimum.  If several weights tie within
    ``objective_tolerance``, the lowest is selected to minimize intervention.
    Every replay row must identify its timezone-aware ``event_at`` and that
    timestamp must be strictly earlier than the timezone-aware ``cutoff_at``.
    """
    normalised_cutoff, cutoff_datetime = _normalise_timestamp(
        cutoff_at,
        field_name="cutoff_at",
    )
    if isinstance(min_events, bool) or int(min_events) != min_events or min_events < 1:
        raise ValueError("min_events must be a positive integer")
    if not math.isfinite(float(objective_tolerance)) or objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and non-negative")

    allowed_sources = tuple(str(value).strip() for value in allowed_source_details)
    if not allowed_sources or any(not value for value in allowed_sources):
        raise ValueError("allowed_source_details must contain non-blank values")
    if len(set(allowed_sources)) != len(allowed_sources):
        raise ValueError("allowed_source_details must not contain duplicates")
    unknown = sorted(set(allowed_sources) - set(ALLOWED_GRID_SOURCE_DETAILS))
    if unknown:
        raise ValueError(f"unsupported allowed_source_details: {', '.join(unknown)}")

    normalised, event_timestamps = _normalise_rows(
        rows,
        allowed_sources,
        cutoff=cutoff_datetime,
    )
    grouped: dict[str, list[_ReplayRow]] = defaultdict(list)
    for row in normalised:
        grouped[row.grid_source_detail].append(row)

    source_calibrated: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for source in allowed_sources:
        source_rows = grouped.get(source, [])
        event_ids = sorted({row.event_id for row in source_rows})
        base: dict[str, Any] = {
            "event_count": len(event_ids),
            "event_ids": event_ids,
            "row_count": len(source_rows),
            "minimum_event_count": int(min_events),
        }
        if len(event_ids) < min_events:
            base["status"] = "no_rows" if not source_rows else "insufficient_events"
            diagnostics[source] = base
            continue

        candidates = _candidate_weights(source_rows)
        objectives = [(weight, _mean_event_mae(source_rows, weight)) for weight in candidates]
        minimum_objective = min(objective for _, objective in objectives)
        tied = [
            weight
            for weight, objective in objectives
            if objective <= minimum_objective + objective_tolerance
        ]
        selected_weight = min(tied)
        selected_objective = _mean_event_mae(source_rows, selected_weight)
        pre_anchor_mae = _mean_event_mae(source_rows, 0.0)
        grid_only_mae = _mean_event_mae(source_rows, 1.0)
        source_calibrated[source] = selected_weight
        base.update(
            {
                "status": "fitted",
                "selected_weight": selected_weight,
                "mean_event_mae": selected_objective,
                "pre_anchor_mean_event_mae": pre_anchor_mae,
                "grid_only_mean_event_mae": grid_only_mae,
                "absolute_mae_improvement": pre_anchor_mae - selected_objective,
                "relative_mae_improvement": (
                    (pre_anchor_mae - selected_objective) / pre_anchor_mae
                    if pre_anchor_mae > 0.0
                    else 0.0
                ),
                "candidate_count": len(candidates),
                "tied_minimizer_count": len(tied),
                "event_row_counts": {
                    event_id: sum(row.event_id == event_id for row in source_rows)
                    for event_id in event_ids
                },
            }
        )
        diagnostics[source] = base

    training_event_ids = sorted(
        event_timestamps,
        key=lambda event_id: (event_timestamps[event_id], event_id),
    )
    return {
        "artifact_type": "grid_anchor_calibration",
        "schema_version": GRID_ANCHOR_CALIBRATION_SCHEMA_VERSION,
        "runtime_activation_allowed": False,
        "source_calibrated": source_calibrated,
        "diagnostics": diagnostics,
        "provenance": {
            "fit_method": "exact_breakpoint_mean_event_mae",
            "event_weighting": "equal_total_weight_per_event",
            "tie_break": "lowest_weight",
            "cutoff_at": normalised_cutoff,
            "training_event_ids": training_event_ids,
            "training_event_timestamps": [
                event_timestamps[event_id] for event_id in training_event_ids
            ],
            "minimum_event_count": int(min_events),
            "objective_tolerance": float(objective_tolerance),
            "allowed_source_details": list(allowed_sources),
            "input_row_count": len(normalised),
            "input_event_count": len({row.event_id for row in normalised}),
            "input_sha256": _input_digest(normalised, cutoff_at=normalised_cutoff),
        },
    }
