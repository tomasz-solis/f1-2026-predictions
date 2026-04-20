"""Generate a calibration and bias evaluation report from saved predictions.

Reads all stored prediction artifacts for a season, pairs each one with its
saved actual results, and produces review-oriented analyses:

  1. Accuracy and segment breakdowns — where does the model hold up or fail?
  2. Calibration — are the Monte Carlo p5/p95 bands empirically honest?
  3. Systematic bias — which drivers and teams does the model consistently
     get wrong in the same direction?
  4. Baseline comparison — does the model beat the naive previous-race
     classifier on MAE and rank correlation?
  5. Error analysis — which weekends and drivers show up repeatedly among
     the biggest misses?

Outputs:
  data/evaluation/<year>_evaluation_report.json   — raw numbers
  docs/MODEL_CALIBRATION.md                       — human-readable summary
  docs/MODEL_ERROR_ANALYSIS.md                    — compact failure-mode summary

Usage:
  python scripts/generate_evaluation_report.py --year 2026
  python scripts/generate_evaluation_report.py --year 2026 --out docs/MODEL_CALIBRATION.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("USE_DB_STORAGE", "file_only")

from src.analysis.model_evaluation import (
    build_confidence_bands,
    compute_calibration_metrics,
    compute_improvement_over_baseline,
    compute_prediction_accuracy,
    identify_systematic_errors,
)
from src.utils.accuracy_targets import (
    CHECKPOINT_ORDER,
    explicit_target_actuals,
    explicit_target_predictions,
    normalize_checkpoint_session,
    sanitize_actual_rows,
    sanitize_prediction_rows,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
_TRACK_CHARACTERISTICS_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "track_characteristics"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_predictions_from_files(year: int, predictions_dir: Path) -> list[dict[str, Any]]:
    """Load all prediction JSON files for the given season year."""
    year_dir = predictions_dir / str(year)
    if not year_dir.exists():
        return []

    predictions: list[dict[str, Any]] = []
    for race_dir in sorted(year_dir.iterdir()):
        if not race_dir.is_dir():
            continue
        for pred_file in sorted(race_dir.glob("*.json")):
            try:
                with open(pred_file) as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict) and "metadata" in payload:
                    predictions.append(payload)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping %s: %s", pred_file, exc)

    return predictions


def _parse_saved_datetime(value: object) -> datetime | None:
    """Parse an ISO timestamp into an aware UTC datetime."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _prediction_sort_key(prediction: dict[str, Any]) -> tuple[datetime, int]:
    """Sort predictions by information cutoff, then by checkpoint order."""
    metadata = prediction.get("metadata", {})
    predicted_at = (
        _parse_saved_datetime(metadata.get("information_cutoff_at"))
        or _parse_saved_datetime(metadata.get("predicted_at"))
        or datetime.min.replace(tzinfo=UTC)
    )
    checkpoint = normalize_checkpoint_session(metadata.get("session_name"))
    checkpoint_order = CHECKPOINT_ORDER.get(checkpoint, -1)
    return predicted_at, checkpoint_order


def _selection_target_key(prediction: dict[str, Any], *, session_kind: str) -> str:
    """Resolve the target key represented by a top-level qualifying/race payload."""
    metadata = prediction.get("metadata", {})
    if session_kind == "qualifying":
        raw_target = metadata.get("top_level_qualifying_target")
    elif session_kind == "race":
        raw_target = metadata.get("top_level_race_target")
    else:
        raise ValueError(f"Unsupported session kind: {session_kind}")

    target_key = str(raw_target or "").strip().lower()
    if target_key:
        return target_key
    return f"legacy_{session_kind}"


def _select_latest_predictions(
    predictions: list[dict[str, Any]],
    *,
    session_kind: str,
) -> list[dict[str, Any]]:
    """Keep only the latest checkpoint artifact per race and top-level target."""
    selected: dict[tuple[str, str], dict[str, Any]] = {}

    for prediction in predictions:
        metadata = prediction.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        if not race_name:
            continue

        predicted_rows, actual_rows = _resolve_session_pair(prediction, session_kind=session_kind)
        if not predicted_rows or not actual_rows:
            continue

        selection_key = (
            race_name,
            _selection_target_key(prediction, session_kind=session_kind),
        )
        existing = selected.get(selection_key)
        if existing is None or _prediction_sort_key(prediction) > _prediction_sort_key(existing):
            selected[selection_key] = prediction

    return list(selected.values())


def _resolve_session_pair(
    prediction: dict[str, Any],
    *,
    session_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical predicted and actual rows for one session.

    The report selects one artifact per race and top-level target. Once that
    selection is made, scoring should read the same target-aware payload that
    the artifact represents. Older artifacts still fall back to the legacy
    top-level session fields when explicit target payloads are unavailable.
    """
    target_key = _selection_target_key(prediction, session_kind=session_kind)
    target_predictions = explicit_target_predictions(prediction)
    target_actuals = explicit_target_actuals(prediction)

    if not target_key.startswith("legacy_"):
        target_payload = target_predictions.get(target_key)
        actual_rows = target_actuals.get(target_key)
        if isinstance(target_payload, dict) and actual_rows:
            predicted_rows = sanitize_prediction_rows(target_payload.get("predicted_order"))
            if predicted_rows:
                return predicted_rows, actual_rows

    payload_key = "predicted_grid" if session_kind == "qualifying" else "predicted_results"
    predicted_rows = sanitize_prediction_rows((prediction.get(session_kind) or {}).get(payload_key))
    actual_rows = sanitize_actual_rows((prediction.get("actuals") or {}).get(session_kind))
    return predicted_rows, actual_rows


def _checkpoint_breakdown(predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Count selected predictions by saved checkpoint session."""
    counts: dict[str, int] = {}
    for prediction in predictions:
        checkpoint = normalize_checkpoint_session(
            (prediction.get("metadata") or {}).get("session_name")
        )
        label = checkpoint or "UNKNOWN"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _target_breakdown(
    predictions: list[dict[str, Any]],
    *,
    session_kind: str,
) -> dict[str, int]:
    """Count selected predictions by top-level target key."""
    counts: dict[str, int] = {}
    for prediction in predictions:
        target_key = _selection_target_key(prediction, session_kind=session_kind)
        counts[target_key] = counts.get(target_key, 0) + 1
    return dict(sorted(counts.items()))


def _extract_qualifying_pairs(
    predictions: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Return (predicted_grids, actual_grids) for predictions that have both."""
    predicted: list[list[dict[str, Any]]] = []
    actual: list[list[dict[str, Any]]] = []

    for pred in predictions:
        grid, actuals = _resolve_session_pair(pred, session_kind="qualifying")

        if not grid or not actuals:
            continue

        predicted.append(grid)
        actual.append(actuals)

    return predicted, actual


def _extract_race_pairs(
    predictions: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Return (predicted_results, actual_results) for predictions that have both."""
    predicted: list[list[dict[str, Any]]] = []
    actual: list[list[dict[str, Any]]] = []

    for pred in predictions:
        results, actuals = _resolve_session_pair(pred, session_kind="race")

        if not results or not actuals:
            continue

        predicted.append(results)
        actual.append(actuals)

    return predicted, actual


def _build_naive_baseline(
    actual_grids: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Predict race N as the actual result of race N-1 (previous-race classifier).

    Returns a list aligned with actual_grids[1:] — the first race has no
    predecessor so the baseline cannot score it.
    """
    return [actual_grids[i] for i in range(len(actual_grids) - 1)]


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _build_calibration_section(
    predicted_grids: list[list[dict[str, Any]]],
    actual_grids: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate p5/p95 coverage across all races with saved band data."""
    all_bands: list[tuple[float, float]] = []
    all_actual_positions: list[int] = []
    races_with_bands = 0

    for pred_grid, act_grid in zip(predicted_grids, actual_grids, strict=True):
        bands = build_confidence_bands(pred_grid)
        if not bands:
            continue

        act_by_driver = {
            row["driver"]: row["position"]
            for row in act_grid
            if "driver" in row and "position" in row
        }

        # Align bands to actual positions in the same driver order as pred_grid
        aligned_bands: list[tuple[float, float]] = []
        aligned_actuals: list[int] = []
        pred_drivers_with_bands = [
            entry["driver"]
            for entry in pred_grid
            if entry.get("p5") is not None and entry.get("p95") is not None
        ]
        for driver, band in zip(pred_drivers_with_bands, bands, strict=True):
            actual_pos = act_by_driver.get(driver)
            if actual_pos is None:
                continue
            aligned_bands.append(band)
            aligned_actuals.append(actual_pos)

        if aligned_bands:
            races_with_bands += 1
            all_bands.extend(aligned_bands)
            all_actual_positions.extend(aligned_actuals)

    calibration = compute_calibration_metrics(all_bands, all_actual_positions)
    calibration["races_with_band_data"] = float(races_with_bands)
    calibration["total_races_evaluated"] = float(len(predicted_grids))
    return calibration


def _build_bias_section(
    predicted: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
    label: str,
) -> dict[str, Any]:
    """Run identify_systematic_errors and return the result with a label."""
    if len(predicted) < 2:
        return {
            "label": label,
            "races_compared": 0,
            "note": "Not enough races to detect systematic bias.",
        }
    result = identify_systematic_errors(predicted, actual)
    result["label"] = label
    return result


def _build_accuracy_section(
    predicted: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
    label: str,
) -> dict[str, Any]:
    """Aggregate ranking accuracy metrics across all aligned events."""
    per_event_metrics = [
        compute_prediction_accuracy(predicted_rows, actual_rows)
        for predicted_rows, actual_rows in zip(predicted, actual, strict=True)
        if predicted_rows and actual_rows
    ]
    if not per_event_metrics:
        return {
            "label": label,
            "events_evaluated": 0,
            "note": "No prediction/actual pairs available.",
        }

    def _mean(key: str) -> float:
        values = [float(row[key]) for row in per_event_metrics if key in row]
        if not values:
            return 0.0
        return sum(values) / len(values)

    return {
        "label": label,
        "events_evaluated": len(per_event_metrics),
        "mae": _mean("mae"),
        "exact_match_rate": _mean("exact_match_rate"),
        "within_3_rate": _mean("within_3_rate"),
        "spearman_rank": _mean("spearman_rank"),
        "kendall_tau": _mean("kendall_tau"),
    }


def _build_format_breakdown(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Summarize coverage by weekend format for reviewer-friendly segmentation."""
    breakdown: dict[str, dict[str, int]] = {}
    for prediction in predictions:
        metadata = prediction.get("metadata", {})
        format_name = str(metadata.get("weekend_format", "unknown") or "unknown").strip().lower()
        bucket = breakdown.setdefault(
            format_name,
            {
                "prediction_files": 0,
                "qualifying_pairs": 0,
                "race_pairs": 0,
            },
        )
        bucket["prediction_files"] += 1
        qualifying_rows, qualifying_actuals = _resolve_session_pair(
            prediction,
            session_kind="qualifying",
        )
        if qualifying_rows and qualifying_actuals:
            bucket["qualifying_pairs"] += 1
        race_rows, race_actuals = _resolve_session_pair(prediction, session_kind="race")
        if race_rows and race_actuals:
            bucket["race_pairs"] += 1
    return breakdown


def _bucket_name(value: object) -> str:
    """Normalize metadata values into stable analysis bucket labels."""
    label = str(value or "unknown").strip().lower()
    return label or "unknown"


def _load_track_type_map(year: int) -> dict[str, str]:
    """Load track types from the local track-characteristics snapshot."""
    candidate_years = [int(year)]
    if int(year) != 2026:
        candidate_years.append(2026)

    for candidate_year in candidate_years:
        path = _TRACK_CHARACTERISTICS_DIR / f"{candidate_year}_track_characteristics.json"
        if not path.exists():
            continue
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping track types from %s: %s", path, exc)
            continue

        tracks = payload.get("tracks", {})
        if not isinstance(tracks, dict):
            continue
        return {
            str(race_name): _bucket_name(track_payload.get("type"))
            for race_name, track_payload in tracks.items()
            if isinstance(track_payload, dict)
        }

    return {}


def _aggregate_metric_dicts(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    """Average event-level metrics into one segment summary."""
    if not metric_rows:
        return {"events": 0}

    keys = (
        "mae",
        "exact_match_rate",
        "within_3_rate",
        "spearman_rank",
        "kendall_tau",
    )
    aggregated = {"events": len(metric_rows)}
    for key in keys:
        values = [float(row[key]) for row in metric_rows if key in row]
        aggregated[key] = (sum(values) / len(values)) if values else 0.0
    return aggregated


def _extract_segment_value(
    metadata: dict[str, Any],
    *,
    track_types: dict[str, str],
    dimension: str,
) -> str:
    """Resolve one segment label from prediction metadata."""
    if dimension == "track_type":
        race_name = str(metadata.get("race_name", "")).strip()
        return _bucket_name(track_types.get(race_name, "unknown"))
    if dimension == "weekend_format":
        return _bucket_name(metadata.get("weekend_format"))
    if dimension == "weather":
        return _bucket_name(metadata.get("weather"))
    raise ValueError(f"Unsupported segment dimension: {dimension}")


def _build_segment_breakdown(
    selected_predictions: dict[str, list[dict[str, Any]]],
    *,
    year: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Build session-level metric slices by weekend format, weather, and track type."""
    track_types = _load_track_type_map(year)
    dimensions = ("weekend_format", "weather", "track_type")
    bucketed: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        "qualifying": {dimension: {} for dimension in dimensions},
        "race": {dimension: {} for dimension in dimensions},
    }

    for session_name, predictions in selected_predictions.items():
        for prediction in predictions:
            metadata = prediction.get("metadata", {})
            predicted_rows, actual_rows = _resolve_session_pair(
                prediction,
                session_kind=session_name,
            )
            if not predicted_rows or not actual_rows:
                continue
            metrics = compute_prediction_accuracy(predicted_rows, actual_rows)
            for dimension in dimensions:
                bucket_name = _extract_segment_value(
                    metadata,
                    track_types=track_types,
                    dimension=dimension,
                )
                bucket = bucketed[session_name][dimension].setdefault(bucket_name, [])
                bucket.append(metrics)

    aggregated: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for session_name, dimension_payload in bucketed.items():
        aggregated[session_name] = {}
        for dimension, buckets in dimension_payload.items():
            aggregated[session_name][dimension] = {
                bucket_name: _aggregate_metric_dicts(metric_rows)
                for bucket_name, metric_rows in sorted(buckets.items())
            }
    return aggregated


def _align_rows_by_driver(
    predicted_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align predicted and actual rows by driver for event-level miss summaries."""
    actual_by_driver = {
        str(row.get("driver", "")).strip(): row
        for row in actual_rows
        if str(row.get("driver", "")).strip()
    }
    aligned: list[dict[str, Any]] = []
    for predicted_row in predicted_rows:
        driver = str(predicted_row.get("driver", "")).strip()
        if not driver or driver not in actual_by_driver:
            continue
        actual_row = actual_by_driver[driver]
        try:
            predicted_position = int(predicted_row.get("position"))
            actual_position = int(actual_row.get("position"))
        except (TypeError, ValueError):
            continue
        signed_error = float(predicted_position - actual_position)
        aligned.append(
            {
                "driver": driver,
                "team": str(predicted_row.get("team") or actual_row.get("team") or "").strip(),
                "predicted_position": predicted_position,
                "actual_position": actual_position,
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
            }
        )
    return aligned


def _describe_event_errors(
    predicted_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract high-signal miss details for one prediction event."""
    aligned = _align_rows_by_driver(predicted_rows, actual_rows)
    top_misses = sorted(
        aligned,
        key=lambda row: (float(row["absolute_error"]), row["driver"]),
        reverse=True,
    )[:3]

    predicted_winner = str(predicted_rows[0].get("driver", "")).strip() if predicted_rows else ""
    actual_winner = str(actual_rows[0].get("driver", "")).strip() if actual_rows else ""
    return {
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
        "winner_correct": bool(predicted_winner and predicted_winner == actual_winner),
        "top_misses": top_misses,
    }


def _collect_error_events(
    predictions: list[dict[str, Any]],
    *,
    year: int,
    session_name: str,
) -> list[dict[str, Any]]:
    """Build event-level error records for one session type."""
    track_types = _load_track_type_map(year)
    events: list[dict[str, Any]] = []

    for prediction in predictions:
        metadata = prediction.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        predicted_rows, actual_rows = _resolve_session_pair(
            prediction,
            session_kind=session_name,
        )
        if not predicted_rows or not actual_rows:
            continue

        metrics = compute_prediction_accuracy(predicted_rows, actual_rows)
        event_errors = _describe_event_errors(predicted_rows, actual_rows)
        events.append(
            {
                "race_name": race_name,
                "weekend_format": _bucket_name(metadata.get("weekend_format")),
                "weather": _bucket_name(metadata.get("weather")),
                "track_type": _bucket_name(track_types.get(race_name, "unknown")),
                "mae": metrics.get("mae"),
                "exact_match_rate": metrics.get("exact_match_rate"),
                "within_3_rate": metrics.get("within_3_rate"),
                **event_errors,
            }
        )

    return events


def _summarize_top_miss_drivers(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count drivers that appear repeatedly among the biggest event misses."""
    counts: dict[str, dict[str, float]] = {}
    for event in events:
        for miss in event.get("top_misses", []):
            driver = str(miss.get("driver", "")).strip()
            if not driver:
                continue
            bucket = counts.setdefault(driver, {"appearances": 0.0, "avg_abs_error": 0.0})
            bucket["appearances"] += 1.0
            bucket["avg_abs_error"] += float(miss.get("absolute_error", 0.0))

    ranked = []
    for driver, payload in counts.items():
        appearances = payload["appearances"]
        ranked.append(
            {
                "driver": driver,
                "appearances": appearances,
                "avg_abs_error": payload["avg_abs_error"] / appearances if appearances else 0.0,
            }
        )

    return sorted(
        ranked,
        key=lambda row: (-float(row["appearances"]), -float(row["avg_abs_error"]), row["driver"]),
    )[:5]


def _build_error_analysis(
    selected_predictions: dict[str, list[dict[str, Any]]],
    *,
    year: int,
) -> dict[str, Any]:
    """Summarize the worst events and recurring misses for qualifying and race sessions."""
    analysis: dict[str, Any] = {}
    for session_name, session_predictions in selected_predictions.items():
        events = _collect_error_events(session_predictions, year=year, session_name=session_name)
        worst_events = sorted(
            events,
            key=lambda row: float(row.get("mae") or 0.0),
            reverse=True,
        )[:5]
        winner_misses = [
            row for row in events if row.get("winner_correct") is False and row.get("actual_winner")
        ]
        analysis[session_name] = {
            "events_evaluated": len(events),
            "worst_events": worst_events,
            "winner_miss_events": winner_misses[:5],
            "frequent_top_miss_drivers": _summarize_top_miss_drivers(events),
        }
    return analysis


def _build_baseline_comparison_section(
    predicted: list[list[dict[str, Any]]],
    actual: list[list[dict[str, Any]]],
    label: str,
) -> dict[str, Any]:
    """Compare model vs naive previous-race baseline."""
    if len(actual) < 2:
        return {
            "label": label,
            "note": "Not enough races for baseline comparison (need ≥ 2).",
        }

    naive_baseline = _build_naive_baseline(actual)
    # Trim model predictions to match — skip the first race (no predecessor)
    model_trimmed = predicted[1:]
    actual_trimmed = actual[1:]

    result = compute_improvement_over_baseline(model_trimmed, actual_trimmed, naive_baseline)
    result["label"] = label
    return result


def build_report(year: int, predictions_dir: Path) -> dict[str, Any]:
    """Load predictions for the year and run all three evaluation analyses."""
    predictions = _load_predictions_from_files(year, predictions_dir)
    logger.info("Loaded %d prediction files for %d", len(predictions), year)

    selected_predictions = {
        "qualifying": _select_latest_predictions(predictions, session_kind="qualifying"),
        "race": _select_latest_predictions(predictions, session_kind="race"),
    }
    pred_quali, act_quali = _extract_qualifying_pairs(selected_predictions["qualifying"])
    pred_race, act_race = _extract_race_pairs(selected_predictions["race"])

    logger.info(
        "Qualifying pairs with actuals: %d, Race pairs with actuals: %d "
        "(from %d/%d selected checkpoints)",
        len(pred_quali),
        len(pred_race),
        len(selected_predictions["qualifying"]),
        len(selected_predictions["race"]),
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "year": year,
        "predictions_analyzed": len(predictions),
        "evaluation_scope": {
            "selection_policy": "latest_checkpoint_per_race_and_target",
            "ignored_intermediate_checkpoints": {
                "qualifying": len(predictions) - len(selected_predictions["qualifying"]),
                "race": len(predictions) - len(selected_predictions["race"]),
            },
            "selected_prediction_counts": {
                "qualifying": len(selected_predictions["qualifying"]),
                "race": len(selected_predictions["race"]),
            },
            "selected_checkpoint_breakdown": {
                "qualifying": _checkpoint_breakdown(selected_predictions["qualifying"]),
                "race": _checkpoint_breakdown(selected_predictions["race"]),
            },
            "selected_target_breakdown": {
                "qualifying": _target_breakdown(
                    selected_predictions["qualifying"],
                    session_kind="qualifying",
                ),
                "race": _target_breakdown(
                    selected_predictions["race"],
                    session_kind="race",
                ),
            },
        },
        "qualifying_pairs": len(pred_quali),
        "race_pairs": len(pred_race),
        "format_breakdown": _build_format_breakdown(predictions),
        "segment_breakdown": _build_segment_breakdown(selected_predictions, year=year),
        "qualifying_accuracy": _build_accuracy_section(pred_quali, act_quali, "qualifying"),
        "race_accuracy": _build_accuracy_section(pred_race, act_race, "race"),
        "calibration": _build_calibration_section(pred_quali, act_quali),
        "error_analysis": _build_error_analysis(selected_predictions, year=year),
        "qualifying_bias": _build_bias_section(pred_quali, act_quali, "qualifying"),
        "race_bias": _build_bias_section(pred_race, act_race, "race"),
        "qualifying_vs_baseline": _build_baseline_comparison_section(
            pred_quali, act_quali, "qualifying"
        ),
        "race_vs_baseline": _build_baseline_comparison_section(pred_race, act_race, "race"),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    """Format a 0-1 fraction as a percentage string."""
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _flt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _render_segment_table(
    title: str,
    buckets: dict[str, dict[str, float]],
) -> list[str]:
    """Render one segmented metric table."""
    if not buckets:
        return [f"#### {title}", "", "*No events available.*", ""]

    lines = [
        f"#### {title}",
        "",
        "| Bucket | Events | MAE | Exact match | Within-3 | Spearman ρ |",
        "|---|---|---|---|---|---|",
    ]
    for bucket_name, metrics in sorted(buckets.items()):
        lines.append(
            f"| {bucket_name} | {int(metrics.get('events', 0))} | "
            f"{_flt(metrics.get('mae'))} | {_flt(metrics.get('exact_match_rate'))}% | "
            f"{_flt(metrics.get('within_3_rate'))}% | {_flt(metrics.get('spearman_rank'))} |"
        )
    lines.append("")
    return lines


def _render_error_session(
    session_name: str,
    payload: dict[str, Any],
) -> list[str]:
    """Render one session's error-analysis block."""
    lines = [f"### {session_name.title()}", ""]
    events_evaluated = int(payload.get("events_evaluated", 0))
    if events_evaluated == 0:
        lines.extend(["*No events with actuals available.*", ""])
        return lines

    lines.append(f"Evaluated **{events_evaluated}** event(s).")
    lines.append("")

    worst_events = payload.get("worst_events", [])
    if worst_events:
        lines.append("Worst weekends:")
        for row in worst_events[:3]:
            lines.append(
                "- "
                f"{row.get('race_name')} "
                f"(`{row.get('track_type')}`, `{row.get('weekend_format')}`, `{row.get('weather')}`) "
                f"MAE={_flt(row.get('mae'))}, winner="
                f"{row.get('predicted_winner')} -> {row.get('actual_winner')}"
            )
        lines.append("")

    frequent_misses = payload.get("frequent_top_miss_drivers", [])
    if frequent_misses:
        lines.append("Drivers that show up repeatedly among the largest misses:")
        for row in frequent_misses[:3]:
            lines.append(
                "- "
                f"{row.get('driver')}: appearances={int(row.get('appearances', 0))}, "
                f"avg_abs_error={_flt(row.get('avg_abs_error'))}"
            )
        lines.append("")

    return lines


def render_error_analysis_markdown(report: dict[str, Any]) -> str:
    """Render a shorter standalone error-analysis document from the main report."""
    error_analysis = report.get("error_analysis", {})
    lines = [
        f"# Model Error Analysis — {report.get('year')}",
        "",
        f"*Generated: {report.get('generated_at', 'unknown')}*",
        "",
        "This companion note focuses on the failures the model needs to explain,",
        "not the averages it would prefer to show.",
        "",
    ]
    for session_name in ("qualifying", "race"):
        lines.extend(_render_error_session(session_name, error_analysis.get(session_name, {})))
    lines.extend(
        [
            "## Context",
            "",
            "- Pair this with [MODEL_CALIBRATION.md](./MODEL_CALIBRATION.md) for calibration and baseline metrics.",
            "- Pair this with [LIMITATIONS.md](../LIMITATIONS.md) for known structural gaps.",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the evaluation report as a markdown document."""
    year = report["year"]
    generated = report.get("generated_at", "unknown")
    n_predictions = report["predictions_analyzed"]
    n_q_pairs = report["qualifying_pairs"]
    n_r_pairs = report["race_pairs"]

    cal = report.get("calibration", {})
    empirical = cal.get("empirical_coverage")
    nominal = cal.get("nominal_coverage", 0.9)
    cal_error = cal.get("calibration_error")
    interval_width = cal.get("mean_interval_width")
    races_with_bands = int(cal.get("races_with_band_data", 0))
    interval_count = int(cal.get("interval_count", 0))

    q_bias = report.get("qualifying_bias", {})
    r_bias = report.get("race_bias", {})
    q_baseline = report.get("qualifying_vs_baseline", {})
    r_baseline = report.get("race_vs_baseline", {})
    q_accuracy = report.get("qualifying_accuracy", {})
    r_accuracy = report.get("race_accuracy", {})
    format_breakdown = report.get("format_breakdown", {})
    segment_breakdown = report.get("segment_breakdown", {})
    error_analysis = report.get("error_analysis", {})
    evaluation_scope = report.get("evaluation_scope", {})
    selected_counts = evaluation_scope.get("selected_prediction_counts", {})
    selected_checkpoints = evaluation_scope.get("selected_checkpoint_breakdown", {})
    selected_targets = evaluation_scope.get("selected_target_breakdown", {})
    ignored_counts = evaluation_scope.get("ignored_intermediate_checkpoints", {})

    lines: list[str] = [
        f"# Model Calibration Report — {year} Season",
        "",
        f"*Generated: {generated}*",
        "",
        "This report measures three things: whether the Monte Carlo uncertainty",
        "bands are empirically honest, whether the model has systematic directional",
        "bias for specific drivers or teams, and whether it beats a naive baseline.",
        "",
        "It is generated from saved prediction artifacts by",
        "`scripts/generate_evaluation_report.py`. Re-run after each race to keep",
        "it current.",
        "",
        "---",
        "",
        "## Coverage",
        "",
        f"- Prediction files analyzed: **{n_predictions}**",
        f"- Latest qualifying checkpoints selected: **{selected_counts.get('qualifying', 0)}**",
        f"- Latest race checkpoints selected: **{selected_counts.get('race', 0)}**",
        f"- Qualifying races with actuals: **{n_q_pairs}**",
        f"- Race results with actuals: **{n_r_pairs}**",
        f"- Intermediate qualifying checkpoints ignored in canonical evaluation: **{ignored_counts.get('qualifying', 0)}**",
        f"- Intermediate race checkpoints ignored in canonical evaluation: **{ignored_counts.get('race', 0)}**",
        "",
        "---",
        "",
        "## 0. Accuracy Overview",
        "",
        "| Session | Events | MAE | Exact match | Within-3 | Spearman ρ | Kendall τ |",
        "|---|---|---|---|---|---|---|",
        (
            f"| Qualifying | {int(q_accuracy.get('events_evaluated', 0))} | "
            f"{_flt(q_accuracy.get('mae'))} | {_flt(q_accuracy.get('exact_match_rate'))}% | "
            f"{_flt(q_accuracy.get('within_3_rate'))}% | {_flt(q_accuracy.get('spearman_rank'))} | "
            f"{_flt(q_accuracy.get('kendall_tau'))} |"
        ),
        (
            f"| Race | {int(r_accuracy.get('events_evaluated', 0))} | "
            f"{_flt(r_accuracy.get('mae'))} | {_flt(r_accuracy.get('exact_match_rate'))}% | "
            f"{_flt(r_accuracy.get('within_3_rate'))}% | {_flt(r_accuracy.get('spearman_rank'))} | "
            f"{_flt(r_accuracy.get('kendall_tau'))} |"
        ),
        "",
    ]

    if format_breakdown:
        lines.extend(
            [
                "### Weekend Format Coverage",
                "",
                "| Format | Prediction files | Qualifying pairs | Race pairs |",
                "|---|---|---|---|",
            ]
        )
        for format_name, counts in sorted(format_breakdown.items()):
            lines.append(
                f"| {format_name} | {counts.get('prediction_files', 0)} | "
                f"{counts.get('qualifying_pairs', 0)} | {counts.get('race_pairs', 0)} |"
            )
        lines.extend(["", "---", ""])

    if selected_checkpoints or selected_targets:
        lines.extend(["## Selection Policy", ""])
        selection_policy = evaluation_scope.get("selection_policy", "unknown")
        lines.append(
            "Canonical evaluation uses "
            f"`{selection_policy}` so each race/target contributes at most one scored forecast."
        )
        lines.append("")
        if selected_checkpoints:
            lines.extend(
                [
                    "### Selected Checkpoints",
                    "",
                    "| Session | Checkpoint | Count |",
                    "|---|---|---|",
                ]
            )
            for session_name, counts in selected_checkpoints.items():
                for checkpoint, count in sorted(counts.items()):
                    lines.append(f"| {session_name} | {checkpoint} | {count} |")
            lines.append("")
        if selected_targets:
            lines.extend(
                [
                    "### Selected Targets",
                    "",
                    "| Session | Target | Count |",
                    "|---|---|---|",
                ]
            )
            for session_name, counts in selected_targets.items():
                for target_name, count in sorted(counts.items()):
                    lines.append(f"| {session_name} | {target_name} | {count} |")
            lines.extend(["", "---", ""])

    lines.extend(["## 1. Segmented Performance", ""])
    for session_name in ("qualifying", "race"):
        lines.append(f"### {session_name.title()}")
        lines.append("")
        session_breakdown = segment_breakdown.get(session_name, {})
        lines.extend(
            _render_segment_table(
                "Weekend Format",
                session_breakdown.get("weekend_format", {}),
            )
        )
        lines.extend(
            _render_segment_table(
                "Weather",
                session_breakdown.get("weather", {}),
            )
        )
        lines.extend(
            _render_segment_table(
                "Track Type",
                session_breakdown.get("track_type", {}),
            )
        )
    lines.extend(["---", ""])

    lines.extend(
        [
            "## 2. Confidence Interval Calibration (Qualifying)",
            "",
            "The Monte Carlo simulation produces a p5–p95 position interval for each",
            "driver. A well-calibrated model should have ~90% of actual outcomes fall",
            "inside that interval.",
            "",
        ]
    )

    if races_with_bands == 0:
        lines += [
            "> **No band data available yet.** Predictions saved before p5/p95 were",
            "> persisted do not carry interval data. Calibration will populate as",
            "> new predictions are generated.",
            "",
        ]
    else:
        calibration_verdict = ""
        if cal_error is not None:
            if abs(cal_error) <= 0.03:
                calibration_verdict = "✅ Well-calibrated (within 3% of nominal)."
            elif cal_error < 0:
                calibration_verdict = (
                    f"⚠️ Intervals are too **tight** — model is overconfident "
                    f"by {abs(cal_error) * 100:.1f}pp."
                )
            else:
                calibration_verdict = (
                    f"⚠️ Intervals are too **wide** — model is underconfident "
                    f"by {cal_error * 100:.1f}pp."
                )

        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Races with interval data | {races_with_bands} |",
            f"| Driver-race predictions covered | {interval_count} |",
            f"| Nominal coverage (target) | {_pct(nominal)} |",
            f"| Empirical coverage (actual) | {_pct(empirical)} |",
            f"| Calibration error | {_flt(cal_error)} ({'+' if (cal_error or 0) > 0 else ''}{_pct(cal_error) if cal_error is not None else 'n/a'}) |",
            f"| Mean interval width | {_flt(interval_width)} positions |",
            "",
            calibration_verdict,
            "",
            "**Interpretation:** A negative calibration error means intervals are",
            "too tight — the model is more certain than it should be. A positive",
            "error means intervals are too wide.",
            "",
        ]

    # --- Systematic bias ---
    lines += [
        "---",
        "",
        "## 3. Error Analysis",
        "",
    ]

    for session_name in ("qualifying", "race"):
        lines.extend(_render_error_session(session_name, error_analysis.get(session_name, {})))

    lines += [
        "---",
        "",
        "## 4. Systematic Bias",
        "",
        "Signed error = predicted position − actual position.",
        "Negative = model predicted *better* than reality (overestimated the driver).",
        "Positive = model predicted *worse* than reality (underestimated the driver).",
        "",
        "### Qualifying",
        "",
    ]

    q_races = q_bias.get("races_compared", 0)
    if q_races < 2:
        lines.append("*Not enough races to detect bias yet.*\n")
    else:
        lines.append(f"Based on {q_races} races.\n")
        for label, key in [
            ("Most overestimated teams", "most_overestimated_teams"),
            ("Most underestimated teams", "most_underestimated_teams"),
            ("Most overestimated drivers", "most_overestimated_drivers"),
            ("Most underestimated drivers", "most_underestimated_drivers"),
        ]:
            rows = q_bias.get(key, [])
            if rows:
                lines.append(f"**{label}:**")
                for row in rows[:3]:
                    entity = row.get("entity", "?")
                    mse = row.get("mean_signed_error")
                    mae = row.get("mean_abs_error")
                    n = int(row.get("samples", 0))
                    lines.append(
                        f"- {entity}: mean signed error {_flt(mse)} (MAE {_flt(mae)}, n={n})"
                    )
                lines.append("")

    lines += ["### Race", ""]
    r_races = r_bias.get("races_compared", 0)
    if r_races < 2:
        lines.append("*Not enough races to detect bias yet.*\n")
    else:
        lines.append(f"Based on {r_races} races.\n")
        for label, key in [
            ("Most overestimated drivers", "most_overestimated_drivers"),
            ("Most underestimated drivers", "most_underestimated_drivers"),
        ]:
            rows = r_bias.get(key, [])
            if rows:
                lines.append(f"**{label}:**")
                for row in rows[:3]:
                    entity = row.get("entity", "?")
                    mse = row.get("mean_signed_error")
                    mae = row.get("mean_abs_error")
                    n = int(row.get("samples", 0))
                    lines.append(
                        f"- {entity}: mean signed error {_flt(mse)} (MAE {_flt(mae)}, n={n})"
                    )
                lines.append("")

    # --- Baseline comparison ---
    lines += [
        "---",
        "",
        "## 5. Baseline Comparison",
        "",
        "Naive baseline: predict race N using the actual results of race N-1",
        "(previous-race classification). This is a realistic lower bar — it",
        "requires no modelling, just memory of last week.",
        "",
        "### Qualifying",
        "",
    ]

    def _baseline_table(section: dict[str, Any]) -> list[str]:
        note = section.get("note")
        if note:
            return [f"*{note}*", ""]
        events = section.get("events_compared", 0)
        model_m = section.get("model_metrics", {})
        base_m = section.get("baseline_metrics", {})
        imp = section.get("improvement", {})
        beats = section.get("model_beats_baseline_on_mae", False)
        verdict = (
            "✅ Model beats naive baseline on MAE"
            if beats
            else "❌ Model does not beat naive baseline on MAE"
        )
        return [
            f"Based on {events} races.",
            "",
            "| Metric | Model | Naive baseline | Δ |",
            "|---|---|---|---|",
            f"| MAE | {_flt(model_m.get('mae'))} | {_flt(base_m.get('mae'))} | {_flt(imp.get('mae_improvement'))} |",
            f"| Within-3 rate | {_pct(model_m.get('within_3_rate', 0) / 100 if model_m.get('within_3_rate') else None)} | {_pct(base_m.get('within_3_rate', 0) / 100 if base_m.get('within_3_rate') else None)} | — |",
            f"| Spearman ρ | {_flt(model_m.get('spearman_rank'))} | {_flt(base_m.get('spearman_rank'))} | {_flt(imp.get('spearman_rank_delta'))} |",
            f"| Kendall τ | {_flt(model_m.get('kendall_tau'))} | {_flt(base_m.get('kendall_tau'))} | {_flt(imp.get('kendall_tau_delta'))} |",
            "",
            verdict,
            "",
        ]

    lines += _baseline_table(q_baseline)
    lines += ["### Race", ""]
    lines += _baseline_table(r_baseline)

    lines += [
        "---",
        "",
        "## Notes",
        "",
        "- Calibration data populates only for predictions generated after",
        "  p5/p95 interval persistence was added. Older artifacts carry no band data.",
        "- Segment breakdowns slice the same event-level metrics by weekend format, weather,",
        "  and track type using saved metadata and local track characteristics.",
        "- Error analysis highlights the worst weekends and repeat offenders, not just mean scores.",
        "- Systematic bias analysis requires ≥ 2 races with saved actuals.",
        "- Baseline comparison requires ≥ 2 races (first race has no predecessor).",
        "- For known model limitations see [LIMITATIONS.md](../LIMITATIONS.md).",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate calibration and bias evaluation report from saved predictions."
    )
    parser.add_argument("--year", type=int, default=2026, help="Season year to evaluate")
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="data/predictions",
        help="Directory containing saved prediction files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/MODEL_CALIBRATION.md",
        help="Path for the rendered markdown report",
    )
    parser.add_argument(
        "--error-out",
        type=str,
        default="docs/MODEL_ERROR_ANALYSIS.md",
        help="Path for the standalone error-analysis markdown",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path for the raw JSON report (default: data/evaluation/<year>_evaluation_report.json)",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        logger.error("Predictions directory not found: %s", predictions_dir)
        return 1

    report = build_report(args.year, predictions_dir)

    json_out = Path(args.json_out or f"data/evaluation/{args.year}_evaluation_report.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote JSON report: %s", json_out)

    md = render_markdown(report)
    md_out = Path(args.out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md)
    logger.info("Wrote markdown report: %s", md_out)

    error_md = render_error_analysis_markdown(report)
    error_out = Path(args.error_out)
    error_out.parent.mkdir(parents=True, exist_ok=True)
    error_out.write_text(error_md)
    logger.info("Wrote error analysis report: %s", error_out)

    # Exit summary
    cal = report.get("calibration", {})
    n_q = report["qualifying_pairs"]
    n_r = report["race_pairs"]
    if n_q == 0 and n_r == 0:
        logger.warning(
            "No races with actuals found — report generated but contains no results. "
            "Run scripts/update_from_race.py first to reconcile actuals."
        )
    else:
        logger.info(
            "Report complete: %d qualifying pairs, %d race pairs, calibration bands from %d races",
            n_q,
            n_r,
            int(cal.get("races_with_band_data", 0)),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
