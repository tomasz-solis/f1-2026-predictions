"""Model evaluation and diagnostic utilities.

These helpers sit above the per-race metric calculators in ``src.utils``.
The goal is to answer the questions a reviewer will ask after the model runs:
how accurate was it, were the uncertainty bands honest, and does it miss the
same teams or drivers in the same direction week after week.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


def build_confidence_bands(
    predicted_grid: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Extract (p5, p95) position intervals from a saved qualifying grid.

    Each entry in ``predicted_grid`` may carry ``p5`` and ``p95`` keys
    populated when the prediction was saved. Entries missing these keys
    (older artifacts saved before the field was added) are skipped, so
    the returned list may be shorter than the input.

    Returns a list of (lower, upper) tuples in the same driver order as
    ``predicted_grid``, suitable for passing to ``compute_calibration_metrics``.
    """
    bands: list[tuple[float, float]] = []
    for entry in predicted_grid:
        p5 = entry.get("p5")
        p95 = entry.get("p95")
        if p5 is None or p95 is None:
            continue
        bands.append((float(p5), float(p95)))
    return bands


_DNF_STATUS_MARKERS = ("dnf", "retired", "did not finish", "disqualified", "dns", "dnq", "dsq")


def row_is_dnf(row: dict[str, Any]) -> bool:
    """Return whether one result row represents a did-not-finish outcome.

    Tolerant of the several shapes actuals can take: an explicit ``dnf`` flag, a
    FastF1-style ``status`` string, or a ``classified``/``finished`` boolean.
    Returns ``False`` when no DNF signal is present, so position-only actuals
    (older artifacts) are simply treated as all-finished.
    """
    if not isinstance(row, dict):
        return False
    if "dnf" in row:
        return bool(row.get("dnf"))
    if "classified" in row and row.get("classified") is not None:
        return not bool(row.get("classified"))
    if "finished" in row and row.get("finished") is not None:
        return not bool(row.get("finished"))
    status = str(row.get("status", "")).strip().lower()
    if status:
        return any(marker in status for marker in _DNF_STATUS_MARKERS)
    return False


def position_weight(
    position: float,
    *,
    field_size: int,
    scheme: str = "reciprocal",
    tau: float = 5.0,
) -> float:
    """Return a top-heavy importance weight for a finishing position.

    Lower (better) positions carry more weight, so an error at the sharp end of
    the grid counts for more than an error among the backmarkers. Schemes:

    - ``reciprocal`` (default): ``1 / position`` — P1 matters most, decaying fast.
    - ``linear``: ``(field_size - position + 1) / field_size`` — gentle gradient.
    - ``exponential``: ``exp(-(position - 1) / tau)`` — tunable decay length.
    """
    pos = max(1.0, float(position))
    normalized_scheme = str(scheme).strip().lower()
    if normalized_scheme == "linear":
        size = max(1, int(field_size))
        return max(0.0, (size - pos + 1.0) / size)
    if normalized_scheme == "exponential":
        return float(np.exp(-(pos - 1.0) / max(1e-6, float(tau))))
    return 1.0 / pos


def _coerce_ranked_rows(rows: Sequence[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ranking inputs into ``driver/team/position`` rows.

    Optional ``dnf``/``status``/``classified`` and ``dnf_probability`` fields are
    preserved when present so finisher-only and DNF-calibration metrics can use
    them without changing the ranking behaviour for callers that ignore them.
    """
    normalized: list[dict[str, Any]] = []
    for default_position, row in enumerate(rows, start=1):
        if isinstance(row, str):
            driver = row.strip()
            if not driver:
                continue
            normalized.append(
                {
                    "driver": driver,
                    "team": "",
                    "position": default_position,
                    "dnf": False,
                    "dnf_probability": None,
                }
            )
            continue

        if not isinstance(row, dict):
            continue

        driver = str(row.get("driver", "")).strip()
        if not driver:
            continue

        raw_position = row.get("position", default_position)
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            position = default_position

        # Accept both the live "dnf_probability" key and the persisted "dnf_risk" alias.
        raw_dnf_probability = row.get("dnf_probability")
        if raw_dnf_probability is None:
            raw_dnf_probability = row.get("dnf_risk")
        try:
            dnf_probability = (
                float(raw_dnf_probability) if raw_dnf_probability is not None else None
            )
        except (TypeError, ValueError):
            dnf_probability = None
        # Some artifacts persist DNF risk as a percentage (0-100); normalise to a 0-1 fraction.
        if dnf_probability is not None and dnf_probability > 1.0:
            dnf_probability = dnf_probability / 100.0

        normalized.append(
            {
                "driver": driver,
                "team": str(row.get("team", "")).strip(),
                "position": position,
                "dnf": row_is_dnf(row),
                "dnf_probability": dnf_probability,
            }
        )

    return normalized


def _aligned_positions(
    predicted: Sequence[str | dict[str, Any]],
    actual: Sequence[str | dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return driver-aligned predicted and actual rows."""
    predicted_rows = _coerce_ranked_rows(predicted)
    actual_rows = _coerce_ranked_rows(actual)
    actual_by_driver = {row["driver"]: row for row in actual_rows}
    return [
        (predicted_row, actual_by_driver[predicted_row["driver"]])
        for predicted_row in predicted_rows
        if predicted_row["driver"] in actual_by_driver
    ]


def compute_prediction_accuracy(
    predicted: Sequence[str | dict[str, Any]],
    actual: Sequence[str | dict[str, Any]],
    *,
    weight_scheme: str = "reciprocal",
    weight_tau: float = 5.0,
) -> dict[str, float]:
    """Compute ranking accuracy for one predicted order.

    This works with plain driver lists or with result rows that already include
    positions and team names. In addition to plain MAE it reports:

    - ``weighted_mae``: position-importance weighted MAE (sharp-end errors count
      more), weighted by the better of the predicted/actual position so both
      "missed a real top finisher" and "wrongly hyped a backmarker" are penalised.
    - ``finisher_mae``: MAE over drivers who actually finished, so a handful of
      unpredictable retirements do not dominate the position score. Equals
      ``mae`` when the actuals carry no DNF information.
    """
    aligned = _aligned_positions(predicted, actual)
    field_size = max(1, len(actual))
    if not aligned:
        return {
            "field_size": float(len(actual)),
            "compared_drivers": 0.0,
            "mae": float("inf"),
            "weighted_mae": float("inf"),
            "finisher_mae": float("inf"),
            "finisher_count": 0.0,
            "dnf_count": 0.0,
            "exact_match_rate": 0.0,
            "within_3_rate": 0.0,
            "spearman_rank": 0.0,
            "kendall_tau": 0.0,
        }

    predicted_positions = np.array([row["position"] for row, _ in aligned], dtype=float)
    actual_positions = np.array([row["position"] for _, row in aligned], dtype=float)
    abs_errors = np.abs(predicted_positions - actual_positions)

    weights = np.array(
        [
            position_weight(
                min(float(predicted_row["position"]), float(actual_row["position"])),
                field_size=field_size,
                scheme=weight_scheme,
                tau=weight_tau,
            )
            for predicted_row, actual_row in aligned
        ],
        dtype=float,
    )
    weight_total = float(np.sum(weights))
    weighted_mae = (
        float(np.sum(weights * abs_errors) / weight_total)
        if weight_total > 0
        else float(np.mean(abs_errors))
    )

    actual_dnf_mask = np.array([bool(actual_row["dnf"]) for _, actual_row in aligned], dtype=bool)
    finisher_errors = abs_errors[~actual_dnf_mask]
    finisher_mae = (
        float(np.mean(finisher_errors)) if finisher_errors.size > 0 else float(np.mean(abs_errors))
    )

    spearman_value = 0.0
    kendall_value = 0.0
    if len(aligned) >= 2:
        spearman_raw, _ = spearmanr(predicted_positions, actual_positions)
        kendall_raw, _ = kendalltau(predicted_positions, actual_positions)
        if np.isfinite(spearman_raw):
            spearman_value = float(spearman_raw)
        if np.isfinite(kendall_raw):
            kendall_value = float(kendall_raw)

    return {
        "field_size": float(len(actual)),
        "compared_drivers": float(len(aligned)),
        "mae": float(np.mean(abs_errors)),
        "weighted_mae": weighted_mae,
        "finisher_mae": finisher_mae,
        "finisher_count": float(int(np.sum(~actual_dnf_mask))),
        "dnf_count": float(int(np.sum(actual_dnf_mask))),
        "exact_match_rate": float(np.mean(abs_errors == 0.0) * 100.0),
        "within_3_rate": float(np.mean(abs_errors <= 3.0) * 100.0),
        "spearman_rank": spearman_value,
        "kendall_tau": kendall_value,
    }


def compute_dnf_calibration(
    predicted: Sequence[str | dict[str, Any]],
    actual: Sequence[str | dict[str, Any]],
) -> dict[str, float]:
    """Score predicted DNF probabilities against actual retirements (Brier score).

    DNF prediction is a calibration problem, not a positioning one. Scoring it
    separately keeps unpredictable retirements from polluting position MAE while
    still tracking whether the per-driver ``dnf_probability`` is honest.

    ``brier_score`` is the mean squared error between predicted probability and
    the realised 0/1 outcome (lower is better). ``baseline_brier`` is the score
    of always predicting the field base rate, so ``brier_skill_score`` > 0 means
    the model beats that naive baseline.
    """
    predicted_rows = _coerce_ranked_rows(predicted)
    actual_rows = _coerce_ranked_rows(actual)
    actual_by_driver = {row["driver"]: row for row in actual_rows}

    probabilities: list[float] = []
    outcomes: list[float] = []
    for predicted_row in predicted_rows:
        actual_row = actual_by_driver.get(predicted_row["driver"])
        if actual_row is None:
            continue
        probability = predicted_row.get("dnf_probability")
        if probability is None:
            continue
        probabilities.append(float(np.clip(float(probability), 0.0, 1.0)))
        outcomes.append(1.0 if bool(actual_row["dnf"]) else 0.0)

    if not probabilities:
        return {
            "scored_drivers": 0.0,
            "actual_dnf_count": 0.0,
            "actual_dnf_rate": 0.0,
            "brier_score": float("nan"),
            "baseline_brier": float("nan"),
            "brier_skill_score": float("nan"),
        }

    probability_array = np.array(probabilities, dtype=float)
    outcome_array = np.array(outcomes, dtype=float)
    base_rate = float(np.mean(outcome_array))
    brier_score = float(np.mean((probability_array - outcome_array) ** 2))
    baseline_brier = float(np.mean((base_rate - outcome_array) ** 2))
    brier_skill = (
        float(1.0 - (brier_score / baseline_brier)) if baseline_brier > 0 else float("nan")
    )

    return {
        "scored_drivers": float(len(probabilities)),
        "actual_dnf_count": float(int(np.sum(outcome_array))),
        "actual_dnf_rate": base_rate,
        "brier_score": brier_score,
        "baseline_brier": baseline_brier,
        "brier_skill_score": brier_skill,
    }


def compute_calibration_metrics(
    confidence_bands: Sequence[tuple[float, float]],
    actual_positions: Sequence[int],
) -> dict[str, float]:
    """Measure whether Monte Carlo finish intervals are honest.

    The intended use is a 90% interval such as ``p5``-``p95``. In that case a
    well-calibrated model should land near 90% empirical coverage over time.
    """
    n = min(len(confidence_bands), len(actual_positions))
    if n == 0:
        return {
            "interval_count": 0.0,
            "empirical_coverage": 0.0,
            "nominal_coverage": 0.90,
            "calibration_error": -0.90,
            "mean_interval_width": 0.0,
            "average_miss_distance": 0.0,
        }

    hits = 0
    miss_distances: list[float] = []
    widths: list[float] = []
    for index in range(n):
        lower, upper = confidence_bands[index]
        actual = float(actual_positions[index])
        lower_bound = min(float(lower), float(upper))
        upper_bound = max(float(lower), float(upper))
        widths.append(max(0.0, upper_bound - lower_bound))
        if lower_bound <= actual <= upper_bound:
            hits += 1
            continue
        if actual < lower_bound:
            miss_distances.append(lower_bound - actual)
        else:
            miss_distances.append(actual - upper_bound)

    empirical_coverage = hits / n
    nominal_coverage = 0.90
    return {
        "interval_count": float(n),
        "empirical_coverage": float(empirical_coverage),
        "nominal_coverage": nominal_coverage,
        "calibration_error": float(empirical_coverage - nominal_coverage),
        "mean_interval_width": float(np.mean(widths)) if widths else 0.0,
        "average_miss_distance": float(np.mean(miss_distances)) if miss_distances else 0.0,
    }


def _extract_history_rows(item: Any) -> tuple[str, list[dict[str, Any]]]:
    """Extract one race label and ordered result rows from a history item."""
    if isinstance(item, list):
        return "", _coerce_ranked_rows(item)

    if not isinstance(item, dict):
        return "", []

    race_name = str(item.get("race_name", "")).strip()
    for key in (
        "rows",
        "predicted_order",
        "actual_order",
        "grid",
        "finish_order",
        "predicted_grid",
        "predicted_results",
    ):
        rows = item.get(key)
        if isinstance(rows, list):
            return race_name, _coerce_ranked_rows(rows)

    return race_name, []


def _summarize_bias_rows(error_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | str]]:
    """Aggregate signed errors into one bias summary per entity."""
    grouped_errors: dict[str, list[float]] = defaultdict(list)
    grouped_abs_errors: dict[str, list[float]] = defaultdict(list)

    for row in error_rows:
        entity = str(row.get("entity", "")).strip()
        if not entity:
            continue
        error = float(row["signed_error"])
        grouped_errors[entity].append(error)
        grouped_abs_errors[entity].append(abs(error))

    summary: dict[str, dict[str, float | str]] = {}
    for entity, errors in grouped_errors.items():
        mean_signed_error = float(np.mean(errors))
        summary[entity] = {
            "samples": float(len(errors)),
            "mean_signed_error": mean_signed_error,
            "mean_abs_error": float(np.mean(grouped_abs_errors[entity])),
            "tendency": (
                "overestimated"
                if mean_signed_error < -0.25
                else "underestimated"
                if mean_signed_error > 0.25
                else "balanced"
            ),
        }
    return summary


def identify_systematic_errors(
    predictions_history: Sequence[Any],
    actuals_history: Sequence[Any],
) -> dict[str, Any]:
    """Identify repeated optimistic or pessimistic bias by team and driver.

    Negative signed error means the model predicted a better finishing position
    than reality delivered. Positive signed error means it was too pessimistic.
    """
    if len(predictions_history) != len(actuals_history):
        raise ValueError("predictions_history and actuals_history must have the same length")

    team_rows: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    races_compared = 0

    for predicted_item, actual_item in zip(predictions_history, actuals_history, strict=True):
        _, predicted_rows = _extract_history_rows(predicted_item)
        _, actual_rows = _extract_history_rows(actual_item)
        aligned = _aligned_positions(predicted_rows, actual_rows)
        if not aligned:
            continue
        races_compared += 1
        for predicted_row, actual_row in aligned:
            signed_error = float(predicted_row["position"] - actual_row["position"])
            driver_rows.append(
                {
                    "entity": predicted_row["driver"],
                    "signed_error": signed_error,
                }
            )

            team_name = str(predicted_row.get("team") or actual_row.get("team") or "").strip()
            if team_name:
                team_rows.append(
                    {
                        "entity": team_name,
                        "signed_error": signed_error,
                    }
                )

    team_bias = _summarize_bias_rows(team_rows)
    driver_bias = _summarize_bias_rows(driver_rows)

    def _sorted_bias(
        source: dict[str, dict[str, float | str]],
        *,
        reverse: bool,
    ) -> list[dict[str, float | str]]:
        items = [
            {"entity": entity, **stats}
            for entity, stats in source.items()
            if isinstance(stats.get("mean_signed_error"), float | int)
        ]
        return sorted(
            items,
            key=lambda row: float(row["mean_signed_error"]),
            reverse=reverse,
        )[:5]

    return {
        "races_compared": races_compared,
        "driver_observations": len(driver_rows),
        "team_bias": team_bias,
        "driver_bias": driver_bias,
        "most_underestimated_teams": _sorted_bias(team_bias, reverse=True),
        "most_overestimated_teams": _sorted_bias(team_bias, reverse=False),
        "most_underestimated_drivers": _sorted_bias(driver_bias, reverse=True),
        "most_overestimated_drivers": _sorted_bias(driver_bias, reverse=False),
    }


def _coerce_event_list(
    rows_or_events: Sequence[Any],
) -> list[Sequence[str | dict[str, Any]]]:
    """Treat a single ranked order and a list of ranked orders consistently."""
    if not rows_or_events:
        return []
    first_item = rows_or_events[0]
    if isinstance(first_item, str | dict):
        return [rows_or_events]
    return list(rows_or_events)


def _mean_metric(events: list[dict[str, float]], key: str) -> float:
    """Average one metric across events, ignoring non-finite values."""
    values = [
        float(event[key]) for event in events if key in event and np.isfinite(float(event[key]))
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def compute_improvement_over_baseline(
    predictions: Sequence[Any],
    actuals: Sequence[Any],
    baseline_predictions: Sequence[Any],
) -> dict[str, Any]:
    """Compare model predictions against a naive baseline across one or more events."""
    prediction_events = _coerce_event_list(predictions)
    actual_events = _coerce_event_list(actuals)
    baseline_events = _coerce_event_list(baseline_predictions)
    event_count = min(len(prediction_events), len(actual_events), len(baseline_events))

    if event_count == 0:
        return {
            "events_compared": 0,
            "model_metrics": {},
            "baseline_metrics": {},
            "improvement": {},
            "model_beats_baseline_on_mae": False,
        }

    model_metrics_by_event = [
        compute_prediction_accuracy(prediction_events[index], actual_events[index])
        for index in range(event_count)
    ]
    baseline_metrics_by_event = [
        compute_prediction_accuracy(baseline_events[index], actual_events[index])
        for index in range(event_count)
    ]

    model_metrics = {
        "mae": _mean_metric(model_metrics_by_event, "mae"),
        "exact_match_rate": _mean_metric(model_metrics_by_event, "exact_match_rate"),
        "within_3_rate": _mean_metric(model_metrics_by_event, "within_3_rate"),
        "spearman_rank": _mean_metric(model_metrics_by_event, "spearman_rank"),
        "kendall_tau": _mean_metric(model_metrics_by_event, "kendall_tau"),
    }
    baseline_metrics = {
        "mae": _mean_metric(baseline_metrics_by_event, "mae"),
        "exact_match_rate": _mean_metric(baseline_metrics_by_event, "exact_match_rate"),
        "within_3_rate": _mean_metric(baseline_metrics_by_event, "within_3_rate"),
        "spearman_rank": _mean_metric(baseline_metrics_by_event, "spearman_rank"),
        "kendall_tau": _mean_metric(baseline_metrics_by_event, "kendall_tau"),
    }

    improvement = {
        "mae_improvement": float(baseline_metrics["mae"] - model_metrics["mae"]),
        "exact_match_rate_delta": float(
            model_metrics["exact_match_rate"] - baseline_metrics["exact_match_rate"]
        ),
        "within_3_rate_delta": float(
            model_metrics["within_3_rate"] - baseline_metrics["within_3_rate"]
        ),
        "spearman_rank_delta": float(
            model_metrics["spearman_rank"] - baseline_metrics["spearman_rank"]
        ),
        "kendall_tau_delta": float(model_metrics["kendall_tau"] - baseline_metrics["kendall_tau"]),
    }

    return {
        "events_compared": event_count,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "improvement": improvement,
        "model_beats_baseline_on_mae": improvement["mae_improvement"] > 0.0,
    }
