"""
Validation metrics for F1 predictions

Compare predicted rankings vs actual qualifying results
"""

from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.utils.team_mapping import canonicalize_team


def compare_rankings(predicted: list[str], actual: list[str]) -> dict[str, float]:
    """
    Compare predicted team ranking vs actual results. Returns dict with all metrics.
    """
    # Normalize team names
    predicted = [canonicalize_team(t) for t in predicted]
    actual = [canonicalize_team(t) for t in actual]

    metrics: dict[str, float] = {}

    # Winner prediction
    if len(predicted) > 0 and len(actual) > 0:
        metrics["winner_correct"] = 1.0 if predicted[0] == actual[0] else 0.0

    # Top N accuracy
    for n in [3, 5, 10]:
        if len(predicted) >= n and len(actual) >= n:
            pred_top_n = set(predicted[:n])
            actual_top_n = set(actual[:n])
            overlap = len(pred_top_n & actual_top_n)
            metrics[f"top{n}_accuracy"] = overlap / n

    # Ranking correlation
    common_teams = [t for t in predicted if t in actual]

    if len(common_teams) >= 3:
        pred_positions = [predicted.index(t) + 1 for t in common_teams]
        actual_positions = [actual.index(t) + 1 for t in common_teams]

        # Spearman correlation
        rho, _ = spearmanr(pred_positions, actual_positions)
        metrics["spearman"] = float(rho)

        # Kendall's tau
        tau, _ = kendalltau(pred_positions, actual_positions)
        metrics["kendall_tau"] = float(tau)

        # Mean absolute error (positions off)
        mae = np.mean([abs(p - a) for p, a in zip(pred_positions, actual_positions, strict=False)])
        metrics["mae_positions"] = float(mae)

    return metrics


def aggregate_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate metrics across multiple races

    Returns mean for each metric
    """
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    aggregated: dict[str, float] = {}

    for key in keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[key] = float(np.mean(values))

    return aggregated


def confidence_calibration(predictions: list[tuple[float, bool]]) -> dict[str, Any]:
    """
    Check if confidence scores are well-calibrated. Returns calibration metrics.
    """
    if not predictions:
        return {}

    # Group by confidence bins
    bins = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
    bin_accuracy: dict[str, dict[str, float | int]] = {}

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_preds = [(conf, correct) for conf, correct in predictions if low <= conf < high]

        if bin_preds:
            accuracy = float(sum(correct for _, correct in bin_preds) / len(bin_preds))
            avg_conf = float(np.mean([conf for conf, _ in bin_preds]))
            bin_accuracy[f"{low:.1f}-{high:.1f}"] = {
                "confidence": avg_conf,
                "accuracy": accuracy,
                "count": len(bin_preds),
            }

    # Brier score (lower is better, 0-1 range)
    brier = np.mean([(conf - (1.0 if correct else 0.0)) ** 2 for conf, correct in predictions])

    return {"brier_score": float(brier), "bins": bin_accuracy}


def analyze_by_track_type(
    results: dict[str, dict[str, float]],
    track_types: dict[str, str],
) -> dict[str, dict[str, float]]:
    """
    Group results by track type. Returns aggregated metrics per type.
    """
    by_type: dict[str, list[dict[str, float]]] = {}

    for race, metrics in results.items():
        track_type = track_types.get(race, "unknown")

        if track_type not in by_type:
            by_type[track_type] = []

        by_type[track_type].append(metrics)

    # Aggregate each type
    aggregated: dict[str, dict[str, float]] = {}
    for track_type, metrics_list in by_type.items():
        aggregated[track_type] = aggregate_metrics(metrics_list)
        aggregated[track_type]["count"] = len(metrics_list)

    return aggregated


def analyze_by_stage(
    results: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """
    Compare prediction quality by stage. Returns aggregated metrics per stage.
    """
    by_stage: dict[str, list[dict[str, float]]] = {}

    for _race, stage_results in results.items():
        for stage, metrics in stage_results.items():
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(metrics)

    # Aggregate each stage
    aggregated: dict[str, dict[str, float]] = {}
    for stage, metrics_list in by_stage.items():
        aggregated[stage] = aggregate_metrics(metrics_list)
        aggregated[stage]["count"] = len(metrics_list)

    return aggregated
