"""Tests for model evaluation and diagnostics utilities."""

from __future__ import annotations

import math

import pytest

from src.analysis.model_evaluation import (
    compute_calibration_metrics,
    compute_improvement_over_baseline,
    compute_prediction_accuracy,
    identify_systematic_errors,
)


def test_compute_prediction_accuracy_perfect_string_order():
    """Perfect ranked lists should score perfectly."""
    predicted = ["VER", "NOR", "LEC", "RUS"]
    actual = ["VER", "NOR", "LEC", "RUS"]

    metrics = compute_prediction_accuracy(predicted, actual)

    assert metrics["mae"] == 0.0
    assert metrics["exact_match_rate"] == 100.0
    assert metrics["within_3_rate"] == 100.0
    assert metrics["spearman_rank"] == pytest.approx(1.0)
    assert metrics["kendall_tau"] == pytest.approx(1.0)


def test_compute_prediction_accuracy_supports_dict_rows():
    """Metric helper should accept row payloads with positions and teams."""
    predicted = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
        {"position": 3, "driver": "LEC", "team": "Ferrari"},
    ]
    actual = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 3, "driver": "LEC", "team": "Ferrari"},
    ]

    metrics = compute_prediction_accuracy(predicted, actual)

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["exact_match_rate"] == pytest.approx(33.3333333333)
    assert metrics["within_3_rate"] == 100.0


def test_compute_prediction_accuracy_ignores_non_overlapping_drivers():
    """Only shared drivers should contribute to metrics."""
    predicted = ["VER", "NOR", "LEC"]
    actual = ["NOR", "VER", "PIA"]

    metrics = compute_prediction_accuracy(predicted, actual)

    assert metrics["compared_drivers"] == 2.0
    assert metrics["mae"] == 1.0


def test_compute_prediction_accuracy_single_driver_zeroes_correlations():
    """Correlation metrics should stay defined for one-driver edge cases."""
    metrics = compute_prediction_accuracy(["VER"], ["VER"])

    assert metrics["mae"] == 0.0
    assert metrics["spearman_rank"] == 0.0
    assert metrics["kendall_tau"] == 0.0


def test_compute_prediction_accuracy_empty_inputs_returns_safe_defaults():
    """Empty rankings should not crash metric computation."""
    metrics = compute_prediction_accuracy([], [])

    assert metrics["compared_drivers"] == 0.0
    assert math.isinf(metrics["mae"])
    assert metrics["exact_match_rate"] == 0.0


def test_compute_calibration_metrics_perfect_coverage():
    """Every actual finish inside the band should produce 100% coverage."""
    metrics = compute_calibration_metrics([(1, 3), (2, 6), (5, 8)], [2, 4, 6])

    assert metrics["empirical_coverage"] == 1.0
    assert metrics["mean_interval_width"] == pytest.approx(3.0)
    assert metrics["average_miss_distance"] == 0.0


def test_compute_calibration_metrics_tracks_miss_distance():
    """Misses outside the interval should record how far off the band was."""
    metrics = compute_calibration_metrics([(1, 2), (3, 4), (6, 8)], [4, 2, 10])

    assert metrics["empirical_coverage"] == pytest.approx(0.0)
    assert metrics["average_miss_distance"] == pytest.approx((2.0 + 1.0 + 2.0) / 3.0)
    assert metrics["calibration_error"] == pytest.approx(-0.9)


def test_identify_systematic_errors_reports_team_and_driver_bias():
    """Bias summary should separate optimistic and pessimistic entities."""
    predictions_history = [
        {
            "race_name": "Bahrain Grand Prix",
            "finish_order": [
                {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 2, "driver": "NOR", "team": "McLaren"},
                {"position": 3, "driver": "LEC", "team": "Ferrari"},
            ],
        },
        {
            "race_name": "Australian Grand Prix",
            "finish_order": [
                {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 2, "driver": "NOR", "team": "McLaren"},
                {"position": 3, "driver": "LEC", "team": "Ferrari"},
            ],
        },
    ]
    actuals_history = [
        {
            "race_name": "Bahrain Grand Prix",
            "finish_order": [
                {"position": 1, "driver": "NOR", "team": "McLaren"},
                {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 3, "driver": "LEC", "team": "Ferrari"},
            ],
        },
        {
            "race_name": "Australian Grand Prix",
            "finish_order": [
                {"position": 1, "driver": "NOR", "team": "McLaren"},
                {"position": 2, "driver": "LEC", "team": "Ferrari"},
                {"position": 3, "driver": "VER", "team": "Red Bull Racing"},
            ],
        },
    ]

    summary = identify_systematic_errors(predictions_history, actuals_history)

    assert summary["races_compared"] == 2
    assert summary["team_bias"]["Red Bull Racing"]["tendency"] == "overestimated"
    assert summary["team_bias"]["McLaren"]["tendency"] == "underestimated"
    assert summary["driver_bias"]["VER"]["mean_signed_error"] < 0.0
    assert summary["driver_bias"]["NOR"]["mean_signed_error"] > 0.0


def test_identify_systematic_errors_requires_matched_histories():
    """Prediction and actual histories should stay aligned by event count."""
    with pytest.raises(ValueError, match="same length"):
        identify_systematic_errors([{"finish_order": []}], [])


def test_compute_improvement_over_baseline_single_event():
    """Improvement summary should flag when the model beats a naive order."""
    predictions = ["VER", "NOR", "LEC", "RUS"]
    actuals = ["NOR", "VER", "LEC", "RUS"]
    baseline = ["LEC", "RUS", "VER", "NOR"]

    summary = compute_improvement_over_baseline(predictions, actuals, baseline)

    assert summary["events_compared"] == 1
    assert summary["improvement"]["mae_improvement"] > 0.0
    assert summary["model_beats_baseline_on_mae"] is True


def test_compute_improvement_over_baseline_multiple_events():
    """Aggregate comparison should average improvements across events."""
    predictions = [
        ["VER", "NOR", "LEC"],
        ["NOR", "VER", "LEC"],
    ]
    actuals = [
        ["VER", "NOR", "LEC"],
        ["NOR", "LEC", "VER"],
    ]
    baseline = [
        ["NOR", "VER", "LEC"],
        ["VER", "NOR", "LEC"],
    ]

    summary = compute_improvement_over_baseline(predictions, actuals, baseline)

    assert summary["events_compared"] == 2
    assert summary["model_metrics"]["mae"] < summary["baseline_metrics"]["mae"]
    assert summary["improvement"]["within_3_rate_delta"] >= 0.0


def test_compute_improvement_over_baseline_handles_empty_events():
    """No events should return an explicit empty comparison payload."""
    summary = compute_improvement_over_baseline([], [], [])

    assert summary["events_compared"] == 0
    assert summary["model_metrics"] == {}
    assert summary["baseline_metrics"] == {}
