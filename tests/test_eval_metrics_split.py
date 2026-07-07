"""Tests for finisher-only MAE, weighted MAE, and DNF Brier calibration."""

import math

import pytest

from src.analysis.model_evaluation import (
    compute_dnf_calibration,
    compute_prediction_accuracy,
    position_weight,
    row_is_dnf,
)


def test_row_is_dnf_detects_variants():
    assert row_is_dnf({"driver": "X", "dnf": True}) is True
    assert row_is_dnf({"driver": "X", "status": "Retired"}) is True
    assert row_is_dnf({"driver": "X", "status": "Disqualified"}) is True
    assert row_is_dnf({"driver": "X", "classified": False}) is True
    assert row_is_dnf({"driver": "X", "status": "Finished"}) is False
    assert row_is_dnf({"driver": "X", "position": 5}) is False  # no signal -> finished


def test_position_weight_is_top_heavy():
    assert position_weight(1, field_size=20, scheme="reciprocal") > position_weight(
        10, field_size=20, scheme="reciprocal"
    )
    assert position_weight(10, field_size=20, scheme="reciprocal") > position_weight(
        20, field_size=20, scheme="reciprocal"
    )
    # Linear stays positive across the field and still decreases.
    assert position_weight(1, field_size=20, scheme="linear") > position_weight(
        20, field_size=20, scheme="linear"
    )


def test_weighted_mae_penalises_top_errors_more():
    actual = [{"driver": d, "position": i} for i, d in enumerate("ABCD", start=1)]
    top_swap = [
        {"driver": "B", "position": 1},
        {"driver": "A", "position": 2},
        {"driver": "C", "position": 3},
        {"driver": "D", "position": 4},
    ]
    bottom_swap = [
        {"driver": "A", "position": 1},
        {"driver": "B", "position": 2},
        {"driver": "D", "position": 3},
        {"driver": "C", "position": 4},
    ]
    top = compute_prediction_accuracy(top_swap, actual)
    bottom = compute_prediction_accuracy(bottom_swap, actual)
    # Plain MAE identical (one swap each); weighted MAE must punish the top swap more.
    assert top["mae"] == pytest.approx(bottom["mae"])
    assert top["weighted_mae"] > bottom["weighted_mae"]


def test_finisher_mae_excludes_actual_dnf():
    # D actually retired from P4; the model had them mid-grid (big error we should not count).
    actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 2, "status": "Finished"},
        {"driver": "C", "position": 3, "status": "Finished"},
        {"driver": "D", "position": 4, "status": "Retired"},
    ]
    predicted = [
        {"driver": "A", "position": 1},
        {"driver": "B", "position": 2},
        {"driver": "C", "position": 3},
        {"driver": "D", "position": 20},
    ]
    metrics = compute_prediction_accuracy(predicted, actual)
    assert metrics["dnf_count"] == 1.0
    assert metrics["finisher_count"] == 3.0
    assert metrics["finisher_mae"] == 0.0  # finishers were perfect
    assert metrics["mae"] > metrics["finisher_mae"]  # the DNF drags raw MAE


def test_finisher_mae_equals_mae_without_dnf_labels():
    actual = [{"driver": d, "position": i} for i, d in enumerate("ABC", start=1)]
    predicted = [
        {"driver": "A", "position": 2},
        {"driver": "B", "position": 1},
        {"driver": "C", "position": 3},
    ]
    metrics = compute_prediction_accuracy(predicted, actual)
    assert metrics["finisher_mae"] == pytest.approx(metrics["mae"])


def test_dnf_calibration_brier_and_skill():
    predicted = [
        {"driver": "A", "position": 1, "dnf_probability": 0.05},
        {"driver": "B", "position": 2, "dnf_probability": 0.10},
        {"driver": "C", "position": 3, "dnf_probability": 0.80},
    ]
    actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 2, "status": "Finished"},
        {"driver": "C", "position": 20, "status": "Retired"},
    ]
    cal = compute_dnf_calibration(predicted, actual)
    assert cal["scored_drivers"] == 3.0
    assert cal["actual_dnf_count"] == 1.0
    # Brier = mean(0.05^2, 0.10^2, (0.80-1)^2) = (0.0025+0.01+0.04)/3
    assert cal["brier_score"] == pytest.approx((0.0025 + 0.01 + 0.04) / 3)
    assert cal["brier_skill_score"] > 0.0  # beats the base-rate baseline


def test_dnf_calibration_handles_missing_probabilities():
    predicted = [{"driver": "A", "position": 1}]  # no dnf_probability
    actual = [{"driver": "A", "position": 1}]
    cal = compute_dnf_calibration(predicted, actual)
    assert cal["scored_drivers"] == 0.0
    assert math.isnan(cal["brier_score"])


def test_dnf_calibration_zero_dnf_event_scores_brier_but_not_skill():
    # An event where nobody retires has baseline_brier 0 (base rate 0), so the
    # skill score is undefined (NaN) while the Brier score itself stays finite.
    predicted = [
        {"driver": "A", "position": 1, "dnf_probability": 0.05},
        {"driver": "B", "position": 2, "dnf_probability": 0.10},
    ]
    actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 2, "status": "Finished"},
    ]
    cal = compute_dnf_calibration(predicted, actual)
    assert cal["scored_drivers"] == 2.0
    assert cal["actual_dnf_count"] == 0.0
    assert cal["brier_score"] == pytest.approx((0.05**2 + 0.10**2) / 2)
    assert cal["baseline_brier"] == 0.0
    assert math.isnan(cal["brier_skill_score"])


def test_dnf_calibration_reads_dnf_risk_alias_and_normalises_percent():
    # Persisted artifacts use the "dnf_risk" alias; some store it as a 0-100 percentage.
    predicted = [
        {"driver": "A", "position": 1, "dnf_risk": 5.0},  # percentage form -> 0.05
        {"driver": "B", "position": 2, "dnf_risk": 80.0},  # -> 0.80
    ]
    actual = [
        {"driver": "A", "position": 1, "status": "Finished"},
        {"driver": "B", "position": 20, "status": "Retired"},
    ]
    cal = compute_dnf_calibration(predicted, actual)
    assert cal["scored_drivers"] == 2.0
    assert cal["actual_dnf_count"] == 1.0
    # Brier = mean(0.05^2, (0.80-1)^2) = (0.0025 + 0.04)/2
    assert cal["brier_score"] == pytest.approx((0.0025 + 0.04) / 2)
