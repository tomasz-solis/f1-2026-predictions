"""
Tests for Prediction Metrics Calculator
"""

import pytest

from src.utils.prediction_metrics import PredictionMetrics


@pytest.fixture
def perfect_prediction():
    """Perfect prediction - all positions correct."""
    predicted = [
        {"position": 1, "driver": "Verstappen"},
        {"position": 2, "driver": "Norris"},
        {"position": 3, "driver": "Leclerc"},
        {"position": 4, "driver": "Piastri"},
        {"position": 5, "driver": "Sainz"},
    ]
    actual = [
        {"position": 1, "driver": "Verstappen"},
        {"position": 2, "driver": "Norris"},
        {"position": 3, "driver": "Leclerc"},
        {"position": 4, "driver": "Piastri"},
        {"position": 5, "driver": "Sainz"},
    ]
    return predicted, actual


@pytest.fixture
def partially_correct():
    """Partially correct prediction."""
    predicted = [
        {"position": 1, "driver": "Verstappen"},
        {"position": 2, "driver": "Norris"},
        {"position": 3, "driver": "Leclerc"},
        {"position": 4, "driver": "Piastri"},
        {"position": 5, "driver": "Sainz"},
    ]
    actual = [
        {"position": 1, "driver": "Norris"},  # Wrong
        {"position": 2, "driver": "Verstappen"},  # Wrong
        {"position": 3, "driver": "Leclerc"},  # Correct
        {"position": 4, "driver": "Sainz"},  # Wrong
        {"position": 5, "driver": "Piastri"},  # Wrong
    ]
    return predicted, actual


def test_position_accuracy_perfect(perfect_prediction):
    """Test position accuracy with perfect prediction."""
    predicted, actual = perfect_prediction
    accuracy = PredictionMetrics.position_accuracy(predicted, actual)
    assert accuracy == 100.0


def test_position_accuracy_partial(partially_correct):
    """Test position accuracy with partially correct prediction."""
    predicted, actual = partially_correct
    accuracy = PredictionMetrics.position_accuracy(predicted, actual)
    assert accuracy == 20.0  # 1 out of 5 correct (Leclerc at P3)


def test_mean_absolute_error_perfect(perfect_prediction):
    """Test MAE with perfect prediction."""
    predicted, actual = perfect_prediction
    mae = PredictionMetrics.mean_absolute_error(predicted, actual)
    assert mae == 0.0


def test_mean_absolute_error_partial(partially_correct):
    """Test MAE with partially correct prediction."""
    predicted, actual = partially_correct
    mae = PredictionMetrics.mean_absolute_error(predicted, actual)
    # Verstappen: |1-2| = 1
    # Norris: |2-1| = 1
    # Leclerc: |3-3| = 0
    # Piastri: |4-5| = 1
    # Sainz: |5-4| = 1
    # Average: (1+1+0+1+1)/5 = 0.8
    assert mae == 0.8


def test_within_n_positions_perfect(perfect_prediction):
    """Test within N positions with perfect prediction."""
    predicted, actual = perfect_prediction
    within_1 = PredictionMetrics.within_n_positions(predicted, actual, 1)
    assert within_1 == 100.0


def test_within_n_positions_partial(partially_correct):
    """Test within N positions with partially correct prediction."""
    predicted, actual = partially_correct
    within_1 = PredictionMetrics.within_n_positions(predicted, actual, 1)
    # All predictions are within 1 position
    assert within_1 == 100.0

    within_0 = PredictionMetrics.within_n_positions(predicted, actual, 0)
    # Only Leclerc is exactly correct
    assert within_0 == 20.0


def test_correlation_coefficient_perfect(perfect_prediction):
    """Test correlation with perfect prediction."""
    predicted, actual = perfect_prediction
    corr = PredictionMetrics.correlation_coefficient(predicted, actual)
    assert pytest.approx(corr, rel=1e-9) == 1.0  # Perfect correlation


def test_correlation_coefficient_partial(partially_correct):
    """Test correlation with partially correct prediction."""
    predicted, actual = partially_correct
    corr = PredictionMetrics.correlation_coefficient(predicted, actual)
    # Not perfect but should still be positive and fairly high
    assert 0.5 < corr < 1.0


def test_podium_accuracy_perfect(perfect_prediction):
    """Test podium accuracy with perfect prediction."""
    predicted, actual = perfect_prediction
    podium = PredictionMetrics.podium_accuracy(predicted, actual)
    assert podium["correct_drivers"] == 3
    assert podium["correct_positions"] == 3
    assert podium["accuracy"] == 100.0


def test_podium_accuracy_partial(partially_correct):
    """Test podium accuracy with partially correct prediction."""
    predicted, actual = partially_correct
    podium = PredictionMetrics.podium_accuracy(predicted, actual)
    # All 3 drivers on podium are correct (Verstappen, Norris, Leclerc)
    assert podium["correct_drivers"] == 3
    # Only Leclerc is in correct position
    assert podium["correct_positions"] == 1
    assert podium["accuracy"] == 100.0


def test_winner_accuracy_correct(perfect_prediction):
    """Test winner accuracy with correct prediction."""
    predicted, actual = perfect_prediction
    winner_correct = PredictionMetrics.winner_accuracy(predicted, actual)
    assert winner_correct is True


def test_winner_accuracy_wrong(partially_correct):
    """Test winner accuracy with wrong prediction."""
    predicted, actual = partially_correct
    winner_correct = PredictionMetrics.winner_accuracy(predicted, actual)
    assert winner_correct is False  # Predicted Verstappen, actual Norris


def test_top_k_overlap_counts_driver_hits(partially_correct):
    """Top-k overlap should count driver-set hits, not exact order."""
    predicted, actual = partially_correct
    overlap = PredictionMetrics.top_k_overlap(predicted, actual, k=3)

    assert overlap["hits"] == 3.0
    assert overlap["pct"] == 100.0


def test_empty_predictions():
    """Test metrics with empty predictions."""
    metrics = PredictionMetrics()

    accuracy = metrics.position_accuracy([], [])
    assert accuracy == 0.0

    mae = metrics.mean_absolute_error([], [])
    assert mae == float("inf")

    within_1 = metrics.within_n_positions([], [], 1)
    assert within_1 == 0.0

    corr = metrics.correlation_coefficient([], [])
    assert corr == 0.0


def test_calculate_all_metrics():
    """Test calculating all metrics at once."""
    prediction_data = {
        "metadata": {"race_name": "Bahrain GP", "session_name": "FP2"},
        "qualifying": {
            "predicted_grid": [
                {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
                {"position": 2, "driver": "Norris", "team": "McLaren"},
                {"position": 3, "driver": "Leclerc", "team": "Ferrari"},
            ]
        },
        "race": {
            "predicted_results": [
                {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
                {"position": 2, "driver": "Norris", "team": "McLaren"},
                {"position": 3, "driver": "Leclerc", "team": "Ferrari"},
            ]
        },
        "actuals": {
            "qualifying": [
                {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
                {"position": 2, "driver": "Leclerc", "team": "Ferrari"},
                {"position": 3, "driver": "Norris", "team": "McLaren"},
            ],
            "race": [
                {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
                {"position": 2, "driver": "Norris", "team": "McLaren"},
                {"position": 3, "driver": "Leclerc", "team": "Ferrari"},
            ],
        },
    }

    metrics = PredictionMetrics.calculate_all_metrics(prediction_data)

    assert metrics is not None
    assert "qualifying" in metrics
    assert "race" in metrics

    # Qualifying: Only Verstappen correct at P1
    assert metrics["qualifying"]["exact_accuracy"] == pytest.approx(33.33, rel=0.1)

    # Race: All positions correct
    assert metrics["race"]["exact_accuracy"] == 100.0
    assert metrics["race"]["winner_correct"] is True


def test_calculate_all_metrics_no_actuals():
    """Test calculating metrics when no actuals available."""
    prediction_data = {
        "metadata": {"race_name": "Bahrain GP", "session_name": "FP2"},
        "qualifying": {"predicted_grid": []},
        "race": {"predicted_results": []},
        "actuals": {"qualifying": None, "race": None},
    }

    metrics = PredictionMetrics.calculate_all_metrics(prediction_data)

    assert metrics is None


def test_calculate_prediction_target_metrics_uses_explicit_targets():
    """Target-aware metrics should read the canonical target payloads first."""
    prediction_data = {
        "metadata": {
            "race_name": "Chinese Grand Prix",
            "session_name": "FP1",
            "weekend_format": "sprint",
        },
        "qualifying": {"predicted_grid": []},
        "race": {"predicted_results": []},
        "targets": {
            "sprint_qualifying": {
                "target_session": "SQ",
                "predicted_order": [
                    {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
                    {"position": 2, "driver": "Norris", "team": "McLaren"},
                    {"position": 3, "driver": "Leclerc", "team": "Ferrari"},
                ],
                "eligible_at_save": True,
            }
        },
        "actuals": {
            "qualifying": None,
            "race": None,
            "targets": {
                "sprint_qualifying": [
                    {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
                    {"position": 2, "driver": "Leclerc", "team": "Ferrari"},
                    {"position": 3, "driver": "Norris", "team": "McLaren"},
                ]
            },
        },
    }

    metrics = PredictionMetrics.calculate_prediction_target_metrics(
        prediction_data,
        is_sprint=True,
    )

    assert "sprint_qualifying" in metrics
    assert metrics["sprint_qualifying"]["top_3_hits"] == 3.0
