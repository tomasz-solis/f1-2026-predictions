"""Tests for model promotion and online learning guardrails."""

from __future__ import annotations

import pytest

from src.analysis.promotion_gate import evaluate_component_promotion_gate
from src.systems.systematic_learning import SystematicLearningSystem
from src.systems.weight_schedule import get_schedule_weights


def test_promotion_gate_requires_real_central_mae_improvement():
    """A challenger with neutral MAE should stay blocked even if it is not harmful."""
    result = evaluate_component_promotion_gate(
        deltas={
            "race_mae_improvement": 0.0,
            "qualifying_mae_improvement": 0.0,
            "top3_accuracy_delta": 0.0,
            "winner_accuracy_delta": 0.0,
        },
        race_delta_summary={
            "races_compared": 2,
            "race_worse_count": 0,
            "race_better_count": 0,
            "qualifying_worse_count": 0,
            "qualifying_better_count": 0,
        },
    )

    assert result["passed"] is False
    assert "combined race and qualifying MAE improvement" in result["reasons"][0]


def test_promotion_gate_allows_improvement_without_accuracy_regression():
    """A challenger can pass when MAE improves and headline accuracy is preserved."""
    result = evaluate_component_promotion_gate(
        deltas={
            "race_mae_improvement": 0.05,
            "qualifying_mae_improvement": 0.03,
            "top3_accuracy_delta": -1.0,
            "winner_accuracy_delta": 0.0,
        },
        race_delta_summary={
            "races_compared": 4,
            "race_worse_count": 1,
            "race_better_count": 2,
            "qualifying_worse_count": 1,
            "qualifying_better_count": 2,
        },
    )

    assert result["passed"] is True
    assert result["reasons"] == []


def test_promotion_gate_blocks_broad_weekend_degradation():
    """A mean win should not hide broad per-weekend degradation."""
    result = evaluate_component_promotion_gate(
        deltas={
            "race_mae_improvement": 0.20,
            "qualifying_mae_improvement": 0.03,
            "top3_accuracy_delta": 0.0,
            "winner_accuracy_delta": 0.0,
        },
        race_delta_summary={
            "races_compared": 5,
            "race_worse_count": 4,
            "race_better_count": 1,
            "qualifying_worse_count": 1,
            "qualifying_better_count": 2,
        },
    )

    assert result["passed"] is False
    assert any("race MAE got worse on more weekends" in reason for reason in result["reasons"])


def test_systematic_learning_adjusts_toward_actual_results(tmp_path):
    """Repeated actuals should produce position corrections in the right direction."""
    learner = SystematicLearningSystem(state_file=tmp_path / "learning_state.json")
    prediction_record = {
        "metadata": {"race_name": "Race 1", "run_id": "run-1"},
        "qualifying": {
            "predicted_grid": [
                {"driver": "NOR", "team": "McLaren", "position": 5},
                {"driver": "PIA", "team": "McLaren", "position": 2},
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            ]
        },
        "actuals": {
            "qualifying": [
                {"driver": "NOR", "team": "McLaren", "position": 3},
                {"driver": "PIA", "team": "McLaren", "position": 4},
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            ]
        },
    }

    update = learner.update_from_prediction_record(prediction_record, alpha=0.5)

    assert update["driver_updates"] == 3
    assert learner.get_driver_position_adjustment("NOR", "qualifying") > 0.0
    assert learner.get_driver_position_adjustment("PIA", "qualifying") < 0.0


def test_reset_schedule_shifts_trust_to_current_season_data():
    """Reset-year weights should let live evidence take over as races accumulate."""
    weights_by_race = [
        get_schedule_weights(race_number=race_number, schedule="rapid_adaptive")
        for race_number in (1, 2, 3, 4)
    ]

    current_weights = [weights["current"] for weights in weights_by_race]
    testing_weights = [weights["testing"] for weights in weights_by_race]

    assert current_weights == sorted(current_weights)
    assert testing_weights == sorted(testing_weights, reverse=True)
    assert current_weights[-1] == pytest.approx(0.95)
