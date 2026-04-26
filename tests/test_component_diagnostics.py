"""Tests for challenger movement diagnostics."""

from __future__ import annotations

from src.analysis.component_diagnostics import build_component_movement_diagnostics


def test_component_movement_diagnostics_counts_closer_and_farther_moves():
    """Movement diagnostics should show whether a challenger helped per driver."""
    champion_report = {
        "race_results": [
            {
                "status": "ok",
                "race_name": "Race 1",
                "qualifying_prediction_rows": [
                    {"driver": "NOR", "position": 5},
                    {"driver": "PIA", "position": 2},
                    {"driver": "VER", "position": 1},
                ],
                "qualifying_actual_rows": [
                    {"driver": "NOR", "position": 3},
                    {"driver": "PIA", "position": 4},
                    {"driver": "VER", "position": 1},
                ],
                "race_prediction_rows": [
                    {"driver": "NOR", "position": 4},
                    {"driver": "PIA", "position": 2},
                ],
                "race_actual_rows": [
                    {"driver": "NOR", "position": 2},
                    {"driver": "PIA", "position": 3},
                ],
            }
        ]
    }
    challenger_report = {
        "race_results": [
            {
                "status": "ok",
                "race_name": "Race 1",
                "qualifying_regime": "practice_backed",
                "race_regime": "predicted_grid",
                "qualifying_prediction_rows": [
                    {"driver": "NOR", "position": 3, "qualifying_residual_adjustment": 1.0},
                    {"driver": "PIA", "position": 1, "qualifying_residual_adjustment": -0.5},
                    {"driver": "VER", "position": 1},
                ],
                "qualifying_actual_rows": [
                    {"driver": "NOR", "position": 3},
                    {"driver": "PIA", "position": 4},
                    {"driver": "VER", "position": 1},
                ],
                "race_prediction_rows": [
                    {"driver": "NOR", "position": 3, "race_residual_adjustment": 0.5},
                    {"driver": "PIA", "position": 4, "race_residual_adjustment": 0.0},
                ],
                "race_actual_rows": [
                    {"driver": "NOR", "position": 2},
                    {"driver": "PIA", "position": 3},
                ],
            }
        ]
    }

    diagnostics = build_component_movement_diagnostics(
        champion_report=champion_report,
        challenger_report=challenger_report,
    )

    qualifying = diagnostics["qualifying"]
    race = diagnostics["race"]

    assert qualifying["closer_count"] == 1
    assert qualifying["farther_count"] == 1
    assert qualifying["unchanged_count"] == 1
    assert qualifying["mae_before"] == 4 / 3
    assert qualifying["mae_after"] == 1.0
    assert qualifying["mean_reported_adjustment"] == 0.25
    assert race["closer_count"] == 1
    assert race["farther_count"] == 0
    assert race["unchanged_count"] == 1
    assert race["mean_reported_adjustment"] == 0.25
