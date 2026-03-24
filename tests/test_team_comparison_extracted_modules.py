"""Direct tests for extracted team-comparison modules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard import team_comparison_fallbacks, team_radar, team_snapshot_history


def test_team_radar_default_team_selection_prefers_big_four_order() -> None:
    selection = team_radar._default_team_selection(
        [
            "Haas",
            "Ferrari",
            "Aston Martin",
            "Mercedes",
            "McLaren",
            "Red Bull Racing",
        ],
        max_teams=4,
    )

    assert selection == ["McLaren", "Mercedes", "Ferrari", "Red Bull Racing"]


def test_team_snapshot_history_marks_weekend_average_team_names() -> None:
    assert (
        team_snapshot_history._comparison_display_team_name(
            "McLaren",
            {"comparison_fallback_source": "same_event_average"},
        )
        == "McLaren*"
    )
    assert (
        team_snapshot_history._comparison_display_team_name(
            "Ferrari",
            {"comparison_fallback_source": "latest_snapshot"},
        )
        == "Ferrari"
    )


def test_team_comparison_fallbacks_apply_same_event_scores_to_missing_metric() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "Team": "McLaren",
                "Overall Pace": 0.74,
                "Overall Performance": 0.68,
                "Slow Corners": 0.71,
                "Medium Corners": 0.72,
                "Fast Corners": 0.73,
                "Braking": 0.50,
                "Top Speed": 0.75,
                "Tire Deg": 0.76,
                "Radar Composite": 0.695,
                "Radar Minus Prior": 0.015,
            }
        ]
    )
    teams_payload = {
        "McLaren": {
            "comparison_fallback_source": "same_event_average",
            "overall_performance": 0.68,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.74,
                    "slow_corner_performance": 0.71,
                    "medium_corner_performance": 0.72,
                    "fast_corner_performance": 0.73,
                    "top_speed": 0.75,
                    "tire_deg_performance": 0.76,
                }
            },
        }
    }

    updated_df, unresolved_missing_count = (
        team_comparison_fallbacks._apply_display_metric_fallbacks(
            comparison_df,
            teams_payload=teams_payload,
            selected_teams=["McLaren"],
            profile="balanced",
            same_event_display_scores={"McLaren": {"Braking": 0.82}},
            latest_reliable_display_scores={},
        )
    )

    row = updated_df.iloc[0]
    expected_composite = (0.71 + 0.72 + 0.73 + 0.82 + 0.75 + 0.76) / 6.0

    assert unresolved_missing_count == 0
    assert row["Braking"] == pytest.approx(0.82, abs=1e-6)
    assert row["Radar Composite"] == pytest.approx(expected_composite, abs=1e-6)
    assert row["Radar Minus Prior"] == pytest.approx(expected_composite - 0.68, abs=1e-6)
