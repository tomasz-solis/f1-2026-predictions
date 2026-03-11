"""Regression tests for performance extractor normalization."""

from src.extractors.performance import extract_all_teams_performance


def test_extract_all_teams_performance_uses_team_keys():
    all_team_data = {
        "Team A": {
            "fp1": {
                "speed_profile": {"top_speed": 320.0},
                "sector_times": {"s1": 30.0},
            }
        },
        "Team B": {
            "fp1": {
                "speed_profile": {"top_speed": 315.0},
                "sector_times": {"s1": 31.0},
            }
        },
    }

    normalized = extract_all_teams_performance(all_team_data, "fp1")

    assert set(normalized.keys()) == {"Team A", "Team B"}
    assert all("top_speed" in metrics for metrics in normalized.values())
    assert all("slow_corner_performance" in metrics for metrics in normalized.values())
    assert normalized["Team A"]["top_speed"] == 1.0
    assert normalized["Team B"]["top_speed"] == 0.0
    assert normalized["Team A"]["slow_corner_performance"] == 1.0
    assert normalized["Team B"]["slow_corner_performance"] == 0.0


def test_extract_all_teams_performance_returns_neutral_scores_for_ties():
    all_team_data = {
        "Team A": {
            "fp1": {
                "speed_profile": {"top_speed": 320.0},
                "sector_times": {"s1": 30.0},
            }
        },
        "Team B": {
            "fp1": {
                "speed_profile": {"top_speed": 320.0},
                "sector_times": {"s1": 30.0},
            }
        },
    }

    normalized = extract_all_teams_performance(all_team_data, "fp1")

    assert normalized["Team A"]["top_speed"] == 0.5
    assert normalized["Team B"]["top_speed"] == 0.5
    assert normalized["Team A"]["slow_corner_performance"] == 0.5
    assert normalized["Team B"]["slow_corner_performance"] == 0.5
