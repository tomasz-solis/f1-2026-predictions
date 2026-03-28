"""Tests for saved-actual team scoring helpers."""

from src.predictors.baseline.data_support import score_teams_from_actual_rows


def test_score_teams_from_actual_rows_uses_rank_spacing():
    actual_rows = [
        {"team": "McLaren", "position": 1},
        {"team": "McLaren", "position": 2},
        {"team": "Ferrari", "position": 3},
        {"team": "Ferrari", "position": 4},
        {"team": "Mercedes", "position": 5},
        {"team": "Mercedes", "position": 6},
        {"team": "Aston Martin", "position": 7},
        {"team": "Aston Martin", "position": 8},
        {"team": "Haas F1 Team", "position": 9},
        {"team": "Haas F1 Team", "position": 10},
    ]

    scores = score_teams_from_actual_rows(
        actual_rows,
        known_teams={"McLaren", "Ferrari", "Mercedes", "Aston Martin", "Haas F1 Team"},
    )

    assert scores == {
        "McLaren": 1.0,
        "Ferrari": 0.75,
        "Mercedes": 0.5,
        "Aston Martin": 0.25,
        "Haas F1 Team": 0.0,
    }


def test_score_teams_from_actual_rows_returns_neutral_score_for_single_team():
    scores = score_teams_from_actual_rows(
        [{"team": "Ferrari", "position": 2}],
        known_teams={"Ferrari"},
    )

    assert scores == {"Ferrari": 0.5}
