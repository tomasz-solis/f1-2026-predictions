"""Tests for actual results fetching utility."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.utils.actual_results_fetcher import fetch_actual_session_results


def test_fetch_actual_session_results_canonicalizes_teams_and_positions():
    """Team names should be mapped and missing positions filled for valid competitive payloads."""
    rows = [
        {
            "Abbreviation": "VER",
            "TeamName": "Oracle Red Bull Racing",
            "Position": 1,
        },
        {"Abbreviation": "LEC", "TeamName": "Scuderia Ferrari", "Position": 2},
    ]
    rows.extend(
        {
            "Abbreviation": f"DRV{i}",
            "TeamName": "McLaren Formula 1 Team",
            "Position": i,
        }
        for i in range(3, 11)
    )

    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(rows)

    with patch("src.utils.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is not None
    assert results[0]["team"] == "Red Bull Racing"
    assert results[0]["position"] == 1
    assert results[1]["team"] == "Ferrari"
    assert results[1]["position"] == 2


def test_fetch_actual_session_results_fails_closed_on_malformed_row():
    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(
        [
            {"Abbreviation": "VER", "TeamName": "Oracle Red Bull Racing", "Position": 1},
            {"Abbreviation": "", "TeamName": "Scuderia Ferrari", "Position": 2},
        ]
    )

    with patch("src.utils.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is None


def test_fetch_actual_session_results_fails_closed_on_missing_position():
    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(
        [
            {"Abbreviation": "VER", "TeamName": "Oracle Red Bull Racing", "Position": 1},
            {"Abbreviation": "LEC", "TeamName": "Scuderia Ferrari", "Position": None},
        ]
    )

    with patch("src.utils.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is None


def test_fetch_actual_session_results_fails_closed_on_too_few_entries():
    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(
        [
            {"Abbreviation": "VER", "TeamName": "Oracle Red Bull Racing", "Position": 1},
            {"Abbreviation": "LEC", "TeamName": "Scuderia Ferrari", "Position": 2},
        ]
    )

    with patch("src.utils.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is None
