"""Tests for actual results fetching utility."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.actual_results_fetcher import fetch_actual_session_results


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
        for i in range(3, 21)
    )

    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(rows)

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
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

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is None


def test_fetch_actual_session_results_fails_closed_on_missing_position():
    rows = [
        {"Abbreviation": "VER", "TeamName": "Oracle Red Bull Racing", "Position": 1},
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
    rows.append({"Abbreviation": "MID", "TeamName": "McLaren Formula 1 Team", "Position": None})
    rows.extend(
        {
            "Abbreviation": f"DRV{i}",
            "TeamName": "McLaren Formula 1 Team",
            "Position": i,
        }
        for i in range(12, 21)
    )

    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(rows)

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
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

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is None


def test_fetch_actual_session_results_rejects_partial_competitive_grid():
    rows = [
        {"Abbreviation": "VER", "TeamName": "Oracle Red Bull Racing", "Position": 1},
        {"Abbreviation": "LEC", "TeamName": "Scuderia Ferrari", "Position": 2},
    ]
    rows.extend(
        {
            "Abbreviation": f"DRV{i}",
            "TeamName": "McLaren Formula 1 Team",
            "Position": i,
        }
        for i in range(3, 18)
    )

    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(rows)

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Bahrain Grand Prix", "Q")

    assert results is None


def test_fetch_actual_session_results_infers_trailing_qualifying_positions():
    rows = [
        {"Abbreviation": "VER", "TeamName": "Oracle Red Bull Racing", "Position": 1},
        {"Abbreviation": "LEC", "TeamName": "Scuderia Ferrari", "Position": 2},
    ]
    rows.extend(
        {
            "Abbreviation": f"DRV{i}",
            "TeamName": "McLaren Formula 1 Team",
            "Position": i,
        }
        for i in range(3, 20)
    )
    rows.extend(
        [
            {"Abbreviation": "STR", "TeamName": "Aston Martin", "Position": None},
            {"Abbreviation": "VER2", "TeamName": "Red Bull Racing", "Position": None},
            {"Abbreviation": "SAI", "TeamName": "Williams", "Position": None},
        ]
    )

    mock_session = MagicMock()
    mock_session.results = pd.DataFrame(rows)

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Australian Grand Prix", "Q")

    assert results is not None
    assert len(results) == 22
    assert [entry["position"] for entry in results[-3:]] == [20, 21, 22]
    assert [entry["driver"] for entry in results[-3:]] == ["STR", "VER2", "SAI"]


def test_fetch_actual_session_results_recomputes_partial_qualifying_positions_before_inference():
    prefix_rows = [
        {
            "Abbreviation": f"DRV{i}",
            "TeamName": "McLaren Formula 1 Team",
            "Position": i,
        }
        for i in range(1, 17)
    ]
    initial_results = pd.DataFrame(
        prefix_rows
        + [
            {"Abbreviation": "ALO", "TeamName": "Aston Martin", "Position": 17},
            {"Abbreviation": "PER", "TeamName": "Cadillac", "Position": 18},
            {"Abbreviation": "BOT", "TeamName": "Cadillac", "Position": 19},
            {"Abbreviation": "STR", "TeamName": "Aston Martin", "Position": None},
            {"Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": None},
            {"Abbreviation": "SAI", "TeamName": "Williams", "Position": None},
        ]
    )

    mock_session = MagicMock()
    mock_session.results = initial_results.copy()

    def _recompute_results(*, force: bool) -> None:
        assert force is True
        mock_session.results = pd.DataFrame(
            prefix_rows
            + [
                {"Abbreviation": "ALO", "TeamName": "Aston Martin", "Position": 17},
                {"Abbreviation": "PER", "TeamName": "Cadillac", "Position": 18},
                {"Abbreviation": "BOT", "TeamName": "Cadillac", "Position": 19},
                {"Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 20},
                {"Abbreviation": "STR", "TeamName": "Aston Martin", "Position": None},
                {"Abbreviation": "SAI", "TeamName": "Williams", "Position": None},
            ]
        )

    mock_session._calculate_quali_like_session_results.side_effect = _recompute_results

    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=mock_session):
        results = fetch_actual_session_results(2026, "Australian Grand Prix", "Q")

    assert results is not None
    assert [entry["driver"] for entry in results[-6:]] == ["ALO", "PER", "BOT", "VER", "STR", "SAI"]
    assert [entry["position"] for entry in results[-6:]] == [17, 18, 19, 20, 21, 22]
