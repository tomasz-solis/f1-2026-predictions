"""Tests for the actual post-penalty starting grid."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.actual_results_fetcher import fetch_actual_starting_grid


def _results(overrides: dict[str, dict] | None = None) -> pd.DataFrame:
    """Build a 20-car race result whose grid matches classification unless overridden."""
    rows = [
        {
            "Abbreviation": f"D{i:02d}",
            "TeamName": "McLaren Formula 1 Team",
            "Position": float(i),
            "GridPosition": float(i),
        }
        for i in range(1, 21)
    ]
    for driver, patch_values in (overrides or {}).items():
        for row in rows:
            if row["Abbreviation"] == driver:
                row.update(patch_values)
    return pd.DataFrame(rows)


def _fetch(results: pd.DataFrame):
    session = MagicMock()
    session.results = results
    with patch("src.data.actual_results_fetcher.fastf1.get_session", return_value=session):
        return fetch_actual_starting_grid(2026, "Belgian Grand Prix")


def test_penalised_driver_takes_his_grid_slot_not_his_qualifying_place():
    # Qualified P2, ten-place penalty: starts P12, and the cars between close up.
    overrides = {"D02": {"GridPosition": 12.0}}
    overrides.update({f"D{i:02d}": {"GridPosition": float(i - 1)} for i in range(3, 13)})

    grid = _fetch(_results(overrides))

    assert grid is not None
    assert [row["position"] for row in grid] == list(range(1, 21))
    by_driver = {row["driver"]: row["position"] for row in grid}
    assert by_driver["D02"] == 12
    assert by_driver["D03"] == 2
    assert {row["start_type"] for row in grid} == {"grid"}


def test_pit_lane_starter_holds_no_grid_slot_and_lines_up_behind():
    # FastF1 reports a pit-lane start as grid slot zero.
    overrides = {"D05": {"GridPosition": 0.0}}
    overrides.update({f"D{i:02d}": {"GridPosition": float(i - 1)} for i in range(6, 21)})

    grid = _fetch(_results(overrides))

    assert grid is not None
    assert grid[-1]["driver"] == "D05"
    assert grid[-1]["start_type"] == "pit_lane"
    assert [row["position"] for row in grid] == list(range(1, 21))


def test_a_grid_that_does_not_reconcile_fails_closed():
    # Two cars cannot share slot 4.
    grid = _fetch(_results({"D05": {"GridPosition": 4.0}}))
    assert grid is None


def test_a_missing_grid_position_fails_closed():
    grid = _fetch(_results({"D07": {"GridPosition": None}}))
    assert grid is None


def test_a_race_that_has_not_run_returns_none():
    assert _fetch(pd.DataFrame([])) is None
