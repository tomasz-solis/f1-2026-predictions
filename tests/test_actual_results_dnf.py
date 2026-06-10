"""Tests for DNF extraction in actual race results."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.data.actual_results_fetcher as arf
from src.data.actual_results_fetcher import (
    _result_row_is_dnf,
    fetch_actual_session_results,
)
from src.utils.grid_validation import validate_qualifying_grid


def test_result_row_is_dnf_classified_position():
    assert _result_row_is_dnf({"ClassifiedPosition": "1", "Status": "Finished"}) is False
    assert _result_row_is_dnf({"ClassifiedPosition": "12", "Status": "+1 Lap"}) is False
    assert _result_row_is_dnf({"ClassifiedPosition": "R", "Status": "Accident"}) is True
    assert _result_row_is_dnf({"ClassifiedPosition": "D", "Status": "Disqualified"}) is True


def test_result_row_is_dnf_status_fallback():
    # No classified position -> fall back to Status string.
    assert _result_row_is_dnf({"Status": "Finished"}) is False
    assert _result_row_is_dnf({"Status": "+2 Laps"}) is False
    assert _result_row_is_dnf({"Status": "Lapped"}) is False
    assert _result_row_is_dnf({"Status": "Engine"}) is True
    assert _result_row_is_dnf({"Status": "Collision"}) is True
    # No signal at all -> assume finished.
    assert _result_row_is_dnf({}) is False
    assert _result_row_is_dnf({"ClassifiedPosition": float("nan")}) is False


def test_validate_qualifying_grid_preserves_dnf():
    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1, "dnf": False},
        {"driver": "NOR", "team": "McLaren", "position": 2, "dnf": True},
    ]
    validated = validate_qualifying_grid(grid, min_entries=2, require_sequential_positions=True)
    assert validated[0]["dnf"] is False
    assert validated[1]["dnf"] is True


def _race_results_frame(n: int = 20, dnf_positions: tuple[int, ...] = (19, 20)) -> pd.DataFrame:
    rows = []
    for pos in range(1, n + 1):
        is_dnf = pos in dnf_positions
        rows.append(
            {
                "Abbreviation": f"D{pos:02d}",
                "TeamName": f"Team{(pos + 1) // 2}",
                "Position": float(pos),
                "ClassifiedPosition": "R" if is_dnf else str(pos),
                "Status": "Retired" if is_dnf else ("Finished" if pos == 1 else "+1 Lap"),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def _direct_resilience(monkeypatch):
    monkeypatch.setattr(arf, "call_with_resilience", lambda _name, fn, labels=None: fn())
    monkeypatch.setattr(arf, "map_team_to_characteristics", lambda name: name)


def test_fetch_race_results_includes_dnf(monkeypatch, _direct_resilience):
    session = MagicMock()
    session.results = _race_results_frame()
    session.date = datetime.now(tz=UTC)
    monkeypatch.setattr(arf.fastf1, "get_session", lambda *a, **k: session)

    grid = fetch_actual_session_results(2026, "Chinese Grand Prix", "R")
    assert grid is not None
    by_driver = {row["driver"]: row for row in grid}
    assert all("dnf" in row for row in grid)
    assert by_driver["D19"]["dnf"] is True
    assert by_driver["D20"]["dnf"] is True
    assert by_driver["D01"]["dnf"] is False
    assert sum(1 for row in grid if row["dnf"]) == 2


def test_fetch_qualifying_results_has_no_dnf_field(monkeypatch, _direct_resilience):
    frame = _race_results_frame(dnf_positions=())
    session = MagicMock()
    session.results = frame
    session.date = datetime.now(tz=UTC)
    monkeypatch.setattr(arf.fastf1, "get_session", lambda *a, **k: session)

    grid = fetch_actual_session_results(2026, "Chinese Grand Prix", "Q")
    assert grid is not None
    assert all("dnf" not in row for row in grid)
