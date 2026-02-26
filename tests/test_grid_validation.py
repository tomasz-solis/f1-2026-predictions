"""Tests for strict qualifying grid validation helpers."""

import pytest

from src.utils.grid_validation import validate_qualifying_grid


def test_validate_qualifying_grid_supports_min_entries_guard():
    grid = [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]

    with pytest.raises(ValueError, match="at least 2 entries"):
        validate_qualifying_grid(grid, min_entries=2)


def test_validate_qualifying_grid_requires_sequential_positions_when_enabled():
    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "LEC", "team": "Ferrari", "position": 3},
    ]

    with pytest.raises(ValueError, match="sequential starting at 1"):
        validate_qualifying_grid(grid, require_sequential_positions=True)


def test_validate_qualifying_grid_accepts_valid_sequential_grid():
    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "LEC", "team": "Ferrari", "position": 2},
    ]

    validated = validate_qualifying_grid(grid, min_entries=2, require_sequential_positions=True)
    assert [entry["position"] for entry in validated] == [1, 2]
