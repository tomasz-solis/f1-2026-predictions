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


def test_validate_qualifying_grid_preserves_uncertainty_fields():
    """Predicted-grid uncertainty fields should survive validation."""
    grid = [
        {
            "driver": "NOR",
            "team": "McLaren",
            "position": 1,
            "median_position": 2,
            "p5": 1,
            "p95": 6,
            "confidence": 48.5,
        },
        {
            "driver": "RUS",
            "team": "Mercedes",
            "position": 2,
            "median_position": 3,
            "p5": 1,
            "p95": 7,
            "confidence": 47.9,
        },
    ]

    validated = validate_qualifying_grid(grid, min_entries=2, require_sequential_positions=True)

    assert validated[0]["median_position"] == 2
    assert validated[0]["p5"] == 1
    assert validated[0]["p95"] == 6
    assert validated[0]["confidence"] == pytest.approx(48.5)


def test_validate_qualifying_grid_requires_positions_inside_interval():
    grid = [
        {
            "driver": "NOR",
            "team": "McLaren",
            "position": 1,
            "median_position": 3,
            "p5": 1,
            "p95": 2,
        },
        {"driver": "RUS", "team": "Mercedes", "position": 2},
    ]

    with pytest.raises(ValueError, match="median_position"):
        validate_qualifying_grid(grid, min_entries=2)

    grid[0]["median_position"] = 1
    grid[0]["position"] = 3
    with pytest.raises(ValueError, match="position"):
        validate_qualifying_grid(grid, min_entries=2)


def test_validate_qualifying_grid_respects_custom_max_position():
    grid = [{"driver": "VER", "team": "Red Bull Racing", "position": 23}]

    with pytest.raises(ValueError, match="position"):
        validate_qualifying_grid(grid, max_position=22)


def test_validate_qualifying_grid_preserves_start_type():
    """Official start metadata must survive validation.

    The validator rebuilds each entry from an allow-list of known keys, so a field it
    does not name is silently dropped. Replay consumers that reconstruct a real race
    require start_type on every row and fail closed without it.
    """
    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1, "start_type": "grid"},
        {"driver": "HAM", "team": "Ferrari", "position": 2, "start_type": "pit_lane"},
    ]

    validated = validate_qualifying_grid(grid, min_entries=2)

    assert [row["start_type"] for row in validated] == ["grid", "pit_lane"]


def test_validate_qualifying_grid_omits_absent_start_type():
    grid = [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]

    assert "start_type" not in validate_qualifying_grid(grid)[0]


def test_validate_qualifying_grid_rejects_blank_start_type():
    grid = [{"driver": "VER", "team": "Red Bull Racing", "position": 1, "start_type": "  "}]

    with pytest.raises(ValueError, match="start_type"):
        validate_qualifying_grid(grid)
