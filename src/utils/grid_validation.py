"""Validation helpers for qualifying grid structures."""

from collections.abc import Mapping, Sequence
from typing import Any

from src.types.prediction_types import QualifyingGridEntry
from src.utils.validation_helpers import validate_position


def validate_qualifying_grid(
    grid: Sequence[QualifyingGridEntry | Mapping[str, Any]],
    *,
    min_entries: int = 1,
    require_sequential_positions: bool = False,
) -> list[QualifyingGridEntry]:
    """Validate and normalize a qualifying grid payload.

    Raises:
        ValueError: If grid structure is invalid.
    """
    if not grid:
        raise ValueError("Grid cannot be empty")

    validated_grid: list[QualifyingGridEntry] = []
    seen_positions: set[int] = set()
    seen_drivers: set[str] = set()

    for entry in grid:
        if not isinstance(entry, dict):
            raise ValueError(f"Grid entry must be a dict, got {type(entry).__name__}")

        if not all(field in entry for field in ("driver", "team", "position")):
            raise ValueError(f"Grid entry missing required keys: {entry}")

        driver = str(entry["driver"]).strip()
        team = str(entry["team"]).strip()
        position = entry["position"]

        if not driver:
            raise ValueError("Grid entry driver cannot be empty")
        if not team:
            raise ValueError("Grid entry team cannot be empty")

        validate_position(position, "position", min_pos=1, max_pos=22)

        if position in seen_positions:
            raise ValueError(f"Duplicate position {position} in grid")
        if driver in seen_drivers:
            raise ValueError(f"Duplicate driver {driver} in grid")

        seen_positions.add(position)
        seen_drivers.add(driver)

        validated_entry: QualifyingGridEntry = {
            "driver": driver,
            "team": team,
            "position": int(position),
        }
        validated_grid.append(validated_entry)

    if len(validated_grid) < min_entries:
        raise ValueError(
            f"Grid must include at least {min_entries} entries, got {len(validated_grid)}"
        )

    if require_sequential_positions:
        sorted_positions = sorted(entry["position"] for entry in validated_grid)
        expected_positions = list(range(1, len(validated_grid) + 1))
        if sorted_positions != expected_positions:
            raise ValueError(
                "Grid positions must be sequential starting at 1 "
                f"(got {sorted_positions}, expected {expected_positions})"
            )

    return validated_grid
