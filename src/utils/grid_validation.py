"""Validation helpers for qualifying grid structures."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.types.prediction_types import QualifyingGridEntry
from src.utils import config_loader
from src.utils.validation_helpers import validate_position


def validate_qualifying_grid(
    grid: Sequence[QualifyingGridEntry | Mapping[str, Any]],
    *,
    min_entries: int = 1,
    require_sequential_positions: bool = False,
    max_position: int | None = None,
) -> list[QualifyingGridEntry]:
    """Validate and normalize a qualifying grid payload.

    Raises:
        ValueError: If grid structure is invalid.
    """
    if not grid:
        raise ValueError("Grid cannot be empty")

    if max_position is None:
        max_position = int(config_loader.get("grid.size", 22))
    max_position = max(int(max_position), 1)

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

        validate_position(position, "position", min_pos=1, max_pos=max_position)

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

        if "start_type" in entry and entry["start_type"] is not None:
            start_type = str(entry["start_type"]).strip()
            if not start_type:
                raise ValueError(f"Grid entry start_type cannot be empty for {driver}")
            validated_entry["start_type"] = start_type

        if "qualifying_position" in entry and entry["qualifying_position"] is not None:
            validate_position(
                entry["qualifying_position"],
                "qualifying_position",
                min_pos=1,
                max_pos=max_position,
            )
            validated_entry["qualifying_position"] = int(entry["qualifying_position"])

        if "median_position" in entry and entry["median_position"] is not None:
            validate_position(
                entry["median_position"],
                "median_position",
                min_pos=1,
                max_pos=max_position,
            )
            validated_entry["median_position"] = int(entry["median_position"])

        if "p5" in entry and entry["p5"] is not None:
            validate_position(entry["p5"], "p5", min_pos=1, max_pos=max_position)
            validated_entry["p5"] = int(entry["p5"])

        if "p95" in entry and entry["p95"] is not None:
            validate_position(entry["p95"], "p95", min_pos=1, max_pos=max_position)
            validated_entry["p95"] = int(entry["p95"])

        if "p5" in validated_entry and "p95" in validated_entry:
            if int(validated_entry["p95"]) < int(validated_entry["p5"]):
                raise ValueError(
                    "Grid percentile positions must satisfy p5 <= p95 "
                    f"(got p5={validated_entry['p5']}, p95={validated_entry['p95']})"
                )
            lower = int(validated_entry["p5"])
            upper = int(validated_entry["p95"])
            if "median_position" in validated_entry:
                median_position = int(validated_entry["median_position"])
                if not lower <= median_position <= upper:
                    raise ValueError(
                        "Grid median_position must lie inside p5-p95 interval "
                        f"(got median_position={median_position}, p5={lower}, p95={upper})"
                    )
            final_position = int(validated_entry["position"])
            if not lower <= final_position <= upper:
                raise ValueError(
                    "Grid position must lie inside p5-p95 interval "
                    f"(got position={final_position}, p5={lower}, p95={upper})"
                )

        if "confidence" in entry and entry["confidence"] is not None:
            try:
                confidence = float(entry["confidence"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Grid confidence must be numeric, got {entry['confidence']!r}"
                ) from exc

            if not math.isfinite(confidence) or confidence < 0.0:
                raise ValueError(
                    "Grid confidence must be a finite non-negative number "
                    f"(got {entry['confidence']!r})"
                )

            validated_entry["confidence"] = confidence

        if "order_confidence" in entry and entry["order_confidence"] is not None:
            try:
                order_confidence = float(entry["order_confidence"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Grid order_confidence must be numeric, got {entry['order_confidence']!r}"
                ) from exc

            if not math.isfinite(order_confidence) or order_confidence < 0.0:
                raise ValueError(
                    "Grid order_confidence must be a finite non-negative number "
                    f"(got {entry['order_confidence']!r})"
                )

            validated_entry["order_confidence"] = order_confidence

        if "dnf" in entry and entry["dnf"] is not None:
            validated_entry["dnf"] = bool(entry["dnf"])

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
