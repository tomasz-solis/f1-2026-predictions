"""Helpers for attaching paired-grid context to race prediction sections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def attach_starting_grid_context(
    race_section: dict[str, Any],
    starting_grid: Sequence[Mapping[str, Any]] | None,
    starting_session_name: str,
) -> dict[str, Any]:
    """Attach the grid used to compare race start and projected finish."""
    if not starting_grid:
        return race_section

    race_section["starting_grid"] = [dict(row) for row in starting_grid]
    race_section["starting_session_name"] = str(starting_session_name).strip().upper()
    return race_section
