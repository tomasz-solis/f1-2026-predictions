"""Post-extraction validation for data coming from FastF1 sessions.

Catches out-of-range values, missing fields, and schema drift before
extracted data reaches the prediction pipeline.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Expected bounds for extracted F1 data
_POSITION_MIN = 1
_POSITION_MAX = 22
_LAP_TIME_MIN_S = 55.0  # fastest ever ~1:02 (Monaco) with margin
_LAP_TIME_MAX_S = 180.0  # slow outlier / safety car restart

_MIN_TEAMS = 5
_MAX_TEAMS = 11


class ExtractionValidationError(ValueError):
    """Raised when extracted data fails sanity checks."""


def validate_session_positions(
    positions: dict[str, int],
    *,
    context: str = "",
) -> list[str]:
    """Validate extracted position data. Returns list of warnings (empty = clean)."""
    warnings: list[str] = []
    prefix = f"[{context}] " if context else ""

    if not positions:
        warnings.append(f"{prefix}No positions extracted")
        return warnings

    for driver, pos in positions.items():
        if not isinstance(pos, int | float):
            warnings.append(f"{prefix}{driver}: position is not numeric ({pos!r})")
        elif pos < _POSITION_MIN or pos > _POSITION_MAX:
            warnings.append(
                f"{prefix}{driver}: position {pos} outside {_POSITION_MIN}-{_POSITION_MAX}"
            )
    return warnings


def validate_team_pace_data(
    team_pace: dict[str, Any] | None,
    *,
    context: str = "",
) -> list[str]:
    """Validate extracted team pace data from FP sessions."""
    warnings: list[str] = []
    prefix = f"[{context}] " if context else ""

    if team_pace is None:
        return warnings  # None is a valid "no data" signal

    num_teams = len(team_pace)
    if num_teams < _MIN_TEAMS:
        warnings.append(f"{prefix}Only {num_teams} teams extracted (expected >= {_MIN_TEAMS})")
    if num_teams > _MAX_TEAMS:
        warnings.append(f"{prefix}{num_teams} teams extracted (expected <= {_MAX_TEAMS})")

    for team, data in team_pace.items():
        if isinstance(data, dict):
            avg_pace = data.get("avg_pace")
            if avg_pace is not None:
                if not isinstance(avg_pace, int | float):
                    warnings.append(f"{prefix}{team}: avg_pace is not numeric ({avg_pace!r})")
                elif avg_pace < _LAP_TIME_MIN_S or avg_pace > _LAP_TIME_MAX_S:
                    warnings.append(
                        f"{prefix}{team}: avg_pace {avg_pace:.1f}s outside "
                        f"{_LAP_TIME_MIN_S}-{_LAP_TIME_MAX_S}s range"
                    )

            degradation = data.get("degradation")

            if degradation is not None and isinstance(degradation, int | float):
                if degradation < 0:
                    warnings.append(f"{prefix}{team}: negative degradation ({degradation:.3f})")
                elif degradation > 1.0:
                    warnings.append(
                        f"{prefix}{team}: degradation {degradation:.3f}s/lap unusually high"
                    )

    return warnings


def validate_fp_team_order(
    team_ranks: dict[str, int],
    *,
    context: str = "",
) -> list[str]:
    """Validate team ordering from FP sessions."""
    warnings: list[str] = []
    prefix = f"[{context}] " if context else ""

    if not team_ranks:
        warnings.append(f"{prefix}No team ranks extracted")
        return warnings

    ranks = list(team_ranks.values())
    if len(set(ranks)) != len(ranks):
        warnings.append(f"{prefix}Duplicate rank values detected: {sorted(ranks)}")

    expected_ranks = set(range(1, len(team_ranks) + 1))
    actual_ranks = set(ranks)
    if actual_ranks != expected_ranks:
        warnings.append(f"{prefix}Non-contiguous ranks: got {sorted(actual_ranks)}")

    return warnings


def log_validation_warnings(warnings: list[str]) -> None:
    """Log extraction validation warnings. Doesn't raise -- just alerts."""
    for w in warnings:
        logger.warning("Extraction validation: %s", w)
