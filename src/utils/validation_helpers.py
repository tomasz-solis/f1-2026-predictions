"""Input validation utilities for critical functions."""

import logging

logger = logging.getLogger(__name__)


def validate_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate value is within range [min_val, max_val]. Raises ValueError if outside."""
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    logger.debug("Validated %s=%s is in range [%s, %s]", name, value, min_val, max_val)


def validate_positive_int(value: int, name: str, min_val: int = 1) -> None:
    """Validate value is positive integer >= min_val. Raises ValueError if not."""
    if not isinstance(value, int) or value < min_val:
        raise ValueError(f"{name} must be an integer >= {min_val}, got {value}")
    logger.debug("Validated %s=%s is positive integer >= %s", name, value, min_val)


def validate_enum(value: str, name: str, valid_values: list[str]) -> None:
    """Validate value is one of allowed enum values. Raises ValueError if not."""
    if value not in valid_values:
        raise ValueError(f"{name} must be one of {valid_values}, got '{value}'")
    logger.debug("Validated %s='%s' is in %s", name, value, valid_values)


def validate_position(value: int | float, name: str, min_pos: int = 1, max_pos: int = 20) -> None:
    """Validate grid or race position is within range. Raises ValueError if outside."""
    # Convert to int if it's a float
    if isinstance(value, float):
        value = int(value)

    if not isinstance(value, int) or not (min_pos <= value <= max_pos):
        raise ValueError(f"{name} must be between {min_pos} and {max_pos}, got {value}")
    logger.debug("Validated %s=%s is valid position", name, value)


def validate_year(
    value: int, name: str = "year", min_year: int = 2020, max_year: int = 2030
) -> None:
    """Validate year is within allowed range. Raises ValueError if outside."""
    if not isinstance(value, int) or not (min_year <= value <= max_year):
        raise ValueError(f"{name} must be between {min_year} and {max_year}, got {value}")
    logger.debug("Validated %s=%s is in range [%s, %s]", name, value, min_year, max_year)
