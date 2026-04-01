"""Validation for compound characteristics data."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _collect_compound_data_errors(compound_data: dict) -> list[str]:
    """Collect validation errors for compound characteristics data."""
    errors = []

    if not isinstance(compound_data, dict):
        errors.append("Compound data must be a dictionary")
        return errors

    # Expected fields for each compound
    required_fields = {
        "degradation_rate",
        "optimal_stint_length",
        "pace_advantage",
    }

    # Validate each compound (C1-C5, SOFT, MEDIUM, HARD, etc.)
    for compound, data in compound_data.items():
        if not isinstance(data, dict):
            errors.append(f"{compound}: must be a dictionary")
            continue

        # Check required fields
        missing = required_fields - set(data.keys())
        if missing:
            errors.append(f"{compound}: missing fields {missing}")

        # Validate degradation_rate
        if "degradation_rate" in data:
            deg_rate = data["degradation_rate"]
            if not isinstance(deg_rate, int | float):
                errors.append(f"{compound}: degradation_rate must be numeric")
            elif deg_rate < 0:
                errors.append(f"{compound}: degradation_rate cannot be negative")
            elif deg_rate > 1.0:
                errors.append(f"{compound}: degradation_rate unusually high (>{1.0})")

        # Validate optimal_stint_length
        if "optimal_stint_length" in data:
            stint = data["optimal_stint_length"]
            if not isinstance(stint, int):
                errors.append(f"{compound}: optimal_stint_length must be integer")
            elif stint < 1:
                errors.append(f"{compound}: optimal_stint_length must be positive")
            elif stint > 100:
                errors.append(f"{compound}: optimal_stint_length unusually high (>100 laps)")

        # Validate pace_advantage
        if "pace_advantage" in data:
            pace = data["pace_advantage"]
            if not isinstance(pace, int | float):
                errors.append(f"{compound}: pace_advantage must be numeric")

    return errors


def validate_compound_data(compound_data: dict) -> None:
    """Validate compound characteristics data.

    Raises:
        ValueError: If validation fails.
    """
    errors = _collect_compound_data_errors(compound_data)
    if errors:
        raise ValueError("; ".join(errors))


validate_compound_data_or_raise = validate_compound_data


def _collect_pirelli_info_errors(pirelli_data: dict) -> list[str]:
    """Collect validation errors for Pirelli track tire stress data."""
    errors = []

    if not isinstance(pirelli_data, dict):
        errors.append("Pirelli data must be a dictionary")
        return errors

    # Each race should have tyre_stress field
    for race, data in pirelli_data.items():
        if not isinstance(data, dict):
            errors.append(f"{race}: must be a dictionary")
            continue

        if "tyre_stress" not in data:
            errors.append(f"{race}: missing tyre_stress field")
            continue

        stress = data["tyre_stress"]
        if not isinstance(stress, dict):
            errors.append(f"{race}: tyre_stress must be a dictionary")
            continue

        # Validate stress components
        expected_stress = {"traction", "braking", "lateral", "asphalt_abrasion"}
        for stress_type in expected_stress:
            if stress_type in stress:
                value = stress[stress_type]
                if not isinstance(value, int | float):
                    errors.append(f"{race}: tyre_stress.{stress_type} must be numeric")
                elif value < 0:
                    errors.append(f"{race}: tyre_stress.{stress_type} cannot be negative")
                elif value > 5:
                    errors.append(f"{race}: tyre_stress.{stress_type} unusually high (>5)")

    return errors


def validate_pirelli_info(pirelli_data: dict) -> None:
    """Validate Pirelli track tire stress dataset.

    Raises:
        ValueError: If validation fails.
    """
    errors = _collect_pirelli_info_errors(pirelli_data)
    if errors:
        raise ValueError("; ".join(errors))


validate_pirelli_info_or_raise = validate_pirelli_info


def load_and_validate_compound_data(file_path: Path) -> dict | None:
    """Load and validate compound data."""
    import json

    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", file_path, e)
        return None
    except FileNotFoundError:
        logger.debug("Compound file not found: %s", file_path)
        return None

    # Determine validation type based on structure
    try:
        if any("tyre_stress" in v for v in data.values() if isinstance(v, dict)):
            validate_pirelli_info(data)
        else:
            validate_compound_data(data)
    except ValueError as e:
        logger.error("Validation failed for %s: %s", file_path, e)
        return None

    return data
