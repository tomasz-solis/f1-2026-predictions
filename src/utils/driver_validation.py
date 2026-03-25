"""Validate driver characteristics for obvious extraction mistakes."""

import logging

logger = logging.getLogger(__name__)


def validate_driver_data(drivers: dict) -> list[str]:
    """Validate driver characteristics and return any error messages."""
    errors = []

    for driver_code, driver_data in drivers.items():
        # Check required fields exist
        if "racecraft" not in driver_data:
            errors.append(f"{driver_code}: Missing 'racecraft' field")
            continue

        if "pace" not in driver_data:
            errors.append(f"{driver_code}: Missing 'pace' field")
            continue

        # Get values
        skill = driver_data.get("racecraft", {}).get("skill_score")
        quali_pace = driver_data.get("pace", {}).get("quali_pace")
        race_pace = driver_data.get("pace", {}).get("race_pace")
        dnf_rate = driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10)

        # Sanity checks (these indicate extraction bugs, not natural variation)
        if skill is not None:
            if skill < 0.01 or skill > 0.99:
                errors.append(
                    f"{driver_code}: skill_score {skill:.3f} out of valid range [0.01, 0.99]"
                )

        if quali_pace is not None and race_pace is not None:
            # Quali/race should be similar (within 20%)
            if abs(quali_pace - race_pace) > 0.20:
                errors.append(
                    f"{driver_code}: Large pace gap (quali {quali_pace:.3f}, race {race_pace:.3f})"
                )

        if dnf_rate < 0.0 or dnf_rate > 0.50:
            errors.append(f"{driver_code}: DNF rate {dnf_rate:.3f} unrealistic (should be 0-50%)")

    if errors:
        logger.warning(f"Found {len(errors)} validation errors:")
        for error in errors[:10]:  # Show first 10
            logger.warning(f"   - {error}")
        if len(errors) > 10:
            logger.warning(f"   ... and {len(errors) - 10} more")

    return errors
