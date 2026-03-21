"""Schema checks for persisted predictor artifacts."""

import logging
from typing import Any

try:
    from jsonschema import ValidationError, validate
except ImportError:
    validate = None
    ValidationError = Exception

logger = logging.getLogger(__name__)

_YEAR_VALUE = {"type": "integer", "minimum": 2020, "maximum": 2030}
_HISTORICAL_YEAR_VALUE = {"type": "integer", "minimum": 1950, "maximum": 2030}
_STRING_OR_NULL = {"type": ["string", "null"]}
_UNIT_INTERVAL_NUMBER = {"type": "number", "minimum": 0.0, "maximum": 1.0}
_NULLABLE_UNIT_INTERVAL_NUMBER = {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0}
_TRACK_PROFILE_PERCENT_KEYS = (
    "straights_pct",
    "slow_corners_pct",
    "medium_corners_pct",
    "high_corners_pct",
)

_DRIVER_EXPERIENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "debut_year": _HISTORICAL_YEAR_VALUE,
        "years_of_experience": {"type": "integer", "minimum": 0},
        "total_seasons": {"type": "integer", "minimum": 0},
        "total_races": {"type": "integer", "minimum": 0},
        "tier": {"type": "string"},
    },
    "additionalProperties": False,
}

_DRIVER_PACE_SCHEMA = {
    "type": "object",
    "required": ["quali_pace", "race_pace"],
    "properties": {
        "quali_pace": _UNIT_INTERVAL_NUMBER,
        "quali_std": {"type": "number", "minimum": 0.0},
        "race_pace": _UNIT_INTERVAL_NUMBER,
        "race_std": {"type": "number", "minimum": 0.0},
        "confidence": {"type": "string"},
    },
    "additionalProperties": False,
}

_DRIVER_RACECRAFT_SCHEMA = {
    "type": "object",
    "required": ["skill_score", "overtaking_skill"],
    "properties": {
        "skill_score": _UNIT_INTERVAL_NUMBER,
        "overtaking_skill": _UNIT_INTERVAL_NUMBER,
        "defensive_skill": _UNIT_INTERVAL_NUMBER,
        "driver_type": {"type": "string"},
        "interpretation": {"type": "string"},
        "races_analyzed": {"type": "integer", "minimum": 0},
        "total_dnfs": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

_DRIVER_DNF_SCHEMA = {
    "type": "object",
    "required": ["dnf_rate"],
    "properties": {
        "dnf_rate": _UNIT_INTERVAL_NUMBER,
        "risk_level": {"type": "string"},
        "total_races": {"type": "integer", "minimum": 0},
        "total_dnfs": {"type": "integer", "minimum": 0},
        "dnf_types": {"type": "object"},
    },
    "additionalProperties": False,
}

_TIRE_MANAGEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "skill": _UNIT_INTERVAL_NUMBER,
        "skill_score": _UNIT_INTERVAL_NUMBER,
        "baseline": {"type": "string"},
        "notes": {"type": "string"},
    },
    "additionalProperties": False,
}

_DRIVER_BAYESIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "rating_mu": {"type": "number"},
        "rating_sigma": {"type": "number", "minimum": 0.0},
        "normalized_skill_score": _UNIT_INTERVAL_NUMBER,
        "blended_skill_score": _UNIT_INTERVAL_NUMBER,
        "blend_weight": _UNIT_INTERVAL_NUMBER,
        "last_session": {"type": "string"},
        "last_updated": _STRING_OR_NULL,
        "season_year": _YEAR_VALUE,
    },
    "additionalProperties": False,
}

_DRIVER_ENTRY_SCHEMA = {
    "type": "object",
    "required": ["racecraft", "pace", "dnf_risk"],
    "properties": {
        "name": {"type": "string"},
        "number": {"type": "integer"},
        "teams": {"type": "array", "items": {"type": "string"}},
        "experience": _DRIVER_EXPERIENCE_SCHEMA,
        "pace": _DRIVER_PACE_SCHEMA,
        "racecraft": _DRIVER_RACECRAFT_SCHEMA,
        "dnf_risk": _DRIVER_DNF_SCHEMA,
        "tire_management": _TIRE_MANAGEMENT_SCHEMA,
        "bayesian": _DRIVER_BAYESIAN_SCHEMA,
    },
    "additionalProperties": False,
}

_DIRECTIONALITY_SCHEMA = {
    "type": "object",
    "required": ["max_speed", "slow_corner_speed", "medium_corner_speed", "high_corner_speed"],
    "properties": {
        "max_speed": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "slow_corner_speed": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "medium_corner_speed": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "high_corner_speed": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

_TESTING_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "slow_corner_performance": _UNIT_INTERVAL_NUMBER,
        "medium_corner_performance": _UNIT_INTERVAL_NUMBER,
        "fast_corner_performance": _UNIT_INTERVAL_NUMBER,
        "braking_performance": _UNIT_INTERVAL_NUMBER,
        "top_speed": _UNIT_INTERVAL_NUMBER,
        "overall_pace": _UNIT_INTERVAL_NUMBER,
        "consistency": _UNIT_INTERVAL_NUMBER,
        "tire_deg_performance": _UNIT_INTERVAL_NUMBER,
        "consistency_performance": _UNIT_INTERVAL_NUMBER,
        "tire_deg_slope": {"type": "number", "minimum": -2.0, "maximum": 2.0},
        "sessions_used": {"type": "number", "minimum": 0.0},
        "session_aggregation": {"type": "string"},
        "run_profile": {"type": "string", "enum": ["balanced", "short_run", "long_run"]},
        "last_updated": _STRING_OR_NULL,
    },
    "additionalProperties": False,
}

_COMPOUND_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "track_name": {"type": "string"},
        "pace_performance": _NULLABLE_UNIT_INTERVAL_NUMBER,
        "tire_deg_performance": _NULLABLE_UNIT_INTERVAL_NUMBER,
        "consistency_performance": _NULLABLE_UNIT_INTERVAL_NUMBER,
        "tire_deg_slope": {"type": ["number", "null"], "minimum": -5.0, "maximum": 5.0},
        "consistency": {"type": "number", "minimum": 0.0},
        "median_lap_time": {"type": "number", "exclusiveMinimum": 0.0},
        "laps_sampled": {"type": "number", "minimum": 0.0},
        "sessions_used": {"type": "number", "minimum": 0.0},
        "last_updated": _STRING_OR_NULL,
    },
    "additionalProperties": False,
}


DRIVER_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "required": ["drivers"],
    "properties": {
        "year": _YEAR_VALUE,
        "version": {"type": "integer", "minimum": 0},
        "method": {"type": "string"},
        "note": {"type": "string"},
        "last_updated": _STRING_OR_NULL,
        "extraction_date": _STRING_OR_NULL,
        "carried_over_from": _YEAR_VALUE,
        "bayesian_last_updated_year": _YEAR_VALUE,
        "years": {"type": "array", "items": _YEAR_VALUE},
        "drivers": {
            "type": "object",
            "patternProperties": {
                "^[A-Z]{3}$": _DRIVER_ENTRY_SCHEMA,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


TEAM_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "required": ["year", "teams"],
    "properties": {
        "year": _YEAR_VALUE,
        "version": {"type": "integer", "minimum": 0},
        "data_freshness": {"type": "string"},
        "note": {"type": "string"},
        "learning_note": {"type": "string"},
        "generated_at": {"type": "string"},
        "last_updated": _STRING_OR_NULL,
        "races_completed": {"type": "integer", "minimum": 0},
        "directionality_source": {"type": "string"},
        "directionality_last_updated": _STRING_OR_NULL,
        "directionality_meta": {"type": "object"},
        "teams": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "required": ["overall_performance"],
                    "properties": {
                        "overall_performance": _UNIT_INTERVAL_NUMBER,
                        "uncertainty": _UNIT_INTERVAL_NUMBER,
                        "note": {"type": "string"},
                        "last_updated": _STRING_OR_NULL,
                        "races_completed": {"type": "integer", "minimum": 0},
                        "current_season_performance": {
                            "type": "array",
                            "items": _UNIT_INTERVAL_NUMBER,
                        },
                        "directionality": _DIRECTIONALITY_SCHEMA,
                        "testing_characteristics": _TESTING_CHARACTERISTICS_SCHEMA,
                        "testing_characteristics_profiles": {
                            "type": "object",
                            "propertyNames": {
                                "enum": ["balanced", "short_run", "long_run"],
                            },
                            "additionalProperties": _TESTING_CHARACTERISTICS_SCHEMA,
                        },
                        "compound_characteristics": {
                            "type": "object",
                            "propertyNames": {"pattern": "^[A-Z_]+$"},
                            "additionalProperties": _COMPOUND_CHARACTERISTICS_SCHEMA,
                        },
                        "drivers": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Z0-9_]+$": _UNIT_INTERVAL_NUMBER,
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


TRACK_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "required": ["year", "tracks"],
    "properties": {
        "year": _YEAR_VALUE,
        "note": {"type": "string"},
        "generated_at": {"type": "string"},
        "generated_from": {"type": "string"},
        "data_freshness": {"type": "string"},
        "tracks": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "required": [
                        "pit_stop_loss",
                        "safety_car_prob",
                        "overtaking_difficulty",
                    ],
                    "properties": {
                        "pit_stop_loss": {"type": "number", "minimum": 10.0, "maximum": 40.0},
                        "safety_car_prob": _UNIT_INTERVAL_NUMBER,
                        "overtaking_difficulty": _UNIT_INTERVAL_NUMBER,
                        "lap1_risk_modifier": _UNIT_INTERVAL_NUMBER,
                        "type": {"type": "string", "enum": ["permanent", "street"]},
                        "has_sprint": {"type": "boolean"},
                        "overtaking_likelihood": _UNIT_INTERVAL_NUMBER,
                        "overtaking_avg_changes_per_lap": {"type": "number", "minimum": 0.0},
                        "overtaking_years_analyzed": {"type": "integer", "minimum": 0},
                        "overtaking_observed_races": {"type": "integer", "minimum": 0},
                        "straights_pct": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        "slow_corners_pct": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        "medium_corners_pct": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        "high_corners_pct": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def validate_json(data: dict[str, Any], schema: dict[str, Any], filename: str) -> None:
    """Validate a payload against a JSON schema."""
    if validate is None:
        logger.warning(
            f"jsonschema library not available. Skipping validation of {filename}. "
            "Install jsonschema to enable validation: pip install jsonschema"
        )
        return

    try:
        validate(instance=data, schema=schema)
        logger.info(f"{filename} validated successfully")
    except ValidationError as exc:
        error_msg = f"Invalid {filename}: {str(exc)}"
        logger.error(f"{filename} validation failed: {error_msg}")
        raise ValueError(error_msg) from exc
    except (AttributeError, TypeError, KeyError, ValueError) as exc:
        error_msg = f"Unexpected validation error in {filename}: {str(exc)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from exc


def _validate_expected_year(
    data: dict[str, Any],
    filename: str,
    expected_year: int | None,
    *,
    allow_missing: bool = False,
) -> None:
    """Ensure a payload matches the season we intended to load."""
    if expected_year is None:
        return

    payload_year = data.get("year")
    if payload_year is None:
        if allow_missing:
            return
        raise ValueError(f"Invalid {filename}: missing year for expected season {expected_year}")

    if int(payload_year) != int(expected_year):
        raise ValueError(
            f"Invalid {filename}: payload year {payload_year} does not match expected "
            f"season {expected_year}"
        )


def _validate_driver_season_alignment(
    data: dict[str, Any],
    expected_year: int | None,
) -> None:
    """Cross-check driver season markers when top-level year is absent."""
    if expected_year is None:
        return

    drivers = data.get("drivers", {})
    if not isinstance(drivers, dict):
        return

    season_years: set[int] = set()
    for driver_data in drivers.values():
        if not isinstance(driver_data, dict):
            continue
        bayesian = driver_data.get("bayesian", {})
        if not isinstance(bayesian, dict):
            continue
        season_year = bayesian.get("season_year")
        if season_year is None:
            continue
        season_years.add(int(season_year))

    if season_years and season_years != {int(expected_year)}:
        raise ValueError(
            "Invalid driver_characteristics.json: "
            f"bayesian season years {sorted(season_years)} do not match expected season "
            f"{expected_year}"
        )


def _validate_track_profile_percentages(data: dict[str, Any]) -> None:
    """Require complete track-composition percentages when any are present."""
    tracks = data.get("tracks", {})
    if not isinstance(tracks, dict):
        return

    for track_name, track_data in tracks.items():
        if not isinstance(track_data, dict):
            continue

        present_keys = [key for key in _TRACK_PROFILE_PERCENT_KEYS if key in track_data]
        if not present_keys:
            continue

        if len(present_keys) != len(_TRACK_PROFILE_PERCENT_KEYS):
            missing_keys = [key for key in _TRACK_PROFILE_PERCENT_KEYS if key not in track_data]
            raise ValueError(
                "Invalid track_characteristics.json: "
                f"{track_name} is missing track profile fields {missing_keys}"
            )

        total = sum(float(track_data[key]) for key in _TRACK_PROFILE_PERCENT_KEYS)
        if total <= 0.0:
            raise ValueError(
                "Invalid track_characteristics.json: "
                f"{track_name} track profile percentages must sum to a positive value"
            )


def validate_driver_characteristics(
    data: dict[str, Any],
    *,
    expected_year: int | None = None,
) -> None:
    """Validate driver characteristics JSON."""
    validate_json(data, DRIVER_CHARACTERISTICS_SCHEMA, "driver_characteristics.json")
    _validate_expected_year(
        data,
        "driver_characteristics.json",
        expected_year,
        allow_missing=True,
    )
    _validate_driver_season_alignment(data, expected_year)


def validate_team_characteristics(
    data: dict[str, Any],
    *,
    expected_year: int | None = None,
) -> None:
    """Validate team characteristics JSON."""
    validate_json(data, TEAM_CHARACTERISTICS_SCHEMA, "team_characteristics.json")
    _validate_expected_year(data, "team_characteristics.json", expected_year)


def validate_track_characteristics(
    data: dict[str, Any],
    *,
    expected_year: int | None = None,
) -> None:
    """Validate track characteristics JSON."""
    validate_json(data, TRACK_CHARACTERISTICS_SCHEMA, "track_characteristics.json")
    _validate_expected_year(data, "track_characteristics.json", expected_year)
    _validate_track_profile_percentages(data)
