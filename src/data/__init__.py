"""Data domain package for track loading, results fetching, and compound helpers."""

from .compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from .compound_validator import (
    load_and_validate_compound_data,
    validate_compound_data,
    validate_pirelli_info,
)
from .data_generator import create_baseline_if_missing
from .track_data_loader import (
    KNOWN_MAIN_RACE_LAPS,
    KNOWN_SPRINT_LAPS,
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_non_competitive_weather_features,
    resolve_race_distance_laps,
    resolve_track_temperature_c,
    resolve_track_temperature_profile,
)

__all__ = [
    "create_baseline_if_missing",
    "KNOWN_MAIN_RACE_LAPS",
    "KNOWN_SPRINT_LAPS",
    "get_available_compounds",
    "get_compound_performance_modifier",
    "get_tire_stress_score",
    "load_and_validate_compound_data",
    "load_track_specific_params",
    "resolve_non_competitive_weather_features",
    "resolve_race_distance_laps",
    "resolve_track_temperature_c",
    "resolve_track_temperature_profile",
    "should_use_compound_adjustments",
    "validate_compound_data",
    "validate_pirelli_info",
]
