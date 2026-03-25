"""Data domain package for track loading, results fetching, and compound helpers."""

from .compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from .compound_validator import (
    load_and_validate_compound_data,
    validate_compound_data,
    validate_compound_data_or_raise,
    validate_pirelli_info,
    validate_pirelli_info_or_raise,
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


def __getattr__(name: str):
    """Lazily import heavy modules on first access."""
    _LAZY = {
        "fetch_actual_session_results": (
            "src.data.actual_results_fetcher",
            "fetch_actual_session_results",
        ),
        "get_competitive_session_completion_state": (
            "src.data.actual_results_fetcher",
            "get_competitive_session_completion_state",
        ),
        "is_competitive_session_completed": (
            "src.data.actual_results_fetcher",
            "is_competitive_session_completed",
        ),
    }
    if name in _LAZY:
        import importlib

        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_baseline_if_missing",
    "fetch_actual_session_results",
    "KNOWN_MAIN_RACE_LAPS",
    "KNOWN_SPRINT_LAPS",
    "get_available_compounds",
    "get_competitive_session_completion_state",
    "get_compound_performance_modifier",
    "get_tire_stress_score",
    "is_competitive_session_completed",
    "load_and_validate_compound_data",
    "load_track_specific_params",
    "resolve_non_competitive_weather_features",
    "resolve_race_distance_laps",
    "resolve_track_temperature_c",
    "resolve_track_temperature_profile",
    "should_use_compound_adjustments",
    "validate_compound_data",
    "validate_compound_data_or_raise",
    "validate_pirelli_info",
    "validate_pirelli_info_or_raise",
]
