"""Data domain package — track loading, data generation, results fetching, and compounds.

Re-exports the canonical implementations from ``src.utils`` so that new
code can ``from src.data import load_track_specific_params`` while existing
imports in ``src.utils`` keep working without any change.

Heavy dependencies (actual_results_fetcher) are imported lazily so that the
package can be loaded in lightweight environments.
"""

from src.utils.compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from src.utils.compound_validator import (
    validate_compound_data,
    validate_compound_data_or_raise,
)
from src.utils.data_generator import create_baseline_if_missing
from src.utils.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_race_distance_laps,
)


def __getattr__(name: str):
    """Lazily import heavy modules on first access."""
    _LAZY = {
        "fetch_actual_session_results": (
            "src.utils.actual_results_fetcher",
            "fetch_actual_session_results",
        ),
        "get_competitive_session_completion_state": (
            "src.utils.actual_results_fetcher",
            "get_competitive_session_completion_state",
        ),
        "is_competitive_session_completed": (
            "src.utils.actual_results_fetcher",
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
    "get_available_compounds",
    "get_competitive_session_completion_state",
    "get_compound_performance_modifier",
    "get_tire_stress_score",
    "is_competitive_session_completed",
    "load_track_specific_params",
    "resolve_race_distance_laps",
    "should_use_compound_adjustments",
    "validate_compound_data",
    "validate_compound_data_or_raise",
]
