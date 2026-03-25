"""Simulation domain package for race engine, tire model, traffic, and strategy code."""

from .pit_strategy import generate_pit_strategy
from .strategy_optimizer import (
    calculate_pit_timing_bias_laps,
    calculate_undercut_window,
)
from .tire_degradation import (
    calculate_fuel_delta,
    calculate_tire_deg_delta,
    get_effective_tire_deg_slope,
    get_fresh_tire_advantage,
)
from .traffic_model import (
    calculate_dirty_air_penalty,
    get_track_downforce_level,
)


def __getattr__(name: str):
    """Lazily expose lap-by-lap simulator helpers to avoid import cycles."""
    if name in {"aggregate_simulation_results", "simulate_race_lap_by_lap"}:
        from src.utils.lap_by_lap_simulator import (
            aggregate_simulation_results,
            simulate_race_lap_by_lap,
        )

        exports = {
            "aggregate_simulation_results": aggregate_simulation_results,
            "simulate_race_lap_by_lap": simulate_race_lap_by_lap,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "aggregate_simulation_results",
    "calculate_dirty_air_penalty",
    "calculate_fuel_delta",
    "calculate_pit_timing_bias_laps",
    "calculate_tire_deg_delta",
    "calculate_undercut_window",
    "generate_pit_strategy",
    "get_effective_tire_deg_slope",
    "get_fresh_tire_advantage",
    "get_track_downforce_level",
    "simulate_race_lap_by_lap",
]
