"""Simulation domain package — race engine, tire degradation, pit strategy, and traffic.

Re-exports the canonical implementations from ``src.utils`` so that new
code can ``from src.simulation import simulate_race_lap_by_lap`` while
existing imports in ``src.utils`` keep working without any change.
"""

from src.utils.lap_by_lap_simulator import (
    aggregate_simulation_results,
    simulate_race_lap_by_lap,
)
from src.utils.pit_strategy import generate_pit_strategy
from src.utils.strategy_optimizer import (
    calculate_pit_timing_bias_laps,
    calculate_undercut_window,
)
from src.utils.tire_degradation import (
    calculate_fuel_delta,
    calculate_tire_deg_delta,
    get_effective_tire_deg_slope,
    get_fresh_tire_advantage,
)
from src.utils.traffic_model import (
    calculate_dirty_air_penalty,
    get_track_downforce_level,
)

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
