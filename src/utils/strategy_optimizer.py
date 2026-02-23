"""Strategy optimization helpers for pit timing decisions."""

from __future__ import annotations

import numpy as np


def calculate_undercut_window(
    race_distance: int,
    car_ahead_tire_age: int,
    pit_loss_time: float,
    fresh_tire_advantage: float = 0.30,
) -> tuple[int, int]:
    """Calculate a practical undercut window as (earliest_lap, latest_lap)."""
    if race_distance <= 0:
        return (1, 1)

    pit_loss_time = max(0.0, pit_loss_time)
    fresh_tire_advantage = max(0.05, fresh_tire_advantage)
    tire_age = max(1, int(car_ahead_tire_age))

    laps_to_recover = pit_loss_time / fresh_tire_advantage
    earliest = max(5, int(round(tire_age - laps_to_recover)))
    latest = min(race_distance - 3, tire_age + 3)
    if latest < earliest:
        latest = earliest

    return (earliest, latest)


def evaluate_overcut(
    track_overtaking: float,
    tire_age: int,
    traffic_penalty: float = 0.0,
) -> float:
    """Evaluate overcut potential (>0 favors staying out, <0 favors undercut)."""
    track_overtaking = float(np.clip(track_overtaking, 0.0, 1.0))
    tire_age = max(1, int(tire_age))
    traffic_penalty = max(0.0, float(traffic_penalty))

    track_position_value = track_overtaking * 2.0
    tire_wear_cost = max(0.0, (tire_age - 15) * 0.05)
    return float(track_position_value - tire_wear_cost - traffic_penalty)


def calculate_pit_timing_bias_laps(
    track_overtaking: float | None,
    grid_position: int | None,
    race_distance: int,
    strategy_signal: float = 0.0,
) -> float:
    """Return pit-lap bias in laps: negative=earlier undercut, positive=later overcut."""
    if race_distance <= 0:
        return 0.0

    overtaking = 0.5 if track_overtaking is None else float(np.clip(track_overtaking, 0.0, 1.0))
    grid_pos = int(grid_position) if grid_position is not None else 11

    # Easy overtaking -> undercut; hard overtaking -> overcut.
    track_component = (overtaking - 0.5) * 4.0  # approx [-2, +2]

    # Front-runners on hard-to-pass tracks protect track position with slight overcut bias.
    front_component = 0.0
    if grid_pos <= 6:
        front_component = overtaking * 0.6
    elif grid_pos >= 14:
        # Backmarkers on easy tracks can attempt earlier undercut.
        front_component = -(1.0 - overtaking) * 0.8

    signal_component = float(np.clip(strategy_signal, -1.0, 1.0)) * 0.7
    bias = track_component + front_component + signal_component

    max_abs_bias = min(4.0, race_distance * 0.08)
    return float(np.clip(bias, -max_abs_bias, max_abs_bias))
