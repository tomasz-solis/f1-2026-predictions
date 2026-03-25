"""Traffic and dirty-air modeling utilities."""

from __future__ import annotations

import numpy as np

DEFAULT_TRACK_DOWNFORCE = 0.70

TRACK_DOWNFORCE_LEVELS: dict[str, float] = {
    "Monaco Grand Prix": 1.00,
    "Hungarian Grand Prix": 0.95,
    "Singapore Grand Prix": 0.90,
    "Azerbaijan Grand Prix": 0.85,
    "Japanese Grand Prix": 0.82,
    "Spanish Grand Prix": 0.80,
    "Dutch Grand Prix": 0.78,
    "Abu Dhabi Grand Prix": 0.75,
    "Miami Grand Prix": 0.72,
    "Bahrain Grand Prix": 0.68,
    "Saudi Arabian Grand Prix": 0.65,
    "Australian Grand Prix": 0.65,
    "Chinese Grand Prix": 0.62,
    "British Grand Prix": 0.60,
    "United States Grand Prix": 0.60,
    "Canadian Grand Prix": 0.58,
    "Austrian Grand Prix": 0.55,
    "Mexico City Grand Prix": 0.55,
    "Qatar Grand Prix": 0.52,
    "Emilia Romagna Grand Prix": 0.50,
    "Brazilian Grand Prix": 0.48,
    "São Paulo Grand Prix": 0.48,
    "Belgian Grand Prix": 0.45,
    "Las Vegas Grand Prix": 0.42,
    "Italian Grand Prix": 0.30,
    "Pre-Season Testing": 0.68,
}


def get_track_downforce_level(
    track_name: str | None,
    track_overtaking: float | None = None,
) -> float:
    """Return track downforce level (0.0 to 1.0) for dirty-air modeling."""
    if track_name and track_name in TRACK_DOWNFORCE_LEVELS:
        return TRACK_DOWNFORCE_LEVELS[track_name]

    if track_overtaking is not None:
        return float(np.clip(track_overtaking, 0.0, 1.0))

    return DEFAULT_TRACK_DOWNFORCE


def calculate_dirty_air_penalty(
    gap_to_car_ahead_s: float,
    track_downforce_level: float,
    dirty_air_window_s: float = 1.8,
    min_penalty_s: float = 0.02,
    max_penalty_s: float = 0.05,
    car_speed_kph: float | None = None,
) -> float:
    """Calculate lap-time penalty from dirty air.

    The penalty is:
    - zero outside the dirty-air window,
    - higher on high-downforce tracks,
    - stronger at closer gaps,
    - mildly speed-sensitive when speed is available.
    """
    if dirty_air_window_s <= 0 or gap_to_car_ahead_s >= dirty_air_window_s:
        return 0.0

    downforce = float(np.clip(track_downforce_level, 0.0, 1.0))
    gap = max(0.0, float(gap_to_car_ahead_s))

    # Exponent keeps Monaco/Monza close-gap penalty ratio near 2x.
    base_penalty = min_penalty_s + ((max_penalty_s - min_penalty_s) * (downforce**1.5))

    closeness = np.clip(1.0 - (gap / dirty_air_window_s), 0.0, 1.0)
    gap_factor = closeness**1.5

    speed_factor = 1.0
    if car_speed_kph is not None:
        normalized_speed = np.clip((car_speed_kph - 150.0) / 150.0, 0.0, 1.0)
        speed_factor = 0.65 + (0.35 * normalized_speed)

    penalty = base_penalty * gap_factor * speed_factor
    return float(np.clip(penalty, 0.0, max_penalty_s))
