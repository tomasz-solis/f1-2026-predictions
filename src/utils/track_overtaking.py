"""Shared overtaking-difficulty priors for track data generation and loading."""

from __future__ import annotations

TRACK_OVERTAKING_BASELINES: dict[str, float] = {
    "Bahrain Grand Prix": 0.40,
    "Saudi Arabian Grand Prix": 0.60,
    "Australian Grand Prix": 0.50,
    "Japanese Grand Prix": 0.50,
    "Chinese Grand Prix": 0.30,
    "Miami Grand Prix": 0.50,
    "Monaco Grand Prix": 0.95,
    "Spanish Grand Prix": 0.40,
    "Canadian Grand Prix": 0.50,
    "Austrian Grand Prix": 0.40,
    "British Grand Prix": 0.40,
    "Hungarian Grand Prix": 0.80,
    "Belgian Grand Prix": 0.30,
    "Dutch Grand Prix": 0.50,
    "Italian Grand Prix": 0.20,
    "Singapore Grand Prix": 0.80,
    "United States Grand Prix": 0.40,
    "Mexico City Grand Prix": 0.40,
    "Brazilian Grand Prix": 0.40,
    "S\u00e3o Paulo Grand Prix": 0.40,
    "Las Vegas Grand Prix": 0.30,
    "Qatar Grand Prix": 0.40,
    "Abu Dhabi Grand Prix": 0.50,
    "Azerbaijan Grand Prix": 0.35,
    "Emilia Romagna Grand Prix": 0.65,
    "Pre-Season Testing": 0.50,
}


def get_track_overtaking_baseline(race_name: str | None, default: float = 0.50) -> float:
    """Return a conservative overtaking-difficulty prior for a track name."""
    if not race_name:
        return float(default)
    return float(TRACK_OVERTAKING_BASELINES.get(str(race_name), default))
