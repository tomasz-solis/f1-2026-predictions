"""Calculate driver-specific FP adjustments for track-level form."""

import logging

import numpy as np
import pandas as pd

from src.utils.fp_blending import get_fp_team_performance

logger = logging.getLogger(__name__)


def _median_lap_seconds(laps: pd.DataFrame) -> float | None:
    """Return median valid lap time in seconds."""
    if laps.empty or "LapTime" not in laps.columns:
        return None

    lap_seconds = laps["LapTime"].dt.total_seconds().dropna()
    if lap_seconds.empty:
        return None
    return float(lap_seconds.median())


def calculate_driver_fp_modifiers(
    year: int,
    race_name: str,
    session_types: list[str],
    scale: float = 0.10,
    smoothing_seconds: float = 0.50,
    preloaded_session_laps: dict[str, pd.DataFrame | None] | None = None,
) -> dict[str, float]:
    """Estimate per-driver modifiers from teammate pace deltas in FP sessions.

    Reuses `preloaded_session_laps` when provided to avoid repeated FastF1 loads
    inside the same prediction call.

    Returns:
        Mapping of driver code to adjustment in [-scale, +scale].
    """
    if not session_types:
        return {}

    deltas: dict[str, float] = {}
    counts: dict[str, int] = {}
    laps_by_session = preloaded_session_laps or {}

    for session_type in session_types:
        if session_type in laps_by_session:
            laps = laps_by_session.get(session_type)
        else:
            try:
                _, laps, _ = get_fp_team_performance(year, race_name, session_type)
            except Exception as e:
                logger.warning(f"Could not read FP data for {session_type}: {e}")
                continue

        if laps is None or not isinstance(laps, pd.DataFrame) or laps.empty:
            continue
        if "Team" not in laps.columns or "Driver" not in laps.columns:
            continue

        for _team, team_laps in laps.groupby("Team"):
            if team_laps.empty:
                continue

            driver_medians: dict[str, float] = {}
            for driver, driver_laps in team_laps.groupby("Driver"):
                median_time = _median_lap_seconds(driver_laps)
                if median_time is not None:
                    driver_medians[str(driver)] = median_time

            if len(driver_medians) < 2:
                continue

            for driver, driver_time in driver_medians.items():
                teammate_times = [t for d, t in driver_medians.items() if d != driver]
                if not teammate_times:
                    continue
                teammate_time = float(np.median(teammate_times))
                delta_seconds = teammate_time - driver_time
                delta = np.clip(delta_seconds / max(1e-6, smoothing_seconds), -1.0, 1.0) * scale
                deltas[driver] = deltas.get(driver, 0.0) + delta
                counts[driver] = counts.get(driver, 0) + 1

    modifiers = {
        driver: float(np.clip(total / counts[driver], -scale, scale))
        for driver, total in deltas.items()
        if counts.get(driver, 0) > 0
    }

    if modifiers:
        logger.info(f"Calculated FP driver modifiers for {len(modifiers)} drivers")
    return modifiers
