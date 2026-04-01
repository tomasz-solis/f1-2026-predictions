"""
Race Pace Extractor

Extract long run pace from FP2 sessions.
Detects race simulation runs (10+ laps on same tire).
"""

import logging

import fastf1 as ff1
import numpy as np
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def extract_fp2_pace(year: int, race_name: str, verbose: bool = False) -> dict | None:
    """Extract long run pace from FP2, detecting race simulations and calculating degradation."""
    try:
        # Load FP2
        session = ff1.get_session(year, race_name, "FP2")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        if not hasattr(session, "laps") or session.laps is None:
            if verbose:
                logger.info("No FP2 lap data available")
            return None

        laps = session.laps

        if verbose:
            logger.info("Analyzing FP2: %s laps total", len(laps))

        # Find long run stints for each team
        team_pace = {}

        for team in laps["Team"].unique():
            if pd.isna(team):
                continue

            team_laps = laps[laps["Team"] == team]

            # Detect long run stints
            long_runs = _detect_long_runs(team_laps, verbose)

            if not long_runs:
                continue

            # Use best/most representative stint
            best_stint = _select_best_stint(long_runs)

            if best_stint:
                team_pace[team] = best_stint

        if not team_pace:
            if verbose:
                logger.info("No long runs detected")
            return None

        # Calculate relative pace (vs median)
        all_paces = [p["avg_pace"] for p in team_pace.values()]
        median_pace = np.median(all_paces)

        for team in team_pace:
            team_pace[team]["relative_pace"] = team_pace[team]["avg_pace"] - median_pace

        if verbose:
            logger.info("Extracted pace for %s teams", len(team_pace))

        from src.extractors.validation import log_validation_warnings, validate_team_pace_data

        warnings = validate_team_pace_data(team_pace, context=f"FP2 {year} {race_name}")
        log_validation_warnings(warnings)

        return team_pace

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.error(
            "Failed to extract FP2 pace for %s (%s): %s. Race pace simulation will be unavailable.",
            race_name,
            year,
            e,
        )
        if verbose:
            logger.warning("FP2 extraction failed: %s", e)
        return None


def _detect_long_runs(team_laps: pd.DataFrame, verbose: bool = False) -> list:
    """Detect long run stints (10+ laps on same compound) with pace metrics."""
    # Filter valid laps
    valid_laps = team_laps[(team_laps["LapTime"].notna()) & (team_laps["Compound"].notna())].copy()

    if len(valid_laps) < 10:
        return []

    # Group consecutive laps on same compound
    valid_laps["CompoundChange"] = (
        valid_laps["Compound"] != valid_laps["Compound"].shift()
    ).cumsum()

    long_runs = []

    for _stint_id, stint_laps in valid_laps.groupby("CompoundChange"):
        # Check if long enough (10+ laps)
        if len(stint_laps) < 10:
            continue

        # Check if it's a race sim (not quali sim)
        # Race sims: consistent lap times, no major outliers
        lap_times = stint_laps["LapTime"].dt.total_seconds()

        # Remove outliers (in/out laps, traffic)
        median_time = lap_times.median()
        std_time = lap_times.std()

        clean_laps = stint_laps[abs(lap_times - median_time) < 3 * std_time]

        if len(clean_laps) < 8:  # Need at least 8 clean laps
            continue

        # Calculate pace metrics
        clean_times = clean_laps["LapTime"].dt.total_seconds()

        avg_pace = clean_times.mean()
        std_pace = clean_times.std()

        # Calculate degradation (lap time increase per lap)
        # Fit linear trend: lap_time = base + deg_rate * lap_num
        lap_nums = np.arange(len(clean_times))
        deg_rate = np.polyfit(lap_nums, clean_times, 1)[0]

        long_runs.append(
            {
                "laps": len(clean_laps),
                "compound": stint_laps.iloc[0]["Compound"],
                "avg_pace": avg_pace,
                "std_pace": std_pace,
                "degradation": max(0, deg_rate),  # Only positive (degradation)
                "stint_laps": stint_laps,
            }
        )

    return long_runs


def _select_best_stint(long_runs: list) -> dict | None:
    """
    Select best/most representative long run stint.

    Prefers:
    1. Longest stint (more representative)
    2. Race compound (not quali sim)
    3. Lower std (consistent pace)
    """
    if not long_runs:
        return None

    # Sort by laps (prefer longer)
    long_runs.sort(key=lambda x: x["laps"], reverse=True)

    # Pick longest stint
    # (Could add logic to prefer certain compounds, but longest is usually race sim)
    best = long_runs[0]

    return {
        "avg_pace": best["avg_pace"],
        "degradation": best["degradation"],
        "laps": best["laps"],
        "compound": best["compound"],
        "consistency": best["std_pace"],
    }
