"""
Session Order Extractor

The issue: FP sessions have LAP TIMES, not positions!
The fix: Extract fastest laps for FP, use positions for Quali/Race.
"""

import logging

import fastf1 as ff1
import numpy as np
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def extract_fp_order_from_laps(
    year: int,
    race_name: str,
    session_type: str,
) -> dict[str, int] | None:
    """Extract team order from an FP session using representative lap times."""
    # Try multiple session name variations
    variations = {
        "FP1": ["FP1", "Practice 1", "Free Practice 1"],
        "FP2": ["FP2", "Practice 2", "Free Practice 2"],
        "FP3": ["FP3", "Practice 3", "Free Practice 3"],
    }

    session_variations = variations.get(session_type, [session_type])

    for variation in session_variations:
        try:
            session = ff1.get_session(year, race_name, variation)

            # Load LAPS for FP sessions (key difference!)
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            if not hasattr(session, "laps") or session.laps is None or len(session.laps) == 0:
                continue

            laps = session.laps

            # Get fastest lap per team (median of drivers)
            team_times = {}

            for team in laps["Team"].unique():
                if pd.isna(team):
                    continue

                team_laps = laps[laps["Team"] == team]

                # Get each driver's fastest lap
                driver_best_times = []
                for driver in team_laps["Driver"].unique():
                    driver_laps = team_laps[team_laps["Driver"] == driver]

                    # Filter valid laps (has time, not deleted)
                    valid_laps = driver_laps[
                        (driver_laps["LapTime"].notna())
                        & (
                            ~driver_laps["IsAccurate"].isna()
                            if "IsAccurate" in driver_laps
                            else True
                        )
                    ]

                    if len(valid_laps) > 0:
                        best_time = valid_laps["LapTime"].min()
                        driver_best_times.append(best_time.total_seconds())

                if driver_best_times:
                    # Median avoids single-driver outliers
                    team_times[team] = np.median(driver_best_times)

            if len(team_times) < 5:  # Need at least 5 teams
                continue

            # Convert to ranks (1 = fastest time)
            sorted_teams = sorted(team_times.items(), key=lambda x: x[1])
            team_ranks = {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}

            from src.extractors.validation import log_validation_warnings, validate_fp_team_order

            warnings = validate_fp_team_order(
                team_ranks, context=f"{session_type} {year} {race_name}"
            )
            log_validation_warnings(warnings)

            return team_ranks

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            # Try next variation
            logger.debug(
                "Session variation %s for %s (%s %s) failed: %s",
                variation,
                session_type,
                year,
                race_name,
                e,
            )
            continue

    return None


def extract_quali_order_from_positions(
    year: int,
    race_name: str,
    session_type: str,
) -> dict[str, int] | None:
    """Extract team order from qualifying-style sessions using classified positions."""
    # Try multiple session name variations
    variations = {
        "Q": ["Q", "Qualifying"],
        "Sprint Qualifying": ["Sprint Qualifying", "Sprint Shootout", "SQ"],
    }

    session_variations = variations.get(session_type, [session_type])

    for variation in session_variations:
        try:
            session = ff1.get_session(year, race_name, variation)
            session.load(laps=False, telemetry=False, weather=False, messages=False)

            if not hasattr(session, "results") or session.results is None:
                continue

            results = session.results

            # Check if Position exists
            if "Position" not in results.columns:
                continue

            # Require enough valid positions to compare teams.
            valid_positions = results["Position"].notna().sum()
            if valid_positions < 5:
                continue

            # Extract team positions (median of drivers)
            team_positions = {}

            for team in results["TeamName"].unique():
                if pd.isna(team):
                    continue

                team_results = results[results["TeamName"] == team]
                positions = team_results["Position"].dropna()

                if len(positions) > 0:
                    team_positions[team] = float(np.median(positions))

            if len(team_positions) < 5:
                continue

            # Convert to ranks (1 = best position)
            sorted_teams = sorted(team_positions.items(), key=lambda x: x[1])
            team_ranks = {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}

            return team_ranks

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.debug(
                "Session variation %s for %s (%s %s) failed: %s",
                variation,
                session_type,
                year,
                race_name,
                e,
            )
            continue

    logger.warning(
        "Could not extract team order for %s (%s) using %s. No session variation succeeded.",
        race_name,
        year,
        session_type,
    )
    return None


def extract_session_order_safe(
    year: int,
    race_name: str,
    session_type: str,
) -> dict[str, int] | None:
    """Extract team order from any supported session using the appropriate method."""
    # Determine extraction method based on session type
    fp_sessions = ["FP1", "FP2", "FP3"]
    quali_sessions = ["Q", "Sprint Qualifying", "Sprint Shootout", "SQ"]

    if session_type in fp_sessions:
        # Use lap times for FP
        return extract_fp_order_from_laps(year, race_name, session_type)
    elif session_type in quali_sessions:
        # Use positions for quali
        return extract_quali_order_from_positions(year, race_name, session_type)
    else:
        # Try both methods
        result = extract_quali_order_from_positions(year, race_name, session_type)
        if result:
            return result
        return extract_fp_order_from_laps(year, race_name, session_type)


def calculate_order_mae(
    predicted_order: dict[str, int], actual_order: dict[str, int]
) -> float | None:
    """Calculate MAE between predicted and actual team order."""
    errors = []

    for team in predicted_order:
        if team in actual_order:
            error = abs(predicted_order[team] - actual_order[team])
            errors.append(error)

    return float(np.mean(errors)) if errors else None
