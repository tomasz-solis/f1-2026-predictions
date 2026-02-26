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


def extract_fp_order_from_laps(year, race_name, session_type):
    """
    Extract team order from FP session using lap times (FP sessions lack position data).
    """
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
                    # Use median of team's drivers (robust to one having issues)
                    team_times[team] = np.median(driver_best_times)

            if len(team_times) < 5:  # Need at least 5 teams
                continue

            # Convert to ranks (1 = fastest time)
            sorted_teams = sorted(team_times.items(), key=lambda x: x[1])
            team_ranks = {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}

            return team_ranks

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            # Try next variation
            logger.debug(
                f"Session variation {variation} for {session_type} ({year} {race_name}) failed: {e}"
            )
            continue

    return None


def extract_quali_order_from_positions(year, race_name, session_type):
    """
    Extract team order from Qualifying/Sprint Quali using position data.
    """
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

            # Check we have enough valid positions
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
                f"Session variation {variation} for {session_type} ({year} {race_name}) failed: {e}"
            )
            continue

    logger.warning(
        f"Could not extract team order for {race_name} ({year}) using {session_type}. No session variation succeeded."
    )
    return None


def extract_session_order_safe(year, race_name, session_type):
    """
    Extract team order from any session, auto-detecting FP (lap times) vs Quali (positions).
    """
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


def calculate_order_mae(predicted_order, actual_order):
    """
    Calculate MAE between predicted and actual team order.
    """
    errors = []

    for team in predicted_order:
        if team in actual_order:
            error = abs(predicted_order[team] - actual_order[team])
            errors.append(error)

    return np.mean(errors) if errors else None


def test_session_as_predictor_fixed(
    year,
    race_name,
    predictor_session,
    target_session="Q",
    driver_ranker=None,
    lineups=None,
    actual_driver_results=None,
):
    """
    Test prediction accuracy of a session against qualifying/race results.
    """
    # Get predictor session order
    predictor_order = extract_session_order_safe(year, race_name, predictor_session)

    if predictor_order is None:
        return {
            "status": "failed",
            "reason": f"{predictor_session} data not available",
            "race": race_name,
        }

    # Get actual qualifying order
    actual_order = extract_session_order_safe(year, race_name, target_session)

    if actual_order is None:
        return {
            "status": "failed",
            "reason": f"{target_session} data not available",
            "race": race_name,
        }

    # Calculate team-level MAE
    team_mae = calculate_order_mae(predictor_order, actual_order)

    result = {
        "status": "success",
        "race": race_name,
        "predictor_session": predictor_session,
        "target_session": target_session,
        "team_mae": team_mae,
        "predictor_order": predictor_order,
        "actual_order": actual_order,
    }

    # If driver ranker provided, test driver-level
    if driver_ranker and lineups and actual_driver_results:
        try:
            # Predict drivers using predictor session order
            driver_preds = driver_ranker.predict_positions(
                team_predictions=predictor_order,
                team_lineups=lineups,
                session_type="qualifying",
            )

            # Calculate driver MAE
            errors = []

            for pred in driver_preds["predictions"]:
                actual_pos = next(
                    (p["position"] for p in actual_driver_results if p["driver"] == pred.driver),
                    None,
                )

                if actual_pos and pd.notna(actual_pos):
                    errors.append(abs(pred.position - actual_pos))

            if errors:
                result["driver_mae"] = np.mean(errors)
                result["driver_within_1"] = sum(1 for e in errors if e <= 1) / len(errors)
                result["driver_within_2"] = sum(1 for e in errors if e <= 2) / len(errors)
                result["driver_within_3"] = sum(1 for e in errors if e <= 3) / len(errors)
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.error(
                f"Error calculating driver-level accuracy for {race_name}: {e}. Driver metrics will be unavailable."
            )
            result["driver_error"] = str(e)

    return result
