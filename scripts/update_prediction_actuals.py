"""
Update Prediction with Actual Results

Use this script to add actual race results to saved predictions for accuracy tracking.

USAGE:
    python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP1 --year 2026

This will attempt to fetch actual qualifying and race results from FastF1 and update
the saved prediction.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1

from src.utils.prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_actual_results(year: int, race_name: str):
    """
    Fetch actual qualifying and race results from FastF1.

    Args:
        year: Season year
        race_name: Race name

    Returns:
        Tuple of (qualifying_results, race_results)
    """
    logger.info(f"Fetching actual results for {race_name} {year}...")

    try:
        # Get qualifying results
        quali_session = fastf1.get_session(year, race_name, "Q")
        quali_session.load()

        quali_results = []
        for pos in range(1, 23):  # Up to 22 drivers
            driver_data = quali_session.results[quali_session.results["Position"] == pos]
            if not driver_data.empty:
                driver_abbr = driver_data.iloc[0]["Abbreviation"]
                team_name = driver_data.iloc[0]["TeamName"]
                quali_results.append({"driver": driver_abbr, "team": team_name})

        logger.info(f"Found {len(quali_results)} qualifying results")

        # Get race results
        race_session = fastf1.get_session(year, race_name, "R")
        race_session.load()

        race_results = []
        for pos in range(1, 23):  # Up to 22 drivers
            driver_data = race_session.results[race_session.results["Position"] == pos]
            if not driver_data.empty:
                driver_abbr = driver_data.iloc[0]["Abbreviation"]
                team_name = driver_data.iloc[0]["TeamName"]
                race_results.append({"driver": driver_abbr, "team": team_name})

        logger.info(f"Found {len(race_results)} race results")

        return quali_results, race_results

    except Exception as e:
        logger.error(f"Failed to fetch actual results: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Update saved prediction with actual race results")
    parser.add_argument("race_name", help="Race name (e.g., 'Bahrain Grand Prix')")
    parser.add_argument("session_name", help="Session name (e.g., 'FP1', 'FP2', 'FP3', 'SQ')")
    parser.add_argument("--year", type=int, default=2026, help="Season year")

    args = parser.parse_args()

    # Initialize prediction logger
    logger_inst = PredictionLogger()

    # Check if prediction exists
    if not logger_inst.has_prediction_for_session(args.year, args.race_name, args.session_name):
        logger.error(
            f"No prediction found for {args.race_name} after {args.session_name}. "
            "Save a prediction first using the dashboard."
        )
        return 1

    # Fetch actual results
    quali_results, race_results = fetch_actual_results(args.year, args.race_name)

    if quali_results is None and race_results is None:
        logger.error("Could not fetch actual results. Race may not have completed yet.")
        return 1

    # Update prediction with actuals
    success = logger_inst.update_actuals(
        year=args.year,
        race_name=args.race_name,
        session_name=args.session_name,
        qualifying_results=quali_results,
        race_results=race_results,
    )

    if success:
        logger.info(
            f"[OK] Successfully updated prediction for {args.race_name} (after {args.session_name}) with actual results"
        )
        logger.info("View accuracy metrics in the 'Prediction Accuracy' tab in the dashboard")
        return 0
    else:
        logger.error("Failed to update prediction")
        return 1


if __name__ == "__main__":
    sys.exit(main())
