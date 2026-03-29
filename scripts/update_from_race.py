"""Update team and driver characteristics after one race weekend."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.systems.updater import update_from_race  # noqa: E402
from src.utils.prediction_logger import PredictionLogger  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the race-update command-line entry point."""
    parser = argparse.ArgumentParser(description="Update characteristics after a race")
    parser.add_argument("race_name", help="Race name (e.g., 'Australian Grand Prix')")
    parser.add_argument("--year", type=int, default=2026, help="Season year")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")

    args = parser.parse_args()

    try:
        update_from_race(args.year, args.race_name, args.data_dir)
        reconciled_predictions = PredictionLogger().reconcile_completed_prediction_actuals(
            args.year
        )
        logger.info("Reconciled actuals for %s saved prediction(s).", reconciled_predictions)
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise


if __name__ == "__main__":
    main()
