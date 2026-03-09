"""Update one saved prediction with actual target results."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime_dependencies() -> tuple[object, ...]:
    """Import project modules after making the repository root importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.accuracy_targets import (
        explicit_target_predictions,
        fastf1_session_name,
        legacy_target_keys_for_prediction,
        synthesize_legacy_targets,
        target_session_name,
    )
    from src.utils.actual_results_fetcher import fetch_actual_session_results
    from src.utils.prediction_logger import PredictionLogger
    from src.utils.weekend import is_sprint_weekend

    return (
        explicit_target_predictions,
        fastf1_session_name,
        legacy_target_keys_for_prediction,
        synthesize_legacy_targets,
        target_session_name,
        fetch_actual_session_results,
        PredictionLogger,
        is_sprint_weekend,
    )


(
    explicit_target_predictions,
    fastf1_session_name,
    legacy_target_keys_for_prediction,
    synthesize_legacy_targets,
    target_session_name,
    fetch_actual_session_results,
    PredictionLogger,
    is_sprint_weekend,
) = _load_runtime_dependencies()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for logger_name in ("requests_cache", "urllib3", "fastf1", "fastf1.req", "req"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def prediction_is_sprint_weekend(
    prediction_data: dict,
    *,
    year: int,
    race_name: str,
) -> bool:
    """Infer whether a saved prediction belongs to a sprint weekend."""
    metadata = prediction_data.get("metadata", {})
    weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    if weekend_format == "sprint":
        return True
    if weekend_format == "normal":
        return False

    target_predictions = explicit_target_predictions(prediction_data)
    if any("sprint" in target_key for target_key in target_predictions):
        return True

    checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
    if checkpoint_session in {"SQ", "SPRINT"}:
        return True
    if checkpoint_session in {"FP2", "FP3"}:
        return False

    try:
        return bool(is_sprint_weekend(year, race_name))
    except Exception as exc:
        logger.warning("Could not confirm weekend format for %s %s: %s", race_name, year, exc)
        return False


def fetch_target_actual_results(
    *,
    year: int,
    race_name: str,
    prediction_data: dict,
    is_sprint: bool,
) -> tuple[dict[str, list[dict] | None], list[dict] | None, list[dict] | None]:
    """Fetch actual results for every stored target without inventing missing sessions."""
    metadata = prediction_data.get("metadata", {})
    checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
    target_predictions = explicit_target_predictions(prediction_data)
    if not target_predictions:
        target_predictions = synthesize_legacy_targets(prediction_data, is_sprint=is_sprint)

    session_results: dict[str, list[dict] | None] = {}
    target_actual_results: dict[str, list[dict] | None] = {}

    for target_key, payload in target_predictions.items():
        target_session = str(payload.get("target_session", target_session_name(target_key)))
        if target_session not in session_results:
            logger.info("Fetching actual %s results for %s %s", target_session, race_name, year)
            session_results[target_session] = fetch_actual_session_results(
                year,
                race_name,
                fastf1_session_name(target_session),
            )
        target_actual_results[target_key] = session_results[target_session]

    qualifying_target = str(metadata.get("top_level_qualifying_target", "")).strip()
    race_target = str(metadata.get("top_level_race_target", "")).strip()
    if not qualifying_target or not race_target:
        inferred_qualifying, inferred_race = legacy_target_keys_for_prediction(
            checkpoint_session,
            is_sprint=is_sprint,
        )
        qualifying_target = qualifying_target or str(inferred_qualifying or "")
        race_target = race_target or str(inferred_race or "")

    qualifying_results = None
    race_results = None
    if qualifying_target in target_predictions:
        qualifying_session = str(
            target_predictions[qualifying_target].get(
                "target_session",
                target_session_name(qualifying_target),
            )
        )
        qualifying_results = session_results.get(qualifying_session)
    if race_target in target_predictions:
        race_session = str(
            target_predictions[race_target].get(
                "target_session",
                target_session_name(race_target),
            )
        )
        race_results = session_results.get(race_session)

    return target_actual_results, qualifying_results, race_results


def main() -> int:
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(description="Update a saved prediction with actual results.")
    parser.add_argument("race_name", help="Race name (for example 'Bahrain Grand Prix')")
    parser.add_argument("session_name", help="Checkpoint session (for example 'FP1' or 'SQ')")
    parser.add_argument("--year", type=int, default=2026, help="Season year")
    args = parser.parse_args()

    logger_inst = PredictionLogger()
    if not logger_inst.has_prediction_for_session(args.year, args.race_name, args.session_name):
        logger.error(
            "No prediction found for %s after %s. Save a prediction first using the dashboard.",
            args.race_name,
            args.session_name,
        )
        return 1

    prediction_data = logger_inst.load_prediction(args.year, args.race_name, args.session_name)
    if prediction_data is None:
        logger.error("Prediction exists but could not be loaded.")
        return 1

    sprint_flag = prediction_is_sprint_weekend(
        prediction_data,
        year=args.year,
        race_name=args.race_name,
    )
    target_actual_results, qualifying_results, race_results = fetch_target_actual_results(
        year=args.year,
        race_name=args.race_name,
        prediction_data=prediction_data,
        is_sprint=sprint_flag,
    )
    if not target_actual_results and qualifying_results is None and race_results is None:
        logger.error("Could not fetch actual results. The target session may not be complete yet.")
        return 1

    success = logger_inst.update_actuals(
        year=args.year,
        race_name=args.race_name,
        session_name=args.session_name,
        qualifying_results=qualifying_results,
        race_results=race_results,
        target_actual_results=target_actual_results,
    )
    if not success:
        logger.error("Failed to update prediction")
        return 1

    logger.info(
        "Successfully updated prediction for %s (checkpoint %s) with actual results",
        args.race_name,
        args.session_name,
    )
    logger.info("View accuracy metrics in the 'Prediction Accuracy' page in the dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
