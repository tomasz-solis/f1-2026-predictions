"""Dashboard prediction orchestration."""

import logging
import time

from .cache import get_predictor

logger = logging.getLogger(__name__)


def fetch_grid_if_available(
    year: int,
    race_name: str,
    session_name: str,
    predicted_grid: list,
) -> tuple[list, str]:
    """Fetch actual grid if session completed, otherwise use predicted grid."""
    from src.utils.actual_results_fetcher import (
        fetch_actual_session_results,
        is_competitive_session_completed,
    )
    from src.utils.grid_validation import validate_qualifying_grid

    logger.info(
        "FastF1 session refresh check started: race=%s year=%s session=%s",
        race_name,
        year,
        session_name,
    )

    session_completed = is_competitive_session_completed(year, race_name, session_name)
    logger.info(
        "FastF1 session completion status: race=%s year=%s session=%s completed=%s",
        race_name,
        year,
        session_name,
        session_completed,
    )

    if session_completed:
        logger.info(f"{session_name} is completed, fetching actual grid from FastF1")
        actual_grid = fetch_actual_session_results(year, race_name, session_name)
        if actual_grid:
            validated_grid = validate_qualifying_grid(actual_grid)
            logger.info(
                "Grid source resolved: ACTUAL race=%s year=%s session=%s drivers=%s",
                race_name,
                year,
                session_name,
                len(validated_grid),
            )
            return validated_grid, "ACTUAL"
        raise RuntimeError(
            f"FastF1 returned no {session_name} results for completed session at "
            f"{race_name} {year}; refusing to fall back to predicted grid."
        )
    else:
        validated_grid = validate_qualifying_grid(predicted_grid)
        logger.info(
            "Grid source resolved: PREDICTED race=%s year=%s session=%s drivers=%s",
            race_name,
            year,
            session_name,
            len(validated_grid),
        )
        return validated_grid, "PREDICTED"


def run_prediction(
    race_name: str,
    weather: str,
    _artifact_versions: dict[str, tuple[int, str]],
    is_sprint: bool = False,
    year: int = 2026,
) -> dict:
    """
    Run full weekend cascade prediction.

    Executes on every user-triggered run so FastF1-dependent session checks refresh.
    """
    valid_weather = ["dry", "rain", "mixed"]
    if weather not in valid_weather:
        raise ValueError(f"Weather must be one of {valid_weather}, got '{weather}'")

    timing: dict[str, float] = {}
    overall_start = time.time()

    try:
        predictor = get_predictor(_artifact_versions, year=year)
    except TypeError:
        predictor = get_predictor(_artifact_versions)
    results = {}

    if is_sprint:
        # SPRINT WEEKEND CASCADE: SQ -> Sprint -> Main Quali -> Main Race

        sq_start = time.time()
        sq_result = predictor.predict_qualifying(
            year=year,
            race_name=race_name,
            qualifying_stage="sprint",
        )
        timing["sprint_quali"] = time.time() - sq_start
        results["sprint_quali"] = sq_result

        sprint_start = time.time()
        sq_grid, grid_source = fetch_grid_if_available(year, race_name, "SQ", sq_result["grid"])
        results["sprint_quali"]["grid_source"] = grid_source

        sprint_result = predictor.predict_sprint_race(
            sprint_quali_grid=sq_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=50,
        )
        timing["sprint_race"] = time.time() - sprint_start
        results["sprint_race"] = sprint_result

        mq_start = time.time()
        mq_result = predictor.predict_qualifying(
            year=year,
            race_name=race_name,
            qualifying_stage="main",
        )
        timing["main_quali"] = time.time() - mq_start
        results["main_quali"] = mq_result

        mr_start = time.time()
        quali_grid, grid_source = fetch_grid_if_available(year, race_name, "Q", mq_result["grid"])
        results["main_quali"]["grid_source"] = grid_source

        main_race_result = predictor.predict_race(
            qualifying_grid=quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=50,
            year=year,
        )
        timing["main_race"] = time.time() - mr_start
        results["main_race"] = main_race_result

    else:
        # NORMAL WEEKEND CASCADE: Quali -> Race

        quali_start = time.time()
        quali_result = predictor.predict_qualifying(
            year=year,
            race_name=race_name,
            qualifying_stage="main",
        )
        timing["qualifying"] = time.time() - quali_start
        results["qualifying"] = quali_result

        race_start = time.time()
        quali_grid, grid_source = fetch_grid_if_available(
            year, race_name, "Q", quali_result["grid"]
        )
        results["qualifying"]["grid_source"] = grid_source

        race_result = predictor.predict_race(
            qualifying_grid=quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=50,
            year=year,
        )
        timing["race"] = time.time() - race_start
        results["race"] = race_result

    timing["total"] = time.time() - overall_start

    for key in results:
        results[key]["timing"] = timing

    return results
