"""Weekend prediction builder used by warmup and automation jobs."""

import logging
import time
from typing import Any

from src.types.prediction_types import QualifyingGridEntry
from src.utils.race_input_confidence import (
    cap_predicted_main_race_input_confidence,
    derive_race_input_confidence,
)

from .cache import get_predictor
from .checkpoint_predictor import build_checkpoint_overlay_predictor
from .precomputed_predictions import get_prediction_precompute_config

logger = logging.getLogger(__name__)


class CompetitiveSessionStatusUnavailableError(RuntimeError):
    """Raised when competitive-session status cannot be verified from FastF1."""


def _session_display_name(session_name: str) -> str:
    """Return a human-readable name for a competitive session code."""
    session_key = str(session_name).strip().upper()
    session_labels = {
        "SQ": "Sprint Qualifying",
        "SPRINT": "Sprint Race",
        "Q": "Qualifying",
        "R": "Race",
    }
    return session_labels.get(session_key, str(session_name).strip() or session_key)


def build_actual_qualifying_section(
    actual_grid: list[QualifyingGridEntry],
    *,
    session_name: str,
) -> dict[str, Any]:
    """Build dashboard payload for a completed qualifying-style session."""
    session_label = _session_display_name(session_name)
    return {
        "grid": list(actual_grid),
        "grid_source": "ACTUAL",
        "result_mode": "ACTUAL",
        "session_name": str(session_name).strip().upper(),
        "data_source": f"Actual {session_label} classification",
        "classification_note": (
            f"Showing ACTUAL {session_label.lower()} classification from the completed session."
        ),
        "classification_caption": (
            "No grid penalties are applied here. This is the raw post-session classification "
            "used as the race input grid."
        ),
    }


def build_actual_race_section(
    actual_results: list[QualifyingGridEntry],
    *,
    session_name: str,
) -> dict[str, Any]:
    """Build dashboard payload for a completed race-style session."""
    session_label = _session_display_name(session_name)
    return {
        "finish_order": list(actual_results),
        "result_mode": "ACTUAL",
        "session_name": str(session_name).strip().upper(),
        "classification_note": (
            f"Showing ACTUAL {session_label.lower()} classification from the completed session."
        ),
        "classification_caption": "This table shows the completed-session classification from FastF1.",
    }


def build_starting_grid_note(session_name: str) -> str:
    """Describe the actual grid source used as race-model input."""
    session_label = _session_display_name(session_name)
    return (
        f"Starting grid: ACTUAL {session_label.lower()} classification from the completed session "
        "(no penalties applied)."
    )


def _derive_race_input_confidence(
    qualifying_result: dict[str, Any],
    *,
    grid_source: str,
) -> float:
    """Estimate race input confidence from qualifying data quality and grid source."""
    return derive_race_input_confidence(qualifying_result, grid_source=grid_source)


def _resolve_dashboard_simulation_count(kind: str) -> int:
    """Resolve dashboard simulation count for qualifying or race predictions."""
    settings = get_prediction_precompute_config()
    if str(kind).strip().lower() == "qualifying":
        return int(settings.get("qualifying_n_simulations", 100))
    return int(settings.get("race_n_simulations", 100))


def _resolve_current_checkpoint_session(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
) -> str:
    """Return the latest completed weekend checkpoint or ``PRE`` when none exist."""
    from src.utils.session_detector import SessionDetector

    try:
        latest_session = SessionDetector().get_latest_completed_session(year, race_name, is_sprint)
    except Exception:
        latest_session = None

    checkpoint_session = str(latest_session or "").strip().upper()
    return checkpoint_session or "PRE"


def fetch_actual_competitive_results_if_completed(
    year: int,
    race_name: str,
    session_name: str,
) -> tuple[list[QualifyingGridEntry] | None, str]:
    """Return actual competitive-session results when the session is complete."""
    from src.data.actual_results_fetcher import (
        fetch_actual_session_results,
        get_competitive_session_completion_state,
    )
    from src.utils.grid_validation import validate_qualifying_grid

    logger.info(
        "FastF1 actual-results refresh check started: race=%s year=%s session=%s",
        race_name,
        year,
        session_name,
    )

    completion_state = get_competitive_session_completion_state(year, race_name, session_name)
    logger.info(
        "FastF1 actual-results completion status: race=%s year=%s session=%s state=%s",
        race_name,
        year,
        session_name,
        completion_state,
    )

    if completion_state == "completed":
        actual_results = fetch_actual_session_results(year, race_name, session_name)
        if actual_results:
            validated_results = validate_qualifying_grid(actual_results)
            logger.info(
                "Actual results resolved: race=%s year=%s session=%s drivers=%s",
                race_name,
                year,
                session_name,
                len(validated_results),
            )
            return validated_results, "ACTUAL"
        raise RuntimeError(
            f"FastF1 returned no {session_name} results for completed session at "
            f"{race_name} {year}; refusing to fall back to predicted output."
        )
    if completion_state == "incomplete":
        return None, "INCOMPLETE"

    raise CompetitiveSessionStatusUnavailableError(
        f"Could not verify completion state for {race_name} {year} {session_name}; "
        "refusing to fall back to predicted output."
    )


def _predict_race_with_optional_confidence(
    predictor: Any,
    *,
    qualifying_grid: list[QualifyingGridEntry],
    weather: str,
    race_name: str,
    year: int,
    input_confidence: float,
) -> dict[str, Any]:
    """Call predictor.predict_race with graceful fallback for legacy signatures."""
    kwargs = {
        "qualifying_grid": qualifying_grid,
        "weather": weather,
        "race_name": race_name,
        "n_simulations": _resolve_dashboard_simulation_count("race"),
        "year": year,
        "input_confidence": input_confidence,
    }
    try:
        return predictor.predict_race(**kwargs)
    except TypeError:
        kwargs.pop("input_confidence", None)
        return predictor.predict_race(**kwargs)


def _predict_sprint_race_with_optional_confidence(
    predictor: Any,
    *,
    sprint_quali_grid: list[QualifyingGridEntry],
    weather: str,
    race_name: str,
    input_confidence: float,
) -> dict[str, Any]:
    """Call predictor.predict_sprint_race with graceful fallback for legacy signatures."""
    kwargs = {
        "sprint_quali_grid": sprint_quali_grid,
        "weather": weather,
        "race_name": race_name,
        "n_simulations": _resolve_dashboard_simulation_count("race"),
        "input_confidence": input_confidence,
    }
    try:
        return predictor.predict_sprint_race(**kwargs)
    except TypeError:
        kwargs.pop("input_confidence", None)
        return predictor.predict_sprint_race(**kwargs)


def fetch_grid_if_available(
    year: int,
    race_name: str,
    session_name: str,
    predicted_grid: list,
) -> tuple[list, str]:
    """Fetch actual grid if session completed, otherwise use predicted grid."""
    from src.data.actual_results_fetcher import (
        fetch_actual_session_results,
        get_competitive_session_completion_state,
    )
    from src.utils.grid_validation import validate_qualifying_grid

    logger.info(
        "FastF1 session refresh check started: race=%s year=%s session=%s",
        race_name,
        year,
        session_name,
    )

    completion_state = get_competitive_session_completion_state(year, race_name, session_name)
    logger.info(
        "FastF1 session completion status: race=%s year=%s session=%s state=%s",
        race_name,
        year,
        session_name,
        completion_state,
    )

    if completion_state == "completed":
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
    if completion_state == "incomplete":
        validated_grid = validate_qualifying_grid(predicted_grid)
        logger.info(
            "Grid source resolved: PREDICTED race=%s year=%s session=%s drivers=%s",
            race_name,
            year,
            session_name,
            len(validated_grid),
        )
        return validated_grid, "PREDICTED"

    raise CompetitiveSessionStatusUnavailableError(
        f"Could not verify completion state for {race_name} {year} {session_name}; "
        "refusing to fall back to predicted grid."
    )


def _resolve_qualifying_section(
    predictor: Any,
    *,
    year: int,
    race_name: str,
    session_name: str,
    qualifying_stage: str,
    checkpoint_session_name: str,
) -> tuple[dict[str, Any], list[QualifyingGridEntry], str]:
    """Return actual completed-session results or a predicted qualifying payload."""
    actual_results, result_source = fetch_actual_competitive_results_if_completed(
        year=year,
        race_name=race_name,
        session_name=session_name,
    )
    if actual_results is not None:
        return (
            build_actual_qualifying_section(actual_results, session_name=session_name),
            actual_results,
            result_source,
        )

    predicted_result = predictor.predict_qualifying(
        year=year,
        race_name=race_name,
        qualifying_stage=qualifying_stage,
        n_simulations=_resolve_dashboard_simulation_count("qualifying"),
        practice_signal_mode="stored_profiles",
        checkpoint_session_name=checkpoint_session_name,
    )
    qualifying_grid, grid_source = fetch_grid_if_available(
        year,
        race_name,
        session_name,
        predicted_result["grid"],
    )
    if grid_source == "ACTUAL":
        return (
            build_actual_qualifying_section(qualifying_grid, session_name=session_name),
            qualifying_grid,
            grid_source,
        )

    predicted_result["grid_source"] = grid_source
    return predicted_result, qualifying_grid, grid_source


def _resolve_race_section(
    predictor: Any,
    *,
    year: int,
    race_name: str,
    session_name: str,
    qualifying_grid: list[QualifyingGridEntry],
    qualifying_grid_source: str,
    grid_session_name: str,
    weather: str,
    input_confidence: float,
) -> dict[str, Any]:
    """Return actual completed-session race results or a predicted race payload."""
    actual_results, _ = fetch_actual_competitive_results_if_completed(
        year=year,
        race_name=race_name,
        session_name=session_name,
    )
    if actual_results is not None:
        actual_payload = build_actual_race_section(actual_results, session_name=session_name)
        actual_payload["grid_source"] = qualifying_grid_source
        if qualifying_grid_source == "ACTUAL":
            actual_payload["starting_grid_note"] = build_starting_grid_note(grid_session_name)
        return actual_payload

    if str(session_name).strip().upper() == "SPRINT":
        race_result = _predict_sprint_race_with_optional_confidence(
            predictor,
            sprint_quali_grid=qualifying_grid,
            weather=weather,
            race_name=race_name,
            input_confidence=input_confidence,
        )
    else:
        race_result = _predict_race_with_optional_confidence(
            predictor,
            qualifying_grid=qualifying_grid,
            weather=weather,
            race_name=race_name,
            year=year,
            input_confidence=input_confidence,
        )
    race_result["grid_source"] = qualifying_grid_source
    if qualifying_grid_source == "ACTUAL":
        race_result["starting_grid_note"] = build_starting_grid_note(grid_session_name)
    return race_result


def run_prediction(
    race_name: str,
    weather: str,
    _artifact_versions: dict[str, tuple[int, str]],
    is_sprint: bool = False,
    year: int = 2026,
) -> dict[str, Any]:
    """
    Run full weekend cascade prediction.

    Warmup and automation use this to build persisted prediction artifacts.
    The interactive dashboard request path loads those stored outputs instead
    of calling this function on a user click.
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
    checkpoint_session_name = _resolve_current_checkpoint_session(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
    )
    predictor = build_checkpoint_overlay_predictor(
        base_predictor=predictor,
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session_name,
        is_sprint=is_sprint,
    )

    if is_sprint:
        # SPRINT WEEKEND CASCADE: SQ -> Sprint -> Main Quali -> Main Race

        sq_start = time.time()
        sq_result, sq_grid, sprint_grid_source = _resolve_qualifying_section(
            predictor,
            year=year,
            race_name=race_name,
            session_name="SQ",
            qualifying_stage="sprint",
            checkpoint_session_name=checkpoint_session_name,
        )
        timing["sprint_quali"] = time.time() - sq_start
        results["sprint_quali"] = sq_result

        sprint_start = time.time()
        sprint_input_confidence = _derive_race_input_confidence(
            sq_result,
            grid_source=sprint_grid_source,
        )

        sprint_result = _resolve_race_section(
            predictor,
            year=year,
            race_name=race_name,
            session_name="Sprint",
            qualifying_grid=sq_grid,
            qualifying_grid_source=sprint_grid_source,
            grid_session_name="SQ",
            weather=weather,
            input_confidence=sprint_input_confidence,
        )
        if str(sprint_result.get("result_mode", "")).upper() != "ACTUAL":
            sprint_result["input_confidence"] = round(float(sprint_input_confidence), 3)
        timing["sprint_race"] = time.time() - sprint_start
        results["sprint_race"] = sprint_result

        mq_start = time.time()
        mq_result, main_grid, main_grid_source = _resolve_qualifying_section(
            predictor,
            year=year,
            race_name=race_name,
            session_name="Q",
            qualifying_stage="main",
            checkpoint_session_name=checkpoint_session_name,
        )
        timing["main_quali"] = time.time() - mq_start
        results["main_quali"] = mq_result

        mr_start = time.time()
        main_race_input_confidence = _derive_race_input_confidence(
            mq_result,
            grid_source=main_grid_source,
        )
        main_race_input_confidence = cap_predicted_main_race_input_confidence(
            main_race_input_confidence,
            qualifying_result=mq_result,
            grid_source=main_grid_source,
            is_sprint_weekend=True,
            boundary_session_name=checkpoint_session_name,
        )

        main_race_result = _resolve_race_section(
            predictor,
            year=year,
            race_name=race_name,
            session_name="R",
            qualifying_grid=main_grid,
            qualifying_grid_source=main_grid_source,
            grid_session_name="Q",
            weather=weather,
            input_confidence=main_race_input_confidence,
        )
        if str(main_race_result.get("result_mode", "")).upper() != "ACTUAL":
            main_race_result["input_confidence"] = round(float(main_race_input_confidence), 3)
        timing["main_race"] = time.time() - mr_start
        results["main_race"] = main_race_result

    else:
        # NORMAL WEEKEND CASCADE: Quali -> Race

        quali_start = time.time()
        quali_result, quali_grid, grid_source = _resolve_qualifying_section(
            predictor,
            year=year,
            race_name=race_name,
            session_name="Q",
            qualifying_stage="main",
            checkpoint_session_name=checkpoint_session_name,
        )
        timing["qualifying"] = time.time() - quali_start
        results["qualifying"] = quali_result

        race_start = time.time()
        race_input_confidence = _derive_race_input_confidence(
            quali_result,
            grid_source=grid_source,
        )

        race_result = _resolve_race_section(
            predictor,
            year=year,
            race_name=race_name,
            session_name="R",
            qualifying_grid=quali_grid,
            qualifying_grid_source=grid_source,
            grid_session_name="Q",
            weather=weather,
            input_confidence=race_input_confidence,
        )
        if str(race_result.get("result_mode", "")).upper() != "ACTUAL":
            race_result["input_confidence"] = round(float(race_input_confidence), 3)
        timing["race"] = time.time() - race_start
        results["race"] = race_result

    timing["total"] = time.time() - overall_start

    for key in results:
        results[key]["timing"] = timing

    return results
