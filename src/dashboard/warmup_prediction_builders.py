"""Helpers for building warmup base features and weather overlays."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any

from src.utils.grid_penalties import apply_grid_penalties

from .prediction_checkpointing import (
    resolve_prediction_checkpoint_session,
    session_is_within_prediction_boundary,
)
from .race_context import attach_starting_grid_context

logger = logging.getLogger(__name__)


def _merge_section_timing(
    prediction_results: dict[str, dict[str, Any]],
    section_timing: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Attach shared timing fields to each rendered prediction section."""
    for section in prediction_results.values():
        if isinstance(section, dict):
            existing_timing = section.get("timing", {})
            merged_timing = dict(existing_timing) if isinstance(existing_timing, dict) else {}
            merged_timing.update(section_timing)
            section["timing"] = merged_timing
    return prediction_results


def compute_base_features(
    year: int,
    target_race: str,
    checkpoint: str,
    artifact_hash: str,
    boundary_signature: str,
    *,
    predictor: Any,
    is_sprint: bool,
    get_prediction_precompute_config_fn: Any,
    fetch_actual_competitive_results_if_completed_fn: Any,
    build_actual_qualifying_section_fn: Any,
    fetch_grid_if_available_fn: Any,
    derive_race_input_confidence_fn: Any,
    cap_predicted_main_race_input_confidence_fn: Any,
    logger_instance: logging.Logger = logger,
) -> dict[str, Any]:
    """Load weather-invariant race inputs once for one race checkpoint."""
    checkpoint_session = resolve_prediction_checkpoint_session(
        checkpoint,
        is_sprint=is_sprint,
    )
    logger_instance.debug(
        "Computing base features: year=%s race=%s checkpoint=%s artifact_hash=%s boundary=%s sprint=%s",
        int(year),
        str(target_race),
        checkpoint_session,
        str(artifact_hash),
        str(boundary_signature),
        bool(is_sprint),
    )
    base_start = time.time()
    precompute_settings = get_prediction_precompute_config_fn()
    qualifying_n_simulations = int(precompute_settings.get("qualifying_n_simulations", 100))

    if is_sprint:
        allow_sq_results = session_is_within_prediction_boundary(
            session_name="SQ",
            checkpoint_session=checkpoint_session,
            is_sprint=True,
        )
        sprint_quali_start = time.time()
        if allow_sq_results:
            sprint_actual_results, sprint_grid_source = (
                fetch_actual_competitive_results_if_completed_fn(
                    year=year,
                    race_name=target_race,
                    session_name="SQ",
                )
            )
        else:
            sprint_actual_results, sprint_grid_source = None, "PREDICTED"
        if sprint_actual_results is not None:
            sprint_grid = sprint_actual_results
            sprint_quali_payload = build_actual_qualifying_section_fn(
                sprint_grid,
                session_name="SQ",
            )
        else:
            sprint_quali = predictor.predict_qualifying(
                year=year,
                race_name=target_race,
                qualifying_stage="sprint",
                n_simulations=qualifying_n_simulations,
                practice_signal_mode="stored_profiles",
                checkpoint_session_name=checkpoint_session,
            )
            if allow_sq_results:
                sprint_grid, sprint_grid_source = fetch_grid_if_available_fn(
                    year,
                    target_race,
                    "SQ",
                    sprint_quali["grid"],
                )
                if sprint_grid_source == "ACTUAL":
                    sprint_quali_payload = build_actual_qualifying_section_fn(
                        sprint_grid,
                        session_name="SQ",
                    )
                else:
                    sprint_quali_payload = deepcopy(sprint_quali)
                    sprint_quali_payload["grid_source"] = sprint_grid_source
            else:
                sprint_grid = list(sprint_quali["grid"])
                sprint_grid_source = "PREDICTED"
                sprint_quali_payload = deepcopy(sprint_quali)
                sprint_quali_payload["grid_source"] = sprint_grid_source
        sprint_quali_elapsed = time.time() - sprint_quali_start
        sprint_input_confidence = derive_race_input_confidence_fn(
            sprint_quali_payload,
            grid_source=sprint_grid_source,
        )

        allow_main_quali_results = session_is_within_prediction_boundary(
            session_name="Q",
            checkpoint_session=checkpoint_session,
            is_sprint=True,
        )
        main_quali_start = time.time()
        if allow_main_quali_results:
            main_actual_results, main_grid_source = (
                fetch_actual_competitive_results_if_completed_fn(
                    year=year,
                    race_name=target_race,
                    session_name="Q",
                )
            )
        else:
            main_actual_results, main_grid_source = None, "PREDICTED"
        if main_actual_results is not None:
            main_grid = main_actual_results
            main_quali_payload = build_actual_qualifying_section_fn(main_grid, session_name="Q")
        else:
            main_quali = predictor.predict_qualifying(
                year=year,
                race_name=target_race,
                qualifying_stage="main",
                n_simulations=qualifying_n_simulations,
                practice_signal_mode="stored_profiles",
                checkpoint_session_name=checkpoint_session,
            )
            if allow_main_quali_results:
                main_grid, main_grid_source = fetch_grid_if_available_fn(
                    year,
                    target_race,
                    "Q",
                    main_quali["grid"],
                )
                if main_grid_source == "ACTUAL":
                    main_quali_payload = build_actual_qualifying_section_fn(
                        main_grid,
                        session_name="Q",
                    )
                else:
                    main_quali_payload = deepcopy(main_quali)
                    main_quali_payload["grid_source"] = main_grid_source
            else:
                main_grid = list(main_quali["grid"])
                main_grid_source = "PREDICTED"
                main_quali_payload = deepcopy(main_quali)
                main_quali_payload["grid_source"] = main_grid_source
        main_penalised = apply_grid_penalties(main_grid, race_name=target_race, year=year)
        main_quali_elapsed = time.time() - main_quali_start
        main_input_confidence = derive_race_input_confidence_fn(
            main_quali_payload,
            grid_source=main_grid_source,
        )
        main_input_confidence = cap_predicted_main_race_input_confidence_fn(
            main_input_confidence,
            qualifying_result=main_quali_payload,
            grid_source=main_grid_source,
            is_sprint_weekend=True,
            boundary_session_name=checkpoint_session,
        )

        return {
            "is_sprint": True,
            "sprint_quali": sprint_quali_payload,
            "sprint_grid_for_race": sprint_grid,
            "sprint_race_input_confidence": float(sprint_input_confidence),
            "main_quali": main_quali_payload,
            "main_grid_for_race": main_penalised.grid,
            "main_grid_penalties": [penalty.to_dict() for penalty in main_penalised.applied],
            "main_race_input_confidence": float(main_input_confidence),
            "timing": {
                "sprint_quali": float(sprint_quali_elapsed),
                "main_quali": float(main_quali_elapsed),
                "base_total": float(time.time() - base_start),
            },
        }

    qualifying_start = time.time()
    allow_main_quali_results = session_is_within_prediction_boundary(
        session_name="Q",
        checkpoint_session=checkpoint_session,
        is_sprint=False,
    )
    if allow_main_quali_results:
        actual_qualifying, grid_source = fetch_actual_competitive_results_if_completed_fn(
            year=year,
            race_name=target_race,
            session_name="Q",
        )
    else:
        actual_qualifying, grid_source = None, "PREDICTED"
    if actual_qualifying is not None:
        qualifying_grid = actual_qualifying
        qualifying_payload = build_actual_qualifying_section_fn(qualifying_grid, session_name="Q")
    else:
        qualifying = predictor.predict_qualifying(
            year=year,
            race_name=target_race,
            qualifying_stage="main",
            n_simulations=qualifying_n_simulations,
            practice_signal_mode="stored_profiles",
            checkpoint_session_name=checkpoint_session,
        )
        if allow_main_quali_results:
            qualifying_grid, grid_source = fetch_grid_if_available_fn(
                year,
                target_race,
                "Q",
                qualifying["grid"],
            )
            if grid_source == "ACTUAL":
                qualifying_payload = build_actual_qualifying_section_fn(
                    qualifying_grid,
                    session_name="Q",
                )
            else:
                qualifying_payload = deepcopy(qualifying)
                qualifying_payload["grid_source"] = grid_source
        else:
            qualifying_grid = list(qualifying["grid"])
            grid_source = "PREDICTED"
            qualifying_payload = deepcopy(qualifying)
            qualifying_payload["grid_source"] = grid_source
    penalised = apply_grid_penalties(qualifying_grid, race_name=target_race, year=year)
    qualifying_elapsed = time.time() - qualifying_start
    race_input_confidence = derive_race_input_confidence_fn(
        qualifying_payload,
        grid_source=grid_source,
    )

    return {
        "is_sprint": False,
        "qualifying": qualifying_payload,
        "qualifying_grid_for_race": penalised.grid,
        "qualifying_grid_penalties": [penalty.to_dict() for penalty in penalised.applied],
        "race_input_confidence": float(race_input_confidence),
        "timing": {
            "qualifying": float(qualifying_elapsed),
            "base_total": float(time.time() - base_start),
        },
    }


def compute_weather_predictions(
    base_features: dict[str, Any],
    weather: str,
    *,
    predictor: Any,
    year: int,
    target_race: str,
    valid_weather_scenarios: set[str] | frozenset[str] | tuple[str, ...],
    fetch_actual_competitive_results_if_completed_fn: Any,
    build_actual_race_section_fn: Any,
    predict_sprint_race_with_optional_confidence_fn: Any,
    predict_race_with_optional_confidence_fn: Any,
    build_starting_grid_note_fn: Any,
) -> dict[str, Any]:
    """Apply a weather scenario to precomputed warmup base features."""
    normalized_weather = str(weather).strip().lower()
    if normalized_weather not in valid_weather_scenarios:
        raise ValueError(
            f"Invalid weather scenario '{weather}'. Expected one of: {sorted(valid_weather_scenarios)}"
        )

    section_timing: dict[str, float] = {}
    if bool(base_features.get("is_sprint")):
        sprint_race_start = time.time()
        sprint_actual_results, _ = fetch_actual_competitive_results_if_completed_fn(
            year=year,
            race_name=target_race,
            session_name="Sprint",
        )
        if sprint_actual_results is not None:
            sprint_race = build_actual_race_section_fn(sprint_actual_results, session_name="Sprint")
        else:
            sprint_race = predict_sprint_race_with_optional_confidence_fn(
                predictor,
                sprint_quali_grid=deepcopy(base_features["sprint_grid_for_race"]),
                weather=normalized_weather,
                race_name=target_race,
                input_confidence=float(base_features.get("sprint_race_input_confidence", 0.0)),
            )
        sprint_race_elapsed = time.time() - sprint_race_start
        sprint_input_confidence = float(base_features.get("sprint_race_input_confidence", 0.0))
        sprint_grid_source = (
            str(base_features.get("sprint_quali", {}).get("grid_source", "PREDICTED"))
            .strip()
            .upper()
            or "PREDICTED"
        )
        sprint_race["grid_source"] = sprint_grid_source
        attach_starting_grid_context(sprint_race, base_features["sprint_grid_for_race"], "SQ")
        if sprint_grid_source == "ACTUAL":
            sprint_race["starting_grid_note"] = build_starting_grid_note_fn("SQ")
        if str(sprint_race.get("result_mode", "")).upper() != "ACTUAL":
            sprint_race["input_confidence"] = round(sprint_input_confidence, 3)

        main_race_start = time.time()
        main_actual_results, _ = fetch_actual_competitive_results_if_completed_fn(
            year=year,
            race_name=target_race,
            session_name="R",
        )
        if main_actual_results is not None:
            main_race = build_actual_race_section_fn(main_actual_results, session_name="R")
        else:
            main_race = predict_race_with_optional_confidence_fn(
                predictor,
                qualifying_grid=deepcopy(base_features["main_grid_for_race"]),
                weather=normalized_weather,
                race_name=target_race,
                year=year,
                input_confidence=float(base_features.get("main_race_input_confidence", 0.0)),
            )
        main_race_elapsed = time.time() - main_race_start
        main_input_confidence = float(base_features.get("main_race_input_confidence", 0.0))
        main_grid_source = str(base_features.get("main_quali", {}).get("grid_source", "PREDICTED"))
        main_grid_source = main_grid_source.strip().upper() or "PREDICTED"
        main_race["grid_source"] = main_grid_source
        attach_starting_grid_context(main_race, base_features["main_grid_for_race"], "Q")
        if base_features.get("main_grid_penalties"):
            main_race["grid_penalties"] = list(base_features["main_grid_penalties"])
        if main_grid_source == "ACTUAL":
            main_race["starting_grid_note"] = build_starting_grid_note_fn("Q")
        if str(main_race.get("result_mode", "")).upper() != "ACTUAL":
            main_race["input_confidence"] = round(main_input_confidence, 3)

        section_timing = {
            "sprint_quali": float(base_features.get("timing", {}).get("sprint_quali", 0.0)),
            "sprint_race": float(sprint_race_elapsed),
            "main_quali": float(base_features.get("timing", {}).get("main_quali", 0.0)),
            "main_race": float(main_race_elapsed),
        }
        section_timing["total"] = (
            section_timing["sprint_quali"]
            + section_timing["sprint_race"]
            + section_timing["main_quali"]
            + section_timing["main_race"]
        )

        prediction_results = {
            "sprint_quali": deepcopy(base_features["sprint_quali"]),
            "sprint_race": sprint_race,
            "main_quali": deepcopy(base_features["main_quali"]),
            "main_race": main_race,
        }
        return _merge_section_timing(prediction_results, section_timing)

    race_start = time.time()
    actual_race_results, _ = fetch_actual_competitive_results_if_completed_fn(
        year=year,
        race_name=target_race,
        session_name="R",
    )
    if actual_race_results is not None:
        race = build_actual_race_section_fn(actual_race_results, session_name="R")
    else:
        race = predict_race_with_optional_confidence_fn(
            predictor,
            qualifying_grid=deepcopy(base_features["qualifying_grid_for_race"]),
            weather=normalized_weather,
            race_name=target_race,
            year=year,
            input_confidence=float(base_features.get("race_input_confidence", 0.0)),
        )
    race_elapsed = time.time() - race_start
    race_input_confidence = float(base_features.get("race_input_confidence", 0.0))
    race_grid_source = str(base_features.get("qualifying", {}).get("grid_source", "PREDICTED"))
    race_grid_source = race_grid_source.strip().upper() or "PREDICTED"
    race["grid_source"] = race_grid_source
    attach_starting_grid_context(race, base_features["qualifying_grid_for_race"], "Q")
    if base_features.get("qualifying_grid_penalties"):
        race["grid_penalties"] = list(base_features["qualifying_grid_penalties"])
    if race_grid_source == "ACTUAL":
        race["starting_grid_note"] = build_starting_grid_note_fn("Q")
    if str(race.get("result_mode", "")).upper() != "ACTUAL":
        race["input_confidence"] = round(race_input_confidence, 3)

    section_timing = {
        "qualifying": float(base_features.get("timing", {}).get("qualifying", 0.0)),
        "race": float(race_elapsed),
    }
    section_timing["total"] = section_timing["qualifying"] + section_timing["race"]

    prediction_results = {
        "qualifying": deepcopy(base_features["qualifying"]),
        "race": race,
    }
    return _merge_section_timing(prediction_results, section_timing)
