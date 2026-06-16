"""Helpers for checkpoint-scoped prediction saves and summaries."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from src.utils.accuracy_targets import (
    TARGET_GRAND_PRIX_RACE,
    TARGET_MAIN_QUALIFYING,
    TARGET_SPRINT_QUALIFYING,
    TARGET_SPRINT_RACE,
    eligible_target_keys,
    legacy_target_keys_for_prediction,
    mean_confidence_from_rows,
    target_session_name,
    weekend_format_name,
)
from src.utils.model_version import get_model_version

PredictionResults = dict[str, Any]
logger = logging.getLogger(__name__)
# ``Q`` is valid for race-only forecasts after qualifying; the qualifying target
# itself is closed by the target-eligibility table.
_NON_COMPETITIVE_PREDICTION_CHECKPOINTS = {
    False: ("PRE", "FP1", "FP2", "FP3", "Q"),
    True: ("PRE", "FP1", "SQ", "Q"),
}
_CHECKPOINT_ORDER = {
    "PRE": 0,
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "SQ": 4,
    "SPRINT": 5,
    "Q": 6,
    "R": 7,
}
_TARGET_SECTION_BINDINGS = {
    False: {
        TARGET_MAIN_QUALIFYING: ("qualifying", "grid"),
        TARGET_GRAND_PRIX_RACE: ("race", "finish_order"),
    },
    True: {
        TARGET_SPRINT_QUALIFYING: ("sprint_quali", "grid"),
        TARGET_SPRINT_RACE: ("sprint_race", "finish_order"),
        TARGET_MAIN_QUALIFYING: ("main_quali", "grid"),
        TARGET_GRAND_PRIX_RACE: ("main_race", "finish_order"),
    },
}

_QUALIFYING_DIAGNOSTIC_KEYS = (
    "data_source",
    "data_regime",
    "blend_used",
    "testing_fallback_used",
    "data_confidence_score",
    "fp_blend_weight_used",
    "qualifying_stage",
    "weather",
    "practice_signal_mode_used",
    "practice_signal_checkpoint",
    "characteristics_profile_used",
    "teams_with_characteristics_profile",
    "qualifying_residual_model_used",
    "qualifying_residual_mean_abs_adjustment",
)
_RACE_DIAGNOSTIC_KEYS = (
    "data_regime",
    "grid_source",
    "input_confidence",
    "characteristics_profile_used",
    "teams_with_characteristics_profile",
    "race_residual_model_used",
    "race_residual_mean_abs_adjustment",
)
_NESTED_RACE_DIAGNOSTIC_KEYS = (
    "track_temperature_context",
    "weather_feature_context",
)


def allowed_prediction_checkpoints(*, is_sprint: bool) -> tuple[str, ...]:
    """Return checkpoints that are allowed to inform a fresh prediction."""
    return _NON_COMPETITIVE_PREDICTION_CHECKPOINTS[bool(is_sprint)]


def _json_safe_diagnostic_value(value: Any, *, max_items: int = 12, depth: int = 2) -> Any:
    """Return a bounded JSON-safe diagnostic value."""
    if value is None or isinstance(value, bool | int | float | str):
        return value

    if isinstance(value, Mapping) and depth > 0:
        bounded: dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= max_items:
                bounded["_truncated"] = True
                break
            key = str(raw_key).strip()[:80]
            if not key:
                continue
            bounded[key] = _json_safe_diagnostic_value(
                raw_value,
                max_items=max_items,
                depth=depth - 1,
            )
        return bounded

    if isinstance(value, list | tuple) and depth > 0:
        return [
            _json_safe_diagnostic_value(item, max_items=max_items, depth=depth - 1)
            for item in list(value)[:max_items]
        ]

    return str(value)[:200]


def _section_model_diagnostics(
    section: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    nested_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Extract bounded model diagnostics from one prediction section."""
    diagnostics: dict[str, Any] = {}
    for key in keys:
        if key in section:
            diagnostics[key] = _json_safe_diagnostic_value(section.get(key))
    for key in nested_keys:
        value = section.get(key)
        if isinstance(value, Mapping):
            diagnostics[key] = _json_safe_diagnostic_value(value)

    compound_strategies = section.get("compound_strategies")
    if isinstance(compound_strategies, Mapping):
        diagnostics["compound_strategy_count"] = len(compound_strategies)
    pit_lap_distribution = section.get("pit_lap_distribution")
    if isinstance(pit_lap_distribution, Mapping):
        diagnostics["pit_lap_distribution_count"] = len(pit_lap_distribution)

    return diagnostics


def prediction_model_diagnostics_for_sections(
    *,
    qualifying_section: Mapping[str, Any],
    race_section: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact model diagnostics suitable for saved prediction metadata."""
    return {
        "model_diagnostics_schema_version": 1,
        "qualifying_model_diagnostics": _section_model_diagnostics(
            qualifying_section,
            keys=_QUALIFYING_DIAGNOSTIC_KEYS,
        ),
        "race_model_diagnostics": _section_model_diagnostics(
            race_section,
            keys=_RACE_DIAGNOSTIC_KEYS,
            nested_keys=_NESTED_RACE_DIAGNOSTIC_KEYS,
        ),
    }


def prediction_payload_for_session(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    session_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pick the qualifying and race payload pair that belongs to one checkpoint."""
    if not is_sprint:
        return (
            prediction_results["qualifying"]["grid"],
            prediction_results["race"]["finish_order"],
            prediction_results.get("qualifying", {}).get("fp_blend_info", {}),
        )

    session_name_upper = resolve_prediction_checkpoint_session(session_name, is_sprint=is_sprint)
    sprint_phase_sessions = {"PRE", "FP1", "SQ", "SPRINT"}
    if session_name_upper in sprint_phase_sessions:
        return (
            prediction_results["sprint_quali"]["grid"],
            prediction_results["sprint_race"]["finish_order"],
            prediction_results.get("sprint_quali", {}).get("fp_blend_info", {}),
        )

    return (
        prediction_results["main_quali"]["grid"],
        prediction_results["main_race"]["finish_order"],
        prediction_results.get("main_quali", {}).get("fp_blend_info", {}),
    )


def prediction_sections_for_session(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    session_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the qualifying and race sections for one checkpoint."""
    if not is_sprint:
        return (
            prediction_results.get("qualifying", {}),
            prediction_results.get("race", {}),
        )

    session_name_upper = resolve_prediction_checkpoint_session(session_name, is_sprint=is_sprint)
    if session_name_upper in {"PRE", "FP1", "SQ", "SPRINT"}:
        return (
            prediction_results.get("sprint_quali", {}),
            prediction_results.get("sprint_race", {}),
        )

    return (
        prediction_results.get("main_quali", {}),
        prediction_results.get("main_race", {}),
    )


def resolve_prediction_checkpoint_session(
    latest_session: Any,
    *,
    is_sprint: bool,
) -> str:
    """Clamp the latest completed session to the last allowed prediction checkpoint."""
    normalized = str(latest_session or "").strip().upper()
    if not normalized:
        return "PRE"

    latest_order = _CHECKPOINT_ORDER.get(normalized, -1)
    for checkpoint in reversed(allowed_prediction_checkpoints(is_sprint=is_sprint)):
        if latest_order >= _CHECKPOINT_ORDER[checkpoint]:
            return checkpoint
    return "PRE"


def session_is_within_prediction_boundary(
    *,
    session_name: str,
    checkpoint_session: str,
    is_sprint: bool,
) -> bool:
    """Return whether session data is allowed to inform predictions at the checkpoint."""
    normalized_session = str(session_name or "").strip().upper()
    resolved_checkpoint = resolve_prediction_checkpoint_session(
        checkpoint_session,
        is_sprint=is_sprint,
    )
    return _CHECKPOINT_ORDER.get(normalized_session, -1) <= _CHECKPOINT_ORDER.get(
        resolved_checkpoint,
        -1,
    )


def prediction_targets_for_checkpoint(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    session_name: str,
) -> dict[str, dict[str, Any]]:
    """Collect still-forecastable targets for one checkpoint save."""
    checkpoint_session = resolve_prediction_checkpoint_session(
        session_name,
        is_sprint=is_sprint,
    )
    target_predictions: dict[str, dict[str, Any]] = {}
    section_bindings = _TARGET_SECTION_BINDINGS[bool(is_sprint)]

    for target_key in eligible_target_keys(checkpoint_session, is_sprint):
        section_name, rows_key = section_bindings[target_key]
        section = prediction_results.get(section_name, {})
        if not isinstance(section, dict):
            continue
        predicted_order = section.get(rows_key, [])
        if not isinstance(predicted_order, list) or not predicted_order:
            continue
        result_mode = str(section.get("result_mode", "PREDICTED")).strip().upper()
        if result_mode == "ACTUAL":
            continue
        fp_blend_info = section.get("fp_blend_info")
        target_predictions[target_key] = {
            "target_session": target_session_name(target_key),
            "predicted_order": predicted_order,
            "result_mode": result_mode,
            "grid_source": str(section.get("grid_source", "PREDICTED")).strip().upper(),
            "fp_blend_info": fp_blend_info if isinstance(fp_blend_info, dict) else {},
            "mean_confidence": mean_confidence_from_rows(predicted_order),
            "eligible_at_save": True,
        }
    return target_predictions


def mean_confidence(entries: Any) -> float | None:
    """Return mean row confidence when entries expose it."""
    return mean_confidence_from_rows(entries)


def persist_prediction_checkpoint_summary(
    *,
    logger_instance: Any,
    prediction_results: PredictionResults,
    year: int,
    race_name: str,
    session_name: str,
    weather: str,
    is_sprint: bool,
    target_predictions: dict[str, dict[str, Any]],
    logger: logging.Logger = logger,
) -> None:
    """Persist a small checkpoint summary for trend views."""
    artifact_store = getattr(logger_instance, "artifact_store", None)
    if artifact_store is None or not hasattr(artifact_store, "save_artifact"):
        return

    qualifying_section, race_section = prediction_sections_for_session(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=session_name,
    )
    if not isinstance(qualifying_section, dict):
        qualifying_section = {}
    if not isinstance(race_section, dict):
        race_section = {}
    qualifying_grid = qualifying_section.get("grid", [])
    race_finish_order = race_section.get("finish_order", [])

    payload = {
        "metadata": {
            "year": int(year),
            "race_name": str(race_name),
            "session_name": str(session_name).strip().upper(),
            "weather": str(weather).strip().lower(),
            "is_sprint_weekend": bool(is_sprint),
            "weekend_format": weekend_format_name(is_sprint),
            "model_version": get_model_version(),
            "generated_at": datetime.now(UTC).isoformat(),
            "source": "dashboard_live_prediction",
        },
        "qualifying": {
            "data_source": qualifying_section.get("data_source"),
            "grid_source": qualifying_section.get("grid_source", "PREDICTED"),
            "data_confidence_score": qualifying_section.get("data_confidence_score"),
            "mean_confidence": mean_confidence(qualifying_grid),
            "driver_count": len(qualifying_grid) if isinstance(qualifying_grid, list) else 0,
        },
        "race": {
            "grid_source": race_section.get("grid_source", "PREDICTED"),
            "input_confidence": race_section.get("input_confidence"),
            "mean_confidence": mean_confidence(race_finish_order),
            "driver_count": len(race_finish_order) if isinstance(race_finish_order, list) else 0,
        },
        "fp_blend_info": qualifying_section.get("fp_blend_info", {}),
        "targets": {
            target_key: {
                "target_session": target_payload.get("target_session"),
                "result_mode": target_payload.get("result_mode"),
                "grid_source": target_payload.get("grid_source"),
                "mean_confidence": target_payload.get("mean_confidence"),
                "driver_count": len(target_payload.get("predicted_order", [])),
            }
            for target_key, target_payload in target_predictions.items()
        },
    }

    try:
        artifact_store.save_artifact(
            artifact_type="prediction_checkpoint",
            artifact_key=f"{int(year)}::{race_name}::{str(session_name).strip().upper()}",
            data=payload,
            version=1,
        )
    except Exception as exc:
        logger.warning(
            "Could not persist prediction checkpoint summary for %s %s %s: %s",
            year,
            race_name,
            session_name,
            exc,
        )


def save_prediction_if_enabled_core(
    *,
    enable_logging: bool,
    prediction_results: PredictionResults,
    is_sprint: bool,
    race_name: str,
    weather: str,
    year: int,
    detector_factory: Callable[[], Any],
    prediction_logger_factory: Callable[[], Any],
    st_module: Any,
    checkpoint_session_override: str | None = None,
    logger: logging.Logger = logger,
) -> None:
    """Save the shown prediction for later accuracy tracking."""
    if not enable_logging:
        return

    logger_inst = prediction_logger_factory()
    prediction_context = prediction_results.get("_prediction_context", {})
    prediction_boundary_session = ""
    if isinstance(prediction_context, dict):
        prediction_boundary_session = str(
            prediction_context.get("boundary_session_name", "")
        ).strip()
    checkpoint_override = str(checkpoint_session_override or "").strip().upper()
    if prediction_boundary_session:
        checkpoint_session = resolve_prediction_checkpoint_session(
            prediction_boundary_session,
            is_sprint=is_sprint,
        )
    elif checkpoint_override:
        checkpoint_session = resolve_prediction_checkpoint_session(
            checkpoint_override,
            is_sprint=is_sprint,
        )
    else:
        detector = detector_factory()
        latest_session = detector.get_latest_completed_session(year, race_name, is_sprint)
        checkpoint_session = resolve_prediction_checkpoint_session(
            latest_session,
            is_sprint=is_sprint,
        )

    if logger_inst.has_prediction_for_session(year, race_name, checkpoint_session):
        st_module.info(f"Prediction for {checkpoint_session} already saved (max 1 per session)")
        return

    target_predictions = prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=checkpoint_session,
    )
    if not target_predictions:
        st_module.info(
            "Skipped saving "
            f"{checkpoint_session} checkpoint because every tracked target already has "
            "completed-session results. Accuracy requires a forecast saved before that "
            "target finishes."
        )
        return

    try:
        quali_grid, race_finish, fp_blend_info = prediction_payload_for_session(
            prediction_results=prediction_results,
            is_sprint=is_sprint,
            session_name=checkpoint_session,
        )
        qualifying_section, race_section = prediction_sections_for_session(
            prediction_results=prediction_results,
            is_sprint=is_sprint,
            session_name=checkpoint_session,
        )
        qualifying_target, race_target = legacy_target_keys_for_prediction(
            checkpoint_session,
            is_sprint=is_sprint,
        )

        logger_inst.save_prediction(
            year=year,
            race_name=race_name,
            session_name=checkpoint_session,
            qualifying_prediction=quali_grid,
            race_prediction=race_finish,
            weather=weather,
            fp_blend_info=fp_blend_info,
            target_predictions=target_predictions,
            metadata={
                "source": "dashboard_live_prediction",
                "weekend_format": weekend_format_name(is_sprint),
                "top_level_qualifying_target": qualifying_target,
                "top_level_race_target": race_target,
                "top_level_qualifying_eligible_at_save": (
                    str(qualifying_section.get("result_mode", "PREDICTED")).strip().upper()
                    != "ACTUAL"
                ),
                "top_level_race_eligible_at_save": (
                    str(race_section.get("result_mode", "PREDICTED")).strip().upper() != "ACTUAL"
                ),
                "top_level_qualifying_result_mode": qualifying_section.get(
                    "result_mode",
                    "PREDICTED",
                ),
                "top_level_race_result_mode": race_section.get("result_mode", "PREDICTED"),
                "top_level_qualifying_grid_source": qualifying_section.get(
                    "grid_source",
                    "PREDICTED",
                ),
                "top_level_race_grid_source": race_section.get("grid_source", "PREDICTED"),
                **prediction_model_diagnostics_for_sections(
                    qualifying_section=qualifying_section,
                    race_section=race_section,
                ),
            },
        )
        persist_prediction_checkpoint_summary(
            logger_instance=logger_inst,
            prediction_results=prediction_results,
            year=year,
            race_name=race_name,
            session_name=checkpoint_session,
            weather=weather,
            is_sprint=is_sprint,
            target_predictions=target_predictions,
            logger=logger,
        )
        st_module.info(f"Prediction saved for accuracy tracking (checkpoint {checkpoint_session})")
    except Exception as exc:
        st_module.warning(f"Could not save prediction: {exc}")
