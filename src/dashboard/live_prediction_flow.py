"""Serve warmed dashboard predictions and checkpoint-save metadata."""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha1
from threading import RLock
from typing import Any, Protocol

from src.dashboard.precomputed_predictions import (
    compute_artifact_hash,
    get_prediction_precompute_config,
    load_precompute_horizon_index,
    load_precomputed_prediction,
)
from src.dashboard.update_flow import _boundary_signature, _build_event_boundary_snapshot
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
from src.utils.operational_observability import (
    drain_recent_alerts,
    snapshot_counters,
)

from . import prediction_boundary as _prediction_boundary
from . import prediction_serving as _prediction_serving

PredictionResults = dict[str, Any]
ArtifactVersion = tuple[int, str]
ArtifactVersions = dict[str, ArtifactVersion]


class PrecomputedPredictionUnavailableError(RuntimeError):
    """Raised when the dashboard cannot serve a request from warmed artifacts."""


class PredictionRunFn(Protocol):
    """Build a full weekend prediction payload."""

    def __call__(
        self,
        race_name: str,
        weather: str,
        artifact_versions: ArtifactVersions,
        /,
        is_sprint: bool = False,
        year: int = 2026,
    ) -> PredictionResults: ...


class AutoUpdateIfNeededFn(Protocol):
    """Run a race-update check."""

    def __call__(self, *, year: int, force_recheck: bool = False) -> None: ...


class DetectEventBoundaryRefreshFn(Protocol):
    """Return current event-boundary state for one race."""

    def __call__(
        self,
        *,
        year: int,
        race_name: str,
        is_sprint: bool,
        session_detector: Any | None = None,
    ) -> dict[str, Any]: ...


class AutoUpdatePracticeIfNeededFn(Protocol):
    """Run a practice-update check."""

    def __call__(
        self,
        *,
        year: int,
        race_name: str,
        is_sprint: bool,
        force_recheck: bool = False,
        session_detector: Any | None = None,
    ) -> dict[str, Any]: ...


class GetArtifactVersionsFn(Protocol):
    """Return the current artifact-version map."""

    def __call__(self, *, year: int) -> ArtifactVersions: ...


logger = logging.getLogger(__name__)
_PREDICTION_RESULT_CACHE_MAX_ENTRIES = 24
_prediction_result_cache: OrderedDict[str, PredictionResults] = OrderedDict()
_prediction_result_cache_lock = RLock()
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


def clear_prediction_result_cache() -> None:
    """Clear the in-memory LRU of served predictions."""
    with _prediction_result_cache_lock:
        _prediction_result_cache.clear()


def _prediction_cache_key(
    *,
    year: int,
    race_name: str,
    weather: str,
    is_sprint: bool,
    artifact_versions: ArtifactVersions,
    boundary_signature: str,
) -> str:
    """Build the RAM-cache key for one served prediction."""
    payload = {
        "year": year,
        "race_name": race_name,
        "weather": weather,
        "is_sprint": is_sprint,
        "artifact_versions": {key: list(value) for key, value in sorted(artifact_versions.items())},
        "boundary_signature": boundary_signature,
    }
    return sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _get_cached_prediction(cache_key: str) -> PredictionResults | None:
    """Return a cached prediction and refresh its LRU position."""
    with _prediction_result_cache_lock:
        cached = _prediction_result_cache.get(cache_key)
        if cached is None:
            return None
        _prediction_result_cache.move_to_end(cache_key)
        return cached


def _store_cached_prediction(cache_key: str, prediction_results: PredictionResults) -> None:
    """Store a prediction in the bounded in-memory cache."""
    with _prediction_result_cache_lock:
        _prediction_result_cache[cache_key] = prediction_results
        _prediction_result_cache.move_to_end(cache_key)
        while len(_prediction_result_cache) > _PREDICTION_RESULT_CACHE_MAX_ENTRIES:
            _prediction_result_cache.popitem(last=False)


def _prediction_persisted_updated_at(prediction_results: PredictionResults) -> str:
    """Read the persisted write timestamp from prediction context."""
    prediction_context = prediction_results.get("_prediction_context")
    if not isinstance(prediction_context, dict):
        return ""
    return str(prediction_context.get("persisted_updated_at", "")).strip()


def _cached_prediction_matches_persisted(
    *,
    cached_prediction: PredictionResults,
    persisted_prediction: PredictionResults,
) -> bool:
    """Return ``True`` when the cached payload matches the latest stored write."""
    cached_updated_at = _prediction_persisted_updated_at(cached_prediction)
    persisted_updated_at = _prediction_persisted_updated_at(persisted_prediction)
    if not cached_updated_at or not persisted_updated_at:
        return False
    return cached_updated_at == persisted_updated_at


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
        # Prefer the boundary that produced the shown payload; overrides can lag.
        checkpoint_session = _resolve_prediction_checkpoint_session(prediction_boundary_session)
    elif checkpoint_override:
        checkpoint_session = _resolve_prediction_checkpoint_session(checkpoint_override)
    else:
        detector = detector_factory()
        latest_session = detector.get_latest_completed_session(year, race_name, is_sprint)
        checkpoint_session = _resolve_prediction_checkpoint_session(latest_session)

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
            f"Skipped saving {checkpoint_session} checkpoint because every tracked target "
            "already has completed-session results. Accuracy requires a forecast saved before "
            "that target finishes."
        )
        return

    try:
        quali_grid, race_finish, fp_blend_info = prediction_payload_for_session(
            prediction_results=prediction_results,
            is_sprint=is_sprint,
            session_name=checkpoint_session,
        )
        qualifying_section, race_section = _prediction_sections_for_session(
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
            },
        )
        _persist_prediction_checkpoint_summary(
            logger_instance=logger_inst,
            prediction_results=prediction_results,
            year=year,
            race_name=race_name,
            session_name=checkpoint_session,
            weather=weather,
            is_sprint=is_sprint,
            target_predictions=target_predictions,
        )
        st_module.info(f"Prediction saved for accuracy tracking (checkpoint {checkpoint_session})")
    except Exception as exc:
        st_module.warning(f"Could not save prediction: {exc}")


def prediction_payload_for_session(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    session_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pick the qualifying/race payload pair that belongs to one checkpoint."""
    if not is_sprint:
        return (
            prediction_results["qualifying"]["grid"],
            prediction_results["race"]["finish_order"],
            prediction_results.get("qualifying", {}).get("fp_blend_info", {}),
        )

    session_name_upper = str(session_name).strip().upper()
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


def _prediction_sections_for_session(
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

    session_name_upper = str(session_name).strip().upper()
    if session_name_upper in {"PRE", "FP1", "SQ", "SPRINT"}:
        return (
            prediction_results.get("sprint_quali", {}),
            prediction_results.get("sprint_race", {}),
        )

    return (
        prediction_results.get("main_quali", {}),
        prediction_results.get("main_race", {}),
    )


def _resolve_prediction_checkpoint_session(latest_session: Any) -> str:
    """Map the latest completed session into the stored checkpoint key."""
    normalized = str(latest_session or "").strip().upper()
    return normalized if normalized else "PRE"


def prediction_targets_for_checkpoint(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    session_name: str,
) -> dict[str, dict[str, Any]]:
    """Collect still-forecastable targets for one checkpoint save."""
    checkpoint_session = _resolve_prediction_checkpoint_session(session_name)
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


def _mean_confidence(entries: Any) -> float | None:
    """Return mean row confidence when entries expose it."""
    return mean_confidence_from_rows(entries)


def _persist_prediction_checkpoint_summary(
    *,
    logger_instance: Any,
    prediction_results: PredictionResults,
    year: int,
    race_name: str,
    session_name: str,
    weather: str,
    is_sprint: bool,
    target_predictions: dict[str, dict[str, Any]],
) -> None:
    """Persist a small checkpoint summary for trend views."""
    artifact_store = getattr(logger_instance, "artifact_store", None)
    if artifact_store is None or not hasattr(artifact_store, "save_artifact"):
        return

    qualifying_section, race_section = _prediction_sections_for_session(
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
            "generated_at": datetime.now(UTC).isoformat(),
            "source": "dashboard_live_prediction",
        },
        "qualifying": {
            "data_source": qualifying_section.get("data_source"),
            "grid_source": qualifying_section.get("grid_source", "PREDICTED"),
            "data_confidence_score": qualifying_section.get("data_confidence_score"),
            "mean_confidence": _mean_confidence(qualifying_grid),
            "driver_count": len(qualifying_grid) if isinstance(qualifying_grid, list) else 0,
        },
        "race": {
            "grid_source": race_section.get("grid_source", "PREDICTED"),
            "input_confidence": race_section.get("input_confidence"),
            "mean_confidence": _mean_confidence(race_finish_order),
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


def render_prediction_results_core(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    display_prediction_result_fn: Callable[[PredictionResults, str, bool], None],
    st_module: Any,
    prediction_cache_hit: bool = False,
    pipeline_timing: Mapping[str, Any] | None = None,
) -> None:
    """Render saved prediction sections for the active weekend format."""

    def _section_title(section: Mapping[str, Any], default_title: str) -> str:
        """Swap Prediction for Result once a section becomes actual."""
        result_mode = str(section.get("result_mode", "")).strip().upper()
        if result_mode == "ACTUAL":
            return default_title.replace("Prediction", "Result")
        return default_title

    first_result = list(prediction_results.values())[0]
    timing = first_result.get("timing", {})
    total_runtime = (
        float(pipeline_timing["total"])
        if isinstance(pipeline_timing, Mapping)
        and isinstance(pipeline_timing.get("total"), int | float)
        else None
    )
    simulated_runtime = (
        float(timing["total"])
        if isinstance(timing, Mapping) and isinstance(timing.get("total"), int | float)
        else None
    )

    if prediction_cache_hit:
        if total_runtime is not None:
            st_module.success(f"Prediction loaded from cache in {total_runtime:.2f}s")
        else:
            st_module.success("Prediction loaded from cache.")
    elif simulated_runtime is not None:
        st_module.success(f"Predictions complete in {simulated_runtime:.2f}s")
    elif total_runtime is not None:
        st_module.success(f"Predictions complete in {total_runtime:.2f}s")
    else:
        st_module.success("Predictions complete.")

    if is_sprint:
        st_module.markdown("---")
        st_module.header("Sprint Weekend Cascade")
        st_module.info(
            "Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race"
        )

        display_prediction_result_fn(
            prediction_results["sprint_quali"],
            _section_title(prediction_results["sprint_quali"], "Sprint Qualifying Prediction"),
            False,
        )
        display_prediction_result_fn(
            prediction_results["sprint_race"],
            _section_title(prediction_results["sprint_race"], "Sprint Race Prediction"),
            True,
        )
        display_prediction_result_fn(
            prediction_results["main_quali"],
            _section_title(prediction_results["main_quali"], "Main Qualifying Prediction"),
            False,
        )
        display_prediction_result_fn(
            prediction_results["main_race"],
            _section_title(prediction_results["main_race"], "Main Race Prediction"),
            True,
        )
    else:
        st_module.markdown("---")
        st_module.header("Normal Weekend Cascade")
        st_module.info("Weekend flow: Qualifying → Race")

        display_prediction_result_fn(
            prediction_results["qualifying"],
            _section_title(prediction_results["qualifying"], "Qualifying Prediction"),
            False,
        )
        display_prediction_result_fn(
            prediction_results["race"],
            _section_title(prediction_results["race"], "Race Prediction"),
            True,
        )


def _get_prediction_precompute_settings() -> dict[str, Any]:
    """Normalize dashboard precompute settings with safe fallbacks."""
    return _prediction_boundary.get_prediction_precompute_settings(
        get_prediction_precompute_config_fn=get_prediction_precompute_config,
        logger=logger,
    )


def _resolve_precompute_targets(
    *,
    year: int,
    race_name: str,
    horizon_races: int,
) -> list[str]:
    """Resolve race targets for boundary-triggered precompute horizon."""
    from src.utils.weekend import get_schedule_rows

    return _prediction_boundary.resolve_precompute_targets(
        year=year,
        race_name=race_name,
        horizon_races=horizon_races,
        get_schedule_rows_fn=get_schedule_rows,
        logger=logger,
    )


def _resolve_race_boundary_context(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    session_detector: Any | None = None,
) -> tuple[str, str]:
    """Return the current boundary signature and checkpoint label for one race."""
    return _prediction_boundary.resolve_race_boundary_context(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        build_event_boundary_snapshot_fn=_build_event_boundary_snapshot,
        boundary_signature_fn=_boundary_signature,
        session_detector=session_detector,
    )


def _resolve_persisted_boundary_fallback(
    *,
    year: int,
    race_name: str,
    artifact_hash: str,
    current_boundary_signature: str,
    current_boundary_session_name: str,
) -> dict[str, str] | None:
    """Resolve warmed-boundary metadata when the current checkpoint is newer than storage."""
    return _prediction_boundary.resolve_persisted_boundary_fallback(
        year=year,
        race_name=race_name,
        artifact_hash=artifact_hash,
        current_boundary_signature=current_boundary_signature,
        current_boundary_session_name=current_boundary_session_name,
        load_precompute_horizon_index_fn=load_precompute_horizon_index,
    )


def _load_warmed_boundary_fallback_prediction(
    *,
    race_name: str,
    weather: str,
    year: int,
    artifact_hash: str,
    fallback_metadata: dict[str, str],
    notify_fn: Callable[[str], None],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Load the latest available warmed prediction when the live boundary is ahead."""
    return _prediction_boundary.load_warmed_boundary_fallback_prediction(
        race_name=race_name,
        weather=weather,
        year=year,
        artifact_hash=artifact_hash,
        fallback_metadata=fallback_metadata,
        notify_fn=notify_fn,
        load_precomputed_prediction_fn=load_precomputed_prediction,
    )


def _served_prediction_boundary_session_name(
    *,
    boundary_session_name: str,
    boundary_fallback: dict[str, str] | None,
) -> str:
    """Return the checkpoint label that matches the prediction actually shown to the user."""
    return _prediction_boundary.served_prediction_boundary_session_name(
        boundary_session_name=boundary_session_name,
        boundary_fallback=boundary_fallback,
    )


def _load_served_prediction_bundle(
    *,
    race_name: str,
    weather: str,
    year: int,
    is_sprint: bool,
    boundary_refresh: dict[str, Any],
    session_detector: Any,
    precompute_settings: dict[str, Any],
    get_artifact_versions_fn: GetArtifactVersionsFn,
    notify_fn: Callable[[str], None],
) -> dict[str, Any]:
    """Load the prediction payload the dashboard should serve for one request."""
    return _prediction_serving.load_served_prediction_bundle(
        race_name=race_name,
        weather=weather,
        year=year,
        is_sprint=is_sprint,
        boundary_refresh=boundary_refresh,
        session_detector=session_detector,
        precompute_settings=precompute_settings,
        get_artifact_versions_fn=get_artifact_versions_fn,
        compute_artifact_hash_fn=compute_artifact_hash,
        resolve_race_boundary_context_fn=_resolve_race_boundary_context,
        resolve_precompute_targets_fn=_resolve_precompute_targets,
        prediction_cache_key_fn=_prediction_cache_key,
        get_cached_prediction_fn=_get_cached_prediction,
        resolve_persisted_boundary_fallback_fn=_resolve_persisted_boundary_fallback,
        load_precomputed_prediction_fn=load_precomputed_prediction,
        cached_prediction_matches_persisted_fn=_cached_prediction_matches_persisted,
        store_cached_prediction_fn=_store_cached_prediction,
        load_warmed_boundary_fallback_prediction_fn=_load_warmed_boundary_fallback_prediction,
        load_precompute_horizon_index_fn=load_precompute_horizon_index,
        served_prediction_boundary_session_name_fn=_served_prediction_boundary_session_name,
        prediction_unavailable_error_type=PrecomputedPredictionUnavailableError,
        notify_fn=notify_fn,
        logger=logger,
    )


def execute_live_prediction_pipeline_core(
    *,
    race_name: str,
    weather: str,
    year: int,
    force_refresh: bool,
    progress_callback: Callable[[str], None] | None,
    clear_fastf1_race_cache_fn: Callable[[int, str], None],
    auto_update_if_needed_fn: AutoUpdateIfNeededFn,
    is_sprint_weekend_fn: Callable[[int, str], bool],
    detect_event_boundary_refresh_if_needed_fn: DetectEventBoundaryRefreshFn,
    auto_update_practice_characteristics_if_needed_fn: AutoUpdatePracticeIfNeededFn,
    clear_resource_cache_fn: Callable[[], None],
    clear_data_cache_fn: Callable[[], None],
    get_artifact_versions_fn: GetArtifactVersionsFn,
    run_prediction_fn: PredictionRunFn,
) -> dict[str, Any]:
    """
    Load a warmed persisted prediction for the selected race and weather.

    This request path does not run warmup work or simulate inline. It only
    resolves boundary state and loads an already-persisted payload.

    """
    pipeline_timing: dict[str, float] = {}
    pipeline_start = time.time()
    _ = (
        clear_fastf1_race_cache_fn,
        auto_update_if_needed_fn,
        auto_update_practice_characteristics_if_needed_fn,
        clear_resource_cache_fn,
        clear_data_cache_fn,
        run_prediction_fn,
    )
    drain_recent_alerts(limit=500)
    logger.info(
        "Live prediction pipeline start: race=%s year=%s force_refresh=%s",
        race_name,
        year,
        force_refresh,
    )
    boundary_refresh: dict[str, Any] = {
        "refresh_needed": False,
        "reason": "not_checked",
        "new_sessions": [],
        "boundary_signature": "",
    }

    def _notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if force_refresh:
        raise PrecomputedPredictionUnavailableError(
            "Manual dashboard refresh is disabled. Run the warmup worker or trigger the "
            "scheduled job manually to refresh persisted predictions."
        )
    pipeline_timing["cache_clear"] = 0.0

    precompute_settings = _get_prediction_precompute_settings()

    _notify("Loading persisted prediction artifacts...")

    weekend_start = time.time()
    _notify("Resolving weekend format...")
    try:
        is_sprint = is_sprint_weekend_fn(year, race_name)
    except Exception as exc:
        pipeline_timing["weekend_lookup"] = time.time() - weekend_start
        raise PrecomputedPredictionUnavailableError(
            f"Could not resolve weekend format for {race_name} {year} because schedule "
            f"lookup failed: {exc}. The dashboard refuses to guess sprint vs conventional."
        ) from exc
    pipeline_timing["weekend_lookup"] = time.time() - weekend_start
    from src.utils.session_detector import SessionDetector

    session_detector = SessionDetector()

    if not force_refresh:
        boundary_check_start = time.time()
        boundary_refresh = detect_event_boundary_refresh_if_needed_fn(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            session_detector=session_detector,
        )
        pipeline_timing["boundary_check"] = time.time() - boundary_check_start
        if bool(boundary_refresh.get("refresh_needed")):
            _notify("A newer checkpoint exists; waiting for warmup to persist it...")
    else:
        pipeline_timing["boundary_check"] = 0.0

    practice_start = time.time()
    practice_update = {"updated": False, "completed_fp_sessions": []}
    _notify("Warmup owns practice refresh; dashboard request path stays read-only...")
    pipeline_timing["practice_update_check"] = time.time() - practice_start

    prediction_start = time.time()
    served_prediction = _load_served_prediction_bundle(
        race_name=race_name,
        weather=weather,
        year=year,
        is_sprint=is_sprint,
        boundary_refresh=boundary_refresh,
        session_detector=session_detector,
        precompute_settings=precompute_settings,
        get_artifact_versions_fn=get_artifact_versions_fn,
        notify_fn=_notify,
    )
    pipeline_timing["prediction_load"] = time.time() - prediction_start
    pipeline_timing["total"] = time.time() - pipeline_start
    logger.info(
        "Live prediction pipeline complete: race=%s year=%s total=%.2fs",
        race_name,
        year,
        pipeline_timing["total"],
    )
    observability = {
        "alerts": drain_recent_alerts(limit=20),
        "counters": snapshot_counters(),
    }

    return {
        "prediction_results": served_prediction["prediction_results"],
        "is_sprint": is_sprint,
        "practice_update": practice_update,
        "boundary_refresh": boundary_refresh,
        "boundary_session_name": served_prediction["boundary_session_name"],
        "precompute_summary": served_prediction["precompute_summary"],
        "prediction_cache_hit": served_prediction["prediction_cache_hit"],
        "pipeline_timing": pipeline_timing,
        "practice_update_error": None,
        "observability": observability,
        "boundary_fallback": served_prediction["boundary_fallback"],
    }
