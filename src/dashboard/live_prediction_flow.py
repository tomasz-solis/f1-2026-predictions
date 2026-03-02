"""Core flow helpers for live prediction dashboard page."""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from hashlib import sha1
from threading import RLock
from typing import Any, Protocol

from src.utils.operational_observability import (
    drain_recent_alerts,
    record_alert,
    record_counter,
    snapshot_counters,
)

PredictionResults = dict[str, Any]
ArtifactVersion = tuple[int, str]
ArtifactVersions = dict[str, ArtifactVersion]


class PredictionRunFn(Protocol):
    """Type contract for prediction orchestration callback."""

    def __call__(
        self,
        race_name: str,
        weather: str,
        artifact_versions: ArtifactVersions,
        /,
        is_sprint: bool = False,
        year: int = 2026,
    ) -> PredictionResults: ...


logger = logging.getLogger(__name__)
_PREDICTION_RESULT_CACHE_MAX_ENTRIES = 24
_prediction_result_cache: OrderedDict[str, PredictionResults] = OrderedDict()
_prediction_result_cache_lock = RLock()


def clear_prediction_result_cache() -> None:
    """Clear in-memory prediction cache in a thread-safe manner."""
    with _prediction_result_cache_lock:
        _prediction_result_cache.clear()


def _invoke_auto_update_if_needed(
    auto_update_if_needed_fn: Callable[..., None],
    *,
    year: int,
    force_recheck: bool,
) -> None:
    """
    Invoke auto-update callback while supporting legacy callable signatures.
    """
    try:
        signature = inspect.signature(auto_update_if_needed_fn)
        parameters: Mapping[str, inspect.Parameter] = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if supports_var_kwargs or "year" in parameters:
        kwargs["year"] = year
    if supports_var_kwargs or "force_recheck" in parameters:
        kwargs["force_recheck"] = force_recheck

    if kwargs:
        auto_update_if_needed_fn(**kwargs)
        return

    if parameters:
        auto_update_if_needed_fn(force_recheck)
        return

    auto_update_if_needed_fn()


def _invoke_get_artifact_versions(
    get_artifact_versions_fn: Callable[..., ArtifactVersions],
    *,
    year: int,
) -> ArtifactVersions:
    """Invoke artifact-version callback with year when supported."""
    try:
        signature = inspect.signature(get_artifact_versions_fn)
        parameters: Mapping[str, inspect.Parameter] = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if supports_var_kwargs or "year" in parameters:
        return get_artifact_versions_fn(year=year)
    if parameters:
        return get_artifact_versions_fn(year)
    return get_artifact_versions_fn()


def _invoke_detect_event_boundary_refresh(
    detect_event_boundary_refresh_fn: Callable[..., dict[str, Any]],
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    session_detector: Any | None = None,
) -> dict[str, Any]:
    """Invoke boundary-refresh detector with compatibility for legacy signatures."""
    try:
        signature = inspect.signature(detect_event_boundary_refresh_fn)
        parameters: Mapping[str, inspect.Parameter] = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if supports_var_kwargs or "year" in parameters:
        kwargs["year"] = year
    if supports_var_kwargs or "race_name" in parameters:
        kwargs["race_name"] = race_name
    if supports_var_kwargs or "is_sprint" in parameters:
        kwargs["is_sprint"] = is_sprint
    if session_detector is not None and (supports_var_kwargs or "session_detector" in parameters):
        kwargs["session_detector"] = session_detector

    try:
        if kwargs:
            result = detect_event_boundary_refresh_fn(**kwargs)
        elif parameters:
            result = detect_event_boundary_refresh_fn(year, race_name, is_sprint)
        else:
            result = detect_event_boundary_refresh_fn()
    except TypeError:
        # Backward-compatible fallback for patched callables with partial signatures.
        result = detect_event_boundary_refresh_fn(year, race_name, is_sprint)

    if isinstance(result, dict):
        return result
    return {"refresh_needed": False, "reason": "invalid_detector_response", "new_sessions": []}


def _invoke_auto_update_practice_if_needed(
    auto_update_practice_if_needed_fn: Callable[..., dict[str, Any]],
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    force_recheck: bool,
    session_detector: Any | None = None,
) -> dict[str, Any]:
    """Invoke practice-update callback with compatibility for legacy signatures."""
    try:
        signature = inspect.signature(auto_update_practice_if_needed_fn)
        parameters: Mapping[str, inspect.Parameter] = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if supports_var_kwargs or "year" in parameters:
        kwargs["year"] = year
    if supports_var_kwargs or "race_name" in parameters:
        kwargs["race_name"] = race_name
    if supports_var_kwargs or "is_sprint" in parameters:
        kwargs["is_sprint"] = is_sprint
    if supports_var_kwargs or "force_recheck" in parameters:
        kwargs["force_recheck"] = force_recheck
    if session_detector is not None and (supports_var_kwargs or "session_detector" in parameters):
        kwargs["session_detector"] = session_detector

    try:
        if kwargs:
            result = auto_update_practice_if_needed_fn(**kwargs)
        elif parameters:
            result = auto_update_practice_if_needed_fn(year, race_name, is_sprint, force_recheck)
        else:
            result = auto_update_practice_if_needed_fn()
    except TypeError:
        result = auto_update_practice_if_needed_fn(year, race_name, is_sprint, force_recheck)

    if isinstance(result, dict):
        return result
    return {"updated": False, "completed_fp_sessions": []}


def _prediction_cache_key(
    *,
    cache_scope_id: str,
    year: int,
    race_name: str,
    weather: str,
    is_sprint: bool,
    artifact_versions: ArtifactVersions,
    boundary_signature: str,
) -> str:
    """Build stable cache key for prediction output reuse."""
    payload = {
        "cache_scope_id": cache_scope_id,
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
    """Fetch cached prediction and mark key as recently used."""
    with _prediction_result_cache_lock:
        cached = _prediction_result_cache.get(cache_key)
        if cached is None:
            return None
        _prediction_result_cache.move_to_end(cache_key)
        return cached


def _store_cached_prediction(cache_key: str, prediction_results: PredictionResults) -> None:
    """Store prediction output in bounded LRU cache."""
    with _prediction_result_cache_lock:
        _prediction_result_cache[cache_key] = prediction_results
        _prediction_result_cache.move_to_end(cache_key)
        while len(_prediction_result_cache) > _PREDICTION_RESULT_CACHE_MAX_ENTRIES:
            _prediction_result_cache.popitem(last=False)


def _is_usable_grid_payload(value: Any) -> bool:
    """Return True when payload can be passed back into qualifying-grid refresh checks."""
    if not isinstance(value, list) or not value:
        return False
    for entry in value:
        if not isinstance(entry, dict):
            return False
        if not all(field in entry for field in ("driver", "team", "position")):
            return False
    return True


def _normalized_grid_signature(value: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """Canonicalized grid representation for stable comparison."""
    normalized: list[tuple[int, str, str]] = []
    for entry in value:
        raw_position = entry.get("position")
        if isinstance(raw_position, int):
            position = raw_position
        elif isinstance(raw_position, float):
            position = int(raw_position)
        elif isinstance(raw_position, str):
            try:
                position = int(raw_position)
            except ValueError:
                position = 0
        else:
            position = 0
        driver = str(entry.get("driver", "")).strip().upper()
        team = str(entry.get("team", "")).strip().upper()
        if not driver:
            continue
        normalized.append((position, driver, team))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    return normalized


def _cache_hit_requires_competitive_refresh(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    year: int,
    race_name: str,
) -> str | None:
    """
    Re-check competitive-session data on cache hits to detect delayed FastF1 result availability.

    Returns a machine-readable reason when cached output must be invalidated.
    """
    from src.dashboard.prediction_flow import fetch_grid_if_available

    session_bindings: tuple[tuple[str, str], ...]
    if is_sprint:
        session_bindings = (("SQ", "sprint_quali"), ("Q", "main_quali"))
    else:
        session_bindings = (("Q", "qualifying"),)

    for session_name, section_key in session_bindings:
        section = prediction_results.get(section_key, {})
        if not isinstance(section, dict):
            continue

        cached_grid_raw = section.get("grid", [])
        if not _is_usable_grid_payload(cached_grid_raw):
            continue
        cached_grid = list(cached_grid_raw)
        cached_source = str(section.get("grid_source", "PREDICTED")).strip().upper() or "PREDICTED"

        try:
            refreshed_grid, refreshed_source_raw = fetch_grid_if_available(
                year=year,
                race_name=race_name,
                session_name=session_name,
                predicted_grid=cached_grid,
            )
        except Exception as exc:
            logger.warning(
                "Could not re-check competitive session %s for %s %s; preserving cached output: %s",
                session_name,
                race_name,
                year,
                exc,
            )
            record_counter(
                "fastf1_cache_recheck_failure_total",
                labels={"year": year, "race_name": race_name, "session_name": session_name},
            )
            record_alert(
                "fastf1_cache_recheck_failure",
                (
                    "Could not refresh competitive session status from FastF1; "
                    f"preserved cached prediction for {race_name} {year} {session_name}."
                ),
                labels={"year": year, "race_name": race_name, "session_name": session_name},
            )
            continue

        refreshed_source = str(refreshed_source_raw).strip().upper() or "PREDICTED"
        if cached_source == "ACTUAL" and refreshed_source != "ACTUAL":
            logger.warning(
                "Ignoring transient grid source downgrade for %s %s %s (%s -> %s)",
                race_name,
                year,
                session_name,
                cached_source,
                refreshed_source,
            )
            record_counter(
                "fastf1_downgrade_prevented_total",
                labels={"year": year, "race_name": race_name, "session_name": session_name},
            )
            record_alert(
                "fastf1_downgrade_prevented",
                (
                    "Prevented ACTUAL -> PREDICTED downgrade during cache refresh for "
                    f"{race_name} {year} {session_name}."
                ),
                labels={"year": year, "race_name": race_name, "session_name": session_name},
            )
            continue
        if refreshed_source != cached_source:
            return f"{session_name.lower()}_grid_source_changed"
        if refreshed_source == "ACTUAL" and _normalized_grid_signature(
            refreshed_grid
        ) != _normalized_grid_signature(cached_grid):
            return f"{session_name.lower()}_actual_grid_changed"

    return None


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
) -> None:
    """Persist prediction artifacts for later accuracy tracking."""
    if not enable_logging:
        return

    detector = detector_factory()
    logger_inst = prediction_logger_factory()
    latest_session = detector.get_latest_completed_session(year, race_name, is_sprint)

    if latest_session:
        if not logger_inst.has_prediction_for_session(year, race_name, latest_session):
            try:
                quali_grid, race_finish, fp_blend_info = prediction_payload_for_session(
                    prediction_results=prediction_results,
                    is_sprint=is_sprint,
                    session_name=str(latest_session),
                )

                logger_inst.save_prediction(
                    year=year,
                    race_name=race_name,
                    session_name=latest_session,
                    qualifying_prediction=quali_grid,
                    race_prediction=race_finish,
                    weather=weather,
                    fp_blend_info=fp_blend_info,
                )
                st_module.info(f"Prediction saved for accuracy tracking (after {latest_session})")
            except Exception as exc:
                st_module.warning(f"Could not save prediction: {exc}")
        else:
            st_module.info(f"Prediction for {latest_session} already saved (max 1 per session)")
    else:
        st_module.info(
            "No completed sessions yet; prediction not saved (will save after FP1/FP2/FP3/SQ)"
        )


def prediction_payload_for_session(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    session_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Select qualifying/race payload pair to persist for a given completed session.

    For sprint weekends, sessions up to and including ``Sprint`` map to the sprint
    cascade payload, while later sessions map to the main qualifying/race payload.
    """
    if not is_sprint:
        return (
            prediction_results["qualifying"]["grid"],
            prediction_results["race"]["finish_order"],
            prediction_results.get("qualifying", {}).get("fp_blend_info", {}),
        )

    session_name_upper = str(session_name).strip().upper()
    sprint_phase_sessions = {"FP1", "SQ", "SPRINT"}
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


def render_prediction_results_core(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    display_prediction_result_fn: Callable[[PredictionResults, str, bool], None],
    st_module: Any,
) -> None:
    """Render prediction result sections for sprint and non-sprint weekends."""
    first_result = list(prediction_results.values())[0]
    timing = first_result.get("timing", {})
    if timing:
        st_module.success(f"Predictions complete in {timing['total']:.2f}s")
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
            "Sprint Qualifying Prediction",
            False,
        )
        display_prediction_result_fn(
            prediction_results["sprint_race"],
            "Sprint Race Prediction",
            True,
        )
        display_prediction_result_fn(
            prediction_results["main_quali"],
            "Main Qualifying Prediction",
            False,
        )
        display_prediction_result_fn(
            prediction_results["main_race"],
            "Main Race Prediction",
            True,
        )
    else:
        st_module.markdown("---")
        st_module.header("Normal Weekend Cascade")
        st_module.info("Weekend flow: Qualifying → Race")

        display_prediction_result_fn(
            prediction_results["qualifying"],
            "Qualifying Prediction",
            False,
        )
        display_prediction_result_fn(
            prediction_results["race"],
            "Race Prediction",
            True,
        )


def execute_live_prediction_pipeline_core(
    *,
    race_name: str,
    weather: str,
    year: int,
    force_refresh: bool,
    progress_callback: Callable[[str], None] | None,
    clear_fastf1_race_cache_fn: Callable[[int, str], None],
    auto_update_if_needed_fn: Callable[..., None],
    is_sprint_weekend_fn: Callable[[int, str], bool],
    detect_event_boundary_refresh_if_needed_fn: Callable[..., dict[str, Any]],
    auto_update_practice_characteristics_if_needed_fn: Callable[..., dict[str, Any]],
    clear_resource_cache_fn: Callable[[], None],
    clear_data_cache_fn: Callable[[], None],
    get_artifact_versions_fn: Callable[..., ArtifactVersions],
    run_prediction_fn: PredictionRunFn,
    cache_scope_id: str = "",
) -> dict[str, Any]:
    """
    Refresh input data and execute a prediction run.

    Kept separate from Streamlit rendering so tests can assert refresh call order.
    """
    pipeline_timing: dict[str, float] = {}
    pipeline_start = time.time()
    drain_recent_alerts(limit=500)
    logger.info(
        "Live prediction pipeline start: race=%s year=%s force_refresh=%s",
        race_name,
        year,
        force_refresh,
    )
    should_clear_runtime_caches = force_refresh
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
        _notify("Clearing FastF1 cache for fresh data...")
        clear_start = time.time()
        clear_fastf1_race_cache_fn(year, race_name)
        pipeline_timing["cache_clear"] = time.time() - clear_start

    update_start = time.time()
    _notify("Checking completed races and model updates...")
    _invoke_auto_update_if_needed(
        auto_update_if_needed_fn,
        year=year,
        force_recheck=force_refresh,
    )
    pipeline_timing["race_update_check"] = time.time() - update_start

    weekend_start = time.time()
    _notify("Resolving weekend format...")
    is_sprint = is_sprint_weekend_fn(year, race_name)
    pipeline_timing["weekend_lookup"] = time.time() - weekend_start
    from src.utils.session_detector import SessionDetector

    session_detector = SessionDetector()

    if not force_refresh:
        boundary_refresh = _invoke_detect_event_boundary_refresh(
            detect_event_boundary_refresh_if_needed_fn,
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            session_detector=session_detector,
        )
        if bool(boundary_refresh.get("refresh_needed")):
            _notify("Event boundary advanced; clearing FastF1 cache and rechecking updates...")
            clear_fastf1_race_cache_fn(year, race_name)
            should_clear_runtime_caches = True

            update_start = time.time()
            _invoke_auto_update_if_needed(
                auto_update_if_needed_fn,
                year=year,
                force_recheck=False,
            )
            pipeline_timing["race_update_check"] += time.time() - update_start

    practice_start = time.time()
    _notify("Checking completed practice sessions...")
    practice_update = _invoke_auto_update_practice_if_needed(
        auto_update_practice_characteristics_if_needed_fn,
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        force_recheck=force_refresh,
        session_detector=session_detector,
    )
    pipeline_timing["practice_update_check"] = time.time() - practice_start

    if practice_update.get("updated") or should_clear_runtime_caches:
        _notify("Refreshing local caches after updates...")
        clear_resource_cache_fn()
        clear_data_cache_fn()

    prediction_start = time.time()
    artifact_versions = _invoke_get_artifact_versions(get_artifact_versions_fn, year=year)
    boundary_signature = str(boundary_refresh.get("boundary_signature", ""))
    prediction_cache_key = _prediction_cache_key(
        cache_scope_id=cache_scope_id,
        year=year,
        race_name=race_name,
        weather=weather,
        is_sprint=is_sprint,
        artifact_versions=artifact_versions,
        boundary_signature=boundary_signature,
    )
    prediction_cache_hit = False

    can_use_cached_prediction = not bool(force_refresh)
    if can_use_cached_prediction:
        cached_prediction = _get_cached_prediction(prediction_cache_key)
        if cached_prediction is not None:
            cache_refresh_reason = _cache_hit_requires_competitive_refresh(
                prediction_results=cached_prediction,
                is_sprint=is_sprint,
                year=year,
                race_name=race_name,
            )

            if cache_refresh_reason is None:
                prediction_results = cached_prediction
                prediction_cache_hit = True
                _notify("Reusing cached prediction (no new sessions or input changes)...")
            else:
                _notify("Competitive results changed; regenerating prediction...")
                prediction_results = run_prediction_fn(
                    race_name,
                    weather,
                    artifact_versions,
                    is_sprint=is_sprint,
                    year=year,
                )
                _store_cached_prediction(prediction_cache_key, prediction_results)
        else:
            _notify("Running qualifying and race simulations...")
            prediction_results = run_prediction_fn(
                race_name,
                weather,
                artifact_versions,
                is_sprint=is_sprint,
                year=year,
            )
            _store_cached_prediction(prediction_cache_key, prediction_results)
    else:
        _notify("Running qualifying and race simulations...")
        prediction_results = run_prediction_fn(
            race_name,
            weather,
            artifact_versions,
            is_sprint=is_sprint,
            year=year,
        )
        _store_cached_prediction(prediction_cache_key, prediction_results)
    pipeline_timing["prediction_run"] = time.time() - prediction_start
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
        "prediction_results": prediction_results,
        "is_sprint": is_sprint,
        "practice_update": practice_update,
        "boundary_refresh": boundary_refresh,
        "prediction_cache_hit": prediction_cache_hit,
        "pipeline_timing": pipeline_timing,
        "practice_update_error": None,
        "observability": observability,
    }
