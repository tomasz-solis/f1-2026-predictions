"""Core flow helpers for live prediction dashboard page."""

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
    save_precompute_horizon_index,
    save_precomputed_prediction,
)
from src.dashboard.update_flow import _boundary_signature, _build_event_boundary_snapshot
from src.persistence.config import should_read_db_first
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
    record_alert,
    record_counter,
    snapshot_counters,
)

PredictionResults = dict[str, Any]
ArtifactVersion = tuple[int, str]
ArtifactVersions = dict[str, ArtifactVersion]


class PrecomputedPredictionUnavailableError(RuntimeError):
    """Raised when the dashboard is configured to load persisted predictions only."""


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


class AutoUpdateIfNeededFn(Protocol):
    """Type contract for race-update callback."""

    def __call__(self, *, year: int, force_recheck: bool = False) -> None: ...


class DetectEventBoundaryRefreshFn(Protocol):
    """Type contract for event-boundary refresh callback."""

    def __call__(
        self,
        *,
        year: int,
        race_name: str,
        is_sprint: bool,
        session_detector: Any | None = None,
    ) -> dict[str, Any]: ...


class AutoUpdatePracticeIfNeededFn(Protocol):
    """Type contract for practice-update callback."""

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
    """Type contract for artifact-version callback."""

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
    """Clear in-memory prediction cache in a thread-safe manner."""
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
    """Build stable cache key for prediction output reuse."""
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
    """Return qualifying/race section dicts corresponding to a checkpoint session."""
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
    """Extract every forecastable target payload for a checkpoint save."""
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
    """Compute mean confidence across prediction rows when values are present."""
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
    """Persist compact checkpoint metadata for session-by-session trend analysis."""
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
    """Render prediction result sections for sprint and non-sprint weekends."""

    def _section_title(section: Mapping[str, Any], default_title: str) -> str:
        """Rename completed-session sections from Prediction to Result."""
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
    """Return normalized settings that control boundary-triggered precompute."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "inline_enabled": True,
        "horizon_races": 3,
        "weather_scenarios": ["dry", "mixed", "rain"],
        "max_file_entries": 2048,
    }
    try:
        loaded = get_prediction_precompute_config()
    except Exception as exc:
        logger.warning("Could not load prediction precompute config: %s", exc)
        return defaults

    if not isinstance(loaded, dict):
        return defaults

    settings: dict[str, Any] = dict(defaults)
    settings.update(loaded)
    weather_scenarios = settings.get("weather_scenarios")
    default_weather_scenarios = [str(item) for item in defaults["weather_scenarios"]]
    if not isinstance(weather_scenarios, list):
        settings["weather_scenarios"] = list(default_weather_scenarios)
    else:
        valid_weather = {"dry", "mixed", "rain"}
        normalized_weather = []
        for weather_option in weather_scenarios:
            normalized = str(weather_option).strip().lower()
            if normalized in valid_weather and normalized not in normalized_weather:
                normalized_weather.append(normalized)
        settings["weather_scenarios"] = (
            normalized_weather if normalized_weather else list(default_weather_scenarios)
        )
    settings["enabled"] = bool(settings.get("enabled", defaults["enabled"]))
    settings["inline_enabled"] = bool(settings.get("inline_enabled", defaults["inline_enabled"]))
    raw_horizon_races = settings.get("horizon_races", defaults["horizon_races"])
    if isinstance(raw_horizon_races, int | float | str):
        try:
            settings["horizon_races"] = max(1, int(raw_horizon_races))
        except (TypeError, ValueError):
            settings["horizon_races"] = defaults["horizon_races"]
    else:
        settings["horizon_races"] = defaults["horizon_races"]
    raw_max_file_entries = settings.get("max_file_entries", defaults["max_file_entries"])
    if isinstance(raw_max_file_entries, int | float | str):
        try:
            settings["max_file_entries"] = max(16, int(raw_max_file_entries))
        except (TypeError, ValueError):
            settings["max_file_entries"] = defaults["max_file_entries"]
    else:
        settings["max_file_entries"] = defaults["max_file_entries"]
    return settings


def _require_persisted_predictions(precompute_settings: Mapping[str, Any]) -> bool:
    """
    Return True when interactive requests must not compute new predictions.

    In DB-backed deployments we want the dashboard request path to stay read-only
    when inline precompute is disabled. That keeps latency stable and avoids
    silent divergence from the persisted warmup horizon.
    """
    if not should_read_db_first():
        return False
    return not bool(precompute_settings.get("inline_enabled", True))


def _resolve_precompute_targets(
    *,
    year: int,
    race_name: str,
    horizon_races: int,
) -> list[str]:
    """Resolve race targets for boundary-triggered precompute horizon."""
    requested_horizon = max(1, int(horizon_races))
    targets = [race_name]
    if requested_horizon <= 1:
        return targets

    try:
        from src.utils.weekend import get_schedule_rows

        rows = list(get_schedule_rows(year))
    except Exception as exc:
        logger.warning("Could not load schedule rows for precompute targeting: %s", exc)
        return targets

    if not rows:
        return targets

    race_names: list[str] = []
    for event_name, event_format in rows:
        normalized = str(event_name).strip()
        normalized_format = str(event_format).strip().lower()
        if not normalized:
            continue
        if "testing" in normalized.lower() or "testing" in normalized_format:
            continue
        race_names.append(normalized)

    if not race_names:
        return targets

    normalized_current = race_name.strip().lower()
    for idx, candidate in enumerate(race_names):
        if candidate.strip().lower() != normalized_current:
            continue
        for next_race in race_names[idx + 1 : idx + requested_horizon]:
            if next_race not in targets:
                targets.append(next_race)
        break

    return targets


def _resolve_race_boundary_context(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    session_detector: Any | None = None,
) -> tuple[str, str]:
    """
    Resolve race-specific boundary signature and checkpoint label.

    Returns:
        Tuple ``(boundary_signature, checkpoint_label)`` where checkpoint label
        is latest elapsed session or ``PRE``.
    """
    snapshot = _build_event_boundary_snapshot(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        session_detector=session_detector,
    )
    checkpoint = str(snapshot.get("latest_elapsed_session") or "PRE").strip().upper() or "PRE"
    if not bool(snapshot.get("has_schedule_data")):
        return "", checkpoint
    return _boundary_signature(snapshot), checkpoint


def _resolve_persisted_boundary_fallback(
    *,
    year: int,
    race_name: str,
    artifact_hash: str,
    current_boundary_signature: str,
    current_boundary_session_name: str,
) -> dict[str, str] | None:
    """Resolve the latest warmed boundary when the current checkpoint is not persisted yet."""
    horizon_index = load_precompute_horizon_index(year=year, artifact_hash=artifact_hash)
    if not isinstance(horizon_index, dict):
        return None

    ready_races_raw = horizon_index.get("ready_races", [])
    ready_races = (
        {str(race).strip() for race in ready_races_raw if str(race).strip()}
        if isinstance(ready_races_raw, list)
        else set()
    )
    if race_name not in ready_races:
        return None

    race_boundaries = horizon_index.get("race_boundaries", {})
    fallback_boundary_signature = (
        str(race_boundaries.get(race_name, "")).strip() if isinstance(race_boundaries, dict) else ""
    )
    if not fallback_boundary_signature:
        fallback_boundary_signature = str(horizon_index.get("boundary_signature", "")).strip()
    if not fallback_boundary_signature or fallback_boundary_signature == current_boundary_signature:
        return None

    anchor_race_name = str(horizon_index.get("anchor_race_name", "")).strip()
    if race_name == anchor_race_name:
        fallback_session_name = str(horizon_index.get("anchor_session_name", "")).strip().upper()
    else:
        fallback_session_name = "PRE"
    if not fallback_session_name:
        fallback_session_name = "PRE"

    return {
        "current_boundary_signature": current_boundary_signature,
        "current_boundary_session_name": current_boundary_session_name,
        "served_boundary_signature": fallback_boundary_signature,
        "served_boundary_session_name": fallback_session_name,
    }


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

    precompute_settings = _get_prediction_precompute_settings()
    require_persisted_predictions = _require_persisted_predictions(precompute_settings)

    if require_persisted_predictions:
        _notify("Persisted-prediction mode active; skipping artifact updates in request path...")
        pipeline_timing["race_update_check"] = 0.0
    else:
        update_start = time.time()
        _notify("Checking completed races and model updates...")
        auto_update_if_needed_fn(
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
        boundary_refresh = detect_event_boundary_refresh_if_needed_fn(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            session_detector=session_detector,
        )
        if bool(boundary_refresh.get("refresh_needed")):
            _notify("Event boundary advanced; clearing FastF1 cache and rechecking updates...")
            clear_fastf1_race_cache_fn(year, race_name)
            should_clear_runtime_caches = True

            if not require_persisted_predictions:
                update_start = time.time()
                auto_update_if_needed_fn(
                    year=year,
                    force_recheck=False,
                )
                pipeline_timing["race_update_check"] += time.time() - update_start

    practice_start = time.time()
    if require_persisted_predictions:
        practice_update = {"updated": False, "completed_fp_sessions": []}
        _notify("Persisted-prediction mode active; waiting for warmup to refresh practice data...")
    else:
        _notify("Checking completed practice sessions...")
        practice_update = auto_update_practice_characteristics_if_needed_fn(
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
    precompute_summary: dict[str, Any] = {
        "triggered": False,
        "generated": 0,
        "reused": 0,
        "targets": [],
        "ready_races": [],
        "errors": [],
        "skipped_reason": "",
    }

    artifact_versions = get_artifact_versions_fn(year=year)
    artifact_hash = compute_artifact_hash(artifact_versions)
    boundary_signature = str(boundary_refresh.get("boundary_signature", ""))
    boundary_session_name = str(boundary_refresh.get("latest_elapsed_session") or "PRE").strip()
    if not boundary_session_name:
        boundary_session_name = "PRE"
    boundary_session_name = boundary_session_name.upper()
    if not boundary_signature:
        try:
            resolved_boundary_signature, resolved_boundary_session_name = (
                _resolve_race_boundary_context(
                    year=year,
                    race_name=race_name,
                    is_sprint=is_sprint,
                    session_detector=session_detector,
                )
            )
            boundary_signature = resolved_boundary_signature
            if boundary_session_name == "PRE":
                boundary_session_name = resolved_boundary_session_name
        except Exception as exc:
            logger.warning(
                "Could not resolve race boundary context for %s %s: %s",
                year,
                race_name,
                exc,
            )
    weather_normalized = str(weather).strip().lower()
    weather_scenarios = [
        str(option).strip().lower() for option in precompute_settings["weather_scenarios"]
    ]
    target_races = _resolve_precompute_targets(
        year=year,
        race_name=race_name,
        horizon_races=int(precompute_settings["horizon_races"]),
    )
    precompute_summary["targets"] = target_races

    prediction_cache_key = _prediction_cache_key(
        year=year,
        race_name=race_name,
        weather=weather,
        is_sprint=is_sprint,
        artifact_versions=artifact_versions,
        boundary_signature=boundary_signature,
    )
    served_boundary_signature = boundary_signature
    served_boundary_session_name = boundary_session_name
    available_boundary_fallback: dict[str, str] | None = None
    boundary_fallback: dict[str, str] | None = None
    prediction_cache_hit = False

    can_use_cached_prediction = not bool(force_refresh)
    cached_prediction = (
        _get_cached_prediction(prediction_cache_key) if can_use_cached_prediction else None
    )
    if can_use_cached_prediction and cached_prediction is None and require_persisted_predictions:
        available_boundary_fallback = _resolve_persisted_boundary_fallback(
            year=year,
            race_name=race_name,
            artifact_hash=artifact_hash,
            current_boundary_signature=boundary_signature,
            current_boundary_session_name=boundary_session_name,
        )
        if available_boundary_fallback is not None:
            fallback_cache_key = _prediction_cache_key(
                year=year,
                race_name=race_name,
                weather=weather,
                is_sprint=is_sprint,
                artifact_versions=artifact_versions,
                boundary_signature=available_boundary_fallback["served_boundary_signature"],
            )
            fallback_cached_prediction = _get_cached_prediction(fallback_cache_key)
            if fallback_cached_prediction is not None:
                cached_prediction = fallback_cached_prediction
                prediction_cache_key = fallback_cache_key
                served_boundary_signature = available_boundary_fallback["served_boundary_signature"]
                served_boundary_session_name = available_boundary_fallback[
                    "served_boundary_session_name"
                ]
                boundary_fallback = available_boundary_fallback

    if can_use_cached_prediction and cached_prediction is None:
        persisted_prediction = load_precomputed_prediction(
            year=year,
            race_name=race_name,
            weather=weather,
            artifact_hash=artifact_hash,
            boundary_signature=boundary_signature,
        )
        if persisted_prediction is not None:
            _store_cached_prediction(prediction_cache_key, persisted_prediction)
            cached_prediction = persisted_prediction
        elif require_persisted_predictions and available_boundary_fallback is not None:
            fallback_cache_key = _prediction_cache_key(
                year=year,
                race_name=race_name,
                weather=weather,
                is_sprint=is_sprint,
                artifact_versions=artifact_versions,
                boundary_signature=available_boundary_fallback["served_boundary_signature"],
            )
            fallback_prediction = load_precomputed_prediction(
                year=year,
                race_name=race_name,
                weather=weather,
                artifact_hash=artifact_hash,
                boundary_signature=available_boundary_fallback["served_boundary_signature"],
            )
            if fallback_prediction is not None:
                _store_cached_prediction(fallback_cache_key, fallback_prediction)
                cached_prediction = fallback_prediction
                prediction_cache_key = fallback_cache_key
                served_boundary_signature = available_boundary_fallback["served_boundary_signature"]
                served_boundary_session_name = available_boundary_fallback[
                    "served_boundary_session_name"
                ]
                boundary_fallback = available_boundary_fallback

    if can_use_cached_prediction:
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
                if boundary_fallback is not None:
                    _notify(
                        "Current checkpoint is not warmed yet; serving latest available "
                        f"persisted checkpoint ({served_boundary_session_name})."
                    )
            else:
                if require_persisted_predictions:
                    refreshed_prediction = load_precomputed_prediction(
                        year=year,
                        race_name=race_name,
                        weather=weather,
                        artifact_hash=artifact_hash,
                        boundary_signature=served_boundary_signature,
                    )
                    if refreshed_prediction is not None:
                        refreshed_reason = _cache_hit_requires_competitive_refresh(
                            prediction_results=refreshed_prediction,
                            is_sprint=is_sprint,
                            year=year,
                            race_name=race_name,
                        )
                        if refreshed_reason is None:
                            prediction_results = refreshed_prediction
                            prediction_cache_hit = True
                            _store_cached_prediction(prediction_cache_key, refreshed_prediction)
                            _notify(
                                "Reloaded updated prediction from persisted store after cache validation..."
                            )
                        else:
                            raise PrecomputedPredictionUnavailableError(
                                "Persisted prediction is stale for the served boundary. "
                                f"Warmup must refresh {race_name} {year} [{weather_normalized}] "
                                f"before the dashboard can serve it ({refreshed_reason})."
                            )
                    else:
                        raise PrecomputedPredictionUnavailableError(
                            "Persisted prediction is missing for the served boundary after cache "
                            f"validation: {race_name} {year} [{weather_normalized}]."
                        )
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
                    save_precomputed_prediction(
                        year=year,
                        race_name=race_name,
                        weather=weather,
                        artifact_hash=artifact_hash,
                        boundary_signature=boundary_signature,
                        is_sprint=is_sprint,
                        prediction_results=prediction_results,
                        metadata={
                            "source_race_name": race_name,
                            "boundary_session_name": boundary_session_name,
                        },
                        max_file_entries=int(precompute_settings["max_file_entries"]),
                    )
        else:
            if require_persisted_predictions:
                raise PrecomputedPredictionUnavailableError(
                    "Persisted prediction is not available for "
                    f"{race_name} {year} [{weather_normalized}] at checkpoint {boundary_session_name}. "
                    "Warm the horizon in the configured storage backend before using the dashboard."
                )
            _notify("Running qualifying and race simulations...")
            prediction_results = run_prediction_fn(
                race_name,
                weather,
                artifact_versions,
                is_sprint=is_sprint,
                year=year,
            )
            _store_cached_prediction(prediction_cache_key, prediction_results)
            save_precomputed_prediction(
                year=year,
                race_name=race_name,
                weather=weather,
                artifact_hash=artifact_hash,
                boundary_signature=boundary_signature,
                is_sprint=is_sprint,
                prediction_results=prediction_results,
                metadata={
                    "source_race_name": race_name,
                    "boundary_session_name": boundary_session_name,
                },
                max_file_entries=int(precompute_settings["max_file_entries"]),
            )
    else:
        if require_persisted_predictions:
            raise PrecomputedPredictionUnavailableError(
                "Forced refresh requested while dashboard is running in persisted-prediction mode. "
                "Warmup must populate the store before the dashboard can serve this race."
            )
        _notify("Running qualifying and race simulations...")
        prediction_results = run_prediction_fn(
            race_name,
            weather,
            artifact_versions,
            is_sprint=is_sprint,
            year=year,
        )
        _store_cached_prediction(prediction_cache_key, prediction_results)
        save_precomputed_prediction(
            year=year,
            race_name=race_name,
            weather=weather,
            artifact_hash=artifact_hash,
            boundary_signature=boundary_signature,
            is_sprint=is_sprint,
            prediction_results=prediction_results,
            metadata={
                "source_race_name": race_name,
                "boundary_session_name": boundary_session_name,
            },
            max_file_entries=int(precompute_settings["max_file_entries"]),
        )

    horizon_index = load_precompute_horizon_index(year=year, artifact_hash=artifact_hash)
    index_matches_current_boundary = False
    if isinstance(horizon_index, dict):
        indexed_boundary = str(horizon_index.get("boundary_signature", "")).strip()
        indexed_anchor = str(horizon_index.get("anchor_race_name", "")).strip()
        indexed_targets_raw = horizon_index.get("expected_targets", [])
        indexed_ready_raw = horizon_index.get("ready_races", [])
        indexed_targets = (
            {str(race).strip() for race in indexed_targets_raw}
            if isinstance(indexed_targets_raw, list)
            else set()
        )
        indexed_ready = (
            [str(race).strip() for race in indexed_ready_raw if str(race).strip()]
            if isinstance(indexed_ready_raw, list)
            else []
        )
        if (
            indexed_boundary == boundary_signature
            and indexed_anchor == race_name
            and indexed_targets == {str(race).strip() for race in target_races}
        ):
            index_matches_current_boundary = True
            precompute_summary["ready_races"] = indexed_ready

    precompute_inline_enabled = bool(precompute_settings.get("inline_enabled", True))
    if not precompute_inline_enabled:
        precompute_summary["skipped_reason"] = "inline_precompute_disabled"

    trigger_precompute = (
        precompute_settings["enabled"]
        and precompute_inline_enabled
        and (
            force_refresh
            or bool(boundary_refresh.get("refresh_needed"))
            or bool(practice_update.get("updated"))
            or not index_matches_current_boundary
        )
    )
    if trigger_precompute:
        precompute_summary["triggered"] = True
        _notify("Precomputing horizon weather scenarios...")
        race_weather_coverage: dict[str, set[str]] = {
            str(target).strip(): set() for target in target_races if str(target).strip()
        }
        race_boundaries: dict[str, str] = {}
        if race_name in race_weather_coverage:
            race_weather_coverage[race_name].add(weather_normalized)

        for target_race in target_races:
            if target_race == race_name:
                target_is_sprint = is_sprint
                target_boundary_signature = boundary_signature
                target_boundary_session_name = boundary_session_name
            else:
                try:
                    target_is_sprint = bool(is_sprint_weekend_fn(year, target_race))
                except Exception as exc:
                    logger.warning(
                        "Could not resolve weekend format for precompute target %s %s: %s",
                        year,
                        target_race,
                        exc,
                    )
                    target_is_sprint = False
                target_boundary_signature, target_boundary_session_name = (
                    _resolve_race_boundary_context(
                        year=year,
                        race_name=target_race,
                        is_sprint=target_is_sprint,
                        session_detector=session_detector,
                    )
                )
            race_boundaries[target_race] = target_boundary_signature

            for target_weather in weather_scenarios:
                if target_race == race_name and target_weather == weather_normalized:
                    # Current request already resolved this scenario above.
                    continue

                precomputed = load_precomputed_prediction(
                    year=year,
                    race_name=target_race,
                    weather=target_weather,
                    artifact_hash=artifact_hash,
                    boundary_signature=target_boundary_signature,
                )
                if precomputed is not None:
                    precompute_summary["reused"] += 1
                    race_weather_coverage.setdefault(target_race, set()).add(target_weather)
                    target_cache_key = _prediction_cache_key(
                        year=year,
                        race_name=target_race,
                        weather=target_weather,
                        is_sprint=target_is_sprint,
                        artifact_versions=artifact_versions,
                        boundary_signature=target_boundary_signature,
                    )
                    _store_cached_prediction(target_cache_key, precomputed)
                    continue

                try:
                    generated_prediction = run_prediction_fn(
                        target_race,
                        target_weather,
                        artifact_versions,
                        is_sprint=target_is_sprint,
                        year=year,
                    )
                    save_precomputed_prediction(
                        year=year,
                        race_name=target_race,
                        weather=target_weather,
                        artifact_hash=artifact_hash,
                        boundary_signature=target_boundary_signature,
                        is_sprint=target_is_sprint,
                        prediction_results=generated_prediction,
                        metadata={
                            "source_race_name": race_name,
                            "boundary_session_name": target_boundary_session_name,
                        },
                        max_file_entries=int(precompute_settings["max_file_entries"]),
                    )
                    target_cache_key = _prediction_cache_key(
                        year=year,
                        race_name=target_race,
                        weather=target_weather,
                        is_sprint=target_is_sprint,
                        artifact_versions=artifact_versions,
                        boundary_signature=target_boundary_signature,
                    )
                    _store_cached_prediction(target_cache_key, generated_prediction)
                    precompute_summary["generated"] += 1
                    race_weather_coverage.setdefault(target_race, set()).add(target_weather)
                except Exception as exc:
                    logger.warning(
                        "Prediction precompute failed for %s %s %s: %s",
                        year,
                        target_race,
                        target_weather,
                        exc,
                    )
                    error_message = f"{target_race} [{target_weather}]: {exc}"
                    precompute_summary["errors"].append(error_message)
                    labels = {
                        "year": year,
                        "race_name": target_race,
                        "weather": target_weather,
                    }
                    record_counter("prediction_precompute_failure_total", labels=labels)
                    record_alert(
                        "prediction_precompute_failure",
                        (
                            "Could not precompute prediction scenario for "
                            f"{target_race} {year} ({target_weather})."
                        ),
                        labels=labels,
                    )
        expected_weather = set(weather_scenarios)
        ready_races = [
            target_race
            for target_race in target_races
            if expected_weather.issubset(race_weather_coverage.get(target_race, set()))
        ]
        precompute_summary["ready_races"] = ready_races
        try:
            save_precompute_horizon_index(
                year=year,
                artifact_hash=artifact_hash,
                boundary_signature=boundary_signature,
                anchor_race_name=race_name,
                anchor_session_name=boundary_session_name,
                expected_targets=target_races,
                ready_races=ready_races,
                weather_scenarios=weather_scenarios,
                race_boundaries=race_boundaries,
            )
        except Exception as exc:
            logger.warning(
                "Could not persist precompute horizon index for %s %s (%s): %s",
                year,
                race_name,
                boundary_signature,
                exc,
            )
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
        "precompute_summary": precompute_summary,
        "prediction_cache_hit": prediction_cache_hit,
        "pipeline_timing": pipeline_timing,
        "practice_update_error": None,
        "observability": observability,
        "boundary_fallback": boundary_fallback,
    }
