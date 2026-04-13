"""Helpers for loading served dashboard predictions from persisted storage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .prediction_checkpointing import resolve_prediction_checkpoint_session


def _empty_precompute_summary(targets: list[str]) -> dict[str, Any]:
    """Build the default request-path precompute summary payload."""
    return {
        "triggered": False,
        "generated": 0,
        "reused": 0,
        "targets": targets,
        "ready_races": [],
        "errors": [],
        "skipped_reason": "",
    }


def _resolve_boundary_state(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    boundary_refresh: dict[str, Any],
    session_detector: Any,
    resolve_race_boundary_context_fn: Any,
    logger: logging.Logger,
) -> tuple[str, str]:
    """Resolve the live boundary signature and checkpoint label for one request."""
    boundary_signature = str(boundary_refresh.get("boundary_signature", ""))
    boundary_session_name = resolve_prediction_checkpoint_session(
        boundary_refresh.get("latest_elapsed_session"),
        is_sprint=is_sprint,
    )

    if boundary_signature:
        return boundary_signature, boundary_session_name

    try:
        resolved_boundary_signature, resolved_boundary_session_name = (
            resolve_race_boundary_context_fn(
                year=year,
                race_name=race_name,
                is_sprint=is_sprint,
                session_detector=session_detector,
            )
        )
        boundary_signature = resolved_boundary_signature
        if boundary_session_name == "PRE":
            boundary_session_name = resolve_prediction_checkpoint_session(
                resolved_boundary_session_name,
                is_sprint=is_sprint,
            )
    except Exception as exc:
        logger.warning(
            "Could not resolve race boundary context for %s %s: %s",
            year,
            race_name,
            exc,
        )

    return boundary_signature, boundary_session_name


def _ready_races_for_current_horizon(
    *,
    year: int,
    race_name: str,
    artifact_hash: str,
    boundary_signature: str,
    target_races: list[str],
    load_precompute_horizon_index_fn: Any,
) -> list[str]:
    """Return ready races only when the stored horizon matches the current request state."""
    horizon_index = load_precompute_horizon_index_fn(year=year, artifact_hash=artifact_hash)
    if not isinstance(horizon_index, dict):
        return []

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
        return indexed_ready
    return []


def load_served_prediction_bundle(
    *,
    race_name: str,
    weather: str,
    year: int,
    is_sprint: bool,
    boundary_refresh: dict[str, Any],
    session_detector: Any,
    precompute_settings: dict[str, Any],
    get_artifact_versions_fn: Any,
    compute_artifact_hash_fn: Any,
    resolve_race_boundary_context_fn: Any,
    resolve_precompute_targets_fn: Any,
    prediction_cache_key_fn: Any,
    get_cached_prediction_fn: Any,
    resolve_persisted_boundary_fallback_fn: Any,
    load_precomputed_prediction_fn: Any,
    cached_prediction_matches_persisted_fn: Any,
    store_cached_prediction_fn: Any,
    load_warmed_boundary_fallback_prediction_fn: Any,
    load_precompute_horizon_index_fn: Any,
    served_prediction_boundary_session_name_fn: Any,
    prediction_unavailable_error_type: type[Exception],
    notify_fn: Callable[[str], None],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Load the persisted payload the dashboard should serve for one request."""
    artifact_versions = get_artifact_versions_fn(year=year)
    artifact_hash = compute_artifact_hash_fn(artifact_versions)
    boundary_signature, boundary_session_name = _resolve_boundary_state(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        boundary_refresh=boundary_refresh,
        session_detector=session_detector,
        resolve_race_boundary_context_fn=resolve_race_boundary_context_fn,
        logger=logger,
    )
    weather_normalized = str(weather).strip().lower()
    target_races = resolve_precompute_targets_fn(
        year=year,
        race_name=race_name,
        horizon_races=int(precompute_settings["horizon_races"]),
    )
    precompute_summary = _empty_precompute_summary(target_races)

    prediction_cache_key = prediction_cache_key_fn(
        year=year,
        race_name=race_name,
        weather=weather,
        is_sprint=is_sprint,
        artifact_versions=artifact_versions,
        boundary_signature=boundary_signature,
    )
    boundary_fallback: dict[str, str] | None = None
    prediction_cache_hit = False

    cached_prediction = get_cached_prediction_fn(prediction_cache_key)
    available_boundary_fallback = resolve_persisted_boundary_fallback_fn(
        year=year,
        race_name=race_name,
        artifact_hash=artifact_hash,
        current_boundary_signature=boundary_signature,
        current_boundary_session_name=boundary_session_name,
    )

    persisted_prediction = load_precomputed_prediction_fn(
        year=year,
        race_name=race_name,
        weather=weather,
        artifact_hash=artifact_hash,
        boundary_signature=boundary_signature,
    )
    if persisted_prediction is not None:
        if cached_prediction is not None and cached_prediction_matches_persisted_fn(
            cached_prediction=cached_prediction,
            persisted_prediction=persisted_prediction,
        ):
            prediction_results = cached_prediction
            prediction_cache_hit = True
            notify_fn("Reusing cached persisted prediction...")
        else:
            prediction_results = persisted_prediction
            store_cached_prediction_fn(prediction_cache_key, persisted_prediction)
            if cached_prediction is not None:
                notify_fn("Loaded updated persisted prediction...")
            else:
                notify_fn("Loaded persisted prediction...")
    else:
        fallback_result = None
        if available_boundary_fallback is not None:
            fallback_result = load_warmed_boundary_fallback_prediction_fn(
                race_name=race_name,
                weather=weather,
                year=year,
                artifact_hash=artifact_hash,
                fallback_metadata=available_boundary_fallback,
                notify_fn=notify_fn,
            )

        if fallback_result is not None:
            prediction_results, boundary_fallback = fallback_result
        else:
            raise prediction_unavailable_error_type(
                "Persisted prediction is not available for "
                f"{race_name} {year} [{weather_normalized}] at checkpoint {boundary_session_name}. "
                "Run warmup or trigger the scheduled job before using the dashboard."
            )

    precompute_summary["ready_races"] = _ready_races_for_current_horizon(
        year=year,
        race_name=race_name,
        artifact_hash=artifact_hash,
        boundary_signature=boundary_signature,
        target_races=target_races,
        load_precompute_horizon_index_fn=load_precompute_horizon_index_fn,
    )
    precompute_summary["skipped_reason"] = "request_path_read_only"

    return {
        "prediction_results": prediction_results,
        "boundary_session_name": served_prediction_boundary_session_name_fn(
            boundary_session_name=boundary_session_name,
            boundary_fallback=boundary_fallback,
        ),
        "precompute_summary": precompute_summary,
        "prediction_cache_hit": prediction_cache_hit,
        "boundary_fallback": boundary_fallback,
    }
