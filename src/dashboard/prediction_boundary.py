"""Helpers for dashboard boundary resolution and warmed fallback serving."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .prediction_checkpointing import resolve_prediction_checkpoint_session


def get_prediction_precompute_settings(
    *,
    get_prediction_precompute_config_fn: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Normalize dashboard precompute settings with safe fallbacks."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "horizon_races": 3,
        "weather_scenarios": ["dry", "mixed", "rain"],
        "max_file_entries": 2048,
    }
    try:
        loaded = get_prediction_precompute_config_fn()
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


def resolve_precompute_targets(
    *,
    year: int,
    race_name: str,
    horizon_races: int,
    get_schedule_rows_fn: Any,
    logger: logging.Logger,
) -> list[str]:
    """Resolve race targets for the warmed precompute horizon."""
    requested_horizon = max(1, int(horizon_races))
    targets = [race_name]
    if requested_horizon <= 1:
        return targets

    try:
        rows = list(get_schedule_rows_fn(year))
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


def resolve_race_boundary_context(
    *,
    year: int,
    race_name: str,
    is_sprint: bool,
    build_event_boundary_snapshot_fn: Any,
    boundary_signature_fn: Any,
    session_detector: Any | None = None,
) -> tuple[str, str]:
    """Return the current boundary signature and checkpoint label for one race."""
    snapshot = build_event_boundary_snapshot_fn(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        session_detector=session_detector,
    )
    checkpoint = resolve_prediction_checkpoint_session(
        snapshot.get("latest_elapsed_session"),
        is_sprint=is_sprint,
    )
    if not bool(snapshot.get("has_schedule_data")):
        return "", checkpoint
    return boundary_signature_fn(snapshot), checkpoint


def resolve_persisted_boundary_fallback(
    *,
    year: int,
    race_name: str,
    artifact_hash: str,
    current_boundary_signature: str,
    current_boundary_session_name: str,
    load_precompute_horizon_index_fn: Any,
) -> dict[str, str] | None:
    """Resolve warmed-boundary metadata when the current checkpoint is newer than storage."""
    if not current_boundary_signature:
        return None

    horizon_index = load_precompute_horizon_index_fn(year=year, artifact_hash=artifact_hash)
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


def load_warmed_boundary_fallback_prediction(
    *,
    race_name: str,
    weather: str,
    year: int,
    artifact_hash: str,
    fallback_metadata: dict[str, str],
    notify_fn: Callable[[str], None],
    load_precomputed_prediction_fn: Any,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    """Load the latest available warmed prediction when the live boundary is ahead."""
    warmed_boundary_signature = str(fallback_metadata.get("served_boundary_signature", "")).strip()
    if not warmed_boundary_signature:
        return None

    fallback_prediction = load_precomputed_prediction_fn(
        year=year,
        race_name=race_name,
        weather=weather,
        artifact_hash=artifact_hash,
        boundary_signature=warmed_boundary_signature,
    )
    if fallback_prediction is None:
        return None

    notify_fn(
        "Current checkpoint is ahead of the warmed horizon; serving the latest "
        "persisted checkpoint until warmup catches up..."
    )
    boundary_fallback = dict(fallback_metadata)
    boundary_fallback["mode"] = "served_warmed_boundary"
    boundary_fallback["warmed_boundary_signature"] = warmed_boundary_signature
    boundary_fallback["warmed_boundary_session_name"] = (
        str(fallback_metadata.get("served_boundary_session_name", "PRE")).strip().upper() or "PRE"
    )
    return fallback_prediction, boundary_fallback


def served_prediction_boundary_session_name(
    *,
    boundary_session_name: str,
    boundary_fallback: dict[str, str] | None,
) -> str:
    """Return the checkpoint label that matches the prediction actually shown to the user."""
    served_checkpoint = str(boundary_session_name).strip().upper() or "PRE"
    if not isinstance(boundary_fallback, dict):
        return served_checkpoint

    warmed_boundary = (
        str(
            boundary_fallback.get(
                "warmed_boundary_session_name",
                boundary_fallback.get("served_boundary_session_name", served_checkpoint),
            )
        )
        .strip()
        .upper()
    )
    return warmed_boundary or served_checkpoint
