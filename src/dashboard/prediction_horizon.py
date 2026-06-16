"""Helpers for dashboard race selection and precompute-horizon gating."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

_RACE_SESSION_POST_START_HOLD = timedelta(hours=4)
_LOOKUP_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _event_value(event: Any, key: str) -> Any:
    """Read a value from a schedule row without depending on a concrete row type."""
    if isinstance(event, Mapping):
        return event.get(key)
    get_method = getattr(event, "get", None)
    if callable(get_method):
        try:
            return get_method(key)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    try:
        return event[key]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _coerce_utc_datetime(value: Any) -> datetime | None:
    """Normalize schedule datetime-like values to UTC-aware datetimes."""
    if value is None:
        return None

    candidate = value
    if hasattr(candidate, "to_pydatetime"):
        try:
            candidate = candidate.to_pydatetime()
        except (AttributeError, TypeError, ValueError):
            return None

    if not isinstance(candidate, datetime):
        return None

    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=UTC)

    return candidate.astimezone(UTC)


def race_window_cutoff_datetime(event: Any) -> datetime | None:
    """
    Return the timestamp after which a schedule row can stop anchoring the dashboard.

    FastF1 ``EventDate`` is often midnight on race day. Using it directly advances
    the dashboard to the next GP before the race has even started, so prefer the
    scheduled race session and hold through the plausible race/result window.
    """
    for session_key in ("Session5DateUtc", "Session5Date"):
        session_start = _coerce_utc_datetime(_event_value(event, session_key))
        if session_start is not None:
            return session_start + _RACE_SESSION_POST_START_HOLD

    event_date = _coerce_utc_datetime(_event_value(event, "EventDate"))
    if event_date is None:
        return None

    if event_date.timetz().replace(tzinfo=None) == time.min:
        return event_date + timedelta(days=1)
    return event_date + _RACE_SESSION_POST_START_HOLD


def parse_refresh_timestamp(value: Any) -> datetime | None:
    """Parse a persisted refresh timestamp into UTC when possible."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def latest_dashboard_refresh_timestamp(
    year: int,
    *,
    get_artifact_versions_fn: Any,
    compute_artifact_hash_fn: Any,
    load_precompute_horizon_index_fn: Any,
    refresh_paths: tuple[Path, ...],
    logger: logging.Logger,
) -> datetime | None:
    """Return the newest refresh stamp that affects dashboard predictions."""
    timestamps: list[datetime] = []
    artifact_hash = ""

    try:
        artifact_versions = get_artifact_versions_fn(year=year)
    except _LOOKUP_ERRORS as exc:
        logger.warning("Could not load artifact versions for refresh stamp: %s", exc)
        artifact_versions = {}

    for version_payload in artifact_versions.values():
        if not isinstance(version_payload, tuple) or len(version_payload) < 2:
            continue
        parsed = parse_refresh_timestamp(version_payload[1])
        if parsed is not None:
            timestamps.append(parsed)

    try:
        artifact_hash = compute_artifact_hash_fn(artifact_versions)
    except _LOOKUP_ERRORS as exc:
        logger.warning("Could not compute artifact hash for refresh stamp: %s", exc)

    if artifact_hash:
        try:
            horizon_index = load_precompute_horizon_index_fn(
                year=year,
                artifact_hash=artifact_hash,
            )
        except _LOOKUP_ERRORS as exc:
            logger.warning("Could not load horizon index for refresh stamp: %s", exc)
            horizon_index = None
        if isinstance(horizon_index, dict):
            parsed = parse_refresh_timestamp(horizon_index.get("updated_at"))
            if parsed is not None:
                timestamps.append(parsed)

    for refresh_path in refresh_paths:
        if not refresh_path.exists():
            continue
        try:
            timestamps.append(datetime.fromtimestamp(refresh_path.stat().st_mtime, tz=UTC))
        except OSError:
            continue

    return max(timestamps) if timestamps else None


def dashboard_refresh_label(
    year: int,
    *,
    latest_dashboard_refresh_timestamp_fn: Any,
    fallback_label: str,
) -> str:
    """Format the dashboard refresh label."""
    latest_refresh = latest_dashboard_refresh_timestamp_fn(year)
    if latest_refresh is None:
        return fallback_label
    return latest_refresh.strftime("%Y-%m-%d %H:%M UTC")


def load_schedule_event_rows(
    year: int,
    *,
    get_event_schedule_fn: Any,
    fallback_schedule_rows_fn: Any,
    logger: logging.Logger,
) -> tuple[tuple[str, str, str], ...]:
    """Load schedule rows with ISO-formatted race-window cutoff timestamps."""
    rows: list[tuple[str, str, str]] = []

    try:
        schedule = get_event_schedule_fn(year)
        if "EventName" in schedule.columns and "EventFormat" in schedule.columns:
            for _, event in schedule.iterrows():
                event_name = str(event.get("EventName", "")).strip()
                event_format = str(event.get("EventFormat", "")).strip()
                if not event_name:
                    continue
                event_cutoff = race_window_cutoff_datetime(event)
                event_cutoff_iso = event_cutoff.isoformat() if event_cutoff else ""
                rows.append((event_name, event_format, event_cutoff_iso))
    except _LOOKUP_ERRORS as exc:
        logger.warning("Could not load dated schedule rows for %s: %s", year, exc)

    if rows:
        return tuple(rows)

    return tuple(
        (str(event_name).strip(), str(event_format).strip(), "")
        for event_name, event_format in fallback_schedule_rows_fn(year)
        if str(event_name).strip()
    )


def resolve_dashboard_race_horizon(
    *,
    schedule_rows: tuple[tuple[str, str, str], ...],
    horizon_races: int,
    now_utc: datetime | None = None,
) -> list[str]:
    """Return the next configured race window from the live schedule."""
    if not schedule_rows:
        return []

    competitive_rows: list[tuple[str, datetime | None]] = []
    for event_name, event_format, event_cutoff_iso in schedule_rows:
        normalized_name = str(event_name).strip()
        normalized_format = str(event_format).strip().lower()
        if not normalized_name:
            continue
        if "testing" in normalized_name.lower() or "testing" in normalized_format:
            continue
        event_cutoff: datetime | None = None
        candidate = str(event_cutoff_iso).strip()
        if candidate:
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    event_cutoff = parsed.replace(tzinfo=UTC)
                else:
                    event_cutoff = parsed.astimezone(UTC)
        competitive_rows.append((normalized_name, event_cutoff))

    if not competitive_rows:
        return []

    horizon_size = max(1, int(horizon_races))
    current_time = now_utc or datetime.now(UTC)
    anchor_index: int | None = None
    for index, (_race_name, event_cutoff) in enumerate(competitive_rows):
        if event_cutoff is None:
            continue
        if event_cutoff >= current_time:
            anchor_index = index
            break

    if anchor_index is None:
        return []

    return [
        race_name
        for race_name, _event_cutoff in competitive_rows[anchor_index : anchor_index + horizon_size]
    ]


def current_anchor_boundary_signature(
    year: int,
    anchor_race_name: str,
    *,
    is_sprint_weekend_fn: Any,
    build_event_boundary_snapshot_fn: Any,
    boundary_signature_fn: Any,
    session_detector_factory: Any,
    logger: logging.Logger,
    now_utc: datetime | None = None,
) -> str | None:
    """Return the live boundary signature for the anchor race, if available."""
    try:
        is_sprint = bool(is_sprint_weekend_fn(year, anchor_race_name))
    except _LOOKUP_ERRORS as exc:
        logger.warning(
            "Could not determine weekend type for anchor boundary validation (%s %s): %s",
            year,
            anchor_race_name,
            exc,
        )
        return None

    try:
        snapshot = build_event_boundary_snapshot_fn(
            year=year,
            race_name=anchor_race_name,
            is_sprint=is_sprint,
            session_detector=session_detector_factory(),
            now_utc=now_utc or datetime.now(UTC),
        )
    except _LOOKUP_ERRORS as exc:
        logger.warning(
            "Could not build anchor boundary snapshot for dropdown filtering (%s %s): %s",
            year,
            anchor_race_name,
            exc,
        )
        return None

    if not bool(snapshot.get("has_schedule_data")):
        return None
    return boundary_signature_fn(snapshot)


def selected_race_persisted_prediction_available(
    *,
    year: int,
    race_name: str,
    weather: str,
    get_artifact_versions_fn: Any,
    compute_artifact_hash_fn: Any,
    current_anchor_boundary_signature_fn: Any,
    load_precomputed_prediction_fn: Any,
    logger: logging.Logger,
) -> bool:
    """Return whether this race already has a served prediction at the live boundary."""
    try:
        artifact_versions = get_artifact_versions_fn(year=year)
        artifact_hash = compute_artifact_hash_fn(artifact_versions)
    except _LOOKUP_ERRORS as exc:
        logger.warning(
            "Could not compute artifact hash while checking selected-race availability: %s",
            exc,
        )
        return False

    current_boundary_signature = current_anchor_boundary_signature_fn(year, race_name)
    if not current_boundary_signature:
        return False

    try:
        payload = load_precomputed_prediction_fn(
            year=year,
            race_name=race_name,
            weather=str(weather).strip().lower(),
            artifact_hash=artifact_hash,
            boundary_signature=current_boundary_signature,
        )
    except _LOOKUP_ERRORS as exc:
        logger.warning(
            "Could not load persisted selected-race prediction for %s %s [%s]: %s",
            year,
            race_name,
            weather,
            exc,
        )
        return False

    return isinstance(payload, dict)


def load_ready_races_from_current_store(
    *,
    year: int,
    race_names: list[str],
    artifact_hash: str,
    weather_scenarios: list[str],
    current_anchor_boundary_signature_fn: Any,
    load_precomputed_prediction_fn: Any,
) -> list[str]:
    """Return races that already have full persisted weather coverage for the live boundary."""
    ready_races: list[str] = []
    expected_weather = [
        str(weather).strip().lower() for weather in weather_scenarios if str(weather).strip()
    ]
    if not race_names or not expected_weather:
        return ready_races

    for race_name in race_names:
        boundary_signature = current_anchor_boundary_signature_fn(year, race_name)
        if not boundary_signature:
            continue
        has_full_coverage = True
        for weather in expected_weather:
            payload = load_precomputed_prediction_fn(
                year=year,
                race_name=race_name,
                weather=weather,
                artifact_hash=artifact_hash,
                boundary_signature=boundary_signature,
            )
            if payload is None:
                has_full_coverage = False
                break
        if has_full_coverage:
            ready_races.append(race_name)

    return ready_races


def maybe_scope_race_options_to_planned_horizon(
    *,
    race_options: list[str],
    planned_races: list[str],
    requested_horizon: int,
) -> tuple[list[str], bool]:
    """Trim a full-season dropdown to the live planned horizon when needed."""
    if not race_options:
        return [], False

    planned_set = {race_name.strip() for race_name in planned_races if race_name.strip()}
    if not planned_set:
        return list(race_options), False

    if len(race_options) <= max(1, int(requested_horizon)):
        return list(race_options), False

    scoped_race_options = [
        option for option in race_options if option.replace(" (Sprint)", "").strip() in planned_set
    ]
    if not scoped_race_options or len(scoped_race_options) == len(race_options):
        return list(race_options), False

    return scoped_race_options, True


def filter_race_options_to_precomputed_horizon(
    *,
    year: int,
    race_options: list[str],
    get_prediction_precompute_config_fn: Any,
    resolve_dashboard_race_horizon_fn: Any,
    get_artifact_versions_fn: Any,
    compute_artifact_hash_fn: Any,
    load_precompute_horizon_index_fn: Any,
    has_precompute_horizon_for_year_fn: Any,
    load_ready_races_from_current_store_fn: Any,
    current_anchor_boundary_signature_fn: Any,
    logger: logging.Logger,
    validate_live_boundary: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Filter race options to the warmed horizon for the active artifact state."""
    if not race_options:
        return race_options, {"applied": False}

    settings = get_prediction_precompute_config_fn()
    requested_horizon = max(1, int(settings.get("horizon_races", 3)))
    planned_races = resolve_dashboard_race_horizon_fn(year, requested_horizon)
    base_race_options = list(race_options)
    scoped_race_options, scope_applied = maybe_scope_race_options_to_planned_horizon(
        race_options=base_race_options,
        planned_races=planned_races,
        requested_horizon=requested_horizon,
    )
    scope_metadata: dict[str, Any] = {
        "applied": False,
        "scope_applied": scope_applied,
        "planned_races": planned_races,
        "requested_horizon": requested_horizon,
    }

    try:
        artifact_versions = get_artifact_versions_fn(year=year)
        artifact_hash = compute_artifact_hash_fn(artifact_versions)
    except _LOOKUP_ERRORS as exc:
        logger.warning("Could not compute artifact hash for dropdown filtering: %s", exc)
        return scoped_race_options, scope_metadata

    index_payload = load_precompute_horizon_index_fn(year=year, artifact_hash=artifact_hash)
    if not isinstance(index_payload, dict):
        ready_races = load_ready_races_from_current_store_fn(
            year=year,
            race_names=planned_races,
            artifact_hash=artifact_hash,
            weather_scenarios=list(settings.get("weather_scenarios", ["dry", "mixed", "rain"])),
        )
        if ready_races:
            ready_set = set(ready_races)
            filtered_options = [
                option
                for option in scoped_race_options
                if option.replace(" (Sprint)", "").strip() in ready_set
            ]
            if filtered_options:
                return filtered_options, {
                    **scope_metadata,
                    "applied": True,
                    "artifact_hash": artifact_hash,
                    "ready_races": ready_races,
                    "expected_targets": planned_races,
                    "source": "storage_scan",
                }
        stale_reason = "missing_horizon_index"
        if has_precompute_horizon_for_year_fn(
            year=year,
            exclude_artifact_hash=artifact_hash,
        ):
            stale_reason = "artifact_hash_mismatch"
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
            "stale_reason": stale_reason,
        }

    anchor_race_name = str(index_payload.get("anchor_race_name", "")).strip()
    indexed_boundary = str(index_payload.get("boundary_signature", "")).strip()
    if not anchor_race_name or not indexed_boundary:
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
            "stale_reason": "missing_anchor_or_boundary",
        }

    ready_races_raw = index_payload.get("ready_races", [])
    ready_races = (
        [str(race).strip() for race in ready_races_raw if str(race).strip()]
        if isinstance(ready_races_raw, list)
        else []
    )

    if not validate_live_boundary:
        option_by_race = {
            option.replace(" (Sprint)", "").strip(): option for option in base_race_options
        }
        filtered_options = [
            option_by_race[race_name] for race_name in ready_races if race_name in option_by_race
        ]
        if filtered_options:
            return filtered_options, {
                **scope_metadata,
                "applied": True,
                "artifact_hash": artifact_hash,
                "anchor_race_name": anchor_race_name,
                "anchor_session_name": str(index_payload.get("anchor_session_name", "")).strip(),
                "boundary_signature": indexed_boundary,
                "boundary_validation_deferred": True,
                "expected_targets": [
                    str(race).strip()
                    for race in index_payload.get("expected_targets", [])
                    if str(race).strip()
                ]
                if isinstance(index_payload.get("expected_targets"), list)
                else [],
                "ready_races": ready_races,
            }

        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
            "boundary_validation_deferred": True,
        }

    current_boundary = current_anchor_boundary_signature_fn(year, anchor_race_name)
    if not current_boundary:
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
            "stale_reason": "boundary_unavailable",
            "anchor_race_name": anchor_race_name,
            "indexed_boundary_signature": indexed_boundary,
        }

    if current_boundary != indexed_boundary:
        ready_set = set(ready_races)
        filtered_options = [
            option
            for option in base_race_options
            if option.replace(" (Sprint)", "").strip() in ready_set
        ]
        if filtered_options:
            return filtered_options, {
                **scope_metadata,
                "applied": True,
                "artifact_hash": artifact_hash,
                "anchor_race_name": anchor_race_name,
                "anchor_session_name": str(index_payload.get("anchor_session_name", "")).strip(),
                "boundary_signature": indexed_boundary,
                "current_boundary_signature": current_boundary,
                "expected_targets": [
                    str(race).strip()
                    for race in index_payload.get("expected_targets", [])
                    if str(race).strip()
                ]
                if isinstance(index_payload.get("expected_targets"), list)
                else [],
                "ready_races": ready_races,
                "fallback_boundary_active": True,
                "stale_reason": "boundary_mismatch",
            }

        ready_races = load_ready_races_from_current_store_fn(
            year=year,
            race_names=planned_races,
            artifact_hash=artifact_hash,
            weather_scenarios=list(settings.get("weather_scenarios", ["dry", "mixed", "rain"])),
        )
        if ready_races:
            ready_set = set(ready_races)
            filtered_options = [
                option
                for option in scoped_race_options
                if option.replace(" (Sprint)", "").strip() in ready_set
            ]
            if filtered_options:
                return filtered_options, {
                    **scope_metadata,
                    "applied": True,
                    "artifact_hash": artifact_hash,
                    "anchor_race_name": anchor_race_name,
                    "ready_races": ready_races,
                    "expected_targets": planned_races,
                    "source": "storage_scan",
                }
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
            "stale_reason": "boundary_mismatch",
            "anchor_race_name": anchor_race_name,
            "indexed_boundary_signature": indexed_boundary,
            "current_boundary_signature": current_boundary,
        }

    if not isinstance(ready_races_raw, list):
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
        }
    if not ready_races:
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
        }

    ready_set = set(ready_races)
    filtered_options = [
        option
        for option in base_race_options
        if option.replace(" (Sprint)", "").strip() in ready_set
    ]
    if not filtered_options:
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
        }

    return filtered_options, {
        **scope_metadata,
        "applied": True,
        "artifact_hash": artifact_hash,
        "anchor_race_name": anchor_race_name,
        "anchor_session_name": str(index_payload.get("anchor_session_name", "")).strip(),
        "boundary_signature": indexed_boundary,
        "expected_targets": [
            str(race).strip()
            for race in index_payload.get("expected_targets", [])
            if str(race).strip()
        ]
        if isinstance(index_payload.get("expected_targets"), list)
        else [],
        "ready_races": ready_races,
    }


def prediction_action_state(
    precompute_filter_meta: dict[str, Any],
    *,
    selected_race_prediction_available: bool = False,
) -> dict[str, Any]:
    """Resolve whether a persisted dashboard prediction is available."""
    if bool(precompute_filter_meta.get("fallback_boundary_active")):
        pending_message = (
            "A newer checkpoint exists beyond the warmed horizon. The selected race will stay "
            "on the latest warmed persisted checkpoint until the next warmup catches up."
        )
        return {
            "disabled": False,
            "pending_message": pending_message,
            "help": pending_message,
        }

    if bool(precompute_filter_meta.get("applied")):
        return {"disabled": False, "pending_message": None}

    if selected_race_prediction_available:
        return {"disabled": False, "pending_message": None}

    stale_reason = str(precompute_filter_meta.get("stale_reason", "")).strip()
    if stale_reason == "artifact_hash_mismatch":
        pending_message = (
            "Stored predictions exist for an older artifact set, but the current artifact set "
            "has not been warmed yet. Run warmup again after artifact or config changes."
        )
    elif stale_reason == "boundary_mismatch":
        pending_message = (
            "Current session boundary is ahead of the warmed horizon. Predictions will be available "
            "after the next hourly warmup persists this checkpoint."
        )
    elif bool(precompute_filter_meta.get("scope_applied")):
        pending_message = (
            "Persisted horizon metadata is still warming for the current checkpoint. "
            "The dashboard will update after the next hourly warmup completes."
        )
    else:
        pending_message = (
            "Persisted-only mode is enabled and no warmed horizon is available yet. "
            "Run warmup before using dashboard predictions."
        )

    return {
        "disabled": True,
        "pending_message": pending_message,
        "help": pending_message,
    }
