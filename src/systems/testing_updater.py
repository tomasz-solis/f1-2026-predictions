"""
Testing/Practice Directionality Updater

Updates team car directionality metrics from pre-season testing or weekend
practice sessions. It can be run manually (CLI) and is also invoked by the
dashboard when new FP sessions are completed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import fastf1

from src.extractors.performance import extract_all_teams_performance
from src.persistence.artifact_store import ArtifactStore
from src.persistence.config import should_write_to_db
from src.systems.compound_analyzer import (
    aggregate_compound_samples,
    extract_compound_metrics,
    normalize_compound_metrics_across_teams,
)
from src.systems.testing_updater_flow import (
    apply_team_updates as _apply_team_updates,
)
from src.systems.testing_updater_flow import (
    collect_sessions_for_events as _collect_sessions_for_events,
)
from src.systems.testing_updater_flow import (
    load_characteristics_payload as _load_characteristics_payload,
)
from src.systems.testing_updater_flow import (
    raise_if_no_loaded_sessions as _raise_if_no_loaded_sessions,
)
from src.systems.testing_updater_flow import (
    validate_update_options as _validate_update_options,
)
from src.systems.testing_updater_flow import (
    write_characteristics_if_needed as _write_characteristics_if_needed,
)
from src.systems.testing_updater_metrics import (
    _aggregate_metric_samples,
    _blend_directionality,
    _build_directionality_from_metrics,
    _canonicalize_team_name,
    _classify_run_laps,
    _estimate_tire_deg_slope,
    _extract_team_payload,
    _filter_valid_laps,
    _median_lap_seconds,
    _median_timedelta_seconds,
    _normalize_lower_better,
    _normalize_tire_deg_scores,
    _select_program_aware_laps,
)
from src.systems.testing_updater_metrics import (
    _collect_session_metrics as _collect_session_metrics_impl,
)
from src.systems.testing_updater_metrics import (
    _count_team_selected_laps as _count_team_selected_laps_impl,
)
from src.systems.testing_updater_metrics import (
    _count_team_valid_laps as _count_team_valid_laps_impl,
)
from src.systems.testing_updater_metrics import (
    _extract_session_compound_metrics as _extract_session_compound_metrics_impl,
)
from src.systems.testing_updater_sessions import (
    coerce_utc_datetime as _coerce_utc_datetime_impl,
)
from src.systems.testing_updater_sessions import (
    extract_testing_day as _extract_testing_day_impl,
)
from src.systems.testing_updater_sessions import (
    extract_testing_number as _extract_testing_number_impl,
)
from src.systems.testing_updater_sessions import (
    get_testing_event_with_backends as _get_testing_event_with_backends_impl,
)
from src.systems.testing_updater_sessions import (
    is_testing_event as _is_testing_event_impl,
)
from src.systems.testing_updater_sessions import (
    load_sessions_for_event as _load_sessions_for_event_impl,
)
from src.systems.testing_updater_sessions import (
    load_testing_session_with_backends as _load_testing_session_with_backends_impl,
)
from src.systems.testing_updater_sessions import (
    normalize_name as _normalize_name_impl,
)
from src.systems.testing_updater_sessions import (
    normalize_testing_event_sessions as _normalize_testing_event_sessions_impl,
)
from src.systems.testing_updater_sessions import (
    resolve_testing_backends as _resolve_testing_backends_impl,
)
from src.systems.testing_updater_sessions import (
    resolve_testing_cache_dir as _resolve_testing_cache_dir_impl,
)
from src.systems.testing_updater_sessions import (
    testing_session_has_started as _testing_session_has_started_impl,
)
from src.utils.file_operations import atomic_json_write

__all__ = [
    "_aggregate_metric_samples",
    "_blend_directionality",
    "_build_directionality_from_metrics",
    "_canonicalize_team_name",
    "_classify_run_laps",
    "_coerce_utc_datetime",
    "_count_team_selected_laps",
    "_count_team_valid_laps",
    "_extract_testing_day",
    "_extract_testing_number",
    "_extract_team_payload",
    "_filter_valid_laps",
    "_get_testing_event_with_backends",
    "_is_testing_event",
    "_load_sessions_for_event",
    "_load_testing_session_with_backends",
    "_median_lap_seconds",
    "_median_timedelta_seconds",
    "_normalize_lower_better",
    "_normalize_testing_event_sessions",
    "_normalize_tire_deg_scores",
    "_resolve_testing_backends",
    "_resolve_testing_cache_dir",
    "_select_program_aware_laps",
    "_testing_session_has_started",
    "extract_all_teams_performance",
    "extract_compound_metrics",
    "normalize_compound_metrics_across_teams",
    "update_from_testing_sessions",
]

logger = logging.getLogger(__name__)
logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logging.getLogger("fastf1.logger").setLevel(logging.CRITICAL)
logging.getLogger("requests_cache").setLevel(logging.CRITICAL)
try:
    fastf1.set_log_level("CRITICAL")
except (AttributeError, TypeError):
    pass


DEFAULT_SESSION_CANDIDATES = [
    "FP1",
    "FP2",
    "FP3",
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Day 1",
    "Day 2",
    "Day 3",
]

_TESTING_BACKENDS = ("f1timing", "fastf1", None)
_TESTING_CACHE_ROOT = Path("data/raw")
_DEFAULT_TESTING_CACHE_DIR = _TESTING_CACHE_ROOT / ".fastf1_cache_testing"

_SESSION_AGGREGATION_MODES = ("mean", "median", "laps_weighted")
_RUN_PROFILE_MODES = ("balanced", "all", "short_run", "long_run")
_PROFILES_FOR_STORAGE = ("balanced", "short_run", "long_run")
_TESTING_CHARACTERISTIC_METRICS = (
    "slow_corner_performance",
    "medium_corner_performance",
    "fast_corner_performance",
    "braking_performance",
    "top_speed",
    "overall_pace",
    "consistency",
    "tire_deg_slope",
    "tire_deg_performance",
)


def _normalize_name(value: str) -> str:
    """Normalize names for robust matching."""
    return _normalize_name_impl(value)


def _is_testing_event(event_name: str) -> bool:
    """Best-effort detection of testing events from user-provided name."""
    return _is_testing_event_impl(event_name)


def _extract_testing_day(session_name: str) -> int | None:
    """Map session label to a testing day number (1..3) if possible."""
    return _extract_testing_day_impl(session_name)


def _extract_testing_number(event_name: str) -> int | None:
    """Parse explicit test number from event name (e.g., 'Testing 2')."""
    return _extract_testing_number_impl(event_name)


def _resolve_testing_backends(
    preferred_backend: str | None = "auto",
) -> tuple[str | None, ...]:
    """Resolve backend preference into an ordered list of backends to try."""
    return _resolve_testing_backends_impl(
        preferred_backend=preferred_backend,
        default_backends=_TESTING_BACKENDS,
    )


def _resolve_testing_cache_dir(cache_dir: str | None = None) -> Path:
    """
    Resolve testing cache location.

    Relative paths are kept under data/raw to avoid repository root clutter.
    """
    return _resolve_testing_cache_dir_impl(
        cache_dir=cache_dir,
        default_cache_dir=_DEFAULT_TESTING_CACHE_DIR,
        cache_root=_TESTING_CACHE_ROOT,
    )


def _coerce_utc_datetime(value) -> datetime | None:
    """Convert FastF1 event datetime values to UTC-aware datetime."""
    return _coerce_utc_datetime_impl(value)


def _testing_session_has_started(
    event: fastf1.events.Event, day_number: int, now_utc: datetime | None = None
) -> bool:
    """Check whether a testing day has started based on UTC session timestamp."""
    return _testing_session_has_started_impl(
        event=event,
        day_number=day_number,
        now_utc=now_utc,
        coerce_utc_datetime_fn=_coerce_utc_datetime,
    )


def _get_testing_event_with_backends(
    year: int,
    test_number: int,
    testing_backends: tuple[str | None, ...],
    error_messages: list[str] | None = None,
) -> fastf1.events.Event | None:
    """Load a testing event, trying explicit backends before auto mode."""
    return _get_testing_event_with_backends_impl(
        year=year,
        test_number=test_number,
        testing_backends=testing_backends,
        error_messages=error_messages,
        fastf1_get_testing_event=fastf1.get_testing_event,
        logger_obj=logger,
    )


def _normalize_testing_event_sessions(event: fastf1.events.Event) -> None:
    """
    Normalize testing session labels to FastF1-compatible names.

    Some schedules expose "Day 1/2/3". FastF1 Session initialization expects
    canonical names like "Practice 1/2/3".
    """
    _normalize_testing_event_sessions_impl(event)


def _load_testing_session_with_backends(
    year: int,
    test_number: int,
    day_number: int,
    testing_backends: tuple[str | None, ...],
    error_messages: list[str] | None = None,
) -> fastf1.core.Session | None:
    """
    Load a testing session and verify laps are actually accessible.

    This avoids reporting sessions as discovered when `session.laps` would still
    raise DataNotLoadedError after `load()`.
    """
    return _load_testing_session_with_backends_impl(
        year=year,
        test_number=test_number,
        day_number=day_number,
        testing_backends=testing_backends,
        error_messages=error_messages,
        fastf1_get_testing_event=fastf1.get_testing_event,
        normalize_testing_event_sessions_fn=_normalize_testing_event_sessions,
        logger_obj=logger,
    )


def _load_sessions_for_event(
    year: int,
    event_name: str,
    session_candidates: list[str],
    testing_backends: tuple[str | None, ...] = _TESTING_BACKENDS,
    error_messages: list[str] | None = None,
) -> list[tuple[str, fastf1.core.Session]]:
    """
    Load available sessions for an event.

    Strategy:
    1) For non-testing events: use regular `get_session(event_name, session_name)`.
    2) For testing events: use `get_testing_event` + `get_testing_session`.
    """
    return _load_sessions_for_event_impl(
        year=year,
        event_name=event_name,
        session_candidates=session_candidates,
        testing_backends=testing_backends,
        error_messages=error_messages,
        is_testing_event_fn=_is_testing_event,
        extract_testing_number_fn=_extract_testing_number,
        extract_testing_day_fn=_extract_testing_day,
        get_testing_event_with_backends_fn=_get_testing_event_with_backends,
        testing_session_has_started_fn=_testing_session_has_started,
        load_testing_session_with_backends_fn=_load_testing_session_with_backends,
        fastf1_get_session=fastf1.get_session,
        logger_obj=logger,
    )


def _count_team_selected_laps(
    session: fastf1.core.Session,
    known_teams: set[str],
    run_profile: str = "all",
) -> dict[str, float]:
    """Compatibility wrapper around extracted lap-count helper."""
    return _count_team_selected_laps_impl(
        session=session,
        known_teams=known_teams,
        run_profile=run_profile,
        canonicalize_team_name_fn=_canonicalize_team_name,
        filter_valid_laps_fn=_filter_valid_laps,
        select_program_aware_laps_fn=_select_program_aware_laps,
    )


def _count_team_valid_laps(
    session: fastf1.core.Session,
    known_teams: set[str],
) -> dict[str, float]:
    """Compatibility wrapper around extracted valid-lap helper."""
    return _count_team_valid_laps_impl(
        session=session,
        known_teams=known_teams,
        canonicalize_team_name_fn=_canonicalize_team_name,
        filter_valid_laps_fn=_filter_valid_laps,
    )


def _collect_session_metrics(
    session: fastf1.core.Session,
    session_key: str,
    known_teams: set[str],
    run_profile: str = "balanced",
    diagnostics: list[str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compatibility wrapper that preserves monkeypatchable module-level dependencies."""
    return _collect_session_metrics_impl(
        session=session,
        session_key=session_key,
        known_teams=known_teams,
        run_profile=run_profile,
        diagnostics=diagnostics,
        canonicalize_team_name_fn=_canonicalize_team_name,
        filter_valid_laps_fn=_filter_valid_laps,
        select_program_aware_laps_fn=_select_program_aware_laps,
        classify_run_laps_fn=_classify_run_laps,
        median_lap_seconds_fn=_median_lap_seconds,
        extract_team_payload_fn=_extract_team_payload,
        estimate_tire_deg_slope_fn=_estimate_tire_deg_slope,
        extract_all_teams_performance_fn=extract_all_teams_performance,
        normalize_lower_better_fn=_normalize_lower_better,
        normalize_tire_deg_scores_fn=_normalize_tire_deg_scores,
    )


def _extract_session_compound_metrics(
    session: fastf1.core.Session,
    event_name: str,
    known_teams: set[str],
) -> dict[str, dict[str, dict[str, float | str | None]]]:
    """Compatibility wrapper that keeps compound helpers patchable in this module."""
    return _extract_session_compound_metrics_impl(
        session=session,
        event_name=event_name,
        known_teams=known_teams,
        canonicalize_team_name_fn=_canonicalize_team_name,
        extract_compound_metrics_fn=extract_compound_metrics,
        normalize_compound_metrics_across_teams_fn=normalize_compound_metrics_across_teams,
    )


def update_from_testing_sessions(
    year: int,
    events: list[str],
    data_dir: str = "data/processed",
    sessions: list[str] | None = None,
    characteristics_year: int | None = None,
    testing_backend: str | None = "auto",
    cache_dir: str = str(_DEFAULT_TESTING_CACHE_DIR),
    force_renew_cache: bool = False,
    new_weight: float = 0.7,
    directionality_scale: float = 0.10,
    session_aggregation: str = "mean",
    run_profile: str = "balanced",
    dry_run: bool = False,
) -> dict:
    """
    Update car directionality from testing or practice sessions.

    This function only updates testing-related fields:
    - teams[*].directionality
    - teams[*].testing_characteristics
    """
    if not events:
        raise ValueError("At least one event name is required")

    target_year = characteristics_year or year
    characteristics_file, characteristics = _load_characteristics_payload(data_dir, target_year)
    _validate_update_options(
        session_aggregation=session_aggregation,
        run_profile=run_profile,
        session_aggregation_modes=_SESSION_AGGREGATION_MODES,
        run_profile_modes=_RUN_PROFILE_MODES,
    )

    session_candidates = sessions or DEFAULT_SESSION_CANDIDATES
    known_teams = set(characteristics["teams"].keys())
    testing_backends = _resolve_testing_backends(testing_backend)

    cache_path = _resolve_testing_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path), force_renew=force_renew_cache)

    collection = _collect_sessions_for_events(
        year=year,
        events=events,
        session_candidates=session_candidates,
        testing_backends=testing_backends,
        known_teams=known_teams,
        run_profile=run_profile,
        profiles_for_storage=_PROFILES_FOR_STORAGE,
        load_sessions_for_event=_load_sessions_for_event,
        collect_session_metrics=_collect_session_metrics,
        count_team_selected_laps=_count_team_selected_laps,
        extract_session_compound_metrics=_extract_session_compound_metrics,
        logger=logger,
    )
    _raise_if_no_loaded_sessions(
        discovered_sessions=collection.discovered_sessions,
        loaded_sessions=collection.loaded_sessions,
        extraction_diagnostics=collection.extraction_diagnostics,
        load_errors=collection.load_errors,
    )

    now_iso = datetime.now().isoformat()
    updated_teams = _apply_team_updates(
        characteristics=characteristics,
        metric_samples=collection.metric_samples,
        profile_metric_samples=collection.profile_metric_samples,
        team_sessions_used=collection.team_sessions_used,
        team_profile_sessions_used=collection.team_profile_sessions_used,
        compound_metrics_by_session=collection.compound_metrics_by_session,
        now_iso=now_iso,
        session_aggregation=session_aggregation,
        run_profile=run_profile,
        directionality_scale=directionality_scale,
        new_weight=new_weight,
        profiles_for_storage=_PROFILES_FOR_STORAGE,
        testing_characteristic_metrics=_TESTING_CHARACTERISTIC_METRICS,
        aggregate_metric_samples=_aggregate_metric_samples,
        build_directionality_from_metrics=_build_directionality_from_metrics,
        blend_directionality=_blend_directionality,
        aggregate_compound_samples=aggregate_compound_samples,
    )

    if not updated_teams:
        raise ValueError(
            "Sessions loaded but no teams were matched to characteristics file team names."
        )

    characteristics["directionality_source"] = "SESSION_EXTRACTION"
    characteristics["directionality_last_updated"] = now_iso
    characteristics["directionality_meta"] = {
        "year": year,
        "characteristics_year": target_year,
        "events": events,
        "sessions_loaded": collection.loaded_sessions,
        "testing_backend": testing_backend or "auto",
        "cache_dir": str(cache_path),
        "force_renew_cache": force_renew_cache,
        "new_weight": new_weight,
        "directionality_scale": directionality_scale,
        "session_aggregation": session_aggregation,
        "run_profile": run_profile,
        "profiles_captured": list(_PROFILES_FOR_STORAGE),
    }

    artifact_key = f"{target_year}::car_characteristics"
    latest_known_version = 0
    artifact_store = ArtifactStore(data_root="data")
    if should_write_to_db():
        try:
            latest_known_version = int(
                artifact_store.get_latest_version("car_characteristics", artifact_key)
            )
        except Exception as exc:
            logger.warning(
                "Could not resolve latest DB version for %s before write: %s",
                artifact_key,
                exc,
            )

    _write_characteristics_if_needed(
        characteristics_file=characteristics_file,
        characteristics=characteristics,
        now_iso=now_iso,
        dry_run=dry_run,
        atomic_json_write=atomic_json_write,
        latest_known_version=latest_known_version,
    )

    if should_write_to_db() and not dry_run:
        try:
            artifact_store.save_artifact(
                artifact_type="car_characteristics",
                artifact_key=artifact_key,
                data=characteristics,
                version=int(characteristics.get("version", 1)),
            )
        except Exception as exc:
            logger.warning(
                "Could not persist testing/practice characteristics to ArtifactStore (%s): %s",
                artifact_key,
                exc,
            )

    return {
        "year": year,
        "characteristics_year": target_year,
        "events": events,
        "loaded_sessions": collection.loaded_sessions,
        "updated_teams": sorted(updated_teams),
        "characteristics_file": str(characteristics_file),
        "testing_backend": testing_backend or "auto",
        "cache_dir": str(cache_path),
        "force_renew_cache": force_renew_cache,
        "session_aggregation": session_aggregation,
        "run_profile": run_profile,
        "profiles_captured": list(_PROFILES_FOR_STORAGE),
        "dry_run": dry_run,
    }
