"""Dashboard pages and page-level orchestration."""

import logging
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fastf1
import streamlit as st

from src.utils.weekend import _get_schedule_rows, is_sprint_weekend

from . import team_comparison as _team_comparison
from .accuracy_view import (
    METRIC_OPTIONS,
    render_overall_accuracy_metrics,
    render_saved_predictions_summary,
    render_target_sections,
)
from .cache import get_artifact_versions
from .layout import BRAND_LAST_UPDATED, BRAND_MODEL_VERSION, ENABLE_PREDICTION_ACCURACY_TAB
from .live_prediction_flow import (
    PrecomputedPredictionUnavailableError,
)
from .live_prediction_flow import (
    execute_live_prediction_pipeline_core as _execute_live_prediction_pipeline_core,
)
from .live_prediction_flow import (
    render_prediction_results_core as _render_prediction_results_core,
)
from .live_prediction_flow import (
    save_prediction_if_enabled_core as _save_prediction_if_enabled_core,
)
from .page_content import (
    CONTACT_PAGE_HTML,
    MODEL_INSIGHTS_MARKDOWN,
    QUALIFYING_HYPERPARAMETERS_MARKDOWN,
    RACE_HYPERPARAMETERS_MARKDOWN,
)
from .precomputed_predictions import (
    compute_artifact_hash,
    get_prediction_precompute_config,
    has_precompute_horizon_for_year,
    load_precompute_horizon_index,
    load_precomputed_prediction,
)
from .prediction_flow import CompetitiveSessionStatusUnavailableError, run_prediction
from .rendering import (
    display_prediction_result,
    render_notice_banner,
    render_prediction_hero_deck,
)
from .update_flow import (
    _boundary_signature,
    _build_event_boundary_snapshot,
    auto_update_if_needed,
    auto_update_practice_characteristics_if_needed,
    detect_event_boundary_refresh_if_needed,
)

logger = logging.getLogger(__name__)
DEFAULT_SEASON = 2026
_MIN_SELECTABLE_SEASON = 2024
_FASTF1_CACHE_DIRS = (
    Path("data/raw/.fastf1_cache"),
    Path("data/raw/.fastf1_cache_testing"),
)
_NON_DISTINCT_RACE_TOKENS = {"grand", "prix", "gp"}
_SESSION_LABELS = {
    "FP1": "Free Practice 1",
    "FP2": "Free Practice 2",
    "FP3": "Free Practice 3",
    "SQ": "Sprint Qualifying",
    "SPRINT": "Sprint Race",
    "Q": "Qualifying",
    "R": "Grand Prix Race",
}
_SESSION_ORDER = {
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "SQ": 4,
    "SPRINT": 5,
    "Q": 6,
    "R": 7,
}

# Backwards-compatible exports for tests and existing imports.
_DEFAULT_TEAM_COLOR = _team_comparison._DEFAULT_TEAM_COLOR
_build_team_comparison_dataframe = _team_comparison._build_team_comparison_dataframe
_coerce_unit_metric = _team_comparison._coerce_unit_metric
_collect_profile_names = _team_comparison._collect_profile_names
_default_team_selection = _team_comparison._default_team_selection
_hex_to_rgba = _team_comparison._hex_to_rgba
_load_team_characteristics_payload = _team_comparison._load_team_characteristics_payload
_render_team_comparison_section = _team_comparison._render_team_comparison_section
_resolve_profile_metrics = _team_comparison._resolve_profile_metrics
_team_brand_color = _team_comparison._team_brand_color


def _available_seasons() -> list[int]:
    """Return season choices shown in the dashboard UI."""
    current_year = datetime.now(UTC).year
    latest = max(DEFAULT_SEASON, current_year)
    earliest = min(DEFAULT_SEASON, _MIN_SELECTABLE_SEASON)
    return list(range(latest, earliest - 1, -1))


def _get_selected_season(default: int = DEFAULT_SEASON) -> int:
    """Read selected season from Streamlit session state with safe fallback."""
    try:
        raw_value = st.session_state.get("selected_season", default)
    except Exception:
        raw_value = default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _set_selected_season(year: int) -> None:
    """Persist selected season in Streamlit session state when available."""
    try:
        st.session_state["selected_season"] = int(year)
    except Exception:
        return


def render_team_comparison_page() -> None:
    """Render standalone team comparison tab."""
    st.header("Team Comparison")
    st.markdown(
        "Compare team characteristic fingerprints from synced session inputs. "
        "Profile metrics and season-prior baseline are separate signals and can diverge."
    )
    _render_team_comparison_section(year=_get_selected_season())


def _clear_fastf1_race_cache(year: int, race_name: str) -> None:
    """
    Clear FastF1 cache for a specific race to force fresh data fetch.

    This invalidates all cached session data for the race, including practice sessions,
    qualifying, and race results. The next FastF1 call will fetch fresh data from the API.
    """
    import shutil

    for cache_dir in _FASTF1_CACHE_DIRS:
        year_dir = cache_dir / str(year)
        if not year_dir.exists():
            continue

        removed_paths: list[Path] = []
        try:
            for event_cache_dir in year_dir.iterdir():
                if not event_cache_dir.is_dir():
                    continue
                if not _cache_dir_matches_race(event_cache_dir.name, race_name):
                    continue
                try:
                    shutil.rmtree(event_cache_dir)
                    removed_paths.append(event_cache_dir)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(f"Could not clear FastF1 cache path {event_cache_dir}: {exc}")
        except Exception as exc:
            logger.warning(f"Could not inspect FastF1 cache at {year_dir}: {exc}")
            continue

        if removed_paths:
            removed_labels = ", ".join(path.name for path in removed_paths[:5])
            if len(removed_paths) > 5:
                removed_labels += ", ..."
            logger.info(
                f"Cleared FastF1 cache for {race_name} {year}: "
                f"{len(removed_paths)} path(s) in {cache_dir} ({removed_labels})"
            )
        else:
            logger.info(f"No FastF1 cache paths matched {race_name} {year} in {cache_dir}")


def _normalize_cache_fragment(value: str) -> str:
    """Normalize cache path fragments for case-insensitive race-name matching."""
    normalized = unicodedata.normalize("NFKD", str(value))
    folded = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch.lower() for ch in folded if ch.isalnum())


def _cache_dir_matches_race(cache_dir_name: str, race_name: str) -> bool:
    """
    Return True when a FastF1 event cache directory corresponds to race_name.

    FastF1 event directories are often date-prefixed (for example
    `2025-04-13_Bahrain_Grand_Prix`), so exact-path matching is insufficient.
    """
    normalized_dir = _normalize_cache_fragment(cache_dir_name)
    normalized_race = _normalize_cache_fragment(race_name)
    if not normalized_dir or not normalized_race:
        return False

    if normalized_race in normalized_dir:
        return True

    race_tokens = [
        token
        for token in (_normalize_cache_fragment(part) for part in race_name.split())
        if token and token not in _NON_DISTINCT_RACE_TOKENS
    ]
    return bool(race_tokens) and all(token in normalized_dir for token in race_tokens)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_race_options_cached(year: int) -> tuple[list[str], str | None]:
    """Load race options and cache schedule fetches for responsiveness."""

    def _options_from_schedule_rows(rows: list[tuple[str, str]]) -> list[str]:
        options: list[str] = []
        for race_name, event_format in rows:
            if not race_name or "testing" in race_name.lower():
                continue
            if "sprint" in str(event_format).lower():
                options.append(f"{race_name} (Sprint)")
            else:
                options.append(race_name)
        return options

    try:
        schedule = fastf1.get_event_schedule(year)
        race_rows: list[tuple[str, str]] = []
        if "EventName" in schedule.columns and "EventFormat" in schedule.columns:
            for _, event in schedule.iterrows():
                race_rows.append(
                    (
                        str(event.get("EventName", "")).strip(),
                        str(event.get("EventFormat", "")).strip(),
                    )
                )
        race_options = _options_from_schedule_rows(race_rows)
        if race_options:
            return race_options, None
        fastf1_error = f"FastF1 returned no race events for {year}"
    except Exception as exc:
        fastf1_error = str(exc)

    fallback_rows = list(_get_schedule_rows(year))
    fallback_options = _options_from_schedule_rows(fallback_rows)
    if fallback_options:
        logger.warning(
            "Using local fallback race options for %s because FastF1 schedule load failed: %s",
            year,
            fastf1_error,
        )
        return fallback_options, None

    return (
        [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
        ],
        fastf1_error,
    )


def _load_race_options(year: int = DEFAULT_SEASON) -> list[str]:
    """Load race options from FastF1 schedule with sprint labels."""
    race_options, error = _load_race_options_cached(year)
    if error:
        st.warning(f"Failed to load {year} calendar: {error}. Using minimal fallback list.")
    return race_options


def _parse_refresh_timestamp(value: Any) -> datetime | None:
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


def _latest_dashboard_refresh_timestamp(year: int) -> datetime | None:
    """
    Return the newest dashboard data-refresh timestamp for a season.

    The dashboard can run in DB-backed or local-file modes, so this resolver
    combines persisted artifact timestamps, horizon-index timestamps, and local
    file mtimes for the dashboard inputs that actually drive predictions.
    """
    timestamps: list[datetime] = []
    artifact_hash = ""

    try:
        artifact_versions = get_artifact_versions(year=year)
    except Exception as exc:
        logger.warning("Could not load artifact versions for refresh stamp: %s", exc)
        artifact_versions = {}

    for version_payload in artifact_versions.values():
        if not isinstance(version_payload, tuple) or len(version_payload) < 2:
            continue
        parsed = _parse_refresh_timestamp(version_payload[1])
        if parsed is not None:
            timestamps.append(parsed)

    try:
        artifact_hash = compute_artifact_hash(artifact_versions)
    except Exception as exc:
        logger.warning("Could not compute artifact hash for refresh stamp: %s", exc)

    if artifact_hash:
        try:
            horizon_index = load_precompute_horizon_index(year=year, artifact_hash=artifact_hash)
        except Exception as exc:
            logger.warning("Could not load horizon index for refresh stamp: %s", exc)
            horizon_index = None
        if isinstance(horizon_index, dict):
            parsed = _parse_refresh_timestamp(horizon_index.get("updated_at"))
            if parsed is not None:
                timestamps.append(parsed)

    season_year = int(year)
    refresh_paths = (
        Path(f"data/processed/car_characteristics/{season_year}_car_characteristics.json"),
        Path(f"data/processed/track_characteristics/{season_year}_track_characteristics.json"),
        Path(f"data/processed/driver_characteristics/{season_year}_driver_characteristics.json"),
        Path("data/processed/driver_characteristics.json"),
        Path("data/systems/practice_characteristics_state.json"),
        Path("data/systems/precompute_horizon_index.json"),
        Path("data/systems/precomputed_predictions.json"),
    )
    for refresh_path in refresh_paths:
        if not refresh_path.exists():
            continue
        try:
            mtime = datetime.fromtimestamp(refresh_path.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        timestamps.append(mtime)

    return max(timestamps) if timestamps else None


def _dashboard_refresh_label(year: int) -> str:
    """Format the latest dashboard refresh stamp for the hero card."""
    latest_refresh = _latest_dashboard_refresh_timestamp(year)
    if latest_refresh is None:
        return BRAND_LAST_UPDATED
    return latest_refresh.strftime("%Y-%m-%d %H:%M UTC")


@st.cache_data(ttl=900, show_spinner=False)
def _load_schedule_event_rows_cached(year: int) -> tuple[tuple[str, str, str], ...]:
    """Load cached schedule rows with serialized event dates for horizon filtering."""
    rows: list[tuple[str, str, str]] = []

    try:
        schedule = fastf1.get_event_schedule(year)
        if "EventName" in schedule.columns and "EventFormat" in schedule.columns:
            for _, event in schedule.iterrows():
                event_name = str(event.get("EventName", "")).strip()
                event_format = str(event.get("EventFormat", "")).strip()
                if not event_name:
                    continue
                event_date = event.get("EventDate")
                if hasattr(event_date, "to_pydatetime"):
                    try:
                        event_date = event_date.to_pydatetime()
                    except Exception:
                        event_date = None
                if isinstance(event_date, datetime):
                    if event_date.tzinfo is None:
                        event_date = event_date.replace(tzinfo=UTC)
                    else:
                        event_date = event_date.astimezone(UTC)
                    event_date_iso = event_date.isoformat()
                else:
                    event_date_iso = ""
                rows.append((event_name, event_format, event_date_iso))
    except Exception as exc:
        logger.warning("Could not load dated schedule rows for %s: %s", year, exc)

    if rows:
        return tuple(rows)

    return tuple(
        (str(event_name).strip(), str(event_format).strip(), "")
        for event_name, event_format in _get_schedule_rows(year)
        if str(event_name).strip()
    )


def _resolve_dashboard_race_horizon(year: int, horizon_races: int) -> list[str]:
    """Return the next configured race window from the live schedule when dates are available."""
    schedule_rows = _load_schedule_event_rows_cached(year)
    if not schedule_rows:
        return []

    competitive_rows: list[tuple[str, datetime | None]] = []
    for event_name, event_format, event_date_iso in schedule_rows:
        normalized_name = str(event_name).strip()
        normalized_format = str(event_format).strip().lower()
        if not normalized_name:
            continue
        if "testing" in normalized_name.lower() or "testing" in normalized_format:
            continue
        event_date: datetime | None = None
        candidate = str(event_date_iso).strip()
        if candidate:
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    event_date = parsed.replace(tzinfo=UTC)
                else:
                    event_date = parsed.astimezone(UTC)
        competitive_rows.append((normalized_name, event_date))

    if not competitive_rows:
        return []

    horizon_size = max(1, int(horizon_races))
    now_utc = datetime.now(UTC)
    anchor_index: int | None = None
    for index, (_, event_date) in enumerate(competitive_rows):
        if event_date is None:
            continue
        if event_date >= now_utc:
            anchor_index = index
            break

    if anchor_index is None:
        return []

    return [
        race_name
        for race_name, _event_date in competitive_rows[anchor_index : anchor_index + horizon_size]
    ]


@st.cache_data(show_spinner=False, ttl=120)
def _current_anchor_boundary_signature(year: int, anchor_race_name: str) -> str | None:
    """
    Resolve current boundary signature for horizon anchor race.

    Returns ``None`` when boundary state cannot be validated.
    """
    try:
        is_sprint = bool(is_sprint_weekend(year, anchor_race_name))
    except Exception as exc:
        logger.warning(
            "Could not determine weekend type for anchor boundary validation (%s %s): %s",
            year,
            anchor_race_name,
            exc,
        )
        return None

    from src.utils.session_detector import SessionDetector

    try:
        snapshot = _build_event_boundary_snapshot(
            year=year,
            race_name=anchor_race_name,
            is_sprint=is_sprint,
            session_detector=SessionDetector(),
            now_utc=datetime.now(UTC),
        )
    except Exception as exc:
        logger.warning(
            "Could not build anchor boundary snapshot for dropdown filtering (%s %s): %s",
            year,
            anchor_race_name,
            exc,
        )
        return None

    if not bool(snapshot.get("has_schedule_data")):
        return None
    return _boundary_signature(snapshot)


def _selected_race_persisted_prediction_available(
    *,
    year: int,
    race_name: str,
    weather: str,
) -> bool:
    """Return whether the selected race already has a persisted prediction at the live boundary.

    The prediction page should only hard-block when the selected race itself cannot
    be served. Full warmed-horizon metadata can lag behind without preventing the
    current race from loading when an exact persisted prediction already exists.
    """
    try:
        artifact_versions = get_artifact_versions(year=year)
        artifact_hash = compute_artifact_hash(artifact_versions)
    except Exception as exc:
        logger.warning(
            "Could not compute artifact hash while checking selected-race availability: %s",
            exc,
        )
        return False

    current_boundary_signature = _current_anchor_boundary_signature(year, race_name)
    if not current_boundary_signature:
        return False

    try:
        payload = load_precomputed_prediction(
            year=year,
            race_name=race_name,
            weather=str(weather).strip().lower(),
            artifact_hash=artifact_hash,
            boundary_signature=current_boundary_signature,
        )
    except Exception as exc:
        logger.warning(
            "Could not load persisted selected-race prediction for %s %s [%s]: %s",
            year,
            race_name,
            weather,
            exc,
        )
        return False

    return isinstance(payload, dict)


def _load_ready_races_from_current_store(
    *,
    year: int,
    race_names: list[str],
    artifact_hash: str,
    weather_scenarios: list[str],
) -> list[str]:
    """Return races that already have full persisted weather coverage for the current boundary."""
    ready_races: list[str] = []
    expected_weather = [
        str(weather).strip().lower() for weather in weather_scenarios if str(weather).strip()
    ]
    if not race_names or not expected_weather:
        return ready_races

    for race_name in race_names:
        boundary_signature = _current_anchor_boundary_signature(year, race_name)
        if not boundary_signature:
            continue
        has_full_coverage = True
        for weather in expected_weather:
            payload = load_precomputed_prediction(
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


def _maybe_scope_race_options_to_planned_horizon(
    *,
    race_options: list[str],
    planned_races: list[str],
    requested_horizon: int,
) -> tuple[list[str], bool]:
    """
    Narrow broad calendar dropdowns to the live planned horizon.

    The live page normally passes the full season calendar into this filter, but
    tests and internal callers can also provide a curated subset. Those smaller
    lists should stay intact so persisted ready-race metadata remains the source
    of truth instead of being pre-trimmed by schedule-based fallback logic.
    """
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


def _filter_race_options_to_precomputed_horizon(
    *,
    year: int,
    race_options: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """
    Filter race dropdown options to precomputed races for the active artifact state.

    Returns:
        Tuple of ``(filtered_options, metadata)`` where metadata includes whether
        filtering was applied and contextual details for user-facing captions.
    """
    if not race_options:
        return race_options, {"applied": False}

    settings = get_prediction_precompute_config()
    requested_horizon = max(1, int(settings.get("horizon_races", 3)))
    planned_races = _resolve_dashboard_race_horizon(year, requested_horizon)
    base_race_options = list(race_options)
    scoped_race_options, scope_applied = _maybe_scope_race_options_to_planned_horizon(
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
        artifact_versions = get_artifact_versions(year=year)
        artifact_hash = compute_artifact_hash(artifact_versions)
    except Exception as exc:
        logger.warning("Could not compute artifact hash for dropdown filtering: %s", exc)
        return scoped_race_options, scope_metadata

    index_payload = load_precompute_horizon_index(year=year, artifact_hash=artifact_hash)
    if not isinstance(index_payload, dict):
        ready_races = _load_ready_races_from_current_store(
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
        if has_precompute_horizon_for_year(
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

    current_boundary = _current_anchor_boundary_signature(year, anchor_race_name)
    if not current_boundary:
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
            "stale_reason": "boundary_unavailable",
            "anchor_race_name": anchor_race_name,
            "indexed_boundary_signature": indexed_boundary,
        }
    if current_boundary != indexed_boundary:
        ready_races_raw = index_payload.get("ready_races", [])
        ready_races = (
            [str(race).strip() for race in ready_races_raw if str(race).strip()]
            if isinstance(ready_races_raw, list)
            else []
        )
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

        ready_races = _load_ready_races_from_current_store(
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

    ready_races_raw = index_payload.get("ready_races", [])
    if not isinstance(ready_races_raw, list):
        return scoped_race_options, {
            **scope_metadata,
            "artifact_hash": artifact_hash,
        }
    ready_races = [str(race).strip() for race in ready_races_raw if str(race).strip()]
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


def _prediction_action_state(
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
            "Current session boundary is ahead of the warmed horizon. Predictions will unlock "
            "after the next hourly warmup persists this checkpoint."
        )
    elif bool(precompute_filter_meta.get("scope_applied")):
        pending_message = (
            "Persisted horizon metadata is still warming for the current checkpoint. "
            "The dashboard will unlock after the next hourly warmup completes."
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


def _save_prediction_if_enabled(
    enable_logging: bool,
    prediction_results: dict,
    is_sprint: bool,
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
    checkpoint_session_override: str | None = None,
) -> None:
    """Persist prediction artifacts for later accuracy tracking."""
    from src.utils.prediction_logger import PredictionLogger
    from src.utils.session_detector import SessionDetector

    _save_prediction_if_enabled_core(
        enable_logging=enable_logging,
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        race_name=race_name,
        weather=weather,
        year=year,
        detector_factory=SessionDetector,
        prediction_logger_factory=PredictionLogger,
        st_module=st,
        checkpoint_session_override=checkpoint_session_override,
    )


def _render_prediction_results(
    prediction_results: dict,
    is_sprint: bool,
    *,
    prediction_cache_hit: bool = False,
    pipeline_timing: dict[str, float] | None = None,
) -> None:
    """Render prediction sections for sprint and normal weekends."""
    _render_prediction_results_core(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        display_prediction_result_fn=display_prediction_result,
        st_module=st,
        prediction_cache_hit=prediction_cache_hit,
        pipeline_timing=pipeline_timing,
    )


def _latest_data_status_message(
    race_name: str,
    year: int,
    boundary_refresh: dict[str, object],
    practice_update: dict[str, object],
) -> str:
    """Build a short user-facing summary of the freshest session data in use."""

    def _session_label(session_name: str) -> str:
        normalized = str(session_name).strip().upper()
        return _SESSION_LABELS.get(normalized, normalized or "Unknown session")

    latest_elapsed = str(boundary_refresh.get("latest_elapsed_session") or "").strip().upper()
    if latest_elapsed:
        latest_label = _session_label(latest_elapsed)
        return (
            "Latest datapoint in use: "
            f"{race_name} {year} - {latest_label} ({latest_elapsed}). "
            "Predictions include all data available through this session."
        )

    completed_fp_raw = practice_update.get("completed_fp_sessions")
    completed_fp = (
        [str(session).strip().upper() for session in completed_fp_raw]
        if isinstance(completed_fp_raw, list)
        else []
    )
    if completed_fp:
        latest_practice = max(
            completed_fp,
            key=lambda session_name: _SESSION_ORDER.get(str(session_name).upper(), 0),
        )
        return (
            "Latest datapoint in use: "
            f"{race_name} {year} - {_session_label(latest_practice)} ({latest_practice}). "
            "No completed qualifying/race sessions yet."
        )

    reason = str(boundary_refresh.get("reason", "")).strip().lower()
    if reason == "schedule_unavailable":
        return (
            "Live session schedule is currently unavailable. "
            "Using the latest persisted artifacts and cached race-weekend state."
        )

    return (
        "Latest datapoint in use: pre-weekend baseline/testing only. "
        f"No completed sessions yet for {race_name} {year}."
    )


def _render_collapsible_runtime_messages(messages: list[tuple[str, str]]) -> None:
    """Render runtime notices compactly to avoid stacked info/warning banners."""
    unique_messages: list[tuple[str, str]] = []
    for level, message in messages:
        normalized_level = str(level).strip().lower()
        normalized_message = str(message).strip()
        if not normalized_message:
            continue
        item = (normalized_level, normalized_message)
        if item not in unique_messages:
            unique_messages.append(item)

    if not unique_messages:
        return

    primary_level, primary_message = unique_messages[0]
    remaining_count = len(unique_messages) - 1
    summary_text = (
        primary_message if remaining_count == 0 else f"{primary_message} (+{remaining_count} more)"
    )
    render_notice_banner(
        summary_text,
        tone=primary_level,
        label="Run context",
        st_module=st,
    )

    if remaining_count <= 0:
        return

    try:
        expander = st.expander("Run Context Details", expanded=False)
    except TypeError:
        expander = st.expander("Run Context Details")

    with expander:
        for level, message in unique_messages:
            prefix = "Info"
            if level == "warning":
                prefix = "Warning"
            elif level == "success":
                prefix = "Success"
            st.markdown(f"- **{prefix}:** {message}")


def execute_live_prediction_pipeline(
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
    force_refresh: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Load the persisted prediction selected in the dashboard.

    Kept separate from Streamlit rendering so tests can assert request-path behavior.

    Args:
        race_name: The name of the race
        weather: Weather forecast for the race
        year: Season year
        force_refresh: Compatibility flag. Manual dashboard refresh is intentionally disabled.
        progress_callback: Optional callback for progress updates
    """
    return _execute_live_prediction_pipeline_core(
        race_name=race_name,
        weather=weather,
        year=year,
        force_refresh=force_refresh,
        progress_callback=progress_callback,
        clear_fastf1_race_cache_fn=_clear_fastf1_race_cache,
        auto_update_if_needed_fn=auto_update_if_needed,
        is_sprint_weekend_fn=is_sprint_weekend,
        detect_event_boundary_refresh_if_needed_fn=detect_event_boundary_refresh_if_needed,
        auto_update_practice_characteristics_if_needed_fn=auto_update_practice_characteristics_if_needed,
        clear_resource_cache_fn=st.cache_resource.clear,
        clear_data_cache_fn=st.cache_data.clear,
        get_artifact_versions_fn=get_artifact_versions,
        run_prediction_fn=run_prediction,
    )


def _prediction_failure_hint(error: Exception) -> str | None:
    """Return the most relevant user-facing hint for a prediction failure."""
    message = str(error).strip()
    normalized_message = message.lower()

    if isinstance(error, CompetitiveSessionStatusUnavailableError) or (
        "could not verify completion state" in normalized_message
        and "predicted grid" in normalized_message
    ):
        return (
            "FastF1 has not exposed a reliable completion state for that session yet. "
            "This is a live-data sync problem, not a missing artifact problem. "
            "Retry shortly; if the session is clearly finished, clear that race's FastF1 cache "
            "and rerun."
        )

    artifact_error_markers = (
        "driver characteristics",
        "track characteristics",
        "extract_driver_characteristics.py",
        "could not locate driver characteristics fallback",
    )
    if any(marker in normalized_message for marker in artifact_error_markers):
        return (
            "Make sure data files are generated. Run: "
            "`python scripts/extract_driver_characteristics.py --years 2023,2024,2025,2026`"
            " (prefer a background job or local shell on Render; web-shell runs can hit memory limits)."
        )

    if isinstance(error, PrecomputedPredictionUnavailableError):
        return (
            "The dashboard is currently in persisted-prediction mode, so it will not simulate on demand. "
            "Warm the 3-race horizon first with "
            "`python scripts/warmup_precompute.py --year 2026` "
            "(add `--require-db` only when you want DB-backed warmup to be mandatory)."
        )

    return None


def render_live_prediction_page(enable_logging: bool) -> None:
    selected_season = _get_selected_season()
    render_prediction_hero_deck(
        title="Race Weekend Prediction",
        summary=(
            "Qualifying and race forecasts from the latest warmed checkpoint. "
            "Session context and data freshness stay visible without crowding the page."
        ),
        eyebrow="Weekend forecast",
        cards=[
            {
                "label": "Model",
                "value": BRAND_MODEL_VERSION,
                "meta": "Current dashboard release.",
                "tone": "accent",
            },
            {
                "label": "Updated",
                "value": _dashboard_refresh_label(selected_season),
                "meta": "Latest persisted data refresh stamp.",
                "tone": "neutral",
            },
            {
                "label": "Logging",
                "value": "ON" if enable_logging else "OFF",
                "meta": "One checkpoint saved per session boundary.",
                "tone": "success" if enable_logging else "warning",
            },
            {
                "label": "Serving",
                "value": "Persisted",
                "meta": "Warmup and cron refresh predictions outside the request path.",
                "tone": "neutral",
            },
        ],
        st_module=st,
    )

    season_options = _available_seasons()
    if selected_season not in season_options:
        season_options = [selected_season, *season_options]
    season_index = season_options.index(selected_season)
    control_col1, control_col2, control_col3 = st.columns([0.9, 1.8, 1.1], gap="large")
    with control_col1:
        selected_season = int(
            st.selectbox(
                "Season",
                options=season_options,
                index=season_index,
                key="selected_season",
                help=("Controls schedule lookup, warmed artifacts, and prediction execution year."),
            )
        )
    _set_selected_season(selected_season)

    race_options = _load_race_options(selected_season)
    race_options, precompute_filter_meta = _filter_race_options_to_precomputed_horizon(
        year=selected_season,
        race_options=race_options,
    )

    with control_col2:
        race_selection = st.selectbox("Grand Prix", race_options)
        race_name = race_selection.replace(" (Sprint)", "")

    with control_col3:
        weather = st.selectbox("Weather", ["dry", "rain", "mixed"])
    selected_race_prediction_available = _selected_race_persisted_prediction_available(
        year=selected_season,
        race_name=race_name,
        weather=weather,
    )

    selected_weekend_label = (
        "Sprint weekend" if race_selection.endswith("(Sprint)") else "Race weekend"
    )
    render_notice_banner(
        (
            f"{selected_weekend_label} selected for {race_name} {selected_season}. "
            "Predictions are served from warmed persisted artifacts; run warmup outside the "
            "dashboard if a newer checkpoint is needed."
        ),
        tone="info",
        label="Run setup",
        st_module=st,
    )

    if bool(precompute_filter_meta.get("applied")):
        ready_count = len(precompute_filter_meta.get("ready_races", []))
        expected_targets = precompute_filter_meta.get("expected_targets", [])
        horizon_count = len(expected_targets) if isinstance(expected_targets, list) else ready_count
        anchor_race = str(precompute_filter_meta.get("anchor_race_name", "")).strip()
        anchor_session = str(precompute_filter_meta.get("anchor_session_name", "")).strip()
        if (
            bool(precompute_filter_meta.get("fallback_boundary_active"))
            and anchor_race
            and anchor_session
        ):
            precompute_message = (
                f"Showing {ready_count}/{horizon_count} precomputed races from "
                f"{anchor_race} checkpoint {anchor_session}. "
                "A newer checkpoint exists, but it is not warmed yet. The selected race stays "
                "on the latest warmed persisted checkpoint while future-race options remain on "
                "the last warmed horizon."
            )
        elif anchor_race and anchor_session:
            precompute_message = (
                f"Showing {ready_count}/{horizon_count} precomputed races from "
                f"{anchor_race} checkpoint {anchor_session}. "
                "Hidden races will appear after the horizon is warmed."
            )
        else:
            precompute_message = (
                f"Showing {ready_count} precomputed races. "
                "Hidden races will appear after the horizon is warmed."
            )
    elif str(precompute_filter_meta.get("stale_reason", "")).strip() == "artifact_hash_mismatch":
        planned_races = precompute_filter_meta.get("planned_races", [])
        visible_count = len(race_options)
        planned_count = len(planned_races) if isinstance(planned_races, list) else visible_count
        precompute_message = (
            f"Showing the next {visible_count}/{planned_count} scheduled races only. "
            "Warmup exists for an older artifact set, but not for the current one yet."
        )
    elif bool(precompute_filter_meta.get("scope_applied")):
        planned_races = precompute_filter_meta.get("planned_races", [])
        visible_count = len(race_options)
        planned_count = len(planned_races) if isinstance(planned_races, list) else visible_count
        if selected_race_prediction_available:
            precompute_message = (
                f"Showing the next {visible_count}/{planned_count} scheduled races only. "
                "The selected race is ready at the current checkpoint, while future-race "
                "horizon metadata is still catching up."
            )
        else:
            precompute_message = (
                f"Showing the next {visible_count}/{planned_count} scheduled races only. "
                "Persisted horizon metadata is not ready for this boundary yet."
            )
    else:
        precompute_message = (
            "No warmed precompute horizon yet. First run builds checkpoint snapshots for the "
            "current race and nearby weekends."
        )
    render_notice_banner(
        precompute_message,
        tone="success" if bool(precompute_filter_meta.get("applied")) else "info",
        label="Precompute horizon",
        st_module=st,
    )

    prediction_action_state = _prediction_action_state(
        precompute_filter_meta,
        selected_race_prediction_available=selected_race_prediction_available,
    )
    pending_message = prediction_action_state.get("pending_message")
    if isinstance(pending_message, str) and pending_message.strip():
        render_notice_banner(
            pending_message,
            tone="warning",
            label="Warmup pending",
            st_module=st,
        )

    predict_clicked = st.button(
        "Predict sprint weekend" if race_selection.endswith("(Sprint)") else "Predict weekend",
        type="primary",
        width="stretch",
        disabled=bool(prediction_action_state.get("disabled")),
        help=str(prediction_action_state.get("help") or ""),
    )

    if predict_clicked:
        status_placeholder = st.empty()

        with st.spinner("Loading prediction data..."):
            try:

                def update_status(message: str) -> None:
                    status_placeholder.info(f"Loading: {message}")

                pipeline_output = execute_live_prediction_pipeline(
                    race_name=race_name,
                    weather=weather,
                    year=selected_season,
                    force_refresh=False,
                    progress_callback=update_status,
                )
                prediction_results = pipeline_output["prediction_results"]
                is_sprint = bool(pipeline_output["is_sprint"])
                practice_update = pipeline_output["practice_update"]
                boundary_refresh = pipeline_output.get("boundary_refresh", {})
                boundary_fallback = pipeline_output.get("boundary_fallback", {})
                precompute_summary = pipeline_output.get("precompute_summary", {})
                prediction_cache_hit = bool(pipeline_output.get("prediction_cache_hit", False))
                pipeline_timing = pipeline_output.get("pipeline_timing", {})
                observability = pipeline_output.get("observability", {})
                prediction_checkpoint = (
                    str(pipeline_output.get("boundary_session_name") or "").strip().upper()
                )
                status_placeholder.empty()

                runtime_messages: list[tuple[str, str]] = []
                if selected_season == 2026:
                    runtime_messages.append(
                        (
                            "warning",
                            "2026 regulation reset: predictions are uncertain until races complete.",
                        )
                    )
                else:
                    runtime_messages.append(
                        (
                            "info",
                            f"{selected_season} season selected: predictions use currently available "
                            "session data and learned artifacts for this season.",
                        )
                    )

                if is_sprint:
                    runtime_messages.append(
                        (
                            "info",
                            "Sprint weekend mode active: Sprint Qualifying → Sprint Race → "
                            "Main Qualifying → Main Race cascade.",
                        )
                    )
                runtime_messages.append(
                    (
                        "info",
                        _latest_data_status_message(
                            race_name=race_name,
                            year=selected_season,
                            boundary_refresh=boundary_refresh,
                            practice_update=practice_update,
                        ),
                    )
                )
                if prediction_cache_hit:
                    runtime_messages.append(
                        (
                            "info",
                            "Prediction reused from cache (inputs unchanged, no new boundary data).",
                        )
                    )
                if isinstance(boundary_fallback, dict) and boundary_fallback:
                    current_checkpoint = str(
                        boundary_fallback.get("current_boundary_session_name", "")
                    ).strip()
                    warmed_checkpoint = str(
                        boundary_fallback.get("warmed_boundary_session_name", "")
                    ).strip()
                    runtime_messages.append(
                        (
                            "warning",
                            "Latest completed checkpoint "
                            f"{current_checkpoint or 'current'} is not warmed yet. "
                            "Serving the latest available persisted checkpoint "
                            f"{warmed_checkpoint or 'PRE'} instead.",
                        )
                    )
                if practice_update.get("updated"):
                    runtime_messages.append(
                        (
                            "success",
                            "Updated car characteristics from completed practice sessions: "
                            f"{', '.join(practice_update['completed_fp_sessions'])} "
                            f"({practice_update['teams_updated']} teams)",
                        )
                    )
                elif practice_update.get("completed_fp_sessions"):
                    runtime_messages.append(
                        (
                            "info",
                            "Practice characteristics already up to date for sessions: "
                            f"{', '.join(practice_update['completed_fp_sessions'])}",
                        )
                    )

                retried_events = practice_update.get("retried_events", [])
                if retried_events:
                    runtime_messages.append(
                        (
                            "warning",
                            "Practice backlog updates deferred due to active processing lock: "
                            f"{', '.join(str(event) for event in retried_events)}",
                        )
                    )

                if boundary_refresh.get("refresh_needed"):
                    new_sessions = boundary_refresh.get("new_sessions", [])
                    reason = boundary_refresh.get("reason", "session_boundary_delta")
                    if new_sessions:
                        runtime_messages.append(
                            (
                                "info",
                                "A newer checkpoint was detected but is still waiting on warmup "
                                f"({reason}): {', '.join(new_sessions)}",
                            )
                        )
                    else:
                        runtime_messages.append(
                            (
                                "info",
                                f"A newer checkpoint was detected but is still waiting on warmup ({reason}).",
                            )
                        )

                if isinstance(precompute_summary, dict) and precompute_summary.get("triggered"):
                    generated = int(precompute_summary.get("generated", 0))
                    reused = int(precompute_summary.get("reused", 0))
                    targets = precompute_summary.get("targets", [])
                    ready_races = precompute_summary.get("ready_races", [])
                    target_label = ", ".join(str(target) for target in targets) or race_name
                    ready_count = len(ready_races) if isinstance(ready_races, list) else 0
                    runtime_messages.append(
                        (
                            "info",
                            "Boundary precompute completed: "
                            f"{generated} scenario(s) generated, {reused} reused "
                            f"for {target_label}. Ready races: {ready_count}.",
                        )
                    )
                    errors = precompute_summary.get("errors", [])
                    if isinstance(errors, list) and errors:
                        runtime_messages.append(
                            (
                                "warning",
                                "Some precompute scenarios failed: "
                                + "; ".join(str(error) for error in errors[:3]),
                            )
                        )

                _render_collapsible_runtime_messages(runtime_messages)

                if pipeline_timing:
                    timing_parts = [
                        f"boundary check {pipeline_timing.get('boundary_check', 0.0):.1f}s",
                        f"weekend lookup {pipeline_timing.get('weekend_lookup', 0.0):.1f}s",
                        f"practice check {pipeline_timing.get('practice_update_check', 0.0):.1f}s",
                        f"prediction load {pipeline_timing.get('prediction_load', 0.0):.1f}s",
                        f"total {pipeline_timing.get('total', 0.0):.1f}s",
                    ]
                    st.caption("Pipeline timing: " + " | ".join(timing_parts))

                alerts = observability.get("alerts", [])
                for alert in alerts:
                    if not isinstance(alert, dict):
                        continue
                    severity = str(alert.get("severity", "warning")).lower()
                    name = str(alert.get("name", "runtime_alert")).strip()
                    message = str(alert.get("message", "")).strip()
                    if not message:
                        continue
                    formatted = f"[{name}] {message}"
                    if severity == "error":
                        st.error(formatted)
                    else:
                        st.warning(formatted)

                counters = observability.get("counters", {})
                if isinstance(counters, dict):
                    watched = [
                        "fastf1_completion_unknown_total",
                        "fastf1_downgrade_prevented_total",
                        "practice_backlog_retry_total",
                        "fastf1_circuit_trip_total",
                    ]
                    active = [
                        f"{key}={int(counters[key])}"
                        for key in watched
                        if key in counters and int(counters[key]) > 0
                    ]
                    if active:
                        st.caption("Runtime health counters: " + " | ".join(active))

                _save_prediction_if_enabled(
                    enable_logging=enable_logging,
                    prediction_results=prediction_results,
                    is_sprint=is_sprint,
                    race_name=race_name,
                    weather=weather,
                    year=selected_season,
                    checkpoint_session_override=prediction_checkpoint or None,
                )

                _render_prediction_results(
                    prediction_results,
                    is_sprint,
                    prediction_cache_hit=prediction_cache_hit,
                    pipeline_timing=pipeline_timing if isinstance(pipeline_timing, dict) else None,
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                hint = _prediction_failure_hint(e)
                if hint:
                    st.info(hint)


def render_model_insights_page() -> None:
    st.header("Model and Learning Runtime")
    st.markdown(MODEL_INSIGHTS_MARKDOWN)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(QUALIFYING_HYPERPARAMETERS_MARKDOWN)

    with col2:
        st.markdown(RACE_HYPERPARAMETERS_MARKDOWN)


def _render_accuracy_page_controls() -> tuple[int, bool]:
    """Render season and refresh controls for the accuracy page."""
    selected_season = _get_selected_season()
    season_options = _available_seasons()
    if selected_season not in season_options:
        season_options = [selected_season, *season_options]
    season_index = season_options.index(selected_season)

    selected_season = int(
        st.selectbox(
            "Season",
            options=season_options,
            index=season_index,
            key="selected_season",
            help="Choose which season's saved predictions and snapshots to inspect.",
        )
    )
    _set_selected_season(selected_season)

    st.caption(
        "Refresh completed qualifying, sprint, and race results, then rebuild the accuracy "
        "cards and charts from the updated checkpoints."
    )
    refresh_requested = st.button(
        "Refresh Actuals",
        type="primary",
        width="stretch",
        help=(
            "Fetch newly completed qualifying, sprint, and race results for saved "
            "predictions. This can take a bit longer than a normal page load."
        ),
    )

    return selected_season, refresh_requested


def render_prediction_accuracy_page() -> None:
    """Render the target-aware accuracy dashboard."""
    st.header("Prediction Accuracy Tracker")

    from .accuracy import AccuracyPipeline

    selected_season, refresh_requested = _render_accuracy_page_controls()
    pipeline = AccuracyPipeline(year=selected_season)
    if refresh_requested:
        with st.spinner("Refreshing completed weekends..."):
            refreshed_predictions = pipeline.reconcile_actuals()
        snapshots_written = pipeline.snapshots_written
        if refreshed_predictions > 0 or snapshots_written > 0:
            st.success(
                "Refresh complete: "
                f"{refreshed_predictions} saved prediction(s) updated, "
                f"{snapshots_written} accuracy snapshot(s) rebuilt."
            )
            st.caption(
                "Overall Accuracy and all charts below were rebuilt from the refreshed data."
            )
        else:
            st.caption("No new completed weekends or snapshot updates were needed.")

    summary = pipeline.build_summary()

    if not pipeline.all_predictions:
        st.info(
            "No predictions saved yet. Run predictions after practice sessions to start "
            "building checkpoint accuracy history."
        )
        return

    st.success(f"Found {summary.n_predictions} saved prediction(s)")
    if pipeline.actuals_reconciled > 0:
        st.caption(f"Reconciled actuals for {pipeline.actuals_reconciled} saved prediction(s).")

    if pipeline.has_actuals:
        metric_name = st.selectbox(
            "Metric",
            options=list(METRIC_OPTIONS),
            format_func=lambda key: METRIC_OPTIONS.get(key, key),
            index=0,
        )
        show_secondary_sprint_targets = st.toggle(
            "Show sprint-only targets",
            value=False,
        )
        render_overall_accuracy_metrics(summary)
        render_target_sections(summary, metric_name, show_secondary_sprint_targets)

        if summary.n_excluded_targets > 0:
            st.caption(
                f"{summary.n_excluded_targets} target save(s) were excluded because they were not "
                "real forecasts at that checkpoint."
            )
    else:
        st.info(
            "Predictions saved, but no actual results added yet. After each race, "
            "you can update predictions with actual results to calculate accuracy."
        )

    render_saved_predictions_summary(pipeline.prediction_status_rows)


def render_contact_page() -> None:
    st.header("Contact")
    st.markdown(CONTACT_PAGE_HTML, unsafe_allow_html=True)


def render_about_page() -> None:
    """Backwards-compatible alias for older routes."""
    render_contact_page()


def render_page(page: str, enable_logging: bool) -> None:
    """Route the selected dashboard page to its renderer."""
    if page in {"Prediction", "Live Prediction"}:
        render_live_prediction_page(enable_logging)
    elif page in {"Model & Learning", "Model Insights"}:
        render_model_insights_page()
    elif page == "Team Comparison":
        render_team_comparison_page()
    elif page == "Prediction Accuracy" and ENABLE_PREDICTION_ACCURACY_TAB:
        render_prediction_accuracy_page()
    elif page in {"Contact", "About"}:
        render_contact_page()
    else:
        render_live_prediction_page(enable_logging)
