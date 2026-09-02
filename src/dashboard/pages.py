"""Page-level dashboard helpers and route handlers."""

import logging
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.utils.data_paths import resolve_repo_data_path
from src.utils.weekend import get_fallback_schedule_rows, get_schedule_rows, is_sprint_weekend

from . import prediction_horizon as _prediction_horizon
from . import prediction_messages as _prediction_messages
from . import team_comparison
from .analytics import track_event
from .cache import get_artifact_versions
from .driver_substitution_admin import render_driver_substitution_editor
from .grid_penalty_admin import render_grid_penalty_editor
from .layout import BRAND_LAST_UPDATED, BRAND_MODEL_VERSION, ENABLE_PREDICTION_ACCURACY_TAB
from .live_prediction_flow import (
    execute_live_prediction_pipeline_core as _execute_live_prediction_pipeline_core,
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
from .prediction_cascade import (
    render_prediction_results_core as _render_prediction_results_core,
)
from .rendering import display_prediction_result

logger = logging.getLogger(__name__)
DEFAULT_SEASON = 2026
_MIN_SELECTABLE_SEASON = 2024
_FASTF1_CACHE_DIRS = (
    resolve_repo_data_path("data/raw/.fastf1_cache"),
    resolve_repo_data_path("data/raw/.fastf1_cache_testing"),
)
_NON_DISTINCT_RACE_TOKENS = {"grand", "prix", "gp"}

# Backwards-compatible exports for tests and existing imports.
_DEFAULT_TEAM_COLOR = "#B6BABD"


def _fastf1_module() -> Any:
    """Import FastF1 only when schedule data is requested."""
    import fastf1 as fastf1_module

    return fastf1_module


def __getattr__(name: str) -> Any:
    """Backwards-compatible lazy access for tests and callers patching FastF1."""
    if name == "fastf1":
        module = _fastf1_module()
        globals()[name] = module
        return module
    if name == "ArtifactStore":
        from src.persistence.artifact_store import ArtifactStore as artifact_store_class

        return artifact_store_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _artifact_store_class() -> Any:
    patched = globals().get("ArtifactStore")
    if patched is not None:
        return patched
    from src.persistence.artifact_store import ArtifactStore as artifact_store_class

    return artifact_store_class


def render_notice_banner(*args, **kwargs):
    from .rendering import render_notice_banner as _render_notice_banner

    return _render_notice_banner(*args, **kwargs)


def render_page_hero_deck(*args, **kwargs):
    from .rendering import render_page_hero_deck as _render_page_hero_deck

    return _render_page_hero_deck(*args, **kwargs)


def render_prediction_hero_deck(*args, **kwargs):
    from .rendering import render_prediction_hero_deck as _render_prediction_hero_deck

    return _render_prediction_hero_deck(*args, **kwargs)


def _run_prediction(*args, **kwargs):
    from .prediction_flow import run_prediction

    return run_prediction(*args, **kwargs)


def _available_seasons() -> list[int]:
    """Return the season list shown in the dashboard selector."""
    current_year = datetime.now(UTC).year
    latest = max(DEFAULT_SEASON, current_year)
    earliest = min(DEFAULT_SEASON, _MIN_SELECTABLE_SEASON)
    return list(range(latest, earliest - 1, -1))


def _get_selected_season(default: int = DEFAULT_SEASON) -> int:
    """Read the selected season from session state."""
    try:
        raw_value = st.session_state.get("selected_season", default)
    except Exception:
        raw_value = default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _set_selected_season(year: int) -> None:
    """Store the selected season in session state when possible."""
    try:
        st.session_state["selected_season"] = int(year)
    except Exception:
        return


def _forecast_view_is_new(view_key: str) -> bool:
    """Return True the first time a forecast selection is shown this session.

    Guards analytics and the accuracy-save so passive Streamlit reruns (e.g.
    toggling an unrelated control) don't re-fire them. Falls back to True when
    session state is unavailable (headless/tests), where there are no reruns to
    dedupe anyway.
    """
    try:
        if st.session_state.get("_forecast_view_key") == view_key:
            return False
        st.session_state["_forecast_view_key"] = view_key
    except Exception:
        return True
    return True


def render_team_comparison_page() -> None:
    """Render the team-comparison page."""
    selected_season = _get_selected_season()
    render_page_hero_deck(
        title="Team Comparison",
        summary=(
            "Compare synced team fingerprints without blending session profile pace into the "
            "season-prior baseline."
        ),
        eyebrow="Team form",
        cards=[
            {
                "label": "Season",
                "value": str(selected_season),
                "meta": "Uses the shared dashboard season.",
                "tone": "neutral",
            },
            {
                "label": "Source",
                "value": "Latest snapshot",
                "meta": "Falls back to the season file when needed.",
                "tone": "accent",
            },
            {
                "label": "View",
                "value": "Radar + trend",
                "meta": "Profile shape and development history.",
                "tone": "neutral",
            },
            {
                "label": "Scale",
                "value": "10-100",
                "meta": "Higher display score is stronger.",
                "tone": "neutral",
            },
        ],
        st_module=st,
    )
    team_comparison._render_team_comparison_section(year=selected_season)


def _clear_fastf1_race_cache(year: int, race_name: str) -> None:
    """Remove cached FastF1 files for one race weekend."""
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
                    logger.warning("Could not clear FastF1 cache path %s: %s", event_cache_dir, exc)
        except Exception as exc:
            logger.warning("Could not inspect FastF1 cache at %s: %s", year_dir, exc)
            continue

        if removed_paths:
            removed_labels = ", ".join(path.name for path in removed_paths[:5])
            if len(removed_paths) > 5:
                removed_labels += ", ..."
            logger.info(
                "Cleared FastF1 cache for %s %s: %s path(s) in %s (%s)",
                race_name,
                year,
                len(removed_paths),
                cache_dir,
                removed_labels,
            )
        else:
            logger.info("No FastF1 cache paths matched %s %s in %s", race_name, year, cache_dir)


def _normalize_cache_fragment(value: str) -> str:
    """Normalize cache path fragments for case-insensitive race-name matching."""
    normalized = unicodedata.normalize("NFKD", str(value))
    folded = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch.lower() for ch in folded if ch.isalnum())


def _cache_dir_matches_race(cache_dir_name: str, race_name: str) -> bool:
    """Return ``True`` when a FastF1 cache directory belongs to the race."""
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
    """Load race options with a cached schedule lookup."""

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
        schedule = _fastf1_module().get_event_schedule(year)
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

    fallback_rows = list(get_schedule_rows(year))
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
            "Australian Grand Prix",
            "Chinese Grand Prix",
            "Japanese Grand Prix",
            "Miami Grand Prix",
            "Canadian Grand Prix",
            "Monaco Grand Prix",
        ],
        fastf1_error,
    )


def _load_race_options(year: int = DEFAULT_SEASON) -> list[str]:
    """Load race options from FastF1 schedule with sprint labels."""
    race_options, error = _load_race_options_cached(year)
    if error:
        st.warning(f"Failed to load {year} calendar: {error}. Using minimal fallback list.")
    return race_options


def _load_local_race_options(year: int = DEFAULT_SEASON) -> list[str]:
    """Load race options from local fallback data without touching FastF1."""
    options: list[str] = []
    for race_name, event_format in get_fallback_schedule_rows(year):
        if not race_name or "testing" in race_name.lower():
            continue
        if "sprint" in str(event_format).lower():
            options.append(f"{race_name} (Sprint)")
        else:
            options.append(race_name)
    return options


@st.cache_data(ttl=3600, show_spinner=False)
def _load_race_round_meta(year: int) -> dict[str, tuple[int, str]]:
    """Map each race (base name) to ``(round_number, iso_event_date)`` from the schedule.

    Sourced from FastF1 so the dropdown can be ordered by calendar round and default
    to the next upcoming Grand Prix. Keyed by the plain event name (no "(Sprint)"
    suffix) so sprint-labelling differences never break the lookup. Empty when the
    schedule can't be loaded (offline) — the dropdown then keeps its raw order and
    shows no numbers.
    """
    meta: dict[str, tuple[int, str]] = {}
    try:
        schedule = _fastf1_module().get_event_schedule(year)
    except Exception as exc:
        logger.info("Race round metadata unavailable for %s: %s", year, exc)
        return meta
    if not {"EventName", "RoundNumber"}.issubset(schedule.columns):
        return meta
    for _, event in schedule.iterrows():
        name = str(event.get("EventName", "")).strip()
        if not name or "testing" in name.lower():
            continue
        try:
            round_number = int(event.get("RoundNumber", 0) or 0)
        except (TypeError, ValueError):
            continue
        if round_number <= 0:  # round 0 is pre-season testing
            continue
        meta[name] = (round_number, str(event.get("EventDate", "") or "")[:10])
    return meta


def _order_races_by_round(
    race_options: list[str],
    round_meta: dict[str, tuple[int, str]],
    *,
    today_iso: str,
) -> tuple[list[str], int]:
    """Sort options by calendar round and pick the next upcoming race as the default.

    Returns the reordered options and the index of the race to open on: the first
    Grand Prix whose event date is today or later. Falls back to the most recent
    race when the season is over, or index 0 when dates are unknown (offline).
    """

    def _base(label: str) -> str:
        return label.replace(" (Sprint)", "")

    ordered = sorted(race_options, key=lambda label: round_meta.get(_base(label), (10_000, ""))[0])

    upcoming = [
        i
        for i, label in enumerate(ordered)
        if round_meta.get(_base(label), (0, ""))[1] >= today_iso
        and round_meta.get(_base(label), (0, ""))[1]
    ]
    if upcoming:
        return ordered, upcoming[0]
    if any(round_meta.get(_base(label), (0, ""))[1] for label in ordered):
        return ordered, len(ordered) - 1  # season over — show the most recent race
    return ordered, 0


def _latest_dashboard_refresh_timestamp(year: int) -> datetime | None:
    """Return the newest refresh stamp that affects dashboard predictions."""
    season_year = int(year)
    return _prediction_horizon.latest_dashboard_refresh_timestamp(
        year=year,
        get_artifact_versions_fn=get_artifact_versions,
        compute_artifact_hash_fn=compute_artifact_hash,
        load_precompute_horizon_index_fn=load_precompute_horizon_index,
        refresh_paths=(
            resolve_repo_data_path(
                f"data/processed/car_characteristics/{season_year}_car_characteristics.json"
            ),
            resolve_repo_data_path(
                f"data/processed/track_characteristics/{season_year}_track_characteristics.json"
            ),
            resolve_repo_data_path(
                f"data/processed/driver_characteristics/{season_year}_driver_characteristics.json"
            ),
            resolve_repo_data_path("data/processed/driver_characteristics.json"),
            resolve_repo_data_path("data/systems/practice_characteristics_state.json"),
            resolve_repo_data_path("data/systems/precompute_horizon_index.json"),
            resolve_repo_data_path("data/systems/precomputed_predictions.json"),
        ),
        logger=logger,
    )


def _dashboard_refresh_label(year: int) -> str:
    """Format the latest dashboard refresh stamp for the hero card."""
    return _prediction_horizon.dashboard_refresh_label(
        year,
        latest_dashboard_refresh_timestamp_fn=_latest_dashboard_refresh_timestamp,
        fallback_label=BRAND_LAST_UPDATED,
    )


def _load_completed_races_count(year: int) -> int | None:
    """Read the completed Grand Prix race count from the active car artifact."""
    season_year = int(year)
    try:
        payload = _artifact_store_class()(data_root="data").load_artifact(
            "car_characteristics",
            f"{season_year}::car_characteristics",
        )
    except Exception as exc:
        logger.warning("Could not read %s completed-race count: %s", season_year, exc)
        return None

    if not isinstance(payload, dict):
        return None

    completed_count = _prediction_messages.coerce_completed_races_count(
        payload.get("races_completed")
    )
    if completed_count is not None:
        return completed_count

    team_payloads = payload.get("teams", {})
    if not isinstance(team_payloads, dict):
        return None

    team_counts: list[int] = []
    for team_payload in team_payloads.values():
        if not isinstance(team_payload, dict):
            continue
        count = _prediction_messages.coerce_completed_races_count(
            team_payload.get("races_completed")
        )
        if count is not None:
            team_counts.append(count)
    if not team_counts:
        return None
    return min(team_counts)


@st.cache_data(ttl=900, show_spinner=False)
def _load_schedule_event_rows_cached(year: int) -> tuple[tuple[str, str, str], ...]:
    """Load cached schedule rows with serialized event dates for horizon filtering."""
    return _prediction_horizon.load_schedule_event_rows(
        year,
        get_event_schedule_fn=_fastf1_module().get_event_schedule,
        fallback_schedule_rows_fn=get_schedule_rows,
        logger=logger,
    )


def _resolve_dashboard_race_horizon(year: int, horizon_races: int) -> list[str]:
    """Return the next configured race window from the live schedule when dates are available."""
    return _prediction_horizon.resolve_dashboard_race_horizon(
        schedule_rows=_load_schedule_event_rows_cached(year),
        horizon_races=horizon_races,
    )


@st.cache_data(show_spinner=False, ttl=120)
def _current_anchor_boundary_signature(year: int, anchor_race_name: str) -> str | None:
    """Return the live boundary signature for the anchor race, if available."""
    from src.utils.session_detector import SessionDetector

    from .update_flow import boundary_signature, build_event_boundary_snapshot

    return _prediction_horizon.current_anchor_boundary_signature(
        year,
        anchor_race_name,
        is_sprint_weekend_fn=is_sprint_weekend,
        build_event_boundary_snapshot_fn=build_event_boundary_snapshot,
        boundary_signature_fn=boundary_signature,
        session_detector_factory=SessionDetector,
        logger=logger,
    )


def _load_ready_races_from_current_store(
    *,
    year: int,
    race_names: list[str],
    artifact_hash: str,
    weather_scenarios: list[str],
) -> list[str]:
    """Return races that already have full persisted weather coverage for the current boundary."""
    return _prediction_horizon.load_ready_races_from_current_store(
        year=year,
        race_names=race_names,
        artifact_hash=artifact_hash,
        weather_scenarios=weather_scenarios,
        current_anchor_boundary_signature_fn=_current_anchor_boundary_signature,
        load_precomputed_prediction_fn=load_precomputed_prediction,
    )


def _filter_race_options_to_precomputed_horizon(
    *,
    year: int,
    race_options: list[str],
    validate_live_boundary: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Filter race options to the warmed horizon for the active artifact state."""
    return _prediction_horizon.filter_race_options_to_precomputed_horizon(
        year=year,
        race_options=race_options,
        get_prediction_precompute_config_fn=get_prediction_precompute_config,
        resolve_dashboard_race_horizon_fn=(
            _resolve_dashboard_race_horizon
            if validate_live_boundary
            else lambda _year, _horizon: []
        ),
        get_artifact_versions_fn=get_artifact_versions,
        compute_artifact_hash_fn=compute_artifact_hash,
        load_precompute_horizon_index_fn=load_precompute_horizon_index,
        has_precompute_horizon_for_year_fn=has_precompute_horizon_for_year,
        load_ready_races_from_current_store_fn=_load_ready_races_from_current_store,
        current_anchor_boundary_signature_fn=_current_anchor_boundary_signature,
        logger=logger,
        validate_live_boundary=validate_live_boundary,
    )


def _filter_race_options_for_initial_render(
    *,
    year: int,
    race_options: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Fast initial-render wrapper that tolerates legacy monkeypatch signatures."""
    try:
        return _filter_race_options_to_precomputed_horizon(
            year=year,
            race_options=race_options,
            validate_live_boundary=False,
        )
    except TypeError as exc:
        if "validate_live_boundary" not in str(exc):
            raise
        return _filter_race_options_to_precomputed_horizon(
            year=year,
            race_options=race_options,
        )


def _prediction_action_state(
    precompute_filter_meta: dict[str, Any],
    *,
    selected_race_prediction_available: bool = False,
) -> dict[str, Any]:
    """Resolve whether a persisted dashboard prediction is available."""
    return _prediction_horizon.prediction_action_state(
        precompute_filter_meta,
        selected_race_prediction_available=selected_race_prediction_available,
    )


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
    return _prediction_messages.latest_data_status_message(
        race_name=race_name,
        year=year,
        boundary_refresh=boundary_refresh,
        practice_update=practice_update,
    )


def _render_collapsible_runtime_messages(messages: list[tuple[str, str]]) -> None:
    """Render runtime notices compactly to avoid stacked info/warning banners."""
    _prediction_messages.render_collapsible_runtime_messages(
        messages,
        render_notice_banner_fn=render_notice_banner,
        st_module=st,
    )


def execute_live_prediction_pipeline(
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
    force_refresh: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Load the warmed persisted prediction selected in the dashboard.

    Kept separate from Streamlit rendering so tests can assert request-path
    behavior without implying that a user click regenerates artifacts.

    Args:
        race_name: The name of the race
        weather: Weather forecast for the race
        year: Season year
        force_refresh: Legacy compatibility flag. Manual request-path refresh is disabled.
        progress_callback: Optional callback for progress updates
    """
    from .update_flow import (
        auto_update_if_needed,
        auto_update_practice_characteristics_if_needed,
        detect_event_boundary_refresh_if_needed,
    )

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
        run_prediction_fn=_run_prediction,
    )


def _build_runtime_messages(
    *,
    selected_season: int,
    race_name: str,
    is_sprint: bool,
    boundary_refresh: dict[str, Any],
    practice_update: dict[str, Any],
    prediction_cache_hit: bool,
    boundary_fallback: dict[str, Any],
    precompute_summary: dict[str, Any],
    completed_races_count: int | None = None,
) -> list[tuple[str, str]]:
    """Build the runtime notices displayed after prediction loading."""
    return _prediction_messages.build_runtime_messages(
        selected_season=selected_season,
        race_name=race_name,
        is_sprint=is_sprint,
        boundary_refresh=boundary_refresh,
        practice_update=practice_update,
        prediction_cache_hit=prediction_cache_hit,
        boundary_fallback=boundary_fallback,
        precompute_summary=precompute_summary,
        completed_races_count=completed_races_count,
        latest_data_status_message_fn=_latest_data_status_message,
    )


def _render_forecast_pending_state(
    *,
    race_name: str,
    selected_season: int,
    pending_message: str | None,
    error: Exception | None = None,
) -> None:
    """Show a calm 'forecast is being prepared' state instead of a raw error.

    Reached when no warmed prediction is available yet for the selected race
    (warmup still running, or a schedule lookup failed). A fan sees plain
    reassurance and a next step, not a stack trace.
    """
    if isinstance(pending_message, str) and pending_message.strip():
        lead = pending_message.strip().rstrip(".") + "."
    else:
        lead = f"The {race_name} {selected_season} forecast isn't published yet."
    body = (
        f"{lead} Predictions appear here as each session's data comes in, starting with first "
        "practice. To see a full forecast right now, pick a Grand Prix that's already run from "
        "the list above."
    )
    render_notice_banner(body, tone="info", label="Forecast updating", st_module=st)
    if error is not None:
        logger.info("Forecast unavailable for %s %s: %s", race_name, selected_season, error)


def render_live_prediction_page(enable_logging: bool) -> None:
    """Render the main live prediction page with qualifying and race tabs."""
    selected_season = _get_selected_season()
    render_prediction_hero_deck(
        title="Race Weekend Prediction",
        summary=(
            "Qualifying and race-day forecasts for the weekend, updated as each session's data "
            "comes in. The latest numbers load the moment you arrive."
        ),
        eyebrow="Weekend forecast",
        cards=[
            {
                "label": "Model",
                "value": BRAND_MODEL_VERSION,
                "meta": "The forecasting model these predictions run on.",
                "tone": "accent",
            },
            {
                "label": "Updated",
                "value": _dashboard_refresh_label(selected_season),
                "meta": "When these forecasts were last refreshed.",
                "tone": "neutral",
            },
            {
                "label": "On record",
                "value": "Saved" if enable_logging else "Off",
                "meta": "Every forecast is saved, so it can be scored against the real result.",
                "tone": "success" if enable_logging else "warning",
            },
            {
                "label": "Forecasts",
                "value": "Ready",
                "meta": "Prepared ahead of each session, so they load instantly.",
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
                help="Choose which season to forecast.",
            )
        )
    _set_selected_season(selected_season)

    race_options = _load_local_race_options(selected_season) or _load_race_options(selected_season)
    race_options, precompute_filter_meta = _filter_race_options_for_initial_render(
        year=selected_season,
        race_options=race_options,
    )

    # The raw option order (from the local track file) is not the calendar order, so
    # order by round and open on the next upcoming Grand Prix — not whatever happened
    # to be first. Numbers come from the schedule; the option value stays the clean
    # race name so downstream matching is untouched.
    round_meta = _load_race_round_meta(selected_season)
    race_options, default_race_index = _order_races_by_round(
        race_options,
        round_meta,
        today_iso=datetime.now(UTC).date().isoformat(),
    )

    def _format_race_option(label: str) -> str:
        entry = round_meta.get(label.replace(" (Sprint)", ""))
        return f"{entry[0]}. {label}" if entry else label

    with control_col2:
        race_selection = st.selectbox(
            "Grand Prix",
            race_options,
            index=default_race_index,
            format_func=_format_race_option,
            help="Pick a race weekend to see its forecast.",
        )
        race_name = race_selection.replace(" (Sprint)", "")

    with control_col3:
        weather = st.selectbox(
            "Weather",
            ["dry", "rain", "mixed"],
            help="Assumed race-day conditions. Switch it to see how wet or mixed weather changes the forecast.",
        )
    selected_race_prediction_available = False

    # Operator-only, token-gated: renders nothing for an ordinary visitor.
    render_grid_penalty_editor(race_name=race_name, year=selected_season)
    render_driver_substitution_editor(race_name=race_name, year=selected_season)

    # Horizon-coverage detail (which upcoming races are warmed yet) is operator
    # plumbing, not a fan's answer — it no longer gets its own banner above the
    # forecast. When a race has no forecast, the single pending state below says
    # everything the fan needs, without doubling up. (impeccable: distill)
    prediction_action_state = _prediction_action_state(
        precompute_filter_meta,
        selected_race_prediction_available=selected_race_prediction_available,
    )
    pending_message = prediction_action_state.get("pending_message")

    # Answer first: the forecast is served from warmed, persisted artifacts and
    # the dashboard request path is read-only (no recompute, no writes), so show
    # it immediately on load instead of gating it behind a click. Changing the
    # Grand Prix or weather selector above reruns this and refreshes the forecast
    # in place. (impeccable: onboard — auto-show next race)
    status_placeholder = st.empty()

    def update_status(message: str) -> None:
        status_placeholder.info(f"Loading: {message}")

    with st.spinner("Loading the latest forecast…"):
        try:
            pipeline_output = execute_live_prediction_pipeline(
                race_name=race_name,
                weather=weather,
                year=selected_season,
                force_refresh=False,
                progress_callback=update_status,
            )
        except Exception as e:  # noqa: BLE001 - surfaced as a calm pending state below
            status_placeholder.empty()
            _render_forecast_pending_state(
                race_name=race_name,
                selected_season=selected_season,
                pending_message=pending_message,
                error=e,
            )
            return

    status_placeholder.empty()

    prediction_results = pipeline_output["prediction_results"]
    is_sprint = bool(pipeline_output["is_sprint"])
    practice_update = pipeline_output["practice_update"]
    boundary_refresh = pipeline_output.get("boundary_refresh", {})
    boundary_fallback = pipeline_output.get("boundary_fallback", {})
    precompute_summary = pipeline_output.get("precompute_summary", {})
    prediction_cache_hit = bool(pipeline_output.get("prediction_cache_hit", False))
    pipeline_timing = pipeline_output.get("pipeline_timing", {})
    observability = pipeline_output.get("observability", {})
    prediction_checkpoint = str(pipeline_output.get("boundary_session_name") or "").strip().upper()

    runtime_messages = _build_runtime_messages(
        selected_season=selected_season,
        race_name=race_name,
        is_sprint=is_sprint,
        boundary_refresh=boundary_refresh,
        practice_update=practice_update,
        prediction_cache_hit=prediction_cache_hit,
        boundary_fallback=boundary_fallback,
        precompute_summary=precompute_summary,
        completed_races_count=_load_completed_races_count(selected_season),
    )
    # Consolidate operator diagnostics (runtime alerts, health counters, timing)
    # into the single collapsible disclosure instead of stacking separate captions
    # and banners above the forecast. A fan sees one "Forecast details" line with
    # everything else one click deep. (impeccable: distill)
    for severity, formatted in _prediction_messages.iter_observability_alerts(observability):
        runtime_messages.append(("warning" if severity == "error" else severity, formatted))

    counters_caption = _prediction_messages.runtime_health_counters_caption(observability)
    if counters_caption:
        runtime_messages.append(("info", counters_caption))

    timing_caption = _prediction_messages.pipeline_timing_caption(
        pipeline_timing if isinstance(pipeline_timing, dict) else None
    )
    if timing_caption:
        runtime_messages.append(("info", timing_caption))

    _render_collapsible_runtime_messages(runtime_messages)

    # Capture the served forecast for accuracy tracking and log the view once per
    # selection per browser session, so passive reruns don't re-fire writes,
    # analytics, or the "already saved" notice on every interaction.
    view_key = f"{selected_season}|{race_name}|{weather}|{prediction_checkpoint}"
    if _forecast_view_is_new(view_key):
        track_event(
            "forecast_viewed",
            race=race_name,
            is_sprint=is_sprint,
            weather=str(weather),
            season=int(selected_season),
        )
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


def render_model_insights_page() -> None:
    """Render the model insights page with hyperparameters and learning state."""
    render_page_hero_deck(
        title="Model and Learning Runtime",
        summary=(
            "Inspect the active forecast path, calibration loop, and guardrails used before "
            "experimental components are promoted."
        ),
        eyebrow="Model notes",
        cards=[
            {
                "label": "Release",
                "value": BRAND_MODEL_VERSION,
                "meta": "Active dashboard model label.",
                "tone": "accent",
            },
            {
                "label": "Learning",
                "value": "Gated",
                "meta": "Saved actuals update calibration state.",
                "tone": "success",
            },
            {
                "label": "Reset year",
                "value": "2026",
                "meta": "Current-season evidence ramps in quickly.",
                "tone": "warning",
            },
            {
                "label": "Outputs",
                "value": "Q + Race",
                "meta": "One predictor serves both sections.",
                "tone": "neutral",
            },
        ],
        st_module=st,
    )
    st.markdown(MODEL_INSIGHTS_MARKDOWN)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(QUALIFYING_HYPERPARAMETERS_MARKDOWN)

    with col2:
        st.markdown(RACE_HYPERPARAMETERS_MARKDOWN)


def render_model_diagnostics_page() -> None:
    """Render persisted model diagnostics."""
    from .model_diagnostics import render_model_diagnostics

    selected_season = _get_selected_season()
    render_page_hero_deck(
        title="Model Diagnostics",
        summary=(
            "Read the persisted replay, leakage, and regulation-reset checks used before "
            "schema migration."
        ),
        eyebrow="Model audit",
        cards=[
            {
                "label": "Season",
                "value": str(selected_season),
                "meta": "Uses the shared dashboard season.",
                "tone": "neutral",
            },
            {
                "label": "Source",
                "value": "Persisted",
                "meta": "No ad hoc dashboard recompute.",
                "tone": "accent",
            },
            {
                "label": "Dry leakage",
                "value": "Measured",
                "meta": "Legacy proxy until schema migration.",
                "tone": "warning",
            },
            {
                "label": "Wet invariant",
                "value": "Guarded",
                "meta": "Depends on weather-routed replay rows.",
                "tone": "neutral",
            },
        ],
        st_module=st,
    )
    render_model_diagnostics(year=selected_season, st_module=st)


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
        "Scheduled workers refresh completed qualifying, sprint, and race results automatically. "
        "Use the repair action only when stored artifacts need a forced rebuild."
    )
    refresh_requested = st.button(
        "Repair Accuracy Data",
        type="secondary",
        width="stretch",
        help=("Force a one-off actuals reconciliation and snapshot rebuild for saved predictions."),
    )

    return selected_season, refresh_requested


def render_prediction_accuracy_page() -> None:
    """Render the target-aware accuracy dashboard."""
    from .accuracy_view import (
        METRIC_OPTIONS,
        render_overall_accuracy_metrics,
        render_saved_predictions_summary,
        render_target_sections,
    )

    selected_season = _get_selected_season()
    render_page_hero_deck(
        title="Prediction Accuracy Tracker",
        summary=(
            "Review saved checkpoint forecasts against completed qualifying, sprint, and race "
            "results."
        ),
        eyebrow="Forecast audit",
        cards=[
            {
                "label": "Season",
                "value": str(selected_season),
                "meta": "Can be changed in the controls below.",
                "tone": "neutral",
            },
            {
                "label": "Refresh",
                "value": "Automatic",
                "meta": "Workers reconcile completed races after warmup.",
                "tone": "accent",
            },
            {
                "label": "Targets",
                "value": "Checkpointed",
                "meta": "Primary and sprint-only targets are tracked.",
                "tone": "neutral",
            },
            {
                "label": "Scope",
                "value": "Saved runs",
                "meta": "Retrospective-only records stay excluded.",
                "tone": "success",
            },
        ],
        st_module=st,
    )

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


def render_checkpoint_viewer_page() -> None:
    """Render a dedicated browser for saved checkpoint artifacts."""
    from .accuracy_view import render_saved_prediction_viewer

    selected_season = _get_selected_season()
    render_page_hero_deck(
        title="Checkpoint Viewer",
        summary=(
            "Browse saved race-weekend artifacts directly, separate from accuracy charts and "
            "summary metrics."
        ),
        eyebrow="Artifact browser",
        cards=[
            {
                "label": "Season",
                "value": str(selected_season),
                "meta": "Can be changed in the controls below.",
                "tone": "neutral",
            },
            {
                "label": "Granularity",
                "value": "Checkpoint",
                "meta": "Open each stored session boundary.",
                "tone": "accent",
            },
            {
                "label": "Mode",
                "value": "Read-only",
                "meta": "No prediction refresh from this view.",
                "tone": "neutral",
            },
            {
                "label": "Source",
                "value": "Artifacts",
                "meta": "Uses persisted prediction records.",
                "tone": "success",
            },
        ],
        st_module=st,
    )

    from .accuracy import AccuracyPipeline

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
            help="Choose which season's saved checkpoints to browse.",
        )
    )
    _set_selected_season(selected_season)

    st.caption("Browse saved race weekends and open each checkpoint without the accuracy charts.")

    pipeline = AccuracyPipeline(year=selected_season)
    if not pipeline.all_predictions:
        st.info("No saved checkpoints yet.")
        return

    st.success(f"Found {len(pipeline.all_predictions)} saved checkpoint artifact(s)")
    render_saved_prediction_viewer(pipeline.all_predictions, season_year=selected_season)


def render_contact_page() -> None:
    """Render the contact page."""
    render_page_hero_deck(
        title="Contact",
        summary=("Project links, scope notes, and the independence disclaimer for Trackside Labs."),
        eyebrow="About the project",
        cards=[
            {
                "label": "Project",
                "value": "GitHub",
                "meta": "Repository link is listed below.",
                "tone": "accent",
            },
            {
                "label": "Scope",
                "value": "Forecasting",
                "meta": "Prediction workflow and accuracy tracking.",
                "tone": "neutral",
            },
            {
                "label": "Status",
                "value": "Independent",
                "meta": "No team or series affiliation.",
                "tone": "neutral",
            },
            {
                "label": "Contact",
                "value": "LinkedIn",
                "meta": "Profile link is listed below.",
                "tone": "success",
            },
        ],
        st_module=st,
    )
    st.markdown(CONTACT_PAGE_HTML, unsafe_allow_html=True)


def render_page(page: str, enable_logging: bool) -> None:
    """Route the selected dashboard page to its renderer."""
    if page in {"Prediction", "Live Prediction"}:
        render_live_prediction_page(enable_logging)
    elif page in {"Model & Learning", "Model Insights"}:
        render_model_insights_page()
    elif page == "Model Diagnostics":
        render_model_diagnostics_page()
    elif page == "Team Comparison":
        render_team_comparison_page()
    elif page == "Prediction Accuracy" and ENABLE_PREDICTION_ACCURACY_TAB:
        render_prediction_accuracy_page()
    elif page == "Checkpoint Viewer" and ENABLE_PREDICTION_ACCURACY_TAB:
        render_checkpoint_viewer_page()
    elif page in {"Contact", "About"}:
        render_contact_page()
    else:
        render_live_prediction_page(enable_logging)
