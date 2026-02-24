"""Dashboard pages and page-level orchestration."""

import json
import logging
import time
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
import streamlit as st

from src.utils import config_loader
from src.utils.team_mapping import canonicalize_team
from src.utils.weekend import is_sprint_weekend

from .cache import get_artifact_versions
from .prediction_flow import run_prediction
from .rendering import display_prediction_result
from .update_flow import auto_update_if_needed, auto_update_practice_characteristics_if_needed

logger = logging.getLogger(__name__)
DEFAULT_SEASON = 2026

_TEAM_RADAR_METRICS: tuple[tuple[str, str], ...] = (
    ("slow_corner_performance", "Slow Corners"),
    ("medium_corner_performance", "Medium Corners"),
    ("fast_corner_performance", "Fast Corners"),
    ("braking_performance", "Braking"),
    ("top_speed", "Top Speed"),
    ("tire_deg_performance", "Tire Deg"),
)
_TEAM_BRAND_COLORS: dict[str, str] = {
    "RED BULL": "#3671C6",
    "MCLAREN": "#FF8700",
    "FERRARI": "#DC0000",
    "MERCEDES": "#00D2BE",
    "ASTON MARTIN": "#006F62",
    "ALPINE": "#2293D1",
    "HAAS": "#B6BABD",
    "RB": "#6692FF",
    "WILLIAMS": "#005AFF",
    "AUDI": "#C4122E",
    "CADILLAC": "#2A4AA0",
}
_DEFAULT_TEAM_COLOR = "#7C8798"
_DEFAULT_BIG4_CANONICAL: tuple[str, ...] = ("MCLAREN", "MERCEDES", "FERRARI", "RED BULL")


def _coerce_unit_metric(value: Any) -> float | None:
    """Normalize metric values into [0.0, 1.0] when possible."""
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not isfinite(value_float):
        return None
    return max(0.0, min(1.0, value_float))


def _team_brand_color(team_name: str) -> str:
    """Resolve canonical team color; use neutral fallback for unknown names."""
    canonical_id = canonicalize_team(team_name)
    if isinstance(canonical_id, str):
        color = _TEAM_BRAND_COLORS.get(canonical_id)
        if color:
            return color
    return _DEFAULT_TEAM_COLOR


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba string with alpha."""
    cleaned = str(hex_color).strip().lstrip("#")
    if len(cleaned) != 6:
        return f"rgba(124, 135, 152, {alpha})"
    try:
        red = int(cleaned[0:2], 16)
        green = int(cleaned[2:4], 16)
        blue = int(cleaned[4:6], 16)
    except ValueError:
        return f"rgba(124, 135, 152, {alpha})"
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _default_team_selection(team_names: list[str], max_teams: int = 4) -> list[str]:
    """Prefer Big 4 team defaults, then backfill using current ordering."""
    if max_teams <= 0:
        return []

    canonical_to_team: dict[str, str] = {}
    for team_name in team_names:
        canonical_id = canonicalize_team(team_name)
        if isinstance(canonical_id, str) and canonical_id not in canonical_to_team:
            canonical_to_team[canonical_id] = team_name

    selected: list[str] = []
    for canonical_id in _DEFAULT_BIG4_CANONICAL:
        team_name = canonical_to_team.get(canonical_id)
        if team_name and team_name not in selected:
            selected.append(team_name)
        if len(selected) >= max_teams:
            return selected

    for team_name in team_names:
        if team_name not in selected:
            selected.append(team_name)
        if len(selected) >= max_teams:
            break

    return selected


def _collect_profile_names(teams_payload: dict[str, Any]) -> list[str]:
    """Collect available testing profile names from team characteristics payload."""
    profile_names: set[str] = {"balanced"}
    for team_data in teams_payload.values():
        if not isinstance(team_data, dict):
            continue
        profiles = team_data.get("testing_characteristics_profiles")
        if isinstance(profiles, dict):
            profile_names.update(str(name) for name in profiles)
        testing_characteristics = team_data.get("testing_characteristics")
        if isinstance(testing_characteristics, dict):
            run_profile = testing_characteristics.get("run_profile")
            if isinstance(run_profile, str) and run_profile.strip():
                profile_names.add(run_profile.strip())

    ordered = ["balanced", "short_run", "long_run"]
    remaining = sorted(profile for profile in profile_names if profile not in ordered)
    return [profile for profile in ordered if profile in profile_names] + remaining


def _resolve_profile_metrics(team_data: dict[str, Any], profile: str) -> dict[str, Any]:
    """Resolve testing metrics for profile, falling back conservatively to balanced payload."""
    profiles = team_data.get("testing_characteristics_profiles")
    if isinstance(profiles, dict):
        profile_payload = profiles.get(profile)
        if isinstance(profile_payload, dict):
            return profile_payload

    testing_payload = team_data.get("testing_characteristics")
    if isinstance(testing_payload, dict):
        if profile == "balanced":
            return testing_payload
        run_profile = testing_payload.get("run_profile")
        if isinstance(run_profile, str) and run_profile == profile:
            return testing_payload

    return {}


def _build_team_comparison_dataframe(
    teams_payload: dict[str, Any],
    selected_teams: list[str],
    profile: str,
) -> tuple[pd.DataFrame, int]:
    """Build team comparison frame and return count of neutral fallbacks used."""
    rows: list[dict[str, Any]] = []
    neutral_fallback_count = 0

    for team_name in selected_teams:
        team_data = teams_payload.get(team_name)
        if not isinstance(team_data, dict):
            continue

        metrics_payload = _resolve_profile_metrics(team_data, profile)
        row: dict[str, Any] = {"Team": team_name}

        overall_pace = _coerce_unit_metric(metrics_payload.get("overall_pace"))
        row["Overall Pace"] = overall_pace if overall_pace is not None else 0.5
        if overall_pace is None:
            neutral_fallback_count += 1

        overall_perf = _coerce_unit_metric(team_data.get("overall_performance"))
        row["Overall Performance"] = overall_perf if overall_perf is not None else 0.5
        if overall_perf is None:
            neutral_fallback_count += 1

        for payload_key, label in _TEAM_RADAR_METRICS:
            metric_value = _coerce_unit_metric(metrics_payload.get(payload_key))
            row[label] = metric_value if metric_value is not None else 0.5
            if metric_value is None:
                neutral_fallback_count += 1

        radar_values = [float(row[label]) for _, label in _TEAM_RADAR_METRICS]
        radar_composite = float(sum(radar_values) / len(radar_values))
        row["Radar Composite"] = radar_composite
        row["Prior Minus Radar"] = float(row["Overall Performance"]) - radar_composite

        rows.append(row)

    if not rows:
        return pd.DataFrame(), neutral_fallback_count

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("Overall Pace", ascending=False).reset_index(drop=True)
    return frame, neutral_fallback_count


@st.cache_data(ttl=300, show_spinner=False)
def _load_team_characteristics_payload(year: int) -> tuple[dict[str, Any] | None, Path]:
    """Load season car characteristics payload used for team-comparison visualizations."""
    processed_path = Path(config_loader.get("paths.processed", "data/processed"))
    characteristics_path = (
        processed_path / "car_characteristics" / f"{year}_car_characteristics.json"
    )
    if not characteristics_path.exists():
        return None, characteristics_path

    try:
        with open(characteristics_path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, characteristics_path

    if not isinstance(payload, dict):
        return None, characteristics_path
    return payload, characteristics_path


def _render_team_comparison_section(year: int = DEFAULT_SEASON) -> None:
    """Render profile-aware team comparison chart and metric table."""
    st.subheader("Testing/Practice Snapshot")

    payload, characteristics_path = _load_team_characteristics_payload(year)
    if not payload:
        st.info(f"Team characteristics unavailable at `{characteristics_path}`.")
        return

    teams_payload = payload.get("teams")
    if not isinstance(teams_payload, dict) or not teams_payload:
        st.info("No team characteristics found for comparison.")
        return

    profile_names = _collect_profile_names(teams_payload)
    profile = st.selectbox(
        "Comparison profile",
        options=profile_names,
        index=0,
        help="Balanced uses mixed-session behavior. Short/long run focus specific session intent.",
    )

    def _profile_sort_key(team: str) -> float:
        team_data = teams_payload.get(team)
        if not isinstance(team_data, dict):
            return 0.0
        metrics_payload = _resolve_profile_metrics(team_data, profile)
        profile_pace = _coerce_unit_metric(metrics_payload.get("overall_pace"))
        if profile_pace is not None:
            return profile_pace
        baseline = _coerce_unit_metric(team_data.get("overall_performance"))
        return baseline if baseline is not None else 0.0

    sorted_team_names = sorted(teams_payload.keys(), key=_profile_sort_key, reverse=True)
    default_selection = _default_team_selection(sorted_team_names, max_teams=4)
    selected_teams = st.multiselect(
        "Teams to compare",
        options=sorted_team_names,
        default=default_selection,
        help="Radar readability is best with 2-4 teams.",
    )

    if not selected_teams:
        st.info("Select at least one team to view comparison metrics.")
        return

    comparison_df, neutral_fallbacks = _build_team_comparison_dataframe(
        teams_payload=teams_payload,
        selected_teams=selected_teams,
        profile=profile,
    )

    if comparison_df.empty:
        st.info("No comparable team metrics available for selected teams.")
        return

    radar_labels = [label for _, label in _TEAM_RADAR_METRICS]
    if len(selected_teams) > 4:
        st.info(
            "Radar readability drops with more than 4 teams; use the table for dense comparisons."
        )

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        team_count = len(comparison_df.index)
        fill_mode = "toself" if team_count <= 4 else "none"
        fill_alpha = 0.16 if team_count <= 3 else 0.08
        marker_size = 6 if team_count <= 3 else 5
        line_width = 2.8 if team_count <= 3 else 2.2
        for _, row in comparison_df.iterrows():
            values = [float(row[label]) for label in radar_labels]
            trace_color = _team_brand_color(str(row["Team"]))
            fig.add_trace(
                go.Scatterpolar(
                    mode="lines+markers",
                    r=values + [values[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill=fill_mode,
                    name=str(row["Team"]),
                    line=dict(color=trace_color, width=line_width),
                    fillcolor=_hex_to_rgba(trace_color, fill_alpha),
                    marker=dict(color=trace_color, size=marker_size),
                    hovertemplate="%{theta}: %{r:.2f}<extra>%{fullData.name}</extra>",
                )
            )

        fig.update_layout(
            height=560,
            margin=dict(t=36, r=24, b=18, l=24),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8EDF2", size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(17,24,38,0.72)",
                bordercolor="rgba(232,237,242,0.14)",
                borderwidth=1,
                font=dict(size=13, color="#E8EDF2"),
            ),
            polar=dict(
                bgcolor="rgba(11,15,20,0.42)",
                radialaxis=dict(
                    visible=True,
                    range=[0.0, 1.0],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["20", "40", "60", "80", "100"],
                    tickfont=dict(color="#AAB4C2", size=12),
                    gridcolor="rgba(232,237,242,0.23)",
                    linecolor="rgba(232,237,242,0.30)",
                    angle=90,
                ),
                angularaxis=dict(
                    tickfont=dict(size=16, color="#E8EDF2"),
                    linecolor="rgba(232,237,242,0.24)",
                    gridcolor="rgba(232,237,242,0.14)",
                ),
            ),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception as exc:
        st.info(f"Radar chart unavailable ({exc}). Showing table only.")
    st.caption("Tip: compare 2-3 teams at a time for the clearest radar view.")

    display_df = comparison_df.copy()
    percent_cols = radar_labels + [
        "Overall Pace",
        "Overall Performance",
        "Radar Composite",
        "Prior Minus Radar",
    ]
    for column in percent_cols:
        display_df[column] = (display_df[column].astype(float) * 100.0).round(1)
    display_df = display_df[
        [
            "Team",
            "Overall Pace",
            "Radar Composite",
            "Overall Performance",
            "Prior Minus Radar",
            *radar_labels,
        ]
    ].rename(
        columns={
            "Overall Pace": "Profile Pace (Testing)",
            "Radar Composite": "Radar Composite (6 Metrics)",
            "Overall Performance": "Season Prior Strength",
            "Prior Minus Radar": "Prior - Radar Gap",
        }
    )

    st.dataframe(display_df, hide_index=True, width="stretch")
    st.caption(
        "Profile pace/radar come from selected testing-practice profile; "
        "Season Prior Strength is a separate baseline signal."
    )
    st.caption(
        f"Source: `{characteristics_path}` | profile=`{profile}` | "
        f"values are normalized (0-100, higher is better)."
    )
    if neutral_fallbacks > 0:
        st.caption(
            f"{neutral_fallbacks} missing metric(s) were filled with neutral value 50.0 for comparability."
        )


def render_team_comparison_page() -> None:
    """Render standalone team comparison tab."""
    st.header("Team Comparison")
    st.markdown(
        "Compare team characteristic fingerprints from testing/practice inputs. "
        "Profile metrics and season-prior baseline are separate signals and can diverge."
    )
    _render_team_comparison_section(year=DEFAULT_SEASON)


def _clear_fastf1_race_cache(year: int, race_name: str) -> None:
    """
    Clear FastF1 cache for a specific race to force fresh data fetch.

    This invalidates all cached session data for the race, including practice sessions,
    qualifying, and race results. The next FastF1 call will fetch fresh data from the API.
    """
    import shutil
    from pathlib import Path

    cache_dirs = [
        Path("data/raw/.fastf1_cache"),
        Path("data/raw/.fastf1_cache_testing"),
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        try:
            # FastF1 cache structure: {cache_dir}/{year}/{race_name}/...
            race_cache_path = cache_dir / str(year) / race_name.replace(" ", "_")
            if race_cache_path.exists():
                shutil.rmtree(race_cache_path)
                logger.info(f"Cleared FastF1 cache for {race_name} {year} at {race_cache_path}")

            # Also try with spaces removed completely
            race_cache_path_alt = cache_dir / str(year) / race_name.replace(" ", "")
            if race_cache_path_alt.exists():
                shutil.rmtree(race_cache_path_alt)
                logger.info(f"Cleared alternate FastF1 cache at {race_cache_path_alt}")

        except Exception as e:
            logger.warning(f"Could not clear FastF1 cache at {cache_dir}: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_race_options_cached(year: int) -> tuple[list[str], str | None]:
    """Load race options and cache schedule fetches for responsiveness."""
    try:
        schedule = fastf1.get_event_schedule(year)
        race_events = schedule[
            (schedule["EventFormat"].notna())
            & (~schedule["EventName"].str.contains("Testing", case=False, na=False))
        ].copy()

        race_options = []
        for _, event in race_events.iterrows():
            race_name = event["EventName"]
            event_format = str(event["EventFormat"]).lower()
            if "sprint" in event_format:
                race_options.append(f"{race_name} (Sprint)")
            else:
                race_options.append(race_name)

        return race_options, None
    except Exception as exc:
        return (
            [
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
                "Australian Grand Prix",
                "Japanese Grand Prix",
                "Chinese Grand Prix",
                "Miami Grand Prix",
            ],
            str(exc),
        )


def _load_race_options() -> list[str]:
    """Load race options from FastF1 schedule with sprint labels."""
    race_options, error = _load_race_options_cached(DEFAULT_SEASON)
    if error:
        st.error(f"Failed to load {DEFAULT_SEASON} calendar: {error}")
    return race_options


def _save_prediction_if_enabled(
    enable_logging: bool,
    prediction_results: dict,
    is_sprint: bool,
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
) -> None:
    if not enable_logging:
        return

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    logger_inst = PredictionLogger()

    latest_session = detector.get_latest_completed_session(year, race_name, is_sprint)

    if latest_session:
        if not logger_inst.has_prediction_for_session(year, race_name, latest_session):
            try:
                if is_sprint:
                    quali_grid = prediction_results["main_quali"]["grid"]
                    race_finish = prediction_results["main_race"]["finish_order"]
                    fp_blend_info = prediction_results.get("main_quali", {}).get(
                        "fp_blend_info", {}
                    )
                else:
                    quali_grid = prediction_results["qualifying"]["grid"]
                    race_finish = prediction_results["race"]["finish_order"]
                    fp_blend_info = prediction_results.get("qualifying", {}).get(
                        "fp_blend_info", {}
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
                st.info(f"Prediction saved for accuracy tracking (after {latest_session})")
            except Exception as e:
                st.warning(f"Could not save prediction: {e}")
        else:
            st.info(f"Prediction for {latest_session} already saved (max 1 per session)")
    else:
        st.info("No completed sessions yet; prediction not saved (will save after FP1/FP2/FP3/SQ)")


def _render_prediction_results(prediction_results: dict, is_sprint: bool) -> None:
    first_result = list(prediction_results.values())[0]
    timing = first_result.get("timing", {})
    if timing:
        st.success(f"Predictions complete in {timing['total']:.2f}s")
    else:
        st.success("Predictions complete.")

    if is_sprint:
        st.markdown("---")
        st.header("Sprint Weekend Cascade")
        st.info("Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race")

        display_prediction_result(
            prediction_results["sprint_quali"],
            "Sprint Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["sprint_race"],
            "Sprint Race Prediction",
            is_race=True,
        )
        display_prediction_result(
            prediction_results["main_quali"],
            "Main Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["main_race"],
            "Main Race Prediction",
            is_race=True,
        )
    else:
        st.markdown("---")
        st.header("Normal Weekend Cascade")
        st.info("Weekend flow: Qualifying → Race")

        display_prediction_result(
            prediction_results["qualifying"],
            "Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["race"],
            "Race Prediction",
            is_race=True,
        )


@st.cache_data(ttl=1800, show_spinner=False)
def _run_prediction_cached(
    race_name: str,
    weather: str,
    artifact_versions_key: tuple[tuple[str, tuple[int, str]], ...],
    is_sprint: bool,
    year: int,
) -> dict:
    """Run prediction with caching for unchanged inputs and artifact versions."""
    artifact_versions = dict(artifact_versions_key)
    return run_prediction(
        race_name,
        weather,
        artifact_versions,
        is_sprint=is_sprint,
        year=year,
    )


def execute_live_prediction_pipeline(
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
    force_refresh: bool = True,
    use_cached_prediction: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Refresh input data and execute a prediction run.

    Kept separate from Streamlit rendering so tests can assert refresh call order.

    Args:
        race_name: The name of the race
        weather: Weather forecast for the race
        year: Season year
        force_refresh: If True, clears FastF1 cache and forces re-check of session completion
        use_cached_prediction: If True, reuses cached prediction results when inputs are unchanged
        progress_callback: Optional callback for progress updates
    """
    pipeline_timing: dict[str, float] = {}
    pipeline_start = time.time()

    def _notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    # Clear FastF1 cache to force fresh data fetch if force_refresh enabled
    if force_refresh:
        _notify("Clearing FastF1 cache for fresh data...")
        clear_start = time.time()
        _clear_fastf1_race_cache(year, race_name)
        pipeline_timing["cache_clear"] = time.time() - clear_start

    update_start = time.time()
    _notify("Checking completed races and model updates...")
    auto_update_if_needed(force_recheck=force_refresh)
    pipeline_timing["race_update_check"] = time.time() - update_start

    weekend_start = time.time()
    _notify("Resolving weekend format...")
    is_sprint = is_sprint_weekend(year, race_name)
    pipeline_timing["weekend_lookup"] = time.time() - weekend_start

    practice_start = time.time()
    _notify("Checking completed practice sessions...")
    practice_update = auto_update_practice_characteristics_if_needed(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        force_recheck=force_refresh,
    )
    pipeline_timing["practice_update_check"] = time.time() - practice_start

    # Refresh cache on the same click after practice updates write new characteristics.
    if practice_update.get("updated") or force_refresh:
        _notify("Refreshing local caches after updates...")
        st.cache_resource.clear()
        st.cache_data.clear()

    # Capture versions after updates so cache invalidation keys include latest writes.
    prediction_start = time.time()
    artifact_versions = get_artifact_versions()
    if use_cached_prediction and not force_refresh:
        _notify("Running qualifying and race simulations (cache enabled)...")
        artifact_versions_key = tuple(sorted(artifact_versions.items()))
        prediction_results = _run_prediction_cached(
            race_name=race_name,
            weather=weather,
            artifact_versions_key=artifact_versions_key,
            is_sprint=is_sprint,
            year=year,
        )
    else:
        _notify("Running qualifying and race simulations...")
        prediction_results = run_prediction(
            race_name,
            weather,
            artifact_versions,
            is_sprint=is_sprint,
            year=year,
        )
    pipeline_timing["prediction_run"] = time.time() - prediction_start
    pipeline_timing["total"] = time.time() - pipeline_start

    return {
        "prediction_results": prediction_results,
        "is_sprint": is_sprint,
        "practice_update": practice_update,
        "pipeline_timing": pipeline_timing,
        "practice_update_error": None,
    }


def render_live_prediction_page(enable_logging: bool) -> None:
    st.header("Race Weekend Prediction")
    st.markdown(
        "Generate a practice-aware weekend forecast for qualifying and race. "
        "Use forced refresh when new sessions have just completed."
    )

    race_options = _load_race_options()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        race_selection = st.selectbox("Select Grand Prix", race_options)
        race_name = race_selection.replace(" (Sprint)", "")

    with col2:
        weather = st.selectbox("Weather Forecast", ["dry", "rain", "mixed"])

    st.subheader("Run Options")
    options_col, action_col = st.columns((2, 1), gap="large")
    with options_col:
        force_refresh = st.toggle(
            "Force Data Refresh",
            value=False,
            help=(
                "When enabled, clears source caches and recomputes from fresh session data. "
                "When disabled, unchanged inputs can reuse cached predictions."
            ),
        )
    with action_col:
        st.caption(" ")
        generate_prediction = st.button(
            "Generate Prediction",
            type="primary",
            width="stretch",
        )

    use_cached_prediction = not force_refresh
    mode_text = "Mode: Force refresh" if force_refresh else "Mode: Use cached prediction"
    st.markdown(f'<div class="run-options-note">{mode_text}</div>', unsafe_allow_html=True)

    if generate_prediction:
        status_placeholder = st.empty()

        with st.spinner("Running simulation pipeline..."):
            try:

                def update_status(message: str) -> None:
                    status_placeholder.info(f"Loading: {message}")

                pipeline_output = execute_live_prediction_pipeline(
                    race_name=race_name,
                    weather=weather,
                    year=DEFAULT_SEASON,
                    force_refresh=force_refresh,
                    use_cached_prediction=use_cached_prediction,
                    progress_callback=update_status,
                )
                prediction_results = pipeline_output["prediction_results"]
                is_sprint = bool(pipeline_output["is_sprint"])
                practice_update = pipeline_output["practice_update"]
                pipeline_timing = pipeline_output.get("pipeline_timing", {})
                status_placeholder.empty()

                st.warning("2026 regulation reset: predictions are uncertain until races complete.")

                if is_sprint:
                    st.info(
                        "**Sprint Weekend** - System predicts Sprint Qualifying (Friday) → "
                        "Sprint Race (Saturday) → Main Qualifying (Saturday) → Main Race (Sunday). "
                        "Sprint predictions use adjusted chaos modeling "
                        "(30% less variance, grid position +10% importance)."
                    )

                if practice_update.get("updated"):
                    st.success(
                        "Updated car characteristics from completed practice sessions: "
                        f"{', '.join(practice_update['completed_fp_sessions'])} "
                        f"({practice_update['teams_updated']} teams)"
                    )
                elif practice_update.get("completed_fp_sessions"):
                    st.info(
                        "Practice characteristics already up to date for sessions: "
                        f"{', '.join(practice_update['completed_fp_sessions'])}"
                    )

                if pipeline_timing:
                    timing_parts = [
                        f"updates {pipeline_timing.get('race_update_check', 0.0):.1f}s",
                        f"weekend lookup {pipeline_timing.get('weekend_lookup', 0.0):.1f}s",
                        f"practice check {pipeline_timing.get('practice_update_check', 0.0):.1f}s",
                        f"prediction {pipeline_timing.get('prediction_run', 0.0):.1f}s",
                        f"total {pipeline_timing.get('total', 0.0):.1f}s",
                    ]
                    st.caption("Pipeline timing: " + " | ".join(timing_parts))

                _save_prediction_if_enabled(
                    enable_logging=enable_logging,
                    prediction_results=prediction_results,
                    is_sprint=is_sprint,
                    race_name=race_name,
                    weather=weather,
                    year=DEFAULT_SEASON,
                )

                _render_prediction_results(prediction_results, is_sprint)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure data files are generated. Run: "
                    "`python scripts/extract_driver_characteristics.py --years 2023,2024,2025`"
                )


def render_model_insights_page() -> None:
    st.header("Model and Learning Runtime")

    st.markdown("""
    ### Runtime path

    The dashboard currently runs `Baseline2026Predictor` for both qualifying and race.

    **1. Team strength**
    - Uses baseline (pre-season), testing directionality, and current-season performance.
    - Applies a race-by-race weight schedule that shifts toward current-season data.
    - Uses compound-aware modifiers when validated compound samples are available.

    **2. Qualifying**
    - Uses best available weekend practice data for team pace blending.
    - Falls back to testing short-run profiles when no weekend practice data is available.
    - Applies model-only stabilization for teammate gaps and experience tiers.
    - Runs Monte Carlo simulations and reports median/interval grid outputs.

    **3. Race**
    - Uses predicted or actual qualifying grid depending on session availability.
    - Uses lap-by-lap simulation with team pace, racecraft, strategy, and reliability.
    - Applies track-aware overtaking and pit timing bias (undercut/overcut tendency).
    - Derives podium probability from ranked simulation outcomes for consistency.

    **4. Learning**
    - Saved predictions with actuals update a persistent calibration state.
    - Driver and teammate residual errors are tracked per session type.
    - Learned adjustments are applied in qualifying and race scoring.

    **5. Supporting systems**
    - Auto-updater ingests completed races into characteristics.
    - Testing updater refreshes run-profile and compound characteristics.
    - Bayesian ranking tools remain available for offline analysis workflows.
    """)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Qualifying (active path):**
        - Team/driver score: 70% team + 30% driver
        - Practice blend when available; testing fallback otherwise
        - Model-only teammate gap controls + learned adjustment offsets
        - Output: Monte Carlo median grid + confidence intervals
        """)

    with col2:
        st.markdown("""
        **Race (active path):**
        - Base pace weight: 40% (track-adjusted)
        - Grid influence: dynamic by overtaking difficulty
        - Driver skill term: 20%
        - DNF probability + chaos + strategy + safety car modifiers
        - Podium probability from ranked outcomes with monotonic smoothing
        """)


def render_prediction_accuracy_page() -> None:
    st.header("Prediction Accuracy Tracker")

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.prediction_metrics import PredictionMetrics

    logger_inst = PredictionLogger()
    metrics_calc = PredictionMetrics()

    all_predictions = logger_inst.get_all_predictions(DEFAULT_SEASON)

    if not all_predictions:
        st.info(
            "No predictions saved yet. Enable 'Save Predictions for Accuracy Tracking' "
            "in the sidebar and generate predictions after practice sessions."
        )
        return

    st.success(f"Found {len(all_predictions)} saved prediction(s)")

    predictions_with_actuals = [
        p
        for p in all_predictions
        if p.get("actuals") and (p["actuals"].get("qualifying") or p["actuals"].get("race"))
    ]

    if predictions_with_actuals:
        st.markdown("---")
        st.subheader("Overall Accuracy")

        agg_metrics = metrics_calc.aggregate_metrics(predictions_with_actuals)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Qualifying Metrics**")
            if "qualifying" in agg_metrics:
                q_metrics = agg_metrics["qualifying"]
                st.metric(
                    "Exact Position Accuracy",
                    f"{q_metrics['exact_accuracy']['mean']:.1f}%",
                    help="% of drivers predicted in exact correct position",
                )
                st.metric(
                    "Mean Position Error (MAE)",
                    f"{q_metrics['mae']['mean']:.2f} positions",
                    help="Average position error",
                )
                st.metric(
                    "Within ±3 Positions",
                    f"{q_metrics['within_3']['mean']:.1f}%",
                    help="% of predictions within 3 positions",
                )
                st.metric(
                    "Correlation",
                    f"{q_metrics['correlation']['mean']:.3f}",
                    help="Spearman correlation (-1 to 1, higher is better)",
                )

        with col2:
            st.markdown("**Race Metrics**")
            if "race" in agg_metrics:
                r_metrics = agg_metrics["race"]
                st.metric(
                    "Exact Position Accuracy",
                    f"{r_metrics['exact_accuracy']['mean']:.1f}%",
                    help="% of drivers predicted in exact correct position",
                )
                st.metric(
                    "Mean Position Error (MAE)",
                    f"{r_metrics['mae']['mean']:.2f} positions",
                    help="Average position error",
                )
                st.metric(
                    "Within ±3 Positions",
                    f"{r_metrics['within_3']['mean']:.1f}%",
                    help="% of predictions within 3 positions",
                )
                st.metric(
                    "Winner Prediction Accuracy",
                    f"{r_metrics['winner_accuracy']['percentage']:.1f}%",
                    help="% of races where winner was correctly predicted",
                )

        st.markdown("---")
        st.subheader("Per-Race Breakdown")

        for pred in predictions_with_actuals:
            metrics = metrics_calc.calculate_all_metrics(pred)
            if metrics:
                race_name = metrics["metadata"]["race_name"]
                session_name = metrics["metadata"]["session_name"]

                with st.expander(f"{race_name} (Predicted after {session_name})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        if "qualifying" in metrics:
                            st.markdown("**Qualifying**")
                            q = metrics["qualifying"]
                            st.write(f"- Exact: {q['exact_accuracy']:.1f}%")
                            st.write(f"- MAE: {q['mae']:.2f} positions")
                            st.write(f"- Within ±1: {q['within_1']:.1f}%")
                            st.write(f"- Correlation: {q['correlation']:.3f}")

                    with col2:
                        if "race" in metrics:
                            st.markdown("**Race**")
                            r = metrics["race"]
                            st.write(f"- Exact: {r['exact_accuracy']:.1f}%")
                            st.write(f"- MAE: {r['mae']:.2f} positions")
                            st.write(f"- Within ±3: {r['within_3']:.1f}%")
                            st.write(
                                f"- Winner: {'Correct' if r['winner_correct'] else 'Incorrect'}"
                            )
                            st.write(
                                f"- Podium: {r['podium']['correct_drivers']}/3 drivers correct"
                            )
    else:
        st.info(
            "Predictions saved, but no actual results added yet. After each race, "
            "you can update predictions with actual results to calculate accuracy."
        )

    st.markdown("---")
    st.subheader("All Saved Predictions")

    for pred in all_predictions:
        metadata = pred["metadata"]
        race_name = metadata["race_name"]
        session_name = metadata["session_name"]
        has_actuals = bool(
            pred.get("actuals")
            and (pred["actuals"].get("qualifying") or pred["actuals"].get("race"))
        )

        status_text = "Results added" if has_actuals else "Awaiting results"
        st.write(f"**{race_name}** (after {session_name}) - {status_text}")


def render_contact_page() -> None:
    st.header("Contact")
    st.markdown(
        """
        <div class="contact-grid">
          <section class="contact-card">
            <h3>Links</h3>
            <div class="contact-link-stack">
              <a class="contact-link-row" href="https://github.com/tomasz-solis" target="_blank" rel="noopener noreferrer">
                <span class="contact-link-row__label">GitHub</span>
                <span class="contact-link-row__value">@tomasz-solis</span>
              </a>
              <a class="contact-link-row" href="https://linkedin.com/in/tomaszsolis" target="_blank" rel="noopener noreferrer">
                <span class="contact-link-row__label">LinkedIn</span>
                <span class="contact-link-row__value">/in/tomaszsolis</span>
              </a>
            </div>
            <p class="contact-muted">Direct email is intentionally not published in the app.</p>
          </section>
          <section class="contact-card">
            <h3>Project Scope</h3>
            <p>Race weekend prediction workflow for the 2026 season with persistent learning and accuracy tracking.</p>
            <ul>
              <li>Baseline/testing/current-season team blending</li>
              <li>Practice-aware qualifying and race simulation</li>
              <li>Session-based logging for post-race accuracy analysis</li>
            </ul>
          </section>
        </div>
        <section class="contact-card contact-card--full">
          <h3>Disclaimer</h3>
          <p>Independent analytics project. Not affiliated with any racing series, team, or governing body.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_about_page() -> None:
    """Backwards-compatible alias for older routes."""
    render_contact_page()


def render_page(page: str, enable_logging: bool) -> None:
    if page in {"Prediction", "Live Prediction"}:
        render_live_prediction_page(enable_logging)
    elif page in {"Model & Learning", "Model Insights"}:
        render_model_insights_page()
    elif page == "Team Comparison":
        render_team_comparison_page()
    elif page == "Prediction Accuracy":
        render_prediction_accuracy_page()
    elif page in {"Contact", "About"}:
        render_contact_page()
    else:
        render_live_prediction_page(enable_logging)
