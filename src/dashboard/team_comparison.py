"""Streamlit entry points for team-comparison views."""

import json
from hashlib import sha1
from pathlib import Path
from typing import Any

import streamlit as st

from src.dashboard import team_comparison_fallbacks, team_radar, team_snapshot_history
from src.persistence.artifact_store import ArtifactStore
from src.persistence.config import should_read_db_first
from src.utils import config_loader
from src.utils.car_snapshot_history import SNAPSHOT_ARTIFACT_TYPE, sort_snapshot_payloads

_TEAM_RADAR_METRICS = team_radar._TEAM_RADAR_METRICS
_TEAM_BRAND_COLORS = team_radar._TEAM_BRAND_COLORS
_DEFAULT_TEAM_COLOR = team_radar._DEFAULT_TEAM_COLOR
_DEFAULT_BIG4_CANONICAL = team_radar._DEFAULT_BIG4_CANONICAL
_UNIT_CHART_RANGE_PADDING = team_radar._UNIT_CHART_RANGE_PADDING
_TIRE_DEG_SLOPE_DISPLAY_RANGE = team_radar._TIRE_DEG_SLOPE_DISPLAY_RANGE
_DISPLAY_SCORE_FLOOR = team_radar._DISPLAY_SCORE_FLOOR
_DISPLAY_SCORE_CEILING = team_radar._DISPLAY_SCORE_CEILING
_DISPLAY_SCORE_RANGE = team_radar._DISPLAY_SCORE_RANGE
_TIRE_DEG_SCORE_DISPLAY_RANGE = team_radar._TIRE_DEG_SCORE_DISPLAY_RANGE
_TOP_SPEED_KPH_BUFFER = team_radar._TOP_SPEED_KPH_BUFFER
_TOP_SPEED_SCORE_DISPLAY_RANGE = team_radar._TOP_SPEED_SCORE_DISPLAY_RANGE
_RADAR_AXIS_DISPLAY_MAX = team_radar._RADAR_AXIS_DISPLAY_MAX
_RAW_SECTOR_MIN_PADDING_SECONDS = team_radar._RAW_SECTOR_MIN_PADDING_SECONDS
_RAW_PACE_MIN_PADDING_SECONDS = team_radar._RAW_PACE_MIN_PADDING_SECONDS
_RAW_BRAKING_MIN_PADDING_PERCENT = team_radar._RAW_BRAKING_MIN_PADDING_PERCENT
_RAW_PACE_FIELD = team_radar._RAW_PACE_FIELD
_RAW_METRIC_FIELDS = team_radar._RAW_METRIC_FIELDS

_coerce_unit_metric = team_radar._coerce_unit_metric
_team_brand_color = team_radar._team_brand_color
_unit_chart_axis_range = team_radar._unit_chart_axis_range
_normalize_tire_deg_slope_for_display = team_radar._normalize_tire_deg_slope_for_display
_project_normalized_value_to_display_score = team_radar._project_normalized_value_to_display_score
_normalize_top_speed_kph_for_display = team_radar._normalize_top_speed_kph_for_display
_normalize_metric_for_display = team_radar._normalize_metric_for_display
_build_raw_metric_display_scale = team_radar._build_raw_metric_display_scale
_build_top_speed_display_scale = team_radar._build_top_speed_display_scale
_build_tire_deg_display_scale = team_radar._build_tire_deg_display_scale
_resolve_top_speed_metric_value = team_radar._resolve_top_speed_metric_value
_resolve_raw_metric_value = team_radar._resolve_raw_metric_value
_resolve_tire_deg_metric_value = team_radar._resolve_tire_deg_metric_value
_hex_to_rgba = team_radar._hex_to_rgba
_default_team_selection = team_radar._default_team_selection
_collect_profile_names = team_radar._collect_profile_names
_resolve_profile_metrics = team_radar._resolve_profile_metrics
_is_missing_payload_value = team_radar._is_missing_payload_value
_merge_team_payload_values = team_radar._merge_team_payload_values
_canonicalize_teams_payload_for_comparison = team_radar._canonicalize_teams_payload_for_comparison
_has_profile_metrics = team_radar._has_profile_metrics
_strip_raw_display_inputs = team_radar._strip_raw_display_inputs
_uses_same_event_average_fallback = team_radar._uses_same_event_average_fallback
_prepare_team_payload_for_comparison_scales = team_radar._prepare_team_payload_for_comparison_scales
_comparison_session_display_columns = team_radar._comparison_session_display_columns
_resolve_profile_overall_pace_display_value = team_radar._resolve_profile_overall_pace_display_value
_resolve_profile_display_metric_value = team_radar._resolve_profile_display_metric_value
_uses_placeholder_braking = team_radar._uses_placeholder_braking
_resolve_team_comparison_row = team_radar._resolve_team_comparison_row
_build_team_comparison_missing_column_map = team_radar._build_team_comparison_missing_column_map
_build_team_comparison_dataframe = team_radar._build_team_comparison_dataframe

_snapshot_label = team_snapshot_history._snapshot_label
_comparison_display_team_name = team_snapshot_history._comparison_display_team_name
_is_comparison_snapshot_session = team_snapshot_history._is_comparison_snapshot_session
_is_history_chart_snapshot_session = team_snapshot_history._is_history_chart_snapshot_session
_latest_snapshot_payload = team_snapshot_history._latest_snapshot_payload
_build_latest_snapshot_comparison_payload = (
    team_snapshot_history._build_latest_snapshot_comparison_payload
)
_resolve_snapshot_team_profiles = team_snapshot_history._resolve_snapshot_team_profiles
_snapshot_identity = team_snapshot_history._snapshot_identity
_average_snapshot_profile_metrics = team_snapshot_history._average_snapshot_profile_metrics
_resolve_same_event_profile_average_fallback = (
    team_snapshot_history._resolve_same_event_profile_average_fallback
)
_resolve_same_event_metric_average_fallback = (
    team_snapshot_history._resolve_same_event_metric_average_fallback
)
_resolve_latest_metric_fallback = team_snapshot_history._resolve_latest_metric_fallback
_resolve_usable_history_metric_value = team_snapshot_history._resolve_usable_history_metric_value
_resolve_latest_tire_deg_fallback = team_snapshot_history._resolve_latest_tire_deg_fallback
_apply_profile_tire_deg_fallbacks = team_snapshot_history._apply_profile_tire_deg_fallbacks
_apply_profile_braking_fallbacks = team_snapshot_history._apply_profile_braking_fallbacks
_build_snapshot_history_dataframe = team_snapshot_history._build_snapshot_history_dataframe
_ordered_snapshot_labels = team_snapshot_history._ordered_snapshot_labels
_build_development_summary_table = team_snapshot_history._build_development_summary_table

_build_same_event_display_metric_fallbacks = (
    team_comparison_fallbacks._build_same_event_display_metric_fallbacks
)
_build_latest_reliable_display_metric_fallbacks = (
    team_comparison_fallbacks._build_latest_reliable_display_metric_fallbacks
)
_apply_display_metric_fallbacks = team_comparison_fallbacks._apply_display_metric_fallbacks


def _resolve_processed_and_data_roots() -> tuple[Path, Path]:
    """Resolve processed-data path plus persistence root for artifact-backed history."""
    processed_path = Path(config_loader.get("paths.processed", "data/processed"))
    data_root = processed_path.parent if processed_path.name == "processed" else processed_path
    return processed_path, data_root


def _run_characteristics_season_sync(year: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Refresh snapshot history from cached sessions without touching live artifacts."""
    from src.systems.testing_updater import backfill_season_snapshot_history

    directionality_meta = payload.get("directionality_meta")
    testing_backend = "auto"
    force_renew_cache = False
    run_profile = "balanced"
    if isinstance(directionality_meta, dict):
        testing_backend = str(directionality_meta.get("testing_backend", "auto"))
        force_renew_cache = bool(directionality_meta.get("force_renew_cache", False))
        run_profile = str(directionality_meta.get("run_profile", "balanced"))

    return backfill_season_snapshot_history(
        year=year,
        characteristics_year=year,
        testing_backend=testing_backend,
        force_renew_cache=force_renew_cache,
        run_profile=run_profile,
        dry_run=False,
    )


def _snapshot_history_cache_token(year: int) -> str:
    """
    Return a freshness token for stored session snapshots.

    Snapshot history can be updated outside the running dashboard via CLI syncs,
    cron warmups, or manual backfills. This token lets the next Streamlit rerun
    notice those writes immediately instead of serving a stale cached list until
    the TTL expires.
    """
    _processed_path, data_root = _resolve_processed_and_data_roots()
    snapshot_root = data_root / SNAPSHOT_ARTIFACT_TYPE / str(int(year))

    if should_read_db_first():
        store = ArtifactStore(data_root=data_root)
        rows = store.list_artifacts(
            artifact_type=SNAPSHOT_ARTIFACT_TYPE,
            key_prefix=f"{year}::",
            limit=600,
        )
        row_fingerprints: list[tuple[str, str, str, str]] = []
        for row in rows:
            payload = row.get("data") if isinstance(row, dict) else None
            row_fingerprints.append(
                (
                    str(row.get("artifact_key", "")).strip() if isinstance(row, dict) else "",
                    str(row.get("created_at", "")).strip() if isinstance(row, dict) else "",
                    str(payload.get("captured_at", "")).strip()
                    if isinstance(payload, dict)
                    else "",
                    str(payload.get("session_started_at", "")).strip()
                    if isinstance(payload, dict)
                    else "",
                )
            )
        fingerprint_payload = json.dumps(
            row_fingerprints,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"rows:{sha1(fingerprint_payload.encode('utf-8')).hexdigest()}"

    if snapshot_root.exists():
        file_count = 0
        newest_mtime_ns = 0
        total_size = 0
        for path in snapshot_root.rglob("*.json"):
            if not path.is_file():
                continue
            try:
                stat_result = path.stat()
            except OSError:
                continue
            file_count += 1
            newest_mtime_ns = max(newest_mtime_ns, int(stat_result.st_mtime_ns))
            total_size += int(stat_result.st_size)
        return f"files:{file_count}:{newest_mtime_ns}:{total_size}"

    return "rows:missing"


@st.cache_data(ttl=300, show_spinner=False)
def _load_team_snapshot_history(year: int, cache_token: str = "") -> list[dict[str, Any]]:
    """Load stored session snapshots for development history charts."""
    del cache_token
    _processed_path, data_root = _resolve_processed_and_data_roots()
    store = ArtifactStore(data_root=data_root)
    rows = store.list_artifacts(
        artifact_type=SNAPSHOT_ARTIFACT_TYPE,
        key_prefix=f"{year}::",
        limit=600,
    )

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        payload = row.get("data") if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            continue
        event_name = str(payload.get("event_name", "")).strip()
        session_name = str(payload.get("session_name", "")).strip()
        if not event_name or not session_name:
            continue
        deduped.setdefault((event_name, session_name), payload)

    return sort_snapshot_payloads(list(deduped.values()))


def _render_development_history_section(
    year: int,
    selected_teams: list[str],
    profile: str,
    characteristics_payload: dict[str, Any],
) -> None:
    """Render per-session development trends from stored snapshot history."""
    st.subheader("Development Over Time")

    st.caption(
        "Sync rebuilds the stored session snapshot history from cached sessions without "
        "changing the live prediction artifact. The chart follows testing, practice, "
        "sprint, qualifying, and race snapshots in chronological order."
    )
    if st.button(
        "Sync Car Stats From Cache",
        key=f"snapshot_season_sync_{year}",
    ):
        try:
            summary = _run_characteristics_season_sync(year, characteristics_payload)
        except Exception as exc:
            st.info(f"Car-stats sync failed ({exc}).")
        else:
            _load_team_snapshot_history.clear()
            st.success(
                f"Synced {len(summary.get('loaded_sessions', []))} cached session snapshot(s)."
            )
            st.rerun()

    snapshots = _load_team_snapshot_history(year, _snapshot_history_cache_token(year))
    if not snapshots:
        st.info("No session snapshot history yet. Use the sync button to build it from cache.")
        return

    history_df = _build_snapshot_history_dataframe(
        snapshots=snapshots,
        selected_teams=selected_teams,
        profile=profile,
    )
    if history_df.empty:
        st.info(
            "No snapshot history matches the selected teams/profile yet. Try a different team set "
            "or wait for more sessions to be ingested."
        )
        return

    metric_options = ["Overall", "Overall Pace", *[label for _, label in _TEAM_RADAR_METRICS]]
    metric_label = st.selectbox(
        "Development metric",
        options=metric_options,
        index=0,
        help=(
            "Overall averages the radar metrics that are available in each snapshot. "
            "Overall Pace tracks the stored pace score separately, and the other options show "
            "one feature at a time."
        ),
    )

    if metric_label not in history_df.columns:
        st.info(f"No stored `{metric_label}` values are available for this selection yet.")
        return

    metric_frame_columns = ["Snapshot Order", "Snapshot", "Team", metric_label]
    if metric_label == "Overall":
        metric_frame_columns.extend(["Metric Count", "Metric Coverage"])
    metric_frame = history_df[metric_frame_columns].copy()
    if metric_frame[metric_label].dropna().empty:
        st.info(f"No stored `{metric_label}` values are available for this selection yet.")
        return
    category_order = _ordered_snapshot_labels(history_df)
    missing_history_points = bool((~history_df["Has Data"].fillna(False)).any())

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for team_name in selected_teams:
            team_frame = metric_frame[metric_frame["Team"] == team_name].sort_values(
                "Snapshot Order"
            )
            if team_frame.empty:
                continue
            trace_color = _team_brand_color(team_name)
            customdata = None
            hovertemplate = f"{metric_label}: %{{y:.2f}}<extra>{team_name}</extra>"
            if metric_label == "Overall":
                coverage_frame = team_frame.reindex(
                    columns=["Metric Count", "Metric Coverage"]
                ).fillna({"Metric Count": 0, "Metric Coverage": 0.0})
                customdata = coverage_frame.to_numpy()
                hovertemplate = (
                    "Overall: %{y:.2f}<br>"
                    "Coverage: %{customdata[0]:.0f}/6 metrics (%{customdata[1]:.0%})"
                    f"<extra>{team_name}</extra>"
                )
            fig.add_trace(
                go.Scatter(
                    x=list(team_frame["Snapshot"]),
                    y=list(team_frame[metric_label]),
                    mode="lines+markers",
                    name=team_name,
                    connectgaps=False,
                    line=dict(color=trace_color, width=3),
                    marker=dict(color=trace_color, size=8),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                )
            )

        fig.update_layout(
            height=420,
            margin=dict(t=24, r=24, b=24, l=24),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8EDF2", size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(17,24,38,0.72)",
                bordercolor="rgba(232,237,242,0.14)",
                borderwidth=1,
            ),
            xaxis=dict(
                tickangle=-30,
                categoryorder="array",
                categoryarray=category_order,
                gridcolor="rgba(232,237,242,0.12)",
                linecolor="rgba(232,237,242,0.20)",
            ),
            yaxis=dict(
                range=_unit_chart_axis_range(),
                tickvals=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                ticktext=["10", "30", "50", "70", "90", "100"],
                gridcolor="rgba(232,237,242,0.18)",
                linecolor="rgba(232,237,242,0.20)",
            ),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception as exc:
        st.info(f"Development chart unavailable ({exc}).")
    if metric_label == "Overall":
        st.caption(
            "Each point is one session snapshot. Overall averages the available radar metrics, "
            "and the hover shows how complete each session snapshot is."
        )
    else:
        st.caption(
            "Each point is one session snapshot. Relative changes matter more than absolute levels."
        )
    if missing_history_points:
        st.caption(
            "Gaps indicate sessions where a selected team has no stored snapshot sample, "
            "for example after a non-classified or double-retirement result."
        )


@st.cache_data(ttl=300, show_spinner=False)
def _load_team_characteristics_payload(year: int) -> tuple[dict[str, Any] | None, Path]:
    """Load season car characteristics payload used for team-comparison visualizations."""
    processed_path, _data_root = _resolve_processed_and_data_roots()
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


def _render_team_comparison_section(year: int = 2026) -> None:
    """Render profile-aware team comparison chart and metric table."""
    st.subheader("Latest Session Snapshot")

    payload, characteristics_path = _load_team_characteristics_payload(year)
    base_teams_payload = payload.get("teams") if isinstance(payload, dict) else {}
    if not isinstance(base_teams_payload, dict):
        base_teams_payload = {}

    snapshots = _load_team_snapshot_history(year, _snapshot_history_cache_token(year))
    latest_snapshot = _latest_snapshot_payload(snapshots)
    latest_snapshot_label = _snapshot_label(latest_snapshot) if latest_snapshot else ""

    teams_payload = _build_latest_snapshot_comparison_payload(
        base_teams_payload=base_teams_payload if isinstance(base_teams_payload, dict) else {},
        latest_snapshot=latest_snapshot,
        snapshot_history=snapshots,
    )
    source_label = f"latest snapshot `{latest_snapshot_label}`" if latest_snapshot_label else None

    if not teams_payload and isinstance(base_teams_payload, dict) and base_teams_payload:
        teams_payload = _canonicalize_teams_payload_for_comparison(base_teams_payload)
        source_label = f"season file `{characteristics_path}`"

    if not teams_payload:
        if not payload:
            st.info(f"Team characteristics unavailable at `{characteristics_path}`.")
        else:
            st.info("No team characteristics found for comparison.")
        return

    profile_names = _collect_profile_names(teams_payload)
    if not profile_names:
        st.info(
            "No session profile metrics are available yet for this season. "
            f'Run `python scripts/update_from_testing.py "Testing 1" --year {year} --apply` '
            "to populate comparison profiles."
        )
        return

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

    teams_with_signal = [
        team_name
        for team_name in selected_teams
        if isinstance(teams_payload.get(team_name), dict)
        and _has_profile_metrics(teams_payload[team_name], profile)
    ]
    teams_without_signal = [
        team_name for team_name in selected_teams if team_name not in teams_with_signal
    ]
    selected_weekend_fallback_teams = [
        team_name
        for team_name in teams_with_signal
        if _uses_same_event_average_fallback(teams_payload.get(team_name))
    ]
    comparison_display_names = {
        team_name: _comparison_display_team_name(team_name, teams_payload.get(team_name))
        for team_name in teams_with_signal
    }

    if not teams_with_signal:
        st.info(
            "Selected teams do not have session profile metrics for this profile yet. "
            "Choose another profile or refresh telemetry with "
            "`scripts/update_from_testing.py --apply`."
        )
        return

    if teams_without_signal:
        excluded_team_list = ", ".join(teams_without_signal)
        st.caption(f"Excluded teams without `{profile}` profile metrics: {excluded_team_list}.")
    if selected_weekend_fallback_teams:
        fallback_team_list = ", ".join(
            comparison_display_names[team_name]
            for team_name in sorted(selected_weekend_fallback_teams)
        )
        st.caption(
            "Weekend-average approximation applied to "
            f"{fallback_team_list} because the latest session snapshot has no stored team sample."
        )
        with st.expander("* What the asterisk means"):
            st.caption(
                "* only marks the latest-comparison profile pace and radar scores in this section."
            )
            st.caption(
                "For these teams, the latest session snapshot has no stored team sample, so the "
                "comparison uses an average of earlier same-weekend comparison scores."
            )
            st.caption(
                "This does not relabel the team, change season priors, or turn the missing race "
                "session in Development Over Time into a proxy point."
            )

    comparison_df, _neutral_fallbacks = _build_team_comparison_dataframe(
        teams_payload=teams_payload,
        selected_teams=teams_with_signal,
        profile=profile,
    )
    comparison_df, unresolved_neutral_fallbacks = _apply_display_metric_fallbacks(
        comparison_df,
        teams_payload=teams_payload,
        selected_teams=teams_with_signal,
        profile=profile,
        same_event_display_scores=_build_same_event_display_metric_fallbacks(
            snapshot_history=snapshots,
            latest_snapshot=latest_snapshot,
            teams_payload=teams_payload,
            selected_teams=teams_with_signal,
            profile=profile,
        ),
        latest_reliable_display_scores=_build_latest_reliable_display_metric_fallbacks(
            snapshot_history=snapshots,
            latest_snapshot=latest_snapshot,
            selected_teams=teams_with_signal,
            profile=profile,
        ),
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
            team_name = str(row["Team"])
            trace_color = _team_brand_color(team_name)
            fig.add_trace(
                go.Scatterpolar(
                    mode="lines+markers",
                    r=values + [values[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill=fill_mode,
                    name=comparison_display_names.get(team_name, team_name),
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
                    range=[0.0, _RADAR_AXIS_DISPLAY_MAX],
                    tickvals=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    ticktext=["10", "30", "50", "70", "90", "100"],
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
        "Radar Minus Prior",
    ]
    for column in percent_cols:
        display_df[column] = (display_df[column].astype(float) * 100.0).round(1)
    display_df["Team"] = display_df["Team"].map(
        lambda team_name: comparison_display_names.get(str(team_name), str(team_name))
    )
    display_df = display_df[
        [
            "Team",
            "Overall Pace",
            "Radar Composite",
            "Overall Performance",
            "Radar Minus Prior",
            *radar_labels,
        ]
    ].rename(
        columns={
            "Overall Pace": "Profile Pace (Latest Session)",
            "Radar Composite": "Radar Composite (6 Metrics)",
            "Overall Performance": "Season Prior Strength",
            "Radar Minus Prior": "Radar - Prior Gap",
        }
    )

    st.dataframe(display_df, hide_index=True, width="stretch")
    st.caption(
        "Profile pace/radar come from the latest synced comparison snapshot when present; "
        "starred teams use a same-weekend approximation in this section only. "
        "Season Prior Strength stays a separate baseline signal."
    )
    st.caption(
        f"Source: {source_label or f'`{characteristics_path}`'} | profile=`{profile}` | "
        "session-derived values use a 10-100 display scale (higher is better)."
    )
    st.caption(
        "When the latest snapshot lacks a usable tire-deg readout, the chart carries forward "
        "the newest available long-run tire signal instead of defaulting to neutral."
    )
    st.caption(
        "Tire-deg prefers raw slope data and normalizes the current snapshot's best and worst "
        "samples to the 10-100 display range; when only one raw slope exists, it falls back to "
        "a stable absolute-slope score."
    )
    st.caption(
        "Top speed prefers raw trap-speed data when the snapshot has it and maps the slowest and "
        "fastest sampled teams to the 10-100 display endpoints."
    )
    st.caption(
        "Cornering and pace also prefer raw session times when the snapshot has them, so the "
        "fastest sampled team reaches 100 and the slowest reaches 10 instead of compressing "
        "everyone into a narrow middle band."
    )
    st.caption(
        "Braking now prefers a stored telemetry-based proxy when snapshots have it; if the latest "
        "session still carries a legacy placeholder, the comparison falls back to earlier "
        "same-weekend braking or the most recent stored session proxy."
    )
    if unresolved_neutral_fallbacks > 0:
        st.caption(
            f"{unresolved_neutral_fallbacks} metric(s) had no trustworthy prior value and remained at neutral 50.0."
        )

    _render_development_history_section(
        year=year,
        selected_teams=selected_teams,
        profile=profile,
        characteristics_payload=payload or {},
    )
