"""Streamlit entry points for team-comparison views."""

import json
import logging
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard import team_comparison_fallbacks, team_radar, team_snapshot_history
from src.persistence.artifact_store import ArtifactStore
from src.persistence.config import should_read_db_first
from src.utils import config_loader
from src.utils.car_snapshot_history import SNAPSHOT_ARTIFACT_TYPE, sort_snapshot_payloads

logger = logging.getLogger(__name__)

_TEAM_RADAR_METRICS = team_radar._TEAM_RADAR_METRICS
_TEAM_BRAND_COLORS = team_radar._TEAM_BRAND_COLORS
_DEFAULT_TEAM_COLOR = team_radar._DEFAULT_TEAM_COLOR
_DEFAULT_BIG4_CANONICAL = team_radar._DEFAULT_BIG4_CANONICAL
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

_SNAPSHOT_HISTORY_ROW_LIMIT = 5000
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
_resolve_carried_tire_deg_source = team_radar._resolve_carried_tire_deg_source
_resolve_latest_tire_deg_fallback = team_snapshot_history._resolve_latest_tire_deg_fallback
_apply_profile_tire_deg_fallbacks = team_snapshot_history._apply_profile_tire_deg_fallbacks
_apply_profile_braking_fallbacks = team_snapshot_history._apply_profile_braking_fallbacks
_build_snapshot_history_dataframe = team_snapshot_history._build_snapshot_history_dataframe
_ordered_snapshot_labels = team_snapshot_history._ordered_snapshot_labels
_smooth_development_history_dataframe = team_snapshot_history._smooth_development_history_dataframe
_rescale_history_dataframe_per_session = (
    team_snapshot_history._rescale_history_dataframe_per_session
)
_recompute_history_composites = team_snapshot_history._recompute_history_composites
_WINDOW_FILLED_SUFFIX = team_snapshot_history._WINDOW_FILLED_SUFFIX
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


def _apply_smoothed_radar_values(
    comparison_df: Any,
    *,
    history_df: Any,
    snapshot_label: str,
) -> tuple[Any, dict[str, set[str]]]:
    """
    Replace the radar's single-session scores with the smoothed ones for that session.

    Returns the updated frame and, per team, the metrics the session itself never
    measured, so the view can say where those came from. `Radar Composite` is
    recomputed from the metrics actually shown, keeping it equal to the mean of the
    drawn spokes.
    """
    radar_labels = [label for _key, label in _TEAM_RADAR_METRICS]
    filled_metrics: dict[str, set[str]] = {}
    if comparison_df.empty or history_df is None or history_df.empty:
        return comparison_df, filled_metrics
    if "Snapshot" not in history_df.columns or not snapshot_label:
        return comparison_df, filled_metrics

    session_rows = history_df[history_df["Snapshot"] == snapshot_label]
    if session_rows.empty:
        return comparison_df, filled_metrics

    by_team = {str(row["Team"]): row for _index, row in session_rows.iterrows()}
    updated_rows: list[dict[str, Any]] = []
    for row in comparison_df.to_dict(orient="records"):
        team_name = str(row.get("Team", "")).strip()
        smoothed_row = by_team.get(team_name)
        updated_row = dict(row)
        if smoothed_row is not None:
            for label in radar_labels:
                value = smoothed_row.get(label)
                if value is None or pd.isna(value):
                    continue
                updated_row[label] = float(value)
                # Mark the value the session itself never measured. Window size is the
                # wrong test: a 2-sample window is simply what the first and last session
                # of a weekend get, and says nothing about the value's provenance.
                if bool(smoothed_row.get(f"{label}{_WINDOW_FILLED_SUFFIX}", False)):
                    filled_metrics.setdefault(team_name, set()).add(label)
            shown = [
                float(updated_row[label])
                for label in radar_labels
                if updated_row.get(label) is not None and not pd.isna(updated_row[label])
            ]
            if shown:
                composite = float(sum(shown) / len(shown))
                updated_row["Radar Composite"] = composite
                updated_row["Radar Minus Prior"] = composite - float(
                    updated_row.get("Overall Performance", 0.0)
                )
        updated_rows.append(updated_row)

    return pd.DataFrame(updated_rows), filled_metrics


def _all_snapshot_team_names(snapshots: list[dict[str, Any]]) -> list[str]:
    """Return every team appearing anywhere in the snapshot history, canonically named."""
    names: set[str] = set()
    for snapshot_payload in snapshots:
        teams_payload = snapshot_payload.get("teams")
        if not isinstance(teams_payload, dict):
            continue
        for raw_team_name in teams_payload:
            mapped_name = team_snapshot_history.map_team_to_characteristics(str(raw_team_name))
            names.add(
                mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
            )
    return sorted(names)


def _build_display_history_frame(
    snapshots: list[dict[str, Any]],
    profile: str,
    *,
    smooth: bool,
) -> Any:
    """
    Build the history frame both the radar and the season chart read from.

    Always built over the whole field, never the current selection: the smoothed
    scores are re-normalised against the other teams in each session, so scoping
    this to the multiselect would make a team's score move when you tick another
    team. Callers filter to their selection afterwards.
    """
    history_df = _build_snapshot_history_dataframe(
        snapshots=snapshots,
        selected_teams=_all_snapshot_team_names(snapshots),
        profile=profile,
    )
    if history_df.empty or not smooth:
        return history_df
    history_df = _smooth_development_history_dataframe(history_df)
    history_df = _rescale_history_dataframe_per_session(history_df)
    return _recompute_history_composites(history_df)


def _development_metric_options(history_df: Any) -> list[str]:
    """Return development metrics that make sense for the available history payload."""
    columns = set(getattr(history_df, "columns", []))
    options: list[str] = []
    if "Overall Pace" in columns:
        options.append("Overall Pace")
    if "Radar Average" in columns:
        options.append("Radar Average")
    elif "Overall" in columns:
        options.append("Overall")
    for label_name in ("Qualifying Pace", "Race Pace"):
        if label_name in columns:
            options.append(label_name)
    options.extend(label for _, label in _TEAM_RADAR_METRICS)
    return options


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
        limit=_SNAPSHOT_HISTORY_ROW_LIMIT,
    )
    if len(rows) >= _SNAPSHOT_HISTORY_ROW_LIMIT:
        # Rows are artifact *versions*, not snapshots, so a re-sync adds a full season
        # of them. Hitting the cap means the oldest weekends were silently dropped
        # (the query is created_at DESC), which would look like missing history.
        logger.warning(
            "Snapshot history hit the %s-row cap for %s; the earliest weekends may be "
            "missing from the chart. Prune old artifact versions or raise the cap.",
            _SNAPSHOT_HISTORY_ROW_LIMIT,
            year,
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


def _short_event_label(event_name: str) -> str:
    """Trim a race name for a compact axis tick (e.g. 'Bahrain Grand Prix' -> 'Bahrain')."""
    name = str(event_name).strip()
    for suffix in (" Grand Prix", " GP"):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _development_race_ticks(
    history_df: Any, category_order: list[str]
) -> tuple[list[str], list[str]]:
    """Return one x tick per race weekend, at that weekend's first session.

    The development axis has a point for every session snapshot across the season, so
    labelling each one makes it unreadable. Only the weekend is named on the axis; the
    session lives in the hover. (impeccable: declutter a dense time axis)
    """
    if history_df.empty or "Event" not in history_df.columns:
        return [], []
    event_by_label = dict(
        zip(
            history_df["Snapshot"],
            history_df["Event"],
            strict=False,
        )
    )
    tickvals: list[str] = []
    ticktext: list[str] = []
    seen: set[str] = set()
    for label in category_order:
        event = str(event_by_label.get(label, "")).strip()
        if event and event not in seen:
            seen.add(event)
            tickvals.append(label)  # categorical axis: tick at this session's position
            ticktext.append(_short_event_label(event))
    return tickvals, ticktext


def _render_development_history_section(
    year: int,
    selected_teams: list[str],
    profile: str,
    characteristics_payload: dict[str, Any],
    history_df: Any,
) -> None:
    """Render per-session development trends from stored snapshot history."""
    st.subheader("Relative Performance Over Time")

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

    history_df = history_df[history_df["Team"].isin(selected_teams)].copy()
    if history_df.empty:
        st.info(
            "No snapshot history matches the selected teams/profile yet. Try a different team set "
            "or wait for more sessions to be ingested."
        )
        return

    metric_options = _development_metric_options(history_df)
    metric_label = st.selectbox(
        "Relative performance metric",
        options=metric_options,
        index=0,
        help=(
            "Overall Pace tracks actual session pace for the selected profile. "
            "Radar Average is the mean of the six radar metrics in that snapshot. "
            "Qualifying Pace always uses the short-run profile when snapshots store it, "
            "Race Pace uses long-run, and the other options show one feature at a time."
        ),
    )
    st.caption(
        "Overall Pace reflects actual lap-time performance for the selected profile. "
        "Radar Average is the mean of the six radar spokes, so the two can move in different directions."
    )

    if metric_label not in history_df.columns:
        st.info(f"No stored `{metric_label}` values are available for this selection yet.")
        return

    metric_frame_columns = ["Snapshot Order", "Snapshot", "Team", metric_label]
    if metric_label in {"Radar Average", "Overall"}:
        metric_frame_columns.extend(["Metric Count", "Metric Coverage"])
    metric_frame = history_df[metric_frame_columns].copy()
    if metric_frame[metric_label].dropna().empty:
        st.info(f"No stored `{metric_label}` values are available for this selection yet.")
        return
    category_order = _ordered_snapshot_labels(history_df)
    race_tickvals, race_ticktext = _development_race_ticks(history_df, category_order)
    missing_history_points = bool(metric_frame[metric_label].isna().any())

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
            # Scores are stored 0-1 but the axis is labelled 0-100, so the hover reads
            # from a pre-scaled copy; Plotly hover templates cannot do arithmetic.
            display_values = [float(value) * 100.0 for value in team_frame[metric_label]]
            customdata: list[list[float]] = [[value] for value in display_values]
            hovertemplate = f"{metric_label}: %{{customdata[0]:.1f}}<extra>{team_name}</extra>"
            if metric_label in {"Radar Average", "Overall"}:
                coverage_frame = team_frame.reindex(
                    columns=["Metric Count", "Metric Coverage"]
                ).fillna({"Metric Count": 0, "Metric Coverage": 0.0})
                customdata = [
                    [display_value, float(metric_count), float(metric_coverage)]
                    for display_value, metric_count, metric_coverage in zip(
                        display_values,
                        coverage_frame["Metric Count"],
                        coverage_frame["Metric Coverage"],
                        strict=False,
                    )
                ]
                hovertemplate = (
                    "Radar Average: %{customdata[0]:.1f}<br>"
                    "Coverage: %{customdata[1]:.0f}/6 metrics (%{customdata[2]:.0%})"
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
            # Unified hover turns each session into one card listing every selected
            # team and its value, so the order of teams at that point reads at a glance.
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(17,24,38,0.94)",
                bordercolor="rgba(232,237,242,0.20)",
                font=dict(color="#E8EDF2", size=13),
            ),
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
                # One label per race weekend instead of one per session; the session
                # stays in the hover. Angle + automargin so back-to-back races
                # (e.g. Barcelona/Austrian) that sit close on the axis don't overlap.
                tickmode="array",
                tickvals=race_tickvals,
                ticktext=race_ticktext,
                tickangle=-40,
                automargin=True,
                categoryorder="array",
                categoryarray=category_order,
                gridcolor="rgba(232,237,242,0.12)",
                linecolor="rgba(232,237,242,0.20)",
            ),
            yaxis=dict(
                range=_unit_chart_axis_range(),
                # Zero is labelled because the axis is anchored there now; a line's
                # height is its score rather than its distance above the lowest point.
                tickvals=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                ticktext=["0", "10", "30", "50", "70", "90", "100"],
                gridcolor="rgba(232,237,242,0.18)",
                linecolor="rgba(232,237,242,0.20)",
            ),
        )
        # Keyed so the CSS can let this season-long time series use the full page
        # width (the default chart cap is the narrower readable width, which crams
        # the race axis). See `.st-key-ts-dev-over-time` in styles.py.
        st.plotly_chart(
            fig,
            width="stretch",
            key="ts-dev-over-time",
            config={"displayModeBar": False},
        )
    except Exception as exc:
        st.info(f"Relative performance chart unavailable ({exc}).")
    st.caption(
        "Each point ranks a team against the field in that same session: the best car scores 100 "
        "and the slowest 10, every session. A flat line means an unchanged position relative to "
        "the field, not unchanged pace. Big swings can come from fuel loads, tires, and run plans."
    )
    if metric_label in {"Radar Average", "Overall"}:
        st.caption(
            "Each point is one session snapshot. Radar Average is the mean of the available radar metrics, "
            "and the hover shows how complete each session snapshot is."
        )
    elif metric_label == "Overall Pace":
        st.caption(
            "Each point is one session snapshot. Overall Pace follows the currently selected "
            "comparison profile."
        )
    elif metric_label == "Qualifying Pace":
        st.caption(
            "Each point is one session snapshot. Qualifying Pace always tracks the short-run "
            "profile when that snapshot stores one."
        )
    elif metric_label == "Race Pace":
        st.caption(
            "Each point is one session snapshot. Race Pace always tracks the long-run profile "
            "when that snapshot stores one."
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
            "No session profile metrics are available yet for this season. Run "
            f'`python scripts/update_from_testing.py "Testing 1" --year {year} --apply` '
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

    # Declared here, above the radar, because both panels read it. Streamlit renders
    # in call order, so a toggle created inside the chart section could not reach the
    # radar drawn before it.
    smooth_history = st.toggle(
        "Weekend average",
        value=True,
        help=(
            "Averages each team across the sessions of one race weekend, never across two, "
            "then re-spreads each session over 10-100 so the fastest car reads 100. Applies to "
            "both the radar and the season chart. Sessions a team missed stay gaps."
        ),
    )
    display_history_df = _build_display_history_frame(snapshots, profile, smooth=smooth_history)

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
                "session in Relative Performance Over Time into a proxy point."
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

    window_filled_metrics: dict[str, set[str]] = {}
    if smooth_history:
        comparison_df, window_filled_metrics = _apply_smoothed_radar_values(
            comparison_df,
            history_df=display_history_df,
            snapshot_label=latest_snapshot_label,
        )

    radar_labels = [label for _, label in _TEAM_RADAR_METRICS]
    # Mark any value the named session did not measure, so it is never read as one it
    # did. Smoothing supersedes the tire-deg carry-forward, so only one of these fires.
    carried_tire_deg_sources: dict[str, str] = {}
    if smooth_history:
        marked_metrics = {label for labels in window_filled_metrics.values() for label in labels}
    else:
        carried_tire_deg_sources = {
            team_name: source
            for team_name in teams_with_signal
            if (
                source := _resolve_carried_tire_deg_source(
                    teams_payload.get(team_name, {}), profile
                )
            )
        }
        marked_metrics = {"Tire Deg"} if carried_tire_deg_sources else set()
    radar_display_labels = [
        f"{label} †" if label in marked_metrics else label for label in radar_labels
    ]
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
            carried_source = carried_tire_deg_sources.get(team_name)
            team_filled_metrics = window_filled_metrics.get(team_name, set())
            radar_hover_notes = [
                f"<br>carried from {carried_source}"
                if label == "Tire Deg" and carried_source
                else (
                    "<br>from other sessions this weekend" if label in team_filled_metrics else ""
                )
                for label in radar_labels
            ]
            trace_color = _team_brand_color(team_name)
            fig.add_trace(
                go.Scatterpolar(
                    mode="lines+markers",
                    r=values + [values[0]],
                    theta=radar_display_labels + [radar_display_labels[0]],
                    fill=fill_mode,
                    name=comparison_display_names.get(team_name, team_name),
                    line=dict(color=trace_color, width=line_width),
                    fillcolor=_hex_to_rgba(trace_color, fill_alpha),
                    marker=dict(color=trace_color, size=marker_size),
                    # Radial ticks are labelled 0-100 over 0-1 data; hover matches them.
                    customdata=[
                        [value * 100.0, note]
                        for value, note in zip(
                            values + [values[0]],
                            radar_hover_notes + [radar_hover_notes[0]],
                            strict=True,
                        )
                    ],
                    hovertemplate=(
                        "%{theta}: %{customdata[0]:.1f}%{customdata[1]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
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
    # Only one of these is ever populated: smoothing fills from the weekend, otherwise
    # the tire-deg carry-forward reaches back to an earlier one.
    if window_filled_metrics:
        dagger_summary = ", ".join(
            f"{comparison_display_names.get(team_name, team_name)}: {', '.join(sorted(labels))}"
            for team_name, labels in sorted(window_filled_metrics.items())
        )
        dagger_note = f"† Supplied by other sessions of this weekend — {dagger_summary}."
        dagger_detail = (
            "This session recorded no reading for these metrics, so the value comes from the "
            "other sessions of the same weekend. Qualifying and sprint-qualifying runs are too "
            "short to measure tire degradation, which is the usual case."
        )
    elif carried_tire_deg_sources:
        dagger_summary = ", ".join(
            f"{comparison_display_names.get(team_name, team_name)} ({source})"
            for team_name, source in sorted(carried_tire_deg_sources.items())
        )
        dagger_note = f"† Tire Deg carried forward for {dagger_summary}."
        dagger_detail = (
            "Qualifying and sprint-qualifying runs are too short to measure tire degradation, "
            "so this session stores none and the comparison reuses each team's last measured "
            "value. It is scored on the absolute degradation scale rather than against this "
            "session's field, so read it as a standing estimate of race pace."
        )
    else:
        dagger_note = ""
        dagger_detail = ""
    if dagger_note:
        st.caption(dagger_note)
        with st.expander("† What the dagger means"):
            st.caption(dagger_detail)

    if smooth_history:
        st.caption(
            "Radar and table show this weekend's average rather than the single session named "
            "above, and Relative Performance Over Time draws the same numbers. Switch off "
            "Weekend average for the raw single-session values."
        )

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
        history_df=display_history_df,
    )
