"""Team-comparison data loading and rendering helpers."""

import json
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.persistence.artifact_store import ArtifactStore
from src.utils import config_loader
from src.utils.car_snapshot_history import (
    SNAPSHOT_ARTIFACT_TYPE,
    session_order_index,
    snapshot_sort_timestamp,
    sort_snapshot_payloads,
)
from src.utils.team_mapping import canonicalize_team, map_team_to_characteristics

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
    "HAAS": "#7C8798",
    "RB": "#6692FF",
    "WILLIAMS": "#005AFF",
    "AUDI": "#C4122E",
    "CADILLAC": "#2A4AA0",
}
_DEFAULT_TEAM_COLOR = "#B6BABD"
_DEFAULT_BIG4_CANONICAL: tuple[str, ...] = ("MCLAREN", "MERCEDES", "FERRARI", "RED BULL")
_UNIT_CHART_RANGE_PADDING = 0.02


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


def _unit_chart_axis_range(padding: float = _UNIT_CHART_RANGE_PADDING) -> list[float]:
    """Return a slightly padded 0-1 range so endpoint markers are not clipped."""
    bounded_padding = max(0.0, float(padding))
    return [0.0 - bounded_padding, 1.0 + bounded_padding]


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
        preferred_team = canonical_to_team.get(canonical_id)
        if preferred_team and preferred_team not in selected:
            selected.append(preferred_team)
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
    profile_names: set[str] = set()
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
            elif (
                any(
                    metric_name in testing_characteristics
                    for metric_name, _label in _TEAM_RADAR_METRICS
                )
                or "overall_pace" in testing_characteristics
            ):
                profile_names.add("balanced")

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


def _is_missing_payload_value(value: Any) -> bool:
    """Return True for payload values that should be considered missing."""
    if value is None:
        return True
    if isinstance(value, float):
        return not isfinite(value)
    return False


def _merge_team_payload_values(existing: Any, incoming: Any) -> Any:
    """
    Merge team payload fragments while preserving existing non-missing values.

    This is used to combine legacy and rebranded team keys that belong to the
    same canonical constructor identity.
    """
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = deepcopy(existing)
        for key, incoming_value in incoming.items():
            if key not in merged:
                merged[key] = deepcopy(incoming_value)
                continue
            merged[key] = _merge_team_payload_values(merged[key], incoming_value)
        return merged

    if _is_missing_payload_value(existing) and not _is_missing_payload_value(incoming):
        return deepcopy(incoming)
    return deepcopy(existing)


def _canonicalize_teams_payload_for_comparison(
    teams_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Canonicalize team labels for UI display and merge alias payloads."""
    canonical_payload: dict[str, dict[str, Any]] = {}
    for raw_team_name, raw_team_data in teams_payload.items():
        if not isinstance(raw_team_data, dict):
            continue

        mapped_name = map_team_to_characteristics(str(raw_team_name))
        display_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )
        team_payload = deepcopy(raw_team_data)
        team_payload["team_name"] = display_name

        existing_data = canonical_payload.get(display_name)
        if existing_data is None:
            canonical_payload[display_name] = team_payload
            continue
        canonical_payload[display_name] = _merge_team_payload_values(existing_data, team_payload)

    return canonical_payload


def _has_profile_metrics(team_data: dict[str, Any], profile: str) -> bool:
    """Return True when at least one profile metric is available for the selected profile."""
    overall_perf = _coerce_unit_metric(team_data.get("overall_performance"))
    team_name = team_data.get("team_name")
    if (
        overall_perf is not None
        and isinstance(team_name, str)
        and canonicalize_team(team_name) == "AUDI"
    ):
        return True

    metrics_payload = _resolve_profile_metrics(team_data, profile)
    if not isinstance(metrics_payload, dict):
        return False

    overall_pace = _coerce_unit_metric(metrics_payload.get("overall_pace"))
    if overall_pace is not None:
        return True

    for payload_key, _label in _TEAM_RADAR_METRICS:
        metric_value = _coerce_unit_metric(metrics_payload.get(payload_key))
        if metric_value is not None:
            return True
    return False


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

        overall_perf = _coerce_unit_metric(team_data.get("overall_performance"))
        row["Overall Performance"] = overall_perf if overall_perf is not None else 0.5
        if overall_perf is None:
            neutral_fallback_count += 1

        overall_pace = _coerce_unit_metric(metrics_payload.get("overall_pace"))
        row["Overall Pace"] = overall_pace if overall_pace is not None else 0.5
        if overall_pace is None:
            neutral_fallback_count += 1

        for payload_key, label in _TEAM_RADAR_METRICS:
            metric_value = _coerce_unit_metric(metrics_payload.get(payload_key))
            row[label] = metric_value if metric_value is not None else 0.5
            if metric_value is None:
                neutral_fallback_count += 1

        radar_values = [float(row[label]) for _, label in _TEAM_RADAR_METRICS]
        radar_composite = float(sum(radar_values) / len(radar_values))
        row["Radar Composite"] = radar_composite
        row["Radar Minus Prior"] = radar_composite - float(row["Overall Performance"])

        rows.append(row)

    if not rows:
        return pd.DataFrame(), neutral_fallback_count

    rows.sort(key=lambda row: float(row.get("Overall Pace", 0.0)), reverse=True)
    frame = pd.DataFrame(rows)
    return frame, neutral_fallback_count


def _resolve_processed_and_data_roots() -> tuple[Path, Path]:
    """Resolve processed-data path plus persistence root for artifact-backed history."""
    processed_path = Path(config_loader.get("paths.processed", "data/processed"))
    data_root = processed_path.parent if processed_path.name == "processed" else processed_path
    return processed_path, data_root


def _snapshot_label(snapshot_payload: dict[str, Any]) -> str:
    """Build a compact label for one stored snapshot."""
    event_name = str(snapshot_payload.get("event_name", "")).strip()
    session_name = str(snapshot_payload.get("session_name", "")).strip()
    if not event_name:
        return session_name or "Unknown Session"
    if not session_name:
        return event_name
    if session_name.startswith(event_name):
        return session_name
    return f"{event_name} {session_name}"


def _is_comparison_snapshot_session(session_name: str) -> bool:
    """Return True for stored snapshots that belong in comparison charts and tables."""
    normalized = "".join(ch for ch in str(session_name).strip().upper() if ch.isalnum())
    if not normalized:
        return False
    return session_order_index(normalized) in {1, 2, 3, 6, 7}


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


@st.cache_data(ttl=300, show_spinner=False)
def _load_team_snapshot_history(year: int) -> list[dict[str, Any]]:
    """Load stored session snapshots for development history charts."""
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


def _latest_snapshot_payload(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest stored comparison snapshot with team profile data, if any."""
    for snapshot_payload in reversed(snapshots):
        session_name = str(snapshot_payload.get("session_name", "")).strip()
        if not _is_comparison_snapshot_session(session_name):
            continue
        teams_payload = snapshot_payload.get("teams")
        if isinstance(teams_payload, dict) and teams_payload:
            return snapshot_payload
    return None


def _build_latest_snapshot_comparison_payload(
    *,
    base_teams_payload: dict[str, Any],
    latest_snapshot: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build comparison input from the newest snapshot while keeping season priors."""
    if not isinstance(latest_snapshot, dict):
        return {}

    snapshot_teams_payload = latest_snapshot.get("teams")
    if not isinstance(snapshot_teams_payload, dict):
        return {}

    canonical_base_payload = _canonicalize_teams_payload_for_comparison(base_teams_payload)
    merged_payload: dict[str, dict[str, Any]] = {}

    for raw_team_name, raw_team_payload in snapshot_teams_payload.items():
        if not isinstance(raw_team_payload, dict):
            continue

        profiles = raw_team_payload.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            continue

        mapped_name = map_team_to_characteristics(str(raw_team_name))
        display_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )
        team_payload: dict[str, Any] = {"testing_characteristics_profiles": deepcopy(profiles)}

        balanced_profile = profiles.get("balanced")
        if isinstance(balanced_profile, dict):
            team_payload["testing_characteristics"] = deepcopy(balanced_profile)

        base_team_payload = canonical_base_payload.get(display_name)
        if isinstance(base_team_payload, dict):
            overall_performance = _coerce_unit_metric(base_team_payload.get("overall_performance"))
            if overall_performance is not None:
                team_payload["overall_performance"] = overall_performance

        merged_payload[display_name] = team_payload

    return _canonicalize_teams_payload_for_comparison(merged_payload)


def _build_snapshot_history_dataframe(
    snapshots: list[dict[str, Any]],
    selected_teams: list[str],
    profile: str,
) -> pd.DataFrame:
    """Flatten stored snapshots into one line-chart-friendly dataframe."""
    rows: list[dict[str, Any]] = []

    for index, snapshot_payload in enumerate(snapshots):
        session_name = str(snapshot_payload.get("session_name", "")).strip()
        if not _is_comparison_snapshot_session(session_name):
            continue

        label = _snapshot_label(snapshot_payload)
        teams_payload = snapshot_payload.get("teams")
        if not isinstance(teams_payload, dict):
            continue

        for team_name in selected_teams:
            team_payload = teams_payload.get(team_name)
            if not isinstance(team_payload, dict):
                continue
            profiles = team_payload.get("profiles")
            if not isinstance(profiles, dict):
                continue
            metrics_payload = profiles.get(profile)
            if not isinstance(metrics_payload, dict):
                continue

            row: dict[str, Any] = {
                "Snapshot": label,
                "Snapshot Order": index,
                "Snapshot Timestamp": snapshot_sort_timestamp(snapshot_payload).isoformat(),
                "Event": str(snapshot_payload.get("event_name", "")).strip(),
                "Session": session_name,
                "Team": team_name,
            }

            metric_values: list[float] = []
            for payload_key, label_name in _TEAM_RADAR_METRICS:
                metric_value = _coerce_unit_metric(metrics_payload.get(payload_key))
                if metric_value is None:
                    continue
                row[label_name] = metric_value
                metric_values.append(metric_value)

            overall_pace = _coerce_unit_metric(metrics_payload.get("overall_pace"))
            if overall_pace is not None:
                row["Overall Pace"] = overall_pace
            metric_count = len(metric_values)
            if metric_count:
                row["Overall"] = float(sum(metric_values) / metric_count)
                row["Metric Count"] = metric_count
                row["Metric Coverage"] = float(metric_count / len(_TEAM_RADAR_METRICS))
            if metric_values or overall_pace is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _ordered_snapshot_labels(history_df: pd.DataFrame) -> list[str]:
    """Return unique snapshot labels in the exact chronological order used by the chart."""
    if history_df.empty:
        return []

    required_columns = {"Snapshot", "Snapshot Order", "Snapshot Timestamp"}
    if not required_columns.issubset(set(history_df.columns)):
        return []

    ordered = (
        history_df[["Snapshot", "Snapshot Order", "Snapshot Timestamp"]]
        .drop_duplicates(subset=["Snapshot"])
        .sort_values(["Snapshot Timestamp", "Snapshot Order", "Snapshot"])
    )
    return [str(label) for label in ordered["Snapshot"]]


def _build_development_summary_table(history_df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Summarize latest level and change for the selected development metric."""
    if history_df.empty or metric_label not in history_df.columns:
        return pd.DataFrame()

    metric_frame = history_df[["Snapshot Order", "Snapshot", "Team", metric_label]].dropna()
    if metric_frame.empty:
        return pd.DataFrame()

    ordered = metric_frame.sort_values(["Team", "Snapshot Order"])
    rows: list[dict[str, Any]] = []
    for team_name, team_frame in ordered.groupby("Team"):
        first_row = team_frame.iloc[0]
        latest_row = team_frame.iloc[-1]
        rows.append(
            {
                "Team": team_name,
                "Latest": round(float(latest_row[metric_label]) * 100.0, 1),
                "Change": round(
                    (float(latest_row[metric_label]) - float(first_row[metric_label])) * 100.0,
                    1,
                ),
                "First Snapshot": str(first_row["Snapshot"]),
                "Latest Snapshot": str(latest_row["Snapshot"]),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("Latest", ascending=False).reset_index(drop=True)


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
        "qualifying, and race snapshots while skipping sprint-only checkpoints."
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

    snapshots = _load_team_snapshot_history(year)
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
    metric_frame = history_df[metric_frame_columns].dropna(subset=[metric_label])
    if metric_frame.empty:
        st.info(f"No stored `{metric_label}` values are available for this selection yet.")
        return
    category_order = _ordered_snapshot_labels(history_df)

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
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["20", "40", "60", "80", "100"],
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

    snapshots = _load_team_snapshot_history(year)
    latest_snapshot = _latest_snapshot_payload(snapshots)
    latest_snapshot_label = _snapshot_label(latest_snapshot) if latest_snapshot else ""

    teams_payload = _build_latest_snapshot_comparison_payload(
        base_teams_payload=base_teams_payload if isinstance(base_teams_payload, dict) else {},
        latest_snapshot=latest_snapshot,
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

    comparison_df, neutral_fallbacks = _build_team_comparison_dataframe(
        teams_payload=teams_payload,
        selected_teams=teams_with_signal,
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
        "Radar Minus Prior",
    ]
    for column in percent_cols:
        display_df[column] = (display_df[column].astype(float) * 100.0).round(1)
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
        "Profile pace/radar come from the latest synced non-sprint snapshot; "
        "Season Prior Strength is a separate baseline signal."
    )
    st.caption(
        f"Source: {source_label or f'`{characteristics_path}`'} | profile=`{profile}` | "
        "values are normalized (0-100, higher is better)."
    )
    if neutral_fallbacks > 0:
        st.caption(
            f"{neutral_fallbacks} missing metric(s) were filled with neutral value 50.0 for comparability."
        )

    _render_development_history_section(
        year=year,
        selected_teams=selected_teams,
        profile=profile,
        characteristics_payload=payload or {},
    )
