"""Team-comparison data loading and rendering helpers."""

import json
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.utils import config_loader
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


def _render_team_comparison_section(year: int = 2026) -> None:
    """Render profile-aware team comparison chart and metric table."""
    st.subheader("Testing/Practice Snapshot")

    payload, characteristics_path = _load_team_characteristics_payload(year)
    if not payload:
        st.info(f"Team characteristics unavailable at `{characteristics_path}`.")
        return

    raw_teams_payload = payload.get("teams")
    if not isinstance(raw_teams_payload, dict) or not raw_teams_payload:
        st.info("No team characteristics found for comparison.")
        return
    teams_payload = _canonicalize_teams_payload_for_comparison(raw_teams_payload)
    if not teams_payload:
        st.info("No team characteristics found for comparison.")
        return

    profile_names = _collect_profile_names(teams_payload)
    if not profile_names:
        st.info(
            "No testing/practice profile metrics are available yet for this season. "
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
            "Selected teams do not have testing/practice profile metrics for this profile yet. "
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
            "Overall Pace": "Profile Pace (Testing)",
            "Radar Composite": "Radar Composite (6 Metrics)",
            "Overall Performance": "Season Prior Strength",
            "Radar Minus Prior": "Radar - Prior Gap",
        }
    )

    st.dataframe(display_df, hide_index=True, width="stretch")
    st.caption(
        "Profile pace/radar come from selected testing-practice profile; "
        "Season Prior Strength is a separate baseline signal."
    )
    st.caption(
        f"Source: `{characteristics_path}` | profile=`{profile}` | "
        "values are normalized (0-100, higher is better)."
    )
    if neutral_fallbacks > 0:
        st.caption(
            f"{neutral_fallbacks} missing metric(s) were filled with neutral value 50.0 for comparability."
        )
