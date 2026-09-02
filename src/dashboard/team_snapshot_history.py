"""Snapshot-history helpers for team comparison views."""

from copy import deepcopy
from math import isfinite
from typing import Any

import pandas as pd

from src.dashboard.team_radar import (
    _DISPLAY_SCORE_CEILING,
    _DISPLAY_SCORE_FLOOR,
    _RAW_METRIC_FIELDS,
    _TEAM_RADAR_METRICS,
    _build_raw_metric_display_scale,
    _build_tire_deg_display_scale,
    _build_top_speed_display_scale,
    _canonicalize_teams_payload_for_comparison,
    _coerce_unit_metric,
    _resolve_profile_display_metric_value,
    _resolve_profile_metrics,
    _resolve_profile_overall_pace_display_value,
    _uses_placeholder_braking,
    _uses_same_event_average_fallback,
)
from src.utils.car_snapshot_history import session_order_index, snapshot_sort_timestamp
from src.utils.team_mapping import canonicalize_team, map_team_to_characteristics

_PROFILE_DEVELOPMENT_PACE_LABELS = {
    "short_run": "Qualifying Pace",
    "long_run": "Race Pace",
}
_RADAR_AVERAGE_LABEL = "Radar Average"
# Columns that describe a point rather than score it, so smoothing and rescaling skip them.
_NON_METRIC_HISTORY_COLUMNS = frozenset(
    {
        "Snapshot",
        "Session",
        "Team",
        "Event",
        "Has Data",
        "Snapshot Order",
        "Snapshot Timestamp",
        "Metric Count",
        "Metric Coverage",
    }
)
# Suffix flagging a value the session itself did not measure, supplied by its weekend.
_WINDOW_FILLED_SUFFIX = " Window Filled"


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


def _comparison_display_team_name(team_name: str, team_payload: dict[str, Any] | None) -> str:
    """Add an asterisk only when the latest comparison scores are approximated."""
    return f"{team_name}*" if _uses_same_event_average_fallback(team_payload) else team_name


def _is_comparison_snapshot_session(session_name: str) -> bool:
    """Return True for stored snapshots that belong in comparison charts and tables."""
    normalized = "".join(ch for ch in str(session_name).strip().upper() if ch.isalnum())
    if not normalized:
        return False
    # Sprint weekends replace FP2/FP3 with SQ/Sprint, so excluding them left those
    # weekends with only FP1 until Qualifying ran.
    return session_order_index(normalized) in {1, 2, 3, 4, 5, 6, 7}


def _is_history_chart_snapshot_session(session_name: str) -> bool:
    """Return True for stored snapshots that should appear in the development chart."""
    normalized = "".join(ch for ch in str(session_name).strip().upper() if ch.isalnum())
    if not normalized:
        return False
    return session_order_index(normalized) in {1, 2, 3, 4, 5, 6, 7}


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
    snapshot_history: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build comparison input from the newest snapshot while keeping season priors."""
    if not isinstance(latest_snapshot, dict):
        return {}

    snapshot_teams_payload = latest_snapshot.get("teams")
    if not isinstance(snapshot_teams_payload, dict):
        return {}

    history = snapshot_history or []
    canonical_base_payload = _canonicalize_teams_payload_for_comparison(base_teams_payload)
    merged_payload: dict[str, dict[str, Any]] = {}
    candidate_team_names: set[str] = set(canonical_base_payload.keys())
    for raw_team_name in snapshot_teams_payload:
        mapped_name = map_team_to_characteristics(str(raw_team_name))
        display_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )
        candidate_team_names.add(display_name)

    for display_name in sorted(candidate_team_names):
        profiles_payload = _resolve_snapshot_team_profiles(latest_snapshot, display_name)
        fallback_source = ""
        if not profiles_payload:
            profiles_payload = _resolve_same_event_profile_average_fallback(
                snapshot_history=history,
                latest_snapshot=latest_snapshot,
                team_name=display_name,
            )
            if profiles_payload:
                fallback_source = "same_event_average"
        if not profiles_payload:
            continue

        normalized_profiles = deepcopy(profiles_payload)
        tire_deg_fallback, tire_deg_fallback_source = _resolve_latest_tire_deg_fallback(
            snapshot_history=history,
            latest_snapshot=latest_snapshot,
            team_name=display_name,
        )
        if tire_deg_fallback:
            _apply_profile_tire_deg_fallbacks(
                normalized_profiles, tire_deg_fallback, tire_deg_fallback_source
            )

        same_event_braking_fallback = _resolve_same_event_metric_average_fallback(
            snapshot_history=history,
            latest_snapshot=latest_snapshot,
            team_name=display_name,
            metric_name="braking_performance",
        )
        latest_braking_fallback = _resolve_latest_metric_fallback(
            snapshot_history=history,
            latest_snapshot=latest_snapshot,
            team_name=display_name,
            metric_name="braking_performance",
        )
        _apply_profile_braking_fallbacks(
            normalized_profiles,
            same_event_braking_fallback=same_event_braking_fallback,
            latest_braking_fallback=latest_braking_fallback,
        )

        team_payload: dict[str, Any] = {"testing_characteristics_profiles": normalized_profiles}

        balanced_profile = normalized_profiles.get("balanced")
        if isinstance(balanced_profile, dict):
            team_payload["testing_characteristics"] = deepcopy(balanced_profile)
        if fallback_source:
            team_payload["comparison_fallback_source"] = fallback_source
            team_payload["comparison_fallback_label"] = (
                f"{str(latest_snapshot.get('event_name', '')).strip()} weekend average"
            )

        base_team_payload = canonical_base_payload.get(display_name)
        if isinstance(base_team_payload, dict):
            overall_performance = _coerce_unit_metric(base_team_payload.get("overall_performance"))
            if overall_performance is not None:
                team_payload["overall_performance"] = overall_performance

        merged_payload[display_name] = team_payload

    return _canonicalize_teams_payload_for_comparison(merged_payload)


def _resolve_snapshot_team_profiles(
    snapshot_payload: dict[str, Any],
    team_name: str,
) -> dict[str, Any]:
    """Return stored profile payloads for one team label inside a snapshot."""
    teams_payload = snapshot_payload.get("teams")
    if not isinstance(teams_payload, dict):
        return {}

    target_canonical = canonicalize_team(team_name)
    normalized_target = str(team_name).strip()
    for raw_team_name, raw_team_payload in teams_payload.items():
        if not isinstance(raw_team_payload, dict):
            continue
        mapped_name = map_team_to_characteristics(str(raw_team_name))
        display_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )
        if (
            display_name != normalized_target
            and canonicalize_team(display_name) != target_canonical
        ):
            continue
        profiles = raw_team_payload.get("profiles")
        return profiles if isinstance(profiles, dict) else {}

    return {}


def _snapshot_identity(snapshot_payload: dict[str, Any]) -> tuple[str, str]:
    """Build a stable `(event_name, session_name)` identity for one snapshot payload."""
    return (
        str(snapshot_payload.get("event_name", "")).strip(),
        str(snapshot_payload.get("session_name", "")).strip(),
    )


def _average_snapshot_profile_metrics(profile_samples: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric profile metrics across multiple stored session snapshots."""
    metric_samples: dict[str, list[float]] = {}
    for metrics_payload in profile_samples:
        if not isinstance(metrics_payload, dict):
            continue
        for metric_name, raw_value in metrics_payload.items():
            if not isinstance(raw_value, int | float):
                continue
            numeric_value = float(raw_value)
            if not isfinite(numeric_value):
                continue
            metric_samples.setdefault(str(metric_name), []).append(numeric_value)

    averaged_metrics: dict[str, float] = {}
    for metric_name, values in metric_samples.items():
        if not values:
            continue
        averaged_metrics[metric_name] = round(sum(values) / len(values), 4)
    return averaged_metrics


def _resolve_same_event_profile_average_fallback(
    *,
    snapshot_history: list[dict[str, Any]],
    latest_snapshot: dict[str, Any],
    team_name: str,
) -> dict[str, dict[str, float]]:
    """
    Build a same-weekend profile fallback for teams missing from the latest snapshot.

    When a team has no usable latest-session sample, for example after a double
    retirement, the comparison chart should keep the team visible without
    pretending that the missing race snapshot contained valid telemetry. We use
    the average of the earlier snapshots from the same event as a conservative
    weekend-level fallback.
    """
    latest_event = str(latest_snapshot.get("event_name", "")).strip()
    latest_identity = _snapshot_identity(latest_snapshot)
    if not latest_event:
        return {}

    profile_samples: dict[str, list[dict[str, Any]]] = {}
    for snapshot_payload in snapshot_history:
        snapshot_identity = _snapshot_identity(snapshot_payload)
        if snapshot_identity == latest_identity:
            break
        if str(snapshot_payload.get("event_name", "")).strip() != latest_event:
            continue
        team_profiles = _resolve_snapshot_team_profiles(snapshot_payload, team_name)
        if not team_profiles:
            continue
        for profile_name, metrics_payload in team_profiles.items():
            if not isinstance(metrics_payload, dict):
                continue
            profile_samples.setdefault(str(profile_name), []).append(metrics_payload)

    averaged_profiles: dict[str, dict[str, float]] = {}
    for profile_name, samples in profile_samples.items():
        averaged_metrics = _average_snapshot_profile_metrics(samples)
        if averaged_metrics:
            averaged_profiles[profile_name] = averaged_metrics
    return averaged_profiles


def _resolve_same_event_metric_average_fallback(
    *,
    snapshot_history: list[dict[str, Any]],
    latest_snapshot: dict[str, Any],
    team_name: str,
    metric_name: str,
) -> dict[str, float]:
    """Average one metric across earlier snapshots from the same event."""
    latest_event = str(latest_snapshot.get("event_name", "")).strip()
    latest_identity = _snapshot_identity(latest_snapshot)
    if not latest_event:
        return {}

    profile_samples: dict[str, list[float]] = {}
    for snapshot_payload in snapshot_history:
        snapshot_identity = _snapshot_identity(snapshot_payload)
        if snapshot_identity == latest_identity:
            break
        if str(snapshot_payload.get("event_name", "")).strip() != latest_event:
            continue

        team_profiles = _resolve_snapshot_team_profiles(snapshot_payload, team_name)
        if not team_profiles:
            continue

        for profile_name, metrics_payload in team_profiles.items():
            if not isinstance(metrics_payload, dict):
                continue
            metric_value = _resolve_usable_history_metric_value(
                metrics_payload,
                metric_name=metric_name,
            )
            if metric_value is None:
                continue
            profile_samples.setdefault(str(profile_name), []).append(metric_value)

    averaged_metrics: dict[str, float] = {}
    for profile_name, values in profile_samples.items():
        if not values:
            continue
        averaged_metrics[profile_name] = round(float(sum(values) / len(values)), 4)
    return averaged_metrics


def _resolve_latest_metric_fallback(
    *,
    snapshot_history: list[dict[str, Any]],
    latest_snapshot: dict[str, Any],
    team_name: str,
    metric_name: str,
) -> dict[str, float]:
    """Resolve the newest previously stored metric values for one team."""
    latest_identity = _snapshot_identity(latest_snapshot)
    seen_latest = False

    for snapshot_payload in reversed(snapshot_history):
        snapshot_identity = _snapshot_identity(snapshot_payload)
        if not seen_latest:
            if snapshot_identity != latest_identity:
                continue
            seen_latest = True
            continue

        team_profiles = _resolve_snapshot_team_profiles(snapshot_payload, team_name)
        if not team_profiles:
            continue

        resolved: dict[str, float] = {}
        for profile_name, metrics_payload in team_profiles.items():
            if not isinstance(metrics_payload, dict):
                continue
            metric_value = _resolve_usable_history_metric_value(
                metrics_payload,
                metric_name=metric_name,
            )
            if metric_value is None:
                continue
            resolved[str(profile_name)] = metric_value

        if resolved:
            return resolved

    return {}


def _resolve_usable_history_metric_value(
    metrics_payload: dict[str, Any],
    *,
    metric_name: str,
) -> float | None:
    """
    Return a fallback metric only when the stored history looks trustworthy.

    Braking needs extra care because older snapshot artifacts copied the slow
    corner score into `braking_performance`. Using those snapshots as fallback
    sources would quietly preserve the exact bug we are trying to avoid.
    """
    if metric_name == "braking_performance" and _uses_placeholder_braking(metrics_payload):
        return None
    return _coerce_unit_metric(metrics_payload.get(metric_name))


def _resolve_latest_tire_deg_fallback(
    *,
    snapshot_history: list[dict[str, Any]],
    latest_snapshot: dict[str, Any],
    team_name: str,
) -> tuple[dict[str, float], str]:
    """
    Resolve the newest usable tire-deg payload for a team from snapshot history.

    The comparison chart can land on sessions like qualifying or sprint
    qualifying where a real long-run deg signal does not exist. In that case,
    fall back to the latest previously known long-run signal for the same team
    instead of filling the chart with a neutral 0.5.

    Returns the resolved metrics and the label of the session they came from, so
    the comparison view can say on screen that the value is carried forward.
    """
    fallback_profiles = ("long_run", "balanced", "short_run")
    latest_identity = _snapshot_identity(latest_snapshot)
    seen_latest = False

    for snapshot_payload in reversed(snapshot_history):
        snapshot_identity = _snapshot_identity(snapshot_payload)
        if not seen_latest:
            if snapshot_identity != latest_identity:
                continue
            seen_latest = True
            continue

        team_profiles = _resolve_snapshot_team_profiles(snapshot_payload, team_name)
        if not team_profiles:
            continue

        for profile_name in fallback_profiles:
            metrics_payload = team_profiles.get(profile_name)
            if not isinstance(metrics_payload, dict):
                continue

            tire_deg_performance = _coerce_unit_metric(metrics_payload.get("tire_deg_performance"))
            tire_deg_slope = metrics_payload.get("tire_deg_slope")
            if not isinstance(tire_deg_slope, int | float):
                tire_deg_slope = None

            if tire_deg_performance is None and tire_deg_slope is None:
                continue

            resolved: dict[str, float] = {}
            if tire_deg_performance is not None:
                resolved["tire_deg_performance"] = tire_deg_performance
            if tire_deg_slope is not None:
                resolved["tire_deg_slope"] = float(tire_deg_slope)
            return resolved, _snapshot_label(snapshot_payload)

    return {}, ""


def _apply_profile_tire_deg_fallbacks(
    profiles_payload: dict[str, Any],
    tire_deg_fallback: dict[str, float],
    source_label: str = "",
) -> None:
    """Fill missing tire-deg fields and identify their historical basis."""
    if not tire_deg_fallback:
        return

    fallback_tire_perf = tire_deg_fallback.get("tire_deg_performance")
    fallback_tire_slope = tire_deg_fallback.get("tire_deg_slope")

    for metrics_payload in profiles_payload.values():
        if not isinstance(metrics_payload, dict):
            continue
        fallback_applied = False
        if (
            _coerce_unit_metric(metrics_payload.get("tire_deg_performance")) is None
            and fallback_tire_perf is not None
        ):
            metrics_payload["tire_deg_performance"] = fallback_tire_perf
            fallback_applied = True
        raw_slope = metrics_payload.get("tire_deg_slope")
        if not isinstance(raw_slope, int | float) and fallback_tire_slope is not None:
            metrics_payload["tire_deg_slope"] = fallback_tire_slope
            fallback_applied = True
        if fallback_applied:
            metrics_payload["_tire_deg_history_fallback"] = True
            if source_label:
                metrics_payload["_tire_deg_history_fallback_source"] = source_label


def _apply_profile_braking_fallbacks(
    profiles_payload: dict[str, Any],
    *,
    same_event_braking_fallback: dict[str, float],
    latest_braking_fallback: dict[str, float],
) -> None:
    """
    Fill missing or placeholder braking metrics from prior session history.

    Prefer a same-weekend average when available because it preserves the local
    event context. If the weekend has no usable earlier braking sample, fall
    back to the most recent stored session as a proxy.
    """
    if not same_event_braking_fallback and not latest_braking_fallback:
        return

    for profile_name, metrics_payload in profiles_payload.items():
        if not isinstance(metrics_payload, dict):
            continue
        if not _uses_placeholder_braking(metrics_payload):
            continue

        fallback_value = same_event_braking_fallback.get(str(profile_name))
        if fallback_value is None:
            fallback_value = latest_braking_fallback.get(str(profile_name))
        if fallback_value is None and str(profile_name) != "balanced":
            fallback_value = same_event_braking_fallback.get("balanced")
        if fallback_value is None and str(profile_name) != "balanced":
            fallback_value = latest_braking_fallback.get("balanced")
        if fallback_value is None:
            continue

        metrics_payload["braking_performance"] = float(fallback_value)


def _build_snapshot_history_dataframe(
    snapshots: list[dict[str, Any]],
    selected_teams: list[str],
    profile: str,
) -> pd.DataFrame:
    """Flatten stored snapshots into one line-chart-friendly dataframe."""
    rows: list[dict[str, Any]] = []

    for index, snapshot_payload in enumerate(snapshots):
        session_name = str(snapshot_payload.get("session_name", "")).strip()
        if not _is_history_chart_snapshot_session(session_name):
            continue

        label = _snapshot_label(snapshot_payload)
        teams_payload = snapshot_payload.get("teams")
        if not isinstance(teams_payload, dict):
            continue

        snapshot_team_payload: dict[str, dict[str, Any]] = {}
        for raw_team_name, raw_team_payload in teams_payload.items():
            if not isinstance(raw_team_payload, dict):
                continue
            profiles = raw_team_payload.get("profiles")
            if not isinstance(profiles, dict):
                continue
            mapped_name = map_team_to_characteristics(str(raw_team_name))
            display_name = (
                mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
            )
            snapshot_team_payload[display_name] = {"testing_characteristics_profiles": profiles}

        tire_deg_display_scale = _build_tire_deg_display_scale(snapshot_team_payload, profile)
        top_speed_display_scale = _build_top_speed_display_scale(snapshot_team_payload, profile)
        raw_metric_display_scales = {
            metric_key: _build_raw_metric_display_scale(
                snapshot_team_payload,
                profile,
                raw_metric_key=raw_metric_key,
                min_padding=min_padding,
            )
            for metric_key, (
                raw_metric_key,
                _higher_is_better,
                min_padding,
            ) in _RAW_METRIC_FIELDS.items()
        }

        for team_name in selected_teams:
            row: dict[str, Any] = {
                "Snapshot": label,
                "Snapshot Order": index,
                "Snapshot Timestamp": snapshot_sort_timestamp(snapshot_payload).isoformat(),
                "Event": str(snapshot_payload.get("event_name", "")).strip(),
                "Session": session_name,
                "Team": team_name,
                "Has Data": False,
            }
            team_payload = snapshot_team_payload.get(team_name)
            if not isinstance(team_payload, dict):
                rows.append(row)
                continue
            metrics_payload = _resolve_profile_metrics(team_payload, profile)
            if not isinstance(metrics_payload, dict):
                rows.append(row)
                continue

            metric_values: list[float] = []
            for payload_key, label_name in _TEAM_RADAR_METRICS:
                metric_value = _resolve_profile_display_metric_value(
                    payload_key,
                    metrics_payload,
                    tire_deg_display_scale=tire_deg_display_scale,
                    top_speed_display_scale=top_speed_display_scale,
                    raw_metric_display_scales=raw_metric_display_scales,
                    skip_placeholder_braking=True,
                )
                if metric_value is None:
                    continue
                row[label_name] = metric_value
                metric_values.append(metric_value)

            overall_pace = _resolve_profile_overall_pace_display_value(
                metrics_payload,
                raw_metric_display_scales=raw_metric_display_scales,
            )
            if overall_pace is not None:
                row["Overall Pace"] = overall_pace
            for profile_name, label_name in _PROFILE_DEVELOPMENT_PACE_LABELS.items():
                profile_pace = _resolve_development_profile_pace_value(
                    snapshot_team_payload=snapshot_team_payload,
                    team_payload=team_payload,
                    profile_name=profile_name,
                )
                if profile_pace is not None:
                    row[label_name] = profile_pace
            metric_count = len(metric_values)
            if metric_count:
                row[_RADAR_AVERAGE_LABEL] = float(sum(metric_values) / metric_count)
                row["Metric Count"] = metric_count
                row["Metric Coverage"] = float(metric_count / len(_TEAM_RADAR_METRICS))
            row["Has Data"] = bool(metric_values or overall_pace is not None)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _resolve_development_profile_pace_value(
    *,
    snapshot_team_payload: dict[str, dict[str, Any]],
    team_payload: dict[str, Any],
    profile_name: str,
) -> float | None:
    """Resolve one named profile pace so history charts can show quali and race trends."""
    metrics_payload = _resolve_profile_metrics(team_payload, profile_name)
    if not isinstance(metrics_payload, dict) or not metrics_payload:
        return None

    raw_metric_key, _higher_is_better, min_padding = _RAW_METRIC_FIELDS["overall_pace"]
    raw_metric_display_scales = {
        "overall_pace": _build_raw_metric_display_scale(
            snapshot_team_payload,
            profile_name,
            raw_metric_key=raw_metric_key,
            min_padding=min_padding,
        )
    }
    return _resolve_profile_overall_pace_display_value(
        metrics_payload,
        raw_metric_display_scales=raw_metric_display_scales,
    )


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


def _history_metric_columns(history_df: pd.DataFrame) -> list[str]:
    """Return the scored columns of a history frame, ignoring its descriptive ones."""
    return [
        column_name
        for column_name in history_df.columns
        if column_name not in _NON_METRIC_HISTORY_COLUMNS
        and not str(column_name).endswith(_WINDOW_FILLED_SUFFIX)
        and pd.api.types.is_numeric_dtype(history_df[column_name])
    ]


def _smooth_development_history_dataframe(
    history_df: pd.DataFrame,
    *,
    window: int = 5,
) -> pd.DataFrame:
    """
    Return a per-team smoothed copy of the development-history dataframe.

    The window never spans two race weekends. Sessions at different circuits
    measure different things, so each event is smoothed on its own and the line
    breaks between weekends rather than blending them.

    It is five wide because a race weekend holds about five sessions. A narrower
    window leaves the first and last session of every weekend averaged over two
    points while the middle gets three, and that uneven smoothing is visible as
    a sawtooth on the season chart.

    Every radar metric also gets a `<label> Window Filled` flag marking a value the
    session itself never measured, so the view can say where it came from.
    """
    if history_df.empty or "Team" not in history_df.columns:
        return history_df.copy()

    smoothed = history_df.copy()
    metric_columns = _history_metric_columns(smoothed)
    if not metric_columns:
        return smoothed

    sort_columns = [
        column_name
        for column_name in ("Team", "Snapshot Timestamp", "Snapshot Order", "Snapshot")
        if column_name in smoothed.columns
    ]
    if sort_columns:
        smoothed = smoothed.sort_values(sort_columns).reset_index(drop=True)

    radar_labels = {label for _key, label in _TEAM_RADAR_METRICS} & set(metric_columns)
    for label in radar_labels:
        smoothed[f"{label}{_WINDOW_FILLED_SUFFIX}"] = False
    # A team missing from a session stays missing; the window may only speak for a
    # session the team actually ran.
    if "Has Data" in smoothed.columns:
        present = smoothed["Has Data"].fillna(False).astype(bool)
    else:
        present = pd.Series(True, index=smoothed.index)

    group_columns = ["Team"] + (["Event"] if "Event" in smoothed.columns else [])
    for _group_key, group_index in smoothed.groupby(
        group_columns, dropna=False, sort=False
    ).groups.items():
        group_present = present.loc[group_index]
        for column_name in metric_columns:
            original_values = smoothed.loc[group_index, column_name]
            rolling_mean = original_values.rolling(window=window, center=True, min_periods=1).mean()
            if column_name in radar_labels:
                # Qualifying measures no tire degradation, so the session's own reading
                # is absent while the weekend around it has one. Letting the window
                # speak for that session is the whole point of averaging by weekend --
                # and it keeps the radar and the season chart on the same number.
                keep = group_present & rolling_mean.notna()
                smoothed.loc[group_index, f"{column_name}{_WINDOW_FILLED_SUFFIX}"] = (
                    keep & original_values.isna()
                ).fillna(False)
            else:
                keep = original_values.notna()
            smoothed.loc[group_index, column_name] = rolling_mean.where(keep)

    return smoothed


def _rescale_history_dataframe_per_session(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stretch each session's smoothed scores back across the display range.

    Averaging pulls every team toward the middle, so a smoothed session spans a
    narrow band and the gaps between teams become hard to read. Re-normalising per
    session puts the best car in that session back at 100 and the slowest at 10,
    which is what makes the differences legible.

    The cost is that a team leading a metric reads exactly 100 every time, so a
    dominant car draws a flat line along the ceiling, and a session's score cannot
    be compared with another session's in absolute terms. The five-wide smoothing
    window absorbs most of the extra noise this would otherwise re-expand.

    `Radar Average` is not rescaled here: it is recomputed from the rescaled
    metrics afterwards so the number always equals the mean of the six spokes
    actually drawn.
    """
    if history_df.empty or "Snapshot" not in history_df.columns:
        return history_df.copy()

    rescaled = history_df.copy()
    metric_columns = [
        column_name
        for column_name in _history_metric_columns(rescaled)
        if column_name != _RADAR_AVERAGE_LABEL
    ]
    if not metric_columns:
        return rescaled

    span = _DISPLAY_SCORE_CEILING - _DISPLAY_SCORE_FLOOR
    for _snapshot_label_value, session_index in rescaled.groupby(
        "Snapshot", dropna=False, sort=False
    ).groups.items():
        for column_name in metric_columns:
            session_values = rescaled.loc[session_index, column_name]
            scored = session_values.dropna()
            # One scored team has nothing to stretch against, and an identical
            # field would divide by zero. Leave both untouched.
            if len(scored) < 2:
                continue
            lowest = float(scored.min())
            highest = float(scored.max())
            if highest <= lowest:
                continue
            rescaled.loc[session_index, column_name] = (
                _DISPLAY_SCORE_FLOOR + ((session_values - lowest) / (highest - lowest)) * span
            )

    return rescaled


def _recompute_history_composites(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild Radar Average and its coverage from the metrics as they now stand.

    Both are computed before smoothing in the original frame, so after smoothing
    and rescaling they describe values that no longer exist. Recomputing keeps the
    hover's "n/6 metrics" honest about the point it is attached to.
    """
    if history_df.empty:
        return history_df.copy()

    recomputed = history_df.copy()
    radar_labels = [label for _key, label in _TEAM_RADAR_METRICS if label in recomputed.columns]
    if not radar_labels:
        return recomputed

    radar_frame = recomputed[radar_labels]
    metric_count = radar_frame.count(axis=1)
    recomputed[_RADAR_AVERAGE_LABEL] = radar_frame.mean(axis=1, skipna=True).where(metric_count > 0)
    recomputed["Metric Count"] = metric_count.astype(float).where(metric_count > 0)
    recomputed["Metric Coverage"] = (metric_count / len(_TEAM_RADAR_METRICS)).where(
        metric_count > 0
    )
    return recomputed


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
