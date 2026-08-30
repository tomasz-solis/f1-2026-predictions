"""History-aware comparison fallbacks for the team-comparison dashboard."""

from typing import Any

import pandas as pd

from src.dashboard.team_radar import (
    _TEAM_RADAR_METRICS,
    _build_team_comparison_missing_column_map,
    _comparison_session_display_columns,
)
from src.dashboard.team_snapshot_history import (
    _build_snapshot_history_dataframe,
    _snapshot_identity,
)


def _build_same_event_display_metric_fallbacks(
    *,
    snapshot_history: list[dict[str, Any]],
    latest_snapshot: dict[str, Any] | None,
    teams_payload: dict[str, Any],
    selected_teams: list[str],
    profile: str,
) -> dict[str, dict[str, float]]:
    """
    Average earlier same-event display scores for teams missing from the latest snapshot.

    A starred comparison row should stay aligned with the Relative Performance Over Time
    chart. That means we need to average the already-rendered event scores, not
    raw session seconds from practice, sprint, and qualifying mixed together.
    """
    if not isinstance(latest_snapshot, dict) or not selected_teams or not snapshot_history:
        return {}

    fallback_teams = [
        team_name
        for team_name in selected_teams
        if str(teams_payload.get(team_name, {}).get("comparison_fallback_source", "")).strip()
        == "same_event_average"
    ]
    if not fallback_teams:
        return {}

    history_df = _build_snapshot_history_dataframe(
        snapshots=snapshot_history,
        selected_teams=selected_teams,
        profile=profile,
    )
    if history_df.empty:
        return {}

    latest_event = str(latest_snapshot.get("event_name", "")).strip()
    latest_session = str(latest_snapshot.get("session_name", "")).strip()
    if not latest_event or not latest_session:
        return {}

    latest_rows = history_df[
        (history_df["Event"] == latest_event) & (history_df["Session"] == latest_session)
    ]
    if latest_rows.empty:
        return {}

    latest_order = int(latest_rows["Snapshot Order"].max())
    event_history = history_df[
        (history_df["Event"] == latest_event)
        & (history_df["Snapshot Order"] < latest_order)
        & history_df["Has Data"].fillna(False)
    ]
    if event_history.empty:
        return {}

    display_columns = list(_comparison_session_display_columns())
    fallback_rows: dict[str, dict[str, float]] = {}
    for team_name in fallback_teams:
        team_history = event_history[event_history["Team"] == team_name]
        if team_history.empty:
            continue

        averaged_scores: dict[str, float] = {}
        for column in display_columns:
            if column not in team_history.columns:
                continue
            series = team_history[column].dropna()
            if series.empty:
                continue
            averaged_scores[column] = float(series.mean())

        if averaged_scores:
            fallback_rows[team_name] = averaged_scores

    return fallback_rows


def _build_latest_reliable_display_metric_fallbacks(
    *,
    snapshot_history: list[dict[str, Any]],
    latest_snapshot: dict[str, Any] | None,
    selected_teams: list[str],
    profile: str,
) -> dict[str, dict[str, float]]:
    """Return the newest trustworthy historical display score for each team/metric."""
    if not selected_teams or not snapshot_history:
        return {}

    history_df = _build_snapshot_history_dataframe(
        snapshots=snapshot_history,
        selected_teams=selected_teams,
        profile=profile,
    )
    if history_df.empty:
        return {}

    latest_order: int | None = None
    latest_identity = (
        _snapshot_identity(latest_snapshot) if isinstance(latest_snapshot, dict) else ""
    )
    for index, snapshot_payload in enumerate(snapshot_history):
        session_name = str(snapshot_payload.get("session_name", "")).strip()
        normalized = "".join(ch for ch in session_name.strip().upper() if ch.isalnum())
        if not normalized or normalized not in {"FP1", "FP2", "FP3", "SPRINT", "SQ", "Q", "R"}:
            continue
        if latest_identity and _snapshot_identity(snapshot_payload) == latest_identity:
            latest_order = index
    if latest_order is None:
        latest_order = int(history_df["Snapshot Order"].max()) + 1

    prior_history = history_df[
        (history_df["Snapshot Order"] < latest_order) & history_df["Has Data"].fillna(False)
    ]
    if prior_history.empty:
        return {}

    display_columns = list(_comparison_session_display_columns())
    fallback_rows: dict[str, dict[str, float]] = {}
    for team_name in selected_teams:
        team_history = prior_history[prior_history["Team"] == team_name].sort_values(
            "Snapshot Order"
        )
        if team_history.empty:
            continue

        latest_scores: dict[str, float] = {}
        for column in display_columns:
            if column not in team_history.columns:
                continue
            series = team_history[column].dropna()
            if series.empty:
                continue
            latest_scores[column] = float(series.iloc[-1])

        if latest_scores:
            fallback_rows[team_name] = latest_scores

    return fallback_rows


def _apply_display_metric_fallbacks(
    comparison_df: pd.DataFrame,
    *,
    teams_payload: dict[str, Any],
    selected_teams: list[str],
    profile: str,
    same_event_display_scores: dict[str, dict[str, float]],
    latest_reliable_display_scores: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, int]:
    """Fill comparison gaps from same-event averages first, then reliable history."""
    if comparison_df.empty:
        return comparison_df, 0

    radar_labels = [label for _, label in _TEAM_RADAR_METRICS]
    missing_column_map = _build_team_comparison_missing_column_map(
        teams_payload=teams_payload,
        selected_teams=selected_teams,
        profile=profile,
    )
    updated_rows: list[dict[str, Any]] = []
    unresolved_missing_count = 0
    for row in comparison_df.to_dict(orient="records"):
        team_name = str(row.get("Team", "")).strip()

        updated_row = dict(row)
        same_event_scores = same_event_display_scores.get(team_name, {})
        if str(teams_payload.get(team_name, {}).get("comparison_fallback_source", "")).strip() == (
            "same_event_average"
        ):
            for column, value in same_event_scores.items():
                updated_row[column] = float(value)

        missing_columns = missing_column_map.get(team_name, set())
        latest_scores = latest_reliable_display_scores.get(team_name, {})
        for column in missing_columns:
            fallback_value = same_event_scores.get(column)
            if fallback_value is None:
                fallback_value = latest_scores.get(column)
            if fallback_value is None:
                unresolved_missing_count += 1
                continue
            updated_row[column] = float(fallback_value)

        radar_values = [float(updated_row[label]) for label in radar_labels]
        radar_composite = float(sum(radar_values) / len(radar_values))
        updated_row["Radar Composite"] = radar_composite
        updated_row["Radar Minus Prior"] = radar_composite - float(
            updated_row["Overall Performance"]
        )
        updated_rows.append(updated_row)

    updated_rows.sort(key=lambda row: float(row.get("Overall Pace", 0.0)), reverse=True)
    return pd.DataFrame(updated_rows), unresolved_missing_count
