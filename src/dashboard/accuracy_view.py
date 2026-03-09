"""Rendering helpers for the target-aware prediction-accuracy page."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.accuracy import SeasonAccuracySummary, TargetAccuracySummary
from src.utils.accuracy_targets import (
    PRIMARY_TARGET_KEYS,
    SECONDARY_SPRINT_TARGET_KEYS,
    target_checkpoint_sequence,
)

METRIC_OPTIONS = {
    "overall_mae": "Overall MAE",
    "top_3_pct": "Top 3 overlap %",
    "top_10_pct": "Top 10 overlap %",
    "exact_accuracy": "Exact accuracy %",
    "within_1": "Within 1 position %",
    "within_3": "Within 3 positions %",
    "correlation": "Correlation",
}


def render_overall_accuracy_metrics(summary: SeasonAccuracySummary) -> None:
    """Render headline KPI cards for the main qualifying and race targets."""
    st.markdown("---")
    st.subheader("Overall Accuracy")

    main_qualifying = summary.targets.get("main_qualifying")
    grand_prix_race = summary.targets.get("grand_prix_race")
    columns = st.columns(6)
    _render_metric_card(columns[0], "Q MAE", main_qualifying, "overall_mae", "{:.2f}")
    _render_metric_card(columns[1], "Q Top 3", main_qualifying, "top_3_pct", "{:.1f}%")
    _render_metric_card(columns[2], "Q Top 10", main_qualifying, "top_10_pct", "{:.1f}%")
    _render_metric_card(columns[3], "R MAE", grand_prix_race, "overall_mae", "{:.2f}")
    _render_metric_card(columns[4], "R Top 3", grand_prix_race, "top_3_pct", "{:.1f}%")
    _render_metric_card(columns[5], "R Top 10", grand_prix_race, "top_10_pct", "{:.1f}%")


def render_target_sections(
    summary: SeasonAccuracySummary,
    metric_name: str,
    show_secondary_sprint_targets: bool,
) -> None:
    """Render weekend progression and season trend charts by target."""
    primary_targets = [key for key in PRIMARY_TARGET_KEYS if key in summary.targets]
    if primary_targets:
        _render_target_tabs(
            title="Main Targets",
            target_keys=primary_targets,
            summary=summary,
            metric_name=metric_name,
        )

    if show_secondary_sprint_targets:
        secondary_targets = [key for key in SECONDARY_SPRINT_TARGET_KEYS if key in summary.targets]
        if secondary_targets:
            _render_target_tabs(
                title="Sprint Targets",
                target_keys=secondary_targets,
                summary=summary,
                metric_name=metric_name,
            )


def render_saved_predictions_summary(status_rows: list[dict[str, Any]]) -> None:
    """Render the saved checkpoint list with target-aware status text."""
    st.markdown("---")
    st.subheader("Saved Predictions")
    if not status_rows:
        st.info("No saved checkpoints yet.")
        return

    for row in status_rows:
        race_name = str(row.get("race_name", "")).strip()
        checkpoint_session = str(row.get("checkpoint_session", "")).strip().upper()
        weekend_format = str(row.get("weekend_format", "")).strip().title()
        target_labels = row.get("target_labels", [])
        targets_text = (
            ", ".join(str(label) for label in target_labels) if target_labels else "No targets"
        )
        status_text = str(row.get("status_text", "")).strip() or "No scoreable targets"
        st.write(
            f"**{race_name}** ({checkpoint_session}, {weekend_format}) "
            f"- {status_text} - {targets_text}"
        )


def _render_metric_card(
    container: Any,
    label: str,
    summary: TargetAccuracySummary | None,
    metric_name: str,
    template: str,
) -> None:
    """Render one KPI card when the metric is available."""
    value = None
    if summary is not None:
        metric_payload = summary.aggregate.get(metric_name)
        if isinstance(metric_payload, dict):
            value = metric_payload.get("mean")
    with container:
        if isinstance(value, int | float):
            st.metric(label, template.format(float(value)))
        else:
            st.metric(label, "N/A")


def _render_target_tabs(
    *,
    title: str,
    target_keys: list[str],
    summary: SeasonAccuracySummary,
    metric_name: str,
) -> None:
    """Render a tabbed section for a group of targets."""
    st.markdown("---")
    st.subheader(title)
    tabs = st.tabs([summary.targets[key].label for key in target_keys])
    for tab, target_key in zip(tabs, target_keys, strict=False):
        target_summary = summary.targets.get(
            target_key, TargetAccuracySummary(target_key, target_key)
        )
        with tab:
            _render_progression_charts(target_summary, metric_name)
            _render_trend_charts(target_summary, metric_name)


def _render_progression_charts(target_summary: TargetAccuracySummary, metric_name: str) -> None:
    """Render weekend progression charts for normal and sprint weekends."""
    st.markdown("**Weekend Progression**")
    left_col, right_col = st.columns(2)
    for container, weekend_format in ((left_col, "normal"), (right_col, "sprint")):
        checkpoint_labels, metric_values, race_counts, missing_checkpoints = (
            build_progression_series(
                target_summary=target_summary,
                metric_name=metric_name,
                weekend_format=weekend_format,
            )
        )
        with container:
            st.caption(f"{weekend_format.title()} weekends")
            if not checkpoint_labels:
                st.info("No data for this format yet.")
                continue

            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=checkpoint_labels,
                    y=metric_values,
                    mode="lines+markers",
                    name=target_summary.label,
                    customdata=[[count] for count in race_counts],
                    hovertemplate=(
                        "Checkpoint: %{x}<br>"
                        "Value: %{y:.2f}<br>"
                        "Races: %{customdata[0]}<extra></extra>"
                    ),
                    connectgaps=False,
                )
            )
            figure.update_layout(
                margin={"l": 12, "r": 12, "t": 12, "b": 12},
                xaxis_title="Checkpoint",
                yaxis_title=METRIC_OPTIONS.get(metric_name, metric_name),
                showlegend=False,
            )
            figure.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=checkpoint_labels,
            )
            st.plotly_chart(figure, width="stretch")
            if missing_checkpoints:
                st.caption(f"Missing saved checkpoints: {', '.join(missing_checkpoints)}")


def _render_trend_charts(target_summary: TargetAccuracySummary, metric_name: str) -> None:
    """Render season trend charts for normal and sprint weekends."""
    st.markdown("**Season Trend**")
    left_col, right_col = st.columns(2)
    for container, weekend_format in ((left_col, "normal"), (right_col, "sprint")):
        points = [
            point
            for point in target_summary.season_trend
            if point.weekend_format == weekend_format and metric_name in point.metrics
        ]
        with container:
            st.caption(f"{weekend_format.title()} weekends")
            if not points:
                st.info("No data for this format yet.")
                continue

            ordered_races = sorted(
                {(point.race_order, point.race_name) for point in points},
                key=lambda item: item[0],
            )
            race_names = [race_name for _, race_name in ordered_races]
            figure = go.Figure()
            checkpoint_sessions = sorted(
                {point.checkpoint_session for point in points},
                key=lambda session_name: min(
                    point.checkpoint_index
                    for point in points
                    if point.checkpoint_session == session_name
                ),
            )
            for checkpoint_session in checkpoint_sessions:
                lookup = {
                    point.race_name: point.metrics[metric_name]
                    for point in points
                    if point.checkpoint_session == checkpoint_session
                }
                figure.add_trace(
                    go.Scatter(
                        x=race_names,
                        y=[lookup.get(race_name) for race_name in race_names],
                        mode="lines+markers",
                        name=checkpoint_session,
                        connectgaps=False,
                    )
                )

            figure.update_layout(
                margin={"l": 12, "r": 12, "t": 12, "b": 12},
                xaxis_title="Race",
                yaxis_title=METRIC_OPTIONS.get(metric_name, metric_name),
                legend_title="Checkpoint",
            )
            st.plotly_chart(figure, width="stretch")


def build_progression_series(
    *,
    target_summary: TargetAccuracySummary,
    metric_name: str,
    weekend_format: str,
) -> tuple[list[str], list[float | None], list[int], list[str]]:
    """Return an ordered checkpoint series with explicit gaps for missing saves."""
    points = [
        point
        for point in target_summary.checkpoint_progression
        if point.weekend_format == weekend_format and metric_name in point.metrics
    ]
    if not points:
        return [], [], [], []

    point_by_session = {point.checkpoint_session: point for point in points}
    expected_sessions = list(target_checkpoint_sequence(target_summary.target_key, weekend_format))
    observed_sessions = [
        point.checkpoint_session
        for point in sorted(points, key=lambda item: item.checkpoint_index)
        if point.checkpoint_session not in expected_sessions
    ]
    ordered_sessions = expected_sessions + observed_sessions
    if not ordered_sessions:
        ordered_sessions = [
            point.checkpoint_session
            for point in sorted(points, key=lambda item: item.checkpoint_index)
        ]

    metric_values: list[float | None] = []
    race_counts: list[int] = []
    missing_checkpoints: list[str] = []
    for checkpoint_session in ordered_sessions:
        point = point_by_session.get(checkpoint_session)
        if point is None:
            metric_values.append(None)
            race_counts.append(0)
            missing_checkpoints.append(checkpoint_session)
            continue
        metric_values.append(float(point.metrics[metric_name]))
        race_counts.append(int(point.race_count))

    return ordered_sessions, metric_values, race_counts, missing_checkpoints
