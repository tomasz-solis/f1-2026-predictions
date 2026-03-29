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
_TARGET_HIGHLIGHT_METRICS = (
    ("overall_mae", "MAE", "{:.2f}"),
    ("exact_accuracy", "Exact", "{:.1f}%"),
    ("within_1", "Within 1", "{:.1f}%"),
    ("within_3", "Within 3", "{:.1f}%"),
    ("correlation", "Correlation", "{:.2f}"),
)


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
        _render_target_section(
            title="Main Targets",
            target_keys=primary_targets,
            summary=summary,
            metric_name=metric_name,
        )

    if show_secondary_sprint_targets:
        secondary_targets = [key for key in SECONDARY_SPRINT_TARGET_KEYS if key in summary.targets]
        if secondary_targets:
            _render_target_section(
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


def build_target_metric_cards(target_summary: TargetAccuracySummary) -> list[dict[str, Any]]:
    """Return formatted metric-card payloads for the selected target summary."""
    cards: list[dict[str, Any]] = []
    for metric_name, label, template in _TARGET_HIGHLIGHT_METRICS:
        value = None
        metric_payload = target_summary.aggregate.get(metric_name)
        if isinstance(metric_payload, dict):
            raw_value = metric_payload.get("mean")
            if isinstance(raw_value, int | float):
                value = float(raw_value)
        cards.append(
            {
                "metric_name": metric_name,
                "label": label,
                "template": template,
                "value": value,
            }
        )
    return cards


def _render_target_metric_cards(target_summary: TargetAccuracySummary) -> None:
    """Render a compact summary row for the currently selected target."""
    cards = build_target_metric_cards(target_summary)
    if not cards:
        return

    st.markdown("**Target Summary**")
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=False):
        with column:
            value = card.get("value")
            template = str(card.get("template", "{:.2f}"))
            label = str(card.get("label", "Metric"))
            if isinstance(value, int | float):
                st.metric(label, template.format(float(value)))
            else:
                st.metric(label, "N/A")


def _render_target_section(
    *,
    title: str,
    target_keys: list[str],
    summary: SeasonAccuracySummary,
    metric_name: str,
) -> None:
    """Render one target group with a deterministic selector."""
    st.markdown("---")
    st.subheader(title)
    selected_target_key = _select_target_key(
        title=title,
        target_keys=target_keys,
        summary=summary,
    )
    target_summary = summary.targets.get(
        selected_target_key,
        TargetAccuracySummary(selected_target_key, selected_target_key),
    )
    _render_target_metric_cards(target_summary)
    _render_progression_charts(target_summary, metric_name)
    _render_trend_charts(target_summary, metric_name)


def _select_target_key(
    *,
    title: str,
    target_keys: list[str],
    summary: SeasonAccuracySummary,
) -> str:
    """Return the user-selected target key for one chart group."""
    labels = {key: summary.targets[key].label for key in target_keys}
    selector_key = f"accuracy_target_selector_{title.lower().replace(' ', '_')}"
    segmented_control = getattr(st, "segmented_control", None)
    selected_target_key: str | None = None
    if callable(segmented_control):
        try:
            selected_target_key = segmented_control(
                "Target",
                options=target_keys,
                selection_mode="single",
                default=target_keys[0],
                format_func=lambda key: labels.get(key, key),
                key=selector_key,
            )
        except TypeError:
            try:
                selected_target_key = segmented_control(
                    "Target",
                    options=target_keys,
                    default=target_keys[0],
                    format_func=lambda key: labels.get(key, key),
                    key=selector_key,
                )
            except TypeError:
                selected_target_key = segmented_control(
                    "Target",
                    options=target_keys,
                    default=target_keys[0],
                    key=selector_key,
                )

    if selected_target_key not in target_keys:
        selected_target_key = st.radio(
            "Target",
            options=target_keys,
            index=0,
            format_func=lambda key: labels.get(key, key),
            horizontal=True,
            key=f"{selector_key}_radio",
            label_visibility="collapsed",
        )

    return selected_target_key if selected_target_key in target_keys else target_keys[0]


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
            checkpoint_state = build_progression_checkpoint_state(
                target_summary=target_summary,
                weekend_format=weekend_format,
            )
            observed_labels, observed_values, observed_counts = build_progression_line_series(
                checkpoint_labels=checkpoint_labels,
                metric_values=metric_values,
                race_counts=race_counts,
            )

            figure = go.Figure()
            if observed_labels:
                figure.add_trace(
                    go.Scatter(
                        x=observed_labels,
                        y=observed_values,
                        mode="lines+markers",
                        name=target_summary.label,
                        customdata=[[count] for count in observed_counts],
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
            valid_checkpoints = ", ".join(observed_labels) if observed_labels else "None"
            caption_parts = [f"Valid checkpoints: {valid_checkpoints}"]
            excluded_checkpoints = checkpoint_state["excluded_checkpoints"]
            pending_checkpoints = checkpoint_state["pending_checkpoints"]
            if excluded_checkpoints:
                caption_parts.append(
                    "Excluded checkpoints: "
                    + ", ".join(excluded_checkpoints)
                    + " (contaminated or no longer a live forecast)"
                )
            if pending_checkpoints:
                caption_parts.append("Pending checkpoints: " + ", ".join(pending_checkpoints))
            if missing_checkpoints:
                caption_parts.append("Missing checkpoints: " + ", ".join(missing_checkpoints))
            st.caption(". ".join(caption_parts))
            if max(race_counts, default=0) <= 1:
                st.caption(
                    "Only one valid race contributes here, so this path is a single-weekend trace, "
                    "not a stable average yet."
                )


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
            if len(race_names) < 2:
                st.caption(
                    "Trend lines need at least two races in this format; only points are available so far."
                )


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
    status_points = [
        point
        for point in target_summary.checkpoint_status
        if point.weekend_format == weekend_format
    ]
    if not points and not status_points:
        return [], [], [], []

    point_by_session = {point.checkpoint_session: point for point in points}
    status_by_session = {point.checkpoint_session: point for point in status_points}
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
            status_point = status_by_session.get(checkpoint_session)
            if status_point is None:
                missing_checkpoints.append(checkpoint_session)
            continue
        metric_values.append(float(point.metrics[metric_name]))
        race_counts.append(int(point.race_count))

    return ordered_sessions, metric_values, race_counts, missing_checkpoints


def build_progression_line_series(
    *,
    checkpoint_labels: list[str],
    metric_values: list[float | None],
    race_counts: list[int],
) -> tuple[list[str], list[float], list[int]]:
    """Return only observed checkpoints so progression lines stay visible across missing saves."""
    observed_rows = [
        (label, float(value), int(count))
        for label, value, count in zip(checkpoint_labels, metric_values, race_counts, strict=False)
        if isinstance(value, int | float)
    ]
    if not observed_rows:
        return [], [], []

    observed_labels = [label for label, _, _ in observed_rows]
    observed_values = [value for _, value, _ in observed_rows]
    observed_counts = [count for _, _, count in observed_rows]
    return observed_labels, observed_values, observed_counts


def build_progression_checkpoint_state(
    *,
    target_summary: TargetAccuracySummary,
    weekend_format: str,
) -> dict[str, list[str]]:
    """Return scored, excluded, and pending checkpoint labels for one target format."""
    status_points = [
        point
        for point in target_summary.checkpoint_status
        if point.weekend_format == weekend_format
    ]
    scored_checkpoints: list[str] = []
    excluded_checkpoints: list[str] = []
    pending_checkpoints: list[str] = []
    for point in sorted(status_points, key=lambda item: item.checkpoint_index):
        if point.scored_count > 0:
            scored_checkpoints.append(point.checkpoint_session)
        elif point.excluded_count > 0:
            excluded_checkpoints.append(point.checkpoint_session)
        elif point.pending_count > 0:
            pending_checkpoints.append(point.checkpoint_session)

    return {
        "scored_checkpoints": scored_checkpoints,
        "excluded_checkpoints": excluded_checkpoints,
        "pending_checkpoints": pending_checkpoints,
    }
