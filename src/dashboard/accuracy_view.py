"""Rendering helpers for the target-aware prediction-accuracy page."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.accuracy import SeasonAccuracySummary, TargetAccuracySummary
from src.dashboard.rendering import display_prediction_result
from src.utils.accuracy_targets import (
    CHECKPOINT_ORDER,
    PRIMARY_TARGET_KEYS,
    SECONDARY_SPRINT_TARGET_KEYS,
    explicit_target_actuals,
    explicit_target_predictions,
    sanitize_prediction_rows,
    synthesize_legacy_actuals,
    synthesize_legacy_targets,
    target_checkpoint_sequence,
    target_label,
)
from src.utils.weekend import get_schedule_rows

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


def build_saved_prediction_browser_rows(
    predictions: list[dict[str, Any]],
    *,
    season_year: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize saved predictions into deterministic selector rows."""
    resolved_season_year = _resolve_saved_prediction_season_year(
        predictions,
        season_year=season_year,
    )
    schedule_rounds = _build_saved_prediction_round_map(resolved_season_year)
    race_order: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for prediction_index, prediction in enumerate(predictions):
        metadata = prediction.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        race_name = str(metadata.get("race_name", "")).strip()
        checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
        if not race_name or not checkpoint_session:
            continue

        race_order.setdefault(race_name, len(race_order))
        round_number = schedule_rounds.get(_normalize_saved_race_name(race_name))
        weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
        if weekend_format not in {"normal", "sprint"}:
            weekend_format = "sprint" if _saved_prediction_is_sprint(prediction) else "normal"

        predicted_at = _parse_saved_timestamp(metadata.get("predicted_at"))
        information_cutoff_at = _parse_saved_timestamp(metadata.get("information_cutoff_at"))
        rows.append(
            {
                "race_name": race_name,
                "checkpoint_session": checkpoint_session,
                "weekend_format": weekend_format,
                "weather": str(metadata.get("weather", "")).strip().lower(),
                "predicted_at": predicted_at,
                "predicted_at_label": _format_saved_timestamp(predicted_at),
                "information_cutoff_at": information_cutoff_at,
                "information_cutoff_label": _format_saved_timestamp(information_cutoff_at),
                "prediction": prediction,
                "checkpoint_option_value": str(prediction_index),
                "round_number": round_number,
                "_race_order": race_order[race_name],
            }
        )

    rows.sort(
        key=lambda row: (
            row.get("round_number") is None,
            int(row.get("round_number") or row.get("_race_order", 0)),
            CHECKPOINT_ORDER.get(str(row.get("checkpoint_session", "")), 99),
            row.get("predicted_at", datetime.min.replace(tzinfo=UTC)),
        )
    )
    duplicate_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        duplicate_key = (
            str(row.get("race_name", "")).strip(),
            str(row.get("checkpoint_session", "")).strip().upper(),
        )
        duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1

    duplicate_seen: dict[tuple[str, str], int] = {}
    for row in rows:
        duplicate_key = (
            str(row.get("race_name", "")).strip(),
            str(row.get("checkpoint_session", "")).strip().upper(),
        )
        duplicate_seen[duplicate_key] = duplicate_seen.get(duplicate_key, 0) + 1
        row["checkpoint_option_label"] = _saved_checkpoint_option_label(
            row,
            duplicate_index=duplicate_seen[duplicate_key],
            duplicate_count=duplicate_counts[duplicate_key],
        )
        row["race_option_label"] = _saved_race_option_label(
            str(row.get("race_name", "")).strip(),
            round_number=row.get("round_number"),
        )
        row.pop("_race_order", None)
    return rows


def build_saved_prediction_view_model(prediction: dict[str, Any]) -> dict[str, Any]:
    """Adapt one saved checkpoint artifact into display payloads."""
    metadata = prediction.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    checkpoint_session = str(metadata.get("session_name", "")).strip().upper() or "UNKNOWN"
    qualifying_target_key = str(metadata.get("top_level_qualifying_target", "")).strip()
    race_target_key = str(metadata.get("top_level_race_target", "")).strip()

    target_predictions = explicit_target_predictions(prediction)
    is_sprint = _saved_prediction_is_sprint(prediction)
    if not target_predictions:
        target_predictions = synthesize_legacy_targets(prediction, is_sprint=is_sprint)
    target_actuals = explicit_target_actuals(prediction)
    if not target_actuals:
        target_actuals = synthesize_legacy_actuals(prediction, is_sprint=is_sprint)

    qualifying_target = target_predictions.get(qualifying_target_key, {})
    race_target = target_predictions.get(race_target_key, {})
    qualifying_rows = sanitize_prediction_rows(
        qualifying_target.get("predicted_order")
        if isinstance(qualifying_target, dict)
        else (prediction.get("qualifying") or {}).get("predicted_grid")
    )
    if not qualifying_rows:
        qualifying_rows = sanitize_prediction_rows(
            (prediction.get("qualifying") or {}).get("predicted_grid")
        )

    race_rows = sanitize_prediction_rows(
        race_target.get("predicted_order")
        if isinstance(race_target, dict)
        else (prediction.get("race") or {}).get("predicted_results")
    )
    if not race_rows:
        race_rows = sanitize_prediction_rows(
            (prediction.get("race") or {}).get("predicted_results")
        )

    qualifying_title = _saved_prediction_section_title(
        qualifying_target_key,
        default_title="Qualifying Checkpoint",
    )
    race_title = _saved_prediction_section_title(
        race_target_key,
        default_title="Race Checkpoint",
    )

    qualifying_result: dict[str, Any] | None = None
    if qualifying_rows:
        qualifying_result = {
            "grid": qualifying_rows,
            "result_mode": str(metadata.get("top_level_qualifying_result_mode", "PREDICTED"))
            .strip()
            .upper(),
            "grid_source": str(metadata.get("top_level_qualifying_grid_source", "PREDICTED"))
            .strip()
            .upper(),
            "data_source": f"Saved checkpoint ({checkpoint_session})",
            "fp_blend_info": (
                qualifying_target.get("fp_blend_info")
                if isinstance(qualifying_target.get("fp_blend_info"), dict)
                else {}
            ),
        }

    race_result: dict[str, Any] | None = None
    if race_rows:
        race_result = {
            "finish_order": race_rows,
            "result_mode": str(metadata.get("top_level_race_result_mode", "PREDICTED"))
            .strip()
            .upper(),
            "grid_source": str(metadata.get("top_level_race_grid_source", "PREDICTED"))
            .strip()
            .upper(),
            "data_source": f"Saved checkpoint ({checkpoint_session})",
            "starting_grid": qualifying_rows,
            "starting_session_name": str(
                qualifying_target.get("target_session")
                or metadata.get("top_level_qualifying_session")
                or "Q"
            )
            .strip()
            .upper(),
            "input_confidence": race_target.get("mean_confidence"),
        }

    target_status_rows: list[dict[str, Any]] = []
    for target_key, payload in target_predictions.items():
        if not isinstance(payload, dict):
            continue
        target_status_rows.append(
            {
                "target_key": target_key,
                "label": target_label(target_key),
                "session_name": str(payload.get("target_session", "")).strip().upper(),
                "eligible_at_save": bool(payload.get("eligible_at_save", True)),
                "has_actuals": bool(target_actuals.get(target_key)),
            }
        )
    target_status_rows.sort(
        key=lambda row: (
            CHECKPOINT_ORDER.get(str(row.get("session_name", "")), 99),
            str(row.get("label", "")),
        )
    )

    weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    if weekend_format not in {"normal", "sprint"}:
        weekend_format = "sprint" if is_sprint else "normal"

    return {
        "checkpoint_session": checkpoint_session,
        "weekend_format": weekend_format,
        "weather": str(metadata.get("weather", "")).strip().lower(),
        "predicted_at": _format_saved_timestamp(
            _parse_saved_timestamp(metadata.get("predicted_at"))
        ),
        "information_cutoff_at": _format_saved_timestamp(
            _parse_saved_timestamp(metadata.get("information_cutoff_at"))
        ),
        "source": str(metadata.get("source", "")).strip(),
        "qualifying_title": qualifying_title,
        "qualifying_result": qualifying_result,
        "race_title": race_title,
        "race_result": race_result,
        "target_status_rows": target_status_rows,
    }


def render_saved_prediction_viewer(
    predictions: list[dict[str, Any]],
    *,
    season_year: int | None = None,
) -> None:
    """Render a browsable viewer for saved checkpoint artifacts."""
    st.subheader("Checkpoint Viewer")

    rows = build_saved_prediction_browser_rows(predictions, season_year=season_year)
    if not rows:
        st.info("No saved checkpoints yet.")
        return

    race_names: list[str] = []
    race_labels: dict[str, str] = {}
    for row in rows:
        race_name = str(row.get("race_name", "")).strip()
        if race_name and race_name not in race_names:
            race_names.append(race_name)
            race_labels[race_name] = str(row.get("race_option_label", race_name))

    selected_race = st.selectbox(
        "Saved race",
        options=race_names,
        index=max(0, len(race_names) - 1),
        key="saved_prediction_viewer_race",
        format_func=lambda race: race_labels.get(race, race),
        help="Browse one saved race weekend and inspect each checkpoint artifact.",
    )
    race_rows = [row for row in rows if row.get("race_name") == selected_race]
    checkpoint_options = [str(row.get("checkpoint_option_value", "")) for row in race_rows]
    checkpoint_labels = {
        str(row.get("checkpoint_option_value", "")): str(row.get("checkpoint_option_label", ""))
        for row in race_rows
    }
    selected_checkpoint = st.selectbox(
        "Saved checkpoint",
        options=checkpoint_options,
        index=max(0, len(checkpoint_options) - 1),
        key=f"saved_prediction_viewer_checkpoint_{selected_race}",
        format_func=lambda checkpoint: checkpoint_labels.get(checkpoint, checkpoint),
        help="Choose the historical cut-off you want to inspect.",
    )

    selected_row = next(
        (
            row
            for row in race_rows
            if str(row.get("checkpoint_option_value", "")) == selected_checkpoint
        ),
        None,
    )
    if selected_row is None:
        st.warning("Could not load the selected checkpoint.")
        return

    view_model = build_saved_prediction_view_model(selected_row["prediction"])
    metadata_bits = [
        _saved_round_caption(selected_row.get("round_number")),
        f"Weekend: {str(view_model.get('weekend_format', 'unknown')).title()}",
        f"Weather: {str(view_model.get('weather', 'unknown')).upper()}",
    ]
    metadata_bits = [bit for bit in metadata_bits if bit]
    source = str(view_model.get("source", "")).strip()
    if source:
        metadata_bits.append(f"Source: {source}")
    st.caption(" | ".join(metadata_bits))

    target_status_rows = view_model.get("target_status_rows", [])
    if isinstance(target_status_rows, list) and target_status_rows:
        st.markdown("**Tracked targets**")
        for row in target_status_rows:
            if not isinstance(row, dict):
                continue
            eligibility_text = (
                "scoreable at save time"
                if bool(row.get("eligible_at_save", True))
                else "saved for inspection, excluded from scoring"
            )
            actuals_text = "actuals attached" if bool(row.get("has_actuals")) else "actuals pending"
            st.write(
                f"{row.get('label', 'Target')} ({row.get('session_name', '')}): "
                f"{eligibility_text}; {actuals_text}."
            )

    qualifying_result = view_model.get("qualifying_result")
    if isinstance(qualifying_result, dict):
        display_prediction_result(
            qualifying_result,
            str(view_model.get("qualifying_title", "Qualifying Checkpoint")),
            False,
        )

    race_result = view_model.get("race_result")
    if isinstance(race_result, dict):
        display_prediction_result(
            race_result,
            str(view_model.get("race_title", "Race Checkpoint")),
            True,
        )


def _saved_prediction_is_sprint(prediction: dict[str, Any]) -> bool:
    """Infer whether a saved checkpoint belongs to a sprint weekend."""
    metadata = prediction.get("metadata", {})
    weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    if weekend_format in {"normal", "sprint"}:
        return weekend_format == "sprint"

    targets = prediction.get("targets", {})
    if isinstance(targets, dict) and any("sprint" in str(key) for key in targets):
        return True

    checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
    return checkpoint_session in {"SQ", "SPRINT"}


def _saved_prediction_section_title(target_key: str, *, default_title: str) -> str:
    """Return a human-readable section title for one saved target."""
    normalized_target = str(target_key).strip()
    if not normalized_target:
        return default_title
    return f"{target_label(normalized_target)} Checkpoint"


def _saved_checkpoint_option_label(
    row: dict[str, Any],
    *,
    duplicate_index: int = 1,
    duplicate_count: int = 1,
) -> str:
    """Build the clean checkpoint label shown in the saved-prediction selector."""
    checkpoint_session = str(row.get("checkpoint_session", "")).strip().upper()
    if duplicate_count <= 1:
        return checkpoint_session
    return f"{checkpoint_session} ({duplicate_index})"


def _saved_race_option_label(race_name: str, *, round_number: Any) -> str:
    """Build the race label shown in the saved-race selector."""
    if isinstance(round_number, int) and round_number > 0:
        return f"Round {round_number} | {race_name}"
    return race_name


def _saved_round_caption(round_number: Any) -> str:
    """Return the short round caption shown above one checkpoint view."""
    if isinstance(round_number, int) and round_number > 0:
        return f"Round: {round_number}"
    return ""


def _resolve_saved_prediction_season_year(
    predictions: list[dict[str, Any]],
    *,
    season_year: int | None,
) -> int | None:
    """Resolve the season year used for schedule-aware checkpoint ordering."""
    if season_year is not None:
        return int(season_year)

    resolved_years: set[int] = set()
    for prediction in predictions:
        metadata = prediction.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        raw_year = metadata.get("year")
        if not isinstance(raw_year, int | float | str):
            continue
        try:
            resolved_years.add(int(raw_year))
        except (TypeError, ValueError):
            continue

    if len(resolved_years) == 1:
        return next(iter(resolved_years))
    return None


def _build_saved_prediction_round_map(season_year: int | None) -> dict[str, int]:
    """Map normalized race names to season round numbers when schedule data exists."""
    if season_year is None:
        return {}

    try:
        schedule_rows = get_schedule_rows(season_year)
    except Exception:
        return {}

    round_map: dict[str, int] = {}
    for round_number, schedule_row in enumerate(schedule_rows, start=1):
        if not isinstance(schedule_row, tuple) or not schedule_row:
            continue
        race_name = str(schedule_row[0]).strip()
        if not race_name:
            continue
        round_map[_normalize_saved_race_name(race_name)] = round_number
    return round_map


def _normalize_saved_race_name(race_name: str) -> str:
    """Normalize one race name for case-insensitive schedule lookups."""
    without_accents = unicodedata.normalize("NFKD", str(race_name)).encode(
        "ascii",
        "ignore",
    )
    return " ".join(without_accents.decode("ascii").split()).lower()


def _parse_saved_timestamp(value: Any) -> datetime:
    """Parse one stored timestamp into UTC with a stable minimum fallback."""
    if not isinstance(value, str):
        return datetime.min.replace(tzinfo=UTC)
    candidate = value.strip()
    if not candidate:
        return datetime.min.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_saved_timestamp(value: datetime) -> str:
    """Return a compact UTC timestamp label for saved artifacts."""
    if value == datetime.min.replace(tzinfo=UTC):
        return "Unknown"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


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
            figure.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=race_names,
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
