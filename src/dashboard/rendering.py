"""Convenience exports for dashboard prediction rendering helpers."""

import pandas as pd

from src.dashboard import rendering_html, rendering_qualifying, rendering_race

st = rendering_html.st

_build_context_cards = rendering_html._build_context_cards
_build_prediction_highlight_cards = rendering_html._build_prediction_highlight_cards
_build_stat_cards_html = rendering_html._build_stat_cards_html
_build_surface_header_html = rendering_html._build_surface_header_html
_build_team_clustering_warning = rendering_html._build_team_clustering_warning
_build_track_temperature_context_card = rendering_html._build_track_temperature_context_card
_build_weather_feature_context_card = rendering_html._build_weather_feature_context_card
_parse_optional_float = rendering_html._parse_optional_float
_prediction_section_summary = rendering_html._prediction_section_summary
_render_collapsible_warnings = rendering_html._render_collapsible_warnings
_short_data_source_label = rendering_html._short_data_source_label
render_notice_banner = rendering_html.render_notice_banner
render_page_hero_deck = rendering_html.render_page_hero_deck
render_prediction_hero_deck = rendering_html.render_prediction_hero_deck
render_stage_timeline = rendering_html.render_stage_timeline
render_stat_cards = rendering_html.render_stat_cards
render_surface_header = rendering_html.render_surface_header

_render_actual_classification = rendering_qualifying._render_actual_classification
_render_qualifying_result = rendering_qualifying._render_qualifying_result
_render_teammate_head_to_head_probabilities = (
    rendering_qualifying._render_teammate_head_to_head_probabilities
)

_build_position_change_frame = rendering_race._build_position_change_frame
_movement_bar_labels = rendering_race._movement_bar_labels
_position_change_chart_figure = rendering_race._position_change_chart_figure
_position_change_chart_title = rendering_race._position_change_chart_title
_render_compound_strategies = rendering_race._render_compound_strategies
_render_pit_lap_distribution = rendering_race._render_pit_lap_distribution
_render_position_change_chart = rendering_race._render_position_change_chart
_render_race_result = rendering_race._render_race_result
_render_track_temperature_context = rendering_race._render_track_temperature_context
_render_weather_feature_context = rendering_race._render_weather_feature_context
_style_race_table = rendering_race._style_race_table


def display_prediction_result(result: dict, prediction_name: str, is_race: bool = False) -> None:
    """Display a single prediction result (qualifying or race)."""
    results_key = "finish_order" if is_race else "grid"
    df = pd.DataFrame(result[results_key])
    df["position"] = df["position"].astype(int)
    df.attrs["input_confidence"] = result.get("input_confidence")
    result_mode = str(result.get("result_mode", "")).strip().upper()
    render_surface_header(
        title=prediction_name,
        summary=_prediction_section_summary(result, is_race=is_race),
        eyebrow="Race projection" if is_race else "Qualifying projection",
    )

    highlight_cards = _build_prediction_highlight_cards(df, result, is_race=is_race)
    render_stat_cards(highlight_cards)

    if result_mode == "ACTUAL":
        classification_note = str(result.get("classification_note", "")).strip()
        classification_caption = str(result.get("classification_caption", "")).strip()
        if classification_note:
            render_notice_banner(classification_note, tone="success", label="Completed session")
        if is_race:
            _render_position_change_chart(
                df,
                result=result,
                prediction_name=prediction_name,
            )
        _render_actual_classification(
            df,
            caption=classification_caption
            or "This table shows the completed-session classification from FastF1.",
        )
        return

    qualifying_warning_messages: list[str] = []

    if not is_race:
        data_source = result.get("data_source", "Unknown")
        blend_used = result.get("blend_used", False)
        fp_blend_weight_used = result.get("fp_blend_weight_used")

        if blend_used:
            source_label = "practice data"
            if isinstance(data_source, str) and "checkpoint profile blend" in data_source.lower():
                source_label = "stored checkpoint snapshot"
            if isinstance(fp_blend_weight_used, int | float):
                practice_share = int(round(float(fp_blend_weight_used) * 100))
                model_share = max(0, 100 - practice_share)
                render_notice_banner(
                    (
                        f"Data source: {data_source} "
                        f"({practice_share}% {source_label} + {model_share}% model)."
                    ),
                    tone="info",
                    label="Input mix",
                )
            else:
                render_notice_banner(
                    f"Data source: {data_source} (70% {source_label} + 30% model).",
                    tone="info",
                    label="Input mix",
                )
        else:
            render_notice_banner(f"Data source: {data_source}.", tone="info", label="Input mix")
            if isinstance(data_source, str) and "Model-only" in data_source:
                qualifying_warning_messages.append(
                    "Low-confidence qualifying mode: no weekend practice/testing signal. "
                    "Early grids can look too team-ordered."
                )
            elif isinstance(data_source, str) and "Testing short-run profile blend" in data_source:
                qualifying_warning_messages.append(
                    "Medium-confidence qualifying mode: using testing-derived team pace without "
                    "weekend laps. Expect wider position ranges."
                )
        if "confidence" in df.columns and not df.empty:
            mean_qualifying_confidence = float(
                pd.to_numeric(df["confidence"], errors="coerce").mean()
            )
            if mean_qualifying_confidence < 56.0:
                qualifying_warning_messages.append(
                    "Wide predicted-order spread: mean order confidence is "
                    f"{mean_qualifying_confidence:.1f}%. This reflects simulation spread, "
                    "not just how many weekends the model has learned."
                )
        else:
            mean_qualifying_confidence = None

        has_quali_ci = "p5" in df.columns and "p95" in df.columns
        if has_quali_ci and not df.empty:
            interval_width = (
                pd.to_numeric(df["p95"], errors="coerce") - pd.to_numeric(df["p5"], errors="coerce")
            ).fillna(0.0)
            wide_ranges = int((interval_width >= 8.0).sum())
            if wide_ranges >= max(6, int(len(df) * 0.35)):
                qualifying_warning_messages.append(
                    f"Wide position ranges: {wide_ranges} drivers have 90% ranges spanning 8+ places."
                )

        team_cluster_warning = _build_team_clustering_warning(
            df,
            mean_confidence=mean_qualifying_confidence,
        )
        if team_cluster_warning:
            qualifying_warning_messages.append(team_cluster_warning)

    compound_strategies = result.get("compound_strategies", {})
    pit_lap_distribution = result.get("pit_lap_distribution", {})

    if not is_race:
        _render_collapsible_warnings(
            qualifying_warning_messages,
            title="Qualifying warnings",
        )

    if is_race:
        _render_position_change_chart(
            df,
            result=result,
            prediction_name=prediction_name,
        )

    context_cards = _build_context_cards(result, is_race=is_race)
    render_stat_cards(context_cards)

    if compound_strategies and is_race:
        _render_compound_strategies(compound_strategies)

    if pit_lap_distribution and is_race:
        _render_pit_lap_distribution(pit_lap_distribution)

    if is_race:
        _render_race_result(df)
    else:
        teammate_head_to_head = result.get("teammate_head_to_head")
        if isinstance(teammate_head_to_head, list):
            _render_teammate_head_to_head_probabilities(teammate_head_to_head)
        _render_qualifying_result(df)
