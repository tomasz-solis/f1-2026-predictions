"""Convenience exports for dashboard prediction rendering helpers."""

import pandas as pd

from src.dashboard import rendering_html, rendering_qualifying, rendering_race

render_notice_banner = rendering_html.render_notice_banner
render_page_hero_deck = rendering_html.render_page_hero_deck
render_prediction_hero_deck = rendering_html.render_prediction_hero_deck


def display_prediction_result(result: dict, prediction_name: str, is_race: bool = False) -> None:
    """Display a single prediction result (qualifying or race)."""
    results_key = "finish_order" if is_race else "grid"
    df = pd.DataFrame(result[results_key])
    df["position"] = df["position"].astype(int)
    df.attrs["input_confidence"] = result.get("input_confidence")
    result_mode = str(result.get("result_mode", "")).strip().upper()
    rendering_html.render_surface_header(
        title=prediction_name,
        summary=rendering_html._prediction_section_summary(result, is_race=is_race),
        eyebrow="Race projection" if is_race else "Qualifying projection",
    )

    highlight_cards = rendering_html._build_prediction_highlight_cards(df, result, is_race=is_race)
    rendering_html.render_stat_cards(highlight_cards)

    if result_mode == "ACTUAL":
        classification_note = str(result.get("classification_note", "")).strip()
        classification_caption = str(result.get("classification_caption", "")).strip()
        if classification_note:
            render_notice_banner(classification_note, tone="success", label="Completed session")
        if is_race:
            rendering_race._render_grid_penalty_notice(result)
            rendering_race._render_position_change_chart(
                df,
                result=result,
                prediction_name=prediction_name,
            )
        rendering_qualifying._render_actual_classification(
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
        confidence_col = (
            "order_confidence"
            if "order_confidence" in df.columns
            and pd.to_numeric(df["order_confidence"], errors="coerce").notna().any()
            else "confidence"
        )
        if confidence_col in df.columns and not df.empty:
            mean_qualifying_confidence = float(
                pd.to_numeric(df[confidence_col], errors="coerce").mean()
            )
            if mean_qualifying_confidence < 50.0:
                qualifying_warning_messages.append(
                    "Tightly-packed grid: mean order confidence is "
                    f"{mean_qualifying_confidence:.1f}% (avg chance a driver qualifies within one "
                    "place of the projected slot). This reflects how separable the field is, "
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

        team_cluster_warning = rendering_html._build_team_clustering_warning(
            df,
            mean_confidence=mean_qualifying_confidence,
        )
        if team_cluster_warning:
            qualifying_warning_messages.append(team_cluster_warning)

    compound_strategies = result.get("compound_strategies", {})
    pit_lap_distribution = result.get("pit_lap_distribution", {})

    if not is_race:
        rendering_html._render_collapsible_warnings(
            qualifying_warning_messages,
            title="Qualifying warnings",
        )

    if is_race:
        rendering_race._render_grid_penalty_notice(result)
        rendering_race._render_position_change_chart(
            df,
            result=result,
            prediction_name=prediction_name,
        )

    context_cards = rendering_html._build_context_cards(result, is_race=is_race)
    rendering_html.render_stat_cards(context_cards)

    if compound_strategies and is_race:
        rendering_race._render_compound_strategies(compound_strategies)

    if pit_lap_distribution and is_race:
        rendering_race._render_pit_lap_distribution(pit_lap_distribution)

    if is_race:
        rendering_race._render_race_result(df)
    else:
        teammate_head_to_head = result.get("teammate_head_to_head")
        if isinstance(teammate_head_to_head, list):
            rendering_qualifying._render_teammate_head_to_head_probabilities(teammate_head_to_head)
        rendering_qualifying._render_qualifying_result(df)
