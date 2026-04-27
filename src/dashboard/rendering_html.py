"""HTML and summary helpers for dashboard prediction rendering."""

from html import escape
from typing import Any

import pandas as pd
import streamlit as st


def _build_surface_header_html(
    *,
    title: str,
    summary: str | None = None,
    eyebrow: str | None = None,
    tone: str = "default",
) -> str:
    """Build the HTML markup for a styled surface header."""
    eyebrow_html = (
        f'<div class="ts-surface-header__eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    )
    summary_html = f'<p class="ts-surface-header__summary">{escape(summary)}</p>' if summary else ""
    return (
        f'<section class="ts-surface-header ts-surface-header--{escape(tone)}">'
        f"{eyebrow_html}"
        f'<h2 class="ts-surface-header__title">{escape(title)}</h2>'
        f"{summary_html}"
        "</section>"
    )


def _build_stat_cards_html(
    cards: list[dict[str, str]],
    *,
    grid_class: str = "ts-stat-grid",
) -> str:
    """Build the HTML markup for a compact stat-card grid."""
    valid_cards = [card for card in cards if card.get("label") and card.get("value")]
    if not valid_cards:
        return ""

    blocks: list[str] = []
    for card in valid_cards:
        label = escape(card["label"])
        value = escape(card["value"])
        meta = escape(card.get("meta", ""))
        tone = escape(card.get("tone", "neutral"))
        meta_html = f'<div class="ts-stat-card__meta">{meta}</div>' if meta else ""
        blocks.append(
            f'<article class="ts-stat-card ts-stat-card--{tone}">'
            f'<div class="ts-stat-card__label">{label}</div>'
            f'<div class="ts-stat-card__value">{value}</div>'
            f"{meta_html}"
            "</article>"
        )
    return f'<div class="{escape(grid_class)}">{"".join(blocks)}</div>'


def render_surface_header(
    *,
    title: str,
    summary: str | None = None,
    eyebrow: str | None = None,
    tone: str = "default",
    st_module: Any = st,
) -> None:
    """Render a styled section header that works inside Streamlit layouts."""
    st_module.markdown(
        _build_surface_header_html(
            title=title,
            summary=summary,
            eyebrow=eyebrow,
            tone=tone,
        ),
        unsafe_allow_html=True,
    )


def render_stat_cards(
    cards: list[dict[str, str]],
    *,
    grid_class: str = "ts-stat-grid",
    st_module: Any = st,
) -> None:
    """Render a compact grid of highlight cards."""
    cards_html = _build_stat_cards_html(cards, grid_class=grid_class)
    if not cards_html:
        return
    st_module.markdown(cards_html, unsafe_allow_html=True)


def render_page_hero_deck(
    *,
    title: str,
    summary: str,
    eyebrow: str,
    cards: list[dict[str, str]],
    st_module: Any = st,
) -> None:
    """Render a page intro and metadata as one aligned dashboard deck."""
    cards_html = _build_stat_cards_html(cards, grid_class="ts-stat-grid ts-stat-grid--hero")
    st_module.markdown(
        (
            '<section class="ts-hero-deck">'
            '<div class="ts-hero-deck__lead">'
            f"{_build_surface_header_html(title=title, summary=summary, eyebrow=eyebrow)}"
            "</div>"
            '<div class="ts-hero-deck__meta">'
            f"{cards_html}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_prediction_hero_deck(
    *,
    title: str,
    summary: str,
    eyebrow: str,
    cards: list[dict[str, str]],
    st_module: Any = st,
) -> None:
    """Render the prediction-page intro with the generic page hero layout."""
    render_page_hero_deck(
        title=title,
        summary=summary,
        eyebrow=eyebrow,
        cards=cards,
        st_module=st_module,
    )


def render_notice_banner(
    message: str,
    *,
    tone: str = "info",
    label: str | None = None,
    st_module: Any = st,
) -> None:
    """Render a compact contextual notice instead of a large default alert."""
    normalized_message = str(message).strip()
    if not normalized_message:
        return

    label_html = f'<div class="ts-notice__label">{escape(label)}</div>' if label else ""
    st_module.markdown(
        (
            f'<div class="ts-notice ts-notice--{escape(tone)}">'
            '<div class="ts-notice__content">'
            f"{label_html}"
            f'<div class="ts-notice__body">{escape(normalized_message)}</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_stage_timeline(stages: list[dict[str, str]], *, st_module: Any = st) -> None:
    """Render a compact weekend-flow timeline for prediction sections."""
    valid_stages = [stage for stage in stages if stage.get("title")]
    if not valid_stages:
        return

    blocks: list[str] = []
    for index, stage in enumerate(valid_stages, start=1):
        title = escape(stage["title"])
        state = escape(stage.get("state", "Forecast"))
        meta = escape(stage.get("meta", ""))
        meta_html = f'<div class="ts-stage-card__meta">{meta}</div>' if meta else ""
        blocks.append(
            '<article class="ts-stage-card">'
            f'<div class="ts-stage-card__index">{index}</div>'
            f'<div class="ts-stage-card__title">{title}</div>'
            f'<div class="ts-stage-card__state">{state}</div>'
            f"{meta_html}"
            "</article>"
        )

    st_module.markdown(
        f'<div class="ts-stage-grid">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def _render_collapsible_warnings(messages: list[str], *, title: str) -> None:
    """Render warnings compactly to avoid notification spam."""
    unique_messages: list[str] = []
    for message in messages:
        normalized = str(message).strip()
        if normalized and normalized not in unique_messages:
            unique_messages.append(normalized)

    if not unique_messages:
        return
    if len(unique_messages) == 1:
        render_notice_banner(unique_messages[0], tone="warning", label="Warnings")
        return

    primary_warning = unique_messages[0]
    remaining_count = len(unique_messages) - 1
    render_notice_banner(
        f"{primary_warning} (+{remaining_count} more)",
        tone="warning",
        label="Warnings",
    )
    try:
        expander = st.expander(title, expanded=False)
    except TypeError:
        expander = st.expander(title)

    with expander:
        for message in unique_messages:
            st.markdown(f"- {message}")


def _build_team_clustering_warning(
    df: pd.DataFrame, *, mean_confidence: float | None = None
) -> str | None:
    """Build a warning when ordering has unusually many adjacent teammate pairs."""
    required_columns = {"team", "position"}
    if df.empty or not required_columns.issubset(df.columns):
        return None

    ordered = df.sort_values("position").reset_index(drop=True)
    if len(ordered) < 4:
        return None

    same_team_adjacent = int((ordered["team"] == ordered["team"].shift(1)).sum())
    cluster_threshold = max(4, int(len(ordered) * 0.20))
    if same_team_adjacent < cluster_threshold:
        return None

    confidence_note = ""
    if mean_confidence is not None and mean_confidence < 56.0:
        confidence_note = (
            " At this confidence level, part of this can come from priors, not only pace."
        )

    return (
        f"Team-clustered ordering: {same_team_adjacent} adjacent teammate pairs detected."
        f"{confidence_note}"
    )


def _short_data_source_label(data_source: object, *, blend_used: bool) -> str:
    """Return a concise label for the qualifying data source."""
    source_text = str(data_source).strip()
    normalized = source_text.lower()
    if "checkpoint profile blend" in normalized:
        return "Checkpoint blend"
    if blend_used:
        return "Practice blend"
    if "model-only" in normalized:
        return "Model only"
    if "testing" in normalized:
        return "Testing blend"
    if "actual" in normalized:
        return "Actual result"
    if source_text:
        return source_text if len(source_text) <= 18 else "Hybrid source"
    return "Unknown"


def _parse_optional_float(value: object) -> float | None:
    """Return a float for numeric inputs and ``None`` for missing or invalid values."""
    if value is None or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_track_temperature_context_card(result: dict) -> dict[str, str] | None:
    """Build a compact card for the race track-temperature context."""
    context = result.get("track_temperature_context")
    if not isinstance(context, dict):
        return None

    track_temp_c = _parse_optional_float(context.get("track_temperature_c"))
    if track_temp_c is None:
        return None

    source = str(context.get("source", "")).strip().lower()
    session_name = str(context.get("session_name", "")).strip()
    session_source = str(context.get("session_temperature_source", "")).strip().lower()
    session_label = session_name or "latest session"
    if session_source == "air_temp_inferred":
        session_label = f"{session_label} air->track inference"
    elif session_source == "track_temp":
        session_label = f"{session_label} track weather"

    meta = "Race-weather input."
    if source == "session_weather_blend":
        session_weight = _parse_optional_float(context.get("session_weight"))
        forecast_weight = _parse_optional_float(context.get("forecast_weight"))
        if session_weight is None or forecast_weight is None:
            meta = f"Blended from {session_label} and race-weather baseline."
        else:
            meta = (
                f"{int(round(session_weight * 100))}% {session_label} + "
                f"{int(round(forecast_weight * 100))}% race-weather baseline."
            )
    elif source == "session_weather":
        meta = f"Using {session_label}."
    elif source == "forecast_fallback":
        weather_bucket = str(context.get("weather_bucket", "dry")).strip().lower() or "dry"
        meta = f"Fallback from {weather_bucket} race-weather baseline."
    elif source == "track_params_override":
        meta = "Track-specific override applied."

    return {
        "label": "Track temp",
        "value": f"{track_temp_c:.1f}C",
        "meta": meta,
        "tone": "neutral",
    }


def _build_weather_feature_context_card(result: dict) -> dict[str, str] | None:
    """Build a compact card for weather feature selection details."""
    context = result.get("weather_feature_context")
    if not isinstance(context, dict) or not context.get("available"):
        return None

    source_session = str(context.get("source_session", "")).strip()
    if not source_session:
        return None

    practice_bucket = str(context.get("practice_weather_bucket", "unknown")).strip().lower()
    selected_bucket = str(context.get("selected_weather", "unknown")).strip().lower()
    chaos_multiplier = context.get("chaos_multiplier")
    meta = f"{source_session} practice was {practice_bucket}; scenario set to {selected_bucket}."
    if isinstance(chaos_multiplier, int | float):
        meta += f" Chaos x{float(chaos_multiplier):.2f}."

    return {
        "label": "Weather mode",
        "value": selected_bucket.upper(),
        "meta": meta,
        "tone": "neutral",
    }


def _build_prediction_highlight_cards(
    df: pd.DataFrame,
    result: dict,
    *,
    is_race: bool,
) -> list[dict[str, str]]:
    """Build quick-scan highlight cards for the active section."""
    if df.empty:
        return []

    ordered = df.sort_values("position", ascending=True).reset_index(drop=True)
    result_mode = str(result.get("result_mode", "")).strip().upper()

    if result_mode == "ACTUAL":
        leader = ordered.iloc[0]
        actual_cards = [
            {
                "label": "Session winner",
                "value": str(leader["driver"]),
                "meta": str(leader["team"]),
                "tone": "accent",
            },
            {
                "label": "Classification",
                "value": "Actual",
                "meta": "Completed-session result from FastF1.",
                "tone": "success",
            },
            {
                "label": "Field size",
                "value": str(len(ordered)),
                "meta": "Drivers in the published classification.",
                "tone": "neutral",
            },
        ]
        if is_race and len(ordered) >= 3:
            podium = " / ".join(str(item) for item in ordered.head(3)["driver"].tolist())
            actual_cards.insert(
                1,
                {
                    "label": "Podium",
                    "value": podium,
                    "meta": "Top three finishers.",
                    "tone": "neutral",
                },
            )
        return actual_cards

    if is_race:
        leader = ordered.iloc[0]
        race_cards: list[dict[str, str]] = [
            {
                "label": "Winner favorite",
                "value": str(leader["driver"]),
                "meta": str(leader["team"]),
                "tone": "accent",
            }
        ]
        if "confidence" in ordered.columns:
            confidence = pd.to_numeric(ordered["confidence"], errors="coerce").mean()
            if pd.notna(confidence):
                race_cards.append(
                    {
                        "label": "Mean confidence",
                        "value": f"{float(confidence):.1f}%",
                        "meta": "Average confidence across the full field.",
                        "tone": "neutral",
                    }
                )
        if "podium_probability" in ordered.columns:
            podium_leader = ordered.sort_values("podium_probability", ascending=False).iloc[0]
            race_cards.append(
                {
                    "label": "Podium leader",
                    "value": str(podium_leader["driver"]),
                    "meta": f"{float(podium_leader['podium_probability']):.1f}% podium chance.",
                    "tone": "neutral",
                }
            )
        if "dnf_probability" in ordered.columns:
            dnf_watch = ordered.sort_values("dnf_probability", ascending=False).iloc[0]
            race_cards.append(
                {
                    "label": "DNF watch",
                    "value": str(dnf_watch["driver"]),
                    "meta": f"{float(dnf_watch['dnf_probability']) * 100:.1f}% DNF risk.",
                    "tone": "warning",
                }
            )
        return race_cards

    pole = ordered.iloc[0]
    qualifying_cards = [
        {
            "label": "Pole favorite",
            "value": str(pole["driver"]),
            "meta": str(pole["team"]),
            "tone": "accent",
        }
    ]
    front_row = ordered.head(2)["driver"].tolist()
    if front_row:
        qualifying_cards.append(
            {
                "label": "Front row",
                "value": " / ".join(str(driver) for driver in front_row),
                "meta": "Current P1 and P2 projection.",
                "tone": "neutral",
            }
        )
    if "confidence" in ordered.columns:
        confidence = pd.to_numeric(ordered["confidence"], errors="coerce").mean()
        if pd.notna(confidence):
            qualifying_cards.append(
                {
                    "label": "Mean confidence",
                    "value": f"{float(confidence):.1f}%",
                    "meta": "Average confidence across the full grid.",
                    "tone": "neutral",
                }
            )
    data_source = result.get("data_source", "Unknown")
    qualifying_cards.append(
        {
            "label": "Data source",
            "value": _short_data_source_label(
                data_source, blend_used=bool(result.get("blend_used"))
            ),
            "meta": str(data_source),
            "tone": "neutral",
        }
    )
    return qualifying_cards


def _build_context_cards(result: dict, *, is_race: bool) -> list[dict[str, str]]:
    """Build supporting context cards that explain model inputs."""
    cards: list[dict[str, str]] = []

    if is_race:
        grid_source = str(result.get("grid_source", "")).strip().upper()
        if grid_source:
            meta = (
                str(result.get("starting_grid_note", "")).strip()
                if grid_source == "ACTUAL"
                else "Race starts from the predicted qualifying order."
            )
            cards.append(
                {
                    "label": "Grid source",
                    "value": "Actual" if grid_source == "ACTUAL" else "Predicted",
                    "meta": meta or "Starting order for this simulation.",
                    "tone": "success" if grid_source == "ACTUAL" else "neutral",
                }
            )

        characteristics_profile = str(result.get("characteristics_profile_used", "")).strip()
        teams_with_profile = result.get("teams_with_characteristics_profile", 0)
        if characteristics_profile and teams_with_profile:
            cards.append(
                {
                    "label": "Car profile",
                    "value": characteristics_profile,
                    "meta": f"{int(teams_with_profile)} teams with characteristics inputs.",
                    "tone": "neutral",
                }
            )

        track_temp_card = _build_track_temperature_context_card(result)
        if track_temp_card:
            cards.append(track_temp_card)

        weather_card = _build_weather_feature_context_card(result)
        if weather_card:
            cards.append(weather_card)
        return cards

    grid_source = str(result.get("grid_source", "")).strip().upper()
    if grid_source:
        cards.append(
            {
                "label": "Grid source",
                "value": "Actual" if grid_source == "ACTUAL" else "Predicted",
                "meta": (
                    "Completed-session classification is already available."
                    if grid_source == "ACTUAL"
                    else "Grid still comes from the qualifying model."
                ),
                "tone": "success" if grid_source == "ACTUAL" else "neutral",
            }
        )
    return cards


def _prediction_section_summary(result: dict, *, is_race: bool) -> str:
    """Return the short section summary shown in the surface header."""
    result_mode = str(result.get("result_mode", "")).strip().upper()
    if result_mode == "ACTUAL":
        return str(result.get("classification_note", "")).strip() or (
            "Completed-session classification from FastF1."
        )

    if is_race:
        return (
            "Race distribution ranked by expected finish, with podium, risk, and strategy "
            "signals summarized before the full-field table."
        )

    return (
        "Grid projection grouped by elimination stage so sprint weekends stay readable even "
        "when two separate qualifying sessions are on the page."
    )
