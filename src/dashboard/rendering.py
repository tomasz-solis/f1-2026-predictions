"""Dashboard rendering helpers for prediction outputs."""

from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
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


def render_stat_cards(cards: list[dict[str, str]], *, st_module: Any = st) -> None:
    """Render a compact grid of highlight cards."""
    cards_html = _build_stat_cards_html(cards)
    if not cards_html:
        return
    st_module.markdown(cards_html, unsafe_allow_html=True)


def render_prediction_hero_deck(
    *,
    title: str,
    summary: str,
    eyebrow: str,
    cards: list[dict[str, str]],
    st_module: Any = st,
) -> None:
    """Render the prediction-page intro and metadata as one aligned deck."""
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
        f"🧩 Team-clustered ordering: {same_team_adjacent} adjacent teammate pairs detected."
        f"{confidence_note}"
    )


def _short_data_source_label(data_source: object, *, blend_used: bool) -> str:
    """Return a concise label for the qualifying data source."""
    source_text = str(data_source).strip()
    normalized = source_text.lower()
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


def _position_change_chart_title(prediction_name: str, result: dict) -> str:
    """Build a concise label for the two-session position comparison."""
    starting_session = str(result.get("starting_session_name", "")).strip().upper()
    if "sprint race" in prediction_name.lower():
        return f"{starting_session or 'SQ'} -> Sprint"
    if "main race" in prediction_name.lower():
        return f"{starting_session or 'Q'} -> Race"
    if "race" in prediction_name.lower():
        return f"{starting_session or 'Q'} -> Race"
    return "Grid -> Finish"


def _movement_bar_labels(rows: pd.DataFrame) -> list[str]:
    """Build concise labels for movement bars."""
    labels: list[str] = []
    for row in rows.itertuples(index=False):
        labels.append(f"{row.driver}  P{int(row.start_position)} -> P{int(row.finish_position)}")
    return labels


def _position_change_chart_figure(
    rows: pd.DataFrame,
    *,
    title: str,
    marker_color: str,
    x_limit: int,
) -> go.Figure:
    """Build a horizontal bar chart for either projected gainers or losers."""
    labels = _movement_bar_labels(rows)
    deltas = rows["positions_gained"].abs().astype(int).tolist()
    text = [f"{delta:+d}" if title == "Gainers" else f"-{delta}" for delta in deltas]

    fig = go.Figure(
        data=[
            go.Bar(
                x=deltas,
                y=labels,
                orientation="h",
                marker={
                    "color": marker_color,
                    "line": {"width": 0},
                },
                text=text,
                textposition="outside",
                customdata=[
                    [
                        row.driver,
                        row.team,
                        int(row.start_position),
                        int(row.finish_position),
                        int(row.positions_gained),
                    ]
                    for row in rows.itertuples(index=False)
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})"
                    "<br>Start: P%{customdata[2]}"
                    "<br>Finish: P%{customdata[3]}"
                    "<br>Net: %{customdata[4]:+d}<extra></extra>"
                ),
                cliponaxis=False,
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        height=max(240, 70 * len(rows) + 40),
        margin={"l": 16, "r": 70, "t": 44, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 16, "color": "rgba(232,237,242,0.94)"},
        },
        xaxis={
            "title": {
                "text": "Places",
                "font": {"size": 11, "color": "rgba(232,237,242,0.72)"},
            },
            "range": [0, max(1, x_limit) + 0.6],
            "tickmode": "linear",
            "dtick": 1,
            "gridcolor": "rgba(232,237,242,0.08)",
            "zeroline": False,
            "tickfont": {"size": 11, "color": "rgba(232,237,242,0.76)"},
            "fixedrange": True,
        },
        yaxis={
            "autorange": "reversed",
            "tickfont": {"size": 12, "color": "rgba(232,237,242,0.88)"},
            "fixedrange": True,
        },
        font={"family": "IBM Plex Sans, sans-serif", "color": "rgba(232,237,242,0.88)"},
    )
    return fig


def _build_position_change_frame(
    finish_df: pd.DataFrame,
    starting_grid: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge starting-grid and finish positions into one comparison frame."""
    if finish_df.empty or not starting_grid:
        return pd.DataFrame()

    start_df = pd.DataFrame(starting_grid)
    required_columns = {"driver", "team", "position"}
    if not required_columns.issubset(start_df.columns) or not required_columns.issubset(
        finish_df.columns
    ):
        return pd.DataFrame()

    start_df = start_df[["driver", "team", "position"]].copy()
    start_df = start_df.rename(columns={"position": "start_position"})
    finish_positions = finish_df[["driver", "team", "position"]].copy()
    finish_positions = finish_positions.rename(columns={"position": "finish_position"})
    start_df["start_position"] = pd.to_numeric(start_df["start_position"], errors="coerce")
    finish_positions["finish_position"] = pd.to_numeric(
        finish_positions["finish_position"], errors="coerce"
    )

    merged = start_df.merge(
        finish_positions,
        on="driver",
        how="inner",
        suffixes=("_start", "_finish"),
    )
    if merged.empty:
        return merged
    merged = merged.dropna(subset=["start_position", "finish_position"]).copy()
    if merged.empty:
        return merged
    merged["start_position"] = merged["start_position"].astype(int)
    merged["finish_position"] = merged["finish_position"].astype(int)

    merged["team"] = merged["team_finish"].fillna(merged["team_start"])
    merged["positions_gained"] = merged["start_position"] - merged["finish_position"]
    merged["abs_change"] = merged["positions_gained"].abs()
    merged = merged.sort_values(
        ["positions_gained", "finish_position", "driver"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return merged[["driver", "team", "start_position", "finish_position", "positions_gained"]]


def _render_position_change_chart(
    finish_df: pd.DataFrame,
    *,
    result: dict,
    prediction_name: str,
) -> None:
    """Render compact gainers and losers bars from grid to projected finish."""
    starting_grid = result.get("starting_grid")
    if not isinstance(starting_grid, list) or not starting_grid:
        return

    comparison = _build_position_change_frame(finish_df, starting_grid)
    if comparison.empty:
        return

    top_gainers = comparison[comparison["positions_gained"] > 0].head(5).copy()
    top_losers = (
        comparison[comparison["positions_gained"] < 0]
        .sort_values(["positions_gained", "start_position", "driver"])
        .head(5)
        .copy()
    )
    if top_gainers.empty:
        summary = "No major gainers in this run; the model expects a fairly grid-shaped race."
    else:
        summary = "Biggest projected gainers: " + ", ".join(
            f"{row.driver} +{int(row.positions_gained)}"
            for row in top_gainers.itertuples(index=False)
        )

    render_surface_header(
        title="Biggest Movers",
        summary=(
            "Net position change from the paired grid to the projected finish. "
            "Bars isolate the clearest gainers and losers instead of showing the full field."
        ),
        eyebrow=_position_change_chart_title(prediction_name, result),
    )
    st.caption(summary)

    max_change = int(comparison["positions_gained"].abs().max())
    gainers_col, losers_col = st.columns(2, gap="large")
    with gainers_col:
        if top_gainers.empty:
            render_notice_banner(
                "No projected gainers in this run.",
                tone="info",
                label="Gainers",
            )
        else:
            st.plotly_chart(
                _position_change_chart_figure(
                    top_gainers,
                    title="Gainers",
                    marker_color="#76D3B3",
                    x_limit=max_change,
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
    with losers_col:
        if top_losers.empty:
            render_notice_banner(
                "No projected losers in this run.",
                tone="info",
                label="Losers",
            )
        else:
            st.plotly_chart(
                _position_change_chart_figure(
                    top_losers,
                    title="Losers",
                    marker_color="#C28657",
                    x_limit=max_change,
                ),
                width="stretch",
                config={"displayModeBar": False},
            )


def _render_track_temperature_context(result: dict) -> None:
    """Render race track-temperature source and blend details when available."""
    context = result.get("track_temperature_context")
    if not isinstance(context, dict):
        return

    raw_temp = context.get("track_temperature_c")
    if raw_temp is None:
        return
    try:
        track_temp_c = float(raw_temp)
    except (TypeError, ValueError):
        return

    source = str(context.get("source", "")).strip().lower()
    session_name_raw = context.get("session_name")
    session_name = str(session_name_raw).strip() if session_name_raw else ""
    session_source = str(context.get("session_temperature_source", "")).strip().lower()

    raw_session_weight = context.get("session_weight")
    raw_forecast_weight = context.get("forecast_weight")
    session_weight: float | None
    forecast_weight: float | None

    if raw_session_weight is None:
        session_weight = None
    else:
        try:
            session_weight = float(raw_session_weight)
        except (TypeError, ValueError):
            session_weight = None

    if raw_forecast_weight is None:
        forecast_weight = None
    else:
        try:
            forecast_weight = float(raw_forecast_weight)
        except (TypeError, ValueError):
            forecast_weight = None

    session_label = session_name or "latest session"
    if session_source == "air_temp_inferred":
        session_label = f"{session_label} weather (air->track inferred)"
    elif session_source == "track_temp":
        session_label = f"{session_label} weather"

    if source == "session_weather_blend":
        if session_weight is not None and forecast_weight is not None:
            session_pct = int(round(session_weight * 100))
            forecast_pct = int(round(forecast_weight * 100))
            st.info(
                f"Track temperature input: {track_temp_c:.1f}C "
                f"({session_pct}% {session_label} + {forecast_pct}% race-weather baseline)"
            )
        else:
            st.info(
                f"Track temperature input: {track_temp_c:.1f}C "
                f"(blended from {session_label} and race-weather baseline)"
            )
        return

    if source == "session_weather":
        st.info(f"Track temperature input: {track_temp_c:.1f}C ({session_label})")
        return

    if source == "forecast_fallback":
        weather_bucket = str(context.get("weather_bucket", "dry")).strip().lower() or "dry"
        st.info(
            f"Track temperature input: {track_temp_c:.1f}C "
            f"(race-weather fallback: {weather_bucket})"
        )
        return

    if source == "track_params_override":
        st.info(f"Track temperature input: {track_temp_c:.1f}C (track-specific override)")
        return

    st.info(f"Track temperature input: {track_temp_c:.1f}C")


def _render_weather_feature_context(result: dict) -> None:
    """Render non-competitive weather feature source and applied modifiers."""
    context = result.get("weather_feature_context")
    if not isinstance(context, dict):
        return
    if not context.get("available"):
        return

    source_session = str(context.get("source_session", "")).strip()
    if not source_session:
        return

    practice_bucket = str(context.get("practice_weather_bucket", "unknown")).strip().lower()
    selected_bucket = str(context.get("selected_weather", "unknown")).strip().lower()
    chaos_multiplier = context.get("chaos_multiplier")

    message = (
        f"Weather feature input: {source_session} practice weather ({practice_bucket}). "
        f"Scenario selected: {selected_bucket}."
    )
    if isinstance(chaos_multiplier, (int | float)):
        message += f" Uncertainty adjustment active (chaos x{float(chaos_multiplier):.2f})."
    st.info(message)


def _render_compound_strategies(compound_strategies: dict) -> None:
    st.subheader("Tire Compound Strategies")

    sorted_strategies = sorted(compound_strategies.items(), key=lambda x: x[1], reverse=True)

    cols = st.columns(min(3, len(sorted_strategies)))
    for idx, (strategy, frequency) in enumerate(sorted_strategies[:3]):
        with cols[idx]:
            percentage = frequency * 100
            st.metric(
                label=strategy,
                value=f"{percentage:.1f}%",
                help="Frequency of this compound sequence across simulations",
            )

    if len(sorted_strategies) > 3:
        with st.expander("View all strategies"):
            for strategy, frequency in sorted_strategies:
                percentage = frequency * 100
                st.write(f"**{strategy}**: {percentage:.1f}%")


def _render_pit_lap_distribution(pit_lap_distribution: dict) -> None:
    st.subheader("Pit Stop Windows")

    sorted_pit_laps = sorted(
        pit_lap_distribution.items(),
        key=lambda x: int(x[0].split("_")[1].split("-")[0]),
    )

    total_stops = sum(count for _, count in sorted_pit_laps) or 1

    windows = []
    for lap_bin, count in sorted_pit_laps:
        label = lap_bin.replace("lap_", "L")  # lap_25-30 -> L25-30
        pct = 100 * (count / total_stops)
        windows.append((label, count, pct))

    top_windows = sorted(windows, key=lambda x: x[2], reverse=True)[:5]

    st.caption(
        "Share of all simulated pit events (all cars × all simulations). "
        "Windows are 5-lap bins, e.g. L25–30."
    )

    most_likely = top_windows[0]
    st.info(f"Most likely pit window: **{most_likely[0]}** ({most_likely[2]:.1f}%)")

    cols = st.columns(len(top_windows))
    for col, (label, count, pct) in zip(cols, top_windows, strict=False):
        with col:
            st.metric(
                label,
                f"{pct:.1f}%",
                help=f"{count:,} of {total_stops:,} simulated pit events",
            )
            st.progress(min(pct / 100, 1.0))

    with st.expander("View full pit stop distribution"):
        dist_df = pd.DataFrame(windows, columns=["Window", "Stops", "Share %"])
        dist_df["Share %"] = dist_df["Share %"].round(2)
        st.dataframe(dist_df, width="stretch", hide_index=True)


def _style_race_table(df_display: pd.DataFrame):
    def color_position(val):
        if val == 1:
            return (
                "background-color: rgba(255,215,0,0.18);"
                "border-left: 4px solid #FFD700;"
                "font-weight: 800;"
                "color: rgba(237,239,243,0.95);"
            )
        if val == 2:
            return (
                "background-color: rgba(192,192,192,0.14);"
                "border-left: 4px solid #C0C0C0;"
                "font-weight: 800;"
                "color: rgba(237,239,243,0.95);"
            )
        if val == 3:
            return (
                "background-color: rgba(205,127,50,0.16);"
                "border-left: 4px solid #CD7F32;"
                "font-weight: 800;"
                "color: rgba(237,239,243,0.95);"
            )

        if val <= 10:
            return (
                "background-color: rgba(227,242,253,0.07);"
                "border-left: 4px solid rgba(227,242,253,0.30);"
                "font-weight: 800;"
                "color: rgba(237,239,243,0.95);"
            )

        return "border-left: 4px solid transparent; color: rgba(237,239,243,0.88);"

    def color_dnf_risk(val):
        if val > 20:
            return "background-color: rgba(198,40,40,0.22); color: rgba(255,255,255,0.92); font-weight: 700;"
        if val >= 10:
            return "background-color: rgba(245,127,23,0.20); color: rgba(255,255,255,0.92); font-weight: 700;"
        return "background-color: rgba(46,125,50,0.18); color: rgba(237,239,243,0.92); font-weight: 700;"

    def highlight_expected_position(val):
        _ = val
        return (
            "background-color: rgba(66,165,245,0.16);"
            "border-left: 3px solid rgba(66,165,245,0.55);"
            "font-weight: 800;"
            "color: rgba(237,239,243,0.96);"
        )

    styled_df = (
        df_display.style.set_properties(
            **{
                "background-color": "#10141c",
                "color": "rgba(237,239,243,0.88)",
                "border-color": "rgba(255,255,255,0.06)",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "td",
                    "props": [
                        ("border-color", "rgba(255,255,255,0.06)"),
                        ("font-variant-numeric", "tabular-nums"),
                    ],
                },
                {
                    "selector": "td:nth-child(1)",
                    "props": [
                        ("font-size", "0.98rem"),
                        ("font-weight", "800"),
                        ("text-align", "center"),
                        ("width", "64px"),
                    ],
                },
            ]
        )
        .map(color_position, subset=["Pos"])
        .map(color_dnf_risk, subset=["DNF Risk %"])
        .format(
            {
                "Expected Pos": "{:.2f}",
                "Confidence %": "{:.1f}",
                "Podium %": "{:.1f}",
                "DNF Risk %": "{:.1f}",
            }
        )
    )
    if "Expected Pos" in df_display.columns:
        styled_df = styled_df.map(highlight_expected_position, subset=["Expected Pos"])

    try:
        styled_df = styled_df.hide(axis="index")
    except (AttributeError, TypeError):
        pass

    return styled_df


def _render_race_result(df: pd.DataFrame) -> None:
    """Render race prediction table and summary cards."""
    race_df = df.copy()
    race_df["confidence"] = race_df["confidence"].round(1)
    race_df["podium_probability"] = race_df["podium_probability"].round(1)
    race_df["dnf_probability"] = (race_df["dnf_probability"] * 100).round(1)
    has_expected_position = "position_blend_score" in race_df.columns
    if has_expected_position:
        race_df["expected_position"] = race_df["position_blend_score"].astype(float).round(2)

    # Build 90% position interval string when percentile columns are present.
    has_ci = "p5" in race_df.columns and "p95" in race_df.columns
    if has_ci:
        race_df["ci_range"] = race_df.apply(lambda r: f"P{int(r['p5'])}–P{int(r['p95'])}", axis=1)

    input_confidence = race_df.attrs.get("input_confidence")

    warnings: list[str] = []
    mean_confidence = float(race_df["confidence"].mean()) if not race_df.empty else 0.0
    if mean_confidence < 56.0:
        warnings.append(
            f"⚠️ Low confidence run: mean confidence is {mean_confidence:.1f}%. "
            "Use this as a rough order; it should move as more weekend data comes in."
        )

    if isinstance(input_confidence, int | float) and float(input_confidence) < 0.60:
        warnings.append(
            f"⚠️ Low input-data confidence ({float(input_confidence):.2f}/1.00). "
            "This run leans heavily on priors."
        )

    if has_ci:
        interval_width = (race_df["p95"] - race_df["p5"]).astype(float)
        median_width = float(interval_width.median())
        wide_ranges = int((interval_width >= 8.0).sum())
        if wide_ranges >= max(6, int(len(race_df) * 0.35)):
            warnings.append(
                f"📏 Wide position ranges: {wide_ranges} drivers have 90% ranges spanning 8+ places "
                f"(median span: {median_width:.1f})."
            )

    high_dnf = race_df[race_df["dnf_probability"] > 20]
    if not high_dnf.empty:
        warnings.append(
            f"🛑 High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}"
        )
    team_cluster_warning = _build_team_clustering_warning(
        race_df,
        mean_confidence=mean_confidence,
    )
    if team_cluster_warning:
        warnings.append(team_cluster_warning)

    _render_collapsible_warnings(warnings, title="⚠️ Race Warnings")

    st.caption(
        "Rows are ranked by expected finishing position across the full simulation "
        "distribution, not by Confidence% or Podium%."
    )
    st.caption(
        "Key signal: `Expected Pos` (lower is better). Use `90% Pos Range` to judge uncertainty."
    )
    st.caption(
        "`90% Pos Range` shows where a driver lands in 90% of simulations (P5 to P95). "
        "Equal Podium% values are normal because podium probabilities are monotonic-smoothed."
    )

    display_cols = ["position", "driver", "team"]
    display_names = ["Pos", "Driver", "Team"]
    if has_expected_position:
        display_cols.append("expected_position")
        display_names.append("Expected Pos")
    if has_ci:
        display_cols.append("ci_range")
        display_names.append("90% Pos Range")
    display_cols += ["podium_probability", "dnf_probability", "confidence"]
    display_names += ["Podium %", "DNF Risk %", "Confidence %"]

    df_display = race_df[display_cols].copy()
    df_display.columns = display_names

    st.subheader("🏁 Predicted Podium")
    podium = race_df[race_df["position"] <= 3].copy().sort_values("position", ascending=True)
    podium_cards: list[dict[str, str]] = []
    podium_order = [2, 1, 3]
    for position in podium_order:
        row = podium[podium["position"] == position]
        if row.empty:
            continue
        podium_row = row.iloc[0]
        podium_cards.append(
            {
                "label": f"P{position}",
                "value": str(podium_row["driver"]),
                "meta": (
                    f"{podium_row['team']} • {float(podium_row['confidence']):.1f}% confidence"
                ),
                "tone": "accent" if position == 1 else "neutral",
            }
        )
    render_stat_cards(podium_cards)

    st.caption("Projected top 10 shown first. Expand for the full P1-P22 simulation table.")
    top_ten_display = df_display.head(10)
    styled_top_ten = _style_race_table(top_ten_display)
    st.markdown(
        f'<div class="rc-table">{styled_top_ten.to_html()}</div>',
        unsafe_allow_html=True,
    )

    try:
        expander = st.expander("Full Simulation Table (P1-P22)", expanded=False)
    except TypeError:
        expander = st.expander("Full Simulation Table (P1-P22)")

    with expander:
        styled_full = _style_race_table(df_display)
        st.markdown(
            f'<div class="rc-table">{styled_full.to_html()}</div>',
            unsafe_allow_html=True,
        )


def _render_qualifying_result(df: pd.DataFrame) -> None:
    """Render qualifying prediction grouped by elimination stage."""
    df_display = df[["position", "driver", "team"]].copy()
    df_display.columns = ["Grid", "Driver", "Team"]
    has_ci = "p5" in df.columns and "p95" in df.columns
    if has_ci:
        df_display["90% Range"] = df.apply(lambda r: f"P{int(r['p5'])}-P{int(r['p95'])}", axis=1)
        st.caption(
            "📊 `90% Range` shows where each driver lands in 90% of qualifying simulations. "
            "Ranges should tighten as weekend and season data accumulates."
        )
    st.caption(
        "Read left to right as qualifying stages (Q1 -> Q2 -> Q3). "
        "`Grid` remains the full projected final order."
    )

    if has_ci and len(df) >= 2:
        top_a = df.iloc[0]
        top_b = df.iloc[1]
        same_team = str(top_a.get("team", "")) == str(top_b.get("team", ""))
        try:
            a_p5 = int(top_a["p5"])
            a_p95 = int(top_a["p95"])
            b_p5 = int(top_b["p5"])
            b_p95 = int(top_b["p95"])
        except (TypeError, ValueError, KeyError):
            same_team = False
            a_p5 = a_p95 = b_p5 = b_p95 = 0

        ranges_overlap = not (a_p95 < b_p5 or b_p95 < a_p5)
        if same_team and ranges_overlap:
            render_notice_banner(
                (
                    "Front-row projection is statistically tight: teammate ranges overlap, "
                    "so the P1/P2 ordering can flip between close scenarios."
                ),
                tone="info",
                label="Front row",
            )

    stage_sections = [
        ("Q1 Eliminated (Final Grid P17-P22)", df_display.iloc[16:22]),
        ("Q2 Eliminated (Final Grid P11-P16)", df_display.iloc[10:16]),
        ("Q3 Shootout (Final Grid P1-P10)", df_display.head(10)),
    ]
    columns = st.columns(len(stage_sections))
    for column, (label, section_df) in zip(columns, stage_sections, strict=False):
        with column:
            st.markdown(f"**{label}**")
            st.markdown(
                f'<div class="rc-table">{section_df.to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )


def _render_actual_classification(df: pd.DataFrame, *, caption: str) -> None:
    """Render a completed-session classification without prediction-specific extras."""
    df_display = df[["position", "driver", "team"]].copy()
    df_display = df_display.sort_values("position", ascending=True)
    df_display.columns = ["Pos", "Driver", "Team"]
    st.caption(caption)
    st.markdown(
        f'<div class="rc-table">{df_display.to_html(index=False)}</div>',
        unsafe_allow_html=True,
    )


def _render_teammate_head_to_head_probabilities(probabilities: list[dict[str, object]]) -> None:
    """Render teammate head-to-head probabilities derived from qualifying simulations."""

    def _describe_probability(probability: float) -> str:
        """Map numeric head-to-head probability to plain-language edge strength."""
        if probability < 55.0:
            return "too close to call"
        if probability < 65.0:
            return "slight edge"
        if probability < 75.0:
            return "moderate edge"
        if probability < 85.0:
            return "clear edge"
        return "strong edge"

    rows: list[tuple[str, str, str, float, int]] = []
    for item in probabilities:
        if not isinstance(item, dict):
            continue
        team = str(item.get("team", "")).strip()
        driver_a = str(item.get("driver_a", "")).strip()
        driver_b = str(item.get("driver_b", "")).strip()
        raw_probability = item.get("p_driver_a_ahead")
        raw_samples = item.get("n_samples")
        if not team or not driver_a or not driver_b:
            continue
        if not isinstance(raw_probability, (int | float)):
            continue
        if not isinstance(raw_samples, int | float | str):
            n_samples = 0
        else:
            try:
                n_samples = int(raw_samples)
            except (TypeError, ValueError):
                n_samples = 0
        if n_samples <= 0:
            continue
        probability = float(raw_probability) * 100.0
        rows.append((team, driver_a, driver_b, probability, n_samples))

    if not rows:
        return

    rows.sort(key=lambda item: item[3], reverse=True)
    try:
        expander = st.expander("Teammate Matchups (Who Has The Edge?)", expanded=False)
    except TypeError:
        expander = st.expander("Teammate Matchups (Who Has The Edge?)")

    with expander:
        st.markdown(
            "How to read: around 50% means a coin flip, 60-70% means a slight edge, "
            "70-80% means a clear edge, and 80%+ means a strong favorite."
        )
        for team, driver_a, driver_b, probability, n_samples in rows:
            edge_strength = _describe_probability(probability)
            st.markdown(
                f"- **{team}**: {driver_a} over {driver_b} -> **{edge_strength}** "
                f"({probability:.1f}%, based on {n_samples} simulations)."
            )


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
            if isinstance(fp_blend_weight_used, (int | float)):
                practice_share = int(round(float(fp_blend_weight_used) * 100))
                model_share = max(0, 100 - practice_share)
                render_notice_banner(
                    (
                        f"Data source: {data_source} "
                        f"({practice_share}% practice data + {model_share}% model)."
                    ),
                    tone="info",
                    label="Input mix",
                )
            else:
                render_notice_banner(
                    f"Data source: {data_source} (70% practice data + 30% model).",
                    tone="info",
                    label="Input mix",
                )
        else:
            render_notice_banner(f"Data source: {data_source}.", tone="info", label="Input mix")
            if isinstance(data_source, str) and "Model-only" in data_source:
                qualifying_warning_messages.append(
                    "⚠️ Low-confidence qualifying mode: no weekend practice/testing signal. "
                    "Early grids can look too team-ordered."
                )
            elif isinstance(data_source, str) and "Testing short-run profile blend" in data_source:
                qualifying_warning_messages.append(
                    "⚠️ Medium-confidence qualifying mode: using testing-derived team pace without "
                    "weekend laps. Expect wider position ranges."
                )
        if "confidence" in df.columns and not df.empty:
            mean_qualifying_confidence = float(
                pd.to_numeric(df["confidence"], errors="coerce").mean()
            )
            if mean_qualifying_confidence < 56.0:
                qualifying_warning_messages.append(
                    f"⚠️ Low confidence run: mean confidence is {mean_qualifying_confidence:.1f}%. "
                    "Use this as a rough order."
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
                    f"📏 Wide position ranges: {wide_ranges} drivers have 90% ranges spanning 8+ places."
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
            title="⚠️ Qualifying Warnings",
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
