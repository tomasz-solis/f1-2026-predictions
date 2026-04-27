"""Qualifying-specific rendering helpers for dashboard prediction output."""

from html import escape
from typing import TypedDict

import pandas as pd
import streamlit as st

from src.dashboard.rendering_html import render_notice_banner


class _TeammateMatchupRow(TypedDict):
    """Display-ready teammate matchup row."""

    team: str
    favorite: str
    underdog: str
    favorite_probability: float
    edge_delta: float
    edge_strength: str
    tone: str
    n_samples: int
    rank: int


def _render_qualifying_result(df: pd.DataFrame) -> None:
    """Render qualifying prediction grouped by elimination stage."""
    df_display = df[["position", "driver", "team"]].copy()
    df_display.columns = ["Grid", "Driver", "Team"]
    has_ci = "p5" in df.columns and "p95" in df.columns
    if has_ci:
        df_display["90% Range"] = df.apply(lambda r: f"P{int(r['p5'])}-P{int(r['p95'])}", axis=1)
        st.caption(
            "`90% Range` shows where each driver lands in 90% of qualifying simulations. "
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


def _coerce_probability_percent(value: object) -> float | None:
    """Return a 0-100 probability value from either a fraction or percent."""
    if not isinstance(value, int | float | str):
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= probability <= 1.0:
        probability *= 100.0
    if not 0.0 <= probability <= 100.0:
        return None
    return probability


def _describe_teammate_edge(probability: float) -> str:
    """Map favorite-ahead probability to a plain-language edge label."""
    if probability < 55.0:
        return "too close to call"
    if probability < 65.0:
        return "slight edge"
    if probability < 75.0:
        return "moderate edge"
    if probability < 85.0:
        return "clear edge"
    return "strong edge"


def _teammate_edge_tone(probability: float) -> str:
    """Return a CSS tone name for a teammate matchup edge."""
    if probability < 55.0:
        return "neutral"
    if probability < 65.0:
        return "slight"
    if probability < 75.0:
        return "moderate"
    if probability < 85.0:
        return "clear"
    return "strong"


def _normalize_teammate_matchups(
    probabilities: list[dict[str, object]],
) -> list[_TeammateMatchupRow]:
    """Clean, flip, and order teammate matchups by strongest simulated edge."""
    rows: list[_TeammateMatchupRow] = []
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
        if raw_probability is None:
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
        probability = _coerce_probability_percent(raw_probability)
        if probability is None:
            continue

        if probability >= 50.0:
            favorite = driver_a
            underdog = driver_b
            favorite_probability = probability
        else:
            favorite = driver_b
            underdog = driver_a
            favorite_probability = 100.0 - probability

        edge_delta = favorite_probability - 50.0
        rows.append(
            {
                "team": team,
                "favorite": favorite,
                "underdog": underdog,
                "favorite_probability": favorite_probability,
                "edge_delta": edge_delta,
                "edge_strength": _describe_teammate_edge(favorite_probability),
                "tone": _teammate_edge_tone(favorite_probability),
                "n_samples": n_samples,
                "rank": 0,
            }
        )

    rows.sort(
        key=lambda item: (
            -item["edge_delta"],
            -item["favorite_probability"],
            item["team"],
            item["favorite"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _render_teammate_matchup_cards(rows: list[_TeammateMatchupRow]) -> None:
    """Render teammate matchup rows as compact, ordered cards."""
    cards: list[str] = []
    for row in rows:
        team = escape(row["team"])
        favorite = escape(row["favorite"])
        underdog = escape(row["underdog"])
        edge_strength = escape(row["edge_strength"])
        tone = escape(row["tone"])
        rank = row["rank"]
        favorite_probability = row["favorite_probability"]
        edge_delta = row["edge_delta"]
        n_samples = row["n_samples"]
        edge_width = max(0.0, min(50.0, edge_delta))
        advantage_label = f"+{edge_delta:.1f} pp toward {favorite}"
        meter_label = f"{favorite} has a {edge_delta:.1f} point simulated advantage over {underdog}"
        cards.append(
            f'<article class="ts-matchup-card ts-matchup-card--{tone}">'
            '<div class="ts-matchup-card__top">'
            '<span class="ts-matchup-card__identity">'
            f'<span class="ts-matchup-card__rank">#{rank}</span>'
            f'<span class="ts-matchup-card__team">{team}</span>'
            "</span>"
            f'<span class="ts-matchup-card__tag">{edge_strength}</span>'
            "</div>"
            f'<div class="ts-matchup-card__line" aria-label="{favorite} over {underdog}">'
            f'<span class="ts-matchup-card__favorite">{favorite}</span>'
            '<span class="ts-matchup-card__vs">over</span>'
            f"<span>{underdog}</span>"
            "</div>"
            '<div class="ts-matchup-card__meta">'
            f"<span>{favorite_probability:.1f}% ahead</span>"
            f'<span class="ts-matchup-card__advantage">{advantage_label}</span>'
            "</div>"
            '<div class="ts-matchup-card__bar ts-matchup-card__bar--favorite-left" '
            f'aria-label="{escape(meter_label)}">'
            f'<span style="width: {edge_width:.1f}%"></span>'
            "</div>"
            '<div class="ts-matchup-card__scale">'
            f"<span>{favorite}</span>"
            "<span>50/50</span>"
            f"<span>{underdog}</span>"
            "</div>"
            f'<div class="ts-matchup-card__samples">{n_samples:,} sims</div>'
            "</article>"
        )

    st.markdown(
        f'<div class="ts-matchup-list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_teammate_head_to_head_probabilities(probabilities: list[dict[str, object]]) -> None:
    """Render teammate head-to-head probabilities derived from qualifying simulations."""
    rows = _normalize_teammate_matchups(probabilities)

    if not rows:
        return
    try:
        expander = st.expander("Teammate Matchups (Who Has The Edge?)", expanded=False)
    except TypeError:
        expander = st.expander("Teammate Matchups (Who Has The Edge?)")

    with expander:
        st.markdown(
            "How to read: sorted by largest simulated teammate edge first. "
            "The meter starts at 50/50; the colored segment points toward the favored driver."
        )
        _render_teammate_matchup_cards(rows)
