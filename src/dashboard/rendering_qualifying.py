"""Qualifying-specific rendering helpers for dashboard prediction output."""

import pandas as pd
import streamlit as st

from src.dashboard.rendering_html import render_notice_banner


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
