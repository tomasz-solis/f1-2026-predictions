"""Race-specific rendering helpers for dashboard prediction output."""

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.rendering_html import (
    _build_team_clustering_warning,
    _render_collapsible_warnings,
    render_notice_banner,
    render_stat_cards,
    render_surface_header,
)

logger = logging.getLogger(__name__)


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


def _build_movement_story_cards(comparison: pd.DataFrame) -> list[dict[str, str]]:
    """Build compact cards that summarize the race movement story."""
    if comparison.empty:
        return []

    cards: list[dict[str, str]] = []
    top_gainers = comparison[comparison["positions_gained"] > 0].copy()
    top_losers = comparison[comparison["positions_gained"] < 0].copy()

    if not top_gainers.empty:
        gainer = top_gainers.iloc[0]
        gain = int(gainer["positions_gained"])
        cards.append(
            {
                "label": "Biggest gainer",
                "value": f"{gainer['driver']} +{gain}",
                "meta": (
                    f"P{int(gainer['start_position'])} -> "
                    f"P{int(gainer['finish_position'])} | {gainer['team']}"
                ),
                "tone": "success",
            }
        )

    if not top_losers.empty:
        loser = top_losers.sort_values(["positions_gained", "start_position", "driver"]).iloc[0]
        loss = int(abs(loser["positions_gained"]))
        cards.append(
            {
                "label": "Biggest drop",
                "value": f"{loser['driver']} -{loss}",
                "meta": (
                    f"P{int(loser['start_position'])} -> "
                    f"P{int(loser['finish_position'])} | {loser['team']}"
                ),
                "tone": "warning",
            }
        )

    unchanged_count = int((comparison["positions_gained"] == 0).sum())
    max_swing = int(comparison["positions_gained"].abs().max())
    cards.append(
        {
            "label": "Movement spread",
            "value": f"{max_swing} places",
            "meta": f"{unchanged_count} driver(s) projected to hold position.",
            "tone": "neutral",
        }
    )
    return cards


def _movement_ladder_rows(comparison: pd.DataFrame, *, row_limit: int = 12) -> pd.DataFrame:
    """Return only projected movers for the movement ladder."""
    if comparison.empty:
        return comparison

    ladder_rows = comparison[comparison["positions_gained"] != 0].copy()
    if ladder_rows.empty:
        return ladder_rows

    ladder_rows["abs_change"] = ladder_rows["positions_gained"].abs()
    ladder_rows = ladder_rows.sort_values(
        ["abs_change", "finish_position", "driver"],
        ascending=[False, True, True],
    )
    ladder_rows = ladder_rows.head(row_limit)
    return ladder_rows.sort_values(["start_position", "finish_position", "driver"])


def _position_change_ladder_figure(rows: pd.DataFrame, *, title: str) -> go.Figure:
    """Build a start-to-finish movement ladder for the top projected movers."""
    fig = go.Figure()
    for row in rows.itertuples(index=False):
        delta = int(row.positions_gained)
        color = "#76D3B3" if delta > 0 else "#C28657" if delta < 0 else "rgba(139,148,158,0.65)"
        line_width = 4 if delta != 0 else 2
        customdata = [
            [row.driver, row.team, int(row.start_position), int(row.finish_position), delta],
            [row.driver, row.team, int(row.start_position), int(row.finish_position), delta],
        ]
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[int(row.start_position), int(row.finish_position)],
                mode="lines+markers",
                line={"color": color, "width": line_width},
                marker={"size": 9, "color": color, "line": {"width": 0}},
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})"
                    "<br>Grid: P%{customdata[2]}"
                    "<br>Finish: P%{customdata[3]}"
                    "<br>Net: %{customdata[4]:+d}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=-0.06,
            y=int(row.start_position),
            text=f"{row.driver} P{int(row.start_position)}",
            showarrow=False,
            xanchor="right",
            font={"size": 11, "color": "rgba(232,237,242,0.82)"},
        )
        fig.add_annotation(
            x=1.06,
            y=int(row.finish_position),
            text=f"P{int(row.finish_position)} {delta:+d}",
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": color},
        )

    fig.update_layout(
        height=max(340, 28 * len(rows) + 180),
        margin={"l": 90, "r": 90, "t": 56, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 16, "color": "rgba(232,237,242,0.94)"},
        },
        xaxis={
            "range": [-0.22, 1.22],
            "tickmode": "array",
            "tickvals": [0, 1],
            "ticktext": ["Grid", "Finish"],
            "fixedrange": True,
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"size": 12, "color": "rgba(232,237,242,0.80)"},
        },
        yaxis={
            "title": {
                "text": "Position",
                "font": {"size": 11, "color": "rgba(232,237,242,0.72)"},
            },
            "autorange": "reversed",
            "dtick": 1,
            "gridcolor": "rgba(232,237,242,0.08)",
            "fixedrange": True,
            "tickfont": {"size": 11, "color": "rgba(232,237,242,0.62)"},
        },
        font={"family": "IBM Plex Sans, sans-serif", "color": "rgba(232,237,242,0.88)"},
    )
    return fig


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
            "This isolates the race story before the full classification table."
        ),
        eyebrow=_position_change_chart_title(prediction_name, result),
    )
    render_stat_cards(
        _build_movement_story_cards(comparison),
        grid_class="ts-stat-grid ts-stat-grid--movement",
    )
    st.caption(
        "Movement ladder shows projected position changes only; unchanged drivers are omitted."
    )
    st.caption(summary)

    if not bool((comparison["positions_gained"] != 0).any()):
        render_notice_banner(
            "No projected position changes in this run.",
            tone="info",
            label="Movement",
        )
        return

    st.plotly_chart(
        _position_change_ladder_figure(
            _movement_ladder_rows(comparison),
            title="Grid to projected finish - movers only",
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
                "Track temperature input: "
                f"{track_temp_c:.1f}C ({session_pct}% {session_label} + "
                f"{forecast_pct}% race-weather baseline)"
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
    """Render the top compound strategies from the race simulations."""
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
    """Render the simulated pit-lap window summary."""
    st.subheader("Pit Stop Windows")

    sorted_pit_laps = sorted(
        pit_lap_distribution.items(),
        key=lambda x: int(x[0].split("_")[1].split("-")[0]),
    )

    total_stops = sum(count for _, count in sorted_pit_laps) or 1

    windows = []
    for lap_bin, count in sorted_pit_laps:
        label = lap_bin.replace("lap_", "L")
        pct = 100 * (count / total_stops)
        windows.append((label, count, pct))

    top_windows = sorted(windows, key=lambda x: x[2], reverse=True)[:5]

    st.caption(
        "Share of all simulated pit events (all cars x all simulations). "
        "Windows are 5-lap bins, e.g. L25-30."
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
    """Apply race-table styling for podium, points, and risk columns."""

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
    )
    if "DNF Risk %" in df_display.columns:
        styled_df = styled_df.map(color_dnf_risk, subset=["DNF Risk %"])

    format_map = {
        column: template
        for column, template in {
            "Expected Pos": "{:.2f}",
            "Order Confidence %": "{:.1f}",
            "Podium %": "{:.1f}",
            "DNF Risk %": "{:.1f}",
        }.items()
        if column in df_display.columns
    }
    if format_map:
        styled_df = styled_df.format(format_map)
    if "Expected Pos" in df_display.columns:
        styled_df = styled_df.map(highlight_expected_position, subset=["Expected Pos"])

    try:
        styled_df = styled_df.hide(axis="index")
    except (AttributeError, TypeError) as exc:
        logger.debug("Could not hide dataframe index: %s", exc)

    return styled_df


def _render_race_result(df: pd.DataFrame) -> None:
    """Render race prediction table and summary cards."""
    race_df = df.copy()
    has_confidence = "confidence" in race_df.columns
    has_podium_probability = "podium_probability" in race_df.columns
    has_dnf_probability = "dnf_probability" in race_df.columns
    if not has_dnf_probability and "dnf_risk" in race_df.columns:
        race_df["dnf_probability"] = race_df["dnf_risk"]
        has_dnf_probability = True
    if has_confidence:
        race_df["confidence"] = pd.to_numeric(race_df["confidence"], errors="coerce").round(1)
    if has_podium_probability:
        race_df["podium_probability"] = pd.to_numeric(
            race_df["podium_probability"],
            errors="coerce",
        ).round(1)
    if has_dnf_probability:
        race_df["dnf_probability"] = (
            pd.to_numeric(race_df["dnf_probability"], errors="coerce") * 100
        ).round(1)
    has_expected_position = "position_blend_score" in race_df.columns
    if has_expected_position:
        race_df["expected_position"] = race_df["position_blend_score"].astype(float).round(2)

    has_ci = "p5" in race_df.columns and "p95" in race_df.columns
    if has_ci:
        race_df["ci_range"] = race_df.apply(lambda r: f"P{int(r['p5'])}–P{int(r['p95'])}", axis=1)

    input_confidence = race_df.attrs.get("input_confidence")

    warnings: list[str] = []
    mean_confidence = (
        float(race_df["confidence"].mean()) if has_confidence and not race_df.empty else None
    )
    if isinstance(mean_confidence, float) and mean_confidence < 56.0:
        warnings.append(
            "Wide projected-finish spread: mean order confidence is "
            f"{mean_confidence:.1f}%. This reflects simulation spread; input-data "
            "confidence is tracked separately."
        )

    if isinstance(input_confidence, int | float) and float(input_confidence) < 0.60:
        warnings.append(
            f"Low input-data confidence ({float(input_confidence):.2f}/1.00). "
            "This run leans heavily on priors."
        )

    if has_ci:
        interval_width = (race_df["p95"] - race_df["p5"]).astype(float)
        median_width = float(interval_width.median())
        wide_ranges = int((interval_width >= 8.0).sum())
        if wide_ranges >= max(6, int(len(race_df) * 0.35)):
            warnings.append(
                f"Wide position ranges: {wide_ranges} drivers have 90% ranges spanning 8+ places "
                f"(median span: {median_width:.1f})."
            )

    if has_dnf_probability:
        high_dnf = race_df[race_df["dnf_probability"] > 20]
        if not high_dnf.empty:
            warnings.append(
                f"High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}"
            )
    team_cluster_warning = _build_team_clustering_warning(
        race_df,
        mean_confidence=mean_confidence,
    )
    if team_cluster_warning:
        warnings.append(team_cluster_warning)

    _render_collapsible_warnings(warnings, title="Race warnings")

    if has_expected_position:
        primary_caption = (
            "Rows are ranked by expected finishing position across the full simulation "
            "distribution, not by Order Confidence% or Podium%."
            if has_podium_probability
            else "Rows are ranked by expected finishing position across the full simulation "
            "distribution, not by Order Confidence%."
        )
    else:
        primary_caption = (
            "Rows are ranked by projected finishing order from the selected checkpoint."
        )
    st.caption(primary_caption)
    if has_expected_position:
        st.caption(
            "Key signal: `Expected Pos` (lower is better). Use `90% Pos Range` to judge uncertainty."
        )
    if has_ci:
        ci_caption = "`90% Pos Range` shows where a driver lands in 90% of simulations (P5 to P95)."
        if has_podium_probability:
            ci_caption += " Equal Podium% values are normal because podium probabilities are monotonic-smoothed."
        st.caption(ci_caption)

    display_cols = ["position", "driver", "team"]
    display_names = ["Pos", "Driver", "Team"]
    if has_expected_position:
        display_cols.append("expected_position")
        display_names.append("Expected Pos")
    if has_ci:
        display_cols.append("ci_range")
        display_names.append("90% Pos Range")
    if has_podium_probability:
        display_cols.append("podium_probability")
        display_names.append("Podium %")
    if has_dnf_probability:
        display_cols.append("dnf_probability")
        display_names.append("DNF Risk %")
    if has_confidence:
        display_cols.append("confidence")
        display_names.append("Order Confidence %")

    df_display = race_df[display_cols].copy()
    df_display.columns = display_names

    st.subheader("Projected podium")
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
                    f"{podium_row['team']} • {float(podium_row['confidence']):.1f}% order confidence"
                    if has_confidence and pd.notna(podium_row.get("confidence"))
                    else str(podium_row["team"])
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
