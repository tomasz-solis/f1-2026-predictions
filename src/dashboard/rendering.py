"""Dashboard rendering helpers for prediction outputs."""

import pandas as pd
import streamlit as st


def _render_track_temperature_context(result: dict) -> None:
    """Render race track-temperature source and blend details when available."""
    context = result.get("track_temperature_context")
    if not isinstance(context, dict):
        return

    raw_temp = context.get("track_temperature_c")
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
    try:
        session_weight = float(raw_session_weight)
    except (TypeError, ValueError):
        session_weight = None
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
                {
                    "selector": "td:nth-child(4), td:nth-child(5), td:nth-child(6)",
                    "props": [
                        ("font-weight", "700"),
                    ],
                },
            ]
        )
        .map(color_position, subset=["Pos"])
        .map(color_dnf_risk, subset=["DNF Risk %"])
        .format({"Confidence %": "{:.1f}", "Podium %": "{:.1f}", "DNF Risk %": "{:.1f}"})
    )

    try:
        styled_df = styled_df.hide(axis="index")
    except (AttributeError, TypeError):
        pass

    return styled_df


def _render_race_result(df: pd.DataFrame) -> None:
    df["confidence"] = df["confidence"].round(1)
    df["podium_probability"] = df["podium_probability"].round(1)
    df["dnf_probability"] = (df["dnf_probability"] * 100).round(1)

    df["dnf_risk"] = df["dnf_probability"].apply(
        lambda x: "High" if x > 20 else "Medium" if x >= 10 else "Low"
    )

    # Build p5–p95 confidence interval string when columns are present.
    has_ci = "p5" in df.columns and "p95" in df.columns
    if has_ci:
        df["ci_range"] = df.apply(lambda r: f"P{int(r['p5'])}–P{int(r['p95'])}", axis=1)

    display_cols = [
        "position",
        "driver",
        "team",
    ]
    display_names = ["Pos", "Driver", "Team"]
    if has_ci:
        display_cols.append("ci_range")
        display_names.append("p5–p95")
    display_cols += ["confidence", "podium_probability", "dnf_probability", "dnf_risk"]
    display_names += ["Confidence %", "Podium %", "DNF Risk %", "Status"]

    df_display = df[display_cols].copy()
    df_display.columns = display_names

    styled_df = _style_race_table(df_display)

    st.markdown(
        f'<div class="rc-table">{styled_df.to_html()}</div>',
        unsafe_allow_html=True,
    )

    high_dnf = df[df["dnf_probability"] > 20]
    if not high_dnf.empty:
        st.warning(
            f"High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}"
        )

    st.subheader("Predicted Podium")
    podium = df[df["position"] <= 3].copy()

    col1, col2, col3 = st.columns(3)
    for i, (_idx, row) in enumerate(podium.iterrows()):
        col = [col2, col1, col3][i]
        with col:
            st.markdown(f"### P{row['position']}")
            st.markdown(f"## **{row['driver']}**")
            st.markdown(f"*{row['team']}*")
            st.metric("Confidence", f"{row['confidence']:.1f}%")
            st.progress(row["confidence"] / 100)


def _render_qualifying_result(df: pd.DataFrame) -> None:
    df_display = df[["position", "driver", "team"]].copy()
    df_display.columns = ["Grid", "Driver", "Team"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**P1-10**")
        st.markdown(
            f'<div class="rc-table">{df_display.head(10).to_html(index=False)}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**P11-15**")
        st.markdown(
            f'<div class="rc-table">{df_display.iloc[10:15].to_html(index=False)}</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("**P16-22**")
        st.markdown(
            f'<div class="rc-table">{df_display.iloc[15:].to_html(index=False)}</div>',
            unsafe_allow_html=True,
        )


def display_prediction_result(result: dict, prediction_name: str, is_race: bool = False) -> None:
    """Display a single prediction result (qualifying or race)."""
    st.markdown("---")
    st.header(prediction_name)

    grid_source = result.get("grid_source")
    if grid_source:
        if grid_source == "ACTUAL":
            st.success("Using ACTUAL grid from completed session")
        else:
            st.info("Using PREDICTED grid")

    if not is_race:
        data_source = result.get("data_source", "Unknown")
        blend_used = result.get("blend_used", False)
        fp_blend_weight_used = result.get("fp_blend_weight_used")

        if blend_used:
            if isinstance(fp_blend_weight_used, (int | float)):
                practice_share = int(round(float(fp_blend_weight_used) * 100))
                model_share = max(0, 100 - practice_share)
                st.success(
                    f"Using {data_source} ({practice_share}% practice data + {model_share}% model)"
                )
            else:
                st.success(f"Using {data_source} (70% practice data + 30% model)")
        else:
            st.info(data_source)

    characteristics_profile = result.get("characteristics_profile_used")
    teams_with_profile = result.get("teams_with_characteristics_profile", 0)
    compound_strategies = result.get("compound_strategies", {})
    pit_lap_distribution = result.get("pit_lap_distribution", {})

    if characteristics_profile and teams_with_profile:
        st.info(
            "Car characteristics profile in use: "
            f"`{characteristics_profile}` ({teams_with_profile} teams)"
        )

    if is_race:
        _render_track_temperature_context(result)
        _render_weather_feature_context(result)

    if compound_strategies and is_race:
        _render_compound_strategies(compound_strategies)

    if pit_lap_distribution and is_race:
        _render_pit_lap_distribution(pit_lap_distribution)

    results_key = "finish_order" if is_race else "grid"
    df = pd.DataFrame(result[results_key])
    df["position"] = df["position"].astype(int)

    if is_race:
        _render_race_result(df)
    else:
        _render_qualifying_result(df)
