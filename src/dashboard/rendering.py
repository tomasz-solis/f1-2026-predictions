"""Dashboard rendering helpers for prediction outputs."""

import pandas as pd
import streamlit as st


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
        st.warning(unique_messages[0])
        return

    primary_warning = unique_messages[0]
    remaining_count = len(unique_messages) - 1
    st.warning(f"{primary_warning} (+{remaining_count} more)")
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
    df["confidence"] = df["confidence"].round(1)
    df["podium_probability"] = df["podium_probability"].round(1)
    df["dnf_probability"] = (df["dnf_probability"] * 100).round(1)
    has_expected_position = "position_blend_score" in df.columns
    if has_expected_position:
        df["expected_position"] = df["position_blend_score"].astype(float).round(2)

    # Build 90% position interval string when percentile columns are present.
    has_ci = "p5" in df.columns and "p95" in df.columns
    if has_ci:
        df["ci_range"] = df.apply(lambda r: f"P{int(r['p5'])}–P{int(r['p95'])}", axis=1)

    input_confidence = df.attrs.get("input_confidence")

    warnings: list[str] = []
    mean_confidence = float(df["confidence"].mean()) if not df.empty else 0.0
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
        interval_width = (df["p95"] - df["p5"]).astype(float)
        median_width = float(interval_width.median())
        wide_ranges = int((interval_width >= 8.0).sum())
        if wide_ranges >= max(6, int(len(df) * 0.35)):
            warnings.append(
                f"📏 Wide position ranges: {wide_ranges} drivers have 90% ranges spanning 8+ places "
                f"(median span: {median_width:.1f})."
            )

    high_dnf = df[df["dnf_probability"] > 20]
    if not high_dnf.empty:
        warnings.append(
            f"🛑 High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}"
        )
    team_cluster_warning = _build_team_clustering_warning(
        df,
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

    df_display = df[display_cols].copy()
    df_display.columns = display_names

    styled_df = _style_race_table(df_display)

    st.markdown(
        f'<div class="rc-table">{styled_df.to_html()}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🏁 Predicted Podium")
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
            st.info(
                "Front-row projection is statistically tight: teammate ranges overlap, "
                "so the P1/P2 ordering can flip between close scenarios."
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
    st.markdown("---")
    st.header(prediction_name)

    results_key = "finish_order" if is_race else "grid"
    df = pd.DataFrame(result[results_key])
    df["position"] = df["position"].astype(int)
    df.attrs["input_confidence"] = result.get("input_confidence")

    grid_source = result.get("grid_source")
    qualifying_warning_messages: list[str] = []
    if grid_source:
        if is_race:
            if grid_source == "ACTUAL":
                st.success("Using ACTUAL grid from completed session")
            else:
                st.info("Using PREDICTED grid")
        else:
            if grid_source == "ACTUAL":
                qualifying_warning_messages.append(
                    "🧭 Grid source: ACTUAL completed-session results."
                )
            else:
                qualifying_warning_messages.append("🧭 Grid source: PREDICTED qualifying grid.")

    if not is_race:
        data_source = result.get("data_source", "Unknown")
        blend_used = result.get("blend_used", False)
        fp_blend_weight_used = result.get("fp_blend_weight_used")

        if blend_used:
            if isinstance(fp_blend_weight_used, (int | float)):
                practice_share = int(round(float(fp_blend_weight_used) * 100))
                model_share = max(0, 100 - practice_share)
                qualifying_warning_messages.append(
                    f"🗂️ Data source: {data_source} ({practice_share}% practice data + {model_share}% model)."
                )
            else:
                qualifying_warning_messages.append(
                    f"🗂️ Data source: {data_source} (70% practice data + 30% model)."
                )
        else:
            qualifying_warning_messages.append(f"🗂️ Data source: {data_source}.")
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

    characteristics_profile = result.get("characteristics_profile_used")
    teams_with_profile = result.get("teams_with_characteristics_profile", 0)
    compound_strategies = result.get("compound_strategies", {})
    pit_lap_distribution = result.get("pit_lap_distribution", {})

    if characteristics_profile and teams_with_profile:
        profile_msg = (
            "🛠️ Car characteristics profile in use: "
            f"{characteristics_profile} ({teams_with_profile} teams)."
        )
        if is_race:
            st.info(profile_msg)
        else:
            qualifying_warning_messages.append(profile_msg)

    if not is_race:
        _render_collapsible_warnings(
            qualifying_warning_messages,
            title="⚠️ Qualifying Warnings",
        )

    if is_race:
        _render_track_temperature_context(result)
        _render_weather_feature_context(result)

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
