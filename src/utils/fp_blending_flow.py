"""Pure helper logic for FP blending extraction and session-policy flow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def build_session_priority(
    *,
    is_sprint: bool,
    qualifying_stage: str,
) -> list[tuple[str, str, float]]:
    """Resolve ordered FP session policy for the qualifying context."""
    if is_sprint:
        if qualifying_stage == "sprint":
            # Sprint Qualifying prediction should be anchored to pre-SQ context.
            return [("FP1", "FP1 short-stint", 1.0)]
        # Main qualifying: blend all available short-run signals.
        return [
            ("Sprint Qualifying", "Sprint Qualifying short-stint", 1.00),
            ("Sprint", "Sprint pace signal", 0.55),
            ("FP1", "FP1 short-stint", 0.70),
        ]

    # Normal weekend: blend short-stint signals instead of hard FP3 fallback.
    return [
        ("FP3", "FP3 short-stint", 1.00),
        ("FP2", "FP2 short-stint", 0.82),
        ("FP1", "FP1 short-stint", 0.68),
    ]


def extract_team_performance_from_laps(
    *,
    laps: pd.DataFrame,
    run_focus: str,
    min_long_run_laps: int,
    preferred_short_run_compound: str | None,
    long_run_outlier_threshold: float,
    long_run_trim_ends: bool,
    extract_representative_lap_time_fn: Callable[..., float | None],
    map_team_to_characteristics_fn: Callable[[str], str | None],
) -> dict[str, float] | None:
    """Build normalized team performance from loaded session laps."""
    best_times: list[dict[str, float | str]] = []

    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver]
        valid_laps = driver_laps[
            (driver_laps["LapTime"].notna()) & (driver_laps["Compound"].notna())
        ]
        if len(valid_laps) == 0:
            continue

        representative_time = extract_representative_lap_time_fn(
            valid_laps,
            run_focus=run_focus,
            min_long_run_laps=min_long_run_laps,
            preferred_short_run_compound=preferred_short_run_compound,
            long_run_outlier_threshold=long_run_outlier_threshold,
            long_run_trim_ends=long_run_trim_ends,
        )
        if representative_time is None:
            continue

        team_raw = driver_laps["Team"].iloc[0]
        if pd.isna(team_raw):
            continue
        team = map_team_to_characteristics_fn(team_raw) or str(team_raw)

        best_times.append({"driver": driver, "team": team, "time": representative_time})

    if not best_times:
        return None

    team_times: dict[str, list[float]] = {}
    for entry in best_times:
        team = str(entry["team"])
        team_times.setdefault(team, []).append(float(entry["time"]))

    team_medians = {team: float(np.median(times)) for team, times in team_times.items()}
    fastest = min(team_medians.values())
    slowest = max(team_medians.values())

    if fastest == slowest:
        return {team: 0.5 for team in team_medians}

    return {
        team: 1.0 - (time - fastest) / (slowest - fastest) for team, time in team_medians.items()
    }


def blend_available_sessions(
    available_sessions: list[dict[str, Any]],
) -> tuple[str, dict[str, float], pd.DataFrame | None, str] | None:
    """
    Combine one or more available session signals into one blend.

    Returns:
    - label
    - blended performance
    - representative laps payload
    - primary session code
    """
    if not available_sessions:
        return None

    if len(available_sessions) == 1:
        selected = available_sessions[0]
        return selected["label"], selected["data"], selected["laps"], selected["code"]

    weighted_totals: dict[str, float] = {}
    weighted_counts: dict[str, float] = {}
    for session_info in available_sessions:
        weight = float(session_info["weight"])
        for team, score in session_info["data"].items():
            weighted_totals[team] = weighted_totals.get(team, 0.0) + (float(score) * weight)
            weighted_counts[team] = weighted_counts.get(team, 0.0) + weight

    blended = {
        team: weighted_totals[team] / weighted_counts[team]
        for team in weighted_totals
        if weighted_counts.get(team, 0.0) > 0.0
    }
    if not blended:
        return None

    included = " + ".join([item["code"] for item in available_sessions])
    primary = max(available_sessions, key=lambda item: item["weight"])
    return f"Short-stint blend ({included})", blended, primary["laps"], primary["code"]
