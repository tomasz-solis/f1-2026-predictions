"""Metrics and lap-selection helpers for testing updater flows."""

from __future__ import annotations

import logging
from typing import Any

import fastf1
import numpy as np
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from src.extractors.performance import extract_all_teams_performance
from src.systems.compound_analyzer import (
    extract_compound_metrics,
    normalize_compound_metrics_across_teams,
)
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)

_DIRECTIONALITY_KEYS = (
    "max_speed",
    "slow_corner_speed",
    "medium_corner_speed",
    "high_corner_speed",
)

_RUN_PROFILE_MODES = ("balanced", "all", "short_run", "long_run")
_SHORT_STINT_MAX_LAPS = 5
_LONG_STINT_MIN_LAPS = 8
_STINT_OUTLIER_BUFFER_SECONDS = 5.0
_MIN_VALID_TEAM_LAPS = 5
_WET_SESSION_RAIN_THRESHOLD = 0.30
# Absolute pace sanity floor: drop laps slower than (team-best valid lap) * (1 + ratio).
# Removes in/out, cooldown, and aborted laps that are not representative pace before any
# stint/representative selection runs. 0.20 keeps genuine heavy-fuel long runs (~+5-8%) while
# discarding the +20%-and-worse non-laps (e.g. a 133s Monaco "lap" against a 74s best).
_GARBAGE_LAP_PACE_RATIO = 0.20
# Short-run pace = median of each driver's quickest N clean green laps (pace-based, not
# stint-length based). Mirrors src/utils/fp_blending.py::_extract_short_run_lap_time.
_SHORT_RUN_TOP_N = 3
# Plausibility bound for a single-lap teammate delta. Real one-lap teammate gaps are well
# under a second; anything larger is a data artifact and is clamped.
_MAX_TEAMMATE_DELTA_SECONDS = 1.5
_RAW_TOP_SPEED_METRIC = "top_speed_kph"
_RAW_OVERALL_PACE_METRIC = "overall_pace_seconds"
_RAW_SLOW_CORNER_METRIC = "slow_corner_seconds"
_RAW_MEDIUM_CORNER_METRIC = "medium_corner_seconds"
_RAW_FAST_CORNER_METRIC = "fast_corner_seconds"
_RAW_BRAKING_METRIC = "braking_pct"


def _canonicalize_team_name(raw_team: str, known_teams: set[str]) -> str | None:
    """Map session team name to canonical team key used in characteristics JSON."""
    return map_team_to_characteristics(raw_team, known_teams=known_teams)


def _drop_implausible_laps(
    laps: pd.DataFrame, pace_ratio: float = _GARBAGE_LAP_PACE_RATIO
) -> pd.DataFrame:
    """Drop laps far slower than the team's own fastest valid lap.

    Practice/qualifying lap logs contain in/out, cooldown, and aborted laps that are
    tens of seconds off representative pace. Earlier selection only used *relative*
    per-stint filtering, so a stint that contained only slow laps (e.g. a 2-lap
    end-of-session cooldown, or a car that broke down) had its slow lap accepted as
    the team's "representative" pace. This applies an absolute floor relative to the
    team's own best clean lap so such non-laps never reach representative selection.
    """
    if laps.empty or "LapTime" not in laps.columns:
        return laps
    lap_seconds = _lap_seconds_series(laps)
    if lap_seconds.empty:
        return laps
    reference = float(lap_seconds.min())
    if not np.isfinite(reference) or reference <= 0:
        return laps
    threshold = reference * (1.0 + max(0.0, float(pace_ratio)))
    keep_idx = lap_seconds[lap_seconds <= threshold].index
    if len(keep_idx) == 0:
        return laps
    return laps.loc[keep_idx].copy()


def _filter_valid_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """Filter to representative non-pit laps."""
    if team_laps.empty:
        return team_laps
    if "LapTime" not in team_laps.columns:
        return team_laps.iloc[0:0].copy()

    # Testing updates favor coverage over strict pit filtering.
    # Timed laps are enough to infer early directionality.
    mask = team_laps["LapTime"].notna()
    # Testing sessions often have sparse/inconsistent IsAccurate flags.
    # Enforce this only when explicit True rows exist.
    if "IsAccurate" in team_laps.columns:
        accurate = team_laps["IsAccurate"]
        accurate_true = accurate.eq(True)
        if accurate.notna().any() and bool(accurate_true.any()):
            mask &= accurate_true

    # Absolute pace floor: reject non-representative laps (in/out, cooldown, aborted)
    # that are far slower than the team's own best clean lap.
    return _drop_implausible_laps(team_laps[mask].copy())


def _session_rain_fraction(session: fastf1.core.Session) -> float | None:
    """Return the share of a session affected by rainfall when weather data exists."""
    try:
        weather_data = getattr(session, "weather_data", None)
    except (AttributeError, DataNotLoadedError, RuntimeError, TypeError, ValueError):
        return None

    if not isinstance(weather_data, pd.DataFrame) or weather_data.empty:
        return None
    if "Rainfall" not in weather_data.columns:
        return None

    rainfall = weather_data["Rainfall"]
    try:
        rainfall_fraction = rainfall.astype(bool).mean()
    except (TypeError, ValueError):
        rainfall_numeric = pd.to_numeric(rainfall, errors="coerce").dropna()
        if rainfall_numeric.empty:
            return None
        rainfall_fraction = rainfall_numeric.gt(0).mean()

    return float(rainfall_fraction)


def _session_is_predominantly_wet(
    session: fastf1.core.Session,
    rain_threshold: float = _WET_SESSION_RAIN_THRESHOLD,
) -> bool:
    """Return True when a session has too much rain to trust dry-pace inference."""
    rain_fraction = _session_rain_fraction(session)
    if rain_fraction is None:
        return False
    return rain_fraction > float(rain_threshold)


def _strip_in_out_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """Remove in-laps/out-laps when pit timing columns are available."""
    if team_laps.empty:
        return team_laps

    filtered = team_laps.copy()
    if "PitOutTime" in filtered.columns:
        filtered = filtered[filtered["PitOutTime"].isna()]
    if "PitInTime" in filtered.columns:
        filtered = filtered[filtered["PitInTime"].isna()]

    return filtered


def _classify_run_laps(team_laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split team laps into short-run and long-run candidate subsets.

    Uses stint lengths when available; otherwise falls back to lap-time quantiles.
    """
    if team_laps.empty:
        return team_laps, team_laps

    cleaned = _strip_in_out_laps(team_laps)
    if cleaned.empty:
        cleaned = team_laps

    short_chunks: list[pd.DataFrame] = []
    long_chunks: list[pd.DataFrame] = []

    has_stint = "Stint" in cleaned.columns and bool(cleaned["Stint"].notna().any())

    if has_stint:
        grouping_cols = ["Driver", "Stint"]
        for _, stint_laps in cleaned.groupby(grouping_cols, dropna=False):
            timed = stint_laps[stint_laps["LapTime"].notna()].copy()
            if len(timed) < 2:
                continue
            stint_len = len(timed)
            if stint_len <= _SHORT_STINT_MAX_LAPS:
                short_chunks.append(timed)
            if stint_len >= _LONG_STINT_MIN_LAPS:
                long_chunks.append(timed)
    else:
        lap_seconds = pd.to_timedelta(cleaned["LapTime"], errors="coerce").dt.total_seconds()
        lap_seconds = lap_seconds.dropna()
        if not lap_seconds.empty:
            short_threshold = lap_seconds.quantile(0.35)
            long_threshold = lap_seconds.quantile(0.65)
            short_chunks.append(cleaned[lap_seconds <= short_threshold].copy())
            long_chunks.append(cleaned[lap_seconds >= long_threshold].copy())

    short_laps = (
        pd.concat(short_chunks, ignore_index=False) if short_chunks else cleaned.iloc[0:0].copy()
    )
    long_laps = (
        pd.concat(long_chunks, ignore_index=False) if long_chunks else cleaned.iloc[0:0].copy()
    )

    return short_laps, long_laps


def _select_stint_representative_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce laps to one representative lap per driver/stint(/compound) slice.

    This avoids over-weighting teams with longer programs in the same session
    while preserving FastF1's `Laps` subclass when it is available.
    """
    if team_laps.empty:
        return team_laps

    grouping_cols = ["Driver"]
    if "Stint" in team_laps.columns and bool(team_laps["Stint"].notna().any()):
        grouping_cols.append("Stint")
    if "Compound" in team_laps.columns and bool(team_laps["Compound"].notna().any()):
        grouping_cols.append("Compound")

    representative_indices = []
    for _, laps in team_laps.groupby(grouping_cols, dropna=False):
        timed = laps[laps["LapTime"].notna()].copy()
        if timed.empty:
            continue

        filtered = _filter_stint_outlier_laps(timed)
        lap_seconds = _lap_seconds_series(filtered)
        if lap_seconds.empty:
            continue

        if len(lap_seconds) <= _SHORT_STINT_MAX_LAPS:
            representative_idx = lap_seconds.idxmin()
        else:
            median_value = float(lap_seconds.median())
            representative_idx = (lap_seconds - median_value).abs().idxmin()
        representative_indices.append(representative_idx)

    if not representative_indices:
        return team_laps

    return team_laps.loc[representative_indices].copy()


def _select_short_run_laps(team_laps: pd.DataFrame, top_n: int = _SHORT_RUN_TOP_N) -> pd.DataFrame:
    """Select each driver's quickest clean green laps as the short-run sample.

    Short-run (qualifying-sim) pace is about single-lap bite on low fuel, not stint
    length. Selecting by lap *pace* per driver — instead of by stint length — keeps a
    fast lap that happened inside a longer stint and rejects a slow 2-lap cooldown
    stint that a driver who ran a full program may leave at the end of a session.
    Mirrors ``src/utils/fp_blending.py::_extract_short_run_lap_time``.
    """
    clean = _strip_in_out_laps(team_laps)
    if clean.empty:
        clean = team_laps
    if clean.empty or "Driver" not in clean.columns:
        return clean

    keep_index: list = []
    for _, driver_laps in clean.groupby("Driver", dropna=False):
        lap_seconds = _lap_seconds_series(driver_laps)
        if lap_seconds.empty:
            continue
        keep_index.extend(list(lap_seconds.nsmallest(max(1, int(top_n))).index))

    if not keep_index:
        return clean
    return clean.loc[keep_index].copy()


def _restore_laps_session(
    source_laps: pd.DataFrame,
    selected_laps: pd.DataFrame,
) -> pd.DataFrame:
    """Keep FastF1's session reference after pandas selection/concat operations.

    FastF1 stores the telemetry owner on ``Laps.session``. Some pandas paths used
    by balanced/long-run selection preserve the ``Laps`` subclass but drop that
    metadata, which makes every selected ``Lap.get_telemetry()`` call fail. Restore
    the reference from the original team slice before telemetry extraction.
    """
    source_session = getattr(source_laps, "session", None)
    if source_session is None or getattr(selected_laps, "session", None) is not None:
        return selected_laps

    try:
        selected_laps.session = source_session
    except (AttributeError, TypeError, ValueError):
        logger.debug("Could not restore FastF1 session metadata on selected laps")
    return selected_laps


def _select_program_aware_laps(team_laps: pd.DataFrame, run_profile: str) -> pd.DataFrame:
    """
    Select representative laps with program-aware run filtering.

    Modes:
    - all: use all valid laps
    - short_run: per-driver quickest clean green laps (pace-based, not stint-length)
    - long_run: prefer long stints
    - balanced: blend short + long stints
    """
    if team_laps.empty:
        return team_laps

    if run_profile not in _RUN_PROFILE_MODES:
        raise ValueError(
            f"Invalid run_profile '{run_profile}'. Use one of: {', '.join(_RUN_PROFILE_MODES)}"
        )

    # Short-run pace is selected by lap pace per driver, not stint length, so a quick
    # lap inside a long stint counts and an end-of-session cooldown stint does not.
    if run_profile == "short_run":
        short_run_laps = _select_short_run_laps(team_laps)
        selected_short_run = short_run_laps if not short_run_laps.empty else team_laps
        return _restore_laps_session(team_laps, selected_short_run)

    if run_profile == "all":
        selected = team_laps
    else:
        short_laps, long_laps = _classify_run_laps(team_laps)
        if run_profile == "long_run":
            selected = long_laps if not long_laps.empty else team_laps
        else:
            if not short_laps.empty and not long_laps.empty:
                ordered_indices = list(short_laps.index)
                ordered_indices.extend(
                    index for index in long_laps.index if index not in short_laps.index
                )
                selected = team_laps.loc[ordered_indices].copy()
            elif not short_laps.empty:
                selected = short_laps
            elif not long_laps.empty:
                selected = long_laps
            else:
                selected = team_laps

    representative = _select_stint_representative_laps(selected)
    selected_representative = representative if not representative.empty else selected
    return _restore_laps_session(team_laps, selected_representative)


def _count_team_selected_laps(
    session: fastf1.core.Session,
    known_teams: set[str],
    run_profile: str = "all",
    canonicalize_team_name_fn=_canonicalize_team_name,
    filter_valid_laps_fn=_filter_valid_laps,
    select_program_aware_laps_fn=_select_program_aware_laps,
) -> dict[str, float]:
    """Count selected laps per team for a specific run-profile strategy."""
    try:
        laps = session.laps
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}

    if laps is None or laps.empty or "Team" not in laps.columns:
        return {}

    if run_profile not in _RUN_PROFILE_MODES:
        raise ValueError(
            f"Invalid run_profile '{run_profile}'. Use one of: {', '.join(_RUN_PROFILE_MODES)}"
        )

    counts: dict[str, float] = {}
    raw_teams = laps["Team"].dropna().unique()
    for raw_team in raw_teams:
        canonical_team = canonicalize_team_name_fn(str(raw_team), known_teams)
        if not canonical_team:
            continue

        team_laps = laps[laps["Team"] == raw_team]
        valid_laps = filter_valid_laps_fn(team_laps)
        if valid_laps.empty:
            continue

        selected_laps = select_program_aware_laps_fn(valid_laps, run_profile=run_profile)
        if selected_laps.empty:
            selected_laps = valid_laps

        counts[canonical_team] = counts.get(canonical_team, 0.0) + float(len(selected_laps))

    return counts


def _median_timedelta_seconds(series: pd.Series) -> float | None:
    """Get median timedelta in seconds if available."""
    if series is None or series.empty:
        return None

    values = pd.to_timedelta(series, errors="coerce").dropna()
    if values.empty:
        return None

    return float(values.dt.total_seconds().median())


def _median_lap_seconds(team_laps: pd.DataFrame) -> float | None:
    """Get median lap time in seconds for a team slice."""
    if "LapTime" not in team_laps.columns or team_laps.empty:
        return None

    lap_seconds = _lap_seconds_series(team_laps)
    if lap_seconds.empty:
        return None

    return float(lap_seconds.median())


def _lap_seconds_series(team_laps: pd.DataFrame) -> pd.Series:
    """Convert lap times into seconds while preserving the original row index."""
    if "LapTime" not in team_laps.columns or team_laps.empty:
        return pd.Series(dtype=float)

    return pd.to_timedelta(team_laps["LapTime"], errors="coerce").dt.total_seconds().dropna()


def _filter_stint_outlier_laps(
    stint_laps: pd.DataFrame,
    max_delta_seconds: float = _STINT_OUTLIER_BUFFER_SECONDS,
) -> pd.DataFrame:
    """
    Drop obviously slow laps from a stint before extracting representative metrics.

    Practice and qualifying runs often contain cooldown, traffic, or aborted laps
    that are tens of seconds slower than the real push-lap pace. Keeping them
    skews both representative-lap selection and tire-degradation estimates.
    """
    lap_seconds = _lap_seconds_series(stint_laps)
    if lap_seconds.empty:
        return stint_laps.iloc[0:0].copy()

    threshold = float(lap_seconds.min()) + max(0.0, float(max_delta_seconds))
    filtered_idx = lap_seconds[lap_seconds <= threshold].index
    if filtered_idx.empty:
        filtered_idx = lap_seconds.index

    return stint_laps.loc[filtered_idx].copy()


def _estimate_tire_deg_slope(team_laps: pd.DataFrame) -> float | None:
    """
    Estimate team tire degradation slope from same-stint runs.

    Returns slope in seconds/lap (higher means more degradation).
    """
    if team_laps.empty or "LapNumber" not in team_laps.columns:
        return None

    grouping_cols = ["Driver"]
    if "Stint" in team_laps.columns:
        grouping_cols.append("Stint")
    if "Compound" in team_laps.columns:
        grouping_cols.append("Compound")

    slopes = []
    for _, stint_laps in team_laps.groupby(grouping_cols, dropna=False):
        stint = _filter_stint_outlier_laps(stint_laps.sort_values("LapNumber")).sort_values(
            "LapNumber"
        )
        if len(stint) < 3:
            continue

        lap_seconds = _lap_seconds_series(stint)
        if len(lap_seconds) < 3:
            continue

        x = np.arange(len(lap_seconds), dtype=float)
        y = lap_seconds.to_numpy(dtype=float)

        slope = float(np.polyfit(x, y, 1)[0])
        if -0.3 <= slope <= 1.0:
            slopes.append(slope)

    if not slopes:
        return None

    return float(np.median(slopes))


def _normalize_tire_deg_scores(
    tire_deg_slopes: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Normalize tire degradation to 0-1 performance scale (1.0 = best tire life)."""
    if not tire_deg_slopes:
        return {}

    min_slope = min(tire_deg_slopes.values())
    max_slope = max(tire_deg_slopes.values())

    normalized = {}
    for team, slope in tire_deg_slopes.items():
        if max_slope > min_slope:
            perf = 1.0 - ((slope - min_slope) / (max_slope - min_slope))
        else:
            perf = 0.5

        normalized[team] = {
            "tire_deg_slope": float(slope),
            "tire_deg_performance": float(np.clip(perf, 0.0, 1.0)),
        }

    return normalized


def _normalize_lower_better(metric_values: dict[str, float]) -> dict[str, float]:
    """Normalize a lower-is-better metric into 0-1 scale using rank-based scoring."""
    if not metric_values:
        return {}

    if len(metric_values) < 2:
        return {team: 0.5 for team in metric_values}

    ranked_items = sorted(metric_values.items(), key=lambda item: item[1])
    team_count = len(ranked_items)

    grouped: list[tuple[float, list[str]]] = []
    for team, value in ranked_items:
        metric_value = float(value)
        if grouped and np.isclose(metric_value, grouped[-1][0]):
            grouped[-1][1].append(team)
        else:
            grouped.append((metric_value, [team]))

    normalized: dict[str, float] = {}
    rank_cursor = 0
    for _value, tied_teams in grouped:
        average_rank = rank_cursor + ((len(tied_teams) - 1) / 2.0)
        score = float(1.0 - (average_rank / (team_count - 1)))
        for team in tied_teams:
            normalized[team] = float(np.clip(score, 0.0, 1.0))
        rank_cursor += len(tied_teams)
    return normalized


def _attach_raw_snapshot_metrics(
    performance_by_team: dict[str, dict[str, float]],
    per_team_payload: dict[str, dict[str, Any]],
    lap_pace_seconds: dict[str, float],
    raw_top_speed_by_team: dict[str, float],
    *,
    session_key: str,
) -> dict[str, dict[str, float]]:
    """Attach raw session metrics to per-team payloads for snapshot consumers."""
    if not raw_top_speed_by_team and not lap_pace_seconds and not per_team_payload:
        return performance_by_team

    merged = {
        str(team_name): dict(metrics)
        for team_name, metrics in performance_by_team.items()
        if isinstance(metrics, dict)
    }

    for team_name, sessions_payload in per_team_payload.items():
        if not isinstance(sessions_payload, dict) or not sessions_payload:
            continue

        session_payload = sessions_payload.get(session_key)
        if not isinstance(session_payload, dict):
            continue

        team_payload = merged.setdefault(str(team_name), {})

        sector_times = session_payload.get("sector_times")
        if isinstance(sector_times, dict):
            s1 = sector_times.get("s1")
            s2 = sector_times.get("s2")
            s3 = sector_times.get("s3")
            if isinstance(s1, int | float):
                team_payload[_RAW_SLOW_CORNER_METRIC] = float(s1)
            if isinstance(s2, int | float):
                team_payload[_RAW_MEDIUM_CORNER_METRIC] = float(s2)
            if isinstance(s3, int | float):
                team_payload[_RAW_FAST_CORNER_METRIC] = float(s3)

        braking_profile = session_payload.get("braking_profile")
        if isinstance(braking_profile, dict):
            braking_pct = braking_profile.get("braking_pct")
            if isinstance(braking_pct, int | float):
                team_payload[_RAW_BRAKING_METRIC] = float(braking_pct)

    for team_name, lap_seconds in lap_pace_seconds.items():
        team_payload = merged.setdefault(str(team_name), {})
        team_payload[_RAW_OVERALL_PACE_METRIC] = float(lap_seconds)

    for team_name, top_speed_kph in raw_top_speed_by_team.items():
        team_payload = merged.setdefault(str(team_name), {})
        team_payload[_RAW_TOP_SPEED_METRIC] = float(top_speed_kph)
    return merged


def _extract_team_payload(valid_laps: pd.DataFrame) -> dict:
    """Build payload expected by extract_all_teams_performance()."""
    payload = {}

    sector_times = {}
    if "Sector1Time" in valid_laps.columns:
        s1 = _median_timedelta_seconds(valid_laps["Sector1Time"])
        if s1 is not None:
            sector_times["s1"] = s1
    if "Sector2Time" in valid_laps.columns:
        s2 = _median_timedelta_seconds(valid_laps["Sector2Time"])
        if s2 is not None:
            sector_times["s2"] = s2
    if "Sector3Time" in valid_laps.columns:
        s3 = _median_timedelta_seconds(valid_laps["Sector3Time"])
        if s3 is not None:
            sector_times["s3"] = s3
    if sector_times:
        payload["sector_times"] = sector_times

    top_speed = _extract_top_speed_value(valid_laps)
    if top_speed is not None:
        payload["speed_profile"] = {"top_speed": top_speed}

    braking_pct = _extract_braking_capability(valid_laps)
    if braking_pct is not None:
        payload["braking_profile"] = {"braking_pct": braking_pct}

    lap_seconds = _lap_seconds_series(valid_laps)
    if len(lap_seconds) >= 2:
        payload["consistency"] = {"std_lap_time": float(lap_seconds.std(ddof=0))}

    return payload


def _iter_lap_objects(valid_laps: pd.DataFrame):
    """Yield FastF1 lap objects from a Laps frame when telemetry access is available."""
    iterlaps = getattr(valid_laps, "iterlaps", None)
    if not callable(iterlaps):
        return

    for item in iterlaps():
        if isinstance(item, tuple) and len(item) == 2:
            _index, lap = item
        else:
            lap = item
        yield lap


def _extract_braking_pct_from_telemetry(telemetry: pd.DataFrame) -> float | None:
    """Return the share of telemetry samples spent on the brake pedal."""
    if telemetry is None or telemetry.empty or "Brake" not in telemetry.columns:
        return None

    brake = pd.to_numeric(telemetry["Brake"], errors="coerce").dropna()
    if brake.empty:
        return None
    return float((brake > 0).mean() * 100.0)


def _extract_braking_capability(valid_laps: pd.DataFrame) -> float | None:
    """
    Estimate a session-relative braking proxy from representative lap telemetry.

    We use brake-on sample share as a simple, stable proxy: on the same circuit,
    cars that can brake later typically spend less of the lap on the pedal. This
    is still a proxy rather than a pure braking-efficiency metric, but it is
    materially better than copying the slow-corner score.
    """
    braking_samples: list[float] = []

    for lap in _iter_lap_objects(valid_laps):
        telemetry = None

        get_telemetry = getattr(lap, "get_telemetry", None)
        if callable(get_telemetry):
            try:
                telemetry = get_telemetry()
            except (
                AttributeError,
                DataNotLoadedError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                telemetry = None

        if telemetry is None:
            get_car_data = getattr(lap, "get_car_data", None)
            if callable(get_car_data):
                try:
                    telemetry = get_car_data()
                except (
                    AttributeError,
                    DataNotLoadedError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    telemetry = None

        if not isinstance(telemetry, pd.DataFrame):
            continue

        braking_pct = _extract_braking_pct_from_telemetry(telemetry)
        if braking_pct is not None:
            braking_samples.append(braking_pct)

    if not braking_samples:
        return None
    return float(np.median(np.asarray(braking_samples, dtype=float)))


def _extract_top_speed_value(valid_laps: pd.DataFrame) -> float | None:
    """
    Estimate top speed from the quickest trap reached on each selected lap.

    `SpeedST` and `SpeedFL` are closer to true terminal-speed traps than the
    intermediate speed points. Use those first, then fall back when necessary.
    """
    lap_top_speeds = _lap_top_speed_series(valid_laps)
    if lap_top_speeds.empty:
        return None

    return float(lap_top_speeds.median())


def _lap_top_speed_series(valid_laps: pd.DataFrame) -> pd.Series:
    """Return one top-speed sample per lap from the best available trap columns."""
    preferred_columns = [col for col in ("SpeedST", "SpeedFL") if col in valid_laps.columns]
    fallback_columns = [col for col in ("SpeedI2", "SpeedI1") if col in valid_laps.columns]
    speed_columns = preferred_columns or fallback_columns
    if not speed_columns:
        return pd.Series(dtype=float)

    speed_frame = valid_laps[speed_columns].apply(pd.to_numeric, errors="coerce")
    return speed_frame.max(axis=1, skipna=True).dropna()


def _extract_top_speed_capability(valid_laps: pd.DataFrame, quantile: float = 0.90) -> float | None:
    """
    Estimate raw straight-line capability from the high end of a team's lap samples.

    Median trap speed is too conservative for mixed programs because it is pulled
    down by cooldown, traffic, and heavy-fuel running. Use a high percentile of
    the per-lap trap maxima to better represent peak straight-line capability.
    """
    lap_top_speeds = _lap_top_speed_series(valid_laps)
    if lap_top_speeds.empty:
        return None
    if len(lap_top_speeds) < 4:
        return float(lap_top_speeds.max())

    bounded_quantile = min(max(float(quantile), 0.0), 1.0)
    return float(lap_top_speeds.quantile(bounded_quantile))


def _collect_session_metrics(
    session: fastf1.core.Session,
    session_key: str,
    known_teams: set[str],
    run_profile: str = "balanced",
    diagnostics: list[str] | None = None,
    canonicalize_team_name_fn=_canonicalize_team_name,
    filter_valid_laps_fn=_filter_valid_laps,
    select_program_aware_laps_fn=_select_program_aware_laps,
    classify_run_laps_fn=_classify_run_laps,
    median_lap_seconds_fn=_median_lap_seconds,
    extract_team_payload_fn=_extract_team_payload,
    estimate_tire_deg_slope_fn=_estimate_tire_deg_slope,
    extract_all_teams_performance_fn=extract_all_teams_performance,
    normalize_lower_better_fn=_normalize_lower_better,
    normalize_tire_deg_scores_fn=_normalize_tire_deg_scores,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Collect normalized directionality metrics and tire degradation metrics per team."""
    if _session_is_predominantly_wet(session):
        if diagnostics is not None:
            rain_fraction = _session_rain_fraction(session) or 0.0
            diagnostics.append(
                f"{session_key}: rejected wet session ({rain_fraction * 100:.0f}% rainfall)"
            )
        logger.warning(
            "Skipping %s because rainfall exceeded the dry-session threshold", session_key
        )
        return {}, {}

    try:
        laps = session.laps
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Session laps unavailable for %s: %s", session_key, exc)
        if diagnostics is not None:
            diagnostics.append(f"{session_key}: laps unavailable ({type(exc).__name__})")
        return {}, {}

    if laps is None or laps.empty:
        if diagnostics is not None:
            diagnostics.append(f"{session_key}: no laps loaded")
        return {}, {}

    if "Team" not in laps.columns:
        if diagnostics is not None:
            diagnostics.append(f"{session_key}: laps missing Team column")
        return {}, {}

    per_team_payload: dict[str, dict[str, float]] = {}
    tire_deg_slopes: dict[str, float] = {}
    lap_pace_seconds: dict[str, float] = {}
    raw_top_speed_by_team: dict[str, float] = {}
    raw_teams = laps["Team"].dropna().unique()
    mapped_team_count = 0
    selected_lap_count = 0

    for raw_team in raw_teams:
        canonical_team = canonicalize_team_name_fn(str(raw_team), known_teams)
        if not canonical_team:
            continue
        mapped_team_count += 1

        team_laps = laps[laps["Team"] == raw_team]
        valid_laps = filter_valid_laps_fn(team_laps)
        if len(valid_laps) < _MIN_VALID_TEAM_LAPS:
            continue

        representative_laps = select_program_aware_laps_fn(valid_laps, run_profile=run_profile)
        if representative_laps.empty:
            representative_laps = valid_laps
        selected_lap_count += len(representative_laps)

        raw_top_speed = _extract_top_speed_capability(valid_laps)
        if raw_top_speed is not None:
            raw_top_speed_by_team[canonical_team] = raw_top_speed

        median_lap_seconds = median_lap_seconds_fn(representative_laps)
        if median_lap_seconds is not None:
            lap_pace_seconds[canonical_team] = median_lap_seconds

        payload = extract_team_payload_fn(representative_laps)
        if payload:
            per_team_payload.setdefault(canonical_team, {})[session_key] = payload

        _, long_laps = classify_run_laps_fn(valid_laps)
        tire_source = long_laps if not long_laps.empty else valid_laps.iloc[0:0].copy()

        slope = estimate_tire_deg_slope_fn(tire_source)
        if slope is not None:
            tire_deg_slopes[canonical_team] = slope

    normalized_perf = extract_all_teams_performance_fn(per_team_payload, session_name=session_key)
    normalized_perf = _attach_raw_snapshot_metrics(
        normalized_perf,
        per_team_payload,
        lap_pace_seconds,
        raw_top_speed_by_team,
        session_key=session_key,
    )
    normalized_pace = normalize_lower_better_fn(lap_pace_seconds)
    for team, pace_score in normalized_pace.items():
        normalized_perf.setdefault(team, {})["overall_pace"] = pace_score

    normalized_tire = normalize_tire_deg_scores_fn(tire_deg_slopes)

    if diagnostics is not None:
        diagnostics.append(
            f"{session_key}: teams={len(raw_teams)} mapped={mapped_team_count} "
            f"perf_teams={len(normalized_perf)} tire_teams={len(normalized_tire)} "
            f"selected_laps={selected_lap_count} profile={run_profile}"
        )

    return normalized_perf, normalized_tire


def _build_directionality_from_metrics(
    metrics: dict[str, float], directionality_scale: float = 0.10
) -> dict[str, float]:
    """
    Convert 0-1 relative performance metrics into centered directionality deltas.

    Centered around 0 so testing modifier remains small in weight schedule blending.
    """
    metric_map = {
        "max_speed": "top_speed",
        "slow_corner_speed": "slow_corner_performance",
        "medium_corner_speed": "medium_corner_performance",
        "high_corner_speed": "fast_corner_performance",
    }

    fallback_pace = metrics.get("overall_pace")
    directionality = {}
    for key, metric_name in metric_map.items():
        if metric_name in metrics:
            value = float(metrics[metric_name])
        elif fallback_pace is not None and metric_name != "top_speed":
            # Conservative fallback: use overall pace only for corner directionality
            # when per-sector telemetry is still sparse.
            value = float(fallback_pace)
        else:
            value = 0.5
        centered = (value - 0.5) * directionality_scale
        directionality[key] = round(float(np.clip(centered, -0.2, 0.2)), 4)

    return directionality


def _blend_directionality(
    old_directionality: dict[str, float],
    new_directionality: dict[str, float],
    new_weight: float,
) -> dict[str, float]:
    """Blend current and newly extracted directionality to reduce noise."""
    bounded_weight = float(np.clip(new_weight, 0.0, 1.0))

    blended = {}
    for key in _DIRECTIONALITY_KEYS:
        old_value = float(old_directionality.get(key, 0.0))
        new_value = float(new_directionality.get(key, 0.0))
        blended[key] = round(((1.0 - bounded_weight) * old_value) + (bounded_weight * new_value), 4)

    return blended


def _count_team_valid_laps(
    session: fastf1.core.Session,
    known_teams: set[str],
    canonicalize_team_name_fn=_canonicalize_team_name,
    filter_valid_laps_fn=_filter_valid_laps,
) -> dict[str, float]:
    """Count valid timed laps per canonical team for session weighting."""
    try:
        laps = session.laps
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}

    if laps is None or laps.empty or "Team" not in laps.columns:
        return {}

    counts: dict[str, float] = {}
    raw_teams = laps["Team"].dropna().unique()
    for raw_team in raw_teams:
        canonical_team = canonicalize_team_name_fn(str(raw_team), known_teams)
        if not canonical_team:
            continue

        team_laps = laps[laps["Team"] == raw_team]
        valid_laps = filter_valid_laps_fn(team_laps)
        if valid_laps.empty:
            continue

        counts[canonical_team] = counts.get(canonical_team, 0.0) + float(len(valid_laps))

    return counts


def _aggregate_metric_samples(
    samples: list[tuple[float, float]],
    session_aggregation: str,
) -> float | None:
    """Aggregate session metric samples with explicit strategy."""
    if not samples:
        return None

    values = np.array([float(value) for value, _ in samples], dtype=float)
    if values.size == 0:
        return None

    if session_aggregation == "median":
        return float(np.median(values))

    if session_aggregation == "laps_weighted":
        weights = np.array([max(0.0, float(weight)) for _, weight in samples], dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight > 0:
            return float(np.average(values, weights=weights))
        return float(np.mean(values))

    # Default and backward-compatible behavior.
    return float(np.mean(values))


def _extract_session_compound_metrics(
    session: fastf1.core.Session,
    event_name: str,
    known_teams: set[str],
    canonicalize_team_name_fn=_canonicalize_team_name,
    extract_compound_metrics_fn=extract_compound_metrics,
    normalize_compound_metrics_across_teams_fn=normalize_compound_metrics_across_teams,
) -> dict[str, dict[str, dict[str, float | str | None]]]:
    """Extract and normalize compound metrics for one session."""
    if _session_is_predominantly_wet(session):
        return {}

    laps = session.laps
    if laps is None or laps.empty or "Team" not in laps.columns:
        return {}

    session_compound_metrics = {}
    raw_teams = laps["Team"].dropna().unique()

    for raw_team in raw_teams:
        canonical_team = canonicalize_team_name_fn(str(raw_team), known_teams)
        if not canonical_team:
            continue

        team_laps = laps[laps["Team"] == raw_team]
        compound_data = extract_compound_metrics_fn(team_laps, canonical_team, event_name)
        if compound_data:
            session_compound_metrics[canonical_team] = compound_data

    if not session_compound_metrics:
        return {}

    return normalize_compound_metrics_across_teams_fn(session_compound_metrics, event_name)
