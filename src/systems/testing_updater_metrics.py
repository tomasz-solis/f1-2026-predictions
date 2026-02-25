"""
Metrics and lap-selection helpers for testing updater flows.

Kept as a separate module so the orchestration layer stays focused on loading
sessions and writing outputs.
"""

from __future__ import annotations

import logging

import fastf1
import numpy as np
import pandas as pd

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


def _canonicalize_team_name(raw_team: str, known_teams: set[str]) -> str | None:
    """Map session team name to canonical team key used in characteristics JSON."""
    return map_team_to_characteristics(raw_team, known_teams=known_teams)


def _filter_valid_laps(team_laps: pd.DataFrame) -> pd.DataFrame:
    """Filter to representative non-pit laps."""
    if team_laps.empty:
        return team_laps
    if "LapTime" not in team_laps.columns:
        return team_laps.iloc[0:0].copy()

    # For testing updates we prioritize availability over strict pit filtering.
    # Timed laps are enough to infer early directionality.
    mask = team_laps["LapTime"].notna()
    # Testing sessions often have sparse/inconsistent IsAccurate flags.
    # Enforce this only when explicit True rows exist.
    if "IsAccurate" in team_laps.columns:
        accurate = team_laps["IsAccurate"]
        accurate_true = accurate.eq(True)
        if accurate.notna().any() and bool(accurate_true.any()):
            mask &= accurate_true

    return team_laps[mask].copy()


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

    This avoids over-weighting teams with longer programs in the same session.
    """
    if team_laps.empty:
        return team_laps

    grouping_cols = ["Driver"]
    if "Stint" in team_laps.columns and bool(team_laps["Stint"].notna().any()):
        grouping_cols.append("Stint")
    if "Compound" in team_laps.columns and bool(team_laps["Compound"].notna().any()):
        grouping_cols.append("Compound")

    rows = []
    for _, laps in team_laps.groupby(grouping_cols, dropna=False):
        timed = laps[laps["LapTime"].notna()].copy()
        if timed.empty:
            continue

        lap_seconds = pd.to_timedelta(timed["LapTime"], errors="coerce").dt.total_seconds()
        valid_idx = lap_seconds.dropna().index
        if valid_idx.empty:
            continue

        median_value = float(lap_seconds.loc[valid_idx].median())
        representative_idx = (lap_seconds.loc[valid_idx] - median_value).abs().idxmin()
        rows.append(timed.loc[representative_idx])

    if not rows:
        return team_laps

    return pd.DataFrame(rows).copy()


def _select_program_aware_laps(team_laps: pd.DataFrame, run_profile: str) -> pd.DataFrame:
    """
    Select representative laps with program-aware run filtering.

    Modes:
    - all: use all valid laps
    - short_run: prefer short stints
    - long_run: prefer long stints
    - balanced: blend short + long stints
    """
    if team_laps.empty:
        return team_laps

    if run_profile not in _RUN_PROFILE_MODES:
        raise ValueError(
            f"Invalid run_profile '{run_profile}'. Use one of: {', '.join(_RUN_PROFILE_MODES)}"
        )

    if run_profile == "all":
        selected = team_laps
    else:
        short_laps, long_laps = _classify_run_laps(team_laps)
        if run_profile == "short_run":
            selected = short_laps if not short_laps.empty else team_laps
        elif run_profile == "long_run":
            selected = long_laps if not long_laps.empty else team_laps
        else:
            if not short_laps.empty and not long_laps.empty:
                selected = pd.concat([short_laps, long_laps], ignore_index=False)
            elif not short_laps.empty:
                selected = short_laps
            elif not long_laps.empty:
                selected = long_laps
            else:
                selected = team_laps

    representative = _select_stint_representative_laps(selected)
    return representative if not representative.empty else selected


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
    except Exception:
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

    lap_seconds = pd.to_timedelta(team_laps["LapTime"], errors="coerce").dt.total_seconds()
    lap_seconds = lap_seconds.dropna()
    if lap_seconds.empty:
        return None

    return float(lap_seconds.median())


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
        stint = stint_laps.sort_values("LapNumber")
        if len(stint) < 3:
            continue

        lap_seconds = pd.to_timedelta(stint["LapTime"], errors="coerce").dt.total_seconds()
        lap_seconds = lap_seconds.dropna()
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
    """Normalize a lower-is-better metric into 0-1 scale."""
    if not metric_values:
        return {}

    best = min(metric_values.values())
    worst = max(metric_values.values())
    if worst <= best:
        return {team: 0.5 for team in metric_values}

    normalized = {}
    for team, value in metric_values.items():
        score = 1.0 - ((value - best) / (worst - best))
        normalized[team] = float(np.clip(score, 0.0, 1.0))

    return normalized


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

    speed_columns = [
        col for col in ("SpeedST", "SpeedFL", "SpeedI2", "SpeedI1") if col in valid_laps
    ]
    if speed_columns:
        speed_values = []
        for col in speed_columns:
            speed_values.extend(valid_laps[col].dropna().tolist())
        if speed_values:
            payload["speed_profile"] = {"top_speed": float(np.nanmedian(speed_values))}

    lap_seconds = pd.to_timedelta(valid_laps["LapTime"], errors="coerce").dt.total_seconds()
    lap_seconds = lap_seconds.dropna()
    if len(lap_seconds) >= 2:
        payload["consistency"] = {"std_lap_time": float(lap_seconds.std(ddof=0))}

    return payload


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
    try:
        laps = session.laps
    except Exception as exc:
        logger.debug(f"Session laps unavailable for {session_key}: {exc}")
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

    per_team_payload = {}
    tire_deg_slopes = {}
    lap_pace_seconds = {}
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
        # Allow early-session partial data (e.g., testing day in progress).
        if len(valid_laps) < 1:
            continue

        representative_laps = select_program_aware_laps_fn(valid_laps, run_profile=run_profile)
        if representative_laps.empty:
            representative_laps = valid_laps
        selected_lap_count += len(representative_laps)

        median_lap_seconds = median_lap_seconds_fn(representative_laps)
        if median_lap_seconds is not None:
            lap_pace_seconds[canonical_team] = median_lap_seconds

        payload = extract_team_payload_fn(representative_laps)
        if payload:
            per_team_payload.setdefault(canonical_team, {})[session_key] = payload

        if run_profile in ("balanced", "long_run"):
            _, long_laps = classify_run_laps_fn(valid_laps)
            tire_source = long_laps if not long_laps.empty else valid_laps
        else:
            tire_source = valid_laps

        slope = estimate_tire_deg_slope_fn(tire_source)
        if slope is not None:
            tire_deg_slopes[canonical_team] = slope

    normalized_perf = extract_all_teams_performance_fn(per_team_payload, session_name=session_key)
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
            # when granular sector telemetry is still sparse.
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
    except Exception:
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
