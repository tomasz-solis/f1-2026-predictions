"""
Compound-Specific Performance Analyzer

Extracts tire compound performance (pace, degradation, consistency) from session data.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard F1 tire compounds
STANDARD_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
# Minimum laps required to consider compound data reliable
# Increased from 3 to 8: first 2-3 laps are tire warmup
MIN_LAPS_PER_COMPOUND = 8


def _as_numeric_metric(value: float | str | None) -> float | None:
    """Safely coerce numeric metric values while ignoring labels/unknowns."""
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalize_compound_name(compound: str | None) -> str | None:
    """Normalize compound names to standard format (SOFT/MEDIUM/HARD)."""
    if compound is None or pd.isna(compound):
        return None

    normalized = str(compound).upper().strip()

    # Handle various FastF1 compound naming conventions
    if "SOFT" in normalized or normalized == "S":
        return "SOFT"
    elif "MEDIUM" in normalized or normalized == "M":
        return "MEDIUM"
    elif "HARD" in normalized or normalized == "H":
        return "HARD"
    elif "INTERMEDIATE" in normalized or normalized == "I":
        return "INTERMEDIATE"
    elif "WET" in normalized or normalized == "W":
        return "WET"

    return normalized


def _median_lap_seconds_series(lap_times: pd.Series) -> float | None:
    """Calculate median lap time in seconds."""
    if lap_times is None or lap_times.empty:
        return None

    lap_seconds = pd.to_timedelta(lap_times, errors="coerce").dt.total_seconds()
    lap_seconds = lap_seconds.dropna()

    if lap_seconds.empty:
        return None

    return float(lap_seconds.median())


def _estimate_compound_tire_deg(compound_laps: pd.DataFrame) -> float | None:
    """Estimate tire degradation slope in seconds/lap (higher = more degradation)."""
    if compound_laps.empty or "LapNumber" not in compound_laps.columns:
        return None

    grouping_cols = ["Driver"]
    if "Stint" in compound_laps.columns:
        grouping_cols.append("Stint")

    slopes = []
    for _, stint_laps in compound_laps.groupby(grouping_cols, dropna=False):
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
        # Filter unrealistic slopes (negative or extreme)
        if -0.3 <= slope <= 1.0:
            slopes.append(slope)

    if not slopes:
        return None

    return float(np.median(slopes))


def _calculate_compound_consistency(compound_laps: pd.DataFrame) -> float | None:
    """Calculate lap time consistency (standard deviation)."""
    if compound_laps.empty or "LapTime" not in compound_laps.columns:
        return None

    lap_seconds = pd.to_timedelta(compound_laps["LapTime"], errors="coerce").dt.total_seconds()
    lap_seconds = lap_seconds.dropna()

    if len(lap_seconds) < 3:
        return None

    return float(lap_seconds.std(ddof=1))


def extract_compound_metrics(
    team_laps: pd.DataFrame,
    canonical_team: str,
    track_name: str,
) -> dict[str, dict[str, float | str | None]]:
    """Extract compound-specific metrics for a team at a specific track."""
    if team_laps.empty or "Compound" not in team_laps.columns:
        return {}

    compound_metrics: dict[str, dict[str, float | str | None]] = {}

    # Filter to valid, clean laps with compound info
    # Remove pit laps, inaccurate laps, and outliers
    valid_mask = team_laps["LapTime"].notna() & team_laps["Compound"].notna()

    # Add FastF1 quality filters if available
    if "IsAccurate" in team_laps.columns:
        valid_mask &= team_laps["IsAccurate"]

    # Filter out pit in/out laps (inflate lap times)
    if "PitInTime" in team_laps.columns:
        valid_mask &= team_laps["PitInTime"].isna()
    if "PitOutTime" in team_laps.columns:
        valid_mask &= team_laps["PitOutTime"].isna()

    valid_laps = team_laps[valid_mask].copy()

    if valid_laps.empty:
        return {}

    # Normalize compound names
    valid_laps["_normalized_compound"] = valid_laps["Compound"].apply(_normalize_compound_name)

    # Group by compound
    for compound_name, compound_laps in valid_laps.groupby("_normalized_compound", dropna=True):
        if compound_name is None:
            continue

        laps_count = len(compound_laps)

        # Skip compounds with insufficient data
        if laps_count < MIN_LAPS_PER_COMPOUND:
            logger.debug(
                f"  {canonical_team} {compound_name}: Only {laps_count} laps, skipping (need {MIN_LAPS_PER_COMPOUND})"
            )
            continue

        # Keep the payload shape stable even when a metric is missing.
        metrics: dict[str, float | str | None] = {
            "laps_count": float(laps_count),
            "track_name": track_name,
        }

        # Median lap time
        median_time = _median_lap_seconds_series(compound_laps["LapTime"])
        metrics["median_lap_time"] = median_time if median_time is not None else None

        # Tire degradation
        tire_deg = _estimate_compound_tire_deg(compound_laps)
        metrics["tire_deg_slope"] = tire_deg if tire_deg is not None else None

        # Consistency
        consistency = _calculate_compound_consistency(compound_laps)
        metrics["consistency"] = consistency if consistency is not None else None

        # Keep compounds that yielded at least one usable metric.
        has_data = (
            metrics["median_lap_time"] is not None
            or metrics["tire_deg_slope"] is not None
            or metrics["consistency"] is not None
        )

        if has_data:
            compound_metrics[compound_name] = metrics
            logger.debug(
                f"  {canonical_team} {compound_name} @ {track_name}: "
                f"{laps_count} laps, "
                f"pace={median_time:.3f}s, "
                if median_time
                else f"deg={tire_deg:.4f}s/lap"
                if tire_deg
                else ""
            )

    return compound_metrics


def normalize_compound_metrics_across_teams(
    all_team_compound_metrics: dict[str, dict[str, dict[str, float | str | None]]],
    track_name: str,
) -> dict[str, dict[str, dict[str, float | str | None]]]:
    """Normalize compound metrics to 0-1 scale (track-specific, avoids cross-track comparison)."""
    normalized_output: dict[str, dict[str, dict[str, float | str | None]]] = {}

    # Collect all values per compound+metric for normalization (within same track)
    compound_metric_values: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for team_name, compounds in all_team_compound_metrics.items():
        for compound, metrics in compounds.items():
            # Only compare metrics from same track
            if metrics.get("track_name") == track_name:
                for metric_name, value in metrics.items():
                    if metric_name not in (
                        "laps_count",
                        "track_name",
                    ):  # Skip non-performance metrics
                        if isinstance(value, int | float):
                            compound_metric_values[compound][metric_name].append(
                                (team_name, float(value))
                            )

    # Normalize each compound+metric independently (within track)
    for team_name, compounds in all_team_compound_metrics.items():
        normalized_output[team_name] = {}

        for compound, metrics in compounds.items():
            normalized_metrics = metrics.copy()

            # Normalize median_lap_time (lower is better)
            median_lap_time = _as_numeric_metric(metrics.get("median_lap_time"))
            if median_lap_time is not None:
                all_values = [
                    v
                    for _, v in compound_metric_values[compound].get("median_lap_time", [])
                    if v is not None
                ]
                if len(all_values) > 1:
                    best = min(all_values)
                    worst = max(all_values)
                    if worst > best:
                        normalized = 1.0 - ((median_lap_time - best) / (worst - best))
                        normalized_metrics["pace_performance"] = float(
                            np.clip(normalized, 0.0, 1.0)
                        )
                    else:
                        normalized_metrics["pace_performance"] = 0.5
                else:
                    normalized_metrics["pace_performance"] = None
            else:
                normalized_metrics["pace_performance"] = None

            # Normalize tire_deg_slope (lower is better - less degradation)
            tire_deg_slope = _as_numeric_metric(metrics.get("tire_deg_slope"))
            if tire_deg_slope is not None:
                all_values = [
                    v
                    for _, v in compound_metric_values[compound].get("tire_deg_slope", [])
                    if v is not None
                ]
                if len(all_values) > 1:
                    best = min(all_values)
                    worst = max(all_values)
                    if worst > best:
                        normalized = 1.0 - ((tire_deg_slope - best) / (worst - best))
                        normalized_metrics["tire_deg_performance"] = float(
                            np.clip(normalized, 0.0, 1.0)
                        )
                    else:
                        normalized_metrics["tire_deg_performance"] = 0.5
                else:
                    normalized_metrics["tire_deg_performance"] = None
            else:
                normalized_metrics["tire_deg_performance"] = None

            # Normalize consistency (lower is better - more consistent)
            consistency = _as_numeric_metric(metrics.get("consistency"))
            if consistency is not None:
                all_values = [
                    v
                    for _, v in compound_metric_values[compound].get("consistency", [])
                    if v is not None
                ]
                if len(all_values) > 1:
                    best = min(all_values)
                    worst = max(all_values)
                    if worst > best:
                        normalized = 1.0 - ((consistency - best) / (worst - best))
                        normalized_metrics["consistency_performance"] = float(
                            np.clip(normalized, 0.0, 1.0)
                        )
                    else:
                        normalized_metrics["consistency_performance"] = 0.5
                else:
                    normalized_metrics["consistency_performance"] = None
            else:
                normalized_metrics["consistency_performance"] = None

            normalized_output[team_name][compound] = normalized_metrics

    return normalized_output


def aggregate_compound_samples(
    existing_compound_chars: dict[str, dict[str, float | str | None]],
    new_compound_metrics: dict[str, dict[str, float | str | None]],
    blend_weight: float = 0.5,
    race_name: str | None = None,
) -> dict[str, dict[str, float | str | None]]:
    """Blend existing compound data with new session data (track-aware, only blends same track)."""
    blended: dict[str, dict[str, float | str | None]] = {}

    # Process all compounds (existing + new)
    all_compounds = set(existing_compound_chars.keys()) | set(new_compound_metrics.keys())

    for compound in all_compounds:
        existing = existing_compound_chars.get(compound, {})
        new = new_compound_metrics.get(compound, {})

        if not new:
            # Keep existing if no new data
            blended[compound] = existing.copy()
            continue

        if not existing:
            # Use new data if no existing
            blended[compound] = new.copy()
            # Rename laps_count to laps_sampled for storage
            if "laps_count" in blended[compound]:
                blended[compound]["laps_sampled"] = blended[compound].pop("laps_count")
            if "sessions_used" not in blended[compound]:
                blended[compound]["sessions_used"] = 1
            continue

        # Check if existing data is from same track
        existing_track = existing.get("track_name")
        new_track = new.get("track_name")

        # Only blend if same track (or track unknown in existing data)
        if existing_track and new_track and existing_track != new_track:
            # Different tracks - use new data only
            blended[compound] = new.copy()
            if "laps_count" in blended[compound]:
                blended[compound]["laps_sampled"] = blended[compound].pop("laps_count")
            if "sessions_used" not in blended[compound]:
                blended[compound]["sessions_used"] = 1
            continue

        # Blend metrics (same track)
        blended_metrics: dict[str, float | str | None] = (
            {"track_name": new_track} if new_track else {}
        )

        # Metrics to blend (handle None values)
        blend_metrics = [
            "pace_performance",
            "tire_deg_slope",
            "tire_deg_performance",
            "consistency",
            "median_lap_time",
        ]

        for metric in blend_metrics:
            old_val = existing.get(metric)
            new_val = new.get(metric)

            if isinstance(old_val, int | float) and isinstance(new_val, int | float):
                blended_metrics[metric] = (1 - blend_weight) * old_val + blend_weight * new_val
            elif new_val is not None:
                blended_metrics[metric] = new_val
            elif old_val is not None:
                blended_metrics[metric] = old_val
            else:
                blended_metrics[metric] = None

        # Update session counts
        old_laps = existing.get("laps_sampled", 0)
        new_laps = new.get("laps_count", 0)
        old_laps_num = float(old_laps) if isinstance(old_laps, int | float) else 0.0
        new_laps_num = float(new_laps) if isinstance(new_laps, int | float) else 0.0
        blended_metrics["laps_sampled"] = old_laps_num + new_laps_num

        old_sessions = existing.get("sessions_used", 0)
        old_sessions_num = float(old_sessions) if isinstance(old_sessions, int | float) else 0.0
        blended_metrics["sessions_used"] = old_sessions_num + 1.0

        blended[compound] = blended_metrics

    return blended
