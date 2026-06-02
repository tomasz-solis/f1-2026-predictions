"""
Data-Driven Team Performance Calculator

Calculates team/car performance from actual race lap time data.
Uses median lap times (normalized) to determine relative team strength.

USAGE:
    python scripts/calculate_team_performance.py --year 2025 --output data/processed/car_characteristics/2025_car_characteristics.json
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import fastf1 as ff1
import numpy as np
import pandas as pd

from src.utils.normalization import rank_normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _coerce_utc_timestamp(value: object) -> pd.Timestamp | None:
    """Coerce datetime-like values to timezone-aware UTC pandas timestamps."""
    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _lap_top_speed_series(team_laps: pd.DataFrame) -> pd.Series:
    """Return one top-speed sample per lap from the best available trap columns."""
    preferred_columns = [column for column in ("SpeedST", "SpeedFL") if column in team_laps.columns]
    fallback_columns = [column for column in ("SpeedI2", "SpeedI1") if column in team_laps.columns]
    speed_columns = preferred_columns or fallback_columns
    if not speed_columns:
        return pd.Series(dtype=float)

    speed_frame = team_laps[speed_columns].apply(pd.to_numeric, errors="coerce")
    return speed_frame.max(axis=1, skipna=True).dropna()


# See compound_analyzer._FUEL_SLOPE_CORRECTION_S_PER_LAP for derivation.
_FUEL_SLOPE_CORRECTION_S_PER_LAP: float = 0.045


def _estimate_tire_deg_slope(team_laps: pd.DataFrame) -> float | None:
    """Estimate fuel-corrected tire degradation slope (s/lap) across all compounds.

    Adds 0.045 s/lap to each raw regression slope to remove the systematic
    improvement from fuel burn. Without this correction, durable compounds
    (HARD/MEDIUM) often show negative slopes that reflect fuel weight loss,
    not actual rubber wear.
    """
    if team_laps.empty or "LapTime" not in team_laps.columns:
        return None

    grouping_columns = ["Driver"]
    if "Stint" in team_laps.columns and bool(team_laps["Stint"].notna().any()):
        grouping_columns.append("Stint")
    if "Compound" in team_laps.columns and bool(team_laps["Compound"].notna().any()):
        grouping_columns.append("Compound")

    slopes: list[float] = []
    for _, stint_laps in team_laps.groupby(grouping_columns, dropna=False):
        ordered_laps = stint_laps
        if "LapNumber" in stint_laps.columns:
            ordered_laps = stint_laps.sort_values("LapNumber")

        lap_seconds = pd.to_timedelta(ordered_laps["LapTime"], errors="coerce").dt.total_seconds()
        lap_seconds = lap_seconds.dropna()
        if len(lap_seconds) < 3:
            continue

        x = np.arange(len(lap_seconds), dtype=float)
        y = lap_seconds.to_numpy(dtype=float)
        raw_slope = float(np.polyfit(x, y, 1)[0])
        corrected_slope = raw_slope + _FUEL_SLOPE_CORRECTION_S_PER_LAP

        if -0.10 <= corrected_slope <= 0.50:
            slopes.append(corrected_slope)

    if not slopes:
        return None
    return float(np.median(np.asarray(slopes, dtype=float)))


def _append_normalized_metric_samples(
    team_metric_samples: dict[str, dict[str, list[float]]],
    *,
    metric_name: str,
    metric_values: dict[str, float],
    higher_is_better: bool,
) -> None:
    """Append one race worth of rank-normalized team metric samples."""
    if len(metric_values) < 2:
        return

    normalized_scores = rank_normalize(metric_values, higher_is_better=higher_is_better)
    for team_name, score in normalized_scores.items():
        team_metric_samples.setdefault(team_name, {}).setdefault(metric_name, []).append(
            float(score)
        )


def calculate_team_performance_from_races(
    year: int,
    *,
    max_races: int | None = None,
) -> dict:
    """
    Calculate team performance ratings from actual race data.

    Method:
    1. For each race, get median lap time for each team
    2. Normalize to fastest team = 1.0
    3. Average across all races
    4. Result: data-driven performance ratings (0.0-1.0)
    """
    logger.info(f"Calculating team performance for {year}...")

    team_race_performances: dict[str, list[float]] = {}
    team_metric_samples: dict[str, dict[str, list[float]]] = defaultdict(dict)

    schedule = ff1.get_event_schedule(year)
    races = schedule[schedule["EventFormat"] != "testing"]

    races_analyzed = 0
    for _, event in races.iterrows():
        if max_races is not None and races_analyzed >= max_races:
            break

        race_name = event["EventName"]
        if not race_name:
            continue

        try:
            session = ff1.get_session(year, race_name, "R")
            session_date = _coerce_utc_timestamp(getattr(session, "date", None))
            if session_date is not None and session_date > pd.Timestamp.now(tz=UTC):
                continue  # Race hasn't happened yet
            races_analyzed += 1

            logger.info(f"  Analyzing {race_name}...")
            session.load(laps=True, telemetry=False)

            laps = session.laps

            # Calculate median lap time per team
            team_times = {}
            team_sector_1 = {}
            team_sector_2 = {}
            team_sector_3 = {}
            team_top_speed = {}
            team_consistency = {}
            team_tire_deg = {}
            for team in laps["Team"].unique():
                if pd.isna(team):
                    continue

                team_laps = laps[laps["Team"] == team]
                clean_laps = team_laps.pick_accurate().pick_quicklaps()

                if len(clean_laps) >= 10:  # Need enough data
                    lap_seconds = clean_laps["LapTime"].dt.total_seconds().dropna()
                    if lap_seconds.empty:
                        continue

                    median_time = lap_seconds.median()
                    team_times[team] = median_time

                    if "Sector1Time" in clean_laps.columns:
                        sector_1 = (
                            pd.to_timedelta(
                                clean_laps["Sector1Time"],
                                errors="coerce",
                            )
                            .dt.total_seconds()
                            .dropna()
                        )
                        if not sector_1.empty:
                            team_sector_1[team] = float(sector_1.median())

                    if "Sector2Time" in clean_laps.columns:
                        sector_2 = (
                            pd.to_timedelta(
                                clean_laps["Sector2Time"],
                                errors="coerce",
                            )
                            .dt.total_seconds()
                            .dropna()
                        )
                        if not sector_2.empty:
                            team_sector_2[team] = float(sector_2.median())

                    if "Sector3Time" in clean_laps.columns:
                        sector_3 = (
                            pd.to_timedelta(
                                clean_laps["Sector3Time"],
                                errors="coerce",
                            )
                            .dt.total_seconds()
                            .dropna()
                        )
                        if not sector_3.empty:
                            team_sector_3[team] = float(sector_3.median())

                    top_speed_samples = _lap_top_speed_series(clean_laps)
                    if not top_speed_samples.empty:
                        if len(top_speed_samples) >= 4:
                            team_top_speed[team] = float(top_speed_samples.quantile(0.90))
                        else:
                            team_top_speed[team] = float(top_speed_samples.max())

                    if len(lap_seconds) >= 5:
                        team_consistency[team] = float(lap_seconds.std(ddof=0))

                    tire_deg_slope = _estimate_tire_deg_slope(clean_laps)
                    if tire_deg_slope is not None:
                        team_tire_deg[team] = tire_deg_slope

            if not team_times:
                logger.warning(f"  No valid data for {race_name}")
                continue

            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_overall_pace",
                metric_values=team_times,
                higher_is_better=False,
            )
            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_slow_corner_performance",
                metric_values=team_sector_1,
                higher_is_better=False,
            )
            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_medium_corner_performance",
                metric_values=team_sector_2,
                higher_is_better=False,
            )
            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_fast_corner_performance",
                metric_values=team_sector_3,
                higher_is_better=False,
            )
            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_top_speed",
                metric_values=team_top_speed,
                higher_is_better=True,
            )
            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_consistency",
                metric_values=team_consistency,
                higher_is_better=False,
            )
            _append_normalized_metric_samples(
                team_metric_samples,
                metric_name="normalized_tire_deg_performance",
                metric_values=team_tire_deg,
                higher_is_better=False,
            )

            # Normalize to fastest team = 1.0
            fastest_time = min(team_times.values())

            for team, time in team_times.items():
                # Performance = faster_time / slower_time
                # e.g., 90s vs 91s = 90/91 = 0.989 (1.1% slower)
                performance = fastest_time / time

                if team not in team_race_performances:
                    team_race_performances[team] = []

                team_race_performances[team].append(performance)

        except Exception as e:
            logger.warning(f"  Failed to load {race_name}: {e}")
            continue

    # Aggregate across season.
    # Require at least 2 races, or all available races if fewer than 3 completed.
    # A fixed minimum of 3 incorrectly excludes teams with DNFs early in the season
    # when only 3 rounds have been run.
    min_races_required = min(3, max(2, races_analyzed - 1))
    team_characteristics = {}

    for team, performances in team_race_performances.items():
        if len(performances) < min_races_required:
            logger.warning(
                f"Skipping {team} - only {len(performances)} races (need {min_races_required})"
            )
            continue

        avg_performance = np.mean(performances)
        std_performance = np.std(performances)

        # Uncertainty = how variable their performance was
        # Lower std = more consistent = lower uncertainty
        uncertainty = np.clip(std_performance * 5.0, 0.10, 0.40)

        normalized_metrics = {
            metric_name: round(float(np.mean(samples)), 3)
            for metric_name, samples in team_metric_samples.get(team, {}).items()
            if samples
        }

        team_characteristics[team] = {
            "overall_performance": round(avg_performance, 3),
            "uncertainty": round(uncertainty, 2),
            "races_analyzed": len(performances),
            "note": f"Calculated from {len(performances)} races in {year}",
            **normalized_metrics,
        }

    return team_characteristics


def rank_teams_by_performance(teams: dict) -> dict:
    """Add championship position based on performance."""
    sorted_teams = sorted(teams.items(), key=lambda x: x[1]["overall_performance"], reverse=True)

    for position, (team, _data) in enumerate(sorted_teams, 1):
        teams[team]["championship_position"] = position

    return teams


def main():
    parser = argparse.ArgumentParser(description="Calculate team performance from race data")
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/car_characteristics/2025_car_characteristics.json",
        help="Output file path",
    )

    args = parser.parse_args()

    # Ensure cache
    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_dir))

    logger.info("")
    logger.info(f"Calculating {args.year} Team Performance")
    logger.info("")
    logger.info("")

    # Calculate
    team_chars = calculate_team_performance_from_races(args.year)

    # Rank
    team_chars = rank_teams_by_performance(team_chars)

    # Package
    output = {
        "year": args.year,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_freshness": "DATA_DRIVEN",
        "method": "Calculated from race lap times (median per team, normalized)",
        "races_completed": max([t["races_analyzed"] for t in team_chars.values()], default=0),
        "teams": team_chars,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("")
    logger.info("")
    logger.info(f"[OK] Calculated {len(team_chars)} team ratings")
    logger.info(f" Saved to: {output_path}")
    logger.info("")
    logger.info("")
    logger.info("Top 5 teams:")
    sorted_teams = sorted(
        team_chars.items(), key=lambda x: x[1]["overall_performance"], reverse=True
    )
    for team, data in sorted_teams[:5]:
        perf = data["overall_performance"]
        unc = data["uncertainty"]
        races = data["races_analyzed"]
        logger.info(
            f"  P{data['championship_position']} {team:20s}: {perf:.3f} (±{unc:.2f}, {races} races)"
        )


if __name__ == "__main__":
    main()
