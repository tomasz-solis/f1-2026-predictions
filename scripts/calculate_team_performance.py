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
from datetime import datetime
from pathlib import Path

import fastf1 as ff1
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_team_performance_from_races(year: int) -> dict:
    """
    Calculate team performance ratings from actual race data.

    Method:
    1. For each race, get median lap time for each team
    2. Normalize to fastest team = 1.0
    3. Average across all races
    4. Result: data-driven performance ratings (0.0-1.0)
    """
    logger.info(f"Calculating team performance for {year}...")

    team_race_performances = {}  # {team: [race1_perf, race2_perf, ...]}

    schedule = ff1.get_event_schedule(year)
    races = schedule[schedule["EventFormat"] != "testing"]

    for _, event in races.iterrows():
        race_name = event["EventName"]
        if not race_name:
            continue

        try:
            session = ff1.get_session(year, race_name, "R")
            if session.date > pd.Timestamp.now(tz="UTC"):
                continue  # Race hasn't happened yet

            logger.info(f"  Analyzing {race_name}...")
            session.load(laps=True, telemetry=False)

            laps = session.laps

            # Calculate median lap time per team
            team_times = {}
            for team in laps["Team"].unique():
                if pd.isna(team):
                    continue

                team_laps = laps[laps["Team"] == team]
                clean_laps = team_laps.pick_accurate().pick_quicklaps()

                if len(clean_laps) >= 10:  # Need enough data
                    median_time = clean_laps["LapTime"].dt.total_seconds().median()
                    team_times[team] = median_time

            if not team_times:
                logger.warning(f"  No valid data for {race_name}")
                continue

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

    # Aggregate across season
    team_characteristics = {}

    for team, performances in team_race_performances.items():
        if len(performances) < 3:
            logger.warning(f"Skipping {team} - only {len(performances)} races")
            continue

        avg_performance = np.mean(performances)
        std_performance = np.std(performances)

        # Uncertainty = how variable their performance was
        # Lower std = more consistent = lower uncertainty
        uncertainty = np.clip(std_performance * 5.0, 0.10, 0.40)

        team_characteristics[team] = {
            "overall_performance": round(avg_performance, 3),
            "uncertainty": round(uncertainty, 2),
            "races_analyzed": len(performances),
            "note": f"Calculated from {len(performances)} races in {year}",
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

    logger.info("=" * 60)
    logger.info(f"Calculating {args.year} Team Performance")
    logger.info("=" * 60)
    logger.info("")

    # Calculate
    team_chars = calculate_team_performance_from_races(args.year)

    # Rank
    team_chars = rank_teams_by_performance(team_chars)

    # Package
    output = {
        "year": args.year,
        "generated_at": datetime.now().isoformat(),
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
    logger.info("=" * 60)
    logger.info(f"[OK] Calculated {len(team_chars)} team ratings")
    logger.info(f" Saved to: {output_path}")
    logger.info("=" * 60)
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
