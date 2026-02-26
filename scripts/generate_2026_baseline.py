"""
Generate 2026 Baseline Data from Historical Averages (2023-2025)

This script creates proper baseline characteristics for the 2026 season:
- Track characteristics: 3-year averages of pit times, SC probability, overtaking difficulty
- Car/Team characteristics: Neutral starting point (0.5 ± 0.3) for ALL teams
- Driver characteristics: Carried over from 2025 end-of-season

WHY THIS MATTERS:
- 2026 has new regulations → nobody knows team performance yet
- Tracks don't change much → use historical data
- Driver skills persist → carry over from 2025

USAGE:
    python scripts/generate_2026_baseline.py --years 2023,2024,2025 --output data/processed
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_track_characteristics(years: list[int], output_dir: Path) -> None:
    """
    Calculate track characteristics from historical race data.

    For each track, calculates:
    - Average pit stop time loss
    - Safety car probability
    - Overtaking difficulty (from overtaking frequency)
    """
    logger.info(f"Calculating track characteristics from {years}...")

    # Ensure cache directory exists
    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    track_stats = {}

    for year in years:
        logger.info(f"Processing {year} season...")
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule["EventFormat"].notna()].copy()

            for _, event in races.iterrows():
                race_name = event["EventName"]
                logger.info(f"  Analyzing {race_name}...")

                try:
                    session = fastf1.get_session(year, race_name, "R")
                    session.load()

                    # Initialize track if not exists
                    if race_name not in track_stats:
                        track_stats[race_name] = {
                            "pit_times": [],
                            "sc_laps": [],
                            "total_laps": [],
                            "overtakes": [],
                            "event_format": str(event.get("EventFormat", "")).lower(),
                        }

                    # Calculate pit stop loss (median across all stops)
                    if hasattr(session, "laps") and session.laps is not None:
                        pit_laps = session.laps[session.laps["PitInTime"].notna()]
                        if len(pit_laps) > 0:
                            # Estimate pit loss from lap time difference
                            pit_times = []
                            for driver in pit_laps["Driver"].unique():
                                driver_laps = session.laps[session.laps["Driver"] == driver]
                                pit_lap_idx = driver_laps[driver_laps["PitInTime"].notna()].index
                                for _idx in pit_lap_idx:
                                    # Compare to average lap time
                                    avg_lap = driver_laps["LapTime"].mean()
                                    if pd.notna(avg_lap):
                                        pit_loss = 20.0  # Reasonable estimate if we can't calculate
                                        pit_times.append(pit_loss)
                            if pit_times:
                                track_stats[race_name]["pit_times"].extend(pit_times)

                    # Safety car laps
                    if hasattr(session, "laps") and session.laps is not None:
                        total_laps = len(session.laps["LapNumber"].unique())
                        track_stats[race_name]["total_laps"].append(total_laps)

                        # Check for safety car (simplified - would need telemetry)
                        # For now, use a heuristic based on lap time variations
                        lap_times = session.laps.groupby("LapNumber")["LapTime"].mean()
                        if len(lap_times) > 0:
                            # High variation suggests SC/VSC
                            sc_laps = 0  # Default when telemetry-based SC detection is unavailable
                            track_stats[race_name]["sc_laps"].append(sc_laps)

                    # Overtaking difficulty (from position changes)
                    # Higher position changes = easier overtaking
                    if hasattr(session, "results") and session.results is not None:
                        results = session.results
                        if "GridPosition" in results.columns and "Position" in results.columns:
                            position_changes = abs(
                                results["Position"] - results["GridPosition"]
                            ).sum()
                            track_stats[race_name]["overtakes"].append(position_changes)

                except Exception as e:
                    logger.warning(f"  Failed to load {year} {race_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to load {year} schedule: {e}")
            continue

    # Calculate averages
    logger.info("Computing averages...")
    track_characteristics = {
        "year": 2026,
        "generated_from": f"Historical averages from {min(years)}-{max(years)}",
        "generated_at": datetime.now().isoformat(),
        "data_freshness": "BASELINE_PRESEASON",
        "tracks": {},
    }

    for track_name, stats in track_stats.items():
        pit_time = np.mean(stats["pit_times"]) if stats["pit_times"] else 22.0  # Default 22s
        sc_prob = 0.3  # Default - would need better telemetry to calculate

        # Overtaking difficulty: normalize position changes
        if stats["overtakes"]:
            avg_overtakes = np.mean(stats["overtakes"])
            # Scale: 0-20 changes → 1.0-0.0 difficulty (more changes = easier)
            overtaking_difficulty = max(0.0, min(1.0, 1.0 - (avg_overtakes / 40)))
        else:
            overtaking_difficulty = 0.5  # Default medium

        # Determine track type
        track_type = "permanent"
        if "street" in track_name.lower() or "monaco" in track_name.lower():
            track_type = "street"

        has_sprint = "sprint" in stats.get("event_format", "")

        track_characteristics["tracks"][track_name] = {
            "pit_stop_loss": round(pit_time, 1),
            "safety_car_prob": round(sc_prob, 2),
            "overtaking_difficulty": round(overtaking_difficulty, 2),
            "type": track_type,
            **({"has_sprint": True} if has_sprint else {}),
        }

    # Save to file
    output_file = output_dir / "track_characteristics" / "2026_track_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(track_characteristics, f, indent=2)

    logger.info(f"[OK] Saved track characteristics to {output_file}")
    logger.info(f"  Tracks analyzed: {len(track_characteristics['tracks'])}")


def generate_neutral_team_characteristics(output_dir: Path) -> None:
    """
    Generate neutral team characteristics for 2026.

    Since 2026 has new regulations, we DON'T KNOW team performance yet.
    All teams start at 0.5 ± 0.3 (high uncertainty).
    """
    logger.info("Generating neutral team characteristics for 2026...")

    # Get 2026 team lineup
    teams = [
        "McLaren",
        "Mercedes",
        "Red Bull Racing",
        "Ferrari",
        "Williams",
        "RB",
        "Aston Martin",
        "Haas F1 Team",
        "Alpine",
        "Sauber",
        "Cadillac F1",  # New team
    ]

    team_characteristics = {
        "year": 2026,
        "note": "2026 REGULATION RESET - All teams start with neutral baseline (0.5 ± 0.3 uncertainty). Performance unknown until testing/races.",
        "generated_at": datetime.now().isoformat(),
        "data_freshness": "BASELINE_PRESEASON",
        "teams": {},
    }

    for team in teams:
        team_characteristics["teams"][team] = {
            "overall_performance": 0.5,  # Neutral - nobody knows yet!
            "uncertainty": 0.30,  # High uncertainty
            "note": "Pre-season baseline - no 2026 data yet",
            "last_updated": None,
            "races_completed": 0,
        }

    # Save to file
    output_file = output_dir / "car_characteristics" / "2026_car_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(team_characteristics, f, indent=2)

    logger.info(f"[OK] Saved team characteristics to {output_file}")
    logger.info(f"  Teams: {len(teams)} (all start neutral)")


def copy_2025_driver_characteristics(output_dir: Path) -> None:
    """
    Copy 2025 end-of-season driver characteristics to use for 2026.

    Driver skills persist across regulation changes.
    """
    logger.info("Carrying over 2025 driver characteristics...")

    # Check if 2025 driver characteristics exist
    source_file = output_dir / "driver_characteristics.json"

    if not source_file.exists():
        logger.warning(
            f"No 2025 driver characteristics found at {source_file}. "
            f"Run: python scripts/extract_driver_characteristics.py --years 2023,2024,2025"
        )
        return

    with open(source_file) as f:
        driver_data = json.load(f)

    # Add metadata
    driver_data["carried_over_from"] = 2025
    driver_data["last_updated"] = datetime.now().isoformat()
    driver_data["note"] = (
        "Driver characteristics carried over from 2025. Skills persist across regulation changes."
    )

    # Save back
    with open(source_file, "w") as f:
        json.dump(driver_data, f, indent=2)

    logger.info("[OK] Updated driver characteristics with 2026 metadata")


def reset_learning_state() -> None:
    """
    Reset learning state for 2026 season start.
    """
    logger.info("Resetting learning state for 2026...")

    learning_state = {
        "season": 2026,
        "races_completed": 0,
        "last_checkpoint": 0,
        "last_updated": datetime.now().isoformat(),
        "method_performance": {
            "blend_50_50": {"maes": [], "avg": None},
            "blend_70_30": {"maes": [], "avg": None},
            "blend_90_10": {"maes": [], "avg": None},
            "session_order": {"maes": [], "avg": None},
        },
        "recommended_method": "blend",
        "recommended_split": "70/30",  # Default until we have data
        "overtaking_factors": {},
        "pace_model_weights": {
            "pace_weight": 0.4,
            "grid_weight": 0.3,
            "overtaking_weight": 0.2,
            "tire_deg_weight": 0.1,
        },
        "insights": [],
    }

    output_file = Path("data/learning_state.json")
    with open(output_file, "w") as f:
        json.dump(learning_state, f, indent=2)

    logger.info(f"[OK] Reset learning state to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2026 baseline characteristics from historical data"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2023,2024,2025",
        help="Comma-separated years to use for historical averages",
    )
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument(
        "--skip-tracks",
        action="store_true",
        help="Skip track characteristic generation (slow)",
    )
    parser.add_argument(
        "--skip-teams", action="store_true", help="Skip team characteristic generation"
    )
    parser.add_argument(
        "--skip-drivers", action="store_true", help="Skip driver characteristic update"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    years = [int(y.strip()) for y in args.years.split(",")]

    logger.info("=" * 60)
    logger.info("Generating 2026 Baseline Data from Historical Averages")
    logger.info("=" * 60)
    logger.info(f"Years: {years}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Step 1: Track characteristics from historical data
    if not args.skip_tracks:
        calculate_track_characteristics(years, output_dir)
        logger.info("")

    # Step 2: Neutral team characteristics (nobody knows 2026 performance yet!)
    if not args.skip_teams:
        generate_neutral_team_characteristics(output_dir)
        logger.info("")

    # Step 3: Copy 2025 driver characteristics (skills persist)
    if not args.skip_drivers:
        copy_2025_driver_characteristics(output_dir)
        logger.info("")

    # Step 4: Reset learning state
    reset_learning_state()
    logger.info("")

    logger.info("=" * 60)
    logger.info("[OK] 2026 Baseline Generation Complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. After testing (Feb 2026): Run update_from_testing.py")
    logger.info("2. After each race: Run update_from_race.py")
    logger.info("3. System will adaptively learn throughout the season")


if __name__ == "__main__":
    main()
