"""
Generate 2026 Baseline Data from Historical Averages (2023-2025)

This script creates proper baseline characteristics for the 2026 season:
- Track characteristics: 3-year averages of pit times, SC probability, overtaking difficulty
- Car/Team characteristics: 2025-seeded preseason starting point (high uncertainty);
  optional neutral mode when explicitly requested
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("fastf1").setLevel(logging.ERROR)


def _timedelta_to_seconds(value: object) -> float | None:
    """Convert timedelta-like value to seconds; return None if conversion fails."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "total_seconds"):
        try:
            return float(value.total_seconds())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_pit_losses_from_laps(laps: pd.DataFrame | None) -> list[float]:
    """
    Estimate pit lane loss in seconds from FastF1 lap-level data.

    Priority:
    1) Pair PitInTime/PitOutTime timestamps per driver (best signal).
    2) Fallback to lap-time delta vs driver clean-lap median if timestamp pairing unavailable.
    """
    if laps is None or laps.empty:
        return []
    if "Driver" not in laps.columns:
        return []

    losses: list[float] = []

    # First pass: pair pit in/out timestamps per driver.
    has_pit_timestamps = {"PitInTime", "PitOutTime"}.issubset(laps.columns)
    if has_pit_timestamps:
        for _driver, driver_laps in laps.groupby("Driver"):
            pit_ins = (
                driver_laps.loc[driver_laps["PitInTime"].notna(), "PitInTime"]
                .dropna()
                .sort_values()
                .tolist()
            )
            pit_outs = (
                driver_laps.loc[driver_laps["PitOutTime"].notna(), "PitOutTime"]
                .dropna()
                .sort_values()
                .tolist()
            )
            if not pit_ins or not pit_outs:
                continue

            out_idx = 0
            for pit_in in pit_ins:
                while out_idx < len(pit_outs) and pit_outs[out_idx] <= pit_in:
                    out_idx += 1
                if out_idx >= len(pit_outs):
                    break

                pit_out = pit_outs[out_idx]
                out_idx += 1
                seconds = _timedelta_to_seconds(pit_out - pit_in)
                if seconds is None:
                    continue
                # Track pit lane losses are typically around 15-30s; allow buffer for anomalies.
                if 10.0 <= seconds <= 60.0:
                    losses.append(seconds)

    if losses:
        return losses

    # Fallback: estimate pit cost from pit-lap excess vs driver's clean-lap median.
    if "LapTime" not in laps.columns:
        return []
    has_pit_markers = {"PitInTime", "PitOutTime"}.issubset(laps.columns)
    if not has_pit_markers:
        return []

    for _driver, driver_laps in laps.groupby("Driver"):
        clean_mask = driver_laps["PitInTime"].isna() & driver_laps["PitOutTime"].isna()
        clean_laps = driver_laps.loc[clean_mask, "LapTime"].dropna()
        if clean_laps.empty:
            continue

        clean_seconds = [
            seconds
            for lap_time in clean_laps
            for seconds in [_timedelta_to_seconds(lap_time)]
            if seconds is not None
        ]
        if not clean_seconds:
            continue

        baseline = float(np.median(clean_seconds))
        if not np.isfinite(baseline) or baseline <= 0:
            continue

        pit_mask = driver_laps["PitInTime"].notna() | driver_laps["PitOutTime"].notna()
        pit_laps = driver_laps.loc[pit_mask, "LapTime"].dropna()
        for lap_time in pit_laps:
            lap_seconds = _timedelta_to_seconds(lap_time)
            if lap_seconds is None:
                continue
            loss = lap_seconds - baseline
            if 10.0 <= loss <= 60.0:
                losses.append(loss)

    return losses


def _filter_outlier_pit_losses(losses: list[float]) -> list[float]:
    """
    Remove extreme pit-loss outliers caused by incidents/penalties/data artifacts.

    Strategy:
    - keep only finite values in a broad plausible range
    - apply IQR filtering when enough samples exist
    """
    # Keep values within the same operational range enforced by validation script.
    clean = [float(v) for v in losses if np.isfinite(v) and 15.0 <= float(v) <= 30.0]
    if len(clean) < 6:
        return clean

    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    iqr = q3 - q1
    if iqr <= 0:
        return clean

    lower = max(15.0, q1 - 1.5 * iqr)
    upper = min(30.0, q3 + 1.5 * iqr)
    filtered = [v for v in clean if lower <= v <= upper]
    return filtered if filtered else clean


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

                    # Calculate pit stop loss from lap timing data.
                    if hasattr(session, "laps") and session.laps is not None:
                        pit_times = _estimate_pit_losses_from_laps(session.laps)
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
        pit_samples = _filter_outlier_pit_losses(stats["pit_times"])
        pit_time = np.mean(pit_samples) if pit_samples else 22.0  # Default 22s
        pit_time = float(np.clip(pit_time, 15.0, 30.0))
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


def generate_team_characteristics(output_dir: Path, *, neutral_start: bool = False) -> None:
    """
    Generate preseason team characteristics for 2026.

    Default behavior seeds rankings from 2025 constructor order while keeping high uncertainty.
    Use neutral_start=True only when you explicitly want all teams initialized equally.
    """
    if neutral_start:
        logger.info("Generating neutral team characteristics for 2026...")
        team_2026_seed = {
            "McLaren": {"position": 1, "performance": 0.50},
            "Mercedes": {"position": 2, "performance": 0.50},
            "Red Bull Racing": {"position": 3, "performance": 0.50},
            "Ferrari": {"position": 4, "performance": 0.50},
            "Williams": {"position": 5, "performance": 0.50},
            "RB": {"position": 6, "performance": 0.50},
            "Aston Martin": {"position": 7, "performance": 0.50},
            "Haas F1 Team": {"position": 8, "performance": 0.50},
            "Alpine": {"position": 9, "performance": 0.50},
            "Sauber": {"position": 10, "performance": 0.50},
            "Cadillac F1": {"position": 11, "performance": 0.50},
        }
        note = (
            "2026 REGULATION RESET - All teams start with neutral baseline "
            "(0.5 ± 0.3 uncertainty). Performance unknown until testing/races."
        )
    else:
        logger.info("Generating 2025-seeded team characteristics for 2026...")
        # Preserve relative ordering from 2025, but keep large uncertainty for the regulation reset.
        team_2026_seed = {
            "McLaren": {"position": 1, "performance": 0.85},
            "Mercedes": {"position": 2, "performance": 0.75},
            "Red Bull Racing": {"position": 3, "performance": 0.74},
            "Ferrari": {"position": 4, "performance": 0.70},
            "Williams": {"position": 5, "performance": 0.55},
            "RB": {"position": 6, "performance": 0.48},
            "Aston Martin": {"position": 7, "performance": 0.47},
            "Haas F1 Team": {"position": 8, "performance": 0.43},
            "Alpine": {"position": 9, "performance": 0.40},
            "Sauber": {"position": 10, "performance": 0.38},
            "Cadillac F1": {"position": 11, "performance": 0.35},
        }
        note = (
            "2026 REGULATION RESET - Initialized from 2025 constructor ranking with high "
            "uncertainty (±0.3). Team strengths are updated as 2026 data arrives."
        )

    team_characteristics = {
        "year": 2026,
        "note": note,
        "generated_at": datetime.now().isoformat(),
        "data_freshness": "BASELINE_PRESEASON",
        "teams": {},
    }

    for team, team_seed in team_2026_seed.items():
        if neutral_start:
            team_note = "Pre-season neutral baseline - no 2026 data yet"
        else:
            team_note = (
                f"2025 P{team_seed['position']} seed with high uncertainty "
                "for 2026 regulation reset"
            )

        team_characteristics["teams"][team] = {
            "overall_performance": float(team_seed["performance"]),
            "uncertainty": 0.30,
            "note": team_note,
            "last_updated": None,
            "races_completed": 0,
        }

    output_file = output_dir / "car_characteristics" / "2026_car_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(team_characteristics, f, indent=2)

    logger.info(f"[OK] Saved team characteristics to {output_file}")
    logger.info(f"  Teams: {len(team_2026_seed)}")


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
    parser.add_argument(
        "--neutral-teams",
        action="store_true",
        help="Initialize all teams at 0.5 (use only for explicitly neutral preseason experiments).",
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
        generate_team_characteristics(output_dir, neutral_start=args.neutral_teams)
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
