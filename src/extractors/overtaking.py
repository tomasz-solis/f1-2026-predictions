"""
Overtaking Likelihood Extraction

Analyzes last 2 years of race data to determine overtaking likelihood at each track.
Monaco is processional (low overtakes), Bahrain/Monza have high overtakes.
"""

import json
import logging
from pathlib import Path
from typing import Any

import fastf1 as ff1
import numpy as np
import pandas as pd

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def extract_overtakes_from_race(year: int, race_name: str) -> dict[str, Any] | None:
    """Extract overtaking data from a race by counting position changes."""
    try:
        session = ff1.get_session(year, race_name, "R")
        session.load(laps=True, telemetry=False, weather=False)

        if not hasattr(session, "laps") or session.laps is None:
            return None

        laps = session.laps

        # Count position changes per lap
        overtakes = []

        # Get all lap numbers
        lap_numbers = sorted(laps["LapNumber"].unique())

        for lap_num in lap_numbers[1:]:  # Skip lap 1 (start chaos)
            # Get positions at previous lap and current lap
            prev_lap = laps[laps["LapNumber"] == lap_num - 1]
            curr_lap = laps[laps["LapNumber"] == lap_num]

            if len(prev_lap) == 0 or len(curr_lap) == 0:
                continue

            # Exclude pit laps (massive position changes)
            curr_lap_clean = curr_lap[curr_lap["PitOutTime"].isna()]

            # Get drivers present in both laps
            drivers_both = set(prev_lap["Driver"]) & set(curr_lap_clean["Driver"])

            if len(drivers_both) < 5:  # Too few cars (safety car/incident)
                continue

            # Count position changes
            position_changes = 0

            for driver in drivers_both:
                prev_pos = prev_lap[prev_lap["Driver"] == driver]["Position"].iloc[0]
                curr_pos = curr_lap_clean[curr_lap_clean["Driver"] == driver]["Position"].iloc[0]

                if pd.notna(prev_pos) and pd.notna(curr_pos):
                    if curr_pos != prev_pos:
                        position_changes += 1

            overtakes.append({"lap": lap_num, "position_changes": position_changes})

        if not overtakes:
            return None

        # Calculate stats
        total_changes = sum(o["position_changes"] for o in overtakes)
        avg_per_lap = total_changes / len(overtakes) if overtakes else 0

        return {
            "year": year,
            "race": race_name,
            "total_position_changes": total_changes,
            "laps_analyzed": len(overtakes),
            "avg_changes_per_lap": avg_per_lap,
            "max_changes_in_lap": max(o["position_changes"] for o in overtakes),
        }

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.error(
            f"Failed to extract overtaking data from {race_name} ({year}): {e}. Overtaking statistics will be unavailable."
        )
        return None


def calculate_overtaking_likelihood(years: list[int] | None = None) -> dict[str, dict[str, Any]]:
    """Calculate overtaking likelihood for all tracks from specified years."""

    if years is None:
        years = [2024, 2025]

    overtaking_data = {}

    logger.info("Extracting overtaking data from races...")

    for year in years:
        logger.info("Processing %s...", year)

        try:
            schedule = ff1.get_event_schedule(year)
        except (
            ConnectionError,
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            logger.error(f"Failed to get event schedule for {year}: {e}. Skipping this season.")
            logger.warning("Could not get schedule for %s: %s", year, type(e).__name__)
            continue

        for _, event in schedule.iterrows():
            race_name = event["EventName"]

            if "Testing" in str(race_name):
                continue

            stats = extract_overtakes_from_race(year, race_name)

            if stats:
                if race_name not in overtaking_data:
                    overtaking_data[race_name] = []

                overtaking_data[race_name].append(stats)
                logger.info(
                    "%s %s: %.1f changes/lap",
                    year,
                    race_name,
                    stats["avg_changes_per_lap"],
                )
            else:
                logger.info("%s %s: no overtaking data", year, race_name)

    # Aggregate across years
    track_likelihood = {}

    for track, years_data in overtaking_data.items():
        if not years_data:
            continue

        # Average across years
        avg_changes = np.mean([d["avg_changes_per_lap"] for d in years_data])
        total_laps = sum(d["laps_analyzed"] for d in years_data)

        track_likelihood[track] = {
            "avg_changes_per_lap": float(avg_changes),
            "years_analyzed": len(years_data),
            "total_laps_analyzed": int(total_laps),
            "years_data": years_data,
        }

    return track_likelihood


def classify_overtaking_difficulty(avg_changes_per_lap: float) -> tuple[str, float]:
    """Classify track overtaking difficulty from average position changes per lap."""
    if avg_changes_per_lap < 2.5:
        return "very_hard", 0.2
    elif avg_changes_per_lap < 4.0:
        return "hard", 0.4
    elif avg_changes_per_lap < 5.5:
        return "moderate", 0.6
    elif avg_changes_per_lap < 7.0:
        return "easy", 0.8
    else:
        return "very_easy", 1.0


def add_overtaking_to_tracks(
    track_characteristics_path: str | Path,
    overtaking_data: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> None:
    """Add overtaking likelihood to existing track characteristics JSON."""
    # Load existing tracks
    with open(track_characteristics_path) as f:
        data = json.load(f)

    # Add overtaking data
    for track_name, track_data in data["tracks"].items():
        if track_name in overtaking_data:
            ot_data = overtaking_data[track_name]

            difficulty, normalized_score = classify_overtaking_difficulty(
                ot_data["avg_changes_per_lap"]
            )

            track_data["overtaking_avg_changes_per_lap"] = ot_data["avg_changes_per_lap"]
            track_data["overtaking_difficulty"] = difficulty
            track_data["overtaking_likelihood"] = normalized_score
            track_data["overtaking_years_analyzed"] = ot_data["years_analyzed"]
        else:
            # No data - use defaults based on track type
            if track_data.get("is_street_circuit", 0) == 1:
                # Street circuits tend to be harder
                track_data["overtaking_avg_changes_per_lap"] = 3.0
                track_data["overtaking_difficulty"] = "hard"
                track_data["overtaking_likelihood"] = 0.4
            else:
                # Permanent circuits moderate
                track_data["overtaking_avg_changes_per_lap"] = 5.0
                track_data["overtaking_difficulty"] = "moderate"
                track_data["overtaking_likelihood"] = 0.6

            track_data["overtaking_years_analyzed"] = 0

    # Save
    if output_path is None:
        output_path = track_characteristics_path

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Updated overtaking data for %s tracks", len(data["tracks"]))
    logger.info("Saved overtaking data to %s", output_path)
