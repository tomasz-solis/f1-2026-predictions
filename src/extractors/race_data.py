"""
Shared Race Data Extraction

Unified extraction logic used by both racecraft and DNF risk scripts.
Keeps extraction logic consistent across all metrics.
"""

import logging
import warnings

import fastf1 as ff1
import pandas as pd

# Suppress noise
logging.getLogger("fastf1").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def extract_race_data(year, race_name):
    """Extract race data from FastF1.

    Returns dict with quali/race/gain/DNF/team/status per driver.
    """
    try:
        quali = ff1.get_session(year, race_name, "Q")
        quali.load(laps=False)

        race = ff1.get_session(year, race_name, "R")
        race.load(laps=False)

        results = {}

        for _idx, row in race.results.iterrows():
            driver = row["Abbreviation"]

            # Detect DNFs across FastF1 result schema variants.
            if hasattr(row, "dnf"):
                is_dnf = bool(row.dnf)
            else:
                status = str(row.get("Status", "")).strip().lower()
                is_dnf = any(marker in status for marker in ("dnf", "retired", "disqualified"))

            # Get race position
            if pd.notna(row["Position"]):
                race_pos = int(row["Position"])
            else:
                # No race position = skip this entry
                continue

            # Get quali position (mark if missing)
            quali_row = quali.results[quali.results["Abbreviation"] == driver]
            if len(quali_row) > 0 and pd.notna(quali_row.iloc[0]["Position"]):
                quali_pos = int(quali_row.iloc[0]["Position"])
                has_quali = True
            else:
                quali_pos = None
                has_quali = False

            # Calculate gain (only if we have quali)
            if has_quali and quali_pos is not None:
                gain = quali_pos - race_pos
            else:
                gain = None

            results[driver] = {
                "quali": quali_pos,
                "race": race_pos,
                "gain": gain,
                # Keep both keys for backward compatibility with downstream consumers.
                "dnf": is_dnf,
                "dn": is_dnf,
                "has_quali": has_quali,
                "team": row["TeamName"],
                "status": row.get("Status", "Unknown"),
            }

        return results

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.error(
            f"Failed to extract race data for {race_name} ({year}): {e}. This race data will be unavailable for analysis."
        )
        # Caller handles missing data
        return None


def extract_season(year, verbose=True):
    """Extract all races from a season. Returns {driver: [race_data, ...]}."""
    from collections import defaultdict

    schedule = ff1.get_event_schedule(year)
    races = [
        event["EventName"]
        for idx, event in schedule.iterrows()
        if "testing" not in str(event.get("EventFormat", "")).lower()
    ]

    if verbose:
        logger.info("Extracting %s races from %s", len(races), year)
        logger.info("%s", "=" * 70)

    driver_data = defaultdict(list)

    for race_name in races:
        results = extract_race_data(year, race_name)

        if results:
            if verbose:
                logger.info("  %s: %s drivers", race_name, len(results))

            for driver, data in results.items():
                driver_data[driver].append(data)
        else:
            if verbose:
                logger.info("  %s: failed", race_name)

    if verbose:
        logger.info("Extracted %s races", len(races))
        logger.info("Data for %s drivers", len(driver_data))

    return driver_data


def count_total_dnfs(driver_races):
    """Count total DNF races for a driver. Returns int."""
    return sum(1 for race in driver_races if race.get("dnf", race.get("dn", False)))


def get_valid_races(driver_races):
    """Filter to races with valid qualifying data. Returns races with has_quali=True."""
    return [race for race in driver_races if race.get("has_quali", True)]


def get_dnf_races(driver_races):
    """Get all DNF races. Returns races where dnf=True."""
    return [race for race in driver_races if race.get("dnf", race.get("dn", False))]


def get_clean_races(driver_races):
    """Get finished races (not DNF). Returns races where dnf=False."""
    return [race for race in driver_races if not race.get("dnf", race.get("dn", False))]
