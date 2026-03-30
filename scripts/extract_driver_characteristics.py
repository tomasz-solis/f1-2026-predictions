"""
Driver Characteristics Extraction - Global Teammate Network Ranking

Uses iterative global solver to calculate absolute driver ratings from
relative teammate comparisons. Like Elo/TrueSkill for F1.

Key improvements:
- No capping/manual overrides
- Solves teammate network globally (HAM vs RUS → both elite)
- Handles mid-season swaps
- Recency and confidence weighting
- Rookie penalties

USAGE:
    python scripts/extract_driver_characteristics.py --years 2023,2024,2025
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import fastf1 as ff1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.fastf1_resilience import FastF1ResiliencePolicy, call_with_resilience  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _logger_name in (
    "fastf1",
    "fastf1.api",
    "fastf1.core",
    "fastf1.ergast",
    "requests_cache",
    "_api",
    "core",
    "req",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

_REQUEST_DELAY_SECONDS = 0.80
_FASTF1_POLICY = FastF1ResiliencePolicy(
    max_attempts=8,
    timeout_budget_seconds=180.0,
    initial_backoff_seconds=1.5,
    max_backoff_seconds=20.0,
    backoff_multiplier=2.0,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_cooldown_seconds=60.0,
)
_RACE_NAME_CACHE: dict[int, list[str]] = {}
_DNF_RATE_FLOOR = 0.03
_DEFAULT_BAYESIAN_SIGMA = 2.5

DRIVER_FULL_NAMES = {
    "VER": "Max Verstappen",
    "NOR": "Lando Norris",
    "LEC": "Charles Leclerc",
    "RUS": "George Russell",
    "HAM": "Lewis Hamilton",
    "PIA": "Oscar Piastri",
    "SAI": "Carlos Sainz",
    "ALO": "Fernando Alonso",
    "GAS": "Pierre Gasly",
    "OCO": "Esteban Ocon",
    "STR": "Lance Stroll",
    "ALB": "Alexander Albon",
    "HUL": "Nico Hulkenberg",
    "TSU": "Yuki Tsunoda",
    "RIC": "Daniel Ricciardo",
    "BEA": "Oliver Bearman",
    "ANT": "Kimi Antonelli",
    "PER": "Sergio Perez",
    "LAW": "Liam Lawson",
    "BOT": "Valtteri Bottas",
    "ZHO": "Guanyu Zhou",
    "MAG": "Kevin Magnussen",
    "HAD": "Isack Hadjar",
    "BOR": "Gabriel Bortoleto",
    "COL": "Franco Colapinto",
    "DOO": "Jack Doohan",
    "SAR": "Logan Sargeant",
    "DEV": "Nyck de Vries",
    "LIN": "Arvid Lindblad",
}


def _fastf1_call(operation_name: str, fn, *, labels: dict | None = None):
    """Run FastF1 network calls with retry/backoff plus small pacing delay."""
    result = call_with_resilience(
        operation_name,
        fn,
        labels=labels,
        policy=_FASTF1_POLICY,
    )
    if _REQUEST_DELAY_SECONDS > 0:
        time.sleep(_REQUEST_DELAY_SECONDS)
    return result


def _read_cgroup_memory_limit_mb() -> int | None:
    """Read cgroup memory limit in MB when available."""
    candidates = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path) as handle:
                raw = handle.read().strip().lower()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit_bytes = int(raw)
        except ValueError:
            continue
        # Ignore unrealistic sentinel values that represent "unlimited".
        if limit_bytes <= 0 or limit_bytes >= 2**60:
            continue
        return max(1, int(limit_bytes / (1024 * 1024)))
    return None


def _apply_rookie_penalty(base_rating: float, experience_data: dict) -> float:
    """Apply a lighter rookie penalty once the sample size becomes meaningful."""
    adjusted_rating = float(base_rating)
    if experience_data.get("tier") != "rookie":
        return adjusted_rating

    total_races = int(experience_data.get("total_races", 0) or 0)
    if total_races >= 20:
        return adjusted_rating * 0.96
    if total_races >= 10:
        return adjusted_rating * 0.93
    return adjusted_rating * 0.90


def _load_current_lineups(lineup_file: Path) -> dict[str, list[str]]:
    """Load current lineup mapping from disk when available."""
    if not lineup_file.exists():
        return {}

    try:
        with open(lineup_file) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", lineup_file, exc)
        return {}

    current_lineups = payload.get("current_lineups", {})
    if not isinstance(current_lineups, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for team_name, raw_drivers in current_lineups.items():
        if not isinstance(raw_drivers, list):
            continue
        normalized[str(team_name)] = [
            str(driver).strip().upper() for driver in raw_drivers if driver
        ]
    return normalized


def _resolve_bayesian_seed_grid_size(
    final_ratings: dict[str, dict],
    current_lineups: dict[str, list[str]],
) -> int:
    """Pick a seed grid size that matches the active season, not the archive size."""
    lineup_grid_size = len({driver for drivers in current_lineups.values() for driver in drivers})
    if lineup_grid_size >= 2:
        return lineup_grid_size
    return max(2, min(len(final_ratings), 22))


def _seed_initial_bayesian_state(
    final_ratings: dict[str, dict],
    *,
    grid_size: int,
) -> None:
    """Seed Bayesian state so file-only loads still expose an in-season prior."""
    for _driver_code, entry in final_ratings.items():
        skill = float(entry["racecraft"]["skill_score"])
        rating_mu = 1.0 + (skill * max(grid_size - 1, 1))
        entry["bayesian"] = {
            "rating_mu": round(rating_mu, 3),
            "rating_sigma": _DEFAULT_BAYESIAN_SIGMA,
            "normalized_skill_score": round(skill, 3),
            "sessions_observed": 0,
            "seeded_from": "extraction_prior",
        }


def _absolute_finish_floor(
    *,
    driver_code: str,
    absolute_finish_baselines: dict[str, float],
    experience_data: dict,
) -> float | None:
    """Return a conservative floor for rookies with enough real race mileage.

    The teammate network can underrate a rookie paired with an elite benchmark.
    Once we have close to a full season of starts, field results are strong
    enough to stop the rating from collapsing to the absolute backmarker floor.
    """
    if experience_data.get("tier") != "rookie":
        return None
    if int(experience_data.get("total_races", 0) or 0) < 20:
        return None
    baseline = absolute_finish_baselines.get(driver_code)
    if baseline is None:
        return None
    return float(np.clip(baseline * 0.75, 0.35, 0.65))


def _is_render_web_instance() -> bool:
    """Return True when running inside Render web service context."""
    if not os.getenv("RENDER"):
        return False
    service_type = str(os.getenv("RENDER_SERVICE_TYPE", "")).strip().lower()
    if service_type:
        return service_type == "web"
    # Fallback for environments where service type is not exposed.
    return bool(os.getenv("PORT"))


def load_driver_debuts(csv_path: str = "data/driver_debuts.csv") -> dict[str, int]:
    """Load driver F1 debut years from CSV."""
    debuts = {}

    # Name to abbreviation mapping
    name_to_abbr = {
        "Fernando Alonso": "ALO",
        "Lewis Hamilton": "HAM",
        "Nico Hülkenberg": "HUL",
        "Sergio Pérez": "PER",
        "Daniel Ricciardo": "RIC",
        "Valtteri Bottas": "BOT",
        "Kevin Magnussen": "MAG",
        "Max Verstappen": "VER",
        "Carlos Sainz": "SAI",
        "Esteban Ocon": "OCO",
        "Pierre Gasly": "GAS",
        "Lance Stroll": "STR",
        "Charles Leclerc": "LEC",
        "Alexander Albon": "ALB",
        "Lando Norris": "NOR",
        "George Russell": "RUS",
        "Yuki Tsunoda": "TSU",
        "Zhou Guanyu": "ZHO",
        "Nyck de Vries": "DEV",
        "Oscar Piastri": "PIA",
        "Logan Sargeant": "SAR",
        "Franco Colapinto": "COL",
        "Oliver Bearman": "BEA",
        "Isack Hadjar": "HAD",
        "Andrea Kimi Antonelli": "ANT",
        "Gabriel Bortoleto": "BOR",
        "Jack Doohan": "DOO",
        "Arvid Lindblad": "LIN",
        "Liam Lawson": "LAW",
    }

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                driver_name = row["Driver"]
                debut_year = int(row["First F1 season"])

                if driver_name in name_to_abbr:
                    abbr = name_to_abbr[driver_name]
                    debuts[abbr] = debut_year

        logger.info(f"Loaded {len(debuts)} driver debuts from CSV")
    except FileNotFoundError:
        logger.warning(f"Driver debuts CSV not found at {csv_path}, using fallback")

    return debuts


def _iter_non_testing_race_names(year: int) -> list[str]:
    """Return race names for a season, excluding testing events."""
    cached = _RACE_NAME_CACHE.get(year)
    if cached is not None:
        return cached

    schedule = _fastf1_call(
        "extract_driver_characteristics_get_event_schedule",
        lambda: ff1.get_event_schedule(year),
        labels={"year": year},
    )
    races = schedule[schedule["EventFormat"] != "testing"]
    race_names = []
    for _, event in races.iterrows():
        race_name = event["EventName"]
        if race_name:
            race_names.append(race_name)
    _RACE_NAME_CACHE[year] = race_names
    return race_names


def _load_completed_race_session(year: int, race_name: str):
    """Load race session only when race is completed (not future scheduled)."""
    session = _fastf1_call(
        "extract_driver_characteristics_get_session",
        lambda: ff1.get_session(year, race_name, "R"),
        labels={"year": year, "race_name": race_name, "session_name": "R"},
    )

    race_date = session.date
    if pd.isna(race_date):
        return None
    if not hasattr(race_date, "tz") or race_date.tz is None:
        race_date = race_date.tz_localize("UTC")
    if race_date > pd.Timestamp.now(tz="UTC"):
        return None

    return session


def calculate_driver_pace_gap(driver_laps, teammate_laps, session_type="R") -> float:
    """
    Calculate pace gap to teammate (%).

    Returns: % gap where negative = faster than teammate
    """
    d_clean = driver_laps.pick_accurate().pick_quicklaps()
    t_clean = teammate_laps.pick_accurate().pick_quicklaps()

    if d_clean.empty or t_clean.empty or len(d_clean) < 3 or len(t_clean) < 3:
        return None

    if session_type == "Q":
        d_time = d_clean["LapTime"].min().total_seconds()
        t_time = t_clean["LapTime"].min().total_seconds()
    else:
        d_time = d_clean["LapTime"].dt.total_seconds().median()
        t_time = t_clean["LapTime"].dt.total_seconds().median()

    if np.isnan(d_time) or np.isnan(t_time):
        return None

    gap_pct = ((d_time - t_time) / t_time) * 100.0
    return gap_pct


def _normalize_finish_position(raw_position) -> int | None:
    """Return integer finish position when valid; otherwise None."""
    if pd.isna(raw_position):
        return None
    try:
        position = int(raw_position)
    except (TypeError, ValueError):
        return None
    if position <= 0:
        return None
    return position


def _build_race_summary_rows(results: pd.DataFrame) -> list[dict]:
    """Convert FastF1 results DataFrame to compact serializable rows."""
    rows: list[dict] = []
    for _, row in results.iterrows():
        driver = str(row.get("Abbreviation", "")).strip().upper()
        if not driver:
            continue
        status = str(row.get("Status", "")).strip().lower()
        team = str(row.get("TeamName", "")).strip()
        rows.append(
            {
                "driver": driver,
                "team": team,
                "status": status,
                "position": _normalize_finish_position(row.get("Position")),
            }
        )
    return rows


def extract_teammate_comparisons(years: list[int]) -> tuple[list[dict], list[dict]]:
    """
    Extract teammate pace comparisons and compact race summaries.

    Returns list of comparisons with confidence and recency weighting.
    """
    logger.info(f"Extracting teammate comparisons from {years}...")

    comparisons = []
    race_summaries: list[dict] = []

    for year in years:
        # Recency weight: 2025=1.0, 2024=0.8, 2023=0.6
        year_weight = 1.0 - (max(years) - year) * 0.2
        year_weight = max(0.4, year_weight)  # Min weight 0.4

        logger.info(f"Processing {year} season (weight={year_weight:.1f})...")

        try:
            race_names = _iter_non_testing_race_names(year)
        except Exception as e:
            logger.error(f"Failed to load {year} schedule: {e}")
            continue

        for race_index, race_name in enumerate(race_names, start=1):
            session = None
            laps = None
            results = None
            try:
                session = _load_completed_race_session(year, race_name)
                if session is None:
                    continue

                logger.info(f"  {race_name}...")
                _fastf1_call(
                    "extract_driver_characteristics_load_laps",
                    lambda session=session: session.load(
                        laps=True,
                        telemetry=False,
                        weather=False,
                        messages=False,
                    ),
                    labels={"year": year, "race_name": race_name},
                )

                laps = session.laps
                results = session.results
                summary_rows = _build_race_summary_rows(results)
                if summary_rows:
                    race_summaries.append(
                        {
                            "year": int(year),
                            "race": race_name,
                            "race_index": int(race_index),
                            "rows": summary_rows,
                        }
                    )

                # For each team, compare teammates
                for team in laps["Team"].unique():
                    if pd.isna(team):
                        continue

                    team_drivers = laps[laps["Team"] == team]["Driver"].unique()
                    if len(team_drivers) != 2:
                        continue

                    d1, d2 = team_drivers[0], team_drivers[1]
                    laps_d1 = laps.pick_drivers(d1)
                    laps_d2 = laps.pick_drivers(d2)

                    # Calculate pace gap
                    gap = calculate_driver_pace_gap(laps_d1, laps_d2, "R")
                    if gap is None:
                        continue

                    # Get driver abbreviations
                    try:
                        d1_code = results.loc[results["Abbreviation"] == d1].iloc[0]["Abbreviation"]
                        d2_code = results.loc[results["Abbreviation"] == d2].iloc[0]["Abbreviation"]
                    except (KeyError, ValueError, TypeError):
                        continue

                    # Sample size confidence (more laps = higher confidence)
                    sample_size = min(len(laps_d1), len(laps_d2))
                    confidence = min(1.0, sample_size / 30.0)  # 30+ laps = full confidence

                    # Store comparison (A vs B)
                    comparisons.append(
                        {
                            "driver_a": d1_code,
                            "driver_b": d2_code,
                            "gap_pct": gap,  # Positive = A slower than B
                            "year": year,
                            "race": race_name,
                            "confidence": confidence,
                            "recency_weight": year_weight,
                            "weight": confidence * year_weight,
                        }
                    )

            except Exception as e:
                logger.debug(f"  Failed: {e}")
                continue
            finally:
                # Release large objects between sessions to keep peak memory lower.
                laps = None
                results = None
                session = None
                gc.collect()

    logger.info(f"Extracted {len(comparisons)} teammate comparisons")
    return comparisons, race_summaries


def solve_global_ratings(comparisons: list[dict], iterations=15) -> dict[str, float]:
    """
    Solve for absolute driver ratings using iterative global optimization.

    Similar to Elo/TrueSkill - all comparisons constrain the solution space.
    """
    logger.info("Solving global driver ratings...")
    if not comparisons:
        logger.warning("No teammate comparisons available; cannot solve global ratings.")
        return {}

    # Get all unique drivers
    drivers = set()
    for comp in comparisons:
        drivers.add(comp["driver_a"])
        drivers.add(comp["driver_b"])

    # Initialize all drivers at 0.70 (average F1 driver)
    ratings = {driver: 0.70 for driver in drivers}

    # Iterative solver
    learning_rate = 0.15  # How fast to adjust ratings

    for iteration in range(iterations):
        adjustments = {driver: 0.0 for driver in drivers}
        total_weight = {driver: 0.0 for driver in drivers}

        for comp in comparisons:
            a, b = comp["driver_a"], comp["driver_b"]
            gap = comp["gap_pct"]  # % gap
            weight = comp["weight"]

            # Current expected gap based on ratings
            # If A=0.80 and B=0.70, we expect A to be faster (negative gap)
            # Rating difference of 0.10 should correspond to ~1% pace advantage
            expected_gap_pct = (ratings[b] - ratings[a]) * 10.0  # 0.1 rating = 1% pace

            # Actual vs expected
            error = gap - expected_gap_pct

            # Adjust ratings to reduce error
            # If A is slower than expected, reduce A's rating
            # If A is faster than expected, increase A's rating
            adjustment = error * 0.01  # Convert % to rating adjustment

            adjustments[a] -= adjustment * weight
            adjustments[b] += adjustment * weight

            total_weight[a] += weight
            total_weight[b] += weight

        # Apply weighted adjustments
        for driver in drivers:
            if total_weight[driver] > 0:
                avg_adjustment = adjustments[driver] / total_weight[driver]
                ratings[driver] += avg_adjustment * learning_rate

        # Log progress
        if iteration % 5 == 0:
            avg_rating = np.mean(list(ratings.values()))
            std_rating = np.std(list(ratings.values()))
            logger.info(f"  Iteration {iteration}: avg={avg_rating:.3f}, std={std_rating:.3f}")

    # Normalize ratings to 0.35-0.95 range (WIDER SPREAD!)
    # Best driver → 0.95, Average → 0.65, Worst → 0.35
    min_rating = min(ratings.values())
    max_rating = max(ratings.values())

    for driver in ratings:
        # Scale to 0-1, then to 0.35-0.95
        normalized = (ratings[driver] - min_rating) / (max_rating - min_rating)
        ratings[driver] = 0.35 + (normalized * 0.60)

    logger.info(f"Solved ratings for {len(drivers)} drivers")
    return ratings


def calculate_racecraft_scores(
    years: list[int], ratings: dict[str, float], race_summaries: list[dict]
) -> dict[str, float]:
    """Calculate racecraft adjustment based on finish position versus pace-expected position."""
    logger.info("Calculating racecraft adjustments...")

    racecraft_scores = defaultdict(list)
    valid_years = set(years)

    for summary in race_summaries:
        year = int(summary.get("year", 0))
        if year not in valid_years:
            continue

        expected_order = []
        for row in summary.get("rows", []):
            driver = str(row.get("driver", "")).strip().upper()
            actual_pos = row.get("position")
            if driver in ratings and isinstance(actual_pos, int) and actual_pos <= 20:
                expected_order.append((driver, ratings[driver], actual_pos))

        expected_order.sort(key=lambda x: x[1], reverse=True)

        # Compare expected vs actual
        for expected_pos, (driver, _rating, actual_pos) in enumerate(expected_order, 1):
            # Positive = beat expectations (good racecraft)
            racecraft_gain = expected_pos - actual_pos
            racecraft_scores[driver].append(racecraft_gain)

    # Average racecraft scores
    racecraft_ratings = {}
    for driver, scores in racecraft_scores.items():
        avg_gain = np.mean(scores) if scores else 0.0
        # +1 position = +0.02 rating (max ±0.05)
        racecraft_ratings[driver] = np.clip(avg_gain * 0.02, -0.05, 0.05)

    logger.info(f"Calculated racecraft for {len(racecraft_ratings)} drivers")
    return racecraft_ratings


def calculate_experience_and_consistency(
    years: list[int], driver_debuts: dict[str, int], race_summaries: list[dict]
) -> dict:
    """
    Calculate experience tiers, total races, and DNF rates.
    """
    logger.info("Calculating experience and consistency...")

    driver_stats = defaultdict(
        lambda: {
            "seasons": set(),
            "total_races": 0,
            "dnf_count": 0,
            "crash_count": 0,
        }
    )

    valid_years = set(years)
    non_finish_markers = (
        "dnf",
        "retired",
        "disqualified",
        "not classified",
        "did not finish",
        "accident",
        "collision",
        "crash",
        "damage",
        "spun",
        "engine",
        "gearbox",
        "hydraulic",
        "brake",
        "electrical",
        "power unit",
        "fuel pressure",
        "transmission",
    )

    for summary in race_summaries:
        year = int(summary.get("year", 0))
        if year not in valid_years:
            continue

        for row in summary.get("rows", []):
            driver = str(row.get("driver", "")).strip().upper()
            if not driver:
                continue
            status = str(row.get("status", "")).lower()

            driver_stats[driver]["seasons"].add(year)
            driver_stats[driver]["total_races"] += 1

            # Count all non-finish outcomes for race-level reliability risk.
            # Keep crash_count separate for optional driver-error diagnostics.
            finish_markers = ("finished", "classified")
            is_explicit_finish = (
                status.startswith("+")
                or " lap" in status
                or any(marker in status for marker in finish_markers)
            )
            is_non_finish = (not is_explicit_finish) and any(
                marker in status for marker in non_finish_markers
            )
            if is_non_finish:
                driver_stats[driver]["dnf_count"] += 1
                if any(
                    marker in status
                    for marker in ("accident", "collision", "crash", "damage", "spun")
                ):
                    driver_stats[driver]["crash_count"] += 1

    # Process into output format
    output = {}
    current_year = max(years)

    for driver, stats in driver_stats.items():
        total_races = stats["total_races"]

        if total_races < 5:
            continue

        # Calculate REAL F1 experience from debut year
        if driver in driver_debuts:
            debut_year = driver_debuts[driver]
            years_of_experience = current_year - debut_year
        else:
            # Fallback: count seasons in our data
            years_of_experience = len(stats["seasons"])
            logger.warning(f"{driver}: No debut year found, using {years_of_experience} seasons")

        # Experience tier (based on ACTUAL F1 career, not just our data window)
        if years_of_experience >= 10:
            tier = "veteran"
        elif years_of_experience >= 5:
            tier = "established"
        elif years_of_experience >= 2:
            tier = "developing"
        else:
            tier = "rookie"

        # DNF rate (all non-finish outcomes)
        dnf_rate = max(stats["dnf_count"] / total_races, _DNF_RATE_FLOOR)

        output[driver] = {
            "years_of_experience": years_of_experience,
            "debut_year": driver_debuts.get(driver, current_year - years_of_experience),
            "total_races": total_races,
            "tier": tier,
            "dnf_rate": dnf_rate,
        }

    logger.info(f"Processed experience for {len(output)} drivers")
    return output


def calculate_absolute_finish_baselines(
    years: list[int],
    race_summaries: list[dict],
) -> dict[str, float]:
    """Translate median race finish into a cautious field-relative pace floor."""
    finish_positions_by_driver: dict[str, list[int]] = defaultdict(list)
    valid_years = set(years)

    for summary in race_summaries:
        year = int(summary.get("year", 0))
        if year not in valid_years:
            continue
        for row in summary.get("rows", []):
            driver = str(row.get("driver", "")).strip().upper()
            position = row.get("position")
            if not driver or not isinstance(position, int):
                continue
            finish_positions_by_driver[driver].append(int(position))

    baselines: dict[str, float] = {}
    for driver_code, positions in finish_positions_by_driver.items():
        if len(positions) < 10:
            continue
        median_finish = float(np.median(positions))
        field_percentile = 1.0 - ((median_finish - 1.0) / 19.0)
        baselines[driver_code] = float(np.clip(0.35 + (field_percentile * 0.50), 0.35, 0.85))

    return baselines


def calculate_championship_overperformance(
    years: list[int],
    pace_ratings: dict[str, float],
    race_summaries: list[dict],
) -> dict[str, float]:
    """Estimate overperformance versus car baseline using latest race per season."""
    logger.info("Calculating championship overperformance bonuses...")

    latest_summary_by_year: dict[int, dict] = {}
    for summary in race_summaries:
        year = int(summary.get("year", 0))
        if year not in years:
            continue
        race_index = int(summary.get("race_index", 0))
        previous = latest_summary_by_year.get(year)
        previous_index = int(previous.get("race_index", 0)) if previous else -1
        if race_index >= previous_index:
            latest_summary_by_year[year] = summary

    championship_adjustments: dict[str, list[float]] = {}
    for year in years:
        summary = latest_summary_by_year.get(year)
        if summary is None:
            continue

        team_points = defaultdict(list)
        for row in summary.get("rows", []):
            driver = str(row.get("driver", "")).strip().upper()
            team = str(row.get("team", "")).strip()
            position = row.get("position")
            if not driver or not team or driver not in pace_ratings:
                continue
            if not isinstance(position, int):
                continue
            team_points[team].append(position)

        team_expected = {
            team: float(np.mean(positions)) for team, positions in team_points.items() if positions
        }

        for row in summary.get("rows", []):
            driver = str(row.get("driver", "")).strip().upper()
            team = str(row.get("team", "")).strip()
            position = row.get("position")
            if (
                driver not in pace_ratings
                or team not in team_expected
                or not isinstance(position, int)
            ):
                continue

            expected_pos = team_expected[team]
            overperformance = expected_pos - position  # Positive = beat expectations
            championship_adjustments.setdefault(driver, []).append(overperformance)

    championship_bonuses: dict[str, float] = {}
    for driver, overperfs in championship_adjustments.items():
        avg_overperf = float(np.mean(overperfs))
        # +1 position vs team = +0.03 rating (max ±0.10)
        championship_bonuses[driver] = float(np.clip(avg_overperf * 0.03, -0.10, 0.10))
        if abs(championship_bonuses[driver]) > 0.05:
            logger.info(f"  {driver}: {championship_bonuses[driver]:+.3f} (overperformed car)")

    return championship_bonuses


def main():
    global _REQUEST_DELAY_SECONDS
    global _FASTF1_POLICY

    parser = argparse.ArgumentParser(description="Extract driver characteristics (fixed)")
    parser.add_argument("--years", type=str, default="2024,2025", help="Comma-separated years")
    parser.add_argument("--output", type=str, default="data/processed/driver_characteristics.json")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.80,
        help="Seconds to sleep after each FastF1 network call",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Max retries per FastF1 request",
    )
    parser.add_argument(
        "--timeout-budget-seconds",
        type=float,
        default=180.0,
        help="Total retry timeout budget per FastF1 request",
    )
    parser.add_argument(
        "--allow-web-instance-run",
        action="store_true",
        help=(
            "Allow extractor execution on Render web instances. "
            "Use only when you understand memory-risk tradeoffs."
        ),
    )

    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    _REQUEST_DELAY_SECONDS = max(0.0, float(args.request_delay))
    max_attempts = max(1, int(args.max_attempts))
    timeout_budget_seconds = max(5.0, float(args.timeout_budget_seconds))
    memory_limit_mb = _read_cgroup_memory_limit_mb()
    if (
        _is_render_web_instance()
        and not args.allow_web_instance_run
        and memory_limit_mb is not None
        and memory_limit_mb <= 768
    ):
        raise RuntimeError(
            "Refusing to run driver extraction on a memory-constrained Render web instance "
            f"({memory_limit_mb}MB limit). "
            "Run this command from a dedicated background worker or local machine. "
            "If you still want to force it, pass --allow-web-instance-run."
        )

    _FASTF1_POLICY = FastF1ResiliencePolicy(
        max_attempts=max_attempts,
        timeout_budget_seconds=timeout_budget_seconds,
        initial_backoff_seconds=max(1.0, _REQUEST_DELAY_SECONDS * 1.5),
        max_backoff_seconds=20.0,
        backoff_multiplier=2.0,
        circuit_breaker_failure_threshold=max(3, min(max_attempts, 5)),
        circuit_breaker_cooldown_seconds=60.0,
    )

    # Setup cache
    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache(str(cache_dir))

    logger.info("=" * 60)
    logger.info("Fixed Driver Characteristics Extraction")
    logger.info("=" * 60)
    logger.info("")

    # Step 0: Load driver debuts
    driver_debuts = load_driver_debuts()

    # Step 1: Extract teammate comparisons
    comparisons, race_summaries = extract_teammate_comparisons(years)
    if not comparisons:
        raise RuntimeError(
            "No completed race comparisons were extracted. "
            "Verify FastF1 data availability and the selected --years range."
        )

    # Step 2: Solve global ratings
    pace_ratings = solve_global_ratings(comparisons, iterations=15)

    # Step 3: Calculate racecraft adjustments
    racecraft_adjustments = calculate_racecraft_scores(years, pace_ratings, race_summaries)

    # Step 4: Calculate experience and consistency
    experience_data = calculate_experience_and_consistency(years, driver_debuts, race_summaries)

    # Step 5: Calculate championship overperformance (car vs driver finish)
    # This rewards drivers who overdeliver in bad cars (ALO, HAM in 2024)
    championship_bonuses = calculate_championship_overperformance(
        years,
        pace_ratings,
        race_summaries,
    )
    absolute_finish_baselines = calculate_absolute_finish_baselines(years, race_summaries)
    current_lineups = _load_current_lineups(Path("data/current_lineups.json"))

    # Step 6: Combine into final ratings
    final_ratings = {}

    for driver in pace_ratings:
        if driver not in experience_data:
            continue

        base_rating = pace_ratings[driver]
        racecraft_bonus = racecraft_adjustments.get(driver, 0.0)
        championship_bonus = championship_bonuses.get(driver, 0.0)
        exp_data = experience_data[driver]

        base_rating = _apply_rookie_penalty(base_rating, exp_data)
        finish_floor = _absolute_finish_floor(
            driver_code=driver,
            absolute_finish_baselines=absolute_finish_baselines,
            experience_data=exp_data,
        )
        if finish_floor is not None:
            base_rating = max(base_rating, finish_floor)

        # Separate dimensions so downstream simulation can distinguish pace,
        # general race execution, and passing skill.
        race_pace_score = np.clip(base_rating + championship_bonus, 0.10, 0.99)
        general_skill = np.clip(base_rating + racecraft_bonus + championship_bonus, 0.10, 0.99)
        overtaking_score = np.clip(
            base_rating + (racecraft_bonus * 1.5),
            0.10,
            0.99,
        )

        final_ratings[driver] = {
            "name": DRIVER_FULL_NAMES.get(driver, driver),
            "pace": {
                "quali_pace": round(base_rating, 3),
                "race_pace": round(race_pace_score, 3),
            },
            "racecraft": {
                "skill_score": round(general_skill, 3),
                "overtaking_skill": round(overtaking_score, 3),
            },
            "experience": {
                "years_of_experience": exp_data["years_of_experience"],
                "debut_year": exp_data["debut_year"],
                "total_races": exp_data["total_races"],
                "tier": exp_data["tier"],
            },
            "dnf_risk": {
                "dnf_rate": round(exp_data["dnf_rate"], 3),
            },
        }

    # Step 7: Fill missing lineup drivers from current lineups with team-based priors.
    for team_name, team_drivers in current_lineups.items():
        for driver_code in team_drivers:
            if driver_code in final_ratings:
                continue

            teammate_ratings = [
                final_ratings[teammate]["racecraft"]["skill_score"]
                for teammate in team_drivers
                if teammate in final_ratings and teammate != driver_code
            ]
            if teammate_ratings:
                prior_rating = float(np.clip(np.mean(teammate_ratings) - 0.08, 0.10, 0.90))
            else:
                prior_rating = 0.40

            logger.info(
                "  %s: no race data, using team-based prior (%s, base=%.3f)",
                driver_code,
                team_name,
                prior_rating,
            )
            final_ratings[driver_code] = {
                "name": DRIVER_FULL_NAMES.get(driver_code, driver_code),
                "pace": {
                    "quali_pace": round(prior_rating * 0.95, 3),
                    "race_pace": round(prior_rating, 3),
                },
                "racecraft": {
                    "skill_score": round(prior_rating, 3),
                    "overtaking_skill": round(prior_rating, 3),
                },
                "experience": {
                    "years_of_experience": 0,
                    "debut_year": 2026,
                    "total_races": 0,
                    "tier": "rookie",
                },
                "dnf_risk": {
                    "dnf_rate": round(_DNF_RATE_FLOOR, 3),
                },
                "prior_source": "team_based_prior",
            }

    # Step 8: Seed initial Bayesian state so file-based fallbacks can use it.
    _seed_initial_bayesian_state(
        final_ratings,
        grid_size=_resolve_bayesian_seed_grid_size(final_ratings, current_lineups),
    )

    # Save
    extraction_timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    output = {
        "extraction_date": extraction_timestamp,
        "years": years,
        "method": "global_teammate_network_ranking",
        "last_updated": extraction_timestamp,
        "bayesian_last_updated_year": pd.Timestamp.now(tz="UTC").year,
        "drivers": final_ratings,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[OK] Extracted {len(final_ratings)} drivers")
    logger.info(f" Saved to: {output_path}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Sample ratings:")

    # Show top drivers
    sorted_drivers = sorted(
        final_ratings.items(),
        key=lambda x: x[1]["racecraft"]["skill_score"],
        reverse=True,
    )
    for driver, data in sorted_drivers[:10]:
        logger.info(f"  {driver}: {data['racecraft']['skill_score']:.3f}")


if __name__ == "__main__":
    main()
