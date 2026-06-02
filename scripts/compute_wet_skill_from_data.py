"""Compute per-driver wet-weather skill ratings from FastF1 lap-time data.

Methodology:
  1. Load race sessions for specified seasons via FastF1.
  2. Classify each session as dry/wet/mixed from the Rainfall column.
  3. Extract cleaned lap times (exclude pit laps, lap 1, last lap, outliers).
  4. Compute teammate-relative pace per session (driver median minus teammate median).
  5. Compare each driver's relative pace in wet sessions vs dry sessions.
  6. Positive wet effect = driver is faster in wet relative to teammate.
  7. Normalize to [0.40, 0.95] scale, centered on neutral (0.70), with
     exponential recency decay so recent seasons count more.

Usage:
    .venv/bin/python scripts/compute_wet_skill_from_data.py --years 2022,2023,2024,2025
    .venv/bin/python scripts/compute_wet_skill_from_data.py --years 2023,2024,2025 --output data/processed/wet_skill_computed.json
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.ergast", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_REQUEST_DELAY = 0.80
_MIN_LAPS_PER_DRIVER = 5
_MIN_WET_SESSIONS = 2
_MIN_DRY_SESSIONS = 3
_NEUTRAL = 0.70
_ROUNDING = 0.05


def classify_session_weather(session: fastf1.core.Session) -> str:
    """Classify a session as 'dry', 'wet', or 'mixed' from weather data.

    Uses the Rainfall column from FastF1. Sessions with >30% wet timestamps
    are 'wet', 5-30% are 'mixed', below 5% are 'dry'.
    """
    try:
        weather = session.weather_data
    except (AttributeError, RuntimeError):
        return "unknown"

    if weather is None or weather.empty:
        return "unknown"

    if "Rainfall" not in weather.columns:
        return "unknown"

    rainfall = weather["Rainfall"].dropna()
    if rainfall.empty:
        return "unknown"

    wet_fraction = float(rainfall.astype(bool).mean())
    if wet_fraction > 0.30:
        return "wet"
    if wet_fraction > 0.05:
        return "mixed"
    return "dry"


def extract_clean_lap_times(session: fastf1.core.Session) -> dict[str, list[float]]:
    """Extract cleaned lap times per driver from a session.

    Excludes pit in/out laps, lap 1, last lap, and statistical outliers
    beyond 2 standard deviations from each driver's own mean.

    Returns dict of driver abbreviation to list of lap times in seconds.
    """
    try:
        laps = session.laps
    except (AttributeError, RuntimeError):
        return {}

    if laps is None or laps.empty:
        return {}

    required = {"Driver", "LapTime", "LapNumber"}
    if not required.issubset(laps.columns):
        return {}

    mask = laps["LapTime"].notna()
    if "PitOutTime" in laps.columns:
        mask &= laps["PitOutTime"].isna()
    if "PitInTime" in laps.columns:
        mask &= laps["PitInTime"].isna()
    mask &= laps["LapNumber"] > 1
    mask &= laps["LapNumber"] < laps["LapNumber"].max()

    clean = laps[mask].copy()
    if clean.empty:
        return {}

    result: dict[str, list[float]] = {}
    for driver, group in clean.groupby("Driver"):
        times = group["LapTime"].dt.total_seconds().values
        if len(times) < _MIN_LAPS_PER_DRIVER:
            continue
        mean_t, std_t = float(np.mean(times)), float(np.std(times))
        if std_t > 0:
            filtered = times[(times > mean_t - 2 * std_t) & (times < mean_t + 2 * std_t)]
        else:
            filtered = times
        if len(filtered) >= 3:
            result[str(driver)] = list(map(float, filtered))

    return result


def compute_teammate_relative_pace(
    lap_times: dict[str, list[float]],
    session_results: pd.DataFrame,
) -> dict[str, float]:
    """Compute each driver's median pace relative to their teammate.

    Returns dict of driver -> relative_pace_seconds.
    Negative = faster than teammate. Positive = slower.
    """
    if session_results is None or session_results.empty:
        return {}

    # Build team -> drivers mapping from session results
    team_col = "TeamName" if "TeamName" in session_results.columns else "Team"
    if team_col not in session_results.columns or "Abbreviation" not in session_results.columns:
        return {}

    team_drivers: dict[str, list[str]] = defaultdict(list)
    for _, row in session_results.iterrows():
        driver = str(row.get("Abbreviation", "")).strip()
        team = str(row.get(team_col, "")).strip()
        if driver and team:
            team_drivers[team].append(driver)

    result: dict[str, float] = {}
    for _team, drivers in team_drivers.items():
        with_data = [d for d in drivers if d in lap_times]
        if len(with_data) < 2:
            continue
        medians = {d: float(np.median(lap_times[d])) for d in with_data}
        team_mean = float(np.mean(list(medians.values())))
        for d in with_data:
            result[d] = medians[d] - team_mean

    return result


def collect_session_data(years: list[int]) -> pd.DataFrame:
    """Load all race sessions, classify weather, extract teammate-relative pace.

    Returns a DataFrame with columns:
      year, race_name, weather, driver, relative_pace_s
    """
    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    records: list[dict] = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:
            logger.warning("Could not load %d schedule: %s", year, exc)
            continue

        race_names = schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()
        logger.info("Year %d: %d events", year, len(race_names))

        for race_name in race_names:
            time.sleep(_REQUEST_DELAY)

            try:
                session = fastf1.get_session(year, race_name, "R")
                session.load(laps=True, telemetry=False, weather=True)
            except Exception as exc:
                logger.debug("Skipping %d %s R: %s", year, race_name, exc)
                continue

            weather = classify_session_weather(session)
            if weather == "unknown":
                continue

            lap_times = extract_clean_lap_times(session)
            if not lap_times:
                continue

            try:
                results = session.results
            except (AttributeError, RuntimeError):
                continue

            relative_paces = compute_teammate_relative_pace(lap_times, results)
            if not relative_paces:
                continue

            for driver, rel_pace in relative_paces.items():
                records.append(
                    {
                        "year": year,
                        "race_name": race_name,
                        "weather": weather,
                        "driver": driver,
                        "relative_pace_s": rel_pace,
                    }
                )

            logger.info("  %s: weather=%s, %d drivers", race_name, weather, len(relative_paces))

    return pd.DataFrame(records)


def compute_ratings(
    df: pd.DataFrame,
    *,
    recency_half_life_years: float = 2.0,
    current_year: int = 2026,
    neutral: float = _NEUTRAL,
    rounding: float = _ROUNDING,
    min_wet: int = _MIN_WET_SESSIONS,
    min_dry: int = _MIN_DRY_SESSIONS,
) -> dict[str, dict]:
    """Compute wet_skill ratings from collected session data.

    For each driver, compares recency-weighted teammate-relative pace in
    wet/mixed sessions against dry sessions. A positive wet effect means
    the driver is faster relative to their teammate in wet conditions.

    Returns dict of driver -> {wet_skill, wet_sessions, mixed_sessions,
    dry_sessions, raw_wet_effect_s, confidence}.
    """
    if df.empty:
        return {}

    df = df.copy()
    df["years_ago"] = current_year - df["year"]
    df["weight"] = 0.5 ** (df["years_ago"] / recency_half_life_years)

    results: dict[str, dict] = {}

    for driver, group in df.groupby("driver"):
        wet_rows = group[group["weather"] == "wet"]
        mixed_rows = group[group["weather"] == "mixed"]
        dry_rows = group[group["weather"] == "dry"]

        n_wet = len(wet_rows)
        n_mixed = len(mixed_rows)
        n_dry = len(dry_rows)
        # Mixed conditions count as 0.4x of a true wet session for threshold purposes.
        effective_wet = n_wet + (n_mixed * 0.4)

        if effective_wet < min_wet or n_dry < min_dry:
            results[str(driver)] = {
                "wet_skill": neutral,
                "wet_sessions": n_wet,
                "mixed_sessions": n_mixed,
                "dry_sessions": n_dry,
                "raw_wet_effect_s": 0.0,
                "confidence": "insufficient",
            }
            continue

        combined_wet = pd.concat(
            [
                wet_rows,
                mixed_rows.assign(weight=mixed_rows["weight"] * 0.4),
            ]
        )
        if combined_wet.empty:
            results[str(driver)] = {
                "wet_skill": neutral,
                "wet_sessions": n_wet,
                "mixed_sessions": n_mixed,
                "dry_sessions": n_dry,
                "raw_wet_effect_s": 0.0,
                "confidence": "insufficient",
            }
            continue

        wet_mean = float(
            np.average(combined_wet["relative_pace_s"], weights=combined_wet["weight"])
        )
        dry_mean = float(np.average(dry_rows["relative_pace_s"], weights=dry_rows["weight"]))

        # Positive = driver is faster (lower pace) in wet relative to teammate
        raw_effect = float(dry_mean - wet_mean)

        # Uncertainty: standard error of the wet-dry difference
        wet_vals = combined_wet["relative_pace_s"].values
        dry_vals = dry_rows["relative_pace_s"].values
        wet_se = float(np.std(wet_vals) / np.sqrt(len(wet_vals))) if len(wet_vals) > 1 else 0.5
        dry_se = float(np.std(dry_vals) / np.sqrt(len(dry_vals))) if len(dry_vals) > 1 else 0.5
        raw_uncertainty_s = float(np.sqrt(wet_se**2 + dry_se**2))

        if n_wet >= 5:
            confidence = "high"
        elif n_wet >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        if n_wet < 1:
            confidence = "mixed_only"

        results[str(driver)] = {
            "wet_skill": raw_effect,  # will be normalized below
            "wet_sessions": n_wet,
            "mixed_sessions": n_mixed,
            "dry_sessions": n_dry,
            "raw_wet_effect_s": round(raw_effect, 4),
            "raw_uncertainty_s": round(raw_uncertainty_s, 4),
            "confidence": confidence,
        }

    # Normalize raw effects to 0-1 scale centered on neutral
    effects = [r["wet_skill"] for r in results.values() if r["confidence"] != "insufficient"]
    if not effects:
        return results

    effect_std = float(np.std(effects)) if len(effects) > 1 else 1.0
    if effect_std < 1e-6:
        effect_std = 1.0

    for info in results.values():
        if info["confidence"] == "insufficient":
            info["wet_skill_uncertainty"] = 0.0
            continue
        # 1 std of wet effect maps to ~0.10 rating points
        normalized = neutral + (info["wet_skill"] / effect_std) * 0.10
        rounded = round(normalized / rounding) * rounding
        info["wet_skill"] = float(np.clip(rounded, 0.40, 0.95))

        # Propagate uncertainty to skill-space (same scaling factor)
        skill_uncertainty = (info.get("raw_uncertainty_s", 0.0) / effect_std) * 0.10
        info["wet_skill_uncertainty"] = round(float(np.clip(skill_uncertainty, 0.01, 0.15)), 3)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute wet_skill from FastF1 data")
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025")
    parser.add_argument("--output", type=str, default="data/processed/wet_skill_computed.json")
    parser.add_argument("--half-life", type=float, default=2.0)
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    logger.info("Computing wet_skill from years: %s", years)

    df = collect_session_data(years)
    logger.info("Collected %d session-driver records", len(df))

    if not df.empty:
        logger.info("Weather distribution:\n%s", df.groupby("weather").size().to_string())

    ratings = compute_ratings(df, recency_half_life_years=args.half_life)
    logger.info("Computed ratings for %d drivers", len(ratings))

    # Print ranked summary
    by_skill = sorted(ratings.items(), key=lambda x: x[1]["wet_skill"], reverse=True)
    print(f"\n{'Driver':<6} {'Rating':>6} {'Wet':>4} {'Dry':>4} {'Effect':>8} {'Conf'}")
    print()
    for driver, info in by_skill:
        print(
            f"{driver:<6} {info['wet_skill']:>6.2f} {info['wet_sessions']:>4} "
            f"{info['dry_sessions']:>4} {info['raw_wet_effect_s']:>+8.4f} {info['confidence']}"
        )

    # Compare to current priors
    chars_path = Path("data/processed/driver_characteristics.json")
    if chars_path.exists():
        with open(chars_path) as f:
            chars = json.load(f)
        drivers = chars.get("drivers", {})
        print(f"\n{'Driver':<6} {'Computed':>8} {'Prior':>6} {'Delta':>6}")
        print()
        for driver, info in by_skill:
            prior = drivers.get(driver, {}).get("wet_skill")
            if prior is not None:
                delta = info["wet_skill"] - prior
                flag = " ***" if abs(delta) >= 0.10 else ""
                print(f"{driver:<6} {info['wet_skill']:>8.2f} {prior:>6.2f} {delta:>+6.2f}{flag}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "method": "teammate_relative_wet_dry_differential",
        "years": years,
        "recency_half_life_years": args.half_life,
        "computed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "drivers": ratings,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
