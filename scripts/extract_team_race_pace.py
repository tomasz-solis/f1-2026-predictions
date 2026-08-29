"""Measure per-team race pace from cached FastF1 lap data.

``team_strength`` (see ``_get_current_season_observations`` in
``src/predictors/baseline/data_mixin.py``) is reconstructed from classified race
results, which conflate pace with reliability, strategy and luck. This script
measures pace directly from green-flag lap times instead, and writes it to
``data/processed/team_race_pace/<year>_team_race_pace.json`` for
``_resolve_team_pace_delta_seconds`` in ``src/utils/lap_by_lap_simulator.py`` to
prefer over the results-derived value.

Method, per race:
    - keep laps with ``TrackStatus == "1"`` (green flag), a valid ``LapTime``, and
      neither ``PitInTime`` nor ``PitOutTime`` set
    - per driver, take the median of those lap times; drop drivers with fewer
      than 10 qualifying laps
    - per team, take the median across its drivers
    - normalise: subtract the fastest team's value that race, giving a per-race
      gap in seconds (0.0 for the fastest team)

Each team's per-race gaps are then averaged across every race it appears in.

Usage:
    uv run python scripts/extract_team_race_pace.py --year 2026
    uv run python scripts/extract_team_race_pace.py --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import fastf1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.session_detector import SessionDetector  # noqa: E402
from src.utils.team_mapping import map_team_to_characteristics  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.ergast", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Loading a race session can fail for reasons unrelated to this script's logic
# (missing cache entry, malformed upstream payload); skip that race rather than
# aborting the whole run.
_LOAD_ERRORS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_MIN_LAPS_PER_DRIVER = 10
_ROUND_PRECISION = 4


def _enable_fastf1_cache() -> None:
    """Enable the repo's shared FastF1 cache so extraction reads cached lap data."""
    cache_dir = PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def measure_race(year: int, race_name: str) -> dict[str, float] | None:
    """Measure one race's per-team pace gap in seconds, normalised to the fastest team.

    Returns a mapping of characteristics team name to gap (0.0 for the fastest
    team that race), or None if the session/laps could not be loaded or no
    driver had enough green-flag laps to measure.
    """
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(laps=True, telemetry=False, weather=False)
    except _LOAD_ERRORS as exc:
        logger.info("Skipping %s: failed to load session (%s)", race_name, exc)
        return None

    laps = getattr(session, "laps", None)
    if laps is None or laps.empty:
        return None

    green_laps = laps[
        (laps["TrackStatus"] == "1")
        & laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ]
    if green_laps.empty:
        return None

    team_driver_medians: dict[str, list[float]] = defaultdict(list)
    for _driver, driver_laps in green_laps.groupby("Driver"):
        if len(driver_laps) < _MIN_LAPS_PER_DRIVER:
            continue
        raw_team = str(driver_laps["Team"].iloc[0])
        team = map_team_to_characteristics(raw_team) or raw_team
        median_s = driver_laps["LapTime"].median().total_seconds()
        team_driver_medians[team].append(median_s)

    if not team_driver_medians:
        return None

    team_medians = {team: statistics.median(values) for team, values in team_driver_medians.items()}
    fastest = min(team_medians.values())
    return {team: value - fastest for team, value in team_medians.items()}


def collect_measurements(year: int) -> list[tuple[str, float]]:
    """Measure every completed race of `year` with loadable lap data."""
    _enable_fastf1_cache()
    schedule = fastf1.get_event_schedule(year)
    detector = SessionDetector()

    measurements: list[tuple[str, float]] = []
    for _, event in schedule.iterrows():
        race_name = str(event["EventName"])
        if "testing" in race_name.lower():
            continue

        # Race sessions that have not happened yet load "successfully" (no
        # exception) but leave FastF1's lap payload unpopulated, so gate on
        # completion rather than treating a load failure as the only signal.
        if not detector.is_session_completed(year, race_name, "R"):
            logger.debug("Skipping %s: race session not completed", race_name)
            continue

        gaps = measure_race(year, race_name)
        if gaps is None:
            logger.debug("No usable lap data for %s", race_name)
            continue

        for team, gap_s in gaps.items():
            measurements.append((team, gap_s))

    return measurements


def aggregate_by_team(measurements: list[tuple[str, float]]) -> dict[str, dict[str, float | int]]:
    """Average each team's per-race gaps across every race it appears in."""
    gaps_by_team: dict[str, list[float]] = defaultdict(list)
    for team, gap_s in measurements:
        gaps_by_team[team].append(gap_s)

    return {
        team: {
            "gap_s": round(statistics.mean(gaps), _ROUND_PRECISION),
            "races": len(gaps),
        }
        for team, gaps in gaps_by_team.items()
    }


def main() -> None:
    """Measure a season's team race pace and write it, unless --dry-run is set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the measured table without writing"
    )
    args = parser.parse_args()

    measurements = collect_measurements(args.year)
    aggregated = aggregate_by_team(measurements)
    print(f"{'team':<20} {'gap_s':>8} {'races':>6}")
    for team in sorted(aggregated, key=lambda name: aggregated[name]["gap_s"]):
        stats = aggregated[team]
        print(f"{team:<20} {stats['gap_s']:>8.3f} {stats['races']:>6}")

    if not aggregated:
        logger.warning("No completed, resolvable races found for %s", args.year)
        return

    if args.dry_run:
        return

    payload = {
        "year": args.year,
        "built_at": datetime.now(UTC).isoformat(),
        "teams": aggregated,
    }
    path = (
        PROJECT_ROOT / "data" / "processed" / "team_race_pace" / f"{args.year}_team_race_pace.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Wrote %d measured teams to %s", len(aggregated), path)


if __name__ == "__main__":
    main()
