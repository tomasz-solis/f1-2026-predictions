"""Measure 2026 overtaking rates from cached FastF1 lap data.

Writes ``overtaking_avg_changes_per_lap`` and ``overtaking_observed_races`` into
``data/processed/track_characteristics/<year>_track_characteristics.json`` for every
circuit with a completed race session, leaving circuits without one untouched.

The measurement counts field-wide **position changes**, not overtakes: it counts BOTH
cars in a swap (A passes B increments the count for A and for B). This intentionally
matches the historical 2022-2024 ``overtaking_avg_changes_per_lap`` prior stored in
these files (see ``src/extractors/overtaking.py``), so the two remain comparable across
regulation eras rather than measuring a different quantity under the same field name.
That prior is not wrong -- it is a valid measurement of the previous (2022-2025)
regulation era's cars, and is being transitioned away from gradually as 2026 evidence
accumulates (see ``_blend_overtaking_with_transition_prior`` in
``src/data/track_data_loader.py``).

Usage:
    uv run python scripts/extract_overtaking_rates.py --year 2026
    uv run python scripts/extract_overtaking_rates.py --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import fastf1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.circuit_registry import CircuitResolutionError, resolve_track_data_key  # noqa: E402
from src.extractors.overtaking import extract_overtakes_from_race  # noqa: E402
from src.utils.session_detector import SessionDetector  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.ergast", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_ROUND_PRECISION = 3


class Measurement(NamedTuple):
    """One completed race's measured position-change rate, keyed to its data file entry."""

    race_name: str
    data_key: str
    avg_changes_per_lap: float
    laps_analyzed: int


def _enable_fastf1_cache() -> None:
    """Enable the repo's shared FastF1 cache so extraction reads cached lap data."""
    cache_dir = PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def collect_measurements(year: int) -> list[Measurement]:
    """Measure every completed race of `year` that resolves to a known circuit.

    Skips testing events, races whose circuit is not yet registered (hard failure in
    ``circuit_registry`` is treated as skip-and-log here, not an extraction error), and
    races with no characteristics data key yet (e.g. a brand-new circuit). Races with no
    cached/loadable lap data (not yet run) are silently skipped -- that is the normal
    steady state for most of the season.
    """
    _enable_fastf1_cache()
    schedule = fastf1.get_event_schedule(year)
    detector = SessionDetector()

    measurements: list[Measurement] = []
    for _, event in schedule.iterrows():
        race_name = str(event["EventName"])
        if "testing" in race_name.lower():
            continue

        try:
            data_key = resolve_track_data_key(race_name, year=year, location=event.get("Location"))
        except CircuitResolutionError as exc:
            logger.info("Skipping unregistered circuit for %s: %s", race_name, exc)
            continue
        if data_key is None:
            logger.info(
                "Skipping %s: circuit is registered but has no characteristics data yet",
                race_name,
            )
            continue

        # Race sessions that have not happened yet load "successfully" (no exception)
        # but leave FastF1's lap/results payloads unpopulated, so gate on completion
        # rather than treating a load failure as the only "not run yet" signal.
        if not detector.is_session_completed(year, race_name, "R"):
            logger.debug("Skipping %s: race session not completed", race_name)
            continue

        stats = extract_overtakes_from_race(year, race_name)
        if stats is None:
            logger.debug("No lap data available for %s (unloadable)", race_name)
            continue

        measurements.append(
            Measurement(
                race_name=race_name,
                data_key=data_key,
                avg_changes_per_lap=float(stats["avg_changes_per_lap"]),
                laps_analyzed=int(stats["laps_analyzed"]),
            )
        )

    return measurements


def aggregate_by_data_key(measurements: list[Measurement]) -> dict[str, tuple[float, int]]:
    """Map each circuit's data key to (measured rate, races measured)."""
    return {
        measurement.data_key: (round(measurement.avg_changes_per_lap, _ROUND_PRECISION), 1)
        for measurement in measurements
    }


def _track_characteristics_path(year: int) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "track_characteristics"
        / f"{year}_track_characteristics.json"
    )


def apply_measurements(path: Path, aggregated: dict[str, tuple[float, int]]) -> dict:
    """Write measured values into the existing track_characteristics payload.

    Only the two measured fields are set on each matching track entry; every other
    field, and every track with no measurement, is left exactly as loaded.
    """
    with open(path) as f:
        data = json.load(f)

    tracks = data.get("tracks", {})
    for data_key, (avg_changes_per_lap, races_measured) in aggregated.items():
        track_entry = tracks.get(data_key)
        if track_entry is None:
            logger.warning(
                "%s has no existing entry in %s; skipping (register the circuit first)",
                data_key,
                path.name,
            )
            continue
        track_entry["overtaking_avg_changes_per_lap"] = avg_changes_per_lap
        track_entry["overtaking_observed_races"] = races_measured

    return data


def print_table(aggregated: dict[str, tuple[float, int]]) -> None:
    print(f"{'circuit':<32} {'changes/lap':>12} {'races':>6}")
    for data_key in sorted(aggregated):
        avg_changes_per_lap, races_measured = aggregated[data_key]
        print(f"{data_key:<32} {avg_changes_per_lap:>12.3f} {races_measured:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the measured table without writing"
    )
    args = parser.parse_args()

    measurements = collect_measurements(args.year)
    aggregated = aggregate_by_data_key(measurements)
    print_table(aggregated)

    if not aggregated:
        logger.warning("No completed, resolvable races found for %s", args.year)
        return

    if args.dry_run:
        return

    path = _track_characteristics_path(args.year)
    data = apply_measurements(path, aggregated)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote %d measured circuits to %s", len(aggregated), path)


if __name__ == "__main__":
    main()
