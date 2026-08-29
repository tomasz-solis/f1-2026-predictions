"""Measure how far a car actually recovers from a back-of-grid start, by car quality.

A penalised driver's predicted recovery needs a bound taken from what races do, not from
judgement. The obvious version of that statistic -- pooling every start from P15 or worse
-- is dominated by backmarkers, who start at the back because that is their pace. Pooling
them with a quick car serving a penalty understates what the quick car can do by roughly
a factor of seven (pooled median +0 against a top-car median +7), which is enough to make
a correct prediction look pessimistic.

So the envelope is bucketed by car quality, proxied by the driver's median finishing
position across his OTHER races that season. A race never contributes to its own bucket.

Two corrections are applied that matter more than they look:

- Only classified finishers are counted, and BOTH grid and finish are re-ranked within
  that set. Raw ``GridPosition - Position`` counts a retirement ahead as a place gained,
  which measures attrition rather than racecraft.
- The pace proxy needs at least five other races, so an injury replacement or a partial
  season does not get bucketed off two results.

The 2026 regulations changed overtaking materially, so earlier seasons are a PRIOR for
what a quick car can do from the back -- not validation data for the 2026 model. 2026 on
its own supplies only two such starts, which is why the wider sample exists at all.

Usage:
    uv run python scripts/probe_recovery_envelope.py
    uv run python scripts/probe_recovery_envelope.py --years 2022 2023 2024 2025 --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fastf1  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.ergast", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

CLASSIFIED_STATUSES = ("Finished", "Lapped")
BACK_OF_GRID_FROM = 15
MIN_OTHER_RACES = 5
BUCKETS = (
    ("top car (season median finish <= 6)", lambda pace: pace <= 6),
    ("upper-mid (6 < median <= 11)", lambda pace: 6 < pace <= 11),
    ("backmarker (median > 11)", lambda pace: pace > 11),
)


def percentile(values: list[float], point: float) -> float:
    """Linear-interpolated percentile, so the envelope needs no numpy."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * point / 100.0
    lower = int(position)
    if lower + 1 >= len(ordered):
        return float(ordered[-1])
    return float(ordered[lower] + (position - lower) * (ordered[lower + 1] - ordered[lower]))


def collect_season(year: int) -> list[dict[str, Any]]:
    """Return one row per classified finisher of every race in `year`."""
    rows: list[dict[str, Any]] = []
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:  # noqa: BLE001 - a season that will not load is skipped, not fatal
        logger.warning("Schedule for %s did not load: %s", year, exc)
        return rows

    for _, event in schedule.iterrows():
        race = str(event["EventName"])
        if "testing" in race.lower():
            continue
        try:
            session = fastf1.get_session(year, race, "R")
            session.load(laps=False, telemetry=False, weather=False)
            results = session.results
        except Exception:  # noqa: BLE001 - a race that has not run yet is expected
            continue
        if results is None or len(results) < 10:
            continue

        classified = results[results["Status"].astype(str).isin(CLASSIFIED_STATUSES)]
        if len(classified) < 8:
            continue
        classified = classified.sort_values("Position")
        grid_rank = {
            driver: index + 1
            for index, driver in enumerate(classified.sort_values("GridPosition")["Abbreviation"])
        }
        finish_rank = {driver: index + 1 for index, driver in enumerate(classified["Abbreviation"])}
        for _, row in classified.iterrows():
            driver = str(row["Abbreviation"])
            rows.append(
                {
                    "year": year,
                    "race": race,
                    "driver": driver,
                    "grid": float(row["GridPosition"]),
                    "finish": float(row["Position"]),
                    "gain": grid_rank[driver] - finish_rank[driver],
                }
            )
    return rows


def back_of_grid_starts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep starts from P15 or worse, tagging each with the driver's season pace proxy."""
    by_driver: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_driver[(row["year"], row["driver"])].append(row)

    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["grid"] < BACK_OF_GRID_FROM:
            continue
        others = [
            other["finish"]
            for other in by_driver[(row["year"], row["driver"])]
            if other["race"] != row["race"]
        ]
        if len(others) < MIN_OTHER_RACES:
            continue
        row["pace"] = st.median(others)
        selected.append(row)
    return selected


def report(starts: list[dict[str, Any]]) -> None:
    """Print the bucketed envelope and the largest top-car recoveries."""
    print(f"\n{len(starts)} driver-races starting P{BACK_OF_GRID_FROM}+ with a pace proxy\n")
    for label, in_bucket in BUCKETS:
        gains = [row["gain"] for row in starts if in_bucket(row["pace"])]
        if not gains:
            print(f"{label:38s} n=0")
            continue
        print(
            f"{label:38s} n={len(gains):3d}  median {percentile(gains, 50):+5.1f}  "
            f"p75 {percentile(gains, 75):+5.1f}  p90 {percentile(gains, 90):+5.1f}  "
            f"max {max(gains):+4.0f}"
        )

    tops = [row for row in starts if row["pace"] <= 6]
    print("\nlargest top-car recoveries:")
    for row in sorted(tops, key=lambda item: -item["gain"])[:12]:
        print(
            f"  {row['year']} {row['driver']:4s} {row['race'][:24]:26s} "
            f"P{row['grid']:.0f} -> P{row['finish']:.0f}  gain {row['gain']:+.0f}"
        )


def main() -> None:
    """Measure the back-of-grid recovery envelope across the requested seasons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    fastf1.Cache.enable_cache(str(PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"))
    rows: list[dict[str, Any]] = []
    for year in args.years:
        season = collect_season(year)
        rows.extend(season)
        print(f"{year}: {len(season)} classified rows", flush=True)

    starts = back_of_grid_starts(rows)
    report(starts)

    if args.json:
        Path(args.json).write_text(
            json.dumps(starts, indent=2, sort_keys=True, default=float), encoding="utf-8"
        )


def _self_check() -> None:
    """Ranking within finishers must not credit a retirement ahead as a place gained."""
    assert percentile([0, 1, 2, 3, 4], 50) == 2
    assert percentile([5], 90) == 5
    rows = [
        {"year": 2026, "race": "A", "driver": "X", "grid": 20.0, "finish": 8.0, "gain": 5},
        *[
            {"year": 2026, "race": f"R{i}", "driver": "X", "grid": 3.0, "finish": 4.0, "gain": 0}
            for i in range(5)
        ],
    ]
    picked = back_of_grid_starts(rows)
    assert len(picked) == 1 and picked[0]["pace"] == 4.0


if __name__ == "__main__":
    _self_check()
    main()
