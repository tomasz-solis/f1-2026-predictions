"""Measure how the simulator's race dynamics compare to what 2026 races actually did.

This is the measurement foundation for ``docs/OVERTAKING_CALIBRATION_PLAN.md``. Nothing
in that plan may be fitted until these numbers are reproducible, so this script is
deterministic and reports every statistic the plan's acceptance criteria reference.

Three statistics, each measured the same way on both sides:

``churn``
    Position changes per lap. The measured side calls ``extract_overtakes_from_race``
    directly rather than reimplementing its counting, so the two sides cannot drift
    apart definitionally.

``displacement``
    Mean ``|grid rank - finish rank|`` among classified finishers, with BOTH ranks
    recomputed within that set. Raw ``GridPosition - Position`` counts a retirement
    ahead as a place gained, which measures attrition rather than racecraft.

``recovery envelope``
    Pooled across races, the distribution of places gained by drivers starting P15 or
    worse. This is the empirical ceiling a penalised driver's predicted recovery has to
    sit inside.

Plus a mechanism decomposition of the simulator's position changes, which decides
whether the pass model is the right thing to calibrate at all.

Three traps this script exists to avoid, all of which cost real debugging time:

1. ``simulate_race_lap_by_lap`` is *injected*, bound in ``prediction_mixin`` and called
   as ``deps.simulate_race_lap_by_lap``. Patching it on the simulator module or on
   ``race_simulation`` silently does nothing.
2. Comparing a Monte Carlo *median* finish order against a single real race understates
   displacement mechanically. Every simulated statistic is computed inside each
   simulated race and then averaged.
3. ``2026_track_characteristics.json`` keys a circuit as "Spanish Grand Prix" while
   FastF1's event is "Barcelona Grand Prix". Circuits resolve through
   ``circuit_registry``, never by raw name.

Usage:
    uv run python scripts/probe_overtaking_realism.py
    uv run python scripts/probe_overtaking_realism.py --sims 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fastf1  # noqa: E402

from src.data.actual_results_fetcher import fetch_actual_session_results  # noqa: E402
from src.data.circuit_registry import CircuitResolutionError, resolve_track_data_key  # noqa: E402
from src.extractors.overtaking import extract_overtakes_from_race  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.ergast", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

CLASSIFIED_STATUSES = ("Finished", "Lapped")
BACK_OF_GRID_FROM = 15


def rank_within(order: list[str]) -> dict[str, int]:
    """Return 1-based ranks for `order`, which must already be sorted.

    Both the grid and the finish are ranked within the same set of drivers, so a
    retirement ahead of a driver does not register as a place he gained.
    """
    return {driver: index + 1 for index, driver in enumerate(order)}


def displacement_of(grid_order: list[str], finish_order: list[str]) -> float:
    """Mean absolute rank change between two orderings of the same driver set."""
    grid_rank = rank_within(grid_order)
    finish_rank = rank_within(finish_order)
    shared = [driver for driver in grid_rank if driver in finish_rank]
    if not shared:
        return float("nan")
    return st.mean(abs(grid_rank[d] - finish_rank[d]) for d in shared)


def percentile(values: list[float], point: float) -> float:
    """Linear-interpolated percentile, so the envelope does not depend on numpy."""
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


def correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, returning nan rather than raising on a degenerate input."""
    if len(xs) < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    denominator = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    if denominator == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denominator


def reliability_of(values: list[float], standard_errors: list[float]) -> tuple[float, float]:
    """Return (reliability, max attainable correlation) for one-observation-per-track data.

    Each circuit is measured from a single race, so part of the spread between circuits
    is sampling noise rather than a real difference. Subtracting the mean sampling
    variance from the observed between-track variance leaves the true signal; its share
    of the total is the reliability, and its square root bounds any correlation a model
    can achieve against these measurements.

    A non-positive reliability means the observed differences are entirely inside the
    noise -- a real result on 2026 displacement, not an error -- so the ceiling is
    reported as nan rather than taking the root of a negative number.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    observed = st.variance(values)
    sampling = st.mean(se**2 for se in standard_errors)
    if observed <= 0:
        return float("nan"), float("nan")
    reliability = (observed - sampling) / observed
    ceiling = reliability**0.5 if reliability > 0 else float("nan")
    return reliability, ceiling


def measured_side(year: int, race: str) -> dict[str, Any] | None:
    """Churn, displacement and back-of-grid gains for one real race."""
    stats = extract_overtakes_from_race(year, race)
    if stats is None:
        return None

    per_lap = [int(count) for count in stats.get("per_lap_changes", [])]
    churn = float(stats["avg_changes_per_lap"])
    churn_se = st.stdev(per_lap) / (len(per_lap) ** 0.5) if len(per_lap) > 1 else float("nan")

    session = fastf1.get_session(year, race, "R")
    session.load(laps=False, telemetry=False, weather=False)
    results = session.results
    classified = results[results["Status"].astype(str).isin(CLASSIFIED_STATUSES)]
    if len(classified) < 5:
        return None

    grid_order = list(classified.sort_values("GridPosition")["Abbreviation"])
    finish_order = list(classified.sort_values("Position")["Abbreviation"])
    grid_rank, finish_rank = rank_within(grid_order), rank_within(finish_order)
    deltas = [abs(grid_rank[d] - finish_rank[d]) for d in grid_rank]

    actual_grid = {
        str(row["Abbreviation"]): float(row["GridPosition"]) for _, row in classified.iterrows()
    }
    back_gains = [
        grid_rank[d] - finish_rank[d]
        for d in grid_rank
        if actual_grid.get(d, 0.0) >= BACK_OF_GRID_FROM
    ]

    return {
        "churn": churn,
        "churn_se": churn_se,
        "displacement": st.mean(deltas),
        "back_gains": back_gains,
    }


def _record_lap(
    driver_states: dict[str, dict[str, Any]],
    run: dict[str, Any],
    totals: dict[str, Any],
) -> None:
    """Record one completed lap: churn, and what caused each position change.

    ``run`` holds state for the simulated race in progress; ``totals`` accumulates across
    every race. Split that way because the simulator is re-entered per run and the
    previous run's final positions must not leak in as this run's "previous lap".
    """
    run["lap"] += 1
    retired_now = {
        driver
        for driver, state in driver_states.items()
        if state["has_dnf"] and driver not in run["retired"]
    }
    run["retired"] |= retired_now
    current = {
        driver: int(state["position"])
        for driver, state in driver_states.items()
        if not state["has_dnf"]
    }

    if run["lap"] > 1 and run["previous"]:
        pitted_now = {driver for driver, laps in run["pit_laps"].items() if run["lap"] in laps}
        # Sim-side analogue of the extractor's PitOutTime filter: it drops a driver whose
        # current lap is an out-lap, and the simulator charges the pit loss to the lap
        # listed in pit_laps.
        counted = 0
        for driver, position in current.items():
            if driver not in run["previous"] or driver in pitted_now:
                continue
            counted += 1
            if position == run["previous"][driver]:
                continue
            run["changes"] += 1
            # Precedence: retirement, then pit, then on-track. A change explained by a car
            # ahead disappearing is never credited to the pass model.
            ahead_retired = any(
                run["previous"].get(other, 10**6) < run["previous"][driver] for other in retired_now
            )
            displaced = [d for d, pos in run["previous"].items() if pos == position]
            if ahead_retired:
                totals["mechanisms"]["retirement"] += 1
            elif pitted_now and displaced and displaced[0] in pitted_now:
                totals["mechanisms"]["pit"] += 1
            else:
                totals["mechanisms"]["on_track"] += 1
        if counted:
            run["laps_counted"] += 1

    run["previous"] = current


def simulated_side(year: int, race: str, sims: int, seed: int) -> dict[str, Any] | None:
    """Run the predictor for one race and return statistics from inside each simulation."""
    import src.predictors.baseline.race.prediction_mixin as prediction_mixin
    import src.utils.lap_by_lap_simulator as simulator
    from src.predictors.baseline_2026 import Baseline2026Predictor

    qualifying = fetch_actual_session_results(year, race, "Q")
    if not qualifying:
        return None
    grid = [
        {"driver": row["driver"], "team": row["team"], "position": row["position"]}
        for row in qualifying
    ]
    grid_order = [row["driver"] for row in sorted(grid, key=lambda r: int(r["position"]))]

    totals: dict[str, Any] = {
        "churn_rates": [],
        "displacements": [],
        "mechanisms": {"retirement": 0, "pit": 0, "on_track": 0},
    }
    run: dict[str, Any] = {}
    real_simulate = prediction_mixin.simulate_race_lap_by_lap
    real_update = simulator._update_positions_from_times

    def wrapped_update(driver_states: dict[str, dict[str, Any]]) -> None:
        real_update(driver_states)
        _record_lap(driver_states, run, totals)

    def wrapped_simulate(**kwargs: Any) -> Any:
        pit_laps = {
            driver: set(strategy.get("pit_laps", []))
            for driver, strategy in kwargs["strategies"].items()
        }
        run.clear()
        run.update(pit_laps=pit_laps, previous={}, retired=set(), lap=0, changes=0, laps_counted=0)
        simulator._update_positions_from_times = wrapped_update
        try:
            result = real_simulate(**kwargs)
        finally:
            simulator._update_positions_from_times = real_update
        if run["laps_counted"]:
            totals["churn_rates"].append(run["changes"] / run["laps_counted"])
        dnf = set(result.get("dnf_drivers", []))
        classified = [d for d in result["finish_order"] if d not in dnf]
        if len(classified) >= 5:
            grid_classified = [d for d in grid_order if d in set(classified)]
            totals["displacements"].append(displacement_of(grid_classified, classified))
        return result

    prediction_mixin.simulate_race_lap_by_lap = wrapped_simulate
    try:
        Baseline2026Predictor(seed=seed).predict_race(
            qualifying_grid=grid,
            year=year,
            race_name=race,
            n_simulations=sims,
            weather="dry",
        )
    finally:
        prediction_mixin.simulate_race_lap_by_lap = real_simulate

    if not totals["churn_rates"] or not totals["displacements"]:
        return None
    return {
        "churn": st.mean(totals["churn_rates"]),
        "displacement": st.mean(totals["displacements"]),
        "mechanisms": dict(totals["mechanisms"]),
        "runs": len(totals["churn_rates"]),
    }


def completed_races(year: int) -> list[str]:
    """Every race of `year` that has run and resolves to a registered circuit."""
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    schedule = fastf1.get_event_schedule(year)
    races = []
    for _, event in schedule.iterrows():
        race = str(event["EventName"])
        if "testing" in race.lower():
            continue
        try:
            if resolve_track_data_key(race, year=year, location=event.get("Location")) is None:
                continue
        except CircuitResolutionError:
            continue
        if detector.is_session_completed(year, race, "R"):
            races.append(race)
    return races


def main() -> None:
    """Measure every completed race of the season and print the calibration report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--sims", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    started = time.time()
    fastf1.Cache.enable_cache(str(PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"))

    rows: list[dict[str, Any]] = []
    mechanisms = {"retirement": 0, "pit": 0, "on_track": 0}
    envelope: list[float] = []

    print(
        f"{'circuit':24s} {'churn meas':>10} {'churn sim':>10} {'ratio':>6} "
        f"{'disp meas':>10} {'disp sim':>9} {'ratio':>6}"
    )
    for race in completed_races(args.year):
        measured = measured_side(args.year, race)
        if measured is None:
            print(f"{race[:24]:24s} SKIPPED (no measured data)")
            continue
        simulated = simulated_side(args.year, race, args.sims, args.seed)
        if simulated is None:
            print(f"{race[:24]:24s} SKIPPED (no qualifying grid)")
            continue

        envelope.extend(measured["back_gains"])
        for key in mechanisms:
            mechanisms[key] += simulated["mechanisms"][key]
        rows.append(
            {
                "race": race,
                **{f"measured_{k}": v for k, v in measured.items() if k != "back_gains"},
                **{f"simulated_{k}": v for k, v in simulated.items() if k != "mechanisms"},
            }
        )
        print(
            f"{race[:24]:24s} {measured['churn']:>10.3f} {simulated['churn']:>10.3f} "
            f"{simulated['churn'] / measured['churn']:>6.2f} "
            f"{measured['displacement']:>10.2f} {simulated['displacement']:>9.2f} "
            f"{simulated['displacement'] / measured['displacement']:>6.2f}",
            flush=True,
        )

    if not rows:
        print("\nNo circuits measured.")
        return

    churn_ratio = st.mean(r["simulated_churn"] / r["measured_churn"] for r in rows)
    disp_ratio = st.mean(r["simulated_displacement"] / r["measured_displacement"] for r in rows)
    churn_rel, churn_ceiling = reliability_of(
        [r["measured_churn"] for r in rows], [r["measured_churn_se"] for r in rows]
    )
    churn_corr = correlation(
        [r["measured_churn"] for r in rows], [r["simulated_churn"] for r in rows]
    )
    total_changes = sum(mechanisms.values()) or 1

    print(f"\nn = {len(rows)} circuits, {args.sims} simulations each, seed {args.seed}")
    print(f"\nC1  displacement ratio  {disp_ratio:.2f}   (target 1.00 +/- 0.15)")
    print(f"C2  churn ratio         {churn_ratio:.2f}   (target 1.00 +/- 0.15)")
    print(f"C3  churn corr          {churn_corr:+.3f}")

    print(f"\nchurn reliability {churn_rel:+.3f}", end="")
    if churn_rel > 0:
        print(f"   ceiling {churn_ceiling:.3f}   C3 target {0.8 * churn_ceiling:.3f}")
    else:
        print("   NO per-track signal at this sample size")

    print("\nsimulated position changes by mechanism:")
    for name in ("on_track", "pit", "retirement"):
        share = 100.0 * mechanisms[name] / total_changes
        print(f"  {name:11s} {mechanisms[name]:>8d}  {share:5.1f}%")

    print(f"\nrecovery envelope, {len(envelope)} driver-races starting P{BACK_OF_GRID_FROM}+:")
    print(
        f"  median {percentile(envelope, 50):.1f}   p75 {percentile(envelope, 75):.1f}   "
        f"p90 {percentile(envelope, 90):.1f}   max {max(envelope):.0f}"
    )

    elapsed = time.time() - started
    print(f"\nwall clock {elapsed:.0f}s for {len(rows)} circuits x {args.sims} simulations")

    if args.json:
        payload = {
            "year": args.year,
            "sims": args.sims,
            "seed": args.seed,
            "rows": rows,
            "churn_ratio": round(churn_ratio, 4),
            "displacement_ratio": round(disp_ratio, 4),
            "churn_corr": round(churn_corr, 4),
            "churn_reliability": round(churn_rel, 4),
            "mechanisms": mechanisms,
            "envelope": {
                "n": len(envelope),
                "median": percentile(envelope, 50),
                "p75": percentile(envelope, 75),
                "p90": percentile(envelope, 90),
                "max": max(envelope),
            },
            "elapsed_s": round(elapsed, 1),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _self_check() -> None:
    """Rank-within-finishers is the piece most likely to be silently wrong."""
    # A retirement ahead must not register as a place gained: BBB starts and finishes
    # behind AAA, and CCC (who retired) is absent from both orderings.
    assert displacement_of(["AAA", "BBB"], ["AAA", "BBB"]) == 0.0
    # A genuine swap moves both drivers one place.
    assert displacement_of(["AAA", "BBB"], ["BBB", "AAA"]) == 1.0
    assert rank_within(["X", "Y", "Z"]) == {"X": 1, "Y": 2, "Z": 3}
    assert percentile([0, 1, 2, 3, 4], 50) == 2


if __name__ == "__main__":
    _self_check()
    main()
