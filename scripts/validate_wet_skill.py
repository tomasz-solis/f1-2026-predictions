"""Validate wet_skill predictions against actual completed wet race results.

Loads a completed wet race from FastF1, runs the prediction model on it,
and compares predicted vs actual finishing positions for drivers with
non-neutral wet_skill. Outputs calibration metrics.

Usage:
    .venv/bin/python scripts/validate_wet_skill.py --year 2024 --race "Sao Paulo Grand Prix"
    .venv/bin/python scripts/validate_wet_skill.py --year 2024 --race "British Grand Prix"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def load_actual_results(year: int, race_name: str) -> dict[str, int]:
    """Load actual finishing positions from FastF1."""
    cache_dir = Path("data/raw/.fastf1_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, race_name, "R")
    session.load(laps=False, telemetry=False, weather=True)

    results: dict[str, int] = {}
    for _, row in session.results.iterrows():
        driver = str(row.get("Abbreviation", "")).strip()
        pos = row.get("Position")
        status = str(row.get("Status", "")).strip()
        if driver and pd.notna(pos):
            try:
                int_pos = int(pos)
            except (TypeError, ValueError):
                continue
            # Only include classified finishers.
            # FastF1 Status is "Finished" or "+N Lap(s)" for classified drivers;
            # mechanical failures, accidents, etc. have other values.
            if status == "Finished" or status.startswith("+"):
                results[driver] = int_pos

    logger.info(
        "Classified finishers: %d (excluded %d DNF/DNS/DSQ)",
        len(results),
        len(session.results) - len(results),
    )

    return results


def load_wet_skill_priors() -> dict[str, float]:
    """Load current wet_skill values from driver characteristics."""
    chars_path = Path("data/processed/driver_characteristics.json")
    if not chars_path.exists():
        return {}
    with open(chars_path) as f:
        data = json.load(f)
    return {code: d.get("wet_skill", 0.70) for code, d in data.get("drivers", {}).items()}


def compute_predicted_advantage(
    priors: dict[str, float],
    weight: float = 0.16,
    neutral: float = 0.70,
    race_laps: int = 60,
) -> dict[str, float]:
    """Compute predicted cumulative wet advantage in seconds per driver."""
    return {driver: (ws - neutral) * weight * race_laps for driver, ws in priors.items()}


def validate(
    actual_positions: dict[str, int],
    priors: dict[str, float],
    neutral: float = 0.70,
) -> dict[str, dict]:
    """Compare predicted wet_skill advantage to actual position performance.

    For each driver pair where one has above-neutral and one below-neutral
    wet_skill, check if the predicted direction (better wet driver finishes
    ahead) matched reality.
    """
    drivers_in_both = set(actual_positions) & set(priors)
    above_neutral = [d for d in drivers_in_both if priors[d] > neutral + 0.02]
    below_neutral = [d for d in drivers_in_both if priors[d] < neutral - 0.02]

    comparisons = []
    for good in above_neutral:
        for weak in below_neutral:
            predicted_direction = "good_ahead"
            actual_direction = (
                "good_ahead" if actual_positions[good] < actual_positions[weak] else "weak_ahead"
            )
            correct = predicted_direction == actual_direction
            comparisons.append(
                {
                    "good_driver": good,
                    "weak_driver": weak,
                    "good_ws": priors[good],
                    "weak_ws": priors[weak],
                    "good_pos": actual_positions[good],
                    "weak_pos": actual_positions[weak],
                    "predicted": predicted_direction,
                    "actual": actual_direction,
                    "correct": correct,
                }
            )

    n_correct = sum(1 for c in comparisons if c["correct"])
    n_total = len(comparisons)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # Position correlation: among drivers with non-neutral wet_skill,
    # does higher wet_skill correlate with better finishing position?
    non_neutral = [
        (priors[d], actual_positions[d]) for d in drivers_in_both if abs(priors[d] - neutral) > 0.02
    ]
    if len(non_neutral) >= 3:
        ws_arr = np.array([x[0] for x in non_neutral])
        pos_arr = np.array([x[1] for x in non_neutral])
        correlation = float(np.corrcoef(ws_arr, pos_arr)[0, 1])
    else:
        correlation = None

    return {
        "pairwise_comparisons": comparisons,
        "n_comparisons": n_total,
        "n_correct": n_correct,
        "pairwise_accuracy": round(accuracy, 3),
        "wet_skill_position_correlation": round(correlation, 3)
        if correlation is not None
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate wet_skill against actual wet race")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--neutral", type=float, default=0.70)
    args = parser.parse_args()

    logger.info("Loading actual results for %d %s...", args.year, args.race)
    actual = load_actual_results(args.year, args.race)
    logger.info("Got %d finishing positions", len(actual))

    priors = load_wet_skill_priors()
    logger.info("Loaded wet_skill for %d drivers", len(priors))

    drivers_in_both = set(actual) & set(priors)
    logger.info("Drivers in both sets: %d", len(drivers_in_both))

    if not drivers_in_both:
        logger.warning("No overlap between actual results and wet_skill priors")
        return

    advantages = compute_predicted_advantage(priors, race_laps=60)

    print(f"\n{'Driver':<6} {'WetSkill':>8} {'Predicted':>10} {'Actual':>6}")
    print()
    for d in sorted(drivers_in_both, key=lambda x: actual[x]):
        ws = priors.get(d, 0.70)
        adv = advantages.get(d, 0.0)
        print(f"{d:<6} {ws:>8.2f} {adv:>+9.2f}s {actual[d]:>6}")

    result = validate(actual, priors, neutral=args.neutral)

    print(
        f"\nPairwise accuracy: {result['n_correct']}/{result['n_comparisons']} "
        f"({result['pairwise_accuracy']:.0%})"
    )
    if result["wet_skill_position_correlation"] is not None:
        corr = result["wet_skill_position_correlation"]
        # Negative correlation = higher wet_skill → lower position (better)
        print(
            f"Wet_skill vs position correlation: {corr:+.3f} "
            f"({'correct direction' if corr < 0 else 'wrong direction'})"
        )


if __name__ == "__main__":
    main()
