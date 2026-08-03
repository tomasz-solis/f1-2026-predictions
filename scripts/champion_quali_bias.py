"""Score current-code champion qualifying bias over the 2026 walk-forward catalog.

Avoids the shelved challenger tree entirely: it calls production
``Baseline2026Predictor.predict_qualifying`` and the tracked
``identify_systematic_errors``, nothing else.

IMPORTANT — this is not a walk-forward. Every event is predicted against the
*current* artifact state, which already contains the results of that event, so
absolute error here is optimistic and is NOT comparable to the walk-forward MAE
numbers recorded in docs/MODEL_LEDGER.md. It is built for A/B work, where both
arms carry the same leakage and the delta is what matters.

Run from the trackside-labs repo root.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_evaluation import identify_systematic_errors  # noqa: E402
from src.predictors.baseline_2026 import Baseline2026Predictor  # noqa: E402

CATALOG = "data/historical_replay/2026/event_catalog.json"
LINEUPS = "data/current_lineups.json"
SEEDS = (17, 42, 91)


def load_events(dry_only: bool) -> list[dict[str, Any]]:
    """Return catalog events that have an actual qualifying grid to score against."""
    with open(CATALOG, encoding="utf-8") as handle:
        events = json.load(handle)["events"]
    return [
        e for e in events if e.get("actual_qualifying_grid") and (e.get("is_dry") or not dry_only)
    ]


def predict_one(
    predictor: Baseline2026Predictor, event: dict[str, Any], simulations: int
) -> list[dict[str, Any]] | None:
    """Return one PRE-checkpoint qualifying grid, or None if the predictor refused."""
    result = predictor.predict_qualifying(
        year=2026,
        race_name=event["race_name"],
        n_simulations=simulations,
        qualifying_stage="auto",
        practice_signal_mode="stored_profiles",
        checkpoint_session_name=None,
        weather="dry",
    )
    grid = result.get("grid") or result.get("qualifying_grid")
    return list(grid) if grid else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--dry-only", action="store_true", default=True)
    parser.add_argument("--all-events", dest="dry_only", action="store_false")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()

    events = load_events(args.dry_only)
    with open(LINEUPS, encoding="utf-8") as handle:
        lineups = json.load(handle)["current_lineups"]

    predictions: list[dict[str, Any]] = []
    actuals: list[dict[str, Any]] = []
    refused: list[str] = []

    for seed in SEEDS:
        predictor = Baseline2026Predictor(seed=seed)
        for event in events:
            try:
                grid = predict_one(predictor, event, args.simulations)
            except Exception as exc:  # noqa: BLE001 - a refusal must be visible, not fatal
                refused.append(f"{event['event_id']} seed={seed}: {type(exc).__name__}: {exc}")
                continue
            if not grid:
                refused.append(f"{event['event_id']} seed={seed}: empty grid")
                continue
            predictions.append({"race_name": event["race_name"], "grid": grid})
            actuals.append(
                {"race_name": event["race_name"], "grid": event["actual_qualifying_grid"]}
            )

    print(f"label={args.label}  events={len(events)}  seeds={SEEDS}  sims={args.simulations}")
    if refused:
        print(f"REFUSED {len(refused)}:")
        for line in refused[:10]:
            print(f"  {line}")
    if not predictions:
        raise SystemExit("no scored predictions - nothing to report")

    result = identify_systematic_errors(predictions, actuals)
    errors = [
        abs(float(p["position"]) - float(a["position"]))
        for pred, act in zip(predictions, actuals, strict=True)
        for p in pred["grid"]
        for a in act["grid"]
        if str(p.get("driver")) == str(a.get("driver"))
    ]
    team_bias, driver_bias = result["team_bias"], result["driver_bias"]

    print(f"races_compared {result['races_compared']}  driver_obs {result['driver_observations']}")
    print(f"MAE {statistics.fmean(errors):.4f} (leakage-inclusive, not walk-forward)")
    mean_abs_bias = statistics.fmean(
        abs(float(v["mean_signed_error"]))
        for v in driver_bias.values()
        if isinstance(v.get("mean_signed_error"), int | float)
    )
    print(f"mean per-driver |bias| {mean_abs_bias:.4f}")

    def mean(stats: dict | None) -> float | None:
        value = stats.get("mean_signed_error") if stats else None
        return value if isinstance(value, int | float) else None

    def show(value: float | None) -> str:
        return f"{value:+.2f}" if value is not None else "n/a"

    print(f"\n{'team':16}{'team':>8} | {'d1':>4}{'bias':>8}  {'d2':>4}{'bias':>8}{'spread':>8}")
    for team, stats in sorted(team_bias.items(), key=lambda kv: -(mean(kv[1]) or 0.0)):
        drivers = lineups.get(team) or []
        drivers = drivers if isinstance(drivers, list) else drivers.get("drivers", [])
        first, second = (list(drivers) + [None, None])[:2]
        a, b = mean(driver_bias.get(first)), mean(driver_bias.get(second))
        spread = f"{abs(a - b):.2f}" if a is not None and b is not None else "n/a"
        print(
            f"{team:16}{show(mean(stats)):>8} | {str(first):>4}{show(a):>8}"
            f"  {str(second):>4}{show(b):>8}{spread:>8}"
        )


if __name__ == "__main__":
    main()
