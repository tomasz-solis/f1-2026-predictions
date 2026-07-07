"""Probe post-hoc DNF-probability calibration transforms on stored predictions.

The 2026 evaluation shows the emitted ``dnf_probability`` scores a worse Brier
than a naive base-rate forecast: the model overforecasts retirement risk against
a low observed DNF rate. Before touching the predictor, this probe scores
simple shrinkage transforms of the *stored* probabilities offline:

    p' = lambda * p + (1 - lambda) * r

where ``r`` is the DNF base rate observed over *prior completed race events
only* (expanding window, leakage-safe), seeded with a documented prior for the
first event. ``lambda = 1`` reproduces the current model output; ``lambda = 0``
is a base-rate-only forecast.

The result feeds the decision whether to add an output-layer shrinkage knob to
the race predictor. Diagnostics only — this script changes no product data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_shadow_challengers import (  # noqa: E402
    _calendar_order,
    _latest_target_rows_per_race_target,
)

from src.analysis.model_evaluation import _coerce_ranked_rows  # noqa: E402
from src.utils.accuracy_targets import (  # noqa: E402
    TARGET_GRAND_PRIX_RACE,
    TARGET_SPRINT_RACE,
)
from src.utils.prediction_logger import PredictionLogger  # noqa: E402

RACE_TARGET_KEYS = (TARGET_SPRINT_RACE, TARGET_GRAND_PRIX_RACE)
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_env_file(env_file: Path) -> None:
    """Load KEY=VALUE lines so the artifact store can reach configured storage."""
    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _event_pairs(row: dict[str, Any]) -> list[tuple[float, float]]:
    """Return (predicted_probability, dnf_outcome) pairs for one scored event."""
    predicted_rows = _coerce_ranked_rows(row["predicted_rows"])
    actual_rows = _coerce_ranked_rows(row["actual_rows"])
    actual_by_driver = {actual["driver"]: actual for actual in actual_rows}
    pairs: list[tuple[float, float]] = []
    for predicted in predicted_rows:
        actual = actual_by_driver.get(predicted["driver"])
        if actual is None:
            continue
        probability = predicted.get("dnf_probability")
        if probability is None:
            continue
        clipped = min(max(float(probability), 0.0), 1.0)
        pairs.append((clipped, 1.0 if bool(actual["dnf"]) else 0.0))
    return pairs


def _event_sort_key(row: dict[str, Any], *, order: dict[str, int]) -> tuple[int, int]:
    """Calendar order; within a weekend the sprint race completes first."""
    race_index = order.get(str(row["race_name"]).strip().casefold(), 10_000)
    target_index = 0 if row["target_key"] == TARGET_SPRINT_RACE else 1
    return race_index, target_index


def build_probe(year: int, *, seed_base_rate: float, lambdas: tuple[float, ...]) -> dict[str, Any]:
    """Score each shrinkage lambda over the season's race events in replay order."""
    predictions = PredictionLogger().get_all_predictions(year)
    order = _calendar_order(year)
    rows = [
        row
        for row in _latest_target_rows_per_race_target(predictions)
        if row["target_key"] in RACE_TARGET_KEYS
    ]
    rows.sort(key=lambda row: _event_sort_key(row, order=order))

    events: list[dict[str, Any]] = []
    squared_errors: dict[float, float] = {lam: 0.0 for lam in lambdas}
    observations = 0
    prior_drivers = 0
    prior_dnfs = 0

    for row in rows:
        pairs = _event_pairs(row)
        if not pairs:
            continue
        expanding_rate = (prior_dnfs / prior_drivers) if prior_drivers > 0 else seed_base_rate
        event_errors: dict[float, float] = {}
        for lam in lambdas:
            event_errors[lam] = sum(
                (lam * probability + (1.0 - lam) * expanding_rate - outcome) ** 2
                for probability, outcome in pairs
            )
            squared_errors[lam] += event_errors[lam]
        observations += len(pairs)
        event_dnfs = int(sum(outcome for _, outcome in pairs))
        events.append(
            {
                "race_name": row["race_name"],
                "target_key": row["target_key"],
                "checkpoint": row["checkpoint"],
                "scored_drivers": len(pairs),
                "actual_dnf_count": event_dnfs,
                "expanding_base_rate": round(expanding_rate, 6),
                "brier_by_lambda": {
                    f"{lam:.2f}": round(event_errors[lam] / len(pairs), 6) for lam in lambdas
                },
            }
        )
        prior_drivers += len(pairs)
        prior_dnfs += event_dnfs

    pooled = {
        f"{lam:.2f}": round(squared_errors[lam] / observations, 6) if observations else None
        for lam in lambdas
    }
    best_lambda = (
        min(pooled, key=lambda key: pooled[key]) if observations else None  # type: ignore[arg-type]
    )
    return {
        "artifact_type": "dnf_calibration_probe",
        "schema_version": 1,
        "year": int(year),
        "transform": "p' = lambda * p + (1 - lambda) * expanding_prior_base_rate",
        "seed_base_rate": round(float(seed_base_rate), 6),
        "lambdas": [round(lam, 2) for lam in lambdas],
        "events_scored": len(events),
        "driver_observations": observations,
        "actual_dnf_count": prior_dnfs,
        "pooled_brier_by_lambda": pooled,
        "best_lambda": best_lambda,
        "events": events,
    }


def render_markdown(probe: dict[str, Any]) -> str:
    """Render the probe as reviewer-friendly markdown."""
    lines = [
        f"# DNF Calibration Probe - {probe['year']}",
        "",
        f"- Transform: `{probe['transform']}`",
        f"- Seed base rate (first event only): **{probe['seed_base_rate']}**",
        f"- Events scored: **{probe['events_scored']}**"
        f" ({probe['driver_observations']} driver observations,"
        f" {probe['actual_dnf_count']} DNFs)",
        f"- Best lambda by pooled Brier: **{probe['best_lambda']}**"
        " (1.00 = current model output, 0.00 = base-rate only)",
        "",
        "## Pooled Brier by lambda",
        "",
        "| Lambda | Pooled Brier |",
        "|---|---:|",
    ]
    for lam, brier in probe["pooled_brier_by_lambda"].items():
        marker = " (current)" if lam == "1.00" else ""
        lines.append(f"| `{lam}`{marker} | {brier} |")
    lines += [
        "",
        "## Per-event Brier",
        "",
        "| Race | Target | Checkpoint | Drivers | DNFs | Prior rate | "
        + " | ".join(f"λ={lam}" for lam in probe["pooled_brier_by_lambda"]) + " |",
        "|---|---|---|---:|---:|---:|" + "---:|" * len(probe["pooled_brier_by_lambda"]),
    ]
    for event in probe["events"]:
        briers = " | ".join(str(value) for value in event["brier_by_lambda"].values())
        lines.append(
            f"| {event['race_name']} | {event['target_key']} | {event['checkpoint']} "
            f"| {event['scored_drivers']} | {event['actual_dnf_count']} "
            f"| {event['expanding_base_rate']} | {briers} |"
        )
    lines += [
        "",
        "Notes: the expanding base rate uses prior completed events only "
        "(leakage-safe), so lambda=0.00 here is a deployable forecast, unlike the "
        "per-event oracle baseline in the evaluation report. Changing the emitted "
        "probability is an output-layer decision; the Monte Carlo DNF sampling "
        "input is out of scope (see docs/MODEL_PROMOTION.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--seed-base-rate", type=float, default=0.10)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/dnf_calibration_probe.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/dnf_calibration_probe.md",
    )
    args = parser.parse_args()

    if args.env_file is not None:
        _load_env_file(args.env_file)

    probe = build_probe(
        args.year,
        seed_base_rate=args.seed_base_rate,
        lambdas=DEFAULT_LAMBDAS,
    )

    json_out = args.json_out or Path(
        f"data/model_diagnostics/{args.year}/dnf_calibration_probe.json"
    )
    md_out = args.md_out or Path(f"data/model_diagnostics/{args.year}/dnf_calibration_probe.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(probe, indent=2, allow_nan=False), encoding="utf-8")
    md_out.write_text(render_markdown(probe), encoding="utf-8")
    print(f"Wrote {json_out} and {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
