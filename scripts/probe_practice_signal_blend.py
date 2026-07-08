"""Probe how much practice-informed reordering deserves trust in qualifying.

The 2026 checkpoint audit shows practice-informed qualifying checkpoints score
WORSE than the pre-weekend forecast (PRE MAE 2.88 vs FP1 3.26 / FP2 3.62 on
live artifacts). This probe quantifies, from stored predictions alone, whether
any partial trust in the practice-informed reordering beats ignoring it:

    blended_rank_score = w * checkpoint_rank + (1 - w) * PRE_rank

re-ranked into positions and scored against actual qualifying results.
``w = 1`` reproduces the stored checkpoint forecast, ``w = 0`` reproduces PRE.
This is an output-space analogue of the stored-checkpoint blend weight; it
bounds the config decision without a multi-hour simulation replay.

Diagnostics only — changes no product data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_shadow_challengers import _target_rows  # noqa: E402

from src.analysis.model_evaluation import compute_prediction_accuracy  # noqa: E402
from src.utils.accuracy_targets import (  # noqa: E402
    TARGET_MAIN_QUALIFYING,
    TARGET_SPRINT_QUALIFYING,
)
from src.utils.prediction_logger import PredictionLogger  # noqa: E402

QUALIFYING_TARGET_KEYS = (TARGET_MAIN_QUALIFYING, TARGET_SPRINT_QUALIFYING)
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


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


def _positions_by_driver(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return driver -> predicted position."""
    return {
        str(row["driver"]): float(row["position"])
        for row in rows
        if row.get("driver") is not None and row.get("position") is not None
    }


def _blend_ranks(
    pre_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    weight: float,
) -> list[dict[str, Any]]:
    """Re-rank drivers by a weighted blend of PRE and checkpoint positions."""
    pre_positions = _positions_by_driver(pre_rows)
    checkpoint_positions = _positions_by_driver(checkpoint_rows)
    shared = sorted(set(pre_positions) & set(checkpoint_positions))
    scored = sorted(
        shared,
        key=lambda driver: (
            weight * checkpoint_positions[driver] + (1.0 - weight) * pre_positions[driver],
            pre_positions[driver],
        ),
    )
    return [{"driver": driver, "position": index} for index, driver in enumerate(scored, start=1)]


def build_probe(year: int, *, weights: tuple[float, ...]) -> dict[str, Any]:
    """Score blended PRE/checkpoint qualifying forecasts across the season."""
    predictions = PredictionLogger().get_all_predictions(year)

    # Collect every scoreable qualifying row keyed by (race, target, checkpoint).
    rows_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for prediction in predictions:
        for row in _target_rows(prediction):
            if row["target_key"] not in QUALIFYING_TARGET_KEYS:
                continue
            checkpoint = str(row["checkpoint"]).upper()
            existing = rows_by_key[(row["race_name"], row["target_key"])].get(checkpoint)
            if existing is None or row["sort_key"] > existing["sort_key"]:
                rows_by_key[(row["race_name"], row["target_key"])][checkpoint] = row

    events: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (race_name, target_key), by_checkpoint in sorted(rows_by_key.items()):
        pre_row = by_checkpoint.get("PRE")
        if pre_row is None:
            continue
        for checkpoint, row in sorted(by_checkpoint.items()):
            if checkpoint == "PRE":
                continue
            mae_by_weight: dict[str, float] = {}
            for weight in weights:
                blended = _blend_ranks(pre_row["predicted_rows"], row["predicted_rows"], weight)
                if not blended:
                    continue
                metrics = compute_prediction_accuracy(blended, row["actual_rows"])
                mae_by_weight[f"{weight:.2f}"] = round(float(metrics["mae"]), 6)
            if not mae_by_weight:
                continue
            for weight_key, mae in mae_by_weight.items():
                pooled[checkpoint][weight_key].append(mae)
            events.append(
                {
                    "race_name": race_name,
                    "target_key": target_key,
                    "checkpoint": checkpoint,
                    "mae_by_weight": mae_by_weight,
                }
            )

    pooled_summary = {
        checkpoint: {
            weight_key: round(sum(values) / len(values), 6)
            for weight_key, values in sorted(weight_map.items())
        }
        for checkpoint, weight_map in sorted(pooled.items())
    }
    best_weight_by_checkpoint = {
        checkpoint: min(weight_map, key=lambda key: weight_map[key])
        for checkpoint, weight_map in pooled_summary.items()
    }
    return {
        "artifact_type": "practice_signal_blend_probe",
        "schema_version": 1,
        "year": int(year),
        "transform": "rerank by w * checkpoint_rank + (1 - w) * PRE_rank",
        "weights": [round(weight, 2) for weight in weights],
        "events_scored": len(events),
        "pooled_mae_by_checkpoint_and_weight": pooled_summary,
        "best_weight_by_checkpoint": best_weight_by_checkpoint,
        "events": events,
    }


def render_markdown(probe: dict[str, Any]) -> str:
    """Render the probe as reviewer-friendly markdown."""
    lines = [
        f"# Practice Signal Blend Probe - {probe['year']}",
        "",
        f"- Transform: `{probe['transform']}`",
        "- `w=1.00` reproduces the stored practice-informed checkpoint forecast;"
        " `w=0.00` reproduces the pre-weekend (PRE) forecast.",
        f"- Event/checkpoint pairs scored: **{probe['events_scored']}**",
        "",
        "## Pooled qualifying MAE by checkpoint and blend weight",
        "",
        "| Checkpoint | " + " | ".join(f"w={w:.2f}" for w in probe["weights"]) + " | Best |",
        "|---|" + "---:|" * (len(probe["weights"]) + 1),
    ]
    for checkpoint, weight_map in probe["pooled_mae_by_checkpoint_and_weight"].items():
        cells = [str(weight_map.get(f"{w:.2f}", "-")) for w in probe["weights"]]
        best = probe["best_weight_by_checkpoint"].get(checkpoint, "-")
        lines.append(f"| {checkpoint} | " + " | ".join(cells) + f" | `{best}` |")
    lines += [
        "",
        "Notes: blending happens in output (rank) space, so it bounds — but is "
        "not identical to — the `stored_checkpoint_blend_weight_*` strength-space "
        "knobs. A best weight of 0.00 means the practice-informed reordering adds "
        "no value over PRE at that checkpoint; small best weights argue for "
        "reducing the stored-checkpoint blend caps.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/practice_signal_blend_probe.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/practice_signal_blend_probe.md",
    )
    args = parser.parse_args()

    if args.env_file is not None:
        _load_env_file(args.env_file)

    probe = build_probe(args.year, weights=DEFAULT_WEIGHTS)

    json_out = args.json_out or Path(
        f"data/model_diagnostics/{args.year}/practice_signal_blend_probe.json"
    )
    md_out = args.md_out or Path(
        f"data/model_diagnostics/{args.year}/practice_signal_blend_probe.md"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(probe, indent=2, allow_nan=False), encoding="utf-8")
    md_out.write_text(render_markdown(probe), encoding="utf-8")
    print(f"Wrote {json_out} and {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
