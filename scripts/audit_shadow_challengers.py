"""Audit target-specific background challengers and checkpoint MAE decay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_evaluation_report import _prediction_sort_key  # noqa: E402

from src.analysis.model_evaluation import compute_prediction_accuracy  # noqa: E402
from src.models.shadow_challenger import (  # noqa: E402
    SHADOW_CHALLENGER_VERSION,
    TARGET_SHADOW_RULES,
    build_shadow_challenger_for_target,
)
from src.utils.accuracy_targets import (  # noqa: E402
    ALL_TARGET_KEYS,
    TARGET_LABELS,
    explicit_target_actuals,
    explicit_target_predictions,
    sanitize_actual_rows,
    sanitize_prediction_rows,
    target_checkpoint_index,
)
from src.utils.prediction_logger import PredictionLogger  # noqa: E402
from src.utils.weekend import get_schedule_rows  # noqa: E402


@dataclass(frozen=True)
class TargetScore:
    """One target-level champion/challenger score."""

    race_name: str
    target_key: str
    checkpoint: str
    champion_mae: float
    challenger_mae: float | None


def _load_env_file(env_file: Path) -> None:
    """Load KEY=VALUE pairs without overriding already-exported values."""
    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip())


def _mean(values: list[float]) -> float | None:
    """Return a mean for non-empty values."""
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    """Return a median for non-empty values."""
    return float(median(values)) if values else None


def _fmt(value: Any) -> str:
    """Format optional numeric values."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _calendar_order(year: int) -> dict[str, int]:
    """Return normalized race-name order."""
    return {
        str(race).strip().casefold(): index
        for index, (race, _) in enumerate(get_schedule_rows(year))
    }


def _target_rows(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Return scoreable target rows from one prediction artifact."""
    metadata = prediction.get("metadata", {})
    checkpoint = str(metadata.get("session_name", "")).strip().upper()
    race_name = str(metadata.get("race_name", "")).strip()
    target_predictions = explicit_target_predictions(prediction)
    target_actuals = explicit_target_actuals(prediction)
    rows: list[dict[str, Any]] = []
    for target_key in ALL_TARGET_KEYS:
        payload = target_predictions.get(target_key)
        if not isinstance(payload, dict):
            continue
        predicted_rows = sanitize_prediction_rows(payload.get("predicted_order"))
        actual_rows = sanitize_actual_rows(target_actuals.get(target_key))
        if not predicted_rows or not actual_rows:
            continue
        rows.append(
            {
                "race_name": race_name,
                "checkpoint": checkpoint,
                "target_key": target_key,
                "predicted_rows": predicted_rows,
                "actual_rows": actual_rows,
                "sort_key": _prediction_sort_key(prediction),
            }
        )
    return rows


def _latest_target_rows_per_race_target(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the latest scoreable row per race and target."""
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for prediction in predictions:
        for row in _target_rows(prediction):
            key = (str(row["race_name"]), str(row["target_key"]))
            existing = selected.get(key)
            if existing is None or row["sort_key"] > existing["sort_key"]:
                selected[key] = row

    return list(selected.values())


def _target_row_sort_key(row: dict[str, Any], *, order: dict[str, int]) -> tuple[int, Any, int]:
    """Sort target rows by calendar event and checkpoint."""
    race_name = str(row.get("race_name", "")).strip().casefold()
    predicted_at, checkpoint_order = row.get("sort_key", (None, 99))
    return order.get(race_name, 10_000), predicted_at, checkpoint_order


def _checkpoint_decay(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize champion MAE by target/checkpoint across all scoreable artifacts."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for prediction in predictions:
        for row in _target_rows(prediction):
            mae = float(
                compute_prediction_accuracy(row["predicted_rows"], row["actual_rows"])["mae"]
            )
            grouped[(row["target_key"], row["checkpoint"])].append(mae)

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (target_key, checkpoint), values in grouped.items():
        by_target[target_key].append(
            {
                "target_key": target_key,
                "checkpoint": checkpoint,
                "events": len(values),
                "mean_mae": _mean(values),
                "median_mae": _median(values),
            }
        )

    return {
        target_key: sorted(
            rows,
            key=lambda row: target_checkpoint_index(
                target_key,
                "sprint" if "sprint" in target_key else "normal",
                str(row["checkpoint"]),
            ),
        )
        for target_key, rows in sorted(by_target.items())
    }


def _target_shadow_scores(
    predictions: list[dict[str, Any]],
    *,
    order: dict[str, int],
) -> dict[str, Any]:
    """Score latest target forecasts and their time-safe shadow challengers."""
    selected = sorted(
        _latest_target_rows_per_race_target(predictions),
        key=lambda row: _target_row_sort_key(row, order=order),
    )
    actual_history: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    scored: dict[str, list[TargetScore]] = defaultdict(list)

    for row in selected:
        target_key = row["target_key"]
        champion_rows = row["predicted_rows"]
        actual_rows = row["actual_rows"]
        champion_mae = float(compute_prediction_accuracy(champion_rows, actual_rows)["mae"])
        challenger_mae = None
        if target_key in TARGET_SHADOW_RULES and actual_history[target_key]:
            challenger_rows = build_shadow_challenger_for_target(
                champion_rows,
                actual_history[target_key],
                target_key=target_key,
            )
            challenger_mae = float(compute_prediction_accuracy(challenger_rows, actual_rows)["mae"])
        scored[target_key].append(
            TargetScore(
                race_name=row["race_name"],
                target_key=target_key,
                checkpoint=row["checkpoint"],
                champion_mae=champion_mae,
                challenger_mae=challenger_mae,
            )
        )
        actual_history[target_key].append(actual_rows)

    return {
        target_key: _summarize_target_scores(rows) for target_key, rows in sorted(scored.items())
    }


def _summarize_target_scores(rows: list[TargetScore]) -> dict[str, Any]:
    """Summarize target score rows."""
    champion_values = [row.champion_mae for row in rows]
    challenger_values = [
        float(row.challenger_mae) for row in rows if row.challenger_mae is not None
    ]
    champion_on_challenger_rows = [
        row.champion_mae for row in rows if row.challenger_mae is not None
    ]
    champion_mean = _mean(champion_values)
    challenger_mean = _mean(challenger_values)
    comparable_champion_mean = _mean(champion_on_challenger_rows)
    return {
        "events": len(rows),
        "challenger_scored_events": len(challenger_values),
        "champion_mean_mae": champion_mean,
        "champion_median_mae": _median(champion_values),
        "comparable_champion_mean_mae": comparable_champion_mean,
        "challenger_mean_mae": challenger_mean,
        "challenger_median_mae": _median(challenger_values),
        "mae_improvement_vs_comparable_champion": None
        if challenger_mean is None or comparable_champion_mean is None
        else comparable_champion_mean - challenger_mean,
        "rows": [row.__dict__ for row in rows],
    }


def build_shadow_challenger_audit(year: int) -> dict[str, Any]:
    """Build the full shadow challenger audit."""
    predictions = PredictionLogger().get_all_predictions(year)
    order = _calendar_order(year)
    return {
        "artifact_type": "shadow_challenger_audit",
        "schema_version": 1,
        "year": int(year),
        "shadow_challenger_version": SHADOW_CHALLENGER_VERSION,
        "target_rules": {
            target_key: rule.__dict__ for target_key, rule in TARGET_SHADOW_RULES.items()
        },
        "prediction_artifacts_loaded": len(predictions),
        "target_shadow_scores": _target_shadow_scores(predictions, order=order),
        "checkpoint_decay": _checkpoint_decay(predictions),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    """Render audit markdown."""
    lines = [
        f"# Shadow Challenger Audit - {audit['year']}",
        "",
        f"- Challenger version: `{audit['shadow_challenger_version']}`",
        f"- Prediction artifacts loaded: **{audit['prediction_artifacts_loaded']}**",
        "",
        "## Target Scores",
        "",
        "| Target | Events | Challenger events | Champion MAE | Challenger MAE | Improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_key, summary in audit.get("target_shadow_scores", {}).items():
        lines.append(
            f"| {TARGET_LABELS.get(target_key, target_key)} | {summary['events']} | "
            f"{summary['challenger_scored_events']} | "
            f"{_fmt(summary['comparable_champion_mean_mae'])} | "
            f"{_fmt(summary['challenger_mean_mae'])} | "
            f"{_fmt(summary['mae_improvement_vs_comparable_champion'])} |"
        )

    lines.extend(["", "## Checkpoint Decay", ""])
    for target_key, rows in audit.get("checkpoint_decay", {}).items():
        lines.extend(
            [
                f"### {TARGET_LABELS.get(target_key, target_key)}",
                "",
                "| Checkpoint | Events | Mean MAE | Median MAE |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['checkpoint']}` | {row['events']} | "
                f"{_fmt(row['mean_mae'])} | {_fmt(row['median_mae'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/shadow_challenger_audit.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Default: data/model_diagnostics/<year>/shadow_challenger_audit.md",
    )
    args = parser.parse_args()

    if args.env_file is not None:
        _load_env_file(args.env_file)

    audit = build_shadow_challenger_audit(args.year)
    json_out = args.json_out or Path(
        f"data/model_diagnostics/{args.year}/shadow_challenger_audit.json"
    )
    md_out = args.md_out or Path(f"data/model_diagnostics/{args.year}/shadow_challenger_audit.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
