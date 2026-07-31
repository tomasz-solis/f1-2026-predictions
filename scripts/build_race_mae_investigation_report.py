#!/usr/bin/env python3
"""Build the phase-2 2026 race-MAE walk-forward comparison report.

Reads whatever variant walk-forward runs have completed under
``data/historical_replay/2026/walk_forward_runs/`` (see
``run_challenger_research_walk_forward.py``) and the run manifest recording refused
variants, and writes one immutable JSON + one markdown report under
``data/model_diagnostics/2026/race_mae_investigation/``.

This script never invents a scored event: a variant that has not been run, or that
was refused (e.g. Q1's 30-event minimum), is reported as such with its exact reason,
never silently omitted or backfilled.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "walk_forward_runs"
CATALOG_PATH = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "event_catalog.json"
PREDICTION_CACHE_ROOT = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "prediction_cache"
DRIVER_DEBUTS_PATH = PROJECT_ROOT / "data" / "driver_debuts.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_diagnostics" / "2026" / "race_mae_investigation"
JSON_OUTPUT_PATH = OUTPUT_DIR / "2026_walk_forward_variant_comparison.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "2026_WALK_FORWARD_VARIANT_COMPARISON.md"

_RACE_METRIC_KEYS = (
    "mae",
    "finisher_mae",
    "weighted_mae",
    "top_heavy_weighted_mae",
    "top_3_pct",
    "top_10_pct",
    "spearman_rank",
    "kendall_tau",
    "winner_accuracy_percent",
    "top3_accuracy_percent",
    "dnf_brier",
)
_RACE_VIEWS = ("conditional_actual_grid", "end_to_end_predicted_grid")


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    finite = [v for v in values if v is not None]
    if not finite:
        return {"mean": None, "std": None, "n_events": 0}
    return {
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0,
        "n_events": len(finite),
    }


def _load_runs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = RUNS_DIR / "run_manifest.json"
    run_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    runs: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return run_manifest, runs


def _catalog_by_id() -> dict[str, dict[str, Any]]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {row["event_id"]: row for row in payload["events"]}


def _variant_table(replay: dict[str, Any]) -> dict[str, Any]:
    """Aggregate one variant's scored events into event-equal-weighted rows."""
    scored_events = replay["scored_events"]
    by_checkpoint: dict[str, dict[str, Any]] = {}
    per_event_rows: list[dict[str, Any]] = []

    for event in scored_events:
        event_id = event["event_id"]
        for checkpoint, checkpoint_payload in event["checkpoints"].items():
            race_views = checkpoint_payload.get("race_views")
            row: dict[str, Any] = {
                "event_id": event_id,
                "session_kind": event["session_kind"],
                "checkpoint": checkpoint,
                "qualifying": {
                    "champion": checkpoint_payload["champion"],
                    "challenger": checkpoint_payload["challenger"],
                },
            }
            if race_views is not None:
                row["race_views"] = race_views
            per_event_rows.append(row)

            bucket = by_checkpoint.setdefault(
                checkpoint,
                {
                    "qualifying": {"champion": {}, "challenger": {}},
                    "race_views": {
                        view: {"champion": {}, "challenger": {}} for view in _RACE_VIEWS
                    },
                },
            )
            for role in ("champion", "challenger"):
                for metric, value in checkpoint_payload[role].items():
                    if not isinstance(value, int | float):
                        continue
                    bucket["qualifying"][role].setdefault(metric, []).append(float(value))
            if race_views is not None:
                for view in _RACE_VIEWS:
                    for role in ("champion", "challenger"):
                        for metric in _RACE_METRIC_KEYS:
                            value = race_views[view][role].get(metric)
                            if value is None:
                                continue
                            bucket["race_views"][view][role].setdefault(metric, []).append(
                                float(value)
                            )

    aggregate: dict[str, Any] = {}
    for checkpoint, bucket in by_checkpoint.items():
        aggregate[checkpoint] = {
            "qualifying": {
                role: {metric: _mean_std(values) for metric, values in metrics.items()}
                for role, metrics in bucket["qualifying"].items()
            },
            "race_views": {
                view: {
                    role: {metric: _mean_std(values) for metric, values in metrics.items()}
                    for role, metrics in roles.items()
                }
                for view, roles in bucket["race_views"].items()
            },
        }
    return {
        "aggregate_by_checkpoint": aggregate,
        "per_event_rows": per_event_rows,
        "skipped_events": replay["skipped_events"],
        "leakage_audit_passed": replay["leakage_audit"]["passed"],
    }


def _decomposition(replay: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Decompose champion+variant race MAE on the user's requested axes."""
    rows: list[dict[str, Any]] = []
    for event in replay["scored_events"]:
        event_id = event["event_id"]
        for checkpoint, checkpoint_payload in event["checkpoints"].items():
            race_views = checkpoint_payload.get("race_views")
            if race_views is None:
                continue
            for view in _RACE_VIEWS:
                for role in ("champion", "challenger"):
                    metrics = race_views[view][role]
                    rows.append(
                        {
                            "event_id": event_id,
                            "session_kind": event["session_kind"],
                            "checkpoint": checkpoint,
                            "view": view,
                            "role": role,
                            "mae": metrics.get("mae"),
                            "finisher_mae": metrics.get("finisher_mae"),
                        }
                    )

    def _group_mean(
        predicate: Any, *, metric: str, role: str, view: str, checkpoint: str
    ) -> dict[str, Any]:
        values = [
            r[metric]
            for r in rows
            if r["role"] == role
            and r["view"] == view
            and r["checkpoint"] == checkpoint
            and predicate(r)
        ]
        return _mean_std([v for v in values if v is not None])

    format_axis = {}
    for checkpoint in {r["checkpoint"] for r in rows}:
        for view in _RACE_VIEWS:
            for role in ("champion", "challenger"):
                format_axis[f"{checkpoint}:{view}:{role}"] = {
                    "sprint": _group_mean(
                        lambda r: r["session_kind"] == "sprint",
                        metric="mae",
                        role=role,
                        view=view,
                        checkpoint=checkpoint,
                    ),
                    "main": _group_mean(
                        lambda r: r["session_kind"] == "main",
                        metric="mae",
                        role=role,
                        view=view,
                        checkpoint=checkpoint,
                    ),
                }

    # Per-round trend: champion finisher_mae in chronological event order (self-learning check).
    chronological_ids = sorted(
        {r["event_id"] for r in rows}, key=lambda eid: catalog[eid]["event_start_at"]
    )
    trend = []
    for event_id in chronological_ids:
        matches = [
            r
            for r in rows
            if r["event_id"] == event_id
            and r["role"] == "champion"
            and r["view"] == "conditional_actual_grid"
        ]
        if not matches:
            continue
        # Prefer the PRE checkpoint if present so the trend compares like-for-like.
        preferred = next((r for r in matches if r["checkpoint"] == "PRE"), matches[0])
        trend.append(
            {
                "event_id": event_id,
                "round_number": catalog[event_id]["round_number"],
                "checkpoint": preferred["checkpoint"],
                "finisher_mae": preferred["finisher_mae"],
            }
        )

    # DNF floor: mae vs finisher_mae, champion, conditional_actual_grid, PRE checkpoint.
    dnf_floor = []
    for event_id in chronological_ids:
        matches = [
            r
            for r in rows
            if r["event_id"] == event_id
            and r["role"] == "champion"
            and r["view"] == "conditional_actual_grid"
            and r["checkpoint"] == "PRE"
        ]
        if not matches:
            continue
        row = matches[0]
        dnf_floor.append(
            {
                "event_id": event_id,
                "all_driver_mae": row["mae"],
                "finisher_mae": row["finisher_mae"],
                "dnf_mae_contribution": (
                    None
                    if row["mae"] is None or row["finisher_mae"] is None
                    else row["mae"] - row["finisher_mae"]
                ),
            }
        )

    return {
        "weekend_format": format_axis,
        "per_round_trend_champion_finisher_mae": trend,
        "dnf_floor_champion_pre_conditional": dnf_floor,
    }


def _driver_cohort(debut_year: int | None, *, event_year: int = 2026) -> str:
    if debut_year is None:
        return "unknown"
    if debut_year >= event_year:
        return "rookie"
    if debut_year == event_year - 1:
        return "second_year"
    return "established"


def _driver_cohort_decomposition(
    runs: dict[str, dict[str, Any]], catalog: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Mine the raw prediction cache for per-driver champion errors by debut cohort.

    Event-level aggregate metrics (mean over all drivers) cannot answer "does the
    model do worse on rookies" -- that needs per-driver absolute error, which only
    the raw cached ``finish_order`` predictions carry (the walk-forward replay
    artifact keeps only the event-level reduction). This reads the champion's
    already-computed ``conditional_actual_grid`` predictions directly from the
    disk-backed prediction cache instead of re-running anything.
    """
    if not DRIVER_DEBUTS_PATH.is_file() or not PREDICTION_CACHE_ROOT.is_dir():
        return {"computed": False, "reason": "driver debut data or prediction cache unavailable"}
    debuts = json.loads(DRIVER_DEBUTS_PATH.read_text(encoding="utf-8"))["driver_debuts"]

    cohort_errors: dict[str, list[float]] = {}
    seen: set[tuple[str, str, int]] = set()
    for path in PREDICTION_CACHE_ROOT.glob("*/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = payload.get("key", {})
        if key.get("kind") != "race_views" or key.get("variant_id") != "champion":
            continue
        event_id = key.get("event_id")
        checkpoint = key.get("checkpoint")
        seed = key.get("seed")
        if event_id not in catalog or (event_id, checkpoint, seed) in seen:
            continue
        seen.add((event_id, checkpoint, seed))
        prediction = payload["value"].get("conditional_actual_grid")
        if prediction is None:
            continue
        predicted_positions = {row["driver"]: row["position"] for row in prediction["finish_order"]}
        for actual_row in catalog[event_id]["actual_race_finish_order"]:
            driver = actual_row["driver"]
            if driver not in predicted_positions or actual_row.get("dnf"):
                continue
            error = abs(predicted_positions[driver] - actual_row["position"])
            cohort = _driver_cohort(debuts.get(driver))
            cohort_errors.setdefault(cohort, []).append(error)

    if not cohort_errors:
        return {"computed": False, "reason": "no champion race_views cache entries found"}
    return {
        "computed": True,
        "basis": "champion conditional_actual_grid, finishers only, all cached checkpoints/seeds pooled",
        "by_cohort": {
            cohort: {
                "finisher_mae": float(statistics.fmean(errors)),
                "n_driver_observations": len(errors),
            }
            for cohort, errors in sorted(cohort_errors.items())
        },
    }


def _wet_events_excluded(catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"event_id": event_id, "session_kind": row["session_kind"]}
        for event_id, row in catalog.items()
        if not row["is_dry"]
    ]


def _q0_invariance_check(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assert q0's conditional_actual_grid race output never diverges from champion."""
    q0 = runs.get("q0_driver_state")
    if q0 is None:
        return {
            "checked": False,
            "reason": "q0_driver_state has not completed a walk-forward run yet",
        }
    replay = q0["replay"]
    mismatches = []
    checked = 0
    for event in replay["scored_events"]:
        for checkpoint, checkpoint_payload in event["checkpoints"].items():
            race_views = checkpoint_payload.get("race_views")
            if race_views is None:
                continue
            view = race_views["conditional_actual_grid"]
            champion = view["champion"]
            challenger = view["challenger"]
            checked += 1
            for metric in ("mae", "finisher_mae", "weighted_mae"):
                champion_value = champion.get(metric)
                challenger_value = challenger.get(metric)
                if champion_value is None or challenger_value is None:
                    continue
                if abs(champion_value - challenger_value) > 1e-9:
                    mismatches.append(
                        {
                            "event_id": event["event_id"],
                            "checkpoint": checkpoint,
                            "metric": metric,
                            "champion": champion_value,
                            "q0": challenger_value,
                        }
                    )
    return {
        "checked": True,
        "checkpoints_checked": checked,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def main() -> int:
    run_manifest, runs = _load_runs()
    catalog = _catalog_by_id()

    variants: dict[str, Any] = {}
    for variant_id, status in run_manifest.get("results", {}).items():
        if status["status"] == "refused":
            variants[variant_id] = {
                "status": "refused",
                "error": status["error"],
                "error_type": status["error_type"],
            }
            continue
        replay_payload = runs.get(variant_id)
        if replay_payload is None:
            variants[variant_id] = {"status": "not_run"}
            continue
        replay = replay_payload["replay"]
        variants[variant_id] = {
            "status": "scored",
            "manifest_sha256": replay_payload["manifest"]["manifest_sha256"],
            "comparison": _variant_table(replay),
            "decomposition": _decomposition(replay, catalog),
        }

    report: dict[str, Any] = {
        "artifact_type": "race_mae_walk_forward_variant_comparison",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_manifest": run_manifest,
        "event_catalog_summary": {
            "included_events": sorted(catalog),
            "wet_events_excluded_from_all_variant_scoring": _wet_events_excluded(catalog),
        },
        "q0_conditional_grid_invariance_check": _q0_invariance_check(runs),
        "driver_cohort_decomposition": _driver_cohort_decomposition(runs, catalog),
        "variants": variants,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# 2026 Walk-Forward Race-MAE Variant Comparison (Phase 2)",
        "",
        f"Generated: {report['generated_at']}",
        f"Events in catalog: {len(catalog)} ({sorted(catalog)})",
        f"Wet events excluded from all variant scoring: "
        f"{[r['event_id'] for r in report['event_catalog_summary']['wet_events_excluded_from_all_variant_scoring']]}",
        "",
        "## Variant status",
        "",
    ]
    for variant_id, payload in sorted(variants.items()):
        if payload["status"] == "refused":
            lines.append(
                f"- `{variant_id}`: REFUSED -- {payload['error_type']}: {payload['error']}"
            )
        elif payload["status"] == "not_run":
            lines.append(f"- `{variant_id}`: NOT RUN in this session (time budget)")
        else:
            n_scored = len(payload["comparison"]["per_event_rows"])
            lines.append(f"- `{variant_id}`: scored ({n_scored} event/checkpoint rows)")
    lines.append("")

    lines.append("## Race-view comparison (mean across scored events, PRE checkpoint)")
    lines.append("")
    header = (
        "| variant | view | role | mae | finisher_mae | weighted_mae | top_heavy_weighted_mae "
        "| top_3_pct | top_10_pct | winner_acc_% | spearman | kendall | n_events |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 13)
    for variant_id, payload in sorted(variants.items()):
        if payload["status"] != "scored":
            continue
        agg = payload["comparison"]["aggregate_by_checkpoint"].get("PRE")
        if agg is None:
            continue
        for view in _RACE_VIEWS:
            for role in ("champion", "challenger"):
                metrics = agg["race_views"][view][role]

                def _fmt(key: str, *, metrics: dict[str, Any] = metrics) -> str:
                    entry = metrics.get(key, {})
                    value = entry.get("mean")
                    return f"{value:.3f}" if isinstance(value, float) else "n/a"

                n_events = metrics.get("mae", {}).get("n_events", 0)
                label = variant_id if role == "challenger" else "champion"
                lines.append(
                    f"| {label} | {view} | {role} | {_fmt('mae')} | {_fmt('finisher_mae')} | "
                    f"{_fmt('weighted_mae')} | {_fmt('top_heavy_weighted_mae')} | {_fmt('top_3_pct')} | "
                    f"{_fmt('top_10_pct')} | {_fmt('winner_accuracy_percent')} | {_fmt('spearman_rank')} | "
                    f"{_fmt('kendall_tau')} | {n_events} |"
                )
    lines.append("")

    lines.append("## q0 conditional_actual_grid invariance check")
    lines.append("")
    q0_check = report["q0_conditional_grid_invariance_check"]
    lines.append(f"```json\n{json.dumps(q0_check, indent=2)}\n```")
    lines.append("")

    # Champion decomposition is identical across every scored variant run (champion
    # is scored the same way regardless of which challenger it is paired against),
    # so any one scored variant's decomposition carries it.
    first_scored = next((p for p in variants.values() if p["status"] == "scored"), None)
    if first_scored is not None:
        decomposition = first_scored["decomposition"]
        lines.append("## Per-round trend (champion finisher_mae, conditional_actual_grid, PRE)")
        lines.append("")
        lines.append("| round | event | finisher_mae |")
        lines.append("|---|---|---|")
        for row in decomposition["per_round_trend_champion_finisher_mae"]:
            lines.append(
                f"| {row['round_number']} | {row['event_id']} | {row['finisher_mae']:.3f} |"
            )
        lines.append("")

        lines.append("## DNF floor (champion, conditional_actual_grid, PRE)")
        lines.append("")
        lines.append("| event | all_driver_mae | finisher_mae | dnf_mae_contribution |")
        lines.append("|---|---|---|---|")
        for row in decomposition["dnf_floor_champion_pre_conditional"]:
            lines.append(
                f"| {row['event_id']} | {row['all_driver_mae']:.3f} | {row['finisher_mae']:.3f} | "
                f"{row['dnf_mae_contribution']:.3f} |"
            )
        lines.append("")

        lines.append(
            "## Sprint vs main weekend format (mae, PRE, conditional_actual_grid, champion)"
        )
        lines.append("")
        key = "PRE:conditional_actual_grid:champion"
        if key in decomposition["weekend_format"]:
            fmt = decomposition["weekend_format"][key]
            lines.append(f"- sprint: {json.dumps(fmt['sprint'])}")
            lines.append(f"- main: {json.dumps(fmt['main'])}")
        lines.append("")

    lines.append(
        "## Driver cohort decomposition (champion, finishers only, all cached checkpoints/seeds)"
    )
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['driver_cohort_decomposition'], indent=2)}\n```")
    lines.append("")

    MARKDOWN_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json: {JSON_OUTPUT_PATH}")
    print(f"markdown: {MARKDOWN_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
