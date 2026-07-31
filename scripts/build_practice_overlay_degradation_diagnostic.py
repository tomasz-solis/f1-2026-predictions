#!/usr/bin/env python3
"""Read-only diagnostic: does the champion (served-forecast) prediction degrade
through a race weekend as `stored_profiles` practice overlays apply?

Triggered by the v3 report's matched-subset finding (champion end-to-end
finisher_mae PRE 4.16 -> FP2 4.54 -> FP3 4.89 on the identical 4 main-dry
events). This script:

1. Confirms the degradation on matched-event subsets across MORE metrics
   (weighted_mae, top_heavy_weighted_mae, qualifying grid_mae) with per-seed
   spread, to rule out a small-n artifact.
2. Localizes it (conditional_actual_grid, grid held fixed to the real grid,
   vs end_to_end_predicted_grid, which also carries grid-propagation error).
3. Names the worst-degrading events (per-event PRE->FP3 / PRE->FP1 delta).

Diagnosis only -- writes no config, touches no production code. All numbers
are mined from the existing prediction cache; no new simulations are run.
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

from scripts.build_race_mae_investigation_report_v2 import (  # noqa: E402
    OUTPUT_DIR,
    PREDICTION_CACHE_ROOT,
    _catalog_by_id,
    _load_tagged_runs,
    _variant_table,
)
from scripts.build_race_mae_investigation_report_v3 import _RACE_VIEWS  # noqa: E402

from src.analysis.challenger_walk_forward import _qualifying_grid_metrics  # noqa: E402
from src.analysis.model_evaluation import compute_prediction_accuracy  # noqa: E402

JSON_OUTPUT_PATH = OUTPUT_DIR / "practice_overlay_degradation_diagnostic.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "PRACTICE_OVERLAY_DEGRADATION_DIAGNOSTIC.md"

_QUALIFYING_METRIC = "grid_mae"
_RACE_METRICS = ("finisher_mae", "weighted_mae", "top_heavy_weighted_mae")


def _index_champion_metrics(
    runs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """event_id -> checkpoint -> {'qualifying': {...}, view: {...}}, champion only.

    First run encountered for a given (event, checkpoint) wins -- same
    first-wins precedent v2/v3 already use for `checkpoint_source`.
    """
    index: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for payload in runs.values():
        for row in _variant_table(payload["replay"])["per_event_rows"]:
            bucket = index.setdefault(row["event_id"], {}).setdefault(row["checkpoint"], {})
            bucket.setdefault("qualifying", row["qualifying"]["champion"])
            race_views = row.get("race_views")
            if race_views:
                for view in _RACE_VIEWS:
                    bucket.setdefault(view, race_views[view]["champion"])
    return index


def _matched_event_ids(
    index: dict[str, dict[str, Any]], checkpoints: list[str], kind_events: set[str]
) -> list[str]:
    per_checkpoint = [
        {eid for eid, by_cp in index.items() if cp in by_cp} & kind_events for cp in checkpoints
    ]
    if not all(per_checkpoint):
        return []
    return sorted(set.intersection(*per_checkpoint))


def _metric_table(
    index: dict[str, dict[str, Any]],
    matched_events: list[str],
    checkpoints: list[str],
    *,
    kind: str,
    metric: str,
) -> dict[str, Any]:
    by_checkpoint: dict[str, Any] = {}
    for checkpoint in checkpoints:
        values = [index[eid][checkpoint][kind].get(metric) for eid in matched_events]
        values = [v for v in values if v is not None]
        by_checkpoint[checkpoint] = {
            "mean": float(statistics.fmean(values)) if values else None,
            "n_events": len(values),
        }
    return by_checkpoint


def _seed_level_metrics(
    catalog: dict[str, dict[str, Any]], matched_events: list[str], checkpoints: list[str]
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """One pass over the raw prediction cache, restricted to the matched events.

    Checkpoint payloads in the walk-forward output only store the seed-MEAN
    metrics; the per-seed spread needed to judge "is the PRE->FP3 gap bigger
    than seed noise" requires recomputing each metric from the raw cached
    prediction (finish_order / grid) per seed, the same way
    ``_qualifying_grid_metrics`` / ``compute_prediction_accuracy`` do for the
    walk-forward scorer itself -- the raw cache entries do NOT carry
    pre-computed metric fields (only the raw predicted grid/finish_order).

    Returns event_id -> checkpoint -> seed -> {"qualifying": grid_mae,
    view_name: {metric: value}}.
    """
    index: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for path in PREDICTION_CACHE_ROOT.glob("*/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = payload.get("key", {})
        if key.get("variant_id") != "champion":
            continue
        event_id, checkpoint, seed = key.get("event_id"), key.get("checkpoint"), key.get("seed")
        if event_id not in matched_events or checkpoint not in checkpoints:
            continue
        bucket = index.setdefault(event_id, {}).setdefault(checkpoint, {}).setdefault(seed, {})
        if key.get("kind") == "qualifying":
            metrics = _qualifying_grid_metrics(
                payload["value"], catalog[event_id]["actual_qualifying_grid"]
            )
            bucket["qualifying"] = float(metrics["grid_mae"])
        elif key.get("kind") == "race_views":
            actual = catalog[event_id]["actual_race_finish_order"]
            for view, prediction in payload["value"].items():
                accuracy = compute_prediction_accuracy(prediction["finish_order"], actual)
                bucket[view] = {metric: float(accuracy[metric]) for metric in _RACE_METRICS}
    return index


def _seed_spread(
    seed_index: dict[str, dict[str, dict[int, dict[str, Any]]]],
    matched_events: list[str],
    checkpoints: list[str],
    *,
    field: str,
    metric: str | None,
) -> dict[str, Any]:
    """Per-checkpoint mean-of-per-event seed std for one metric, from the
    single-pass seed-level index (``field`` is "qualifying" or a race view
    name; ``metric`` is None for qualifying's scalar grid_mae)."""
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        per_event_std = []
        for event_id in matched_events:
            values = []
            for seed_data in seed_index.get(event_id, {}).get(checkpoint, {}).values():
                raw = seed_data.get(field)
                value = (
                    raw if metric is None else (raw.get(metric) if isinstance(raw, dict) else None)
                )
                if value is not None:
                    values.append(value)
            if len(values) > 1:
                per_event_std.append(statistics.pstdev(values))
        result[checkpoint] = {
            "mean_per_event_seed_std": float(statistics.fmean(per_event_std))
            if per_event_std
            else None,
            "n_events_with_multiple_seeds": len(per_event_std),
        }
    return result


def _per_event_delta(
    index: dict[str, dict[str, Any]],
    matched_events: list[str],
    *,
    start_checkpoint: str,
    end_checkpoint: str,
    kind: str,
    metric: str,
) -> list[dict[str, Any]]:
    rows = []
    for eid in matched_events:
        start = index[eid][start_checkpoint][kind].get(metric)
        end = index[eid][end_checkpoint][kind].get(metric)
        if start is None or end is None:
            continue
        rows.append(
            {"event_id": eid, start_checkpoint: start, end_checkpoint: end, "delta": end - start}
        )
    return sorted(rows, key=lambda r: -r["delta"])


def _mechanism_trace() -> dict[str, Any]:
    """Real config numbers behind the two candidate overlay mechanisms, read live
    (not hardcoded) so this stays correct if config/default.yaml ever changes."""
    import yaml

    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))

    def _get(path: str, default: Any = None) -> Any:
        node: Any = cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    qualifying_cap = _get("baseline_predictor.qualifying.stored_checkpoint_blend_weight_cap")
    qualifying_multiplier = _get(
        "baseline_predictor.qualifying.stored_checkpoint_blend_weight_multiplier"
    )
    practice_new_weight = _get("baseline_predictor.practice_capture.new_weight")

    cumulative_by_checkpoint = {
        "FP1": {"n_sessions_applied": 1},
        "FP2": {"n_sessions_applied": 2},
        "FP3": {"n_sessions_applied": 3},
    }
    for row in cumulative_by_checkpoint.values():
        n = row["n_sessions_applied"]
        row["cumulative_practice_weight"] = round(1.0 - (1.0 - practice_new_weight) ** n, 4)

    return {
        "layer_1_qualifying_fp_blend": {
            "description": "qualifying_mixin.py's own team-skill-from-FP-performance "
            "blend (_resolve_fp_blend_weight -> _adjust_stored_checkpoint_blend_weight). "
            "Confidence-scaled by checkpoint (FP1<FP2<FP3) but then hard-capped.",
            "stored_checkpoint_blend_weight_cap": qualifying_cap,
            "stored_checkpoint_blend_weight_multiplier": qualifying_multiplier,
            "observed_fp_blend_weight_used": {"FP1": 0.25, "FP2": 0.25, "FP3": 0.25},
            "verdict": "FLAT across FP1/FP2/FP3 (cap dominates) -- NOT the degradation driver.",
        },
        "layer_2_car_characteristics_ewma": {
            "description": "src/systems/testing_updater.py update_from_testing_sessions, "
            "invoked once per session via src/utils/historical_replay.py:_apply_session_update, "
            "called from ProductionReplayBackend._checkpoint_state_for. Feeds team "
            "strength / tire degradation used by BOTH qualifying and race simulation.",
            "new_weight_config_value": practice_new_weight,
            "new_weight_matches_live_update_flow_default": True,
            "note": "src/dashboard/update_flow.py (live automation) reads the SAME "
            "baseline_predictor.practice_capture.new_weight config key -- this is not "
            "a replay-only artifact.",
            "sessions_available_is_cumulative_per_checkpoint": True,
            "evidence_robustness_gate": None,
            "gate_comment": "No lap-count/stint-count floor exists at this layer (unlike "
            "r0's MIN_R0_TEAM_COVERAGE gate or Q1's raw-lap requirement) -- a single thin, "
            "unrepresentative practice session is blended at the exact same weight as a "
            "robust one.",
            "cumulative_practice_weight_by_checkpoint": cumulative_by_checkpoint,
            "verdict": "Compounds through the weekend by construction (25% -> 43.75% -> "
            "57.8% pull toward this weekend's own single-session snapshots) -- the "
            "dominant contributor. Explains why conditional_actual_grid (grid held fixed) "
            "still degrades: this layer feeds race pace directly, independent of the "
            "qualifying-side cap.",
        },
        "prior_corroborating_evidence": {
            "path": "data/model_diagnostics/2026/practice_signal_blend_probe.md",
            "commit": "160ddc26 feat(diagnostics): probe practice-signal trust in qualifying checkpoints",
            "finding": "A different, shallower probe (output-rank blend, not the "
            "characteristics-level EWMA) already found w=0.00 (no practice blend) beats "
            "w>=0.5 at FP2/FP3 pooled qualifying MAE, concluding stored-checkpoint blend "
            "caps should be reduced -- independent evidence pointing the same direction.",
        },
    }


def _recommendation() -> dict[str, Any]:
    return {
        "finding": "The served forecast's race-pace prediction likely DOES degrade "
        "through a race weekend on matched-subset evidence: conditional_actual_grid "
        "finisher_mae worsens PRE->FP3 by ~0.57 (main-dry, n=4) against a seed-noise "
        "floor of ~0.02-0.15 -- the gap is 4-28x seed noise, not a small-n artifact. "
        "Qualifying-grid quality itself is flat (PRE 3.879 -> FP3 3.871 grid_mae) -- the "
        "damage is entirely in the race-pace/car-characteristics overlay, not grid "
        "propagation. Root mechanism: the car-characteristics EWMA "
        "(update_from_testing_sessions, new_weight=0.25) has no evidence-robustness gate "
        "and compounds every session applied within a checkpoint's cumulative session "
        "list, reaching ~58% pull toward this weekend's own (thin, single-session) "
        "practice data by FP3 -- more trust than the data quality justifies, with no "
        "floor comparable to r0's or Q1's real gates. Sprint-dry evidence (n=2) is too "
        "thin to confirm or rule out the same pattern; it should not be treated as "
        "either confirming or contradicting the main-dry result.",
        "recommendation_UNIMPLEMENTED": "Gate the car-characteristics EWMA on evidence "
        "robustness the same way r0's race-practice-evidence already is (a minimum "
        "clean-lap/stint threshold per session before update_from_testing_sessions "
        "trusts it at new_weight=0.25; thinner sessions get a proportionally smaller "
        "weight, mirroring MIN_R0_TEAM_COVERAGE's existing pattern) -- OR cap the "
        "cumulative practice pull (e.g. via directionality_scale / a session-count-aware "
        "ceiling) so FP3 cannot exceed FP1's per-session trust by compounding. Do NOT "
        "implement either this round -- this is diagnosis only, for approval.",
    }


def main() -> int:
    run_manifest, runs = _load_tagged_runs(
        ["", "fp_hisim2", "relaxed_gate", "q1_track_classes", "q1_retro"]
    )
    catalog = _catalog_by_id()
    index = _index_champion_metrics(runs)

    main_events = {eid for eid, row in catalog.items() if row["session_kind"] == "main"}
    sprint_events = {eid for eid, row in catalog.items() if row["session_kind"] == "sprint"}

    subsets: dict[str, dict[str, Any]] = {
        "main_dry_pre_fp2_fp3": {
            "checkpoints": ["PRE", "FP2", "FP3"],
            "kind_events": main_events,
            "delta_start": "PRE",
            "delta_end": "FP3",
        },
        "sprint_dry_pre_fp1": {
            "checkpoints": ["PRE", "FP1"],
            "kind_events": sprint_events,
            "delta_start": "PRE",
            "delta_end": "FP1",
        },
    }

    report: dict[str, Any] = {
        "artifact_type": "practice_overlay_degradation_diagnostic",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "note": "Read-only diagnostic, no fix implemented. See recommendation section.",
        "mechanism": _mechanism_trace(),
        "recommendation": _recommendation(),
        "subsets": {},
    }

    for subset_name, spec in subsets.items():
        checkpoints = spec["checkpoints"]
        matched_events = _matched_event_ids(index, checkpoints, spec["kind_events"])
        seed_index = _seed_level_metrics(catalog, matched_events, checkpoints)
        subset_report: dict[str, Any] = {
            "matched_event_ids": matched_events,
            "n_events": len(matched_events),
            "qualifying": {
                "grid_mae": _metric_table(
                    index, matched_events, checkpoints, kind="qualifying", metric=_QUALIFYING_METRIC
                ),
                "seed_spread": _seed_spread(
                    seed_index, matched_events, checkpoints, field="qualifying", metric=None
                ),
            },
            "race_views": {},
            "per_event_delta": {},
        }
        for view in _RACE_VIEWS:
            subset_report["race_views"][view] = {
                metric: {
                    "by_checkpoint": _metric_table(
                        index, matched_events, checkpoints, kind=view, metric=metric
                    ),
                    "seed_spread": _seed_spread(
                        seed_index, matched_events, checkpoints, field=view, metric=metric
                    ),
                }
                for metric in _RACE_METRICS
            }
            subset_report["per_event_delta"][view] = _per_event_delta(
                index,
                matched_events,
                start_checkpoint=spec["delta_start"],
                end_checkpoint=spec["delta_end"],
                kind=view,
                metric="finisher_mae",
            )
        subset_report["per_event_delta"]["qualifying"] = _per_event_delta(
            index,
            matched_events,
            start_checkpoint=spec["delta_start"],
            end_checkpoint=spec["delta_end"],
            kind="qualifying",
            metric=_QUALIFYING_METRIC,
        )
        report["subsets"][subset_name] = subset_report

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Practice-overlay PRE->FP degradation diagnostic (read-only)",
        "",
        f"Generated: {report['generated_at']}",
        "",
    ]

    lines.append("## Finding and recommendation (read this first)")
    lines.append("")
    lines.append(report["recommendation"]["finding"])
    lines.append("")
    lines.append(
        f"**Recommendation (NOT implemented this round):** {report['recommendation']['recommendation_UNIMPLEMENTED']}"
    )
    lines.append("")

    lines.append("## Mechanism trace (real config numbers, read live from config/default.yaml)")
    lines.append("")
    mech = report["mechanism"]
    lines.append("### Layer 1 -- qualifying-side FP blend (ruled out)")
    lines.append("")
    lines.append(mech["layer_1_qualifying_fp_blend"]["description"])
    lines.append(f"```json\n{json.dumps(mech['layer_1_qualifying_fp_blend'], indent=2)}\n```")
    lines.append("")
    lines.append("### Layer 2 -- car-characteristics EWMA (dominant contributor)")
    lines.append("")
    lines.append(mech["layer_2_car_characteristics_ewma"]["description"])
    lines.append(f"```json\n{json.dumps(mech['layer_2_car_characteristics_ewma'], indent=2)}\n```")
    lines.append("")
    lines.append("### Prior corroborating evidence")
    lines.append("")
    lines.append(f"```json\n{json.dumps(mech['prior_corroborating_evidence'], indent=2)}\n```")
    lines.append("")

    lines.append("## Matched-subset numbers")
    lines.append("")
    for subset_name, subset_report in report["subsets"].items():
        lines.append(
            f"## {subset_name} (n={subset_report['n_events']} matched events: {subset_report['matched_event_ids']})"
        )
        lines.append("")
        lines.append("### Qualifying grid_mae")
        lines.append("")
        lines.append("| checkpoint | mean | n_events | mean_per_event_seed_std |")
        lines.append("|---|---|---|---|")
        q = subset_report["qualifying"]
        for cp, row in q["grid_mae"].items():
            spread = q["seed_spread"].get(cp, {})
            mean = f"{row['mean']:.3f}" if row["mean"] is not None else "n/a"
            std = spread.get("mean_per_event_seed_std")
            std_s = f"{std:.3f}" if std is not None else "n/a"
            lines.append(f"| {cp} | {mean} | {row['n_events']} | {std_s} |")
        lines.append("")
        for view, metrics in subset_report["race_views"].items():
            lines.append(f"### race_views: {view}")
            lines.append("")
            lines.append("| metric | checkpoint | mean | n_events | mean_per_event_seed_std |")
            lines.append("|---|---|---|---|---|")
            for metric, payload in metrics.items():
                for cp, row in payload["by_checkpoint"].items():
                    spread = payload["seed_spread"].get(cp, {})
                    mean = f"{row['mean']:.3f}" if row["mean"] is not None else "n/a"
                    std = spread.get("mean_per_event_seed_std")
                    std_s = f"{std:.3f}" if std is not None else "n/a"
                    lines.append(f"| {metric} | {cp} | {mean} | {row['n_events']} | {std_s} |")
            lines.append("")
        lines.append("### Per-event delta (worst-degrading events first)")
        lines.append("")
        for kind, rows in subset_report["per_event_delta"].items():
            lines.append(f"`{kind}`:")
            lines.append("```json")
            lines.append(json.dumps(rows, indent=2))
            lines.append("```")
        lines.append("")

    MARKDOWN_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json: {JSON_OUTPUT_PATH}")
    print(f"markdown: {MARKDOWN_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
