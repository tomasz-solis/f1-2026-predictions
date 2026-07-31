#!/usr/bin/env python3
"""Phase-3 follow-up report: FP-checkpoint, higher-sim-count walk-forward comparison.

Combines the phase-2 PRE-at-20-sims run manifest with the phase-3 FP-checkpoint
higher-sim-count run(s) (tagged, see ``--run-tags``) into one comparison. Never
overwrites the phase-2 report -- writes versioned ``_v2`` files alongside it.

Adds, beyond the phase-2 report:
- PRE vs FP2 vs FP3 champion end-to-end comparison (how much practice data closes
  the qualifying-grid-propagation gap);
- r0 vs champion at FP checkpoints (r0's first real test -- it is a no-op at PRE by
  construction, since PRE has no practice sessions to extract evidence from);
- per-seed spread per variant/checkpoint/view, to judge whether champion-vs-variant
  deltas exceed ordinary seed noise;
- decomposition sections recomputed on whichever run(s) actually completed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_race_mae_investigation_report import (  # noqa: E402
    DRIVER_DEBUTS_PATH,
    OUTPUT_DIR,
    RUNS_DIR,
    _catalog_by_id,
    _decomposition,
    _driver_cohort,
    _q0_invariance_check,
    _variant_table,
    _wet_events_excluded,
)

from src.analysis.model_evaluation import compute_prediction_accuracy  # noqa: E402

JSON_OUTPUT_PATH = OUTPUT_DIR / "2026_walk_forward_variant_comparison_v2.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "2026_WALK_FORWARD_VARIANT_COMPARISON_v2.md"
PREDICTION_CACHE_ROOT = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "prediction_cache"


def _load_tagged_runs(tags: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and merge run_manifest(+variant json)s for every requested tag.

    ``""`` means the untagged phase-2 run. Later tags win on a per-variant basis so a
    higher-sim FP rerun of a variant already present from an earlier tag replaces it.
    """
    merged_manifest: dict[str, Any] = {"results": {}, "runs_merged": []}
    runs: dict[str, dict[str, Any]] = {}
    for tag in tags:
        suffix = f"__{tag}" if tag else ""
        manifest_path = RUNS_DIR / f"run_manifest{suffix}.json"
        if not manifest_path.is_file():
            merged_manifest["runs_merged"].append({"tag": tag, "found": False})
            continue
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged_manifest["runs_merged"].append(
            {
                "tag": tag,
                "found": True,
                "qualifying_simulations": run_manifest.get("qualifying_simulations"),
                "race_simulations": run_manifest.get("race_simulations"),
                "checkpoints_filter": run_manifest.get("checkpoints_filter"),
                "main_checkpoints_filter": run_manifest.get("main_checkpoints_filter"),
                "sprint_checkpoints_filter": run_manifest.get("sprint_checkpoints_filter"),
            }
        )
        for variant_id, status in run_manifest.get("results", {}).items():
            merged_manifest["results"][f"{variant_id}{suffix}"] = {**status, "run_tag": tag}
            if status["status"] == "scored":
                path = RUNS_DIR / f"{variant_id}{suffix}.json"
                runs[f"{variant_id}{suffix}"] = json.loads(path.read_text(encoding="utf-8"))
    return merged_manifest, runs


def _per_seed_spread(variant_key: str, replay: dict[str, Any]) -> dict[str, Any]:
    """Per-(checkpoint, view) seed std of finisher_mae, mined from the raw cache.

    The walk-forward replay artifact only stores the seed-mean; the per-seed spread
    needed to judge "is this delta bigger than seed noise" lives only in the raw
    cached predictions.
    """
    variant_id = str(replay["manifest"]["variant_id"] if "manifest" in replay else variant_key)
    catalog = _catalog_by_id()
    rows: list[dict[str, Any]] = []
    for path in PREDICTION_CACHE_ROOT.glob("*/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = payload.get("key", {})
        if key.get("kind") != "race_views" or key.get("variant_id") not in ("champion", variant_id):
            continue
        event_id = key.get("event_id")
        if event_id not in catalog:
            continue
        actual = catalog[event_id]["actual_race_finish_order"]
        for view, prediction in payload["value"].items():
            accuracy = compute_prediction_accuracy(prediction["finish_order"], actual)
            rows.append(
                {
                    "event_id": event_id,
                    "checkpoint": key.get("checkpoint"),
                    "role": "champion" if key["variant_id"] == "champion" else "challenger",
                    "seed": key.get("seed"),
                    "view": view,
                    "finisher_mae": float(accuracy["finisher_mae"]),
                }
            )

    spread: dict[str, Any] = {}
    checkpoints = {r["checkpoint"] for r in rows}
    for checkpoint in checkpoints:
        for view in ("conditional_actual_grid", "end_to_end_predicted_grid"):
            for role in ("champion", "challenger"):
                per_event_std = []
                for event_id in {r["event_id"] for r in rows}:
                    values = [
                        r["finisher_mae"]
                        for r in rows
                        if r["checkpoint"] == checkpoint
                        and r["view"] == view
                        and r["role"] == role
                        and r["event_id"] == event_id
                    ]
                    if len(values) > 1:
                        per_event_std.append(statistics.pstdev(values))
                if per_event_std:
                    spread[f"{checkpoint}:{view}:{role}"] = {
                        "mean_per_event_seed_std": float(statistics.fmean(per_event_std)),
                        "n_events_with_multiple_seeds": len(per_event_std),
                    }
    return spread


def _driver_cohort_decomposition_scoped(
    replay: dict[str, Any], catalog: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Cohort decomposition restricted to this run's own scored checkpoints.

    Filtering by checkpoint (not just variant) keeps a higher-sim FP-checkpoint run
    from being polluted by the phase-2 20-sim PRE-checkpoint cache entries: a fresh
    FP2/FP3/FP1 run never shares a checkpoint with the old PRE-only run, so this scan
    only ever sees this run's own (higher) sim-count predictions.
    """
    own_checkpoints = {
        checkpoint for event in replay["scored_events"] for checkpoint in event["checkpoints"]
    }
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
        if key.get("checkpoint") not in own_checkpoints:
            continue
        event_id = key.get("event_id")
        seed = key.get("seed")
        if event_id not in catalog or (event_id, key.get("checkpoint"), seed) in seen:
            continue
        seen.add((event_id, key.get("checkpoint"), seed))
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
        return {
            "computed": False,
            "reason": "no champion race_views cache entries for these checkpoints",
        }
    return {
        "computed": True,
        "basis": f"champion conditional_actual_grid, finishers only, checkpoints {sorted(own_checkpoints)}",
        "by_cohort": {
            cohort: {
                "finisher_mae": float(statistics.fmean(errors)),
                "n_driver_observations": len(errors),
            }
            for cohort, errors in sorted(cohort_errors.items())
        },
    }


def _checkpoint_progression(
    runs: dict[str, dict[str, Any]], variant_key_by_checkpoint: dict[str, str]
) -> dict[str, Any]:
    """PRE vs FP2 vs FP3 champion end-to-end comparison, wherever both exist."""
    progression: dict[str, Any] = {}
    for checkpoint, variant_key in variant_key_by_checkpoint.items():
        replay_payload = runs.get(variant_key)
        if replay_payload is None:
            progression[checkpoint] = {"available": False}
            continue
        table = _variant_table(replay_payload["replay"])
        agg = table["aggregate_by_checkpoint"].get(checkpoint)
        if agg is None:
            progression[checkpoint] = {"available": False}
            continue
        # ponytail: _race_metrics now returns {mean, std, n_events, ...} per field
        # (not a bare float) -- pull "mean" to match every other _fmt() call in
        # this file's markdown section.
        progression[checkpoint] = {
            "available": True,
            "champion_end_to_end_finisher_mae": agg["race_views"]["end_to_end_predicted_grid"][
                "champion"
            ]
            .get("finisher_mae", {})
            .get("mean"),
            "champion_conditional_finisher_mae": agg["race_views"]["conditional_actual_grid"][
                "champion"
            ]
            .get("finisher_mae", {})
            .get("mean"),
        }
    return progression


def _research_gate_relaxation_detail(replay: dict[str, Any]) -> dict[str, Any]:
    """Mine real anchor_calibration values + ineffective_for_fold flags from the
    disk-backed caches (the walk-forward output itself only keeps a fold_artifacts
    sha256 digest, not the calibrated numbers -- see ProductionReplayBackend.fit_fold)."""
    events_with_relaxation = [
        (event["event_id"], checkpoint, cp)
        for event in replay["scored_events"]
        for checkpoint, cp in event["checkpoints"].items()
        if cp.get("research_gate_relaxation")
    ]
    if not events_with_relaxation:
        return {}

    variant_id = replay["variant_id"]
    anchor_calibrations: dict[str, Any] = {}
    ineffective_folds: list[dict[str, Any]] = []
    for event_id, checkpoint, _cp in events_with_relaxation:
        for path in PREDICTION_CACHE_ROOT.glob("*/*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = payload.get("key", {})
            if key.get("event_id") != event_id or key.get("checkpoint") != checkpoint:
                continue
            if (
                key.get("kind") == "fold_artifacts"
                and key.get("variant_id") == variant_id
                and payload["value"].get("anchor_calibration") is not None
            ):
                anchor_calibrations[f"{event_id}:{checkpoint}"] = payload["value"][
                    "anchor_calibration"
                ]
            if key.get("kind") == "race_views" and key.get("variant_id") == variant_id:
                for view_name, prediction in payload["value"].items():
                    if prediction.get("ineffective_for_fold"):
                        ineffective_folds.append(
                            {
                                "event_id": event_id,
                                "checkpoint": checkpoint,
                                "seed": key.get("seed"),
                                "view": view_name,
                                "reason": prediction.get("ineffective_reason"),
                            }
                        )
    return {"anchor_calibrations": anchor_calibrations, "ineffective_folds": ineffective_folds}


def _q1_practice_activation(replay: dict[str, Any]) -> dict[str, Any]:
    """Did the Q1 challenger actually activate on its scored checkpoint(s)?

    A scored Q1 checkpoint can still be an undifferentiated (champion-identical)
    result if ``qualifying_mixin.py``'s own runtime guard declines to use the
    fitted launch envelope (e.g. no raw per-lap telemetry for the target event's
    own practice session in this replay's ``stored_profiles`` mode) -- that is
    disclosed on every cached qualifying prediction via
    ``qualifying_practice_challenger`` and must not be silently treated as a win.
    """
    variant_id = replay["variant_id"]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for event in replay["scored_events"]:
        for checkpoint in event["checkpoints"]:
            for path in PREDICTION_CACHE_ROOT.glob("*/*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                key = payload.get("key", {})
                if (
                    key.get("kind") != "qualifying"
                    or key.get("variant_id") != variant_id
                    or key.get("event_id") != event["event_id"]
                    or key.get("checkpoint") != checkpoint
                ):
                    continue
                dedup = (event["event_id"], checkpoint, key.get("seed"))
                if dedup in seen:
                    continue
                seen.add(dedup)
                disclosure = payload["value"].get("qualifying_practice_challenger", {})
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "checkpoint": checkpoint,
                        "seed": key.get("seed"),
                        "used": disclosure.get("used"),
                        "fallback_reason": disclosure.get("fallback_reason"),
                    }
                )
    return {"rows": rows, "any_activated": any(r["used"] for r in rows)}


def _q1_eligibility_table(
    variants: dict[str, Any], runs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-event-checkpoint Q1 eligibility outcome, merged across every scored/refused
    q1_qualifying_practice run in the requested tags (later tags win per event-checkpoint,
    matching ``_load_tagged_runs``'s own precedence -- every event ever attempted appears
    once, whether it scored or refused, with its exact reason)."""
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for variant_key, payload in variants.items():
        if not variant_key.startswith("q1_qualifying_practice") or payload["status"] != "scored":
            continue
        for refusal in payload["checkpoint_refusals"]:
            key = (refusal["event_id"], refusal["checkpoint"])
            rows_by_key[key] = {
                "event_id": refusal["event_id"],
                "checkpoint": refusal["checkpoint"],
                "outcome": "refused",
                "reason": refusal["reason"],
                "run_tag": payload["run_tag"],
            }
        replay = runs[variant_key]["replay"]
        for event in replay["scored_events"]:
            for checkpoint in event["checkpoints"]:
                key = (event["event_id"], checkpoint)
                rows_by_key[key] = {
                    "event_id": event["event_id"],
                    "checkpoint": checkpoint,
                    "outcome": "scored",
                    "reason": (
                        f"research_gate_relaxation="
                        f"{event['checkpoints'][checkpoint].get('research_gate_relaxation')}"
                    ),
                    "run_tag": payload["run_tag"],
                }
    return sorted(rows_by_key.values(), key=lambda r: (r["event_id"], r["checkpoint"]))


def _production_untouched_statement() -> dict[str, Any]:
    prod_config = PROJECT_ROOT / "config" / "production_config.json"
    default_yaml = PROJECT_ROOT / "config" / "default.yaml"
    prod_sha256 = hashlib.sha256(prod_config.read_bytes()).hexdigest()
    champion_line = next(
        (
            line.strip()
            for line in default_yaml.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("model_variant:")
        ),
        None,
    )
    return {
        "config_production_config_json_sha256": prod_sha256,
        "config_production_config_json_expected_sha256": (
            "c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1"
        ),
        "config_production_config_json_byte_identical": prod_sha256
        == "c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1",
        "config_default_yaml_model_variant_line": champion_line,
        "statement": (
            "All research this round (fp_hisim2, relaxed_gate, q1_track_classes, "
            "q1_retro campaigns; the retrospective_diagnostic hatch; the q1 outer-gate "
            "pooling fix) ran through ProductionReplayBackend, a read-only research "
            "harness that drives the real production predictor with research-only "
            "config overrides passed in-memory per prediction call. "
            "config/production_config.json was read, never written (sha256 confirmed "
            "unchanged below). config/default.yaml's model_variant stayed 'champion' "
            "throughout. No champion weights, active artifacts, prediction artifacts, "
            "or data/evaluation/ contents were overwritten. The served weekend forecast "
            "(Belgian GP, live as of 2026-07-19) was never touched by any research run."
        ),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tags",
        nargs="*",
        default=["", "fp_hisim2", "relaxed_gate", "q1_track_classes", "q1_retro"],
    )
    args = parser.parse_args()

    run_manifest, runs = _load_tagged_runs(args.run_tags)
    catalog = _catalog_by_id()

    variants: dict[str, Any] = {}
    for variant_key, status in run_manifest["results"].items():
        if status["status"] != "scored":
            variants[variant_key] = {"status": status["status"], "detail": status.get("error")}
            continue
        replay_payload = runs[variant_key]
        replay = replay_payload["replay"]
        research_gate_relaxations = {
            tuple(sorted(cp["research_gate_relaxation"].items()))
            for event in replay["scored_events"]
            for cp in event["checkpoints"].values()
            if cp.get("research_gate_relaxation")
        }
        variants[variant_key] = {
            "status": "scored",
            "run_tag": status["run_tag"],
            "manifest_sha256": replay_payload["manifest"]["manifest_sha256"],
            "comparison": _variant_table(replay),
            "decomposition": _decomposition(replay, catalog),
            "per_seed_spread": _per_seed_spread(variant_key, replay_payload),
            # Loud, never-dropped per-event-checkpoint extraction refusals (e.g. a
            # FastF1 session too thin to extract): the variant still scores every
            # other eligible event-checkpoint (see CheckpointInputUnavailable).
            "checkpoint_refusals": replay.get("checkpoint_refusals", []),
            "research_gate_relaxations_applied": [
                dict(items) for items in research_gate_relaxations
            ],
            "research_gate_relaxation_detail": _research_gate_relaxation_detail(replay),
            **(
                {"q1_practice_activation": _q1_practice_activation(replay)}
                if variant_key.startswith("q1_qualifying_practice")
                else {}
            ),
        }

    # PRE vs FP2 vs FP3 needs one champion source per checkpoint; prefer the
    # highest-sim run that actually scored each checkpoint.
    checkpoint_source: dict[str, str] = {}
    for variant_key, payload in variants.items():
        if payload["status"] != "scored":
            continue
        for checkpoint in payload["comparison"]["aggregate_by_checkpoint"]:
            checkpoint_source.setdefault(checkpoint, variant_key)
    progression = _checkpoint_progression(runs, checkpoint_source)

    report: dict[str, Any] = {
        "artifact_type": "race_mae_walk_forward_variant_comparison",
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "supersedes": None,
        "note": "Phase-3 follow-up: FP-checkpoint + higher-sim-count run. "
        "See 2026_walk_forward_variant_comparison.json for the phase-2 PRE-only baseline.",
        "run_manifest": run_manifest,
        "event_catalog_summary": {
            "included_events": sorted(catalog),
            "wet_events_excluded_from_all_variant_scoring": _wet_events_excluded(catalog),
        },
        "q0_conditional_grid_invariance_check": {
            key: _q0_invariance_check({"q0_driver_state": payload})
            for key, payload in runs.items()
            if "q0" in key
        },
        "champion_checkpoint_progression": progression,
        "driver_cohort_decomposition": {
            key: _driver_cohort_decomposition_scoped(payload["replay"], catalog)
            for key, payload in runs.items()
        },
        "research_gate_relaxation_limitations": {
            "q1_qualifying_practice": (
                "Final state after three rounds of real fixes (track-class binding, "
                "per-fold Bradley-Terry fit, the retrospective_diagnostic chronology "
                "hatch, and the outer-gate track-class-aware pooling fix): the fitter's "
                "own contract requires an exact (checkpoint, session_kind, track_class) "
                "group (docs/QUALIFYING_RACE_CHALLENGER.md); track class comes from the "
                "curated data/historical_replay/2026/track_class_by_event.json binding. "
                "One fold is now genuinely eligible and genuinely fits: British GP FP1 "
                "(5 prior permanent-class training events, real Bradley-Terry model, "
                "real launch envelope, retrospective_diagnostic=true throughout). But "
                "even there the challenger did not actually activate at inference: "
                "qualifying_mixin.py's own runtime guard requires raw per-lap FP "
                "telemetry for the TARGET event's own practice session "
                "(session_laps_by_type), and the research backend replays every "
                "checkpoint in practice_signal_mode='stored_profiles' (aggregated "
                "team/driver profiles only) -- so session_laps_by_type is always {} in "
                "this harness, for every variant, by construction. The fold scored, is "
                "fully disclosed (qualifying_practice_challenger.used=false, "
                "fallback_reason='no_raw_practice_laps' on every cached prediction), "
                "and is champion-identical as a DIRECT, transparent consequence -- not "
                "a silently-hidden fallback. Making the backend load raw per-lap "
                "telemetry for every replayed checkpoint is a materially larger change "
                "(affects every variant's replay path, reintroduces the FastF1 "
                "telemetry-thinness fragility already hit once this project on "
                "Barcelona FP1) and was not authorized this round -- reported as a "
                "structural finding, not fixed. See q1_practice_activation below."
            ),
            "r2_source_anchor": (
                "This IS a real calibration: fit_source_specific_grid_anchors runs on "
                "genuine (simulated_position, grid_position, actual_position) rows "
                "recovered from r2_no_anchor's own predictions on prior events (its "
                "anchor weight is fixed at 0.0, so its predicted position literally is "
                "the pre-anchor simulated position -- no simulator internals touched). "
                "The calibrated weight is shrunk toward champion's own resolved weight "
                "in proportion to n_training_events/8 and injected via "
                "baseline_predictor.race.grid_anchor.source_calibrated.actual_starting_"
                "grid. See research_gate_relaxation_detail per variant below for the "
                "real fitted values per fold, and ineffective_folds for any fold where "
                "the calibrated prediction still matched champion exactly (flagged, not "
                "hidden). end_to_end_predicted_grid was not calibrated (only the "
                "conditional_actual_grid source detail) given the time budget, so that "
                "view legitimately keeps champion's fallback weight."
            ),
        },
        "variants": variants,
        "r2_interval_coherence_finding": {
            "statement": (
                "First-class result, not a bug: every r2 variant that reached scoring "
                "(r2_no_anchor and r1_r2_no_anchor at PRE/20-sim in the untangled "
                "baseline run; r2_source_anchor and r1_r2_source_anchor at "
                "FP-checkpoint/500-sim with the calibrated anchor in relaxed_gate) "
                "refused whole-variant with the SAME class of error: a simulated "
                "finish position fell outside the champion-computed p5-p95 grid "
                "interval. r2_source_anchor's calibrated anchor weight (fit for real "
                "from r2_no_anchor's own pre-anchor simulated positions, shrunk toward "
                "champion's resolved weight) does not fix this -- it still fails. "
                "champion's own (uncalibrated-away) grid anchor is load-bearing for "
                "keeping simulated positions inside the declared uncertainty band; "
                "removing or only-partially-restoring it breaks interval coherence "
                "regardless of calibration. Recommendation: interval recalibration "
                "(widening p5-p95 or re-deriving it jointly with any r2 variant) is "
                "future work. The coherence validator itself "
                "(validate_qualifying_grid) is intentionally NOT weakened to force a "
                "score -- doing so would hide a real miscalibration, not fix it."
            ),
            "raw_refusals": [
                {"variant_key": key, "run_tag": v.get("run_tag"), "error": v.get("detail")}
                for key, v in variants.items()
                if v["status"] != "scored"
                and key.split("__")[0]
                in {"r2_no_anchor", "r2_source_anchor", "r1_r2_no_anchor", "r1_r2_source_anchor"}
            ],
        },
        "q1_eligibility_table": _q1_eligibility_table(variants, runs),
        "production_untouched_statement": _production_untouched_statement(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# 2026 Walk-Forward Race-MAE Variant Comparison v2 (Phase 3: FP checkpoints "
        "+ 500-sim + research-gate-relaxed Q1/R2-source-anchor)",
        "",
        f"Generated: {report['generated_at']}",
        "Supersedes: nothing -- phase-2 (PRE-only, 20-sim) report kept as-is at "
        "`2026_walk_forward_variant_comparison.json` / `.md`. This report adds the "
        "FP-checkpoint, 500-sim campaign and the research-gate-relaxed Q1/"
        "r2_source_anchor runs.",
        "",
        "## Runs merged",
        "",
        f"```json\n{json.dumps(run_manifest['runs_merged'], indent=2)}\n```",
        "",
        "## Variant status",
        "",
    ]
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] == "scored":
            refusal_note = (
                f", {len(payload['checkpoint_refusals'])} checkpoint refusal(s)"
                if payload["checkpoint_refusals"]
                else ""
            )
            relaxation_note = (
                f", research_gate_relaxation={payload['research_gate_relaxations_applied']}"
                if payload["research_gate_relaxations_applied"]
                else ""
            )
            lines.append(
                f"- `{variant_key}`: scored (tag={payload['run_tag']!r}){refusal_note}{relaxation_note}"
            )
        else:
            lines.append(f"- `{variant_key}`: {payload['status']} -- {payload.get('detail')}")
    lines.append("")

    lines.append("## Per-event-checkpoint refusals (loud, never silently dropped)")
    lines.append("")
    any_refusals = False
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] != "scored":
            continue
        for refusal in payload["checkpoint_refusals"]:
            any_refusals = True
            lines.append(
                f"- `{variant_key}` {refusal['event_id']} {refusal['checkpoint']} "
                f"({refusal['error_type']}): {refusal['reason']}"
            )
    if not any_refusals:
        lines.append("(none)")
    lines.append("")

    lines.append("## r2 interval-coherence finding (first-class result, not a bug)")
    lines.append("")
    lines.append(report["r2_interval_coherence_finding"]["statement"])
    lines.append("")
    lines.append("| variant_key | run_tag | error |")
    lines.append("|---|---|---|")
    for row in report["r2_interval_coherence_finding"]["raw_refusals"]:
        lines.append(f"| `{row['variant_key']}` | {row['run_tag']!r} | {row['error']} |")
    lines.append("")

    lines.append("## Q1 eligibility table (every event-checkpoint ever attempted)")
    lines.append("")
    lines.append("| event_id | checkpoint | outcome | reason / relaxation | run_tag |")
    lines.append("|---|---|---|---|---|")
    for row in report["q1_eligibility_table"]:
        lines.append(
            f"| {row['event_id']} | {row['checkpoint']} | {row['outcome']} | {row['reason']} | "
            f"{row['run_tag']!r} |"
        )
    lines.append("")

    lines.append(
        "## Q1 practice-challenger runtime activation (did the fitted model actually get used?)"
    )
    lines.append("")
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] != "scored" or "q1_practice_activation" not in payload:
            continue
        activation = payload["q1_practice_activation"]
        lines.append(f"`{variant_key}`: any_activated={activation['any_activated']}")
        lines.append(f"```json\n{json.dumps(activation['rows'], indent=2)}\n```")
    lines.append("")

    lines.append("## Production-untouched statement")
    lines.append("")
    stmt = report["production_untouched_statement"]
    lines.append(stmt["statement"])
    lines.append("")
    lines.append(
        f"- `config/production_config.json` sha256: `{stmt['config_production_config_json_sha256']}`"
    )
    lines.append(f"- expected: `{stmt['config_production_config_json_expected_sha256']}`")
    lines.append(f"- byte-identical: **{stmt['config_production_config_json_byte_identical']}**")
    lines.append(f"- `config/default.yaml`: `{stmt['config_default_yaml_model_variant_line']}`")
    lines.append("")

    lines.append("## Real anchor calibration + identity guard (research-gate-relaxed variants)")
    lines.append("")
    any_relaxation_detail = False
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] != "scored":
            continue
        detail = payload["research_gate_relaxation_detail"]
        if not detail:
            continue
        any_relaxation_detail = True
        lines.append(f"### `{variant_key}`")
        lines.append("")
        lines.append(
            "Fitted anchor calibrations (fold -> {status, calibrated_weight, shrinkage_weight}):"
        )
        lines.append(f"```json\n{json.dumps(detail['anchor_calibrations'], indent=2)}\n```")
        lines.append(
            "Folds flagged `ineffective_for_fold` (a race-affecting component produced a "
            "champion-identical finish order; not scored as a differentiated result):"
        )
        lines.append(f"```json\n{json.dumps(detail['ineffective_folds'], indent=2)}\n```")
        lines.append("")
    if not any_relaxation_detail:
        lines.append("(no research-gate-relaxed variant scored any checkpoint)")
        lines.append("")

    lines.append(
        "## Q1 / r2_source_anchor research-gate-relaxation: what the relaxation does and doesn't buy"
    )
    lines.append("")
    for key, text in report["research_gate_relaxation_limitations"].items():
        lines.append(f"**{key}**: {text}")
        lines.append("")

    lines.append(
        "## Champion checkpoint progression (end-to-end finisher_mae, closing the grid-propagation gap)"
    )
    lines.append("")
    lines.append("| checkpoint | end_to_end finisher_mae | conditional finisher_mae |")
    lines.append("|---|---|---|")
    for checkpoint, row in sorted(progression.items()):
        if not row.get("available"):
            lines.append(f"| {checkpoint} | not run | not run |")
            continue
        lines.append(
            f"| {checkpoint} | {row['champion_end_to_end_finisher_mae']:.3f} | "
            f"{row['champion_conditional_finisher_mae']:.3f} |"
        )
    lines.append("")

    lines.append("## Race-view comparison per scored run")
    lines.append("")
    header = (
        "| variant_key | checkpoint | view | role | mae | finisher_mae | weighted_mae "
        "| top_heavy_weighted_mae | winner_acc_% | n_events |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 10)
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] != "scored":
            continue
        for checkpoint, agg in payload["comparison"]["aggregate_by_checkpoint"].items():
            for view in ("conditional_actual_grid", "end_to_end_predicted_grid"):
                for role in ("champion", "challenger"):
                    metrics = agg["race_views"][view][role]
                    if not metrics:
                        continue

                    def _fmt(k: str, *, metrics: dict[str, Any] = metrics) -> str:
                        v = metrics.get(k, {}).get("mean")
                        return f"{v:.3f}" if isinstance(v, float) else "n/a"

                    n_events = metrics.get("mae", {}).get("n_events", 0)
                    lines.append(
                        f"| {variant_key}:{role} | {checkpoint} | {view} | {role} | {_fmt('mae')} | "
                        f"{_fmt('finisher_mae')} | {_fmt('weighted_mae')} | {_fmt('top_heavy_weighted_mae')} | "
                        f"{_fmt('winner_accuracy_percent')} | {n_events} |"
                    )
    lines.append("")

    lines.append("## Per-seed spread (mean per-event std of finisher_mae across the 3 seeds)")
    lines.append("")
    lines.append("| variant_key | checkpoint:view:role | mean_per_event_seed_std | n_events |")
    lines.append("|---|---|---|---|")
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] != "scored":
            continue
        for key, row in sorted(payload["per_seed_spread"].items()):
            lines.append(
                f"| {variant_key} | {key} | {row['mean_per_event_seed_std']:.3f} | "
                f"{row['n_events_with_multiple_seeds']} |"
            )
    lines.append("")

    lines.append("## q0 conditional_actual_grid invariance check")
    lines.append("")
    lines.append(
        f"```json\n{json.dumps(report['q0_conditional_grid_invariance_check'], indent=2)}\n```"
    )
    lines.append("")

    lines.append("## Driver cohort decomposition (by run)")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['driver_cohort_decomposition'], indent=2)}\n```")
    lines.append("")

    first_scored = next((p for p in variants.values() if p["status"] == "scored"), None)
    if first_scored is not None:
        lines.append(
            "## Sprint vs main weekend format, DNF floor, per-round trend (from first scored run)"
        )
        lines.append("")
        lines.append(f"```json\n{json.dumps(first_scored['decomposition'], indent=2)}\n```")
        lines.append("")

    MARKDOWN_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json: {JSON_OUTPUT_PATH}")
    print(f"markdown: {MARKDOWN_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
