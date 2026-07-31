#!/usr/bin/env python3
"""Phase-4 follow-up report: generalized identity guard + matched-subset progression.

Two truth-of-the-record fixes on top of v2 (Fable review findings), both computed
retroactively from the existing prediction cache -- no re-simulation:

1. Generalized structural-identity guard: ANY challenger prediction (qualifying
   grid or either race view) that is byte-identical to champion at a matched
   (event, checkpoint, seed) is flagged, not silently presented as a scored,
   differentiated result. q0's and r1's conditional_actual_grid invariance is
   architectural (that view discards the predicted grid, so a grid/qualifying-only
   component -- q0, q1, r1 -- has nothing to act on) and is kept distinguished
   from pathological identity (a component that SHOULD differ, like r0's
   race-only long-run pace, but didn't activate in this harness).
2. Matched-subset checkpoint progression: PRE vs FP2 vs FP3 restricted to the
   identical main-dry event subset, and PRE vs FP1 restricted to the identical
   sprint-dry event subset, instead of comparing checkpoints scored over
   different event counts.

Never overwrites v2 -- writes versioned ``_v3`` files alongside it.
"""

from __future__ import annotations

import glob
import json
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
    _checkpoint_progression,
    _decomposition,
    _driver_cohort_decomposition_scoped,
    _load_tagged_runs,
    _per_seed_spread,
    _production_untouched_statement,
    _q0_invariance_check,
    _q1_eligibility_table,
    _q1_practice_activation,
    _research_gate_relaxation_detail,
    _variant_table,
    _wet_events_excluded,
)

from src.models.challenger_variants import VARIANT_COMPONENTS  # noqa: E402

JSON_OUTPUT_PATH = OUTPUT_DIR / "2026_walk_forward_variant_comparison_v3.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "2026_WALK_FORWARD_VARIANT_COMPARISON_v3.md"
RAW_LAPS_HANDOFF_PATH = "docs/RAW_LAPS_REPLAY_HANDOFF.md"

# ponytail: grid/qualifying-only components have nothing to act on once a race
# view conditions on the REAL grid (conditional_actual_grid discards whatever
# the predicted grid would have been) -- their identity there is architectural,
# not a sign the component failed to activate. Any component outside this set
# (r0's race-only long-run pace, r2's anchor calibration) is expected to differ
# in BOTH race views regardless of grid source; identity there is pathological.
GRID_ONLY_COMPONENTS = frozenset({"q0", "q1", "r1"})

_RACE_VIEWS = ("conditional_actual_grid", "end_to_end_predicted_grid")


def _base_variant_id(variant_key: str) -> str:
    return variant_key.split("__", 1)[0]


def _finish_order(prediction: dict[str, Any]) -> list[str]:
    rows = prediction.get("grid") or prediction.get("finish_order") or []
    return [row.get("driver") for row in rows]


def _disclosed_fallback_reason(
    challenger: dict[str, Any], components: frozenset[str]
) -> str | None:
    """Best-effort reason mining from the known per-component disclosure fields.

    Both disclosure fields are present on EVERY prediction regardless of variant
    (e.g. q0_driver_state's payload still carries `qualifying_practice_challenger`
    with `used: false, fallback_reason: None` simply because that variant never
    requested q1) -- only mine the field that belongs to a component this variant
    actually claims, or the reason is a red herring ("q1 wasn't requested") rather
    than an explanation for THIS variant's own identity.
    """
    # ponytail: field name <-> owning component is a fixed, small mapping; add a
    # pair here if a third practice-evidence-gated component shows up.
    for field, owning_component in (
        ("qualifying_practice_challenger", "q1"),
        ("race_practice_challenger", "r0"),
    ):
        if owning_component not in components:
            continue
        disclosure = challenger.get(field)
        if isinstance(disclosure, dict) and disclosure.get("used") is False:
            return disclosure.get("fallback_reason")
        if isinstance(disclosure, dict) and disclosure.get("applied") is False:
            return disclosure.get("fallback_reason")
    return None


def _structural_identity_scan(variant_key: str, replay: dict[str, Any]) -> list[dict[str, Any]]:
    """Retroactively flag every champion-identical challenger prediction in this
    run's own scored checkpoints, mined straight from the prediction cache
    (no re-simulation)."""
    variant_id = _base_variant_id(variant_key)
    components = VARIANT_COMPONENTS.get(variant_id, frozenset())
    touches_qualifying = bool(components & {"q0", "q1"})
    own_checkpoints = {
        (event["event_id"], checkpoint)
        for event in replay["scored_events"]
        for checkpoint in event["checkpoints"]
    }
    by_key: dict[tuple[str, str, int, str, str | None], dict[str, Any]] = {}
    for path in glob.glob(str(PREDICTION_CACHE_ROOT / "*" / "*.json")):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = payload.get("key", {})
        kind = key.get("kind")
        if kind not in ("qualifying", "race_views"):
            continue
        if key.get("variant_id") not in ("champion", variant_id):
            continue
        dedup_key = (key.get("event_id"), key.get("checkpoint"))
        if dedup_key not in own_checkpoints:
            continue
        role = "champion" if key["variant_id"] == "champion" else "challenger"
        views = _RACE_VIEWS if kind == "race_views" else (None,)
        for view in views:
            store_key = (key["event_id"], key["checkpoint"], key.get("seed"), kind, view)
            by_key.setdefault(store_key, {})[role] = (
                payload["value"][view] if view else payload["value"]
            )

    flags: list[dict[str, Any]] = []
    for (event_id, checkpoint, seed, kind, view), pair in by_key.items():
        if "champion" not in pair or "challenger" not in pair:
            continue
        if kind == "qualifying" and not touches_qualifying:
            continue  # r0/r2 never claim to touch qualifying -- identity there is trivial
        champion, challenger = pair["champion"], pair["challenger"]
        identical = (
            _finish_order(champion) == _finish_order(challenger)
            if kind == "qualifying"
            else (
                [row.get("driver") for row in champion.get("finish_order", [])]
                == [row.get("driver") for row in challenger.get("finish_order", [])]
            )
        )
        if not identical:
            continue
        is_architectural = (
            kind == "race_views"
            and view == "conditional_actual_grid"
            and components <= GRID_ONLY_COMPONENTS
        )
        flags.append(
            {
                "event_id": event_id,
                "checkpoint": checkpoint,
                "seed": seed,
                "kind": kind,
                "view": view,
                "classification": "legitimate_architectural_invariance"
                if is_architectural
                else "structural_identity",
                "reason": _disclosed_fallback_reason(challenger, components),
            }
        )
    return sorted(
        flags,
        key=lambda f: (f["event_id"], f["checkpoint"], f["kind"], f["view"] or "", f["seed"] or 0),
    )


def _matched_subset_progression(
    runs: dict[str, dict[str, Any]], catalog: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """PRE vs FP2/FP3 on the identical main-dry event subset; PRE vs FP1 on the
    identical sprint-dry event subset. Recomputed from each run's own
    per_event_rows (already-cached seed-mean finisher_mae per event/checkpoint)
    -- no re-simulation."""
    # One event_id -> {view: finisher_mae} map per checkpoint, champion only,
    # preferring whichever tagged run actually scored that checkpoint.
    per_checkpoint_champion: dict[str, dict[str, dict[str, float]]] = {}
    for payload in runs.values():
        replay = payload["replay"]
        table = _variant_table(replay)
        for row in table["per_event_rows"]:
            race_views = row.get("race_views")
            if race_views is None:
                continue
            checkpoint = row["checkpoint"]
            bucket = per_checkpoint_champion.setdefault(checkpoint, {})
            bucket.setdefault(
                row["event_id"],
                {view: race_views[view]["champion"].get("finisher_mae") for view in _RACE_VIEWS},
            )

    def _matched_table(checkpoints: list[str], session_kind: str) -> dict[str, Any]:
        kind_events = {eid for eid, row in catalog.items() if row["session_kind"] == session_kind}
        available_per_checkpoint = {
            cp: set(per_checkpoint_champion.get(cp, {})) & kind_events for cp in checkpoints
        }
        # A checkpoint with no scored events at all must not empty the matched
        # subset for the checkpoints that do have data -- it is simply reported
        # unavailable while the rest intersect over their common events.
        non_empty = [events for events in available_per_checkpoint.values() if events]
        matched_events = set.intersection(*non_empty) if non_empty else set()
        rows: dict[str, Any] = {}
        for checkpoint in checkpoints:
            if not matched_events or not available_per_checkpoint[checkpoint]:
                rows[checkpoint] = {"available": False}
                continue
            per_view = {}
            for view in _RACE_VIEWS:
                values = [
                    per_checkpoint_champion[checkpoint][eid][view]
                    for eid in matched_events
                    if per_checkpoint_champion[checkpoint][eid].get(view) is not None
                ]
                per_view[view] = sum(values) / len(values) if values else None
            rows[checkpoint] = {"available": True, **per_view}
        return {
            "matched_event_ids": sorted(matched_events),
            "n_events": len(matched_events),
            "by_checkpoint": rows,
        }

    return {
        "main_dry_pre_fp2_fp3": _matched_table(["PRE", "FP2", "FP3"], "main"),
        "sprint_dry_pre_fp1": _matched_table(["PRE", "FP1"], "sprint"),
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
        identity_flags = _structural_identity_scan(variant_key, replay)
        variants[variant_key] = {
            "status": "scored",
            "run_tag": status["run_tag"],
            "manifest_sha256": replay_payload["manifest"]["manifest_sha256"],
            "comparison": _variant_table(replay),
            "decomposition": _decomposition(replay, catalog),
            "per_seed_spread": _per_seed_spread(variant_key, replay_payload),
            "checkpoint_refusals": replay.get("checkpoint_refusals", []),
            "research_gate_relaxations_applied": [
                dict(items) for items in research_gate_relaxations
            ],
            "research_gate_relaxation_detail": _research_gate_relaxation_detail(replay),
            "structural_identity_flags": identity_flags,
            **(
                {"q1_practice_activation": _q1_practice_activation(replay)}
                if variant_key.startswith("q1_qualifying_practice")
                else {}
            ),
        }

    checkpoint_source: dict[str, str] = {}
    for variant_key, payload in variants.items():
        if payload["status"] != "scored":
            continue
        for checkpoint in payload["comparison"]["aggregate_by_checkpoint"]:
            checkpoint_source.setdefault(checkpoint, variant_key)
    unmatched_progression = _checkpoint_progression(runs, checkpoint_source)
    matched_progression = _matched_subset_progression(runs, catalog)

    # r0's FP-checkpoint rows are champion-identical for a structural reason
    # (race_practice_challenger never activates in this harness's stored_profiles
    # replay mode -- see structural_identity_flags on r0_long_run__fp_hisim2) --
    # reclassify rather than present as a normal scored, differentiated result.
    r0_pathological = any(
        f["classification"] == "structural_identity" and f["kind"] == "race_views"
        for key, payload in variants.items()
        if _base_variant_id(key) == "r0_long_run" and payload["status"] == "scored"
        for f in payload["structural_identity_flags"]
    )

    report: dict[str, Any] = {
        "artifact_type": "race_mae_walk_forward_variant_comparison",
        "schema_version": 3,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "supersedes": "2026_walk_forward_variant_comparison_v2.json",
        "note": "Phase-4 follow-up: generalized structural-identity guard (retroactive, "
        "no re-sim) + matched-subset checkpoint progression. v2 kept as-is.",
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
        "champion_checkpoint_progression_UNMATCHED_SUPERSEDED": unmatched_progression,
        "champion_checkpoint_progression_matched_subset": matched_progression,
        "r0_structural_reclassification": {
            "pathological": r0_pathological,
            "statement": (
                "r0_long_run's FP-checkpoint results in fp_hisim2 are RECLASSIFIED: "
                "structurally untested (champion-identical: no raw practice laps in "
                "replay), not a scored differentiated result. Every r0 challenger "
                "prediction at FP1/FP2/FP3 (qualifying grid is trivially identical -- "
                "r0 has no qualifying component -- but BOTH race views are also "
                "identical, which is not expected: r0's long-run pace effect should "
                "show up in race simulation regardless of which grid it started "
                "from). Cause: race_practice_challenger.applied=false, "
                "fallback_reason='insufficient_field_evidence_coverage' on every "
                "cached FP-checkpoint prediction -- the same class of gap that "
                "neutralized Q1 (this harness's stored_profiles checkpoint-replay "
                "mode never loads raw per-driver practice evidence, for any "
                "variant). r0 at PRE remains a legitimate, documented no-op (no "
                "practice sessions exist pre-weekend, so there is nothing to "
                "extract regardless of harness mode)."
            ),
        },
        "driver_cohort_decomposition": {
            key: _driver_cohort_decomposition_scoped(payload["replay"], catalog)
            for key, payload in runs.items()
        },
        "future_work_raw_laps_replay": (
            f"Loading raw per-lap FastF1 telemetry for every replayed checkpoint "
            f"(instead of the current stored_profiles aggregated-only mode) would "
            f"let both Q1 and r0 actually activate in this harness. Scoped as its "
            f"own future handoff, not implemented this round: see {RAW_LAPS_HANDOFF_PATH}."
        ),
        "q1_eligibility_table": _q1_eligibility_table(variants, runs),
        "production_untouched_statement": _production_untouched_statement(),
        "variants": variants,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# 2026 Walk-Forward Race-MAE Variant Comparison v3 (Phase 4: structural-identity "
        "guard + matched-subset progression)",
        "",
        f"Generated: {report['generated_at']}",
        f"Supersedes: {report['supersedes']} (kept, not deleted).",
        "",
        "## r0 structural reclassification (first-class finding)",
        "",
        report["r0_structural_reclassification"]["statement"],
        "",
        "## Structural-identity flags (generalized guard, retroactive from cache)",
        "",
        "Legend: `legitimate_architectural_invariance` = expected (grid-only component, "
        "conditional_actual_grid view discards the predicted grid) -- not a bug. "
        "`structural_identity` = champion-identical where the component was expected to "
        "differ -- flagged, not silently scored.",
        "",
    ]
    any_flags = False
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] != "scored" or not payload["structural_identity_flags"]:
            continue
        any_flags = True
        counts: dict[str, int] = {}
        for flag in payload["structural_identity_flags"]:
            counts[flag["classification"]] = counts.get(flag["classification"], 0) + 1
        lines.append(f"### `{variant_key}` -- {counts}")
        reasons = sorted({f["reason"] for f in payload["structural_identity_flags"] if f["reason"]})
        if reasons:
            lines.append(f"Disclosed reasons: {reasons}")
        lines.append("")
    if not any_flags:
        lines.append("(no scored variant had any champion-identical prediction)")
        lines.append("")

    lines.append(
        "## Champion checkpoint progression -- MATCHED event subsets (restated conclusion)"
    )
    lines.append("")
    for label, key in (
        ("Main-dry PRE vs FP2 vs FP3", "main_dry_pre_fp2_fp3"),
        ("Sprint-dry PRE vs FP1", "sprint_dry_pre_fp1"),
    ):
        table = matched_progression[key]
        lines.append(
            f"### {label} (n={table['n_events']} matched events: {table['matched_event_ids']})"
        )
        lines.append("")
        lines.append("| checkpoint | end_to_end finisher_mae | conditional finisher_mae |")
        lines.append("|---|---|---|")
        for checkpoint, row in table["by_checkpoint"].items():
            if not row.get("available"):
                lines.append(f"| {checkpoint} | not available on matched subset | not available |")
                continue
            e2e = row.get("end_to_end_predicted_grid")
            cond = row.get("conditional_actual_grid")
            lines.append(
                f"| {checkpoint} | {e2e:.3f} | {cond:.3f} |"
                if e2e is not None and cond is not None
                else f"| {checkpoint} | n/a | n/a |"
            )
        lines.append("")

    lines.append(
        "## Champion checkpoint progression -- UNMATCHED (SUPERSEDED, kept for audit trail only)"
    )
    lines.append("")
    lines.append(
        "This table compares checkpoints scored over DIFFERENT event counts (e.g. PRE "
        "over 7 events vs FP1 over 2) -- confounded, do not draw conclusions from it. "
        "See the matched-subset tables above for the restated conclusion."
    )
    lines.append("")
    lines.append("| checkpoint | end_to_end finisher_mae | conditional finisher_mae |")
    lines.append("|---|---|---|")
    for checkpoint, row in sorted(unmatched_progression.items()):
        if not row.get("available"):
            lines.append(f"| {checkpoint} | not run | not run |")
            continue
        lines.append(
            f"| {checkpoint} | {row['champion_end_to_end_finisher_mae']:.3f} | "
            f"{row['champion_conditional_finisher_mae']:.3f} |"
        )
    lines.append("")

    lines.append("## Future work: raw-laps replay handoff")
    lines.append("")
    lines.append(report["future_work_raw_laps_replay"])
    lines.append("")

    lines.append("## Q1 eligibility table")
    lines.append("")
    lines.append("| event_id | checkpoint | outcome | reason / relaxation | run_tag |")
    lines.append("|---|---|---|---|---|")
    for row in report["q1_eligibility_table"]:
        lines.append(
            f"| {row['event_id']} | {row['checkpoint']} | {row['outcome']} | {row['reason']} | "
            f"{row['run_tag']!r} |"
        )
    lines.append("")

    lines.append("## Production-untouched statement")
    lines.append("")
    stmt = report["production_untouched_statement"]
    lines.append(stmt["statement"])
    lines.append("")
    lines.append(
        f"- `config/production_config.json` sha256: `{stmt['config_production_config_json_sha256']}`"
    )
    lines.append(
        f"- byte-identical to expected: **{stmt['config_production_config_json_byte_identical']}**"
    )
    lines.append(f"- `config/default.yaml`: `{stmt['config_default_yaml_model_variant_line']}`")
    lines.append("")

    lines.append(
        "## Full variant status (see v2 report for the full race-view/decomposition tables, unchanged)"
    )
    lines.append("")
    for variant_key, payload in sorted(variants.items()):
        if payload["status"] == "scored":
            lines.append(f"- `{variant_key}`: scored (tag={payload['run_tag']!r})")
        else:
            lines.append(f"- `{variant_key}`: {payload['status']} -- {payload.get('detail')}")
    lines.append("")

    MARKDOWN_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json: {JSON_OUTPUT_PATH}")
    print(f"markdown: {MARKDOWN_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
