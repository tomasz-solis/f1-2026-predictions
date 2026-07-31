#!/usr/bin/env python3
"""Round-9/10 fix-validation backtest: does capping the cumulative practice-
overlay pull (Fix A) or gating it on field-evidence coverage (Fix B) recover
champion's PRE-level race MAE at FP2/FP3 (main-dry) and FP1 (sprint-dry),
without harming PRE or qualifying grid MAE?

Compares every champion variant side by side on the SAME matched event subset
(intersection of what baseline AND every fix variant actually scored):
- baseline500: a CLEAN uncapped 500-sim baseline (run_tag "baseline500") --
  apples-to-apples with every fix variant, which all also run at 500/500 sims.
  (Supersedes an earlier confounded comparison that mixed 20-sim PRE against
  500-sim FP numbers.)
- pullcap025 / pullcap035: `research_cumulative_pull_cap` (Fix A).
- covgate050 / covgate070: `research_min_field_coverage` (Fix B).

This is diagnosis/backtest evidence only -- no production file is changed by
this script, and none of the candidate parameters are applied to
config/production_config.json or config/default.yaml. See
production_change_implied for what shipping any of this would require, and
NOT_IMPLEMENTED for the explicit disclaimer.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_practice_overlay_degradation_diagnostic import (  # noqa: E402
    _QUALIFYING_METRIC,
    _RACE_METRICS,
    _index_champion_metrics,
    _matched_event_ids,
    _metric_table,
    _seed_level_metrics,
    _seed_spread,
)
from scripts.build_race_mae_investigation_report_v2 import (  # noqa: E402
    OUTPUT_DIR,
    _catalog_by_id,
    _load_tagged_runs,
)
from scripts.build_race_mae_investigation_report_v3 import _RACE_VIEWS  # noqa: E402

JSON_OUTPUT_PATH = OUTPUT_DIR / "practice_overlay_fix_comparison.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "PRACTICE_OVERLAY_FIX_COMPARISON.md"

_BASELINE_NAME = "baseline500"
_FIX_VARIANTS = ("pullcap025", "pullcap035", "covgate050", "covgate070")
_VARIANTS = {_BASELINE_NAME: [_BASELINE_NAME], **{name: [name] for name in _FIX_VARIANTS}}

NOT_IMPLEMENTED = (
    "NONE of these candidates are implemented in production. This script backtests "
    "research-only, default-off overrides confined to ProductionReplayBackend / "
    "the replay harness (research_cumulative_pull_cap, research_min_field_coverage) "
    "-- config/production_config.json, config/default.yaml, and "
    "src/systems/testing_updater.py's live defaults are untouched. Any decision to "
    "ship one of these requires explicit user approval AND wider-season validation "
    "(n=4 main-dry / n=2 sprint-dry here is directional evidence, not proof)."
)

PRODUCTION_CHANGE_IMPLIED = {
    "pullcap025": {
        "file": "src/systems/testing_updater.py (update_from_testing_sessions call sites) "
        "or a new config-driven ceiling consumed there",
        "config_key": "a new baseline_predictor.practice_capture.cumulative_pull_cap "
        "(does not exist today)",
        "value": 0.25,
        "effective_behavior": "Only the FIRST practice session of a weekend moves team "
        "characteristics; every later session in the same weekend is ignored at the "
        "characteristics-EWMA layer (FP-blend-weight-based qualifying signals untouched).",
    },
    "pullcap035": {
        "file": "same as above",
        "config_key": "same as above",
        "value": 0.35,
        "effective_behavior": "First session gets its full normal weight; a second session "
        "tops up to the 0.35 ceiling; any third session contributes nothing further.",
    },
    "covgate050": {
        "file": "src/systems/testing_updater.py (update_from_testing_sessions) -- a "
        "coverage check before/around the characteristics write, analogous to "
        "src/features/race_practice_evidence.py's summarize_race_practice_coverage",
        "config_key": "a new baseline_predictor.practice_capture.min_field_coverage "
        "(does not exist today; MIN_R0_TEAM_COVERAGE=0.50 is the closest existing precedent)",
        "value": 0.50,
        "effective_behavior": "A session that updates fewer than 50% of the field's teams "
        "contributes NOTHING to characteristics (full revert), instead of being trusted at "
        "the normal EWMA weight regardless of how few teams it actually covered.",
    },
    "covgate070": {
        "file": "same as above",
        "config_key": "same as above",
        "value": 0.70,
        "effective_behavior": "Same mechanism, stricter bar (70% of the field).",
    },
}


def _load_variant_index(tags: list[str]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    _run_manifest, runs = _load_tagged_runs(tags)
    return _index_champion_metrics(runs)


def _checkpoint_identical(
    variants: dict[str, Any], baseline_name: str, fix_name: str, checkpoint: str
) -> bool:
    """True iff EVERY recomputed metric (qualifying grid_mae + both race views'
    finisher/weighted/top_heavy MAE) is exact-float-equal between the fix and
    baseline at this checkpoint -- i.e. the gate/cap never actually diverged
    from default behavior here."""
    baseline_q = variants[baseline_name]["qualifying"]["grid_mae"][checkpoint]["mean"]
    fix_q = variants[fix_name]["qualifying"]["grid_mae"][checkpoint]["mean"]
    if baseline_q != fix_q:
        return False
    for view in _RACE_VIEWS:
        for metric in _RACE_METRICS:
            b = variants[baseline_name]["race_views"][view][metric]["by_checkpoint"][checkpoint][
                "mean"
            ]
            f = variants[fix_name]["race_views"][view][metric]["by_checkpoint"][checkpoint]["mean"]
            if b != f:
                return False
    return True


def main() -> int:
    catalog = _catalog_by_id()
    main_events = {eid for eid, row in catalog.items() if row["session_kind"] == "main"}
    sprint_events = {eid for eid, row in catalog.items() if row["session_kind"] == "sprint"}

    indices = {name: _load_variant_index(tags) for name, tags in _VARIANTS.items()}
    variant_order = [_BASELINE_NAME, *_FIX_VARIANTS]

    subsets = {
        "main_dry_pre_fp2_fp3": {"checkpoints": ["PRE", "FP2", "FP3"], "kind_events": main_events},
        "sprint_dry_pre_fp1": {"checkpoints": ["PRE", "FP1"], "kind_events": sprint_events},
    }

    report: dict[str, Any] = {
        "artifact_type": "practice_overlay_fix_comparison",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "not_implemented": NOT_IMPLEMENTED,
        "production_change_implied": PRODUCTION_CHANGE_IMPLIED,
        "subsets": {},
    }

    for subset_name, spec in subsets.items():
        checkpoints = spec["checkpoints"]
        # Matched across EVERY variant -- an event only counts if baseline AND
        # every fix variant scored it at every checkpoint being compared, so the
        # comparison is never confounded by one variant covering more events.
        per_variant_matched = {
            name: set(_matched_event_ids(indices[name], checkpoints, spec["kind_events"]))
            for name in variant_order
        }
        matched_events = sorted(set.intersection(*per_variant_matched.values()))

        subset_report: dict[str, Any] = {
            "matched_event_ids": matched_events,
            "n_events": len(matched_events),
            "per_variant_own_matched_event_ids": {
                k: sorted(v) for k, v in per_variant_matched.items()
            },
            "variants": {},
        }

        for name in variant_order:
            index = indices[name]
            seed_index = _seed_level_metrics(catalog, matched_events, checkpoints)
            variant_report: dict[str, Any] = {
                "qualifying": {
                    "grid_mae": _metric_table(
                        index,
                        matched_events,
                        checkpoints,
                        kind="qualifying",
                        metric=_QUALIFYING_METRIC,
                    ),
                    "seed_spread": _seed_spread(
                        seed_index, matched_events, checkpoints, field="qualifying", metric=None
                    ),
                },
                "race_views": {},
            }
            for view in _RACE_VIEWS:
                variant_report["race_views"][view] = {
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
            subset_report["variants"][name] = variant_report

        # Per-event, per-fix "did this specific event get better or worse than
        # baseline at the last checkpoint, on BOTH race views" -- guards against a
        # mean-only view hiding a fix that helps on average but wrecks one event,
        # and against a fix that only ever shows up in the conditional view (grid
        # held fixed) without ever reaching the end_to_end view users actually see.
        last_checkpoint = checkpoints[-1]
        subset_report["per_event_vs_baseline_at_last_checkpoint"] = {}
        for fix_name in _FIX_VARIANTS:
            per_view_rows: dict[str, list[dict[str, Any]]] = {}
            for view in _RACE_VIEWS:
                rows = []
                for eid in matched_events:
                    baseline_val = indices[_BASELINE_NAME][eid][last_checkpoint][view].get(
                        "finisher_mae"
                    )
                    fix_val = indices[fix_name][eid][last_checkpoint][view].get("finisher_mae")
                    if baseline_val is None or fix_val is None:
                        continue
                    rows.append(
                        {
                            "event_id": eid,
                            "baseline": baseline_val,
                            "fix": fix_val,
                            "delta_fix_minus_baseline": fix_val - baseline_val,
                            "worse_under_fix": fix_val > baseline_val,
                        }
                    )
                per_view_rows[view] = sorted(rows, key=lambda r: -r["delta_fix_minus_baseline"])
            subset_report["per_event_vs_baseline_at_last_checkpoint"][fix_name] = per_view_rows

        # Identity check: did this fix variant actually DIVERGE from baseline at
        # this checkpoint, or is it byte-identical (exact float equality on the
        # recomputed metric means -- if the underlying raw predictions never
        # differed, the metric means won't either)? A gate that never diverges is
        # itself a first-class finding (the sessions it would have blocked
        # already clear its bar) -- must be stated plainly, not buried as a
        # silent "no effect."
        subset_report["identity_check_vs_baseline"] = {
            fix_name: {
                cp: _checkpoint_identical(subset_report["variants"], _BASELINE_NAME, fix_name, cp)
                for cp in checkpoints
            }
            for fix_name in _FIX_VARIANTS
        }

        report["subsets"][subset_name] = subset_report

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Practice-overlay fix comparison (Fix A cumulative-pull-cap + Fix B field-coverage-gate, NOT implemented in production)",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"**{NOT_IMPLEMENTED}**",
        "",
        "## Production change each variant would imply (NOT applied this round)",
        "",
        f"```json\n{json.dumps(report['production_change_implied'], indent=2)}\n```",
        "",
    ]
    for subset_name, subset_report in report["subsets"].items():
        checkpoints = subsets[subset_name]["checkpoints"]
        lines.append(
            f"## {subset_name} (n={subset_report['n_events']} events matched across ALL {len(variant_order)} variants: {subset_report['matched_event_ids']})"
        )
        lines.append("")
        if not subset_report["matched_event_ids"]:
            lines.append(
                "No events matched across every variant -- see per_variant_own_matched_event_ids in the JSON for why."
            )
            lines.append("")
            continue

        lines.append(
            "### Identity check: did each fix actually diverge from baseline500 at each checkpoint?"
        )
        lines.append("")
        lines.append(
            "A `False` cell below means this fix variant is BYTE-IDENTICAL to baseline500 at that "
            "checkpoint (exact float equality on every recomputed metric mean, qualifying + both race "
            "views) -- the gate never actually fired there. That is a first-class finding in its own "
            "right when it happens, not a null result to bury: it means the sessions this fix would "
            "have blocked already clear its bar, so this specific mechanism cannot be the fix."
        )
        lines.append("")
        lines.append("| checkpoint | " + " | ".join(_FIX_VARIANTS) + " |")
        lines.append("|" + "---|" * (len(_FIX_VARIANTS) + 1))
        for cp in checkpoints:
            cells = [
                str(subset_report["identity_check_vs_baseline"][fix_name][cp])
                for fix_name in _FIX_VARIANTS
            ]
            lines.append(f"| {cp} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append(
            "### Qualifying grid_mae (should be ~unchanged -- every fix here only touches the race-pace overlay)"
        )
        lines.append("")
        header_cols = ["checkpoint", *variant_order]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * len(header_cols))
        for cp in checkpoints:
            cells = [
                f"{subset_report['variants'][name]['qualifying']['grid_mae'][cp]['mean']:.3f}"
                if subset_report["variants"][name]["qualifying"]["grid_mae"][cp]["mean"] is not None
                else "n/a"
                for name in variant_order
            ]
            lines.append(f"| {cp} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Seed spread (mean per-event seed std) for the same checkpoints:")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * len(header_cols))
        for cp in checkpoints:
            stds = [
                f"{subset_report['variants'][name]['qualifying']['seed_spread'][cp]['mean_per_event_seed_std']:.3f}"
                if subset_report["variants"][name]["qualifying"]["seed_spread"][cp][
                    "mean_per_event_seed_std"
                ]
                is not None
                else "n/a"
                for name in variant_order
            ]
            lines.append(f"| {cp} | " + " | ".join(stds) + " |")
        lines.append("")

        for view in _RACE_VIEWS:
            lines.append(f"### race_views: {view}")
            lines.append("")
            for metric in _RACE_METRICS:
                lines.append(f"**{metric}**")
                lines.append("| " + " | ".join(header_cols) + " |")
                lines.append("|" + "---|" * len(header_cols))
                for cp in checkpoints:
                    cells = []
                    for name in variant_order:
                        row = subset_report["variants"][name]["race_views"][view][metric][
                            "by_checkpoint"
                        ][cp]
                        cells.append(f"{row['mean']:.3f}" if row["mean"] is not None else "n/a")
                    lines.append(f"| {cp} | " + " | ".join(cells) + " |")
                lines.append("")
                lines.append(f"{metric} seed spread:")
                lines.append("| " + " | ".join(header_cols) + " |")
                lines.append("|" + "---|" * len(header_cols))
                for cp in checkpoints:
                    stds = []
                    for name in variant_order:
                        spread = subset_report["variants"][name]["race_views"][view][metric][
                            "seed_spread"
                        ][cp]
                        s = spread["mean_per_event_seed_std"]
                        stds.append(f"{s:.3f}" if s is not None else "n/a")
                    lines.append(f"| {cp} | " + " | ".join(stds) + " |")
                lines.append("")

        lines.append(
            f"### Per-event vs baseline at {checkpoints[-1]} -- events made WORSE by each fix are flagged, both race views"
        )
        lines.append("")
        for fix_name, per_view_rows in subset_report[
            "per_event_vs_baseline_at_last_checkpoint"
        ].items():
            lines.append(f"#### `{fix_name}`")
            for view, rows in per_view_rows.items():
                worse = [r for r in rows if r["worse_under_fix"]]
                lines.append(f"`{view}`: {len(worse)}/{len(rows)} events made WORSE.")
                lines.append(f"```json\n{json.dumps(rows, indent=2)}\n```")
            lines.append("")

    MARKDOWN_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"json: {JSON_OUTPUT_PATH}")
    print(f"markdown: {MARKDOWN_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
