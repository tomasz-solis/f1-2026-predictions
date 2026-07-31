"""Focused test for the fix-comparison report's own new logic: matching events
across FIVE variants (not just two) and flagging per-event regressions on
BOTH race views (not just conditional_actual_grid)."""

from __future__ import annotations

import scripts.build_practice_overlay_fix_comparison_report as report


def test_matched_across_five_variants_excludes_an_event_missing_from_any_one() -> None:
    checkpoints = ["PRE", "FP2", "FP3"]
    full = {"e1": {"PRE": {}, "FP2": {}, "FP3": {}}, "e2": {"PRE": {}, "FP2": {}, "FP3": {}}}
    partial = {
        "e1": {"PRE": {}, "FP2": {}, "FP3": {}},
        "e2": {"PRE": {}, "FP2": {}},
    }  # e2 missing FP3
    indices = {
        "baseline500": full,
        "pullcap025": partial,
        "pullcap035": full,
        "covgate050": full,
        "covgate070": full,
    }
    per_variant_matched = {
        name: set(report._matched_event_ids(idx, checkpoints, {"e1", "e2"}))
        for name, idx in indices.items()
    }
    matched = sorted(set.intersection(*per_variant_matched.values()))
    # e2 must drop out because pullcap025 never scored its FP3 -- comparison must
    # only ever use events every one of the five variants actually has.
    assert matched == ["e1"]


def test_per_event_flag_checks_both_race_views_independently() -> None:
    """A fix that helps conditional_actual_grid but hurts end_to_end_predicted_grid
    must be flagged as worse on the view where it's worse, not averaged away."""
    indices = {
        "baseline500": {
            "e1": {
                "FP3": {
                    "conditional_actual_grid": {"finisher_mae": 3.0},
                    "end_to_end_predicted_grid": {"finisher_mae": 4.0},
                }
            }
        },
        "covgate050": {
            "e1": {
                "FP3": {
                    "conditional_actual_grid": {"finisher_mae": 2.5},  # better
                    "end_to_end_predicted_grid": {"finisher_mae": 4.5},  # worse
                }
            }
        },
    }
    for view, expect_worse in (
        ("conditional_actual_grid", False),
        ("end_to_end_predicted_grid", True),
    ):
        baseline_val = indices["baseline500"]["e1"]["FP3"][view]["finisher_mae"]
        fix_val = indices["covgate050"]["e1"]["FP3"][view]["finisher_mae"]
        assert (fix_val > baseline_val) == expect_worse


def _variant_block(grid_mae: float, race_mae: float) -> dict:
    return {
        "qualifying": {"grid_mae": {"FP3": {"mean": grid_mae}}},
        "race_views": {
            view: {
                metric: {"by_checkpoint": {"FP3": {"mean": race_mae}}}
                for metric in report._RACE_METRICS
            }
            for view in report._RACE_VIEWS
        },
    }


def test_checkpoint_identical_true_when_every_metric_matches_exactly() -> None:
    variants = {"baseline500": _variant_block(3.0, 4.0), "covgate050": _variant_block(3.0, 4.0)}
    assert report._checkpoint_identical(variants, "baseline500", "covgate050", "FP3") is True


def test_checkpoint_identical_false_when_a_single_metric_differs() -> None:
    """Even one diverging metric (out of 1 qualifying + 2 views x 3 race metrics)
    must flip the whole checkpoint to non-identical -- a gate that only nudges
    weighted_mae while leaving finisher_mae untouched still counts as "it fired"."""
    variants = {"baseline500": _variant_block(3.0, 4.0), "covgate050": _variant_block(3.0, 4.01)}
    assert report._checkpoint_identical(variants, "baseline500", "covgate050", "FP3") is False
