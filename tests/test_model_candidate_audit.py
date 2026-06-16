from __future__ import annotations

import pytest
from scripts.audit_model_candidates import (
    _bias_corrected_rank,
    _blend_with_previous,
    _build_promotion_gate,
    _diagnostic_coverage,
    _rolling_actual_rank,
)


def test_blend_with_previous_uses_recent_actual_rank_to_break_model_order():
    model_rows = [
        {"driver": "BBB", "team": "B", "position": 1},
        {"driver": "AAA", "team": "A", "position": 2},
    ]
    previous_actual_rows = [
        {"driver": "AAA", "team": "A", "position": 1},
        {"driver": "BBB", "team": "B", "position": 2},
    ]

    blended = _blend_with_previous(model_rows, previous_actual_rows, model_weight=0.4)

    assert [row["driver"] for row in blended] == ["AAA", "BBB"]


def test_rolling_actual_rank_averages_only_past_actual_rows():
    model_rows = [
        {"driver": "CCC", "team": "C", "position": 1},
        {"driver": "AAA", "team": "A", "position": 2},
        {"driver": "BBB", "team": "B", "position": 3},
    ]
    history = [
        [
            {"driver": "AAA", "team": "A", "position": 1},
            {"driver": "BBB", "team": "B", "position": 2},
            {"driver": "CCC", "team": "C", "position": 3},
        ],
        [
            {"driver": "AAA", "team": "A", "position": 2},
            {"driver": "BBB", "team": "B", "position": 1},
            {"driver": "CCC", "team": "C", "position": 3},
        ],
    ]

    ranked = _rolling_actual_rank(model_rows, history, window=2)

    assert [row["driver"] for row in ranked] == ["AAA", "BBB", "CCC"]


def test_bias_corrected_rank_uses_prior_signed_errors():
    previous_prediction = [
        {"driver": "AAA", "team": "A", "position": 3},
        {"driver": "BBB", "team": "B", "position": 1},
        {"driver": "CCC", "team": "C", "position": 2},
    ]
    previous_actual = [
        {"driver": "AAA", "team": "A", "position": 1},
        {"driver": "BBB", "team": "B", "position": 2},
        {"driver": "CCC", "team": "C", "position": 3},
    ]
    model_rows = [
        {"driver": "BBB", "team": "B", "position": 1},
        {"driver": "CCC", "team": "C", "position": 2},
        {"driver": "AAA", "team": "A", "position": 3},
    ]

    ranked = _bias_corrected_rank(
        model_rows,
        [previous_prediction],
        [previous_actual],
        alpha=1.0,
        level="driver",
    )

    assert [row["driver"] for row in ranked] == ["AAA", "BBB", "CCC"]


def test_bias_corrected_rank_rejects_mismatched_history_lengths():
    with pytest.raises(ValueError):
        _bias_corrected_rank(
            [{"driver": "AAA", "team": "A", "position": 1}],
            [[{"driver": "AAA", "team": "A", "position": 1}]],
            [],
            alpha=1.0,
            level="driver",
        )


def test_promotion_gate_promotes_model_aware_challenger_with_enough_events():
    summary = {
        "overall": [
            {
                "candidate": "fixed_blend_model_0.6",
                "events": 9,
                "mean_mae": 2.00,
            },
            {"candidate": "raw_model", "events": 9, "mean_mae": 2.30},
        ],
        "by_format": {
            "conventional": [
                {"candidate": "fixed_blend_model_0.6", "events": 6, "mean_mae": 2.10},
                {"candidate": "raw_model", "events": 6, "mean_mae": 2.25},
            ],
            "sprint": [
                {"candidate": "fixed_blend_model_0.6", "events": 3, "mean_mae": 1.80},
                {"candidate": "raw_model", "events": 3, "mean_mae": 2.40},
            ],
        },
    }

    gate = _build_promotion_gate(summary)

    assert gate["status"] == "promote"
    assert gate["best_candidate_family"] == "model_recent_actual_blend"


def test_promotion_gate_holds_recent_actual_only_winner():
    summary = {
        "overall": [
            {"candidate": "rolling_actual_2", "events": 9, "mean_mae": 1.90},
            {"candidate": "raw_model", "events": 9, "mean_mae": 2.30},
        ],
        "by_format": {},
    }

    gate = _build_promotion_gate(summary)

    assert gate["status"] == "hold"
    assert gate["best_candidate_family"] == "recent_actual_only"
    assert any("recent-actual-only" in reason for reason in gate["reasons"])


def test_diagnostic_coverage_counts_saved_model_metadata():
    coverage = _diagnostic_coverage(
        [
            {
                "metadata": {
                    "qualifying_model_diagnostics": {
                        "data_regime": "practice",
                        "characteristics_profile_used": "short_run",
                        "qualifying_residual_model_used": False,
                    }
                }
            },
            {"metadata": {}},
        ],
        session_kind="qualifying",
    )

    assert coverage["events_with_model_diagnostics"] == 1
    assert coverage["events_with_characteristics_profile"] == 1
    assert coverage["events_with_residual_model_flag"] == 1
    assert coverage["data_regime_counts"] == {"practice": 1}
