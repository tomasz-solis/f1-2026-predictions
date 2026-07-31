from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.qualifying_practice_challenger import FittedQualifyingPracticeModel
from src.models.qualifying_practice_runtime import (
    _candidate_utilities_against_event_field,
    _session_has_rainfall,
    build_qualifying_practice_feature_rows,
    predict_q1_position_records,
)


def _evidence(best_a: float, best_b: float, session: str) -> dict:
    return {
        "eligibility": {"eligible": True},
        "normalization": {"measurement_uncertainty_s": 0.20},
        "drivers": {
            "AAA": {
                "team": "Team A",
                "features": {
                    "best_adjusted_lap_s": best_a,
                    "best_two_mean_adjusted_lap_s": best_a + 0.05,
                    "q20_adjusted_lap_s": best_a + 0.02,
                    "theoretical_adjusted_lap_s": best_a - 0.03,
                    "execution_loss_s": 0.03,
                    "mad_s": 0.08,
                    "measurement_uncertainty_s": 0.20,
                    "effective_lap_count": 3.0,
                },
                "run_feature_candidates": [
                    {
                        "run_id": f"AAA:{session}:1",
                        "run_class": "quali_sim",
                        "compound": "SOFT",
                        "clean_lap_count": 3,
                        "effective_lap_count": 3.0,
                        "best_adjusted_lap_s": best_a,
                        "best_two_mean_adjusted_lap_s": best_a + 0.05,
                        "q20_adjusted_lap_s": best_a + 0.02,
                        "theoretical_adjusted_lap_s": best_a - 0.03,
                        "execution_loss_s": 0.03,
                        "mad_s": 0.08,
                        "measurement_uncertainty_s": 0.20,
                    }
                ],
                "counts": {
                    "clean_laps": 5,
                    "runs": {"quali_sim": 1},
                    "compounds": ["SOFT"],
                },
            },
            "BBB": {
                "team": "Team A",
                "features": {
                    "best_adjusted_lap_s": best_b,
                    "best_two_mean_adjusted_lap_s": best_b + 0.05,
                    "q20_adjusted_lap_s": best_b + 0.02,
                    "theoretical_adjusted_lap_s": best_b - 0.03,
                    "execution_loss_s": 0.03,
                    "mad_s": 0.10,
                    "measurement_uncertainty_s": 0.25,
                    "effective_lap_count": 3.0,
                },
                "run_feature_candidates": [
                    {
                        "run_id": f"BBB:{session}:1",
                        "run_class": "quali_sim",
                        "compound": "SOFT",
                        "clean_lap_count": 3,
                        "effective_lap_count": 3.0,
                        "best_adjusted_lap_s": best_b,
                        "best_two_mean_adjusted_lap_s": best_b + 0.05,
                        "q20_adjusted_lap_s": best_b + 0.02,
                        "theoretical_adjusted_lap_s": best_b - 0.03,
                        "execution_loss_s": 0.03,
                        "mad_s": 0.10,
                        "measurement_uncertainty_s": 0.25,
                    }
                ],
                "counts": {
                    "clean_laps": 5,
                    "runs": {"quali_sim": 1},
                    "compounds": ["SOFT"],
                },
            },
        },
        "session": {"session_code": session},
    }


def _drivers() -> list[dict]:
    return [
        {
            "driver": "AAA",
            "team": "Team A",
            "team_strength": 0.7,
            "quali_pace": 0.6,
            "skill": 0.7,
        },
        {
            "driver": "BBB",
            "team": "Team A",
            "team_strength": 0.7,
            "quali_pace": 0.6,
            "skill": 0.7,
        },
    ]


@pytest.mark.parametrize("rainfall", [[False, True], [0, 1], ["false", "rain"]])
def test_dry_only_runtime_detects_any_recorded_rainfall(rainfall: list[object]) -> None:
    assert _session_has_rainfall(pd.DataFrame({"Rainfall": rainfall})) is True


def test_dry_only_runtime_allows_explicitly_dry_or_missing_rainfall() -> None:
    assert _session_has_rainfall(pd.DataFrame({"Rainfall": [False, 0, "false"]})) is False
    assert _session_has_rainfall(pd.DataFrame({"LapTime": [90.0]})) is False


def test_runtime_adapter_builds_whole_field_and_session_change_features() -> None:
    rows, summary = build_qualifying_practice_feature_rows(
        {
            "FP1": _evidence(90.5, 90.4, "FP1"),
            "FP2": _evidence(90.0, 90.3, "FP2"),
        },
        all_drivers=_drivers(),
    )
    indexed = rows.set_index("driver")

    assert summary["eligible"] is True
    assert summary["eligible_drivers"] == 2
    assert indexed.loc["AAA", "session_improvement_s"] == 0.5
    assert indexed.loc["AAA", "teammate_gap_s"] == pytest.approx(-0.3)
    assert indexed.loc["AAA", "evidence_session_count"] == 2
    assert indexed.loc["AAA", "direct_soft_flag"] == 1.0
    assert [row["session_code"] for row in summary["run_feature_rows_by_driver"]["AAA"]] == [
        "FP1",
        "FP2",
    ]


def test_runtime_adapter_keeps_prior_only_driver_in_full_grid() -> None:
    drivers = _drivers() + [
        {
            "driver": "CCC",
            "team": "Team B",
            "team_strength": 0.5,
            "quali_pace": 0.5,
            "skill": 0.5,
        }
    ]
    rows, summary = build_qualifying_practice_feature_rows(
        {"FP2": _evidence(90.0, 90.3, "FP2")},
        all_drivers=drivers,
    )

    assert set(rows["driver"]) == {"AAA", "BBB", "CCC"}
    ccc = rows.set_index("driver").loc["CCC"]
    assert np.isnan(ccc["best_adjusted_lap_s"])
    assert ccc["evidence_quality_score"] == 0.0
    assert summary["field_size"] == 3


def test_runtime_adapter_fails_closed_when_practice_coverage_is_too_sparse() -> None:
    drivers = _drivers() + [
        {
            "driver": f"D{index}",
            "team": f"Team {team}",
            "team_strength": 0.5,
            "quali_pace": 0.5,
            "skill": 0.5,
        }
        for index, team in enumerate(("B", "B", "C", "C"), start=1)
    ]

    _rows, summary = build_qualifying_practice_feature_rows(
        {"FP2": _evidence(90.0, 90.3, "FP2")},
        all_drivers=drivers,
    )

    assert summary["eligible"] is False
    assert summary["eligible_drivers"] == 2
    assert summary["required_eligible_drivers"] == 3
    assert summary["eligible_teams"] == 1
    assert summary["required_eligible_teams"] == 2
    assert summary["fallback_reason"] == "insufficient_grid_evidence_coverage"


def test_runtime_prediction_uses_same_coherent_samples() -> None:
    rows, summary = build_qualifying_practice_feature_rows(
        {"FP2": _evidence(90.0, 90.3, "FP2")},
        all_drivers=_drivers(),
    )
    model = FittedQualifyingPracticeModel(
        checkpoint="FP2",
        feature_columns=("prior_utility", "best_adjusted_lap_s", "measurement_se_s"),
        coefficients=(0.2, -1.0, 0.0),
        feature_medians=(0.5, 90.15, 0.2),
        feature_scales=(0.1, 0.2, 0.1),
        temperature=0.5,
        training_events=30,
        generated_at=datetime.now(UTC).isoformat(),
    )

    records, scenarios, diagnostics = predict_q1_position_records(
        model=model,
        feature_rows=rows,
        n_simulations=200,
        rng=np.random.default_rng(17),
        evidence_summary=summary,
    )

    assert len(scenarios) == 200
    assert diagnostics["eligible_drivers"] == 2
    assert diagnostics["run_bootstrap_driver_count"] == 2
    assert diagnostics["run_bootstrap_candidate_count"] == 2
    assert diagnostics["run_bootstrap_mode"] == "compatible_run_utility"
    assert "run_feature_rows_by_driver" not in diagnostics
    for index, scenario in enumerate(scenarios):
        for position, driver in enumerate(scenario, start=1):
            assert records[driver][index] == position


def test_run_candidate_utility_is_aligned_to_central_event_coordinates() -> None:
    model = FittedQualifyingPracticeModel(
        checkpoint="FP2",
        feature_columns=("best_adjusted_lap_s",),
        coefficients=(-1.0,),
        feature_medians=(0.0,),
        feature_scales=(1.0,),
        temperature=1.0,
        training_events=30,
        generated_at=datetime.now(UTC).isoformat(),
    )
    central = pd.DataFrame(
        [
            {"driver": "AAA", "best_adjusted_lap_s": 100.0},
            {"driver": "BBB", "best_adjusted_lap_s": 101.0},
            {"driver": "CCC", "best_adjusted_lap_s": 102.0},
        ]
    )

    utilities = _candidate_utilities_against_event_field(
        model=model,
        central_rows=central,
        driver="AAA",
        candidate_rows=[{"best_adjusted_lap_s": 103.0}],
    )

    # Fixed against the central event median (101s), the candidate is two
    # seconds slower. A fresh median for the substituted frame must not hide one.
    assert utilities.tolist() == pytest.approx([-2.0])
