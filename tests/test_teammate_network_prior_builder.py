"""Tests for the teammate-network prior builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts.build_teammate_network_prior import (
    PriorFitConfig,
    build_network_prior,
    build_teammate_network_prior,
    evaluate_validation,
    format_validation_report,
    write_prior_artifacts,
)


def _aggregate_row(
    *,
    reference: str,
    comparison: str,
    gap_s: float,
    session_kind: str = "race",
    race_name: str = "Example Grand Prix",
    team: str = "Example",
    year: int = 2024,
    n_matched_pairs: int = 12,
) -> dict[str, Any]:
    """Build one aggregate observation row for prior-builder tests."""
    return {
        "reference_driver_code": reference,
        "comparison_driver_code": comparison,
        "team": team,
        "year": year,
        "race_name": race_name,
        "session_name": "Race" if session_kind == "race" else "Qualifying",
        "session_kind": session_kind,
        "matched_gap_median_s": gap_s,
        "matched_gap_se_s": 0.10,
        "n_matched_pairs": n_matched_pairs,
        "weather_bucket": "dry",
        "skip_reason": pd.NA,
    }


def test_network_fit_preserves_positive_faster_sign_convention() -> None:
    """A positive A-B gap fits A as faster than B under the model convention."""
    observations = pd.DataFrame(
        [
            _aggregate_row(reference="AAA", comparison="BBB", gap_s=0.40),
            _aggregate_row(reference="BBB", comparison="CCC", gap_s=0.20),
            _aggregate_row(reference="AAA", comparison="CCC", gap_s=0.60),
        ]
    )
    config = PriorFitConfig(bootstrap_replicates=25, min_driver_observations=1)

    network = build_network_prior(observations, session_kind="race", config=config)

    drivers = network["drivers"]
    assert drivers["AAA"]["mu_s"] > drivers["BBB"]["mu_s"] > drivers["CCC"]["mu_s"]
    assert drivers["AAA"]["mu_s"] - drivers["BBB"]["mu_s"] == pytest.approx(0.40)
    assert network["components"][0]["anchored"] is True


def test_network_fit_rejects_multiple_large_components() -> None:
    """The builder stops when no component satisfies the dominance rule."""
    observations = pd.DataFrame(
        [
            _aggregate_row(reference="AAA", comparison="BBB", gap_s=0.20, team="Team 1"),
            _aggregate_row(reference="CCC", comparison="DDD", gap_s=0.10, team="Team 2"),
        ]
    )
    config = PriorFitConfig(bootstrap_replicates=5)

    with pytest.raises(ValueError, match="no dominant component"):
        build_network_prior(observations, session_kind="race", config=config)


def test_prior_artifact_validation_reports_threshold_failures(tmp_path: Path) -> None:
    """Ambiguous PACETEQ rows stay external and do not fake a hard pass."""
    artifact = {
        "race_network": {
            "drivers": {
                "VER": {"mu_s": 0.25},
                "PER": {"mu_s": -0.10},
                "BOT": {"mu_s": 0.02},
                "ZHO": {"mu_s": -0.01},
            }
        },
        "quali_network": {"drivers": {}},
    }

    validation = evaluate_validation(artifact)

    first = validation["context_checks"][0]
    assert first["check_id"] == "verstappen_perez_race_2022"
    assert first["observed_delta_s"] == pytest.approx(0.35)
    assert first["passed"] is True
    assert first["failure_analysis"] == "passed"
    assert validation["all_hard_checks_passed"] is False
    assert validation["failed_hard_check_ids"] == []
    assert validation["hard_validation_state"] == "provisional_no_same_construct_hard_checks"
    assert validation["hard_race_total"] == 0
    assert validation["hard_quali_total"] == 0
    assert validation["context_checks"][0]["tier"] == "EXTERNAL_CONTEXT"
    assert validation["supplemental_checks"][0]["tier"] == "SUPPLEMENTAL"

    written = write_prior_artifacts(
        {
            **artifact,
            "built_at": "2026-05-14T10:00:00+00:00",
            "config": {},
            "validation": validation,
        },
        output_dir=tmp_path,
    )
    assert Path(written["latest"]).exists()
    assert Path(written["validation_report"]).exists()
    assert (
        json.loads(Path(written["latest"]).read_text())["validation"]["all_hard_checks_passed"]
        is False
    )
    assert "HARD Race Checks" in Path(written["validation_report"]).read_text()


def test_validation_reports_direct_pair_diagnostics() -> None:
    """External context diagnostics still expose direct source-scope rows."""
    observations = pd.DataFrame(
        [
            _aggregate_row(
                reference="VER",
                comparison="PER",
                gap_s=0.70,
                year=2024,
                race_name="Example Grand Prix",
                team="Red Bull",
            )
        ]
    )
    artifact = {
        "race_network": {
            "drivers": {
                "VER": {"mu_s": 0.21},
                "PER": {"mu_s": -0.21},
                "BOT": {"mu_s": 0.02},
                "ZHO": {"mu_s": -0.01},
            }
        },
        "quali_network": {"drivers": {}},
    }

    validation = evaluate_validation(
        artifact,
        observations=observations,
        config=PriorFitConfig(min_driver_observations=1),
    )

    target = next(
        row
        for row in validation["context_checks"]
        if row["check_id"] == "verstappen_perez_race_2024"
    )
    diagnostics = target["direct_pair_diagnostics"]
    assert target["failure_analysis"] == "pooled_prior_below_source_scope_direct_delta"
    assert diagnostics["scope_year"] == 2024
    assert diagnostics["source_scope"]["weighted_mean_delta_s"] == pytest.approx(0.70)
    assert diagnostics["source_scope"]["n_observations"] == 1


def test_validation_report_contains_failure_context() -> None:
    """The Markdown report includes source-backed failures and diagnosis text."""
    artifact = {
        "built_at": "2026-05-14T10:00:00+00:00",
        "validation": {
            "source_backed_checks": [
                {
                    "check_id": "example_check",
                    "network_key": "race_network",
                    "source": "Example Source",
                    "threshold_s": 0.5,
                    "observed_delta_s": 0.4,
                    "passed": False,
                    "failure_analysis": "matched_lap_direct_delta_below_source_threshold",
                    "direct_pair_diagnostics": {
                        "source_scope": {"weighted_mean_delta_s": 0.35},
                        "all_years": {"weighted_mean_delta_s": 0.36},
                    },
                }
            ],
            "supplemental_checks": [],
            "context_checks": [],
            "cut_checks": [],
            "hard_race_passed": 0,
            "hard_race_total": 1,
            "hard_quali_passed": 0,
            "hard_quali_total": 0,
            "failed_hard_check_ids": ["example_check"],
            "hard_validation_state": "ready",
            "all_hard_checks_passed": False,
            "validation_contract_note": "Validation contract note.",
            "quali_validation_note": "Qualifying validation note.",
            "smoke_only_note": "Smoke-only note.",
        },
    }

    report = format_validation_report(artifact)

    assert "`example_check`" in report
    assert "matched-lap direct delta below source threshold" in report
    assert "Example Source" in report


def test_full_artifact_contains_race_and_quali_networks() -> None:
    """The full builder emits the required top-level artifact sections."""
    observations = pd.DataFrame(
        [
            _aggregate_row(reference="AAA", comparison="BBB", gap_s=0.40, session_kind="race"),
            _aggregate_row(reference="BBB", comparison="CCC", gap_s=0.20, session_kind="race"),
            _aggregate_row(reference="AAA", comparison="CCC", gap_s=0.60, session_kind="race"),
            _aggregate_row(
                reference="AAA",
                comparison="BBB",
                gap_s=0.30,
                session_kind="qualifying",
            ),
            _aggregate_row(
                reference="BBB",
                comparison="CCC",
                gap_s=0.10,
                session_kind="qualifying",
            ),
            _aggregate_row(
                reference="AAA",
                comparison="CCC",
                gap_s=0.40,
                session_kind="qualifying",
            ),
        ]
    )
    config = PriorFitConfig(bootstrap_replicates=10, min_driver_observations=1)

    artifact = build_teammate_network_prior(
        observations,
        config=config,
        built_at="2026-05-14T10:00:00+00:00",
    )

    assert set(artifact) >= {"built_at", "config", "race_network", "quali_network", "validation"}
    assert artifact["race_network"]["drivers"]["AAA"]["mu_s"] > 0
    assert artifact["quali_network"]["drivers"]["AAA"]["mu_s"] > 0
