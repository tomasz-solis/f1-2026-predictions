"""Tests for the qualifying construct-probe diagnostic runner."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.probe_teammate_network_constructs import (
    QualifyingProbeTarget,
    build_parser,
    build_probe_report,
    format_probe_report,
    qualifying_probe_targets,
)

from src.extractors.matched_laps import MatchedLapConfig


def test_build_parser_defaults_to_offline_construct_probe() -> None:
    """The probe defaults to stored observations and offline cache reads."""
    args = build_parser().parse_args([])
    assert args.observations.parts == (
        "data",
        "processed",
        "teammate_network_observations",
        "latest",
        "aggregated_observations.csv",
    )
    assert args.cache_dir.parts == ("data", "raw", ".fastf1_cache")
    assert args.output_dir.parts == (
        "data",
        "diagnostics",
        "teammate_network_construct_probe",
    )
    assert args.online is False


def test_probe_report_summarizes_phase5_and_cache_constructs() -> None:
    """The report keeps stored and fresh-cache construct summaries separate."""
    target = QualifyingProbeTarget(
        check_id="aaa_bbb_quali_2024",
        year=2024,
        faster_driver="AAA",
        slower_driver="BBB",
        threshold_s=0.50,
        source="Example Source",
    )
    observations = pd.DataFrame(
        [
            _observation_row(race_name="One Grand Prix", gap_s=0.20, se_s=0.10, n_pairs=3),
            _observation_row(race_name="Two Grand Prix", gap_s=0.60, se_s=0.20, n_pairs=4),
            _observation_row(
                race_name="Three Grand Prix",
                gap_s=pd.NA,
                se_s=pd.NA,
                n_pairs=2,
                skip_reason="insufficient_matched_pairs",
            ),
        ]
    )
    session_rows = pd.DataFrame(
        [
            _session_row(
                race_name="One Grand Prix",
                phase5_delta_s=0.20,
                current_delta_s=0.20,
                highest_common_s=0.30,
                any_valid_s=0.35,
            ),
            _session_row(
                race_name="Two Grand Prix",
                phase5_delta_s=0.60,
                current_delta_s=0.58,
                highest_common_s=0.70,
                any_valid_s=0.75,
            ),
        ]
    )

    report = build_probe_report(
        observations=observations,
        session_rows=session_rows,
        targets=[target],
        config=MatchedLapConfig(),
        built_at="2026-05-17T10:00:00+00:00",
    )
    summary = report["summaries"][0]
    markdown = format_probe_report(report)

    assert summary["phase5_valid_rows"] == 2
    assert summary["phase5_equal_mean_delta_s"] == pytest.approx(0.40)
    assert summary["phase5_wls_mean_delta_s"] == pytest.approx(0.30)
    assert summary["cache_current_equal_mean_delta_s"] == pytest.approx(0.39)
    assert summary["cache_highest_common_best_equal_mean_delta_s"] == pytest.approx(0.50)
    assert summary["cache_any_valid_best_equal_mean_delta_s"] == pytest.approx(0.55)
    assert summary["artifact_cache_delta_mismatch_count"] == 1
    assert "`aaa_bbb_quali_2024`" in markdown
    assert "0.300s" in markdown


def test_qualifying_probe_targets_follow_external_context_rows() -> None:
    """Construct probes keep tracking qualifying rows after HARD demotion."""
    check_ids = {target.check_id for target in qualifying_probe_targets()}
    assert "russell_hamilton_quali_2024" in check_ids


def _observation_row(
    *,
    race_name: str,
    gap_s: object,
    se_s: object,
    n_pairs: int,
    skip_reason: object = pd.NA,
) -> dict[str, object]:
    """Build one stored Phase 5 aggregate row."""
    return {
        "reference_driver_code": "AAA",
        "comparison_driver_code": "BBB",
        "year": 2024,
        "race_name": race_name,
        "session_kind": "qualifying",
        "matched_gap_median_s": gap_s,
        "matched_gap_se_s": se_s,
        "n_matched_pairs": n_pairs,
        "weather_bucket": "dry",
        "skip_reason": skip_reason,
    }


def _session_row(
    *,
    race_name: str,
    phase5_delta_s: float,
    current_delta_s: float,
    highest_common_s: float,
    any_valid_s: float,
) -> dict[str, object]:
    """Build one fresh-cache qualifying probe row."""
    return {
        "check_id": "aaa_bbb_quali_2024",
        "year": 2024,
        "race_name": race_name,
        "current_construct_delta_s": current_delta_s,
        "highest_common_best_delta_s": highest_common_s,
        "any_valid_best_delta_s": any_valid_s,
        "phase5_delta_s": phase5_delta_s,
    }
