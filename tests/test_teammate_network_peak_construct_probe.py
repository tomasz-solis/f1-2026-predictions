"""Tests for the network-wide qualifying peak construct probe."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.probe_teammate_network_peak_constructs import (
    QualifyingNetworkTarget,
    build_network_peak_report,
    build_pair_season_summaries,
    format_network_peak_report,
    qualifying_network_targets,
)

from src.extractors.matched_laps import MatchedLapConfig


def test_qualifying_network_targets_drop_incomplete_pairs() -> None:
    """Network targets keep only complete stored qualifying teammate pairs."""
    observations = pd.DataFrame(
        [
            _observation_row(
                year=2024,
                team="Example",
                reference="AAA",
                comparison="BBB",
                race_name="One Grand Prix",
            ),
            _observation_row(
                year=2024,
                team="Example",
                reference="AAA",
                comparison="BBB",
                race_name="Two Grand Prix",
            ),
            _observation_row(
                year=2024,
                team="Solo",
                reference="CCC",
                comparison=pd.NA,
                race_name="One Grand Prix",
            ),
        ]
    )

    targets = qualifying_network_targets(observations)

    assert targets == [
        QualifyingNetworkTarget(
            year=2024,
            team="Example",
            reference_driver="AAA",
            comparison_driver="BBB",
        )
    ]


def test_network_peak_report_summarizes_coverage_and_dispersion() -> None:
    """The network report keeps row coverage and within-pair dispersion separate."""
    targets = [
        QualifyingNetworkTarget(
            year=2024,
            team="Example",
            reference_driver="AAA",
            comparison_driver="BBB",
        ),
        QualifyingNetworkTarget(
            year=2024,
            team="Other",
            reference_driver="CCC",
            comparison_driver="DDD",
        ),
    ]
    observations = pd.DataFrame(
        [
            _observation_row(
                year=2024,
                team="Example",
                reference="AAA",
                comparison="BBB",
                race_name="One Grand Prix",
                gap_s=0.20,
                se_s=0.10,
                n_pairs=3,
            ),
            _observation_row(
                year=2024,
                team="Example",
                reference="AAA",
                comparison="BBB",
                race_name="Two Grand Prix",
                gap_s=0.40,
                se_s=0.20,
                n_pairs=4,
            ),
        ]
    )
    session_rows = pd.DataFrame(
        [
            _session_row(
                target_id="2024:Example:AAA-BBB",
                current_delta_s=0.20,
                peak_delta_s=0.30,
                any_delta_s=0.35,
                highest_common_segment="Q3",
                phase5_delta_s=0.20,
            ),
            _session_row(
                target_id="2024:Example:AAA-BBB",
                current_delta_s=0.40,
                peak_delta_s=0.50,
                any_delta_s=0.55,
                highest_common_segment="Q2",
                phase5_delta_s=0.40,
            ),
            _session_row(
                target_id="2024:Other:CCC-DDD",
                current_delta_s=None,
                peak_delta_s=0.10,
                any_delta_s=0.20,
                highest_common_segment="Q1",
                phase5_delta_s=None,
            ),
        ]
    )

    pair_rows = build_pair_season_summaries(
        observations=observations,
        session_rows=session_rows,
        targets=targets,
    )
    report = build_network_peak_report(
        session_rows=session_rows,
        pair_rows=pair_rows,
        targets=targets,
        config=MatchedLapConfig(),
        built_at="2026-05-17T10:00:00+00:00",
    )
    markdown = format_network_peak_report(report)
    summary = report["summary"]

    assert summary["pair_seasons"] == 2
    assert summary["current_rows"] == 2
    assert summary["highest_common_best_rows"] == 3
    assert summary["peak_row_gain_vs_current"] == 1
    assert summary["highest_common_best_abs_gt_1s_rows"] == 0
    assert summary["highest_common_best_abs_gt_2s_rows"] == 0
    assert summary["artifact_cache_delta_mismatch_count"] == 0
    assert summary["highest_common_segment_counts"] == {"Q1": 1, "Q2": 1, "Q3": 1}
    assert summary["current_pair_season_sd_s"]["count"] == 1
    assert summary["current_pair_season_sd_s"]["median"] == pytest.approx(0.1414213562)
    assert summary["highest_common_best_pair_season_sd_s"]["count"] == 1
    assert summary["phase5_wls_vs_equal_abs_shift_s"]["median"] == pytest.approx(0.05)
    assert "Highest-common best" in markdown
    assert "Largest Absolute Peak Session Deltas" in markdown
    assert "`2024:Other:CCC-DDD`" in markdown


def _observation_row(
    *,
    year: int,
    team: str,
    reference: object,
    comparison: object,
    race_name: str,
    gap_s: object = 0.20,
    se_s: object = 0.10,
    n_pairs: int = 3,
) -> dict[str, object]:
    """Build one stored qualifying aggregate row."""
    return {
        "reference_driver_code": reference,
        "comparison_driver_code": comparison,
        "team": team,
        "year": year,
        "race_name": race_name,
        "session_kind": "qualifying",
        "matched_gap_median_s": gap_s,
        "matched_gap_se_s": se_s,
        "n_matched_pairs": n_pairs,
        "weather_bucket": "dry",
        "skip_reason": pd.NA,
    }


def _session_row(
    *,
    target_id: str,
    current_delta_s: float | None,
    peak_delta_s: float | None,
    any_delta_s: float | None,
    highest_common_segment: str,
    phase5_delta_s: float | None,
    race_name: str = "Example Grand Prix",
) -> dict[str, object]:
    """Build one fresh-cache network probe row."""
    return {
        "target_id": target_id,
        "race_name": race_name,
        "current_construct_delta_s": current_delta_s,
        "highest_common_best_delta_s": peak_delta_s,
        "any_valid_best_delta_s": any_delta_s,
        "highest_common_segment": highest_common_segment,
        "phase5_delta_s": phase5_delta_s,
    }
