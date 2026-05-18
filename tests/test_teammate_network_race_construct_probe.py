"""Tests for the race construct-probe diagnostic runner."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.probe_teammate_network_race_constructs import (
    RaceProbeTarget,
    build_parser,
    build_race_probe_report,
    format_race_probe_report,
    race_probe_targets,
)

from src.extractors.matched_laps import MatchedLapConfig


def test_build_parser_defaults_to_offline_race_probe() -> None:
    """The race probe defaults to stored observations and offline cache reads."""
    args = build_parser().parse_args([])
    assert str(args.observations) == (
        "data/processed/teammate_network_observations/latest/aggregated_observations.csv"
    )
    assert str(args.cache_dir) == "data/raw/.fastf1_cache"
    assert str(args.output_dir) == "data/diagnostics/teammate_network_construct_probe"
    assert args.online is False


def test_race_probe_report_summarizes_paired_and_broad_constructs() -> None:
    """The report keeps strict paired and broader valid-lap race views separate."""
    target = RaceProbeTarget(
        check_id="aaa_bbb_race_2024",
        year=2024,
        team="Example",
        faster_driver="AAA",
        slower_driver="BBB",
        threshold_s=0.50,
        source="Example Source",
    )
    observations = pd.DataFrame(
        [
            _observation_row(race_name="One Grand Prix", gap_s=0.20, se_s=0.10, n_pairs=3),
            _observation_row(race_name="Two Grand Prix", gap_s=0.60, se_s=0.20, n_pairs=4),
        ]
    )
    session_rows = pd.DataFrame(
        [
            _session_row(
                race_name="One Grand Prix",
                phase5_delta_s=0.20,
                current_delta_s=0.20,
                broad_median_s=0.30,
                broad_mean_s=0.35,
            ),
            _session_row(
                race_name="Two Grand Prix",
                phase5_delta_s=0.60,
                current_delta_s=0.58,
                broad_median_s=0.70,
                broad_mean_s=0.75,
            ),
        ]
    )

    report = build_race_probe_report(
        observations=observations,
        session_rows=session_rows,
        targets=[target],
        config=MatchedLapConfig(),
        built_at="2026-05-17T10:00:00+00:00",
    )
    summary = report["summaries"][0]
    markdown = format_race_probe_report(report)

    assert summary["phase5_valid_rows"] == 2
    assert summary["phase5_equal_mean_delta_s"] == pytest.approx(0.40)
    assert summary["phase5_wls_mean_delta_s"] == pytest.approx(0.30)
    assert summary["cache_current_equal_mean_delta_s"] == pytest.approx(0.39)
    assert summary["cache_broad_valid_median_equal_mean_delta_s"] == pytest.approx(0.50)
    assert summary["cache_broad_valid_mean_equal_mean_delta_s"] == pytest.approx(0.55)
    assert summary["artifact_cache_delta_mismatch_count"] == 1
    assert "`aaa_bbb_race_2024`" in markdown
    assert "Broad valid-lap median" in markdown


def test_race_probe_targets_follow_external_context_rows() -> None:
    """Race probes keep tracking rows after PACETEQ HARD demotion."""
    observations = pd.DataFrame(
        [
            _observation_row(
                race_name="One Grand Prix",
                gap_s=0.20,
                se_s=0.10,
                n_pairs=3,
                reference=reference,
                comparison=comparison,
                team=team,
                year=year,
            )
            for year, team, reference, comparison in [
                (2022, "Red Bull Racing", "VER", "PER"),
                (2023, "Red Bull Racing", "VER", "PER"),
                (2024, "Red Bull Racing", "VER", "PER"),
                (2023, "Aston Martin", "ALO", "STR"),
                (2024, "Aston Martin", "ALO", "STR"),
                (2023, "Williams", "ALB", "SAR"),
                (2024, "Williams", "ALB", "SAR"),
            ]
        ]
    )

    check_ids = {target.check_id for target in race_probe_targets(observations)}

    assert "verstappen_perez_race_2022" in check_ids


def _observation_row(
    *,
    race_name: str,
    gap_s: object,
    se_s: object,
    n_pairs: int,
    reference: str = "AAA",
    comparison: str = "BBB",
    team: str = "Example",
    year: int = 2024,
) -> dict[str, object]:
    """Build one stored Phase 5 race aggregate row."""
    return {
        "reference_driver_code": reference,
        "comparison_driver_code": comparison,
        "team": team,
        "year": year,
        "race_name": race_name,
        "session_kind": "race",
        "matched_gap_median_s": gap_s,
        "matched_gap_se_s": se_s,
        "n_matched_pairs": n_pairs,
        "weather_bucket": "dry",
        "skip_reason": pd.NA,
    }


def _session_row(
    *,
    race_name: str,
    phase5_delta_s: float,
    current_delta_s: float,
    broad_median_s: float,
    broad_mean_s: float,
) -> dict[str, object]:
    """Build one fresh-cache race probe row."""
    return {
        "check_id": "aaa_bbb_race_2024",
        "year": 2024,
        "race_name": race_name,
        "current_construct_delta_s": current_delta_s,
        "broad_valid_median_delta_s": broad_median_s,
        "broad_valid_mean_delta_s": broad_mean_s,
        "phase5_delta_s": phase5_delta_s,
    }
