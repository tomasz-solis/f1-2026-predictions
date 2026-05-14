"""Tests for the bulk matched-lap observation builder."""

from __future__ import annotations

import pandas as pd
from scripts.build_matched_lap_observations import (
    BulkSessionSpec,
    build_diagnostic_report,
    build_parser,
    format_diagnostic_report,
)

from src.extractors.matched_laps import MatchedLapConfig


def _session_specs() -> list[BulkSessionSpec]:
    """Return two synthetic bulk session specs."""
    return [
        BulkSessionSpec(
            year=2024,
            race_name="Example Grand Prix",
            event_format="conventional",
            session_code="R",
            session_kind="race",
        ),
        BulkSessionSpec(
            year=2024,
            race_name="Example Grand Prix",
            event_format="conventional",
            session_code="Q",
            session_kind="qualifying",
        ),
    ]


def test_build_parser_defaults_to_phase5_window() -> None:
    """The bulk runner defaults to the agreed 2022-2025 extraction window."""
    args = build_parser().parse_args([])
    assert args.years == [2022, 2023, 2024, 2025]
    assert str(args.cache_dir) == "data/raw/.fastf1_cache"
    assert str(args.output_dir) == "data/processed/teammate_network_observations/latest"
    assert args.online is False


def test_diagnostic_report_includes_required_phase5_sections() -> None:
    """The report exposes coverage, skips, weather, components, and zero sessions."""
    raw = pd.DataFrame(
        [
            {
                "row_type": "matched_pair",
                "year": 2024,
                "race_name": "Example Grand Prix",
                "session_name": "Race",
                "session_kind": "race",
                "team": "Example",
                "reference_driver_code": "AAA",
                "comparison_driver_code": "BBB",
                "weather_bucket": "dry",
                "compound": "MEDIUM",
                "skip_reason": pd.NA,
            },
            {
                "row_type": "skipped_pair",
                "year": 2024,
                "race_name": "Example Grand Prix",
                "session_name": "Qualifying",
                "session_kind": "qualifying",
                "team": "Example",
                "reference_driver_code": "AAA",
                "comparison_driver_code": "BBB",
                "weather_bucket": pd.NA,
                "compound": pd.NA,
                "skip_reason": "insufficient_matched_pairs",
            },
        ]
    )
    aggregate = pd.DataFrame(
        [
            {
                "reference_driver_code": "AAA",
                "comparison_driver_code": "BBB",
                "team": "Example",
                "year": 2024,
                "race_name": "Example Grand Prix",
                "session_name": "Race",
                "session_kind": "race",
                "matched_gap_median_s": 0.25,
                "matched_gap_se_s": 0.05,
                "n_matched_pairs": 8,
                "weather_bucket": "dry",
                "skip_reason": pd.NA,
            },
            {
                "reference_driver_code": "CCC",
                "comparison_driver_code": "DDD",
                "team": "Example 2",
                "year": 2024,
                "race_name": "Example Grand Prix",
                "session_name": "Race",
                "session_kind": "race",
                "matched_gap_median_s": pd.NA,
                "matched_gap_se_s": pd.NA,
                "n_matched_pairs": 1,
                "weather_bucket": "wet",
                "skip_reason": "insufficient_matched_pairs",
            },
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "session_kind": "race",
                "skip_reason": pd.NA,
                "pit_laps": 4,
                "non_green_laps": 2,
                "sc_vsc_laps": 1,
                "stint_outlier_laps": 3,
                "candidate_matched_pairs": 8,
                "matched_pair_rows": 8,
            }
        ]
    )

    report = build_diagnostic_report(
        session_specs=_session_specs(),
        raw_observations=raw,
        aggregate_observations=aggregate,
        filter_diagnostics=diagnostics,
        errors=[],
        config=MatchedLapConfig(),
    )
    markdown = format_diagnostic_report(report)

    assert report["n_target_sessions"] == 2
    assert report["n_loaded_sessions"] == 2
    assert report["n_valid_aggregate_rows"] == 1
    assert report["weather_bucket_counts"] == [
        {"session_kind": "race", "weather_bucket": "dry", "count": 1}
    ]
    assert report["zero_observation_sessions"] == [
        {"year": 2024, "race_name": "Example Grand Prix", "session_kind": "qualifying"}
    ]
    assert report["connected_components"][0]["component_sizes"] == [2]
    assert report["connected_components"][0]["component_observation_counts"] == [1]
    assert report["connected_components"][0]["component_observation_shares"] == [1.0]
    assert "## Connected Components" in markdown
