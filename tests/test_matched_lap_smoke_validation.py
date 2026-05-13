"""Tests for the matched-lap smoke validation harness."""

from __future__ import annotations

import pandas as pd
from scripts.validate_matched_lap_smoke_sessions import (
    SmokeSessionSpec,
    build_parser,
    build_smoke_summary,
    format_smoke_summary,
)


def test_build_parser_defaults_to_offline_cache_run() -> None:
    """The smoke validator defaults to local-cache evidence output."""
    args = build_parser().parse_args([])
    assert str(args.cache_dir) == "data/raw/.fastf1_cache"
    assert str(args.output_dir) == "data/diagnostics/matched_lap_extractor_smoke"
    assert args.online is False


def test_strategy_summary_reports_filter_totals() -> None:
    """Strategy smoke summaries expose SC/VSC, pit, and outlier counts."""
    spec = SmokeSessionSpec(
        category="strategy_asymmetric_race",
        year=2024,
        event_name="Miami Grand Prix",
        session_code="R",
        session_kind="race",
        weather_mode="dry",
        min_matched_pairs=50,
        max_matched_pairs=None,
    )
    matched = pd.DataFrame(
        {
            "row_type": ["matched_pair"] * 55,
            "weather_bucket": ["dry"] * 55,
            "skip_reason": [pd.NA] * 55,
        }
    )
    diagnostics = pd.DataFrame(
        {
            "non_green_laps": [10],
            "sc_vsc_laps": [6],
            "pit_laps": [12],
            "stint_outlier_laps": [3],
            "lap_level_weather_unreliable_laps": [0],
            "weather_mode_excluded_laps": [0],
        }
    )

    summary = build_smoke_summary(spec, matched, diagnostics)
    text = format_smoke_summary(summary)

    assert summary["status"] == "pass"
    assert summary["filter_totals"]["sc_vsc_laps"] == 6
    assert "sc_vsc=6" in text
    assert "stint_outlier=3" in text
