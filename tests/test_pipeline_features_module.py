"""Tests for feature pipeline orchestration utilities."""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from src.pipelines.features import F1FeaturePipeline, RelativePerformanceCalculator


def test_relative_performance_calculator_normalize_and_percentiles():
    df = pd.DataFrame(
        [
            {"driver_number": "1", "fastest_lap": 90.0, "avg_speed_full_throttle": 305.0},
            {"driver_number": "2", "fastest_lap": 91.0, "avg_speed_full_throttle": 302.0},
            {"driver_number": "3", "fastest_lap": 92.0, "avg_speed_full_throttle": 299.0},
        ]
    )

    calc = RelativePerformanceCalculator(use_median=True)
    normalized = calc.normalize_features(df)
    ranked = calc.add_percentile_ranks(normalized)

    assert "fastest_lap_rel" in normalized.columns
    assert "avg_speed_full_throttle_rel" in normalized.columns
    assert "fastest_lap_pct" in ranked.columns
    assert "avg_speed_full_throttle_pct" in ranked.columns


def test_relative_performance_calculator_mean_baseline():
    df = pd.DataFrame(
        [
            {"driver_number": "1", "metric": 10.0},
            {"driver_number": "2", "metric": 13.0},
            {"driver_number": "3", "metric": 16.0},
        ]
    )
    calc = RelativePerformanceCalculator(use_median=False)
    out = calc.normalize_features(df)
    assert out.loc[0, "metric_rel"] == -3.0
    assert out.loc[2, "metric_rel"] == 3.0


def _fake_session(name: str, event_name: str = "Bahrain Grand Prix"):
    return SimpleNamespace(
        event={"EventDate": datetime(2026, 3, 1), "EventName": event_name},
        name=name,
        date=datetime(2026, 3, 1, 12, 0),
    )


def test_f1_feature_pipeline_process_session_with_metadata(patcher):
    pipeline = F1FeaturePipeline()
    patcher.setattr(
        pipeline.session_aggregator,
        "extract_all_drivers",
        lambda _session: pd.DataFrame(
            [
                {"driver_number": "1", "fastest_lap": 90.0, "avg_speed_full_throttle": 304.0},
                {"driver_number": "2", "fastest_lap": 91.0, "avg_speed_full_throttle": 301.0},
            ]
        ),
    )

    out = pipeline.process_session(_fake_session("FP2"), add_metadata=True)

    assert len(out) == 2
    assert {"year", "event", "session_type", "session_date"}.issubset(out.columns)
    assert "fastest_lap_rel" in out.columns


def test_f1_feature_pipeline_process_session_empty_features(patcher):
    pipeline = F1FeaturePipeline()
    patcher.setattr(
        pipeline.session_aggregator, "extract_all_drivers", lambda _session: pd.DataFrame()
    )

    out = pipeline.process_session(_fake_session("FP1"))
    assert out.empty


def test_f1_feature_pipeline_process_multiple_sessions(patcher):
    pipeline = F1FeaturePipeline()

    def _process(session, add_metadata=True):
        if session.name == "FP1":
            return pd.DataFrame([{"driver_number": "1", "event": "Bahrain Grand Prix"}])
        return pd.DataFrame([{"driver_number": "2", "event": "Saudi Arabian Grand Prix"}])

    patcher.setattr(pipeline, "process_session", _process)

    combined = pipeline.process_multiple_sessions(
        [_fake_session("FP1"), _fake_session("FP2", event_name="Saudi Arabian Grand Prix")],
        verbose=False,
    )

    assert len(combined) == 2
    assert set(combined["driver_number"]) == {"1", "2"}
