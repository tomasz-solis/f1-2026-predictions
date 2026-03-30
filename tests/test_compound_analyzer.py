"""Tests for compound_analyzer module."""

import numpy as np
import pandas as pd
import pytest

from src.systems.compound_analyzer import (
    MIN_LAPS_PER_COMPOUND,
    _calculate_compound_consistency,
    _estimate_compound_tire_deg,
    _median_lap_seconds_series,
    _normalize_compound_name,
    aggregate_compound_samples,
    extract_compound_metrics,
    normalize_compound_metrics_across_teams,
)


@pytest.fixture
def sample_team_laps():
    """Create sample lap data for testing."""
    rng = np.random.default_rng(42)
    laps = []
    # Create 10 laps on SOFT compound with realistic degradation
    for lap_num in range(1, 11):
        base_time = 90.0
        deg = lap_num * 0.05  # 0.05s/lap degradation
        noise = rng.uniform(-0.2, 0.2)
        lap_time = pd.Timedelta(seconds=base_time + deg + noise)

        laps.append(
            {
                "LapNumber": lap_num,
                "LapTime": lap_time,
                "Compound": "SOFT",
                "Driver": "VER",
                "Team": "Red Bull",
                "Stint": 1,
                "IsAccurate": True,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            }
        )

    # Add 9 laps on MEDIUM compound (need MIN_LAPS_PER_COMPOUND)
    for lap_num in range(1, 10):
        base_time = 91.0
        deg = lap_num * 0.03
        noise = rng.uniform(-0.2, 0.2)
        lap_time = pd.Timedelta(seconds=base_time + deg + noise)

        laps.append(
            {
                "LapNumber": lap_num + 10,
                "LapTime": lap_time,
                "Compound": "MEDIUM",
                "Driver": "VER",
                "Team": "Red Bull",
                "Stint": 2,
                "IsAccurate": True,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            }
        )

    return pd.DataFrame(laps)


@pytest.fixture
def sample_team_laps_with_bad_data():
    """Create sample lap data with pit laps and inaccurate laps."""
    laps = []
    for lap_num in range(1, 11):
        is_pit_lap = lap_num == 5
        is_inaccurate = lap_num == 7

        lap_time = pd.Timedelta(seconds=90.0 + lap_num * 0.05)
        pit_in = pd.Timedelta(seconds=100) if is_pit_lap else pd.NaT

        laps.append(
            {
                "LapNumber": lap_num,
                "LapTime": lap_time,
                "Compound": "SOFT",
                "Driver": "VER",
                "Team": "Red Bull",
                "Stint": 1,
                "IsAccurate": not is_inaccurate,
                "PitInTime": pit_in,
                "PitOutTime": pd.NaT,
            }
        )

    return pd.DataFrame(laps)


def test_normalize_compound_name():
    """Test compound name normalization."""
    assert _normalize_compound_name("SOFT") == "SOFT"
    assert _normalize_compound_name("soft") == "SOFT"
    assert _normalize_compound_name("S") == "SOFT"
    assert _normalize_compound_name("MEDIUM") == "MEDIUM"
    assert _normalize_compound_name("M") == "MEDIUM"
    assert _normalize_compound_name("HARD") == "HARD"
    assert _normalize_compound_name("H") == "HARD"
    assert _normalize_compound_name("INTERMEDIATE") == "INTERMEDIATE"
    assert _normalize_compound_name("WET") == "WET"
    assert _normalize_compound_name(None) is None
    assert _normalize_compound_name(pd.NA) is None


def test_median_lap_seconds_series():
    """Test median lap time calculation."""
    times = pd.Series(
        [
            pd.Timedelta(seconds=90.0),
            pd.Timedelta(seconds=90.5),
            pd.Timedelta(seconds=91.0),
        ]
    )
    result = _median_lap_seconds_series(times)
    assert result == pytest.approx(90.5, abs=0.01)


def test_median_lap_seconds_series_empty():
    """Test median calculation with empty series."""
    times = pd.Series([])
    result = _median_lap_seconds_series(times)
    assert result is None


def test_estimate_compound_tire_deg(sample_team_laps):
    """Test tire degradation estimation."""
    soft_laps = sample_team_laps[sample_team_laps["Compound"] == "SOFT"]
    deg = _estimate_compound_tire_deg(soft_laps)

    # Should detect roughly 0.05 s/lap degradation
    assert deg is not None
    assert 0.03 < deg < 0.08  # Allow some variance from noise


def test_estimate_compound_tire_deg_insufficient_data():
    """Test degradation with insufficient data."""
    laps = pd.DataFrame(
        {
            "LapNumber": [1, 2],
            "LapTime": [pd.Timedelta(seconds=90), pd.Timedelta(seconds=90.1)],
            "Stint": [1, 1],
            "Driver": ["VER", "VER"],
        }
    )
    deg = _estimate_compound_tire_deg(laps)
    assert deg is None


def test_calculate_compound_consistency(sample_team_laps):
    """Test consistency calculation."""
    soft_laps = sample_team_laps[sample_team_laps["Compound"] == "SOFT"]
    consistency = _calculate_compound_consistency(soft_laps)

    # Should return standard deviation
    assert consistency is not None
    assert consistency > 0


def test_extract_compound_metrics_basic(sample_team_laps):
    """Test basic compound metrics extraction."""
    metrics = extract_compound_metrics(sample_team_laps, "Red Bull", "Bahrain")

    assert "SOFT" in metrics
    assert "MEDIUM" in metrics

    soft = metrics["SOFT"]
    assert soft["laps_count"] == 10.0
    assert soft["track_name"] == "Bahrain"
    assert soft["median_lap_time"] is not None
    assert soft["tire_deg_slope"] is not None
    assert soft["consistency"] is not None


def test_extract_compound_metrics_filters_bad_laps(sample_team_laps_with_bad_data):
    """Test that pit laps and inaccurate laps are filtered out."""
    metrics = extract_compound_metrics(sample_team_laps_with_bad_data, "Red Bull", "Bahrain")

    # Should have SOFT compound
    assert "SOFT" in metrics

    # Should have filtered out pit lap (5) and inaccurate lap (7)
    # So 10 - 2 = 8 laps
    assert metrics["SOFT"]["laps_count"] == 8.0


def test_extract_compound_metrics_insufficient_laps():
    """Test that compounds with insufficient laps are skipped."""
    laps = pd.DataFrame(
        {
            "LapNumber": range(1, MIN_LAPS_PER_COMPOUND),
            "LapTime": [
                pd.Timedelta(seconds=90 + i * 0.05) for i in range(MIN_LAPS_PER_COMPOUND - 1)
            ],
            "Compound": ["SOFT"] * (MIN_LAPS_PER_COMPOUND - 1),
            "Driver": ["VER"] * (MIN_LAPS_PER_COMPOUND - 1),
            "Team": ["Red Bull"] * (MIN_LAPS_PER_COMPOUND - 1),
            "Stint": [1] * (MIN_LAPS_PER_COMPOUND - 1),
            "IsAccurate": [True] * (MIN_LAPS_PER_COMPOUND - 1),
            "PitInTime": [pd.NaT] * (MIN_LAPS_PER_COMPOUND - 1),
            "PitOutTime": [pd.NaT] * (MIN_LAPS_PER_COMPOUND - 1),
        }
    )

    metrics = extract_compound_metrics(laps, "Red Bull", "Bahrain")

    # Should skip due to insufficient laps
    assert "SOFT" not in metrics


def test_extract_compound_metrics_empty_dataframe():
    """Test with empty DataFrame."""
    empty_df = pd.DataFrame()
    metrics = extract_compound_metrics(empty_df, "Red Bull", "Bahrain")
    assert metrics == {}


def test_normalize_compound_metrics_across_teams():
    """Test track-specific normalization across teams."""
    all_metrics = {
        "Red Bull": {
            "SOFT": {
                "median_lap_time": 90.0,
                "tire_deg_slope": 0.10,
                "consistency": 0.5,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Mercedes": {
            "SOFT": {
                "median_lap_time": 91.0,
                "tire_deg_slope": 0.15,
                "consistency": 0.6,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
    }

    normalized = normalize_compound_metrics_across_teams(all_metrics, "Bahrain")

    # Red Bull should have best pace (1.0), Mercedes worst (0.0)
    assert normalized["Red Bull"]["SOFT"]["pace_performance"] == pytest.approx(1.0)
    assert normalized["Mercedes"]["SOFT"]["pace_performance"] == pytest.approx(0.0)

    # Red Bull should have best tire deg (1.0), Mercedes worst (0.0)
    assert normalized["Red Bull"]["SOFT"]["tire_deg_performance"] == pytest.approx(1.0)
    assert normalized["Mercedes"]["SOFT"]["tire_deg_performance"] == pytest.approx(0.0)


def test_normalize_compound_metrics_uses_rank_scale_for_outliers():
    """One slow outlier should not flatten the midfield compound scores."""
    all_metrics = {
        "Team A": {
            "SOFT": {
                "median_lap_time": 90.0,
                "tire_deg_slope": 0.11,
                "consistency": 0.45,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Team B": {
            "SOFT": {
                "median_lap_time": 90.8,
                "tire_deg_slope": 0.12,
                "consistency": 0.48,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Team C": {
            "SOFT": {
                "median_lap_time": 91.0,
                "tire_deg_slope": 0.14,
                "consistency": 0.50,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Team D": {
            "SOFT": {
                "median_lap_time": 97.0,
                "tire_deg_slope": 0.40,
                "consistency": 1.20,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
    }

    normalized = normalize_compound_metrics_across_teams(all_metrics, "Bahrain")

    assert normalized["Team A"]["SOFT"]["pace_performance"] == pytest.approx(1.0)
    assert normalized["Team D"]["SOFT"]["pace_performance"] == pytest.approx(0.0)
    assert normalized["Team B"]["SOFT"]["pace_performance"] > 0.5
    assert normalized["Team C"]["SOFT"]["pace_performance"] > 0.3


def test_normalize_compound_metrics_handles_ties():
    """Teams with identical compound values should land on the same score."""
    all_metrics = {
        "Team A": {
            "SOFT": {
                "median_lap_time": 90.0,
                "tire_deg_slope": 0.10,
                "consistency": 0.50,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Team B": {
            "SOFT": {
                "median_lap_time": 90.0,
                "tire_deg_slope": 0.10,
                "consistency": 0.50,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Team C": {
            "SOFT": {
                "median_lap_time": 91.5,
                "tire_deg_slope": 0.20,
                "consistency": 0.80,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
    }

    normalized = normalize_compound_metrics_across_teams(all_metrics, "Bahrain")

    assert (
        normalized["Team A"]["SOFT"]["pace_performance"]
        == normalized["Team B"]["SOFT"]["pace_performance"]
    )
    assert normalized["Team C"]["SOFT"]["pace_performance"] == pytest.approx(0.0)


def test_normalize_compound_metrics_different_tracks():
    """Test that different tracks are not normalized together."""
    all_metrics = {
        "Red Bull": {
            "SOFT": {
                "median_lap_time": 90.0,
                "track_name": "Bahrain",
                "laps_count": 10.0,
            }
        },
        "Mercedes": {
            "SOFT": {
                "median_lap_time": 70.0,  # Monaco is faster
                "track_name": "Monaco",
                "laps_count": 10.0,
            }
        },
    }

    # Normalize for Bahrain
    normalized = normalize_compound_metrics_across_teams(all_metrics, "Bahrain")

    # Only Red Bull should be normalized (Bahrain track)
    assert "Red Bull" in normalized
    # Mercedes Monaco data should not affect normalization


def test_aggregate_compound_samples_new_data():
    """Test aggregation with no existing data."""
    existing = {}
    new = {
        "SOFT": {
            "pace_performance": 0.8,
            "tire_deg_performance": 0.7,
            "laps_count": 10,
            "track_name": "Bahrain",
        }
    }

    result = aggregate_compound_samples(existing, new, blend_weight=0.5, race_name="Bahrain")

    assert "SOFT" in result
    assert result["SOFT"]["pace_performance"] == 0.8
    assert result["SOFT"]["laps_sampled"] == 10
    assert result["SOFT"]["sessions_used"] == 1


def test_aggregate_compound_samples_blending():
    """Test blending of existing and new data."""
    existing = {
        "SOFT": {
            "pace_performance": 0.6,
            "tire_deg_performance": 0.5,
            "laps_sampled": 20,
            "sessions_used": 2,
            "track_name": "Bahrain",
        }
    }
    new = {
        "SOFT": {
            "pace_performance": 0.8,
            "tire_deg_performance": 0.9,
            "laps_count": 10,
            "track_name": "Bahrain",
        }
    }

    # 50% blend: (0.5 * 0.6) + (0.5 * 0.8) = 0.7
    result = aggregate_compound_samples(existing, new, blend_weight=0.5, race_name="Bahrain")

    assert result["SOFT"]["pace_performance"] == pytest.approx(0.7)
    assert result["SOFT"]["tire_deg_performance"] == pytest.approx(0.7)
    assert result["SOFT"]["laps_sampled"] == 30
    assert result["SOFT"]["sessions_used"] == 3


def test_aggregate_compound_samples_different_tracks():
    """Test that different tracks don't blend together."""
    existing = {
        "SOFT": {
            "pace_performance": 0.6,
            "track_name": "Monaco",
            "laps_sampled": 20,
            "sessions_used": 2,
        }
    }
    new = {
        "SOFT": {
            "pace_performance": 0.8,
            "track_name": "Bahrain",
            "laps_count": 10,
        }
    }

    result = aggregate_compound_samples(existing, new, blend_weight=0.5, race_name="Bahrain")

    # Should use new data only (different tracks)
    assert result["SOFT"]["pace_performance"] == 0.8
    assert result["SOFT"]["track_name"] == "Bahrain"
    assert result["SOFT"]["laps_sampled"] == 10
    assert result["SOFT"]["sessions_used"] == 1


def test_aggregate_compound_samples_handles_none_values():
    """Test that None values in metrics are handled correctly."""
    existing = {
        "SOFT": {
            "pace_performance": 0.6,
            "tire_deg_performance": None,
            "laps_sampled": 20,
            "track_name": "Bahrain",
        }
    }
    new = {
        "SOFT": {
            "pace_performance": 0.8,
            "tire_deg_performance": 0.7,
            "laps_count": 10,
            "track_name": "Bahrain",
        }
    }

    result = aggregate_compound_samples(existing, new, blend_weight=0.5, race_name="Bahrain")

    # Pace should blend
    assert result["SOFT"]["pace_performance"] == pytest.approx(0.7)
    # Tire deg should use new value (old was None)
    assert result["SOFT"]["tire_deg_performance"] == 0.7
