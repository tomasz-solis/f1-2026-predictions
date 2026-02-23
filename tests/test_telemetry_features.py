"""Tests for telemetry feature extraction."""

import pandas as pd

from src.features.telemetry import LapFeatureExtractor, SessionFeatureAggregator


class _FakeLap:
    def __init__(self, telemetry: pd.DataFrame | None):
        self._telemetry = telemetry

    def get_telemetry(self):
        return self._telemetry


class _FakeLaps(pd.DataFrame):
    @property
    def _constructor(self):
        return _FakeLaps

    def pick_fastest(self):
        idx = self["LapTime"].idxmin()
        return self.loc[idx]

    def pick_drivers(self, driver: str):
        return _FakeLaps(self[self["Driver"] == driver].copy())


def _sample_telemetry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Speed": [280, 230, 170, 120, 160, 210, 260, 300],
            "Throttle": [100, 95, 88, 72, 84, 92, 98, 100],
            "Brake": [0, 0, 20, 60, 10, 0, 0, 0],
            "nGear": [8, 7, 6, 4, 5, 6, 7, 8],
            "DRS": [1, 1, 0, 0, 0, 1, 1, 1],
        }
    )


def test_lap_feature_extractor_methods():
    extractor = LapFeatureExtractor()
    telemetry = _sample_telemetry()

    corner = extractor.extract_corner_speeds(telemetry)
    throttle = extractor.extract_throttle_metrics(telemetry)
    braking = extractor.extract_braking_metrics(telemetry)
    straight = extractor.extract_straight_line_speed(telemetry)
    drs = extractor.extract_drs_usage(telemetry)

    assert {"slow_corner_speed", "medium_corner_speed", "high_corner_speed"} <= set(corner)
    assert throttle["pct_full_throttle"] > 0
    assert braking["braking_zones"] >= 1
    assert straight["max_speed"] == 300
    assert drs["drs_active_pct"] > 0


def test_lap_feature_extractor_extract_features_handles_empty():
    extractor = LapFeatureExtractor()
    assert extractor.extract_features(_FakeLap(pd.DataFrame())) == {}
    assert extractor.extract_features(_FakeLap(None)) == {}


def _sample_session_laps() -> _FakeLaps:
    base = pd.to_timedelta([90.0, 90.5, 91.0, 90.2], unit="s")
    return _FakeLaps(
        {
            "IsAccurate": [True, True, True, False],
            "TrackStatus": ["1", "1", "1", "4"],
            "LapTime": base,
            "DriverNumber": ["4", "4", "4", "4"],
            "Driver": ["NOR", "NOR", "NOR", "NOR"],
            "Team": ["McLaren", "McLaren", "McLaren", "McLaren"],
        }
    )


def test_session_feature_aggregator_extract_driver_session(patcher):
    lap_extractor = LapFeatureExtractor()
    aggregator = SessionFeatureAggregator(lap_extractor)

    patcher.setattr(
        lap_extractor,
        "extract_features",
        lambda _lap: {
            "slow_corner_speed": 120.0,
            "avg_speed_full_throttle": 300.0,
            "pct_full_throttle": 50.0,
        },
    )

    driver_features = aggregator.extract_driver_session(_sample_session_laps())

    assert driver_features["driver_code"] == "NOR"
    assert driver_features["clean_laps"] == 3
    assert "fastest_lap" in driver_features
    assert "slow_corner_speed_std" in driver_features


def test_session_feature_aggregator_extract_all_drivers(patcher):
    lap_extractor = LapFeatureExtractor()
    aggregator = SessionFeatureAggregator(lap_extractor)

    patcher.setattr(
        lap_extractor,
        "extract_features",
        lambda _lap: {"slow_corner_speed": 120.0, "avg_speed_full_throttle": 301.0},
    )

    laps = _FakeLaps(
        {
            "IsAccurate": [True, True, True, True],
            "TrackStatus": ["1", "1", "1", "1"],
            "LapTime": pd.to_timedelta([90.0, 91.0, 89.5, 90.5], unit="s"),
            "DriverNumber": ["4", "4", "16", "16"],
            "Driver": ["NOR", "NOR", "LEC", "LEC"],
            "Team": ["McLaren", "McLaren", "Ferrari", "Ferrari"],
        }
    )
    session = type("S", (), {"laps": laps})()

    out = aggregator.extract_all_drivers(session)

    assert len(out) == 2
    assert set(out["driver_code"]) == {"NOR", "LEC"}
