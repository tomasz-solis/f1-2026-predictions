"""Tests for session-order extraction utilities."""

from types import SimpleNamespace

import pandas as pd

from src.extractors import session as session_extractor


def _make_session(*, laps: pd.DataFrame | None = None, results: pd.DataFrame | None = None):
    class _Session:
        def __init__(self, _laps, _results):
            self.laps = _laps
            self.results = _results

        def load(self, laps=False, telemetry=False, weather=False, messages=False):
            return None

    return _Session(laps, results)


def test_extract_fp_order_from_laps_handles_variation_fallback(patcher):
    laps = pd.DataFrame(
        [
            {
                "Team": "McLaren",
                "Driver": "NOR",
                "LapTime": pd.to_timedelta(90.0, unit="s"),
                "IsAccurate": True,
            },
            {
                "Team": "Ferrari",
                "Driver": "LEC",
                "LapTime": pd.to_timedelta(90.4, unit="s"),
                "IsAccurate": True,
            },
            {
                "Team": "Mercedes",
                "Driver": "RUS",
                "LapTime": pd.to_timedelta(90.8, unit="s"),
                "IsAccurate": True,
            },
            {
                "Team": "Red Bull Racing",
                "Driver": "VER",
                "LapTime": pd.to_timedelta(91.0, unit="s"),
                "IsAccurate": True,
            },
            {
                "Team": "Williams",
                "Driver": "ALB",
                "LapTime": pd.to_timedelta(91.2, unit="s"),
                "IsAccurate": True,
            },
        ]
    )

    def _get_session(_year: int, _race: str, variation: str):
        if variation == "FP1":
            raise ValueError("missing")
        return _make_session(laps=laps)

    patcher.setattr(session_extractor.ff1, "get_session", _get_session)

    order = session_extractor.extract_fp_order_from_laps(2026, "Bahrain Grand Prix", "FP1")

    assert order["McLaren"] == 1
    assert len(order) == 5


def test_extract_fp_order_from_laps_requires_minimum_teams(patcher):
    laps = pd.DataFrame(
        [
            {
                "Team": "McLaren",
                "Driver": "NOR",
                "LapTime": pd.to_timedelta(90.0, unit="s"),
                "IsAccurate": True,
            },
            {
                "Team": "Ferrari",
                "Driver": "LEC",
                "LapTime": pd.to_timedelta(90.4, unit="s"),
                "IsAccurate": True,
            },
        ]
    )
    patcher.setattr(
        session_extractor.ff1,
        "get_session",
        lambda *_args, **_kwargs: _make_session(laps=laps),
    )

    assert session_extractor.extract_fp_order_from_laps(2026, "Bahrain Grand Prix", "FP1") is None


def test_extract_quali_order_from_positions(patcher):
    results = pd.DataFrame(
        [
            {"TeamName": "McLaren", "Position": 1},
            {"TeamName": "Ferrari", "Position": 2},
            {"TeamName": "Mercedes", "Position": 3},
            {"TeamName": "Red Bull Racing", "Position": 4},
            {"TeamName": "Williams", "Position": 5},
        ]
    )
    patcher.setattr(
        session_extractor.ff1,
        "get_session",
        lambda *_args, **_kwargs: _make_session(results=results),
    )

    order = session_extractor.extract_quali_order_from_positions(2026, "Bahrain Grand Prix", "Q")

    assert order["McLaren"] == 1
    assert order["Williams"] == 5


def test_extract_session_order_safe_switches_methods(patcher):
    patcher.setattr(session_extractor, "extract_fp_order_from_laps", lambda *_args: {"FP": 1})
    patcher.setattr(
        session_extractor,
        "extract_quali_order_from_positions",
        lambda *_args: {"Q": 1},
    )

    assert session_extractor.extract_session_order_safe(2026, "Bahrain Grand Prix", "FP2") == {
        "FP": 1
    }
    assert session_extractor.extract_session_order_safe(2026, "Bahrain Grand Prix", "Q") == {"Q": 1}


def test_calculate_order_mae():
    predicted = {"McLaren": 1, "Ferrari": 2, "Mercedes": 4}
    actual = {"McLaren": 2, "Ferrari": 2, "Mercedes": 3}

    mae = session_extractor.calculate_order_mae(predicted, actual)
    assert mae == (1 + 0 + 1) / 3


def test_test_session_as_predictor_fixed_failure_path(patcher):
    patcher.setattr(session_extractor, "extract_session_order_safe", lambda *_args: None)

    result = session_extractor.test_session_as_predictor_fixed(
        2026, "Bahrain Grand Prix", "FP2", target_session="Q"
    )

    assert result["status"] == "failed"
    assert "FP2 data not available" in result["reason"]


def test_test_session_as_predictor_fixed_with_driver_metrics(patcher):
    def _extract(_year: int, _race: str, session_type: str):
        if session_type == "FP2":
            return {"McLaren": 1, "Ferrari": 2}
        return {"McLaren": 2, "Ferrari": 1}

    patcher.setattr(session_extractor, "extract_session_order_safe", _extract)

    class _Ranker:
        def predict_positions(self, team_predictions, team_lineups, session_type):
            assert team_predictions == {"McLaren": 1, "Ferrari": 2}
            assert session_type == "qualifying"
            return {
                "predictions": [
                    SimpleNamespace(driver="NOR", position=1),
                    SimpleNamespace(driver="LEC", position=2),
                ]
            }

    result = session_extractor.test_session_as_predictor_fixed(
        2026,
        "Bahrain Grand Prix",
        "FP2",
        target_session="Q",
        driver_ranker=_Ranker(),
        lineups={"McLaren": ["NOR"], "Ferrari": ["LEC"]},
        actual_driver_results=[{"driver": "NOR", "position": 2}, {"driver": "LEC", "position": 1}],
    )

    assert result["status"] == "success"
    assert result["team_mae"] == 1.0
    assert result["driver_mae"] == 1.0
    assert result["driver_within_1"] == 1.0
