"""Tests for FP2 long-run pace extraction."""

import pandas as pd

from src.extractors import race_pace


def _timedelta_series(seconds: list[float]) -> list[pd.Timedelta]:
    return [pd.to_timedelta(s, unit="s") for s in seconds]


def _make_laps_for_team(team: str, driver: str, base: float, n_laps: int = 10) -> pd.DataFrame:
    values = [base + (idx * 0.08) for idx in range(n_laps)]
    return pd.DataFrame(
        {
            "Team": [team] * n_laps,
            "Driver": [driver] * n_laps,
            "LapTime": _timedelta_series(values),
            "Compound": ["MEDIUM"] * n_laps,
        }
    )


def _make_session(laps: pd.DataFrame):
    class _Session:
        def __init__(self, _laps: pd.DataFrame):
            self.laps = _laps

        def load(self, laps=True, telemetry=False, weather=False, messages=False):
            return None

    return _Session(laps)


def test_detect_long_runs_extracts_stint_metrics():
    laps = _make_laps_for_team("McLaren", "NOR", base=90.0, n_laps=12)

    long_runs = race_pace._detect_long_runs(laps)

    assert len(long_runs) == 1
    stint = long_runs[0]
    assert stint["laps"] >= 8
    assert stint["compound"] == "MEDIUM"
    assert stint["avg_pace"] > 0
    assert stint["degradation"] >= 0


def test_detect_long_runs_returns_empty_when_laps_insufficient():
    short_laps = _make_laps_for_team("Ferrari", "LEC", base=90.5, n_laps=6)
    assert race_pace._detect_long_runs(short_laps) == []


def test_select_best_stint_prefers_longest():
    selected = race_pace._select_best_stint(
        [
            {
                "laps": 10,
                "avg_pace": 91.0,
                "degradation": 0.08,
                "compound": "MEDIUM",
                "std_pace": 0.2,
            },
            {
                "laps": 14,
                "avg_pace": 91.2,
                "degradation": 0.07,
                "compound": "HARD",
                "std_pace": 0.3,
            },
        ]
    )

    assert selected["laps"] == 14
    assert selected["compound"] == "HARD"


def test_extract_fp2_pace_returns_relative_team_pace(patcher):
    laps = pd.concat(
        [
            _make_laps_for_team("McLaren", "NOR", base=90.0, n_laps=12),
            _make_laps_for_team("Ferrari", "LEC", base=91.0, n_laps=12),
        ],
        ignore_index=True,
    )

    patcher.setattr(race_pace.ff1, "get_session", lambda *_args, **_kwargs: _make_session(laps))

    result = race_pace.extract_fp2_pace(2026, "Bahrain Grand Prix")

    assert set(result.keys()) == {"McLaren", "Ferrari"}
    assert "relative_pace" in result["McLaren"]
    assert round(result["McLaren"]["relative_pace"] + result["Ferrari"]["relative_pace"], 6) == 0.0


def test_extract_fp2_pace_returns_none_without_laps(patcher):
    session = _make_session(pd.DataFrame())
    session.laps = None
    patcher.setattr(race_pace.ff1, "get_session", lambda *_args, **_kwargs: session)

    assert race_pace.extract_fp2_pace(2026, "Bahrain Grand Prix") is None


def test_extract_fp2_pace_handles_exceptions(patcher):
    patcher.setattr(
        race_pace.ff1,
        "get_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("offline")),
    )

    assert race_pace.extract_fp2_pace(2026, "Bahrain Grand Prix") is None
