"""Tests for qualifying-stage-aware FP blending session selection."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.fp_blending import _circuit_breaker, get_best_fp_performance


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker before each test to avoid cross-test contamination."""
    _circuit_breaker.reset()
    yield
    _circuit_breaker.reset()


def test_sprint_qualifying_stage_uses_fp1_only():
    """Sprint-qualifying context should only probe FP1 on sprint weekends."""
    calls = []

    class _NoScheduleEvent:
        def get_session_date(self, _session_name: str):
            return None

    def _mock_get_fp_team_performance(year, race_name, session_type):
        calls.append(session_type)
        if session_type == "FP1":
            return {"McLaren": 0.9}, pd.DataFrame({"Driver": ["NOR"]}), None
        return None, None, None

    with patch("src.utils.fp_blending.ff1.get_event", return_value=_NoScheduleEvent()):
        with patch(
            "src.utils.fp_blending.get_fp_team_performance",
            side_effect=_mock_get_fp_team_performance,
        ):
            session_label, perf, laps = get_best_fp_performance(
                year=2026,
                race_name="Chinese Grand Prix",
                is_sprint=True,
                qualifying_stage="sprint",
            )

    assert calls == ["FP1"]
    assert session_label == "FP1 short-stint"
    assert perf == {"McLaren": 0.9}
    assert laps is not None


def test_main_qualifying_stage_blends_sq_sprint_and_fp1():
    """Main-qualifying context should blend all available short-stint signals."""
    calls = []

    class _NoScheduleEvent:
        def get_session_date(self, _session_name: str):
            return None

    def _mock_get_fp_team_performance(year, race_name, session_type):
        calls.append(session_type)
        if session_type == "Sprint Qualifying":
            return (
                {
                    "Mercedes": 1.0,
                    "Ferrari": 0.6,
                },
                pd.DataFrame({"Driver": ["RUS"]}),
                None,
            )
        if session_type == "Sprint":
            return (
                {
                    "Mercedes": 0.6,
                    "Ferrari": 0.9,
                },
                pd.DataFrame({"Driver": ["HAM"]}),
                None,
            )
        if session_type == "FP1":
            return (
                {
                    "Mercedes": 0.8,
                    "Ferrari": 0.7,
                },
                pd.DataFrame({"Driver": ["ANT"]}),
                None,
            )
        return None, None, None

    with patch("src.utils.fp_blending.ff1.get_event", return_value=_NoScheduleEvent()):
        with patch(
            "src.utils.fp_blending.get_fp_team_performance",
            side_effect=_mock_get_fp_team_performance,
        ):
            session_label, perf, laps = get_best_fp_performance(
                year=2026,
                race_name="Chinese Grand Prix",
                is_sprint=True,
                qualifying_stage="main",
            )

    assert calls == ["Sprint Qualifying", "Sprint", "FP1"]
    assert session_label == "Short-stint blend (Sprint Qualifying + Sprint + FP1)"
    assert perf["Mercedes"] == pytest.approx((1.0 + (0.6 * 0.55) + (0.8 * 0.70)) / 2.25)
    assert perf["Ferrari"] == pytest.approx((0.6 + (0.9 * 0.55) + (0.7 * 0.70)) / 2.25)
    assert laps is not None


def test_invalid_qualifying_stage_raises_value_error():
    with pytest.raises(ValueError, match="qualifying_stage"):
        get_best_fp_performance(
            year=2026,
            race_name="Chinese Grand Prix",
            is_sprint=True,
            qualifying_stage="invalid",
        )


def test_main_stage_skips_future_session_before_fetching_laps():
    """Future/in-progress sessions should be skipped before expensive FastF1 lap loading."""
    calls = []
    now = datetime.now(UTC)

    class _FakeEvent:
        def get_session_date(self, session_name: str):
            if session_name == "FP3":
                return now + timedelta(hours=1)
            if session_name == "FP2":
                return now - timedelta(hours=4)
            if session_name == "FP1":
                return now - timedelta(hours=7)
            return None

    def _mock_get_fp_team_performance(year, race_name, session_type):
        calls.append(session_type)
        if session_type == "FP2":
            return {"McLaren": 0.9, "Ferrari": 0.7}, pd.DataFrame({"Driver": ["NOR"]}), None
        if session_type == "FP1":
            return {"McLaren": 0.7, "Ferrari": 0.9}, pd.DataFrame({"Driver": ["PIA"]}), None
        return None, None, None

    with patch("src.utils.fp_blending.ff1.get_event", return_value=_FakeEvent()):
        with patch(
            "src.utils.fp_blending.get_fp_team_performance",
            side_effect=_mock_get_fp_team_performance,
        ):
            session_label, perf, laps = get_best_fp_performance(
                year=2026,
                race_name="Chinese Grand Prix",
                is_sprint=False,
                qualifying_stage="main",
            )

    assert calls == ["FP2", "FP1"]
    assert session_label == "Short-stint blend (FP2 + FP1)"
    assert perf is not None
    assert laps is not None
