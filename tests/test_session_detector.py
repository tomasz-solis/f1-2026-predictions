"""Tests for session completion detection."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from src.utils.session_detector import SessionDetector


class _FakeEvent:
    def __init__(self, session_date: datetime | None):
        self._session_date = session_date

    def get_session_date(self, _session_name: str):
        return self._session_date


def test_is_session_completed_returns_false_before_session_end():
    detector = SessionDetector()
    future_start = datetime.now(UTC) + timedelta(hours=1)

    with patch(
        "src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(future_start)
    ):
        with patch("src.utils.session_detector.fastf1.get_session") as mock_get_session:
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "Q") is False

    mock_get_session.assert_not_called()


def test_is_session_completed_practice_requires_non_empty_laps():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=4)
    mock_session = MagicMock()
    mock_session.laps = pd.DataFrame([{"LapTime": pd.Timedelta(seconds=90)}])

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=mock_session):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "FP1") is True


def test_is_session_completed_practice_without_laps_uses_elapsed_fallback():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=3)

    session = MagicMock()
    session.laps = pd.DataFrame()
    session.session_status = pd.DataFrame()

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=session):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "FP1") is True


def test_is_session_completed_practice_load_failure_uses_elapsed_fallback():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=3)

    session = MagicMock()
    session.load.side_effect = RuntimeError("temporary FastF1 lapse")

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=session):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "FP2") is True


def test_is_session_completed_competitive_requires_results():
    past_start = datetime.now(UTC) - timedelta(hours=4)

    empty_results_session = MagicMock()
    empty_results_session.results = pd.DataFrame()
    detector = SessionDetector()
    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch(
            "src.utils.session_detector.fastf1.get_session", return_value=empty_results_session
        ):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "Q") is False

    populated_results_session = MagicMock()
    populated_results_session.results = pd.DataFrame([{"Position": 1}])
    detector = SessionDetector()
    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch(
            "src.utils.session_detector.fastf1.get_session", return_value=populated_results_session
        ):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "Q") is True


def test_get_completed_sessions_uses_per_session_validation(patcher):
    detector = SessionDetector()

    def _mock_completion_state(year: int, race_name: str, session_name: str) -> str:
        return "completed" if session_name in {"FP1", "FP3"} else "incomplete"

    patcher.setattr(detector, "get_session_completion_state", _mock_completion_state)
    completed = detector.get_completed_sessions(2026, "Bahrain Grand Prix", is_sprint=False)

    assert completed == ["FP1", "FP3"]


def test_is_session_completed_practice_uses_final_status_when_available():
    detector = SessionDetector()
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    session = MagicMock()
    session.laps = pd.DataFrame([{"LapTime": pd.Timedelta(seconds=90)}])
    session.session_status = pd.DataFrame({"Status": ["Finished"]})

    with patch(
        "src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(recent_start)
    ):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=session):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "FP1") is True


def test_is_session_completed_practice_blocks_active_status_with_laps():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=2)

    session = MagicMock()
    session.laps = pd.DataFrame([{"LapTime": pd.Timedelta(seconds=90)}])
    session.session_status = pd.DataFrame({"Status": ["Started"]})

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=session):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "FP2") is False


def test_is_session_completed_competitive_blocks_active_status_even_with_results():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=3)

    session = MagicMock()
    session.results = pd.DataFrame([{"Position": 1}])
    session.session_status = pd.DataFrame({"Status": ["Running"]})

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=session):
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "Q") is False


def test_get_session_completion_state_returns_unknown_on_fastf1_error():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=3)

    broken_session = MagicMock()
    broken_session.load.side_effect = RuntimeError("api timeout")

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=broken_session):
            assert (
                detector.get_session_completion_state(2026, "Bahrain Grand Prix", "Q") == "unknown"
            )
            assert detector.is_session_completed(2026, "Bahrain Grand Prix", "Q") is False


def test_is_session_completed_competitive_ignores_unloaded_session_status():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=4)

    class _CompetitiveSessionWithoutStatus:
        """Competitive session stub with results loaded but no status feed."""

        def __init__(self) -> None:
            self.results = pd.DataFrame([{"Position": 1}])

        def load(self, **_kwargs) -> None:
            """Simulate a successful FastF1 load call."""

        @property
        def session_status(self):
            """Match FastF1 behavior when status data was never loaded."""
            raise DataNotLoadedError("The data you are trying to access has not been loaded yet.")

    session = _CompetitiveSessionWithoutStatus()

    with patch("src.utils.session_detector.fastf1.get_event", return_value=_FakeEvent(past_start)):
        with patch("src.utils.session_detector.fastf1.get_session", return_value=session):
            assert detector.get_session_completion_state(2026, "Australian Grand Prix", "Q") == (
                "completed"
            )
            assert detector.is_session_completed(2026, "Australian Grand Prix", "Q") is True


def test_get_completed_sessions_uses_single_event_lookup_per_race():
    detector = SessionDetector()
    past_start = datetime.now(UTC) - timedelta(hours=4)

    mock_session = MagicMock()
    mock_session.laps = pd.DataFrame([{"LapTime": pd.Timedelta(seconds=90)}])
    mock_session.session_status = pd.DataFrame({"Status": ["Finished"]})

    with patch(
        "src.utils.session_detector.fastf1.get_event",
        return_value=_FakeEvent(past_start),
    ) as mock_get_event:
        with patch("src.utils.session_detector.fastf1.get_session", return_value=mock_session):
            completed = detector.get_completed_sessions(2026, "Bahrain Grand Prix", is_sprint=False)

    assert completed == ["FP1", "FP2", "FP3"]
    assert mock_get_event.call_count == 1
