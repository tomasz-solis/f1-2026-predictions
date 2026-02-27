"""
Tests for src/utils/auto_updater.py - Automatic race detection and updating

Critical path testing for dashboard auto-update functionality.
"""

import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir)

    # Create directory structure
    char_dir = data_dir / "processed" / "car_characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)

    # Create initial characteristics file
    char_file = char_dir / "2026_car_characteristics.json"
    initial_data = {
        "year": 2026,
        "version": 1,
        "races_completed": 0,
        "last_updated": "2026-01-01T00:00:00",
        "last_learned_race": None,
        "teams": {},
    }

    with open(char_file, "w") as f:
        json.dump(initial_data, f, indent=2)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestAutoUpdaterDetection:
    """Test automatic race completion detection."""

    def test_needs_update_no_races(self, temp_data_dir):
        """Test needs_update returns False when no races completed."""
        from src.utils.auto_updater import needs_update

        with patch("src.utils.auto_updater.get_completed_races") as mock_completed:
            mock_completed.return_value = []

            with patch("src.utils.auto_updater.get_learned_races") as mock_learned:
                mock_learned.return_value = []

                result, new_races = needs_update()

        assert result is False
        assert new_races == []

    def test_needs_update_with_new_race(self, temp_data_dir):
        """Test needs_update returns True when new race completed."""
        from src.utils.auto_updater import needs_update

        with patch("src.utils.auto_updater.get_completed_races") as mock_completed:
            mock_completed.return_value = ["Bahrain Grand Prix"]

            with patch("src.utils.auto_updater.get_learned_races") as mock_learned:
                mock_learned.return_value = []

                result, new_races = needs_update()

        assert result is True
        assert "Bahrain Grand Prix" in new_races

    def test_needs_update_all_learned(self, temp_data_dir):
        """Test needs_update returns False when all races already learned."""
        from src.utils.auto_updater import needs_update

        with patch("src.utils.auto_updater.get_completed_races") as mock_completed:
            mock_completed.return_value = [
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
            ]

            with patch("src.utils.auto_updater.get_learned_races") as mock_learned:
                mock_learned.return_value = [
                    "Bahrain Grand Prix",
                    "Saudi Arabian Grand Prix",
                ]

                result, new_races = needs_update()

        assert result is False
        assert new_races == []

    def test_needs_update_passes_year_to_sources(self, temp_data_dir):
        from src.utils.auto_updater import needs_update

        with patch("src.utils.auto_updater.get_completed_races") as mock_completed:
            mock_completed.return_value = []
            with patch("src.utils.auto_updater.get_learned_races") as mock_learned:
                mock_learned.return_value = []
                needs_update(year=2027)

        mock_completed.assert_called_once_with(year=2027)
        mock_learned.assert_called_once_with(year=2027)


class TestAutoUpdaterExecution:
    """Test automatic update execution."""

    def test_auto_update_from_races_success(self, temp_data_dir):
        """Test successful auto-update from completed races."""
        from src.utils.auto_updater import auto_update_from_races

        with patch("src.utils.auto_updater.needs_update") as mock_needs:
            mock_needs.return_value = (True, ["Bahrain Grand Prix"])

            with patch("src.systems.updater.update_from_race") as mock_update:
                with patch("src.utils.auto_updater.mark_race_as_learned"):

                    def progress_callback(current, total, message):
                        pass  # Mock progress callback

                    updated_count = auto_update_from_races(progress_callback=progress_callback)

        assert updated_count == 1
        mock_update.assert_called_once_with(2026, "Bahrain Grand Prix")

    def test_auto_update_multiple_races(self, temp_data_dir):
        """Test auto-update handles multiple races."""
        from src.utils.auto_updater import auto_update_from_races

        with patch("src.utils.auto_updater.needs_update") as mock_needs:
            mock_needs.return_value = (
                True,
                [
                    "Bahrain Grand Prix",
                    "Saudi Arabian Grand Prix",
                    "Australian Grand Prix",
                ],
            )

            with patch("src.systems.updater.update_from_race") as mock_update:
                with patch("src.utils.auto_updater.mark_race_as_learned"):

                    def progress_callback(current, total, message):
                        assert total == 3

                    updated_count = auto_update_from_races(progress_callback=progress_callback)

        assert updated_count == 3
        assert mock_update.call_count == 3

    def test_auto_update_handles_failure(self, temp_data_dir):
        """Test auto-update continues on single race failure."""
        from src.utils.auto_updater import auto_update_from_races

        with patch("src.utils.auto_updater.needs_update") as mock_needs:
            mock_needs.return_value = (
                True,
                ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"],
            )

            with patch("src.systems.updater.update_from_race") as mock_update:
                with patch("src.utils.auto_updater.mark_race_as_learned"):
                    # First update succeeds, second fails
                    mock_update.side_effect = [None, Exception("Update failed")]

                    def progress_callback(current, total, message):
                        pass

                    updated_count = auto_update_from_races(progress_callback=progress_callback)

        # Should have updated 1 race (first one) before failure
        assert updated_count == 1

    def test_auto_update_from_races_passes_year_to_update_calls(self, temp_data_dir):
        from src.utils.auto_updater import auto_update_from_races

        with patch("src.utils.auto_updater.needs_update") as mock_needs:
            mock_needs.return_value = (True, ["Bahrain Grand Prix"])
            with patch("src.systems.updater.update_from_race") as mock_update:
                with patch("src.utils.auto_updater.mark_race_as_learned"):
                    updated_count = auto_update_from_races(year=2027)

        assert updated_count == 1
        mock_update.assert_called_once_with(2027, "Bahrain Grand Prix")

    def test_auto_update_from_races_uses_explicit_race_list(self, temp_data_dir):
        """Explicit race lists should bypass needs_update recomputation."""
        from src.utils.auto_updater import auto_update_from_races

        with patch(
            "src.utils.auto_updater.needs_update",
            side_effect=AssertionError("needs_update should not be called"),
        ):
            with patch("src.systems.updater.update_from_race") as mock_update:
                with patch("src.utils.auto_updater.mark_race_as_learned"):
                    updated_count = auto_update_from_races(
                        races_to_update=[
                            "Bahrain Grand Prix",
                            "Bahrain Grand Prix",
                            "Saudi Arabian Grand Prix",
                        ]
                    )

        assert updated_count == 2
        assert mock_update.call_count == 2
        assert mock_update.call_args_list[0].args == (2026, "Bahrain Grand Prix")
        assert mock_update.call_args_list[1].args == (2026, "Saudi Arabian Grand Prix")


class TestCompletedRacesDetection:
    """Test detection of completed races from FastF1."""

    def test_get_completed_races_recent(self, temp_data_dir):
        """Test get_completed_races returns only recent races."""
        from src.utils.auto_updater import get_completed_races

        # Mock FastF1 schedule
        mock_schedule = pd.DataFrame(
            {
                "EventName": [
                    "Bahrain Grand Prix",
                    "Saudi Arabian Grand Prix",
                    "Australian Grand Prix",
                ],
                "EventDate": [
                    datetime.now() - timedelta(days=10),  # Completed
                    datetime.now() - timedelta(days=5),  # Completed
                    datetime.now() + timedelta(days=5),  # Future
                ],
            }
        )

        with patch("fastf1.get_event_schedule") as mock_get_schedule:
            with patch("fastf1.get_session") as mock_get_session:
                mock_get_schedule.return_value = mock_schedule
                mock_session = MagicMock()
                mock_session.results = pd.DataFrame([{"Position": 1}])
                mock_get_session.return_value = mock_session

                completed = get_completed_races(year=2026)

        assert len(completed) == 2
        assert "Bahrain Grand Prix" in completed
        assert "Saudi Arabian Grand Prix" in completed
        assert "Australian Grand Prix" not in completed

    def test_get_completed_races_handles_timezone_aware_dates(self, temp_data_dir):
        """Timezone-aware event dates should be handled without naive/aware comparison errors."""
        from src.utils.auto_updater import get_completed_races

        mock_schedule = pd.DataFrame(
            {
                "EventName": ["Bahrain Grand Prix", "Australian Grand Prix"],
                "EventDate": [
                    datetime.now(UTC) - timedelta(days=2),
                    datetime.now(UTC) + timedelta(days=2),
                ],
            }
        )

        with patch("fastf1.get_event_schedule") as mock_get_schedule:
            with patch("fastf1.get_session") as mock_get_session:
                mock_get_schedule.return_value = mock_schedule
                mock_session = MagicMock()
                mock_session.results = pd.DataFrame([{"Position": 1}])
                mock_get_session.return_value = mock_session

                completed = get_completed_races(year=2026)

        assert completed == ["Bahrain Grand Prix"]

    def test_get_completed_races_requires_session_load(self, temp_data_dir):
        """Race is treated as completed only after FastF1 session load succeeds."""
        from src.utils.auto_updater import get_completed_races

        mock_schedule = pd.DataFrame(
            {
                "EventName": ["Bahrain Grand Prix"],
                "EventDate": [datetime.now() - timedelta(days=2)],
            }
        )
        mock_session = MagicMock()
        mock_session.results = pd.DataFrame([{"Position": 1}])

        with patch("fastf1.get_event_schedule") as mock_get_schedule:
            with patch("fastf1.get_session") as mock_get_session:
                mock_get_schedule.return_value = mock_schedule
                mock_get_session.return_value = mock_session

                completed = get_completed_races(year=2026)

        assert completed == ["Bahrain Grand Prix"]
        mock_session.load.assert_called_once_with(
            laps=False,
            telemetry=False,
            weather=False,
            messages=False,
        )

    def test_get_completed_races_skips_when_session_load_fails(self, temp_data_dir):
        """Session load failures should not mark race as completed."""
        from src.utils.auto_updater import get_completed_races

        mock_schedule = pd.DataFrame(
            {
                "EventName": ["Bahrain Grand Prix"],
                "EventDate": [datetime.now() - timedelta(days=2)],
            }
        )
        mock_session = MagicMock()
        mock_session.load.side_effect = RuntimeError("Data not yet loaded")

        with patch("fastf1.get_event_schedule") as mock_get_schedule:
            with patch("fastf1.get_session") as mock_get_session:
                mock_get_schedule.return_value = mock_schedule
                mock_get_session.return_value = mock_session

                completed = get_completed_races(year=2026)

        assert completed == []

    def test_get_completed_races_skips_when_results_missing(self, temp_data_dir):
        """Session metadata loads are ignored when results payload is empty."""
        from src.utils.auto_updater import get_completed_races

        mock_schedule = pd.DataFrame(
            {
                "EventName": ["Bahrain Grand Prix"],
                "EventDate": [datetime.now() - timedelta(days=2)],
            }
        )
        mock_session = MagicMock()
        mock_session.results = pd.DataFrame()

        with patch("fastf1.get_event_schedule") as mock_get_schedule:
            with patch("fastf1.get_session") as mock_get_session:
                mock_get_schedule.return_value = mock_schedule
                mock_get_session.return_value = mock_session

                completed = get_completed_races(year=2026)

        assert completed == []

    def test_get_completed_races_excludes_testing_events(self, temp_data_dir):
        """Testing events should never be treated as completed races to learn from."""
        from src.utils.auto_updater import get_completed_races

        mock_schedule = pd.DataFrame(
            {
                "EventName": [
                    "Pre-Season Testing",
                    "Bahrain Grand Prix",
                ],
                "EventFormat": [
                    "testing",
                    "conventional",
                ],
                "RoundNumber": [0, 1],
                "EventDate": [
                    datetime.now() - timedelta(days=10),
                    datetime.now() - timedelta(days=5),
                ],
            }
        )

        with patch("fastf1.get_event_schedule") as mock_get_schedule:
            with patch("fastf1.get_session") as mock_get_session:
                mock_get_schedule.return_value = mock_schedule
                mock_session = MagicMock()
                mock_session.results = pd.DataFrame([{"Position": 1}])
                mock_get_session.return_value = mock_session

                completed = get_completed_races(year=2026)

        assert completed == ["Bahrain Grand Prix"]


class TestLearnedRacesTracking:
    """Test tracking of already-learned races."""

    def test_get_learned_races_from_file(self, temp_data_dir):
        """Test get_learned_races reads from learning_state.json."""
        from src.utils.auto_updater import get_learned_races

        # Create learning_state.json with learned races
        learning_file = Path("data/learning_state.json")
        learning_file.parent.mkdir(parents=True, exist_ok=True)

        learning_data = {
            "history": [
                {"race": "Bahrain Grand Prix", "date": "2026-03-15"},
                {"race": "Saudi Arabian Grand Prix", "date": "2026-03-22"},
            ]
        }

        with open(learning_file, "w") as f:
            json.dump(learning_data, f, indent=2)

        try:
            learned = get_learned_races()

            # Should have 2 learned races
            assert len(learned) == 2
            assert "Bahrain Grand Prix" in learned
            assert "Saudi Arabian Grand Prix" in learned
        finally:
            # Cleanup
            if learning_file.exists():
                learning_file.unlink()

    def test_get_learned_races_empty(self, temp_data_dir):
        """Test get_learned_races returns empty when no races learned."""
        from src.utils.auto_updater import get_learned_races

        learned = get_learned_races()

        assert learned == []

    def test_mark_race_as_learned_recovers_from_corrupted_state(self, temp_data_dir):
        """Corrupted learning state should be rebuilt instead of crashing."""
        from src.utils.auto_updater import mark_race_as_learned

        learning_file = Path("data/learning_state.json")
        learning_file.parent.mkdir(parents=True, exist_ok=True)
        learning_file.write_text("{not: valid json")

        try:
            mark_race_as_learned("Bahrain Grand Prix")

            with open(learning_file) as f:
                repaired = json.load(f)
            assert any(entry.get("race") == "Bahrain Grand Prix" for entry in repaired["history"])
        finally:
            if learning_file.exists():
                learning_file.unlink()

    def test_mark_race_as_learned_yearless_legacy_entry_does_not_block_new_year(
        self, temp_data_dir
    ):
        """Legacy yearless entries should not be treated as duplicates for a different season."""
        from src.utils.auto_updater import mark_race_as_learned

        learning_file = Path("data/learning_state.json")
        learning_file.parent.mkdir(parents=True, exist_ok=True)
        learning_file.write_text(
            json.dumps(
                {
                    "season": 2026,
                    "history": [{"race": "Bahrain Grand Prix", "date": "2026-03-15T00:00:00"}],
                    "races_completed": 1,
                }
            )
        )

        try:
            mark_race_as_learned("Bahrain Grand Prix", year=2027)

            with open(learning_file) as f:
                state = json.load(f)

            history = state.get("history", [])
            assert len(history) == 2
            assert any(
                entry.get("race") == "Bahrain Grand Prix" and entry.get("year") == 2027
                for entry in history
            )
        finally:
            if learning_file.exists():
                learning_file.unlink()

    def test_mark_race_as_learned_yearless_entry_ignores_mutated_state_season(self, temp_data_dir):
        """Yearless legacy rows must stay mapped to default season even after season rollover."""
        from src.utils.auto_updater import mark_race_as_learned

        learning_file = Path("data/learning_state.json")
        learning_file.parent.mkdir(parents=True, exist_ok=True)
        learning_file.write_text(
            json.dumps(
                {
                    "season": 2027,
                    "history": [
                        {"race": "Saudi Arabian Grand Prix", "date": "2026-03-22T00:00:00"}
                    ],
                    "races_completed": 1,
                }
            )
        )

        try:
            mark_race_as_learned("Saudi Arabian Grand Prix", year=2027)

            with open(learning_file) as f:
                state = json.load(f)

            history = state.get("history", [])
            assert len(history) == 2
            assert any(
                entry.get("race") == "Saudi Arabian Grand Prix" and entry.get("year") == 2027
                for entry in history
            )
        finally:
            if learning_file.exists():
                learning_file.unlink()

    def test_get_learned_races_prefers_runtime_state_store_when_db_reads_enabled(self):
        from src.utils.auto_updater import get_learned_races

        class _Store:
            def get_record(self, namespace: str, state_key: str):
                assert namespace == "race_learning"
                assert state_key == "2027"
                return {
                    "season": 2027,
                    "history": [
                        {"race": "Bahrain Grand Prix", "year": 2027},
                        {"race": "Saudi Arabian Grand Prix", "year": 2026},
                    ],
                }

        with patch("src.utils.auto_updater.should_read_db_first", lambda: True):
            with patch("src.utils.auto_updater.should_write_to_db", lambda: True):
                with patch("src.utils.auto_updater._get_runtime_state_store", lambda: _Store()):
                    learned = get_learned_races(year=2027)

        assert learned == ["Bahrain Grand Prix"]

    def test_mark_race_as_learned_writes_runtime_state_store_when_db_writes_enabled(self):
        from src.utils.auto_updater import mark_race_as_learned

        captured: dict[str, dict] = {}

        class _Store:
            def get_record(self, namespace: str, state_key: str):
                assert namespace == "race_learning"
                assert state_key == "2027"
                return {"season": 2027, "history": [], "races_completed": 0}

            def upsert_record(self, namespace: str, state_key: str, payload: dict):
                captured["namespace"] = {"value": namespace}
                captured["state_key"] = {"value": state_key}
                captured["payload"] = payload

        with patch("src.utils.auto_updater.should_read_db_first", lambda: True):
            with patch("src.utils.auto_updater.should_write_to_db", lambda: True):
                with patch("src.utils.auto_updater.should_write_to_file", lambda: False):
                    with patch("src.utils.auto_updater._get_runtime_state_store", lambda: _Store()):
                        mark_race_as_learned("Bahrain Grand Prix", year=2027)

        assert captured["namespace"]["value"] == "race_learning"
        assert captured["state_key"]["value"] == "2027"
        assert captured["payload"]["season"] == 2027
        assert captured["payload"]["races_completed"] == 1
        assert captured["payload"]["history"][0]["race"] == "Bahrain Grand Prix"
