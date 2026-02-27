"""
Comprehensive tests for utils modules

Increases test coverage for lineups, weekend, and config utilities.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestLineupsModule:
    """Test lineup extraction and management"""

    def test_get_lineups_for_2026(self):
        """Test getting lineups for 2026 races"""
        from src.utils.lineups import get_lineups

        lineups = get_lineups(2026, "Bahrain Grand Prix")

        # Should return 11 teams
        assert len(lineups) == 11
        assert "McLaren" in lineups
        assert "Cadillac F1" in lineups

        # Each team should have 2 drivers
        for _team, drivers in lineups.items():
            assert len(drivers) == 2

    def test_lineup_driver_codes_valid(self):
        """Test all driver codes are 3-letter abbreviations"""
        from src.utils.lineups import get_lineups

        lineups = get_lineups(2026, "Bahrain Grand Prix")

        for _team, drivers in lineups.items():
            for driver in drivers:
                assert len(driver) == 3, f"Invalid driver code: {driver}"
                assert driver.isupper(), f"Driver code not uppercase: {driver}"

    @patch("src.utils.lineups.ff1.get_session")
    def test_get_lineups_from_session_handles_missing_data(self, mock_session):
        """Test handling when session data is unavailable"""
        from src.utils.lineups import get_lineups_from_session

        # Mock session with no results
        mock_session_obj = MagicMock()
        mock_session_obj.results = None
        mock_session.return_value = mock_session_obj

        result = get_lineups_from_session(2025, "Bahrain Grand Prix", "Q")

        # Should return None when no data available
        assert result is None


class TestWeekendModule:
    """Test weekend type detection"""

    @staticmethod
    def _mock_schedule():
        return pd.DataFrame(
            {
                "EventName": [
                    "Bahrain Grand Prix",
                    "Chinese Grand Prix",
                    "Miami Grand Prix",
                    "Monaco Grand Prix",
                ],
                "EventFormat": [
                    "conventional",
                    "sprint",
                    "sprint_shootout",
                    "conventional",
                ],
            }
        )

    def test_sprint_weekend_detection_2026(self):
        """Sprint detection should come from FastF1 EventFormat."""
        from src.utils import weekend

        weekend.refresh_schedule_cache()
        with patch(
            "src.utils.weekend.fastf1.get_event_schedule",
            return_value=self._mock_schedule(),
        ):
            assert weekend.is_sprint_weekend(2026, "Chinese Grand Prix") is True
            assert weekend.is_sprint_weekend(2026, "Miami Grand Prix") is True

    def test_normal_weekend_detection_2026(self):
        """Conventional events should resolve to non-sprint weekends."""
        from src.utils import weekend

        weekend.refresh_schedule_cache()
        with patch(
            "src.utils.weekend.fastf1.get_event_schedule",
            return_value=self._mock_schedule(),
        ):
            assert weekend.is_sprint_weekend(2026, "Bahrain Grand Prix") is False
            assert weekend.is_sprint_weekend(2026, "Monaco Grand Prix") is False

    def test_get_all_sprint_races(self):
        """Sprint race list should be derived from schedule metadata."""
        from src.utils import weekend

        weekend.refresh_schedule_cache()
        with patch(
            "src.utils.weekend.fastf1.get_event_schedule",
            return_value=self._mock_schedule(),
        ):
            sprint_races = weekend.get_all_sprint_races(2026)

        assert sprint_races == ["Chinese Grand Prix", "Miami Grand Prix"]

    def test_get_best_qualifying_session(self):
        """Best qualifying source should follow weekend format."""
        from src.utils import weekend

        weekend.refresh_schedule_cache()
        with patch(
            "src.utils.weekend.fastf1.get_event_schedule",
            return_value=self._mock_schedule(),
        ):
            assert (
                weekend.get_best_qualifying_session(2026, "Chinese Grand Prix")
                == "Sprint Qualifying"
            )
            assert weekend.get_best_qualifying_session(2026, "Bahrain Grand Prix") == "FP3"

    def test_invalid_race_name_handling(self):
        """Test graceful handling of invalid race names"""
        from src.utils.weekend import get_weekend_type

        with pytest.raises(ValueError) as exc_info:
            get_weekend_type(2026, "Invalid Race Name")

        assert "not found" in str(exc_info.value).lower()


class TestConfigModule:
    """Test configuration loading and management"""

    def test_production_config_file_exists(self):
        """Test production config file exists and is valid JSON"""
        import json
        from pathlib import Path

        config_path = Path("config/production_config.json")
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "qualifying_methods" in config


class TestDataValidation:
    """Test data file integrity"""

    def test_2026_car_characteristics_complete(self):
        """Test all 11 teams have car characteristics"""
        import json

        with open("data/processed/car_characteristics/2026_car_characteristics.json") as f:
            data = json.load(f)

        teams = data["teams"]

        # Should have all 11 teams
        assert len(teams) == 11

        # Check required fields
        required_fields = ["overall_performance", "uncertainty", "note"]
        for team, values in teams.items():
            for field in required_fields:
                assert field in values, f"{team} missing {field}"

            # Performance should be 0-1
            assert 0 <= values["overall_performance"] <= 1
            assert 0 <= values["uncertainty"] <= 1

    def test_2026_track_characteristics_complete(self):
        """Test all 24 tracks have characteristics"""
        import json

        with open("data/processed/track_characteristics/2026_track_characteristics.json") as f:
            data = json.load(f)

        tracks = data["tracks"]

        # Should have 24 races
        assert len(tracks) >= 23  # At least 23 (allow for calendar changes)

        # Check required fields
        for _track, values in tracks.items():
            assert "pit_stop_loss" in values
            assert "safety_car_prob" in values
            assert "overtaking_difficulty" in values
            assert "type" in values

            # Validate ranges
            assert 15 <= values["pit_stop_loss"] <= 30  # Realistic pit stop loss
            assert 0 <= values["safety_car_prob"] <= 1
            assert 0 <= values["overtaking_difficulty"] <= 1
            assert values["type"] in ["permanent", "street"]

    def test_current_lineups_valid(self):
        """Test current lineups file is valid"""
        import json

        with open("data/current_lineups.json") as f:
            data = json.load(f)

        lineups = data["current_lineups"]

        # 11 teams for 2026
        assert len(lineups) == 11

        # Each team has 2 drivers
        for _team, drivers in lineups.items():
            assert len(drivers) == 2
            for driver in drivers:
                assert len(driver) == 3  # 3-letter code
                assert driver.isupper()
