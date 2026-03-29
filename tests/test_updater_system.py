"""
Tests for src/systems/updater.py - Post-race update system

Critical path testing for the update_from_race functionality.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "processed"

    # Create directory structure
    char_dir = data_dir / "car_characteristics"
    char_dir.mkdir(parents=True, exist_ok=True)

    # Create initial characteristics file
    char_file = char_dir / "2026_car_characteristics.json"
    initial_data = {
        "year": 2026,
        "version": 1,
        "races_completed": 0,
        "last_updated": "2026-01-01T00:00:00",
        "teams": {
            "Red Bull": {
                "overall_performance": 0.95,
                "directionality": {
                    "max_speed": 0.0,
                    "slow_corner_speed": 0.0,
                    "medium_corner_speed": 0.0,
                    "high_corner_speed": 0.0,
                },
                "current_season_performance": [],
                "uncertainty": 0.30,
                "drivers": {"VER": 0.90, "PER": 0.80},
            },
            "McLaren": {
                "overall_performance": 0.85,
                "directionality": {
                    "max_speed": 0.0,
                    "slow_corner_speed": 0.0,
                    "medium_corner_speed": 0.0,
                    "high_corner_speed": 0.0,
                },
                "current_season_performance": [],
                "uncertainty": 0.30,
                "drivers": {"NOR": 0.85, "PIA": 0.78},
            },
        },
    }

    with open(char_file, "w") as f:
        json.dump(initial_data, f, indent=2)

    yield str(data_dir)  # Return the processed directory

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_race_results():
    """Mock FastF1 race results."""
    mock_results = MagicMock()
    mock_results.to_dict.return_value = [
        {
            "DriverNumber": "1",
            "Abbreviation": "VER",
            "TeamName": "Red Bull Racing",
            "Position": 1,
        },
        {
            "DriverNumber": "4",
            "Abbreviation": "NOR",
            "TeamName": "McLaren",
            "Position": 2,
        },
        {
            "DriverNumber": "11",
            "Abbreviation": "PER",
            "TeamName": "Red Bull Racing",
            "Position": 3,
        },
        {
            "DriverNumber": "81",
            "Abbreviation": "PIA",
            "TeamName": "McLaren",
            "Position": 4,
        },
    ]
    return mock_results


@pytest.fixture
def mock_session():
    """Mock FastF1 session."""
    mock_sess = MagicMock()
    mock_sess.event = {"EventName": "Bahrain Grand Prix"}
    mock_sess.name = "Race"
    return mock_sess


class TestUpdaterCore:
    """Test core updater functionality."""

    def test_update_increments_version(self, temp_data_dir, mock_race_results, mock_session):
        """Test that update_from_race increments version number."""
        from src.systems.updater import update_from_race

        with patch("src.systems.updater.load_race_session") as mock_load:
            mock_load.return_value = (mock_race_results, mock_session)
            with patch("src.systems.updater.load_qualifying_session") as mock_quali:
                mock_quali.return_value = (pd.DataFrame(), mock_session)

                with patch(
                    "src.systems.updater.extract_team_performance_from_telemetry"
                ) as mock_extract:
                    mock_extract.return_value = {"Red Bull": 0.95, "McLaren": 0.85}

                    update_from_race(2026, "Bahrain Grand Prix", temp_data_dir)

        # Verify version incremented
        char_file = Path(temp_data_dir) / "car_characteristics" / "2026_car_characteristics.json"
        with open(char_file) as f:
            data = json.load(f)

        assert data["version"] == 2, "Version should increment from 1 to 2"
        assert data["races_completed"] == 1, "Races completed should be 1"

    def test_update_appends_performance(self, temp_data_dir, mock_race_results, mock_session):
        """Test that update appends to current_season_performance."""
        from src.systems.updater import update_from_race

        with patch("src.systems.updater.load_race_session") as mock_load:
            mock_load.return_value = (mock_race_results, mock_session)
            with patch("src.systems.updater.load_qualifying_session") as mock_quali:
                mock_quali.return_value = (pd.DataFrame(), mock_session)

                with patch(
                    "src.systems.updater.extract_team_performance_from_telemetry"
                ) as mock_extract:
                    mock_extract.return_value = {"Red Bull": 0.95, "McLaren": 0.82}

                    update_from_race(2026, "Bahrain Grand Prix", temp_data_dir)

        # Verify performance appended
        char_file = Path(temp_data_dir) / "car_characteristics" / "2026_car_characteristics.json"
        with open(char_file) as f:
            data = json.load(f)

        assert len(data["teams"]["McLaren"]["current_season_performance"]) == 1
        assert data["teams"]["McLaren"]["current_season_performance"][0] == 0.82

    def test_update_reduces_uncertainty(self, temp_data_dir, mock_race_results, mock_session):
        """Test that uncertainty decreases after race update."""
        from src.systems.updater import update_from_race

        # Get initial uncertainty
        char_file = Path(temp_data_dir) / "car_characteristics" / "2026_car_characteristics.json"
        with open(char_file) as f:
            initial_data = json.load(f)
        initial_uncertainty = initial_data["teams"]["McLaren"]["uncertainty"]

        with patch("src.systems.updater.load_race_session") as mock_load:
            mock_load.return_value = (mock_race_results, mock_session)
            with patch("src.systems.updater.load_qualifying_session") as mock_quali:
                mock_quali.return_value = (pd.DataFrame(), mock_session)

                with patch(
                    "src.systems.updater.extract_team_performance_from_telemetry"
                ) as mock_extract:
                    mock_extract.return_value = {"Red Bull": 0.95, "McLaren": 0.85}

                    update_from_race(2026, "Bahrain Grand Prix", temp_data_dir)

        # Verify uncertainty reduced
        with open(char_file) as f:
            updated_data = json.load(f)
        updated_uncertainty = updated_data["teams"]["McLaren"]["uncertainty"]

        assert updated_uncertainty < initial_uncertainty, "Uncertainty should decrease after update"

    def test_update_moves_baseline_toward_observed_form(
        self, temp_data_dir, mock_race_results, mock_session
    ):
        """Repeated race updates should nudge baseline performance toward observed form."""
        from src.systems.updater import update_from_race

        char_file = Path(temp_data_dir) / "car_characteristics" / "2026_car_characteristics.json"
        with open(char_file) as f:
            initial_data = json.load(f)
        initial_baseline = initial_data["teams"]["McLaren"]["overall_performance"]

        with patch("src.systems.updater.load_race_session") as mock_load:
            mock_load.return_value = (mock_race_results, mock_session)
            with patch("src.systems.updater.load_qualifying_session") as mock_quali:
                mock_quali.return_value = (pd.DataFrame(), mock_session)

                with patch(
                    "src.systems.updater.extract_team_performance_from_telemetry"
                ) as mock_extract:
                    mock_extract.side_effect = [
                        {"Red Bull": 0.95, "McLaren": 0.60},
                        {"Red Bull": 0.95, "McLaren": 0.50},
                    ]

                    update_from_race(2026, "Bahrain Grand Prix", temp_data_dir)
                    update_from_race(2026, "Saudi Arabian Grand Prix", temp_data_dir)

        with open(char_file) as f:
            updated_data = json.load(f)
        updated_baseline = updated_data["teams"]["McLaren"]["overall_performance"]

        assert updated_baseline < initial_baseline
        assert updated_baseline > 0.50
        assert updated_data["teams"]["McLaren"]["current_season_performance"] == [0.6, 0.5]


class TestUpdaterEdgeCases:
    """Test edge cases and error handling."""

    def test_update_with_missing_team(self, temp_data_dir, mock_session):
        """Test update handles new teams gracefully."""
        from src.systems.updater import update_from_race

        # Mock results with a new team
        mock_results_new_team = MagicMock()
        mock_results_new_team.to_dict.return_value = [
            {
                "DriverNumber": "1",
                "Abbreviation": "VER",
                "TeamName": "Red Bull Racing",
                "Position": 1,
            },
            {
                "DriverNumber": "99",
                "Abbreviation": "NEW",
                "TeamName": "Cadillac",
                "Position": 11,
            },
        ]

        with patch("src.systems.updater.load_race_session") as mock_load:
            mock_load.return_value = (mock_results_new_team, mock_session)
            with patch("src.systems.updater.load_qualifying_session") as mock_quali:
                mock_quali.return_value = (pd.DataFrame(), mock_session)

                with patch(
                    "src.systems.updater.extract_team_performance_from_telemetry"
                ) as mock_extract:
                    mock_extract.return_value = {"Red Bull": 0.95, "Cadillac": 0.50}

                    # Should not crash
                    update_from_race(2026, "Bahrain Grand Prix", temp_data_dir)

    def test_backup_created(self, temp_data_dir, mock_race_results, mock_session):
        """Test that backup file is created before update."""
        from src.systems.updater import update_from_race

        char_file = Path(temp_data_dir) / "car_characteristics" / "2026_car_characteristics.json"
        backup_file = Path(str(char_file) + ".backup")

        # Mock Supabase to force file fallback (which creates backups)
        with patch("src.persistence.db.get_supabase_client") as mock_supabase:
            # Make Supabase fail so it falls back to file mode
            mock_supabase.side_effect = RuntimeError("Supabase unavailable in test")

            with patch("src.systems.updater.load_race_session") as mock_load:
                mock_load.return_value = (mock_race_results, mock_session)
                with patch("src.systems.updater.load_qualifying_session") as mock_quali:
                    mock_quali.return_value = (pd.DataFrame(), mock_session)

                    with patch(
                        "src.systems.updater.extract_team_performance_from_telemetry"
                    ) as mock_extract:
                        mock_extract.return_value = {"Red Bull": 0.95, "McLaren": 0.85}

                        update_from_race(2026, "Bahrain Grand Prix", temp_data_dir)

        # Check backup exists
        assert backup_file.exists(), f"Backup file should be created at {backup_file}"

    def test_extract_team_performance_canonicalizes_session_team_names(self):
        """Telemetry extraction should map sponsor names to characteristics team names."""
        from src.systems.updater import extract_team_performance_from_telemetry

        laps = pd.DataFrame(
            [
                {
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=91.0),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 2,
                },
                {
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=91.2),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 3,
                },
                {
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=91.1),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 4,
                },
                {
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=91.3),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 5,
                },
                {
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=91.0),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 6,
                },
                {
                    "Team": "Oracle Red Bull Racing",
                    "LapTime": pd.Timedelta(seconds=91.4),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 7,
                },
                {
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=91.4),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 2,
                },
                {
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=91.5),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 3,
                },
                {
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=91.6),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 4,
                },
                {
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=91.5),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 5,
                },
                {
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=91.7),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 6,
                },
                {
                    "Team": "Scuderia Ferrari",
                    "LapTime": pd.Timedelta(seconds=91.8),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": 7,
                },
            ]
        )

        session = MagicMock()
        session.laps = laps

        perf = extract_team_performance_from_telemetry(
            session=session,
            team_names=["Red Bull Racing", "Ferrari"],
        )

        assert "Red Bull Racing" in perf
        assert "Ferrari" in perf

    def test_update_bayesian_driver_ratings_uses_observation_mapping(self):
        """Bayesian updater should receive a full observations mapping, not per-driver scalars."""
        from src.systems.updater import update_bayesian_driver_ratings

        race_results = pd.DataFrame(
            {
                "Abbreviation": ["VER", "NOR", "PIA"],
                "Position": [1, 2, pd.NA],
                "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix", "Bahrain Grand Prix"],
            }
        )

        with patch("src.models.priors_factory.PriorsFactory.create_priors", return_value={}) as _:
            with patch("src.systems.updater.BayesianDriverRanking") as mock_bayesian_cls:
                mock_bayesian = mock_bayesian_cls.return_value

                update_bayesian_driver_ratings(race_results)

        mock_bayesian.update.assert_called_once()
        kwargs = mock_bayesian.update.call_args.kwargs
        assert kwargs["observations"] == {"VER": 1, "NOR": 2}
        assert kwargs["session_name"] == "Bahrain Grand Prix"
        assert kwargs["confidence"] == 1.0
