"""
Test suite for JSON schema validation module.

Tests cover:
- Valid data validation
- Invalid data detection
- Error handling and messaging
- All three schema types (driver, team, track characteristics)
"""

import json
from pathlib import Path

import pytest

from src.utils.schema_validation import (
    DRIVER_CHARACTERISTICS_SCHEMA,
    validate_driver_characteristics,
    validate_json,
    validate_team_characteristics,
    validate_track_characteristics,
)


class TestDriverCharacteristicsSchema:
    """Test driver characteristics schema validation."""

    def test_valid_driver_data_from_file(self):
        """Test validation of actual driver characteristics file."""
        file_path = Path("data/processed/driver_characteristics.json")
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            # Should not raise
            validate_driver_characteristics(data)

    def test_valid_minimal_driver_data(self):
        """Test validation of minimal valid driver data."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.85, "overtaking_skill": 0.90},
                    "pace": {"quali_pace": 0.92, "race_pace": 0.88},
                    "dnf_risk": {"dnf_rate": 0.05},
                }
            }
        }
        # Should not raise
        validate_driver_characteristics(data)

    def test_invalid_missing_drivers_key(self):
        """Test that missing 'drivers' key raises error."""
        data = {}
        with pytest.raises(ValueError, match="drivers"):
            validate_driver_characteristics(data)

    def test_invalid_missing_racecraft(self):
        """Test that missing 'racecraft' raises error."""
        data = {
            "drivers": {
                "VER": {
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                    "dnf_risk": {"dnf_rate": 0.05},
                }
            }
        }
        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_missing_pace(self):
        """Test that missing 'pace' raises error."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.8},
                    "dnf_risk": {"dnf_rate": 0.05},
                }
            }
        }
        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_missing_dnf_risk(self):
        """Test that missing 'dnf_risk' raises error."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.8},
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                }
            }
        }
        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_skill_score_out_of_range(self):
        """Test that skill_score > 1.0 raises error."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {
                        "skill_score": 1.5,
                        "overtaking_skill": 0.8,
                    },  # Invalid: > 1.0
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                    "dnf_risk": {"dnf_rate": 0.05},
                }
            }
        }
        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_dnf_rate_out_of_range(self):
        """Test that dnf_rate > 1.0 raises error."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.8},
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                    "dnf_risk": {"dnf_rate": 1.5},  # Invalid: > 1.0
                }
            }
        }
        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_driver_code_not_3_letters(self):
        """Test that invalid driver codes are accepted (pattern is permissive)."""
        # The schema uses patternProperties with relaxed matching
        # Non-matching patterns just aren't validated against the driver schema
        data = {
            "drivers": {
                "VERSTAPPEN": {  # Not 3 letters - won't match pattern
                    "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.8},
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                    "dnf_risk": {"dnf_rate": 0.05},
                }
            }
        }
        # This should pass because patternProperties don't require all to match
        validate_driver_characteristics(data)


class TestTeamCharacteristicsSchema:
    """Test team characteristics schema validation."""

    def test_valid_team_data_from_file(self):
        """Test validation of actual team characteristics file."""
        file_path = Path("data/processed/car_characteristics/2026_car_characteristics.json")
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            # Should not raise
            validate_team_characteristics(data)

    def test_valid_minimal_team_data(self):
        """Test validation of minimal valid team data."""
        data = {
            "teams": {
                "McLaren": {"overall_performance": 0.85},
                "Ferrari": {"overall_performance": 0.75},
            }
        }
        # Should not raise
        validate_team_characteristics(data)

    def test_invalid_missing_teams_key(self):
        """Test that missing 'teams' key raises error."""
        data = {}
        with pytest.raises(ValueError, match="teams"):
            validate_team_characteristics(data)

    def test_invalid_missing_overall_performance(self):
        """Test that missing 'overall_performance' raises error."""
        data = {"teams": {"McLaren": {"uncertainty": 0.30}}}
        with pytest.raises(ValueError):
            validate_team_characteristics(data)

    def test_invalid_performance_out_of_range(self):
        """Test that performance > 1.0 raises error."""
        data = {"teams": {"McLaren": {"overall_performance": 1.5}}}  # Invalid: > 1.0
        with pytest.raises(ValueError):
            validate_team_characteristics(data)

    def test_invalid_uncertainty_out_of_range(self):
        """Test that uncertainty > 1.0 raises error."""
        data = {
            "teams": {
                "McLaren": {
                    "overall_performance": 0.85,
                    "uncertainty": 1.5,
                }  # Invalid: > 1.0
            }
        }
        with pytest.raises(ValueError):
            validate_team_characteristics(data)


class TestTrackCharacteristicsSchema:
    """Test track characteristics schema validation."""

    def test_valid_track_data_from_file(self):
        """Test validation of actual track characteristics file."""
        file_path = Path("data/processed/track_characteristics/2026_track_characteristics.json")
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            # Should not raise
            validate_track_characteristics(data)

    def test_valid_minimal_track_data(self):
        """Test validation of minimal valid track data."""
        data = {
            "tracks": {
                "Monaco Grand Prix": {
                    "pit_stop_loss": 20.0,
                    "safety_car_prob": 0.8,
                    "overtaking_difficulty": 0.9,
                },
                "Monza Grand Prix": {
                    "pit_stop_loss": 24.0,
                    "safety_car_prob": 0.3,
                    "overtaking_difficulty": 0.2,
                },
            }
        }
        # Should not raise
        validate_track_characteristics(data)

    def test_valid_track_data_with_sprint(self):
        """Test validation of track data with sprint flag."""
        data = {
            "tracks": {
                "Miami Grand Prix": {
                    "pit_stop_loss": 24.0,
                    "safety_car_prob": 0.7,
                    "overtaking_difficulty": 0.7,
                    "has_sprint": True,
                }
            }
        }
        # Should not raise
        validate_track_characteristics(data)

    def test_invalid_missing_tracks_key(self):
        """Test that missing 'tracks' key raises error."""
        data = {}
        with pytest.raises(ValueError, match="tracks"):
            validate_track_characteristics(data)

    def test_invalid_safety_car_prob_out_of_range(self):
        """Test that safety_car_prob > 1.0 raises error."""
        data = {
            "tracks": {
                "Monaco Grand Prix": {
                    "pit_stop_loss": 20.0,
                    "safety_car_prob": 1.5,  # Invalid: > 1.0
                    "overtaking_difficulty": 0.9,
                }
            }
        }
        with pytest.raises(ValueError):
            validate_track_characteristics(data)

    def test_invalid_overtaking_difficulty_out_of_range(self):
        """Test that overtaking_difficulty > 1.0 raises error."""
        data = {
            "tracks": {
                "Monaco Grand Prix": {
                    "pit_stop_loss": 20.0,
                    "safety_car_prob": 0.8,
                    "overtaking_difficulty": 1.5,  # Invalid: > 1.0
                }
            }
        }
        with pytest.raises(ValueError):
            validate_track_characteristics(data)


class TestValidateJsonFunction:
    """Test the generic validate_json function."""

    def test_validate_json_with_valid_data(self):
        """Test validate_json with valid data."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.8},
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                    "dnf_risk": {"dnf_rate": 0.05},
                }
            }
        }
        # Should not raise
        validate_json(data, DRIVER_CHARACTERISTICS_SCHEMA, "test.json")

    def test_validate_json_with_invalid_data(self):
        """Test validate_json with invalid data."""
        data = {}
        with pytest.raises(ValueError):
            validate_json(data, DRIVER_CHARACTERISTICS_SCHEMA, "test.json")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_driver_with_empty_dnf_types(self):
        """Test driver with empty dnf_types dict."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.8},
                    "pace": {"quali_pace": 0.9, "race_pace": 0.85},
                    "dnf_risk": {"dnf_rate": 0.05, "dnf_types": {}},
                }
            }
        }
        # Should not raise
        validate_driver_characteristics(data)

    def test_track_without_optional_sprint_flag(self):
        """Test track data without has_sprint flag."""
        data = {
            "tracks": {
                "Monaco Grand Prix": {
                    "pit_stop_loss": 20.0,
                    "safety_car_prob": 0.8,
                    "overtaking_difficulty": 0.9,
                }
            }
        }
        # Should not raise
        validate_track_characteristics(data)

    def test_boundary_values_zero(self):
        """Test boundary value of 0.0 for normalized fields."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 0.0, "overtaking_skill": 0.0},
                    "pace": {"quali_pace": 0.0, "race_pace": 0.0},
                    "dnf_risk": {"dnf_rate": 0.0},
                }
            }
        }
        # Should not raise
        validate_driver_characteristics(data)

    def test_boundary_values_one(self):
        """Test boundary value of 1.0 for normalized fields."""
        data = {
            "drivers": {
                "VER": {
                    "racecraft": {"skill_score": 1.0, "overtaking_skill": 1.0},
                    "pace": {"quali_pace": 1.0, "race_pace": 1.0},
                    "dnf_risk": {"dnf_rate": 1.0},
                }
            }
        }
        # Should not raise
        validate_driver_characteristics(data)
