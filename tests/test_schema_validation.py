"""Tests for runtime JSON schema validation."""

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


def _driver_payload() -> dict:
    """Return a valid minimal driver payload."""
    return {
        "drivers": {
            "VER": {
                "name": "Max Verstappen",
                "experience": {
                    "debut_year": 2015,
                    "total_races": 210,
                    "years_of_experience": 11,
                    "tier": "elite",
                },
                "racecraft": {"skill_score": 0.85, "overtaking_skill": 0.90},
                "pace": {"quali_pace": 0.92, "race_pace": 0.88},
                "dnf_risk": {"dnf_rate": 0.05},
            }
        }
    }


def _team_payload() -> dict:
    """Return a valid nested team payload."""
    return {
        "year": 2026,
        "teams": {
            "McLaren": {
                "overall_performance": 0.85,
                "uncertainty": 0.12,
                "current_season_performance": [0.80, 0.83],
                "directionality": {
                    "max_speed": 0.10,
                    "slow_corner_speed": 0.02,
                    "medium_corner_speed": -0.01,
                    "high_corner_speed": 0.05,
                },
                "testing_characteristics": {
                    "run_profile": "balanced",
                    "overall_pace": 0.78,
                    "overall_pace_seconds": 99.214,
                    "top_speed": 0.72,
                    "top_speed_kph": 321.5,
                    "slow_corner_seconds": 31.2,
                    "medium_corner_seconds": 43.9,
                    "fast_corner_seconds": 24.1,
                    "braking_pct": 98.4,
                },
                "testing_characteristics_profiles": {
                    "balanced": {
                        "run_profile": "balanced",
                        "overall_pace": 0.78,
                        "overall_pace_seconds": 99.214,
                        "top_speed": 0.72,
                        "top_speed_kph": 321.5,
                        "slow_corner_seconds": 31.2,
                        "medium_corner_seconds": 43.9,
                        "fast_corner_seconds": 24.1,
                        "braking_pct": 98.4,
                    }
                },
                "compound_characteristics": {
                    "SOFT": {
                        "pace_performance": 0.82,
                        "tire_deg_performance": 0.61,
                        "consistency_performance": 0.75,
                        "tire_deg_slope": 0.12,
                        "laps_sampled": 18,
                    }
                },
            }
        },
    }


def _track_payload() -> dict:
    """Return a valid track payload."""
    return {
        "year": 2026,
        "tracks": {
            "Bahrain Grand Prix": {
                "pit_stop_loss": 22.0,
                "safety_car_prob": 0.35,
                "overtaking_difficulty": 0.60,
                "lap1_risk_modifier": 0.22,
                "type": "permanent",
                "straights_pct": 30.0,
                "slow_corners_pct": 25.0,
                "medium_corners_pct": 25.0,
                "high_corners_pct": 20.0,
            }
        },
    }


class TestDriverCharacteristicsSchema:
    """Test driver characteristics schema validation."""

    def test_valid_driver_data_from_season_scoped_file(self):
        """Validate the shipped season-scoped driver file."""
        file_path = Path("data/processed/driver_characteristics/2026_driver_characteristics.json")
        if file_path.exists():
            with open(file_path) as file_obj:
                data = json.load(file_obj)

            validate_driver_characteristics(data, expected_year=2026)

    def test_valid_minimal_driver_data(self):
        """Accept a small but structurally valid driver payload."""
        validate_driver_characteristics(_driver_payload())

    def test_invalid_missing_drivers_key(self):
        """Reject payloads without the required drivers map."""
        with pytest.raises(ValueError, match="drivers"):
            validate_driver_characteristics({})

    def test_invalid_skill_score_out_of_range(self):
        """Reject racecraft scores outside the normalized range."""
        data = _driver_payload()
        data["drivers"]["VER"]["racecraft"]["skill_score"] = 1.5

        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_driver_code_not_3_letters(self):
        """Reject driver keys that do not match canonical three-letter codes."""
        data = _driver_payload()
        data["drivers"] = {"VERSTAPPEN": data["drivers"]["VER"]}

        with pytest.raises(ValueError):
            validate_driver_characteristics(data)

    def test_invalid_driver_bayesian_year_mismatch(self):
        """Reject season markers that conflict with the expected year."""
        data = _driver_payload()
        data["drivers"]["VER"]["bayesian"] = {"season_year": 2025}

        with pytest.raises(ValueError, match="expected season 2026"):
            validate_driver_characteristics(data, expected_year=2026)

    def test_valid_driver_bayesian_payload_without_normalized_skill_score(self):
        """Allow newer Bayesian payloads that omit the cached normalized score."""
        data = _driver_payload()
        data["drivers"]["VER"]["bayesian"] = {
            "rating_mu": 18.5,
            "rating_sigma": 1.6,
            "blended_skill_score": 0.84,
            "blend_weight": 0.25,
            "last_session": "Japanese Grand Prix",
            "season_year": 2026,
        }

        validate_driver_characteristics(data, expected_year=2026)


class TestTeamCharacteristicsSchema:
    """Test team characteristics schema validation."""

    def test_valid_team_data_from_file(self):
        """Validate the shipped team characteristics file."""
        file_path = Path("data/processed/car_characteristics/2026_car_characteristics.json")
        if file_path.exists():
            with open(file_path) as file_obj:
                data = json.load(file_obj)

            validate_team_characteristics(data, expected_year=2026)

    def test_valid_nested_team_data(self):
        """Accept the richer nested team structure the predictor consumes."""
        validate_team_characteristics(_team_payload())

    def test_valid_team_data_with_checkpoint_snapshot_metadata(self):
        """Allow checkpoint snapshot metadata on overlaid car-characteristics payloads."""
        data = _team_payload()
        data["checkpoint_snapshot"] = {
            "event_name": "Australian Grand Prix",
            "session_name": "FP1",
            "source": "testing_practice_extraction",
            "captured_at": "2026-03-14T23:32:08+00:00",
            "session_started_at": "2026-03-06T01:30:00+00:00",
        }

        validate_team_characteristics(data)

    def test_valid_team_data_with_blend_provenance_fields(self):
        """Allow replayed testing payloads to keep blend provenance details."""
        data = _team_payload()
        data["teams"]["McLaren"]["testing_characteristics"].update(
            {
                "sessions_blended": 2,
                "effective_blend_weight": 0.18,
                "circuits_observed": ["Testing 1", "Testing 2"],
            }
        )

        validate_team_characteristics(data)

    def test_invalid_missing_teams_key(self):
        """Reject payloads without the teams map."""
        with pytest.raises(ValueError, match="teams"):
            validate_team_characteristics({"year": 2026})

    def test_invalid_missing_overall_performance(self):
        """Reject team entries without baseline strength."""
        data = {"year": 2026, "teams": {"McLaren": {"uncertainty": 0.30}}}

        with pytest.raises(ValueError):
            validate_team_characteristics(data)

    def test_invalid_current_season_performance_out_of_range(self):
        """Reject current-season observations outside the normalized range."""
        data = _team_payload()
        data["teams"]["McLaren"]["current_season_performance"] = [0.5, 1.2]

        with pytest.raises(ValueError):
            validate_team_characteristics(data)

    def test_invalid_directionality_missing_axis(self):
        """Reject directionality payloads that omit a required axis."""
        data = _team_payload()
        del data["teams"]["McLaren"]["directionality"]["high_corner_speed"]

        with pytest.raises(ValueError):
            validate_team_characteristics(data)

    def test_invalid_expected_year_mismatch(self):
        """Reject team payloads whose embedded season conflicts with the load target."""
        data = _team_payload()
        data["year"] = 2025

        with pytest.raises(ValueError, match="expected season 2026"):
            validate_team_characteristics(data, expected_year=2026)


class TestTrackCharacteristicsSchema:
    """Test track characteristics schema validation."""

    def test_valid_track_data_from_file(self):
        """Validate the shipped track characteristics file."""
        file_path = Path("data/processed/track_characteristics/2026_track_characteristics.json")
        if file_path.exists():
            with open(file_path) as file_obj:
                data = json.load(file_obj)

            validate_track_characteristics(data, expected_year=2026)

    def test_valid_track_data_with_profile_percentages(self):
        """Accept valid track payloads with optional composition percentages."""
        validate_track_characteristics(_track_payload())

    def test_invalid_missing_tracks_key(self):
        """Reject payloads without the tracks map."""
        with pytest.raises(ValueError, match="tracks"):
            validate_track_characteristics({"year": 2026})

    def test_invalid_missing_required_track_field(self):
        """Reject track entries that omit a required core field."""
        data = _track_payload()
        del data["tracks"]["Bahrain Grand Prix"]["overtaking_difficulty"]

        with pytest.raises(ValueError):
            validate_track_characteristics(data)

    def test_invalid_partial_track_profile_percentages(self):
        """Reject partial track profile payloads because weighting would be ambiguous."""
        data = _track_payload()
        del data["tracks"]["Bahrain Grand Prix"]["high_corners_pct"]

        with pytest.raises(ValueError, match="track profile fields"):
            validate_track_characteristics(data)

    def test_invalid_overtaking_difficulty_out_of_range(self):
        """Reject overtaking difficulty values outside the normalized range."""
        data = _track_payload()
        data["tracks"]["Bahrain Grand Prix"]["overtaking_difficulty"] = 1.5

        with pytest.raises(ValueError):
            validate_track_characteristics(data)

    def test_invalid_expected_year_mismatch(self):
        """Reject track payloads whose embedded season conflicts with the load target."""
        data = _track_payload()
        data["year"] = 2025

        with pytest.raises(ValueError, match="expected season 2026"):
            validate_track_characteristics(data, expected_year=2026)


class TestValidateJsonFunction:
    """Test the generic validate_json wrapper."""

    def test_validate_json_with_valid_data(self):
        """Accept valid payloads when called through the generic helper."""
        validate_json(_driver_payload(), DRIVER_CHARACTERISTICS_SCHEMA, "test.json")

    def test_validate_json_with_invalid_data(self):
        """Raise ValueError for invalid payloads in the generic helper."""
        with pytest.raises(ValueError):
            validate_json({}, DRIVER_CHARACTERISTICS_SCHEMA, "test.json")


class TestEdgeCases:
    """Test schema edge cases and boundaries."""

    def test_driver_with_empty_dnf_types(self):
        """Allow an empty DNF type breakdown."""
        data = _driver_payload()
        data["drivers"]["VER"]["dnf_risk"]["dnf_types"] = {}

        validate_driver_characteristics(data)

    def test_track_without_optional_sprint_flag(self):
        """Allow track payloads that omit optional weekend metadata."""
        data = _track_payload()
        data["tracks"]["Bahrain Grand Prix"].pop("type")

        validate_track_characteristics(data)

    def test_boundary_values_zero(self):
        """Allow zero-valued normalized driver metrics."""
        data = _driver_payload()
        data["drivers"]["VER"]["racecraft"] = {"skill_score": 0.0, "overtaking_skill": 0.0}
        data["drivers"]["VER"]["pace"] = {"quali_pace": 0.0, "race_pace": 0.0}
        data["drivers"]["VER"]["dnf_risk"] = {"dnf_rate": 0.0}

        validate_driver_characteristics(data)

    def test_boundary_values_one(self):
        """Allow one-valued normalized driver metrics."""
        data = _driver_payload()
        data["drivers"]["VER"]["racecraft"] = {"skill_score": 1.0, "overtaking_skill": 1.0}
        data["drivers"]["VER"]["pace"] = {"quali_pace": 1.0, "race_pace": 1.0}
        data["drivers"]["VER"]["dnf_risk"] = {"dnf_rate": 1.0}

        validate_driver_characteristics(data)
