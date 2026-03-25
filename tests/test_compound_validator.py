"""Tests for compound characteristics validation."""

import json

from src.data.compound_validator import (
    load_and_validate_compound_data,
    validate_compound_data,
    validate_pirelli_info,
)


def _errors_from_validation(validator, data: dict) -> list[str]:
    """Return validation errors from exception-based validators."""
    try:
        validator(data)
    except ValueError as exc:
        return str(exc).split("; ")
    return []


class TestValidateCompoundData:
    """Tests for validate_compound_data function."""

    def test_validate_compound_data_valid(self):
        """Valid compound data passes validation."""
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
                "optimal_stint_length": 25,
                "pace_advantage": 0.3,
            },
            "MEDIUM": {
                "degradation_rate": 0.10,
                "optimal_stint_length": 35,
                "pace_advantage": 0.0,
            },
            "HARD": {
                "degradation_rate": 0.05,
                "optimal_stint_length": 45,
                "pace_advantage": -0.3,
            },
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert errors == []

    def test_validate_compound_data_invalid_structure(self):
        """Invalid structure fails validation."""
        data = "not a dict"
        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_validate_compound_data_missing_fields(self):
        """Missing required fields detected."""
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "missing fields" in errors[0]
        assert "optimal_stint_length" in errors[0]
        assert "pace_advantage" in errors[0]

    def test_validate_compound_data_negative_degradation(self):
        """Negative degradation rate detected."""
        data = {
            "SOFT": {
                "degradation_rate": -0.1,
                "optimal_stint_length": 25,
                "pace_advantage": 0.3,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "degradation_rate cannot be negative" in errors[0]

    def test_validate_compound_data_unusually_high_degradation(self):
        """Unusually high degradation rate detected."""
        data = {
            "SOFT": {
                "degradation_rate": 1.5,
                "optimal_stint_length": 25,
                "pace_advantage": 0.3,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "unusually high" in errors[0]

    def test_validate_compound_data_invalid_stint_length(self):
        """Invalid stint length detected."""
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
                "optimal_stint_length": -5,
                "pace_advantage": 0.3,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "optimal_stint_length must be positive" in errors[0]

    def test_validate_compound_data_unusually_high_stint(self):
        """Unusually high stint length detected."""
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
                "optimal_stint_length": 150,
                "pace_advantage": 0.3,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "unusually high (>100 laps)" in errors[0]

    def test_validate_compound_data_non_numeric_pace(self):
        """Non-numeric pace advantage detected."""
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
                "optimal_stint_length": 25,
                "pace_advantage": "fast",
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "pace_advantage must be numeric" in errors[0]

    def test_validate_compound_data_multiple_errors(self):
        """Multiple validation errors detected."""
        data = {
            "SOFT": {
                "degradation_rate": -0.1,
                "optimal_stint_length": "twenty",
                "pace_advantage": 0.3,
            },
            "MEDIUM": {
                "degradation_rate": 0.10,
            },
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 3
        assert any("SOFT" in e and "degradation_rate" in e for e in errors)
        assert any("SOFT" in e and "optimal_stint_length" in e for e in errors)
        assert any("MEDIUM" in e and "missing fields" in e for e in errors)

    def test_validate_compound_data_non_dict_compound(self):
        """Non-dict compound data triggers continue."""
        data = {
            "SOFT": "not a dict",
            "MEDIUM": {
                "degradation_rate": 0.10,
                "optimal_stint_length": 35,
                "pace_advantage": 0.0,
            },
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "must be a dictionary" in errors[0]

    def test_validate_compound_data_non_int_stint_length(self):
        """Non-integer stint length detected."""
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
                "optimal_stint_length": 25.5,
                "pace_advantage": 0.3,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "optimal_stint_length must be integer" in errors[0]

    def test_validate_compound_data_non_numeric_degradation(self):
        """Non-numeric degradation rate detected."""
        data = {
            "SOFT": {
                "degradation_rate": "high",
                "optimal_stint_length": 25,
                "pace_advantage": 0.3,
            }
        }

        errors = _errors_from_validation(validate_compound_data, data)
        assert len(errors) == 1
        assert "SOFT" in errors[0]
        assert "degradation_rate must be numeric" in errors[0]


class TestValidatePirelliInfo:
    """Tests for validate_pirelli_info function."""

    def test_validate_pirelli_info_valid(self):
        """Valid Pirelli data passes validation."""
        data = {
            "Bahrain Grand Prix": {
                "tyre_stress": {
                    "traction": 3.5,
                    "braking": 4.0,
                    "lateral": 2.5,
                    "asphalt_abrasion": 4.5,
                }
            },
            "Saudi Arabian Grand Prix": {
                "tyre_stress": {
                    "traction": 3.0,
                    "braking": 3.5,
                    "lateral": 4.5,
                    "asphalt_abrasion": 2.0,
                }
            },
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert errors == []

    def test_validate_pirelli_info_invalid_structure(self):
        """Invalid structure fails validation."""
        data = "not a dict"
        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_validate_pirelli_info_missing_tyre_stress(self):
        """Missing tyre_stress field detected."""
        data = {
            "Bahrain Grand Prix": {
                "other_field": "value",
            }
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "Bahrain Grand Prix" in errors[0]
        assert "missing tyre_stress" in errors[0]

    def test_validate_pirelli_info_invalid_stress_type(self):
        """Invalid stress type detected."""
        data = {
            "Bahrain Grand Prix": {
                "tyre_stress": {
                    "traction": "high",
                    "braking": 4.0,
                }
            }
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "Bahrain Grand Prix" in errors[0]
        assert "tyre_stress.traction" in errors[0]
        assert "must be numeric" in errors[0]

    def test_validate_pirelli_info_negative_stress(self):
        """Negative stress value detected."""
        data = {
            "Bahrain Grand Prix": {
                "tyre_stress": {
                    "traction": -1.0,
                    "braking": 4.0,
                }
            }
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "Bahrain Grand Prix" in errors[0]
        assert "tyre_stress.traction" in errors[0]
        assert "cannot be negative" in errors[0]

    def test_validate_pirelli_info_unusually_high_stress(self):
        """Unusually high stress value detected."""
        data = {
            "Bahrain Grand Prix": {
                "tyre_stress": {
                    "traction": 10.0,
                    "braking": 4.0,
                }
            }
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "Bahrain Grand Prix" in errors[0]
        assert "tyre_stress.traction" in errors[0]
        assert "unusually high" in errors[0]

    def test_validate_pirelli_info_multiple_races_with_errors(self):
        """Multiple races with errors detected."""
        data = {
            "Bahrain Grand Prix": {
                "tyre_stress": {
                    "traction": -1.0,
                }
            },
            "Saudi Arabian Grand Prix": {
                "other_field": "value",
            },
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 2
        assert any("Bahrain" in e and "traction" in e for e in errors)
        assert any("Saudi" in e and "missing tyre_stress" in e for e in errors)

    def test_validate_pirelli_info_non_dict_race(self):
        """Non-dict race data triggers continue."""
        data = {
            "Bahrain Grand Prix": "not a dict",
            "Saudi Arabian Grand Prix": {
                "tyre_stress": {
                    "traction": 3.0,
                }
            },
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "Bahrain" in errors[0]
        assert "must be a dictionary" in errors[0]

    def test_validate_pirelli_info_non_dict_stress(self):
        """Non-dict tyre_stress triggers continue."""
        data = {
            "Bahrain Grand Prix": {"tyre_stress": "not a dict"},
        }

        errors = _errors_from_validation(validate_pirelli_info, data)
        assert len(errors) == 1
        assert "Bahrain" in errors[0]
        assert "tyre_stress must be a dictionary" in errors[0]


class TestLoadAndValidateCompoundData:
    """Tests for load_and_validate_compound_data function."""

    def test_load_and_validate_compound_data_valid_compound(self, tmp_path):
        """Valid compound data loaded successfully."""
        data_file = tmp_path / "compounds.json"
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
                "optimal_stint_length": 25,
                "pace_advantage": 0.3,
            }
        }
        data_file.write_text(json.dumps(data))

        result = load_and_validate_compound_data(data_file)
        assert result == data

    def test_load_and_validate_compound_data_valid_pirelli(self, tmp_path):
        """Valid Pirelli data loaded successfully."""
        data_file = tmp_path / "pirelli.json"
        data = {
            "Bahrain Grand Prix": {
                "tyre_stress": {
                    "traction": 3.5,
                    "braking": 4.0,
                    "lateral": 2.5,
                    "asphalt_abrasion": 4.5,
                }
            }
        }
        data_file.write_text(json.dumps(data))

        result = load_and_validate_compound_data(data_file)
        assert result == data

    def test_load_and_validate_compound_data_invalid_json(self, tmp_path):
        """Invalid JSON returns None."""
        data_file = tmp_path / "invalid.json"
        data_file.write_text("{ invalid json }")

        result = load_and_validate_compound_data(data_file)
        assert result is None

    def test_load_and_validate_compound_data_file_not_found(self, tmp_path):
        """Missing file returns None."""
        data_file = tmp_path / "nonexistent.json"

        result = load_and_validate_compound_data(data_file)
        assert result is None

    def test_load_and_validate_compound_data_validation_failure(self, tmp_path):
        """Data that fails validation returns None."""
        data_file = tmp_path / "invalid_data.json"
        data = {
            "SOFT": {
                "degradation_rate": -0.1,
                "optimal_stint_length": 25,
                "pace_advantage": 0.3,
            }
        }
        data_file.write_text(json.dumps(data))

        result = load_and_validate_compound_data(data_file)
        assert result is None

    def test_load_and_validate_compound_data_missing_required_fields(self, tmp_path):
        """Compound data with missing fields returns None."""
        data_file = tmp_path / "incomplete.json"
        data = {
            "SOFT": {
                "degradation_rate": 0.15,
            }
        }
        data_file.write_text(json.dumps(data))

        result = load_and_validate_compound_data(data_file)
        assert result is None
