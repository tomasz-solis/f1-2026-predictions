"""
Configuration loader for F1 prediction system.

Loads settings from YAML config files with environment variable overrides.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
ValidationSpec = tuple[str, type[int] | type[float], float, float]


class Config:
    """Central configuration manager."""

    _instance: "Config | None" = None
    _config: dict[str, Any] | None = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self._load()

    def _load(self) -> None:
        """Load config from YAML file."""
        # Find config file
        config_file = os.getenv("F1_CONFIG", "config/default.yaml")
        config_path = Path(config_file)

        if not config_path.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Validate required sections exist
        self._validate_config()
        logger.info(f"Configuration loaded successfully from {config_path}")

    def _validate_config(self) -> None:
        """Validate that required config sections exist, with correct structure and value ranges."""
        config = self._config
        if config is None:
            raise ValueError("Config validation failed: configuration is not loaded")

        # Fast-fail validation using Pydantic schema (if available)
        try:
            from src.utils.config_schema import validate_config

            # Validate against Pydantic schema for structured validation
            try:
                validate_config(config)
                logger.info("Config passed Pydantic schema validation")
            except Exception as pydantic_error:
                logger.warning(
                    f"Pydantic validation failed (falling back to legacy): {pydantic_error}"
                )
                # Continue with legacy validation below
        except ImportError:
            logger.debug("Pydantic schemas not available, using legacy validation")

        required_sections = [
            "paths",
            "bayesian",
            "race",
            "qualifying",
            "baseline_predictor",
        ]

        missing = []
        for section in required_sections:
            if section not in config:
                missing.append(section)

        if missing:
            raise ValueError(
                f"Config validation failed. Missing required sections: {missing}. "
                "Check your config file structure."
            )

        baseline_predictor_config = config.get("baseline_predictor")
        if not isinstance(baseline_predictor_config, dict):
            raise ValueError(
                "Config missing baseline_predictor section or it has an invalid structure"
            )
        if "qualifying" not in baseline_predictor_config:
            raise ValueError("Config missing baseline_predictor.qualifying section")
        if "race" not in baseline_predictor_config:
            raise ValueError("Config missing baseline_predictor.race section")

        validations: list[ValidationSpec] = [
            # Bayesian model parameters
            ("bayesian.base_volatility", float, 0.0, 1.0),
            ("bayesian.base_observation_noise", float, 0.0, 100.0),
            ("bayesian.shock_threshold", float, 0.0, 10.0),
            # Qualifying parameters
            ("baseline_predictor.qualifying.noise_std_sprint", float, 0.0, 0.5),
            ("baseline_predictor.qualifying.noise_std_normal", float, 0.0, 0.5),
            ("baseline_predictor.qualifying.team_weight", float, 0.0, 1.0),
            ("baseline_predictor.qualifying.skill_weight", float, 0.0, 1.0),
            ("baseline_predictor.qualifying.fp_blend_weight", float, 0.0, 1.0),
            ("baseline_predictor.qualifying.confidence_cap", int, 1, 100),
            ("baseline_predictor.qualifying.confidence_min", int, 1, 100),
            ("baseline_predictor.qualifying.default_skill", float, 0.0, 1.0),
            ("baseline_predictor.qualifying.default_team_strength", float, 0.0, 1.0),
            # Race parameters - base chaos
            ("baseline_predictor.race.base_chaos.dry", float, 0.0, 1.0),
            ("baseline_predictor.race.base_chaos.wet", float, 0.0, 1.0),
            ("baseline_predictor.race.track_chaos_multiplier", float, 0.0, 1.0),
            # Race parameters - safety car
            ("baseline_predictor.race.sc_base_probability.dry", float, 0.0, 1.0),
            ("baseline_predictor.race.sc_base_probability.wet", float, 0.0, 1.0),
            ("baseline_predictor.race.sc_track_modifier", float, 0.0, 1.0),
            ("baseline_predictor.race.safety_car_trigger_lap", int, 1, 70),
            # Race parameters - grid and pace
            ("baseline_predictor.race.grid_weight_min", float, 0.0, 1.0),
            ("baseline_predictor.race.grid_weight_multiplier", float, 0.0, 1.0),
            ("baseline_predictor.race.grid_divisor", int, 18, 22),
            ("baseline_predictor.race.pace_weight_base", float, 0.0, 1.0),
            ("baseline_predictor.race.pace_weight_track_modifier", float, 0.0, 1.0),
            # Race parameters - DNF caps
            ("baseline_predictor.race.dnf_rate_historical_cap", float, 0.0, 1.0),
            ("baseline_predictor.race.dnf_rate_final_cap", float, 0.0, 1.0),
            ("baseline_predictor.race.dnf_rate_floor", float, 0.0, 1.0),
            ("baseline_predictor.race.team_uncertainty_dnf_multiplier", float, 0.0, 1.0),
            ("baseline_predictor.race.missing_driver_teammate_weight", float, 0.0, 1.0),
            ("baseline_predictor.race.missing_driver_default_dnf_rate", float, 0.0, 1.0),
            ("baseline_predictor.race.missing_driver_rookie_dnf_penalty", float, 0.0, 0.5),
            ("baseline_predictor.race.missing_driver_second_year_penalty_scale", float, 0.0, 1.0),
            ("baseline_predictor.race.min_laps_for_compound_data", int, 1, 500),
            ("baseline_predictor.race.weekend_long_run_min_laps", int, 1, 70),
            ("baseline_predictor.race.long_run_outlier_threshold", float, 0.1, 5.0),
            # Race parameters - low-confidence interval floor
            (
                "baseline_predictor.race.position_interval_floor.apply_below_input_confidence",
                float,
                0.0,
                1.0,
            ),
            ("baseline_predictor.race.position_interval_floor.top_n", int, 1, 22),
            ("baseline_predictor.race.position_interval_floor.min_width", int, 0, 21),
            ("baseline_predictor.race.position_interval_floor.max_extra_width", int, 0, 21),
            # Race parameters - overtaking transition blend
            (
                "baseline_predictor.race.overtaking_transition.min_observed_weight",
                float,
                0.0,
                1.0,
            ),
            (
                "baseline_predictor.race.overtaking_transition.max_observed_weight",
                float,
                0.0,
                1.0,
            ),
            (
                "baseline_predictor.race.overtaking_transition.races_to_full_weight",
                int,
                1,
                50,
            ),
            (
                "baseline_predictor.race.overtaking_transition.max_delta_from_prior",
                float,
                0.0,
                1.0,
            ),
        ]

        errors = []
        for key, expected_type, min_val, max_val in validations:
            value = self.get(key)

            # Check if value exists
            if value is None:
                errors.append(f"Missing required config key: {key}")
                continue

            # Check type
            if not isinstance(value, expected_type):
                expected_type_name = expected_type.__name__
                errors.append(
                    f"Invalid type for {key}: expected {expected_type_name}, "
                    f"got {type(value).__name__}"
                )
                continue

            # Check range
            numeric_value = float(value)
            if not (min_val <= numeric_value <= max_val):
                errors.append(
                    f"Value out of range for {key}: {numeric_value} "
                    f"(must be between {min_val} and {max_val})"
                )

        if errors:
            error_msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)

        confidence_cap = self.get("baseline_predictor.qualifying.confidence_cap", 60)
        confidence_min = self.get("baseline_predictor.qualifying.confidence_min", 40)
        if confidence_min > confidence_cap:
            raise ValueError(
                "baseline_predictor.qualifying.confidence_min must be <= confidence_cap"
            )

        transition_min_weight = self.get(
            "baseline_predictor.race.overtaking_transition.min_observed_weight",
            0.12,
        )
        transition_max_weight = self.get(
            "baseline_predictor.race.overtaking_transition.max_observed_weight",
            0.65,
        )
        if transition_min_weight > transition_max_weight:
            raise ValueError(
                "baseline_predictor.race.overtaking_transition.min_observed_weight must be <= max_observed_weight"
            )

        fallback_tier = self.get("baseline_predictor.race.default_experience_tier", "developing")
        if fallback_tier not in {
            "rookie",
            "second_year",
            "sophomore",
            "developing",
            "established",
            "veteran",
            "sunset",
        }:
            raise ValueError(
                "baseline_predictor.race.default_experience_tier must be one of: "
                "rookie, second_year, developing, established, veteran, sunset"
            )

        clip_range = self.get("baseline_predictor.race.testing_modifier_clip_range", [-0.04, 0.04])
        if not isinstance(clip_range, list) or len(clip_range) != 2:
            raise ValueError(
                "baseline_predictor.race.testing_modifier_clip_range must be a 2-item list"
            )
        if clip_range[0] >= clip_range[1]:
            raise ValueError(
                "baseline_predictor.race.testing_modifier_clip_range lower bound must be < upper bound"
            )

        position_scaling = self.get("baseline_predictor.race.position_scaling", {})
        front_threshold = position_scaling.get("front_threshold", 3)
        upper_threshold = position_scaling.get("upper_threshold", 7)
        mid_threshold = position_scaling.get("mid_threshold", 12)
        if not (front_threshold < upper_threshold < mid_threshold):
            raise ValueError(
                "baseline_predictor.race.position_scaling thresholds must satisfy "
                "front_threshold < upper_threshold < mid_threshold"
            )

        team_weight = self.get("baseline_predictor.qualifying.team_weight", 0.7)
        skill_weight = self.get("baseline_predictor.qualifying.skill_weight", 0.3)
        weight_sum = team_weight + skill_weight

        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"baseline_predictor.qualifying weights must sum to 1.0, got {weight_sum:.3f} "
                f"(team_weight={team_weight}, skill_weight={skill_weight})"
            )

        logger.debug("Config validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation, returning default if not found."""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire config section."""
        if self._config is None:
            return {}
        value = self._config.get(section, {})
        return value if isinstance(value, dict) else {}

    def reload(self) -> None:
        """Force reload config from file."""
        self._config = None
        self._load()


# Singleton instance
_config = Config()


def get(key: str, default: Any = None) -> Any:
    """Get config value."""
    return _config.get(key, default)


def get_section(section: str) -> dict[str, Any]:
    """Get config section."""
    return _config.get_section(section)


def reload() -> None:
    """Reload config."""
    _config.reload()
