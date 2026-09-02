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
        logger.info("Configuration loaded successfully from %s", config_path)

    def _validate_config(self) -> None:
        """Validate against the Pydantic schema, then check constraints it cannot express."""
        config = self._config
        if config is None:
            raise ValueError("Config validation failed: configuration is not loaded")

        from src.utils.config_schema import validate_config

        validate_config(config)

        required_sections = [
            "paths",
            "bayesian",
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
