"""Tests for configuration schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config_schema import (
    BayesianConfig,
    BlendConfig,
    DNFConfig,
    RaceWeightsConfig,
    validate_config,
)


def test_bayesian_config_validates_volatility_range():
    """Volatility must stay between 0.0 and 1.0."""
    valid = BayesianConfig(base_volatility=0.5)
    assert valid.base_volatility == 0.5

    with pytest.raises(ValidationError):
        BayesianConfig(base_volatility=1.5)

    with pytest.raises(ValidationError):
        BayesianConfig(base_volatility=-0.1)


def test_race_weights_cap_each_component():
    """Top-level race weights are individually bounded."""
    valid = RaceWeightsConfig(
        pace_weight=0.4,
        grid_weight=0.3,
        overtaking_weight=0.15,
        tire_deg_weight=0.15,
    )
    assert valid.pace_weight == 0.4

    with pytest.raises(ValidationError):
        RaceWeightsConfig(pace_weight=1.5)


def test_dnf_config_validates_probability_fields():
    """DNF probabilities must stay inside valid ranges."""
    valid = DNFConfig(base_risk=0.05, driver_error_factor=0.15)
    assert valid.base_risk == 0.05

    with pytest.raises(ValidationError):
        DNFConfig(base_risk=1.5)


def test_blend_config_validates_weights():
    """Qualifying blend weights must stay in the unit interval."""
    valid = BlendConfig(default=0.7, fp3_only=0.8, fp1_only=0.4)
    assert valid.default == 0.7

    with pytest.raises(ValidationError):
        BlendConfig(default=1.2)


def test_validate_config_accepts_default_yaml():
    """The shipped YAML config should validate cleanly against the strict schema."""
    config_path = Path("config/default.yaml")
    config_dict = yaml.safe_load(config_path.read_text())

    validated = validate_config(config_dict)

    assert validated.grid.size == 22
    assert validated.learning.min_samples == 3
    assert validated.baseline_predictor.qualifying.fp_blend_weight == pytest.approx(0.62)
    assert (
        validated.baseline_predictor.race.overtake_model.zone_front_probability_scale
        == pytest.approx(0.55)
    )
    assert validated.dashboard.prediction_precompute.reconcile_accuracy_after_warmup is True
    assert validated.dashboard.prediction_precompute.learn_completed_races_before_warmup is True
    assert validated.dashboard.prediction_precompute.accuracy_reconcile_lookback_days == 14


def test_validate_config_rejects_unknown_nested_keys():
    """Any YAML key missing from the schema should fail validation."""
    config_dict = yaml.safe_load(Path("config/default.yaml").read_text())
    config_dict["baseline_predictor"]["race"]["unknown_new_knob"] = 123

    with pytest.raises(ValidationError) as exc_info:
        validate_config(config_dict)

    assert "unknown_new_knob" in str(exc_info.value)


def test_validate_config_rejects_invalid_numeric_values():
    """Strict schema validation should still reject bad scalar values."""
    config_dict = yaml.safe_load(Path("config/default.yaml").read_text())
    config_dict["bayesian"]["base_volatility"] = 2.5

    with pytest.raises(ValidationError) as exc_info:
        validate_config(config_dict)

    assert "base_volatility" in str(exc_info.value)


def test_config_loader_integration():
    """Config loader should continue to validate the real config file."""
    from src.utils.config_loader import Config

    config = Config()
    assert config._config is not None

    base_volatility = config.get("bayesian.base_volatility")
    assert isinstance(base_volatility, int | float)
    assert 0.0 <= base_volatility <= 1.0
