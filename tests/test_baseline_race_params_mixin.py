from __future__ import annotations

import numpy as np

import src.predictors.baseline.race.params_mixin as params_module
from src.predictors.baseline.race.params_mixin import BaselineRaceParamsMixin


class DummyRaceParams(BaselineRaceParamsMixin):
    pass


def _base_params() -> dict:
    return {
        "grid_weight_min": 0.2,
        "grid_weight_multiplier": 0.4,
        "race_advantage_multiplier": 0.5,
        "overtaking_skill_multiplier": 0.3,
        "overtaking_grid_threshold": 5,
        "overtaking_track_threshold": 0.5,
        "lap1_front_row_chaos": 0.0,
        "lap1_upper_midfield_chaos": 0.0,
        "lap1_midfield_chaos": 0.0,
        "lap1_back_field_chaos": 0.0,
        "strategy_variance_base": 0.0,
        "strategy_track_modifier": 0.5,
        "safety_car_luck_range": 0.0,
        "pace_weight_base": 0.5,
        "pace_weight_track_modifier": 0.1,
        "teammate_variance_std": 0.0,
        "grid_divisor": 21,
        "position_scaling_front_threshold": 3,
        "position_scaling_front_scale": 0.1,
        "position_scaling_upper_threshold": 7,
        "position_scaling_upper_scale": 0.3,
        "position_scaling_mid_threshold": 12,
        "position_scaling_mid_scale": 0.6,
        "position_scaling_back_scale": 1.0,
        "base_chaos": {"dry": 0.0, "wet": 0.0},
    }


def test_calculate_driver_race_score_dnf_branch():
    helper = DummyRaceParams()
    rng = np.random.default_rng(123)

    info = {
        "grid_pos": 8,
        "race_advantage": 0.2,
        "overtaking_skill": 0.7,
        "team_strength": 0.8,
        "skill": 0.9,
        "dnf_probability": 1.0,
    }

    score, dnf = helper._calculate_driver_race_score(
        info=info,
        track_overtaking=0.4,
        weather="dry",
        safety_car=False,
        params=_base_params(),
        rng=rng,
    )

    assert dnf is True
    assert -11.0 < score < -10.0


def test_calculate_driver_race_score_overtaking_and_weather_paths():
    helper = DummyRaceParams()
    params = _base_params()

    info = {
        "grid_pos": 12,
        "race_advantage": 0.3,
        "overtaking_skill": 0.8,
        "team_strength": 0.7,
        "skill": 0.6,
        "dnf_probability": 0.0,
    }

    score, dnf = helper._calculate_driver_race_score(
        info=info,
        track_overtaking=0.2,
        weather="rain",
        safety_car=True,
        params=params,
        rng=np.random.default_rng(7),
    )

    assert dnf is False
    assert isinstance(score, float)


def test_calculate_driver_race_score_without_overtaking_boost():
    helper = DummyRaceParams()
    params = _base_params()
    info = {
        "grid_pos": 3,
        "race_advantage": 0.1,
        "overtaking_skill": 0.9,
        "team_strength": 0.8,
        "skill": 0.85,
        "dnf_probability": 0.0,
    }

    score, dnf = helper._calculate_driver_race_score(
        info=info,
        track_overtaking=0.9,
        weather="dry",
        safety_car=False,
        params=params,
        rng=np.random.default_rng(99),
    )

    assert dnf is False
    assert score > 0


def test_load_race_params_reads_expected_config_keys(patcher):
    helper = DummyRaceParams()
    calls = []

    def _fake_get(key: str, default):
        calls.append(key)
        return default

    patcher.setattr(params_module.config_loader, "get", _fake_get)

    params = helper._load_race_params()

    assert len(params) == 30
    assert "base_chaos_dry" in params
    assert "base_chaos_wet" in params
    assert "teammate_variance_std" in params
    assert "grid_divisor" in params
    assert "position_scaling_front_threshold" in params
    assert "baseline_predictor.race.base_chaos.dry" in calls
    assert "baseline_predictor.race.overtaking_skill_multiplier" in calls
    assert "baseline_predictor.race.teammate_variance_std" in calls
    assert "baseline_predictor.race.grid_divisor" in calls
