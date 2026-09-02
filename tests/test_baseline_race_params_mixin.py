from __future__ import annotations

import src.predictors.baseline.race.params_mixin as params_module
from src.predictors.baseline.race.params_mixin import BaselineRaceParamsMixin


class DummyRaceParams(BaselineRaceParamsMixin):
    pass


def test_load_race_params_reads_expected_config_keys(patcher):
    helper = DummyRaceParams()
    calls = []

    def _fake_get(key: str, default):
        calls.append(key)
        return default

    patcher.setattr(params_module.config_loader, "get", _fake_get)

    params = helper._load_race_params()

    assert len(params) >= 30
    assert "base_chaos_dry" in params
    assert "base_chaos_wet" in params
    assert "mixed_weather_chaos_blend" in params
    assert "teammate_variance_std" in params
    assert "teammate_setup_offset_ratio" in params
    assert "teammate_variance_lap_ratio" in params
    assert "grid_divisor" in params
    assert "position_scaling_front_threshold" in params
    assert "baseline_predictor.race.base_chaos.dry" in calls
    assert "baseline_predictor.race.base_chaos.mixed_blend" in calls
    assert "baseline_predictor.race.overtaking_skill_multiplier" in calls
    assert "baseline_predictor.race.teammate_variance_std" in calls
    assert "baseline_predictor.race.teammate_setup_offset_ratio" in calls
    assert "baseline_predictor.race.teammate_variance_lap_ratio" in calls
    assert "baseline_predictor.race.grid_divisor" in calls
