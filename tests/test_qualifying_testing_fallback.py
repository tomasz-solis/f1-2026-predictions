from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import pytest

import src.predictors.baseline.qualifying_mixin as qualifying_module
from src.predictors.baseline.data_mixin import BaselineDataMixin
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin


class DummyConfig:
    def __init__(self, overrides: dict[str, object] | None = None):
        self._overrides = overrides or {}

    def get(self, key: str, default=None):
        return self._overrides.get(key, default)


class DummyQualifyingPredictor(BaselineQualifyingMixin, BaselineDataMixin):
    def __init__(self, config_overrides: dict[str, object] | None = None):
        BaselineDataMixin.__init__(self)
        self.seed = 123
        self.config = DummyConfig(config_overrides)
        self.teams = {
            "Team A": {
                "testing_characteristics_profiles": {
                    "short_run": {"overall_pace": 0.90},
                }
            },
            "Team B": {
                "testing_characteristics_profiles": {
                    "short_run": {"overall_pace": 0.10},
                }
            },
        }
        self.drivers = {
            "AAA": {
                "pace": {"quali_pace": 0.5},
                "racecraft": {"skill_score": 0.5},
                "experience": {"tier": "established"},
            },
            "BBB": {
                "pace": {"quali_pace": 0.5},
                "racecraft": {"skill_score": 0.5},
                "experience": {"tier": "established"},
            },
        }
        self._base_strengths = {"Team A": 0.40, "Team B": 0.60}

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        _ = race_name
        return self._base_strengths[team]


@contextmanager
def _patched_prediction_dependencies():
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(qualifying_module, "is_sprint_weekend", lambda year, race_name: False)
        )
        stack.enter_context(
            patch.object(
                qualifying_module,
                "get_lineups",
                lambda year, race_name: {"Team A": ["AAA"], "Team B": ["BBB"]},
            )
        )
        stack.enter_context(
            patch.object(
                qualifying_module,
                "get_best_fp_performance",
                lambda **kwargs: (None, None, None),
            )
        )
        yield


def test_predict_qualifying_uses_testing_short_run_fallback():
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.enable_driver_fp_adjustment": False,
            "baseline_predictor.qualifying.testing_short_run_modifier_scale": 0.0,
            "baseline_predictor.qualifying.testing_fallback_min_teams": 2,
            "baseline_predictor.qualifying.testing_fallback_modifier_scale": 0.10,
            "baseline_predictor.qualifying.testing_fallback_modifier_clip_range": [-0.05, 0.05],
            "baseline_predictor.race.testing_profile_weights.short_run": {"overall_pace": 1.0},
        }
    )

    captured: dict[str, object] = {}

    def _fake_run(all_drivers, n_simulations, is_sprint, has_practice_data, rng):
        _ = (n_simulations, is_sprint, rng)
        captured["has_practice_data"] = has_practice_data
        captured["all_drivers"] = all_drivers
        return {"AAA": [1], "BBB": [2]}

    def _fake_aggregate(position_records, all_drivers):
        _ = position_records
        ordered = sorted(all_drivers, key=lambda driver: driver["team_strength"], reverse=True)
        return [
            {
                "position": index + 1,
                "driver": driver_info["driver"],
                "team": driver_info["team"],
                "median_position": index + 1,
                "position_distribution": [index + 1],
            }
            for index, driver_info in enumerate(ordered)
        ]

    with _patched_prediction_dependencies():
        with patch.object(predictor, "_run_qualifying_simulations", _fake_run):
            with patch.object(predictor, "_aggregate_grid_results", _fake_aggregate):
                result = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=1)

    assert captured["has_practice_data"] is False
    assert result["blend_used"] is False
    assert result["testing_fallback_used"] is True
    assert result["data_source"] == "Testing short-run profile blend (no weekend practice data)"

    strengths = {driver["team"]: driver["team_strength"] for driver in captured["all_drivers"]}
    assert strengths["Team A"] == pytest.approx(0.55)
    assert strengths["Team B"] == pytest.approx(0.45)
    assert all("experience_total_races" in driver for driver in captured["all_drivers"])


def test_predict_qualifying_remains_model_only_without_testing_profiles():
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.enable_driver_fp_adjustment": False,
            "baseline_predictor.qualifying.testing_short_run_modifier_scale": 0.0,
            "baseline_predictor.qualifying.testing_fallback_min_teams": 2,
            "baseline_predictor.race.testing_profile_weights.short_run": {"overall_pace": 1.0},
        }
    )
    predictor.teams = {"Team A": {}, "Team B": {}}

    captured: dict[str, object] = {}

    def _fake_run(all_drivers, n_simulations, is_sprint, has_practice_data, rng):
        _ = (n_simulations, is_sprint, rng)
        captured["has_practice_data"] = has_practice_data
        captured["all_drivers"] = all_drivers
        return {"AAA": [1], "BBB": [2]}

    with _patched_prediction_dependencies():
        with patch.object(predictor, "_run_qualifying_simulations", _fake_run):
            with patch.object(
                predictor,
                "_aggregate_grid_results",
                lambda position_records, all_drivers: [
                    {
                        "position": 1,
                        "driver": all_drivers[0]["driver"],
                        "team": all_drivers[0]["team"],
                        "median_position": 1,
                        "position_distribution": [1],
                    }
                ],
            ):
                result = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=1)

    assert captured["has_practice_data"] is False
    assert result["blend_used"] is False
    assert result["testing_fallback_used"] is False
    assert result["data_source"] == "Model-only (no practice/testing data)"

    strengths = {driver["team"]: driver["team_strength"] for driver in captured["all_drivers"]}
    assert strengths["Team A"] == pytest.approx(0.40)
    assert strengths["Team B"] == pytest.approx(0.60)


def test_model_only_teammate_anchor_reduces_extreme_inversions():
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.12,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.04,
        }
    )

    all_drivers = [
        {
            "driver": "HAD",
            "team": "Red Bull Racing",
            "team_strength": 0.59,
            "skill": 0.67,
            "quali_pace": 0.705,
            "experience_tier": "rookie",
        },
        {
            "driver": "VER",
            "team": "Red Bull Racing",
            "team_strength": 0.59,
            "skill": 0.99,
            "quali_pace": 0.95,
            "experience_tier": "veteran",
        },
    ]

    position_records = predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1500,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    had_ahead_count = sum(
        1
        for had_pos, ver_pos in zip(position_records["HAD"], position_records["VER"], strict=True)
        if had_pos < ver_pos
    )
    had_ahead_ratio = had_ahead_count / 1500

    assert had_ahead_ratio < 0.35


def test_run_qualifying_simulations_applies_learned_position_adjustments():
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.learning.position_to_score_scale": 0.04,
            "baseline_predictor.qualifying.learning.with_practice_multiplier": 0.65,
        }
    )

    all_drivers = [
        {
            "driver": "AAA",
            "team": "Team A",
            "team_strength": 0.5,
            "skill": 0.5,
            "quali_pace": 0.5,
            "experience_tier": "established",
            "learned_position_adjustment": 2.0,
        },
        {
            "driver": "BBB",
            "team": "Team B",
            "team_strength": 0.5,
            "skill": 0.5,
            "quali_pace": 0.5,
            "experience_tier": "established",
            "learned_position_adjustment": -2.0,
        },
    ]

    position_records = predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=400,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    aaa_ahead_count = sum(
        1
        for aaa_pos, bbb_pos in zip(position_records["AAA"], position_records["BBB"], strict=True)
        if aaa_pos < bbb_pos
    )
    aaa_ahead_ratio = aaa_ahead_count / 400

    assert aaa_ahead_ratio > 0.90


def test_testing_short_run_fallback_blends_toward_balanced_on_profile_divergence():
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.testing_fallback_min_teams": 2,
            "baseline_predictor.qualifying.testing_fallback_short_weight_min": 0.35,
            "baseline_predictor.qualifying.testing_fallback_short_weight_max": 0.85,
            "baseline_predictor.qualifying.testing_fallback_divergence_scale": 1.4,
        }
    )
    predictor.teams = {
        "Team A": {
            "testing_characteristics_profiles": {
                "short_run": {"overall_pace": 0.90},
                "balanced": {"overall_pace": 0.20},
            }
        },
        "Team B": {
            "testing_characteristics_profiles": {
                "short_run": {"overall_pace": 0.60},
                "balanced": {"overall_pace": 0.55},
            }
        },
    }

    fallback_scores = predictor._build_testing_short_run_fallback(
        lineups={"Team A": ["AAA"], "Team B": ["BBB"]},
        metric_weights={"overall_pace": 1.0},
    )

    assert fallback_scores is not None
    # Team A has large short-vs-balanced disagreement, so fallback should avoid
    # trusting short-run alone and land near a blended mid-point.
    assert fallback_scores["Team A"] < 0.60
    # Team B has low disagreement, so score should stay near short-run pace.
    assert fallback_scores["Team B"] > 0.55


def test_model_only_negative_delta_shrink_prevents_extreme_rookie_drop():
    all_drivers = [
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_strength": 0.75,
            "skill": 0.90,
            "quali_pace": 0.88,
            "experience_tier": "established",
        },
        {
            "driver": "ANT",
            "team": "Mercedes",
            "team_strength": 0.75,
            "skill": 0.27,
            "quali_pace": 0.32,
            "experience_tier": "rookie",
        },
        {
            "driver": "DRV1",
            "team": "Team1",
            "team_strength": 0.55,
            "skill": 0.55,
            "quali_pace": 0.55,
            "experience_tier": "established",
        },
        {
            "driver": "DRV2",
            "team": "Team2",
            "team_strength": 0.52,
            "skill": 0.52,
            "quali_pace": 0.52,
            "experience_tier": "established",
        },
        {
            "driver": "DRV3",
            "team": "Team3",
            "team_strength": 0.50,
            "skill": 0.50,
            "quali_pace": 0.50,
            "experience_tier": "established",
        },
        {
            "driver": "DRV4",
            "team": "Team4",
            "team_strength": 0.48,
            "skill": 0.48,
            "quali_pace": 0.48,
            "experience_tier": "established",
        },
    ]

    predictor_without_extra = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.model_only_negative_delta_shrink_cap": 0.0,
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.12,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.04,
        }
    )
    predictor_with_extra = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.model_only_negative_delta_shrink_cap": 0.25,
            "baseline_predictor.qualifying.model_only_negative_delta_shrink_scale": 1.0,
            "baseline_predictor.qualifying.model_only_negative_delta_threshold": 0.08,
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.12,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.04,
        }
    )

    baseline_records = predictor_without_extra._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1200,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )
    adjusted_records = predictor_with_extra._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1200,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    ant_avg_baseline = float(np.mean(baseline_records["ANT"]))
    ant_avg_adjusted = float(np.mean(adjusted_records["ANT"]))
    assert ant_avg_adjusted < ant_avg_baseline


def test_effective_experience_tier_upgrades_second_year_driver():
    predictor = DummyQualifyingPredictor()

    driver_data = {
        "experience": {
            "tier": "rookie",
            "years_of_experience": 0,
            "debut_year": 2025,
        }
    }

    tier_2025 = predictor._resolve_effective_experience_tier(driver_data, prediction_year=2025)
    tier_2026 = predictor._resolve_effective_experience_tier(driver_data, prediction_year=2026)

    assert tier_2025 == "rookie"
    assert tier_2026 == "second_year"


def test_model_only_anchor_penalty_scaled_by_experience_tier():
    all_drivers = [
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.90,
            "quali_pace": 0.88,
            "experience_tier": "established",
        },
        {
            "driver": "ANT",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.32,
            "quali_pace": 0.35,
            "experience_tier": "developing",
        },
        {
            "driver": "DRV1",
            "team": "Team1",
            "team_strength": 0.58,
            "skill": 0.58,
            "quali_pace": 0.58,
            "experience_tier": "established",
        },
        {
            "driver": "DRV2",
            "team": "Team2",
            "team_strength": 0.56,
            "skill": 0.56,
            "quali_pace": 0.56,
            "experience_tier": "established",
        },
        {
            "driver": "DRV3",
            "team": "Team3",
            "team_strength": 0.54,
            "skill": 0.54,
            "quali_pace": 0.54,
            "experience_tier": "established",
        },
        {
            "driver": "DRV4",
            "team": "Team4",
            "team_strength": 0.52,
            "skill": 0.52,
            "quali_pace": 0.52,
            "experience_tier": "established",
        },
    ]

    predictor_unscaled = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.0,
            "baseline_predictor.qualifying.noise_std_sprint": 0.0,
            "baseline_predictor.qualifying.teammate_setup_std": 0.0,
            "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.0,
            "baseline_predictor.qualifying.model_only_experience_shrink": {
                "rookie": 0.0,
                "developing": 0.0,
                "unknown": 0.0,
            },
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.20,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.06,
            "baseline_predictor.qualifying.model_only_teammate_anchor_experience_multiplier": {
                "rookie": 1.0,
                "developing": 1.0,
                "unknown": 1.0,
            },
        }
    )
    predictor_scaled = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.0,
            "baseline_predictor.qualifying.noise_std_sprint": 0.0,
            "baseline_predictor.qualifying.teammate_setup_std": 0.0,
            "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.0,
            "baseline_predictor.qualifying.model_only_experience_shrink": {
                "rookie": 0.0,
                "developing": 0.0,
                "unknown": 0.0,
            },
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.20,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.06,
            "baseline_predictor.qualifying.model_only_teammate_anchor_experience_multiplier": {
                "rookie": 0.30,
                "developing": 0.55,
                "unknown": 0.45,
            },
        }
    )

    unscaled = predictor_unscaled._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )
    scaled = predictor_scaled._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    ant_pos_unscaled = unscaled["ANT"][0]
    ant_pos_scaled = scaled["ANT"][0]
    assert ant_pos_scaled <= ant_pos_unscaled


def test_model_only_teammate_gap_cap_limits_second_year_extreme_drop():
    all_drivers = [
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.90,
            "quali_pace": 0.88,
            "experience_tier": "established",
            "experience_total_races": 130,
        },
        {
            "driver": "ANT",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.27,
            "quali_pace": 0.31,
            "experience_tier": "second_year",
            "experience_total_races": 24,
        },
        {
            "driver": "DRV1",
            "team": "Team1",
            "team_strength": 0.58,
            "skill": 0.58,
            "quali_pace": 0.58,
            "experience_tier": "established",
        },
        {
            "driver": "DRV2",
            "team": "Team2",
            "team_strength": 0.56,
            "skill": 0.56,
            "quali_pace": 0.56,
            "experience_tier": "established",
        },
        {
            "driver": "DRV3",
            "team": "Team3",
            "team_strength": 0.54,
            "skill": 0.54,
            "quali_pace": 0.54,
            "experience_tier": "established",
        },
        {
            "driver": "DRV4",
            "team": "Team4",
            "team_strength": 0.52,
            "skill": 0.52,
            "quali_pace": 0.52,
            "experience_tier": "established",
        },
    ]

    predictor_without_cap = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.0,
            "baseline_predictor.qualifying.noise_std_sprint": 0.0,
            "baseline_predictor.qualifying.teammate_setup_std": 0.0,
            "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.0,
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.20,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.06,
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience": {},
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience": {},
        }
    )
    predictor_with_cap = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.0,
            "baseline_predictor.qualifying.noise_std_sprint": 0.0,
            "baseline_predictor.qualifying.teammate_setup_std": 0.0,
            "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.0,
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.20,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.06,
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience": {
                "rookie": 0.16,
                "second_year": 0.10,
                "unknown": 0.12,
            },
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience": {
                "rookie": 40,
                "second_year": 55,
                "unknown": 45,
            },
        }
    )

    uncapped = predictor_without_cap._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )
    capped = predictor_with_cap._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    ant_pos_uncapped = uncapped["ANT"][0]
    ant_pos_capped = capped["ANT"][0]
    assert ant_pos_capped < ant_pos_uncapped


def test_model_only_teammate_gap_cap_respects_max_races_threshold():
    all_drivers = [
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.90,
            "quali_pace": 0.88,
            "experience_tier": "established",
            "experience_total_races": 130,
        },
        {
            "driver": "ANT",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.27,
            "quali_pace": 0.31,
            "experience_tier": "second_year",
            "experience_total_races": 90,
        },
        {
            "driver": "DRV1",
            "team": "Team1",
            "team_strength": 0.58,
            "skill": 0.58,
            "quali_pace": 0.58,
            "experience_tier": "established",
        },
        {
            "driver": "DRV2",
            "team": "Team2",
            "team_strength": 0.56,
            "skill": 0.56,
            "quali_pace": 0.56,
            "experience_tier": "established",
        },
    ]

    base_overrides = {
        "baseline_predictor.qualifying.noise_std_normal": 0.0,
        "baseline_predictor.qualifying.noise_std_sprint": 0.0,
        "baseline_predictor.qualifying.teammate_setup_std": 0.0,
        "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.0,
        "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.20,
        "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.06,
    }

    predictor_without_cap = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience": {},
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience": {},
        }
    )
    predictor_with_cap = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience": {
                "second_year": 0.10
            },
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience": {
                "second_year": 55
            },
        }
    )

    uncapped = predictor_without_cap._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )
    capped = predictor_with_cap._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    assert capped["ANT"][0] == uncapped["ANT"][0]


def test_model_only_teammate_gap_cap_scales_with_sample_size():
    base_drivers = [
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.90,
            "quali_pace": 0.88,
            "experience_tier": "established",
            "experience_total_races": 130,
        },
        {
            "driver": "ANT",
            "team": "Mercedes",
            "team_strength": 0.74,
            "skill": 0.27,
            "quali_pace": 0.31,
            "experience_tier": "second_year",
        },
        {
            "driver": "DRV1",
            "team": "Team1",
            "team_strength": 0.58,
            "skill": 0.58,
            "quali_pace": 0.58,
            "experience_tier": "established",
        },
        {
            "driver": "DRV2",
            "team": "Team2",
            "team_strength": 0.56,
            "skill": 0.56,
            "quali_pace": 0.56,
            "experience_tier": "established",
        },
        {
            "driver": "DRV3",
            "team": "Team3",
            "team_strength": 0.54,
            "skill": 0.54,
            "quali_pace": 0.54,
            "experience_tier": "established",
        },
        {
            "driver": "DRV4",
            "team": "Team4",
            "team_strength": 0.52,
            "skill": 0.52,
            "quali_pace": 0.52,
            "experience_tier": "established",
        },
    ]
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.0,
            "baseline_predictor.qualifying.noise_std_sprint": 0.0,
            "baseline_predictor.qualifying.teammate_setup_std": 0.0,
            "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.0,
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.20,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.06,
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience": {
                "second_year": 0.10
            },
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience": {
                "second_year": 55
            },
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_min_scale": 0.35,
        }
    )

    low_sample_drivers = [dict(item) for item in base_drivers]
    low_sample_drivers[1]["experience_total_races"] = 10
    high_sample_drivers = [dict(item) for item in base_drivers]
    high_sample_drivers[1]["experience_total_races"] = 50

    low_sample = predictor._run_qualifying_simulations(
        all_drivers=low_sample_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )
    high_sample = predictor._run_qualifying_simulations(
        all_drivers=high_sample_drivers,
        n_simulations=1,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    assert low_sample["ANT"][0] <= high_sample["ANT"][0]
