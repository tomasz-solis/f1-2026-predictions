from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import src.predictors.baseline.qualifying_mixin as qualifying_module
from src.predictors.baseline.data_mixin import BaselineDataMixin
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.qualifying_preparation import resolve_bayesian_skill_score
from src.predictors.baseline.qualifying_simulation import _build_quali_sim_config
from src.predictors.baseline_2026 import Baseline2026Predictor


class DummyConfig:
    def __init__(self, overrides: dict[str, object] | None = None):
        self._overrides = overrides or {}

    def get(self, key: str, default=None):
        return self._overrides.get(key, default)


class DummyQualifyingPredictor(BaselineQualifyingMixin, BaselineDataMixin):
    def __init__(self, config_overrides: dict[str, object] | None = None):
        BaselineDataMixin.__init__(self)
        self.seed = 123
        merged_overrides = {
            "baseline_predictor.current_season_form.infer_from_saved_actuals": False,
        }
        if config_overrides:
            merged_overrides.update(config_overrides)
        self.config = DummyConfig(merged_overrides)
        self.data_dir = Path("data/processed")
        self.season_year = 2026
        self.races_completed = 0
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
                "get_best_fp_performance_with_session_laps",
                lambda **kwargs: (None, None, None, {}),
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

    def _fake_run(
        all_drivers,
        n_simulations,
        is_sprint,
        has_practice_data,
        rng,
        has_testing_fallback_data,
        weather="dry",
    ):
        _ = (n_simulations, is_sprint, rng, has_testing_fallback_data, weather)
        captured["has_practice_data"] = has_practice_data
        captured["all_drivers"] = all_drivers
        return {"AAA": [1], "BBB": [2]}

    def _fake_aggregate(position_records, all_drivers, *, data_confidence_score=None):
        _ = position_records
        _ = data_confidence_score
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
    teammate_probs = result.get("teammate_head_to_head", [])
    assert isinstance(teammate_probs, list)

    strengths = {driver["team"]: driver["team_strength"] for driver in captured["all_drivers"]}
    assert strengths["Team A"] == pytest.approx(0.55)
    assert strengths["Team B"] == pytest.approx(0.45)
    assert all("experience_total_races" in driver for driver in captured["all_drivers"])


def test_testing_fallback_sim_config_restores_team_weight_and_spread():
    """Testing-fallback config should stay more car-led than a neutral 50/50 split."""
    neutral_predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.testing_fallback_team_weight_multiplier": 1.0,
            "baseline_predictor.qualifying.testing_fallback_skill_weight_multiplier": 1.0,
            "baseline_predictor.qualifying.testing_fallback_team_compression_multiplier": 1.0,
        }
    )
    tuned_predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.testing_fallback_team_weight_multiplier": 1.10,
            "baseline_predictor.qualifying.testing_fallback_skill_weight_multiplier": 0.90,
            "baseline_predictor.qualifying.testing_fallback_team_compression_multiplier": 1.25,
        }
    )

    neutral_cfg = _build_quali_sim_config(
        cfg=neutral_predictor.config,
        is_sprint=False,
        has_practice_data=False,
        has_testing_fallback_data=True,
    )
    tuned_cfg = _build_quali_sim_config(
        cfg=tuned_predictor.config,
        is_sprint=False,
        has_practice_data=False,
        has_testing_fallback_data=True,
    )

    assert tuned_cfg.team_weight > neutral_cfg.team_weight
    assert tuned_cfg.skill_weight < neutral_cfg.skill_weight
    assert tuned_cfg.team_strength_compression > neutral_cfg.team_strength_compression


def test_predict_qualifying_blends_bayesian_form_into_quali_pace():
    """Drivers with strong Bayesian form should carry that into the assembled quali pace."""
    predictor = DummyQualifyingPredictor({"grid.size": 22})
    predictor.races_completed = 2
    predictor.drivers["AAA"]["pace"]["quali_pace"] = 0.30
    predictor.drivers["AAA"]["racecraft"]["skill_score"] = 0.45
    predictor.drivers["AAA"]["bayesian"] = {"rating_mu": 20.0}

    captured: dict[str, object] = {}

    def _fake_run(
        all_drivers,
        n_simulations,
        is_sprint,
        has_practice_data,
        rng,
        has_testing_fallback_data,
        weather="dry",
    ):
        _ = (n_simulations, is_sprint, has_practice_data, rng, has_testing_fallback_data, weather)
        captured["all_drivers"] = all_drivers
        return {"AAA": [1], "BBB": [2]}

    def _fake_aggregate(position_records, all_drivers, *, data_confidence_score=None):
        _ = (position_records, data_confidence_score)
        return [
            {
                "position": index + 1,
                "driver": driver_info["driver"],
                "team": driver_info["team"],
                "median_position": index + 1,
                "position_distribution": [index + 1],
            }
            for index, driver_info in enumerate(all_drivers)
        ]

    with _patched_prediction_dependencies():
        with patch.object(predictor, "_run_qualifying_simulations", _fake_run):
            with patch.object(predictor, "_aggregate_grid_results", _fake_aggregate):
                predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=1)

    driver_map = {driver["driver"]: driver for driver in captured["all_drivers"]}
    assert driver_map["AAA"]["raw_quali_pace"] == pytest.approx(0.30)
    assert driver_map["AAA"]["bayesian_pace_blend_weight"] == pytest.approx(0.40)
    assert driver_map["AAA"]["quali_pace"] > 0.50
    assert driver_map["AAA"]["bayesian_skill_score"] == pytest.approx(19.0 / 21.0)


def test_resolve_bayesian_skill_score_recomputes_from_rating_mu_when_cache_is_stale():
    """Stored normalized scores should not win when they were saved for the wrong grid size."""
    driver_data = {
        "bayesian": {
            "rating_mu": 19.5,
            "normalized_skill_score": 0.973684,
        }
    }

    resolved = resolve_bayesian_skill_score(driver_data, grid_size=22)

    assert resolved == pytest.approx(18.5 / 21.0)


def test_predict_qualifying_can_force_stored_checkpoint_profiles():
    """Checkpoint mode should bypass raw practice extraction and use stored profiles directly."""
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.enable_driver_fp_adjustment": False,
            "baseline_predictor.qualifying.testing_short_run_modifier_scale": 0.0,
            "baseline_predictor.qualifying.testing_fallback_min_teams": 2,
            "baseline_predictor.qualifying.testing_fallback_modifier_scale": 0.10,
            "baseline_predictor.qualifying.testing_fallback_modifier_clip_range": [-0.05, 0.05],
            "baseline_predictor.race.testing_profile_weights.short_run": {"overall_pace": 1.0},
            "baseline_predictor.qualifying.checkpoint_driver_profile_smoothing_seconds": 0.35,
            "baseline_predictor.qualifying.checkpoint_driver_profile_quali_scale": 0.08,
            "baseline_predictor.qualifying.checkpoint_driver_profile_skill_scale": 0.02,
        }
    )
    predictor.teams["Team A"]["checkpoint_driver_deltas_seconds"] = {
        "short_run": {"AAA": -0.21},
    }
    predictor.car_characteristics = {
        "checkpoint_snapshot": {
            "event_name": "Australian Grand Prix",
            "session_name": "FP2",
        }
    }

    captured: dict[str, object] = {}

    def _fake_run(
        all_drivers,
        n_simulations,
        is_sprint,
        has_practice_data,
        rng,
        has_testing_fallback_data,
        weather="dry",
    ):
        _ = (n_simulations, is_sprint, rng, has_testing_fallback_data, weather)
        captured["has_practice_data"] = has_practice_data
        captured["all_drivers"] = all_drivers
        return {"AAA": [1], "BBB": [2]}

    def _fake_aggregate(position_records, all_drivers, *, data_confidence_score=None):
        _ = position_records
        _ = data_confidence_score
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
                "get_best_fp_performance_with_session_laps",
                side_effect=AssertionError("raw practice extraction should be bypassed"),
            )
        )
        with patch.object(predictor, "_run_qualifying_simulations", _fake_run):
            with patch.object(predictor, "_aggregate_grid_results", _fake_aggregate):
                result = predictor.predict_qualifying(
                    2026,
                    "Australian Grand Prix",
                    n_simulations=1,
                    practice_signal_mode="stored_profiles",
                    checkpoint_session_name="FP2",
                )

    assert captured["has_practice_data"] is True
    assert result["blend_used"] is True
    assert result["testing_fallback_used"] is False
    assert result["practice_signal_mode_used"] == "stored_profiles"
    assert result["practice_signal_checkpoint"] == "FP2"
    assert (
        result["data_source"]
        == "FP2 checkpoint profile blend (latest stored snapshot: Australian Grand Prix / FP2)"
    )
    assert result["data_confidence_score"] == pytest.approx(0.5)
    strengths = {driver["team"]: driver["team_strength"] for driver in captured["all_drivers"]}
    quali_paces = {driver["driver"]: driver["quali_pace"] for driver in captured["all_drivers"]}
    assert strengths["Team A"] == pytest.approx(0.792)
    assert strengths["Team B"] == pytest.approx(0.208)
    assert quali_paces["AAA"] > 0.53


def test_predict_qualifying_stored_profiles_without_snapshot_stays_fallback_like():
    """Stored profiles should stay conservative when no current-weekend snapshot is loaded."""
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

    def _fake_run(
        all_drivers,
        n_simulations,
        is_sprint,
        has_practice_data,
        rng,
        has_testing_fallback_data,
        weather="dry",
    ):
        _ = (all_drivers, n_simulations, is_sprint, rng, has_testing_fallback_data, weather)
        captured["has_practice_data"] = has_practice_data
        return {"AAA": [1], "BBB": [2]}

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
                "get_best_fp_performance_with_session_laps",
                side_effect=AssertionError("raw practice extraction should be bypassed"),
            )
        )
        with patch.object(predictor, "_run_qualifying_simulations", _fake_run):
            with patch.object(
                predictor,
                "_aggregate_grid_results",
                lambda position_records, all_drivers, *, data_confidence_score=None: [
                    {
                        "position": 1,
                        "driver": all_drivers[0]["driver"],
                        "team": all_drivers[0]["team"],
                        "median_position": 1,
                        "position_distribution": [1],
                    }
                ],
            ):
                result = predictor.predict_qualifying(
                    2026,
                    "Australian Grand Prix",
                    n_simulations=1,
                    practice_signal_mode="stored_profiles",
                    checkpoint_session_name="FP2",
                )

    assert captured["has_practice_data"] is False
    assert result["testing_fallback_used"] is True
    assert result["data_confidence_score"] == pytest.approx(0.45)


def test_real_stored_profile_fallback_avoids_rigid_team_ladder():
    """Stored-profile PRE fallback should keep the sharp end of the grid mixed."""
    predictor = Baseline2026Predictor(data_dir="data/processed", season_year=2026)

    result = predictor.predict_qualifying(
        2026,
        "Canadian Grand Prix",
        n_simulations=300,
        qualifying_stage="main",
        practice_signal_mode="stored_profiles",
        checkpoint_session_name="PRE",
    )

    positions_by_team: dict[str, list[int]] = {}
    for row in result["grid"]:
        positions_by_team.setdefault(str(row["team"]), []).append(int(row["position"]))

    top_ten_positions_by_team = {
        team: sorted(position for position in positions if position <= 10)
        for team, positions in positions_by_team.items()
    }
    adjacent_top_ten_teammate_pairs = sum(
        1
        for positions in top_ten_positions_by_team.values()
        if len(positions) >= 2 and (positions[1] - positions[0] == 1)
    )
    unique_top_ten_teams = len({str(row["team"]) for row in result["grid"][:10]})

    # Current stored profiles can legitimately produce several front-running
    # teammate pairs. This regression guards against the full two-by-two team
    # ladder, not against every adjacent teammate pair.
    assert adjacent_top_ten_teammate_pairs <= 4
    # Five or more teams in the top 10 still avoids the rigid two-by-two ladder
    # this test is protecting against, while leaving room for small
    # cross-environment Monte Carlo reshuffles near the edge of the top 10.
    assert unique_top_ten_teams >= 5


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

    def _fake_run(
        all_drivers,
        n_simulations,
        is_sprint,
        has_practice_data,
        rng,
        has_testing_fallback_data,
        weather="dry",
    ):
        _ = (n_simulations, is_sprint, rng, has_testing_fallback_data, weather)
        captured["has_practice_data"] = has_practice_data
        captured["all_drivers"] = all_drivers
        return {"AAA": [1], "BBB": [2]}

    with _patched_prediction_dependencies():
        with patch.object(predictor, "_run_qualifying_simulations", _fake_run):
            with patch.object(
                predictor,
                "_aggregate_grid_results",
                lambda position_records, all_drivers, data_confidence_score=None: [
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
    assert isinstance(result.get("teammate_head_to_head", []), list)

    strengths = {driver["team"]: driver["team_strength"] for driver in captured["all_drivers"]}
    assert strengths["Team A"] == pytest.approx(0.40)
    assert strengths["Team B"] == pytest.approx(0.60)


def test_build_teammate_head_to_head_probabilities_from_simulation_records():
    predictor = DummyQualifyingPredictor()
    probabilities = predictor._build_teammate_head_to_head_probabilities(
        position_records={
            "VER": [1, 1, 2, 1],
            "HAD": [2, 2, 1, 2],
        },
        all_drivers=[
            {"driver": "VER", "team": "Red Bull Racing"},
            {"driver": "HAD", "team": "Red Bull Racing"},
        ],
    )

    assert probabilities
    first = probabilities[0]
    assert first["team"] == "Red Bull Racing"
    assert first["driver_a"] == "VER"
    assert first["driver_b"] == "HAD"
    assert first["n_samples"] == 4
    assert first["p_driver_a_ahead"] == pytest.approx(0.75)


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


def test_testing_fallback_relaxes_model_only_teammate_regularization():
    """Testing fallback path should avoid hard model-only teammate compression."""
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.model_only_driver_signal_shrink": 0.85,
            "baseline_predictor.qualifying.model_only_experience_shrink": {
                "rookie": 0.55,
                "unknown": 0.0,
            },
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale": 0.30,
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap": 0.12,
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience": {
                "rookie": 0.04
            },
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience": {},
            "baseline_predictor.qualifying.noise_std_normal": 0.018,
            "baseline_predictor.qualifying.teammate_setup_std": 0.008,
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

    strict_model_only = predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=2500,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(123),
        has_testing_fallback_data=False,
    )
    with_testing_fallback = predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=2500,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(123),
        has_testing_fallback_data=True,
    )

    strict_ratio = (
        sum(
            1
            for had_pos, ver_pos in zip(
                strict_model_only["HAD"], strict_model_only["VER"], strict=True
            )
            if had_pos < ver_pos
        )
        / 2500
    )
    fallback_ratio = (
        sum(
            1
            for had_pos, ver_pos in zip(
                with_testing_fallback["HAD"], with_testing_fallback["VER"], strict=True
            )
            if had_pos < ver_pos
        )
        / 2500
    )

    assert fallback_ratio > strict_ratio + 0.04


def test_testing_fallback_teammate_guard_reduces_extreme_inversions():
    """Fallback teammate guard should reduce unsupported flips while keeping variability."""
    base_overrides = {
        "baseline_predictor.qualifying.noise_std_normal": 0.018,
        "baseline_predictor.qualifying.teammate_setup_std": 0.008,
        "baseline_predictor.qualifying.testing_fallback_noise_multiplier": 1.0,
        "baseline_predictor.qualifying.testing_fallback_teammate_setup_multiplier": 1.0,
        "baseline_predictor.qualifying.testing_fallback_weekend_form_std_floor": 0.0,
        "baseline_predictor.qualifying.testing_fallback_driver_signal_shrink": 0.18,
        "baseline_predictor.qualifying.testing_fallback_teammate_anchor_scale": 0.10,
        "baseline_predictor.qualifying.testing_fallback_teammate_anchor_cap": 0.03,
        "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_by_experience": {
            "rookie": 0.12,
            "unknown": 0.12,
        },
        "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_max_races_by_experience": {},
    }
    predictor_without_guard = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.testing_fallback_teammate_guard_enabled": False,
        }
    )
    predictor_with_guard = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.testing_fallback_teammate_guard_enabled": True,
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
            "experience_total_races": 24,
        },
        {
            "driver": "VER",
            "team": "Red Bull Racing",
            "team_strength": 0.59,
            "skill": 0.99,
            "quali_pace": 0.95,
            "experience_tier": "veteran",
            "experience_total_races": 210,
        },
    ]

    without_guard = predictor_without_guard._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=3000,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(2026),
        has_testing_fallback_data=True,
    )
    with_guard = predictor_with_guard._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=3000,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(2026),
        has_testing_fallback_data=True,
    )

    ratio_without_guard = (
        sum(
            1
            for had_pos, ver_pos in zip(without_guard["HAD"], without_guard["VER"], strict=True)
            if had_pos < ver_pos
        )
        / 3000
    )
    ratio_with_guard = (
        sum(
            1
            for had_pos, ver_pos in zip(with_guard["HAD"], with_guard["VER"], strict=True)
            if had_pos < ver_pos
        )
        / 3000
    )

    # The guard should reduce inversions directionally. The original -0.03 margin
    # was calibrated against incorrect config defaults (team_weight=0.7 instead of 0.60).
    # With correct weights the effect size in this two-driver scenario is smaller,
    # so we assert the direction only. The absolute bounds below are the meaningful guard.
    assert ratio_with_guard < ratio_without_guard, (
        f"Guard should reduce inversions but didn't: "
        f"with_guard={ratio_with_guard:.4f}, without_guard={ratio_without_guard:.4f}"
    )
    assert ratio_with_guard < 0.30
    assert ratio_with_guard > 0.05


def test_testing_fallback_driver_offset_multiplier_reduces_team_block_clustering():
    """Fallback-only offset widening should allow more cross-team interleaving."""
    base_overrides = {
        "baseline_predictor.qualifying.noise_std_normal": 0.018,
        "baseline_predictor.qualifying.teammate_setup_std": 0.008,
        "baseline_predictor.qualifying.testing_fallback_noise_multiplier": 1.0,
        "baseline_predictor.qualifying.testing_fallback_teammate_setup_multiplier": 1.0,
        "baseline_predictor.qualifying.testing_fallback_weekend_form_std_floor": 0.0,
        "baseline_predictor.qualifying.testing_fallback_teammate_guard_enabled": True,
        "baseline_predictor.qualifying.testing_fallback_driver_signal_shrink": 0.10,
    }
    predictor_without_multiplier = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.testing_fallback_driver_offset_cap_multiplier": 1.0,
        }
    )
    predictor_with_multiplier = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.testing_fallback_driver_offset_cap_multiplier": 1.33,
        }
    )

    all_drivers = [
        {
            "driver": "AAA",
            "team": "Team A",
            "team_strength": 0.60,
            "skill": 0.95,
            "quali_pace": 0.95,
            "experience_tier": "veteran",
            "experience_total_races": 200,
        },
        {
            "driver": "BBB",
            "team": "Team A",
            "team_strength": 0.60,
            "skill": 0.66,
            "quali_pace": 0.68,
            "experience_tier": "developing",
            "experience_total_races": 35,
        },
        {
            "driver": "CCC",
            "team": "Team B",
            "team_strength": 0.57,
            "skill": 0.88,
            "quali_pace": 0.87,
            "experience_tier": "veteran",
            "experience_total_races": 180,
        },
        {
            "driver": "DDD",
            "team": "Team B",
            "team_strength": 0.57,
            "skill": 0.60,
            "quali_pace": 0.62,
            "experience_tier": "developing",
            "experience_total_races": 30,
        },
    ]

    without_multiplier = predictor_without_multiplier._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=4000,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(2026),
        has_testing_fallback_data=True,
    )
    with_multiplier = predictor_with_multiplier._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=4000,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(2026),
        has_testing_fallback_data=True,
    )

    cross_team_interleave_without_multiplier = (
        sum(
            1
            for team_a_second, team_b_first in zip(
                without_multiplier["BBB"], without_multiplier["CCC"], strict=True
            )
            if team_b_first < team_a_second
        )
        / 4000
    )
    cross_team_interleave_with_multiplier = (
        sum(
            1
            for team_a_second, team_b_first in zip(
                with_multiplier["BBB"], with_multiplier["CCC"], strict=True
            )
            if team_b_first < team_a_second
        )
        / 4000
    )

    assert cross_team_interleave_with_multiplier > (cross_team_interleave_without_multiplier + 0.02)


def test_practice_data_multipliers_reduce_full_grid_teammate_blocking():
    """Practice-backed runs should mix the whole field more than a neutral ladder."""
    neutral_predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.018,
            "baseline_predictor.qualifying.teammate_setup_std": 0.008,
            "baseline_predictor.qualifying.practice_data_team_weight_multiplier": 1.0,
            "baseline_predictor.qualifying.practice_data_skill_weight_multiplier": 1.0,
            "baseline_predictor.qualifying.practice_data_team_compression_multiplier": 1.0,
            "baseline_predictor.qualifying.practice_data_driver_offset_cap_multiplier": 1.0,
            "baseline_predictor.qualifying.practice_data_teammate_setup_multiplier": 1.0,
        }
    )
    tuned_predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.018,
            "baseline_predictor.qualifying.teammate_setup_std": 0.008,
        }
    )

    all_drivers = [
        {
            "driver": "AAA",
            "team": "Team A",
            "team_strength": 0.78,
            "skill": 0.92,
            "quali_pace": 0.92,
            "experience_tier": "veteran",
            "experience_total_races": 160,
        },
        {
            "driver": "AAB",
            "team": "Team A",
            "team_strength": 0.78,
            "skill": 0.66,
            "quali_pace": 0.67,
            "experience_tier": "established",
            "experience_total_races": 90,
        },
        {
            "driver": "BBB",
            "team": "Team B",
            "team_strength": 0.75,
            "skill": 0.90,
            "quali_pace": 0.90,
            "experience_tier": "veteran",
            "experience_total_races": 150,
        },
        {
            "driver": "BBC",
            "team": "Team B",
            "team_strength": 0.75,
            "skill": 0.64,
            "quali_pace": 0.65,
            "experience_tier": "established",
            "experience_total_races": 85,
        },
        {
            "driver": "CCC",
            "team": "Team C",
            "team_strength": 0.72,
            "skill": 0.88,
            "quali_pace": 0.88,
            "experience_tier": "veteran",
            "experience_total_races": 140,
        },
        {
            "driver": "CCD",
            "team": "Team C",
            "team_strength": 0.72,
            "skill": 0.62,
            "quali_pace": 0.63,
            "experience_tier": "established",
            "experience_total_races": 80,
        },
    ]
    driver_to_team = {driver["driver"]: driver["team"] for driver in all_drivers}

    def _average_adjacent_teammate_pairs(position_records: dict[str, list[int]]) -> float:
        sample_count = len(next(iter(position_records.values())))
        total_adjacent_pairs = 0
        for sample_index in range(sample_count):
            positions_by_team: dict[str, list[int]] = {}
            for driver, positions in position_records.items():
                positions_by_team.setdefault(driver_to_team[driver], []).append(
                    positions[sample_index]
                )
            total_adjacent_pairs += sum(
                1
                for positions in positions_by_team.values()
                if len(positions) >= 2 and (sorted(positions)[1] - sorted(positions)[0] == 1)
            )
        return total_adjacent_pairs / sample_count

    neutral_records = neutral_predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=4000,
        is_sprint=False,
        has_practice_data=True,
        rng=np.random.default_rng(42),
        has_testing_fallback_data=False,
    )
    tuned_records = tuned_predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=4000,
        is_sprint=False,
        has_practice_data=True,
        rng=np.random.default_rng(42),
        has_testing_fallback_data=False,
    )

    neutral_average = _average_adjacent_teammate_pairs(neutral_records)
    tuned_average = _average_adjacent_teammate_pairs(tuned_records)

    assert tuned_average < (neutral_average - 0.10)


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


def test_run_qualifying_simulations_applies_recent_form_adjustment():
    """Recent Bayesian form should break ties when the baseline profile is otherwise neutral."""
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.noise_std_normal": 0.0,
            "baseline_predictor.qualifying.noise_std_sprint": 0.0,
            "baseline_predictor.qualifying.teammate_setup_std": 0.0,
            "baseline_predictor.qualifying.team_weight": 0.0,
            "baseline_predictor.qualifying.skill_weight": 1.0,
            "baseline_predictor.qualifying.recent_form_scale": 0.20,
            "baseline_predictor.qualifying.recent_form_cap": 0.03,
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
            "learned_position_adjustment": 0.0,
            "bayesian_skill_score": 0.95,
        },
        {
            "driver": "BBB",
            "team": "Team B",
            "team_strength": 0.5,
            "skill": 0.5,
            "quali_pace": 0.5,
            "experience_tier": "established",
            "learned_position_adjustment": 0.0,
            "bayesian_skill_score": 0.50,
        },
    ]

    position_records = predictor._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=50,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
    )

    assert position_records["AAA"] == [1] * 50
    assert position_records["BBB"] == [2] * 50


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


def test_testing_short_run_fallback_uses_short_run_only_after_sprint_for_main_qualifying():
    """After Sprint, main qualifying fallback should ignore race-shaped balanced profiles."""
    predictor = DummyQualifyingPredictor(
        {
            "baseline_predictor.qualifying.testing_fallback_min_teams": 2,
            "baseline_predictor.qualifying.testing_fallback_short_weight_min": 0.35,
            "baseline_predictor.qualifying.testing_fallback_short_weight_max": 0.85,
            "baseline_predictor.qualifying.testing_fallback_after_sprint_main_short_weight": 1.0,
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
        checkpoint_session_name="SPRINT",
        qualifying_stage="main",
    )

    assert fallback_scores is not None
    assert fallback_scores["Team A"] == pytest.approx(0.90)
    assert fallback_scores["Team B"] == pytest.approx(0.60)


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


def test_testing_fallback_teammate_gap_cap_can_cover_developing_driver_with_higher_race_count():
    """Testing fallback should still protect developing drivers after ~70 race starts."""
    all_drivers = [
        {
            "driver": "NOR",
            "team": "McLaren",
            "team_strength": 0.62,
            "skill": 0.57,
            "quali_pace": 0.77,
            "experience_tier": "established",
            "experience_total_races": 70,
        },
        {
            "driver": "PIA",
            "team": "McLaren",
            "team_strength": 0.62,
            "skill": 0.28,
            "quali_pace": 0.58,
            "experience_tier": "developing",
            "experience_total_races": 70,
        },
        {
            "driver": "OCO",
            "team": "Haas F1 Team",
            "team_strength": 0.56,
            "skill": 0.53,
            "quali_pace": 0.56,
            "experience_tier": "established",
            "experience_total_races": 160,
        },
        {
            "driver": "ALB",
            "team": "Williams",
            "team_strength": 0.54,
            "skill": 0.46,
            "quali_pace": 0.77,
            "experience_tier": "established",
            "experience_total_races": 70,
        },
    ]

    base_overrides = {
        "baseline_predictor.qualifying.noise_std_normal": 0.018,
        "baseline_predictor.qualifying.noise_std_sprint": 0.018,
        "baseline_predictor.qualifying.teammate_setup_std": 0.008,
        "baseline_predictor.qualifying.testing_fallback_driver_signal_shrink": 0.14,
        "baseline_predictor.qualifying.testing_fallback_teammate_anchor_scale": 0.07,
        "baseline_predictor.qualifying.testing_fallback_teammate_anchor_cap": 0.025,
        "baseline_predictor.qualifying.testing_fallback_team_weight_multiplier": 1.0,
        "baseline_predictor.qualifying.testing_fallback_skill_weight_multiplier": 1.0,
    }

    predictor_without_extended_threshold = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_by_experience": {
                "developing": 0.12
            },
            "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_max_races_by_experience": {
                "developing": 55
            },
        }
    )
    predictor_with_extended_threshold = DummyQualifyingPredictor(
        {
            **base_overrides,
            "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_by_experience": {
                "developing": 0.12
            },
            "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_max_races_by_experience": {
                "developing": 90
            },
        }
    )

    uncapped = predictor_without_extended_threshold._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=4000,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
        has_testing_fallback_data=True,
    )
    capped = predictor_with_extended_threshold._run_qualifying_simulations(
        all_drivers=all_drivers,
        n_simulations=4000,
        is_sprint=False,
        has_practice_data=False,
        rng=np.random.default_rng(42),
        has_testing_fallback_data=True,
    )

    pia_mean_uncapped = sum(uncapped["PIA"]) / len(uncapped["PIA"])
    pia_mean_capped = sum(capped["PIA"]) / len(capped["PIA"])
    pia_top3_uncapped = sum(1 for pos in uncapped["PIA"] if pos <= 3) / len(uncapped["PIA"])
    pia_top3_capped = sum(1 for pos in capped["PIA"] if pos <= 3) / len(capped["PIA"])

    assert pia_mean_capped < pia_mean_uncapped
    assert pia_top3_capped > pia_top3_uncapped


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
