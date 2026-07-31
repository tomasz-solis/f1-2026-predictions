from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import src.predictors.baseline.race.prediction_mixin as race_prediction_module
import src.predictors.baseline.race.race_simulation as race_simulation_module
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.race.prediction_mixin import BaselineRacePredictionMixin
from src.predictors.baseline.race.race_simulation import (
    _consume_marginal_grid_randomness,
    _sample_probabilistic_grid_positions,
)
from src.utils.grid_scenarios import (
    build_grid_scenario_schedule,
    build_grid_scenarios,
    grid_scenario_digest,
    validate_grid_scenarios,
)


class _VariantConfig:
    def __init__(self, variant: str) -> None:
        self.variant = variant

    def get(self, key: str, default: Any = None) -> Any:
        if key == "baseline_predictor.model_variant":
            return self.variant
        return default


def test_build_grid_scenarios_uses_shared_simulation_index():
    scenarios = build_grid_scenarios(
        position_records={
            "A": [1, 3, 2],
            "B": [2, 1, 3],
            "C": [3, 2, 1],
        },
        expected_drivers=["A", "B", "C"],
    )

    assert scenarios == [
        ["A", "B", "C"],
        ["B", "C", "A"],
        ["C", "A", "B"],
    ]
    assert grid_scenario_digest(scenarios) == grid_scenario_digest([list(row) for row in scenarios])


def test_validate_grid_scenarios_rejects_incomplete_or_duplicate_permutations():
    for invalid in (["A", "A", "C"], ["A", "B"], ["A", "B", "D"]):
        try:
            validate_grid_scenarios([invalid], expected_drivers=["A", "B", "C"])
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid scenario to fail validation: {invalid}")


def test_grid_scenario_schedule_is_deterministic_and_balanced():
    first = build_grid_scenario_schedule(scenario_count=3, simulation_count=11, base_seed=42)
    second = build_grid_scenario_schedule(scenario_count=3, simulation_count=11, base_seed=42)

    assert first == second
    usage = Counter(first)
    assert set(usage) == {0, 1, 2}
    assert max(usage.values()) - min(usage.values()) <= 1


class _QualifyingScenarioPredictor(BaselineQualifyingMixin):
    seed = 19
    config = _VariantConfig("r1_joint_grid")

    def _prepare_qualifying_prediction_inputs(self, **_kwargs):
        return {
            "all_drivers": [
                {"driver": "A", "team": "Team A"},
                {"driver": "B", "team": "Team B"},
                {"driver": "C", "team": "Team C"},
            ],
            "testing_fallback_used": False,
            "normalized_practice_signal_mode": "auto",
            "practice_like_stored_profiles": False,
            "data_confidence_score": 0.5,
            "effective_fp_blend_weight": 0.0,
            "data_source": "test",
            "data_source_mode": "model_only",
            "teams_with_short_profile": 0,
            "checkpoint_label": "PRE",
            "has_practice_like_data": False,
        }

    def _load_qualifying_residual_model(self):
        return None

    def _run_qualifying_simulations(self, *_args, **_kwargs):
        return {
            "A": [1, 3, 2],
            "B": [2, 1, 3],
            "C": [3, 2, 1],
        }

    def _aggregate_grid_results_with_compat(self, *_args, **_kwargs):
        return [
            {"driver": "A", "team": "Team A", "position": 1},
            {"driver": "B", "team": "Team B", "position": 2},
            {"driver": "C", "team": "Team C", "position": 3},
        ]


def test_qualifying_grid_scenarios_are_opt_in_and_raw_scenarios_stay_internal_by_default():
    predictor = _QualifyingScenarioPredictor()

    champion = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=3)
    challenger = predictor.predict_qualifying(
        2026,
        "Australian Grand Prix",
        n_simulations=3,
        include_grid_scenarios=True,
    )

    assert "grid_scenarios" not in champion
    assert champion["grid_source_detail"] == "predicted_marginal_fallback"
    assert champion["grid_scenario_count"] == 0
    assert challenger["grid_source_detail"] == "predicted_joint"
    assert challenger["grid_scenario_count"] == 3
    assert challenger["grid_scenario_digest"] == grid_scenario_digest(challenger["grid_scenarios"])
    expected_drivers = {row["driver"] for row in challenger["grid"]}
    assert all(set(scenario) == expected_drivers for scenario in challenger["grid_scenarios"])


class _JointGridRacePredictor(BaselineRacePredictionMixin):
    seed = 37
    config = _VariantConfig("r1_joint_grid")

    def _load_race_params(self) -> dict:
        return {}

    def _prepare_driver_info_with_compounds(
        self, qualifying_grid: list[dict], race_name: str | None
    ) -> tuple[dict, int]:
        _ = race_name
        return (
            {
                row["driver"]: {
                    "driver": row["driver"],
                    "team": row["team"],
                    "grid_pos": row["position"],
                    "team_strength": 0.5,
                    "team_strength_by_compound": {
                        "SOFT": 0.5,
                        "MEDIUM": 0.5,
                        "HARD": 0.5,
                    },
                    "tire_deg_by_compound": {
                        "SOFT": 0.1,
                        "MEDIUM": 0.1,
                        "HARD": 0.1,
                    },
                    "skill": 0.5,
                    "race_advantage": 0.0,
                    "overtaking_skill": 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
                for row in qualifying_grid
            },
            0,
        )


def test_race_joint_scenarios_bypass_marginal_sampling_and_are_used_evenly():
    predictor = _JointGridRacePredictor()
    qualifying_grid = [
        {
            "driver": driver,
            "team": f"Team {driver}",
            "position": position,
            "median_position": position,
            "p5": 1,
            "p95": 3,
            "confidence": 45.0,
        }
        for position, driver in enumerate(("A", "B", "C"), start=1)
    ]
    scenarios = [["A", "B", "C"], ["C", "B", "A"]]
    seen_starts: list[tuple[str, ...]] = []

    def _simulate(**kwargs):
        ordered = tuple(
            driver
            for driver, _info in sorted(
                kwargs["driver_info_map"].items(),
                key=lambda item: (item[1]["grid_pos"], item[0]),
            )
        )
        seen_starts.append(ordered)
        return {
            "finish_order": list(ordered),
            "dnf_drivers": [],
            "strategies_used": kwargs["strategies"],
        }

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(race_prediction_module, "load_track_specific_params", lambda *_a, **_k: {})
        )
        stack.enter_context(
            patch.object(race_prediction_module, "get_tire_stress_score", lambda *_a, **_k: 3.0)
        )
        stack.enter_context(
            patch.object(
                race_prediction_module,
                "get_available_compounds",
                lambda *_a, **_k: ["SOFT", "MEDIUM", "HARD"],
            )
        )
        stack.enter_context(
            patch.object(
                race_prediction_module,
                "resolve_track_temperature_c",
                lambda *_a, **_k: None,
            )
        )
        stack.enter_context(
            patch.object(
                race_prediction_module,
                "resolve_track_temperature_profile",
                lambda *_a, **_k: None,
            )
        )
        stack.enter_context(
            patch.object(
                race_prediction_module,
                "resolve_non_competitive_weather_features",
                lambda *_a, **_k: None,
            )
        )
        stack.enter_context(
            patch.object(
                race_prediction_module,
                "resolve_race_distance_laps",
                lambda *_a, **_k: 10,
            )
        )
        stack.enter_context(
            patch.object(
                race_prediction_module,
                "generate_pit_strategy",
                lambda **_kwargs: {
                    "num_stops": 0,
                    "pit_laps": [],
                    "compound_sequence": ["SOFT"],
                    "stint_lengths": [10],
                },
            )
        )
        stack.enter_context(
            patch.object(race_prediction_module, "simulate_race_lap_by_lap", _simulate)
        )
        stack.enter_context(
            patch.object(
                race_simulation_module,
                "_sample_probabilistic_grid_positions",
                side_effect=AssertionError("marginal sampling must be bypassed"),
            )
        )

        result = predictor.predict_race(
            qualifying_grid=qualifying_grid,
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=7,
            is_sprint=True,
            grid_scenarios=scenarios,
        )

    usage = Counter(seen_starts)
    assert set(usage) == {("A", "B", "C"), ("C", "B", "A")}
    assert max(usage.values()) - min(usage.values()) <= 1
    assert result["grid_source_detail"] == "predicted_joint"
    assert result["grid_uncertainty_mode"] == "joint_scenarios"
    assert result["grid_scenario_count"] == 2
    assert result["grid_scenario_digest"] == grid_scenario_digest(scenarios)
    assert result["grid_anchor_diagnostics"]["requested"] == "champion"
    assert result["grid_anchor_diagnostics"]["source_detail"] == "predicted_joint"
    assert "grid_scenarios" not in result


def test_champion_cannot_activate_joint_grid_interfaces() -> None:
    qualifying_predictor = _QualifyingScenarioPredictor()
    qualifying_predictor.config = _VariantConfig("champion")
    with pytest.raises(ValueError, match="explicit model variant containing R1"):
        qualifying_predictor.predict_qualifying(
            2026,
            "Australian Grand Prix",
            n_simulations=3,
            include_grid_scenarios=True,
        )

    race_predictor = _JointGridRacePredictor()
    race_predictor.config = _VariantConfig("champion")
    with pytest.raises(ValueError, match="explicit model variant containing R1"):
        race_predictor.predict_race(
            qualifying_grid=[
                {"driver": "A", "team": "Team A", "position": 1},
                {"driver": "B", "team": "Team B", "position": 2},
            ],
            n_simulations=1,
            grid_scenarios=[["A", "B"]],
        )


def test_joint_and_marginal_paths_preserve_common_random_numbers() -> None:
    grid = [
        {"driver": "A", "team": "Team A", "position": 1},
        {"driver": "B", "team": "Team B", "position": 2},
        {"driver": "C", "team": "Team C", "position": 3},
    ]
    profile = {
        "A": {"center": 1.2, "std": 0.4},
        "B": {"center": 2.0, "std": 0.0},
        "C": {"center": 2.8, "std": 0.6},
    }
    joint_rng = np.random.default_rng(91)
    marginal_rng = np.random.default_rng(91)

    _consume_marginal_grid_randomness(
        validated_grid=grid,
        grid_uncertainty_profile=profile,
        rng=joint_rng,
    )
    _sample_probabilistic_grid_positions(
        validated_grid=grid,
        grid_uncertainty_profile=profile,
        rng=marginal_rng,
    )

    assert joint_rng.random(8).tolist() == marginal_rng.random(8).tolist()
