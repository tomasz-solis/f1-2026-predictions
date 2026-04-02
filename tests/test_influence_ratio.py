"""Regression tests for the race team-to-driver influence balance."""

from __future__ import annotations

import copy

import numpy as np

from src.data.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_race_distance_laps,
)
from src.predictors.baseline_2026 import Baseline2026Predictor
from src.simulation.pit_strategy import generate_pit_strategy
from src.utils.config_loader import Config
from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap

_RACE_NAME = "Australian Grand Prix"
_TEAM_NAME = "McLaren"


def _driver_profile(skill_score: float) -> dict[str, object]:
    """Build a minimal race-ready driver payload for an intra-team matchup."""
    return {
        "pace": {"quali_pace": 0.60, "race_pace": 0.60},
        "racecraft": {"skill_score": skill_score, "overtaking_skill": 0.55},
        "dnf_risk": {"dnf_rate": 0.02},
        "experience": {"tier": "established", "years_of_experience": 5},
    }


def _build_race_context(predictor: Baseline2026Predictor) -> dict[str, object]:
    """Prepare stable race inputs for the raw lap-by-lap simulator check."""
    race_params = {
        **predictor._load_race_params(),
        **load_track_specific_params(_RACE_NAME, year=2026),
    }
    race_params["track_name"] = _RACE_NAME
    race_params["fuel"] = {
        "initial_load_kg": predictor.config.get(
            "baseline_predictor.race.fuel.initial_load_kg", 110.0
        ),
        "effect_per_lap": predictor.config.get(
            "baseline_predictor.race.fuel.effect_per_lap", 0.035
        ),
        "burn_rate_kg_per_lap": predictor.config.get(
            "baseline_predictor.race.fuel.burn_rate_kg_per_lap", 1.5
        ),
    }
    race_params["lap_time"] = {
        "reference_base": predictor.config.get(
            "baseline_predictor.race.lap_time.reference_base", 90.0
        ),
        "team_pace_penalty_range": predictor.config.get(
            "baseline_predictor.race.lap_time.team_pace_penalty_range",
            5.0,
        ),
        "skill_improvement_max": predictor.config.get(
            "baseline_predictor.race.lap_time.skill_improvement_max",
            0.75,
        ),
        "bounds": predictor.config.get("baseline_predictor.race.lap_time.bounds", [70.0, 120.0]),
        "elite_skill_threshold": predictor.config.get(
            "baseline_predictor.race.lap_time.elite_skill_threshold",
            0.88,
        ),
        "elite_skill_lap_bonus_max": predictor.config.get(
            "baseline_predictor.race.lap_time.elite_skill_lap_bonus_max",
            0.09,
        ),
        "elite_skill_exponent": predictor.config.get(
            "baseline_predictor.race.lap_time.elite_skill_exponent",
            1.3,
        ),
    }
    race_params["team_strength_compression"] = predictor.config.get(
        "baseline_predictor.race.lap_time.team_strength_compression",
        0.35,
    )
    race_params["start_grid_gap_seconds"] = predictor.config.get(
        "baseline_predictor.race.start_grid_gap_seconds",
        0.32,
    )
    race_params["race_advantage_lap_impact"] = predictor.config.get(
        "baseline_predictor.race.race_advantage_lap_impact",
        0.35,
    )
    race_params["safety_car_trigger_lap"] = predictor.config.get(
        "baseline_predictor.race.safety_car_trigger_lap",
        10,
    )
    race_params["overtake_model"] = {
        "dirty_air_window_s": predictor.config.get(
            "baseline_predictor.race.overtake_model.dirty_air_window_s",
            1.8,
        ),
        "pace_weight": predictor.config.get(
            "baseline_predictor.race.overtake_model.pace_weight",
            0.55,
        ),
        "racecraft_weight": predictor.config.get(
            "baseline_predictor.race.overtake_model.racecraft_weight",
            0.25,
        ),
        "track_factor": predictor.config.get(
            "baseline_predictor.race.overtake_model.track_factor",
            0.35,
        ),
        "pass_chance_base": predictor.config.get(
            "baseline_predictor.race.overtake_model.pass_chance_base",
            0.30,
        ),
    }
    return {
        "available_compounds": get_available_compounds(_RACE_NAME, weather="dry"),
        "race_distance": resolve_race_distance_laps(
            year=2026,
            race_name=_RACE_NAME,
            is_sprint=False,
        ),
        "race_params": race_params,
        "tire_stress_score": get_tire_stress_score(_RACE_NAME, year=2026),
    }


def _intra_team_win_rate(
    skill_a: float,
    skill_b: float,
    *,
    n_races: int = 80,
    base_seed: int = 100,
) -> float:
    """Return the share of raw race sims where driver A beats driver B in the same car."""
    predictor = Baseline2026Predictor(seed=base_seed)
    predictor.drivers["DRV_A"] = _driver_profile(skill_a)
    predictor.drivers["DRV_B"] = _driver_profile(skill_b)
    context = _build_race_context(predictor)

    wins = 0
    for offset in range(n_races):
        a_starts_ahead = offset % 2 == 0
        grid = [
            {
                "driver": "DRV_A",
                "team": _TEAM_NAME,
                "position": 1 if a_starts_ahead else 2,
            },
            {
                "driver": "DRV_B",
                "team": _TEAM_NAME,
                "position": 2 if a_starts_ahead else 1,
            },
        ]
        driver_info_map, _ = predictor._prepare_driver_info_with_compounds(grid, _RACE_NAME)
        rng = np.random.default_rng(base_seed + offset)
        race_params = copy.deepcopy(context["race_params"])
        strategies = {}
        for driver, info in driver_info_map.items():
            strategies[driver] = generate_pit_strategy(
                race_distance=int(context["race_distance"]),
                tire_stress_score=float(context["tire_stress_score"]),
                available_compounds=list(context["available_compounds"]),
                rng=rng,
                driver_risk_profile=float(info.get("dnf_probability", 0.0)),
                enforce_two_compound_rule=True,
                track_overtaking=float(race_params.get("track_overtaking", 0.5)),
                grid_position=int(info.get("grid_pos", 99)),
                strategy_signal=float(info.get("race_advantage", 0.0)),
            )
        result = simulate_race_lap_by_lap(
            driver_info_map=driver_info_map,
            strategies=strategies,
            race_params=race_params,
            race_distance=int(context["race_distance"]),
            weather="dry",
            rng=rng,
        )
        if result["finish_order"][0] == "DRV_A":
            wins += 1

    return wins / n_races


def test_live_lap_time_ratio_keeps_driver_signal_visible():
    """Race pace defaults should leave enough room for driver skill to matter."""
    config = Config()
    team_pace_penalty_range = float(
        config.get("baseline_predictor.race.lap_time.team_pace_penalty_range", 5.0)
    )
    skill_improvement_max = float(
        config.get("baseline_predictor.race.lap_time.skill_improvement_max", 0.75)
    )
    team_strength_compression = float(
        config.get("baseline_predictor.race.lap_time.team_strength_compression", 0.40)
    )

    team_range = 2.0 * 0.5 * team_strength_compression * team_pace_penalty_range
    driver_range = skill_improvement_max
    influence_ratio = team_range / driver_range

    assert influence_ratio <= 2.40, (
        f"Team:driver lap-time ratio is {influence_ratio:.2f}:1. "
        "Target is <=2.4:1 so the driver signal remains visible."
    )


def test_higher_skill_driver_wins_majority_of_intra_team_battles():
    """A clear skill edge should show up in the raw lap-by-lap race simulation."""
    win_rate = _intra_team_win_rate(skill_a=0.70, skill_b=0.50)

    assert win_rate >= 0.60, (
        f"Higher-skill driver won only {win_rate:.1%} of same-car matchups. "
        "Driver influence is still too muted."
    )


def test_equal_skill_teammates_split_roughly_evenly():
    """Equal-skill teammates should stay close to a 50/50 split over many runs."""
    win_rate = _intra_team_win_rate(skill_a=0.50, skill_b=0.50)

    assert 0.40 <= win_rate <= 0.60, (
        f"Equal-skill teammates split {win_rate:.1%}. "
        "The matchup is leaning too hard toward one side."
    )
