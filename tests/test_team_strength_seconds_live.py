"""Live prediction tests for Phase 7 team-strength seconds mapping."""

from __future__ import annotations

import numpy as np

from src.predictors.baseline.qualifying_simulation import (
    QualiSimConfig,
    _score_single_driver_in_simulation,
)
from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap


def _qualifying_sim_config() -> QualiSimConfig:
    """Return a noise-free qualifying config for score-component tests."""
    return QualiSimConfig(
        noise_std=0.0,
        team_weight=0.60,
        skill_weight=0.40,
        team_strength_compression=1.0,
        team_strength_seconds_score_scale=2.0,
        driver_offset_cap=0.18,
        driver_signal_softness=0.20,
        driver_quali_pace_weight=0.70,
        driver_skill_weight=0.30,
        effective_learning_scale=0.0,
        weekend_form_std=0.0,
        teammate_setup_std=0.0,
        recent_form_scale=0.0,
        recent_form_cap=0.0,
        apply_regularization=False,
        apply_recent_form_adjustment=False,
        regularization=None,
    )


def test_qualifying_score_uses_mapped_team_seconds_delta() -> None:
    """Qualifying scoring should consume the Phase 7 seconds delta when present."""
    sim_cfg = _qualifying_sim_config()
    base_driver = {
        "driver": "AAA",
        "team": "Example",
        "team_strength": 0.5,
        "quali_pace": 0.5,
        "skill": 0.5,
    }

    neutral_score = _score_single_driver_in_simulation(
        driver_info=base_driver,
        raw_driver_signal=0.5,
        regularized_signal=0.5,
        gap_cap=None,
        team_driver_signal_means={"Example": 0.5},
        sim_cfg=sim_cfg,
        weekend_form_offset=0.0,
        wet_skill_adjustment=0.0,
        rng=np.random.default_rng(1),
    )
    mapped_score = _score_single_driver_in_simulation(
        driver_info={**base_driver, "team_strength_seconds_delta": 0.5},
        raw_driver_signal=0.5,
        regularized_signal=0.5,
        gap_cap=None,
        team_driver_signal_means={"Example": 0.5},
        sim_cfg=sim_cfg,
        weekend_form_offset=0.0,
        wet_skill_adjustment=0.0,
        rng=np.random.default_rng(1),
    )

    assert mapped_score > neutral_score


def test_qualifying_score_composes_driver_seconds_with_team_seconds() -> None:
    """Qualifying scoring should read the new seconds-native driver residual."""
    sim_cfg = _qualifying_sim_config()
    base_driver = {
        "driver": "AAA",
        "team": "Example",
        "team_strength": 0.5,
        "team_strength_seconds_delta": 0.0,
        "quali_pace": 0.5,
        "skill": 0.5,
    }

    neutral_score = _score_single_driver_in_simulation(
        driver_info=base_driver,
        raw_driver_signal=0.5,
        regularized_signal=0.5,
        gap_cap=None,
        team_driver_signal_means={"Example": 0.5},
        sim_cfg=sim_cfg,
        weekend_form_offset=0.0,
        wet_skill_adjustment=0.0,
        rng=np.random.default_rng(1),
    )
    driver_score = _score_single_driver_in_simulation(
        driver_info={**base_driver, "quali_rating_mu_s": 0.35},
        raw_driver_signal=0.5,
        regularized_signal=0.5,
        gap_cap=None,
        team_driver_signal_means={"Example": 0.5},
        sim_cfg=sim_cfg,
        weekend_form_offset=0.0,
        wet_skill_adjustment=0.0,
        rng=np.random.default_rng(1),
    )

    assert driver_score > neutral_score


def test_race_simulation_uses_mapped_team_seconds_delta() -> None:
    """Race simulation should use mapped seconds instead of unit strength when present."""
    race_params = {
        "fuel": {
            "initial_load_kg": 100.0,
            "effect_per_lap": 0.0,
            "burn_rate_kg_per_lap": 1.5,
        },
        "lap_time": {
            "reference_base": 90.0,
            "team_pace_penalty_range": 1.0,
            "skill_improvement_max": 0.0,
            "bounds": [70.0, 120.0],
        },
        "team_strength_compression": 1.0,
        "race_advantage_lap_impact": 0.0,
        "start_grid_gap_seconds": 0.4,
        "base_chaos": {"dry": 0.0, "wet": 0.0},
        "lap1_chaos": {
            "front_row": 0.0,
            "upper_midfield": 0.0,
            "midfield": 0.0,
            "back_field": 0.0,
        },
        "pit_stops": {"loss_duration": 22.0, "overtake_loss_range": [0.0, 0.0]},
        "sc_probability": 0.0,
        "safety_car_luck_range": 0.0,
        "teammate_variance_std": 0.0,
        "teammate_setup_offset_ratio": 0.0,
        "teammate_variance_lap_ratio": 0.0,
        "track_overtaking": 0.5,
        "overtake_model": {
            "dirty_air_window_s": 1.8,
            "dirty_air_penalty_base": 0.0,
            "dirty_air_penalty_track_scale": 0.0,
            "pass_window_s": 1.2,
            "pass_threshold_base": 0.1,
            "pass_threshold_track_scale": 0.0,
            "pass_probability_base": 0.0,
            "pass_probability_scale": 0.0,
            "pass_time_bonus_range": [0.1, 0.1],
            "pace_diff_scale": 0.5,
            "skill_scale": 0.2,
            "race_adv_scale": 0.2,
            "track_ease_scale": 0.2,
        },
    }
    strategy = {
        "num_stops": 0,
        "pit_laps": [],
        "compound_sequence": ["MEDIUM"],
        "stint_lengths": [3],
    }
    driver_info_map = {
        "AAA": {
            "grid_pos": 1,
            "dnf_probability": 0.0,
            "team_strength": 0.5,
            "team_strength_by_compound": {"MEDIUM": 0.5},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
        "BBB": {
            "grid_pos": 2,
            "dnf_probability": 0.0,
            "team_strength": 0.5,
            "team_strength_seconds_delta": 1.0,
            "team_strength_by_compound": {"MEDIUM": 0.5},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        },
    }

    def _race(delta_s: float) -> dict:
        info = {name: dict(row) for name, row in driver_info_map.items()}
        info["BBB"]["team_strength_seconds_delta"] = delta_s
        # BBB leads, so the queue never clamps his cumulative time to a car ahead and
        # masks the pace difference being measured.
        info["BBB"]["grid_pos"], info["AAA"]["grid_pos"] = 1, 2
        return simulate_race_lap_by_lap(
            driver_info_map=info,
            strategies={"AAA": strategy, "BBB": strategy},
            race_params=race_params,
            race_distance=3,
            weather="dry",
            rng=np.random.default_rng(1),
        )

    # Read the mapping out of lap time, not finishing order: passing is switched off
    # here, and a position change now requires a completed pass, so a quicker car
    # starting second correctly stays second.
    quick = _race(1.0)["total_times"]["BBB"]
    slow = _race(0.0)["total_times"]["BBB"]

    assert quick < slow, (
        f"The mapped team seconds delta did not reach the simulation: BBB ran "
        f"{quick:.3f}s with a 1.0 s/lap delta against {slow:.3f}s without it."
    )


def test_race_simulation_uses_seconds_native_driver_residual() -> None:
    """Race lap times should consume the driver residual in seconds."""
    race_params = {
        "fuel": {
            "initial_load_kg": 100.0,
            "effect_per_lap": 0.0,
            "burn_rate_kg_per_lap": 1.5,
        },
        "lap_time": {
            "reference_base": 90.0,
            "team_pace_penalty_range": 1.0,
            "skill_improvement_max": 0.0,
            "bounds": [70.0, 120.0],
        },
        "team_strength_compression": 1.0,
        "race_advantage_lap_impact": 0.0,
        "start_grid_gap_seconds": 0.4,
        "base_chaos": {"dry": 0.0, "wet": 0.0},
        "lap1_chaos": {
            "front_row": 0.0,
            "upper_midfield": 0.0,
            "midfield": 0.0,
            "back_field": 0.0,
        },
        "pit_stops": {"loss_duration": 22.0, "overtake_loss_range": [0.0, 0.0]},
        "sc_probability": 0.0,
        "safety_car_luck_range": 0.0,
        "teammate_variance_std": 0.0,
        "teammate_setup_offset_ratio": 0.0,
        "teammate_variance_lap_ratio": 0.0,
        "track_overtaking": 0.5,
        "overtake_model": {
            "dirty_air_window_s": 1.8,
            "dirty_air_penalty_base": 0.0,
            "dirty_air_penalty_track_scale": 0.0,
            "pass_window_s": 1.2,
            "pass_threshold_base": 0.1,
            "pass_threshold_track_scale": 0.0,
            "pass_probability_base": 0.0,
            "pass_probability_scale": 0.0,
            "pass_time_bonus_range": [0.1, 0.1],
            "pace_diff_scale": 0.5,
            "skill_scale": 0.2,
            "race_adv_scale": 0.2,
            "track_ease_scale": 0.2,
        },
    }
    strategy = {
        "num_stops": 0,
        "pit_laps": [],
        "compound_sequence": ["MEDIUM"],
        "stint_lengths": [3],
    }
    shared = {
        "dnf_probability": 0.0,
        "team_strength": 0.5,
        "team_strength_by_compound": {"MEDIUM": 0.5},
        "tire_deg_by_compound": {"MEDIUM": 0.0},
        "skill": 0.5,
        "race_advantage": 0.0,
        "overtaking_skill": 0.5,
    }

    def _race(mu_s: float) -> dict:
        return simulate_race_lap_by_lap(
            # BBB leads: a car held up in traffic has its cumulative time clamped to the
            # car ahead, which would mask the very pace difference under test.
            driver_info_map={
                "BBB": {**shared, "grid_pos": 1, "race_rating_mu_s": mu_s},
                "AAA": {**shared, "grid_pos": 2},
            },
            strategies={"AAA": strategy, "BBB": strategy},
            race_params=race_params,
            race_distance=3,
            weather="dry",
            rng=np.random.default_rng(1),
        )

    # Finishing order cannot show this any more: passing is switched off in this config,
    # and a position change now requires a completed pass, so a quicker car starting
    # second correctly finishes second. The mapping is observable in lap time instead.
    quick = _race(1.0)["total_times"]["BBB"]
    slow = _race(0.0)["total_times"]["BBB"]

    assert quick < slow, (
        f"A 1.0 s/lap seconds delta did not reach the simulation: BBB ran {quick:.3f}s "
        f"with the delta against {slow:.3f}s without it."
    )
