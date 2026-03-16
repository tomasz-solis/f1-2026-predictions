"""Race scoring/parameter helpers for Baseline2026Predictor."""

from __future__ import annotations

import numpy as np

from src.utils import config_loader


class BaselineRaceParamsMixin:
    """Race scoring and parameter-loading methods for Baseline2026Predictor."""

    def _calculate_driver_race_score(
        self,
        info: dict,
        track_overtaking: float,
        weather: str,
        safety_car: bool,
        params: dict,
        rng: np.random.Generator,
    ) -> tuple[float, bool]:
        """Calculate single driver's race score for one simulation."""
        grid_divisor = params.get("grid_divisor", 21)
        front_threshold = params.get("position_scaling_front_threshold", 3)
        front_scale = params.get("position_scaling_front_scale", 0.1)
        upper_threshold = params.get("position_scaling_upper_threshold", 7)
        upper_scale = params.get("position_scaling_upper_scale", 0.3)
        mid_threshold = params.get("position_scaling_mid_threshold", 12)
        mid_scale = params.get("position_scaling_mid_scale", 0.6)
        back_scale = params.get("position_scaling_back_scale", 1.0)

        grid_weight = params["grid_weight_min"] + (
            track_overtaking * params["grid_weight_multiplier"]
        )
        grid_advantage = 1.0 - ((info["grid_pos"] - 1) / grid_divisor)

        if info["grid_pos"] <= front_threshold:
            position_scaling = front_scale
        elif info["grid_pos"] <= upper_threshold:
            position_scaling = upper_scale
        elif info["grid_pos"] <= mid_threshold:
            position_scaling = mid_scale
        else:
            position_scaling = back_scale

        race_pace_boost = (
            info["race_advantage"] * params["race_advantage_multiplier"] * position_scaling
        )

        if (
            info["grid_pos"] > params["overtaking_grid_threshold"]
            and track_overtaking < params["overtaking_track_threshold"]
        ):
            overtaking_boost = (
                (info["overtaking_skill"] - 0.5)
                * params["overtaking_skill_multiplier"]
                * position_scaling
            )
        else:
            overtaking_boost = 0.0

        if info["grid_pos"] <= 3:
            lap1_chaos = rng.normal(0, params["lap1_front_row_chaos"])
        elif info["grid_pos"] <= 10:
            lap1_chaos = rng.normal(0, params["lap1_upper_midfield_chaos"])
        elif info["grid_pos"] <= 15:
            lap1_chaos = rng.normal(0, params["lap1_midfield_chaos"])
        else:
            lap1_chaos = rng.normal(0, params["lap1_back_field_chaos"])

        strategy_std = params["strategy_variance_base"] * (
            1.0 - track_overtaking * params["strategy_track_modifier"]
        )
        strategy_factor = rng.uniform(-strategy_std, strategy_std)

        sc_luck = (
            rng.uniform(-params["safety_car_luck_range"], params["safety_car_luck_range"])
            if safety_car
            else 0.0
        )

        pace_weight = params["pace_weight_base"] - (
            track_overtaking * params["pace_weight_track_modifier"]
        )
        driver_weight = 0.20
        total_weight = grid_weight + pace_weight + driver_weight

        normalized_grid = grid_weight / total_weight
        normalized_pace = pace_weight / total_weight
        normalized_skill = driver_weight / total_weight

        base_score = (
            (grid_advantage * normalized_grid)
            + (info["team_strength"] * normalized_pace)
            + (info["skill"] * normalized_skill)
        )

        teammate_variance = rng.normal(0, params["teammate_variance_std"])

        if weather == "rain":
            base_chaos_std = params["base_chaos"]["wet"]
        elif weather == "mixed":
            mixed_blend = float(np.clip(params.get("mixed_weather_chaos_blend", 0.55), 0.0, 1.0))
            dry_std = params["base_chaos"]["dry"]
            wet_std = params["base_chaos"]["wet"]
            base_chaos_std = dry_std + ((wet_std - dry_std) * mixed_blend)
        else:
            base_chaos_std = params["base_chaos"]["dry"]

        dnf_occurred = rng.random() < info["dnf_probability"]

        if dnf_occurred:
            score = -10.0 + rng.uniform(-1.0, 0.0)
        else:
            score = (
                base_score
                + race_pace_boost
                + overtaking_boost
                + rng.normal(0, base_chaos_std)
                + lap1_chaos
                + strategy_factor
                + sc_luck
                + teammate_variance
            )

        return float(score), bool(dnf_occurred)

    def _load_race_params(self) -> dict:
        """Load all race parameters from config once."""
        cfg = getattr(self, "config", config_loader)
        return {
            "base_chaos_dry": cfg.get("baseline_predictor.race.base_chaos.dry", 0.35),
            "base_chaos_wet": cfg.get("baseline_predictor.race.base_chaos.wet", 0.45),
            "mixed_weather_chaos_blend": cfg.get(
                "baseline_predictor.race.base_chaos.mixed_blend", 0.55
            ),
            "track_chaos_multiplier": cfg.get(
                "baseline_predictor.race.track_chaos_multiplier", 0.4
            ),
            "sc_base_prob_dry": cfg.get("baseline_predictor.race.sc_base_probability.dry", 0.45),
            "sc_base_prob_wet": cfg.get("baseline_predictor.race.sc_base_probability.wet", 0.70),
            "sc_track_modifier": cfg.get("baseline_predictor.race.sc_track_modifier", 0.25),
            "grid_weight_min": cfg.get("baseline_predictor.race.grid_weight_min", 0.15),
            "grid_weight_multiplier": cfg.get(
                "baseline_predictor.race.grid_weight_multiplier", 0.35
            ),
            "race_advantage_multiplier": cfg.get(
                "baseline_predictor.race.race_advantage_multiplier", 0.5
            ),
            "overtaking_skill_multiplier": cfg.get(
                "baseline_predictor.race.overtaking_skill_multiplier", 0.25
            ),
            "overtaking_grid_threshold": cfg.get(
                "baseline_predictor.race.overtaking_grid_threshold", 5
            ),
            "overtaking_track_threshold": cfg.get(
                "baseline_predictor.race.overtaking_track_threshold", 0.5
            ),
            "lap1_front_row_chaos": cfg.get("baseline_predictor.race.lap1_chaos.front_row", 0.15),
            "lap1_upper_midfield_chaos": cfg.get(
                "baseline_predictor.race.lap1_chaos.upper_midfield", 0.32
            ),
            "lap1_midfield_chaos": cfg.get("baseline_predictor.race.lap1_chaos.midfield", 0.38),
            "lap1_back_field_chaos": cfg.get("baseline_predictor.race.lap1_chaos.back_field", 0.28),
            "strategy_variance_base": cfg.get(
                "baseline_predictor.race.strategy_variance_base", 0.30
            ),
            "strategy_track_modifier": cfg.get(
                "baseline_predictor.race.strategy_track_modifier", 0.5
            ),
            "safety_car_luck_range": cfg.get("baseline_predictor.race.safety_car_luck_range", 0.25),
            "pace_weight_base": cfg.get("baseline_predictor.race.pace_weight_base", 0.40),
            "pace_weight_track_modifier": cfg.get(
                "baseline_predictor.race.pace_weight_track_modifier", 0.10
            ),
            "teammate_variance_std": cfg.get("baseline_predictor.race.teammate_variance_std", 0.15),
            "teammate_setup_offset_ratio": cfg.get(
                "baseline_predictor.race.teammate_setup_offset_ratio",
                0.30,
            ),
            "teammate_variance_lap_ratio": cfg.get(
                "baseline_predictor.race.teammate_variance_lap_ratio",
                0.45,
            ),
            "grid_divisor": cfg.get("baseline_predictor.race.grid_divisor", 21),
            "position_scaling_front_threshold": cfg.get(
                "baseline_predictor.race.position_scaling.front_threshold", 3
            ),
            "position_scaling_front_scale": cfg.get(
                "baseline_predictor.race.position_scaling.front_scale", 0.1
            ),
            "position_scaling_upper_threshold": cfg.get(
                "baseline_predictor.race.position_scaling.upper_threshold", 7
            ),
            "position_scaling_upper_scale": cfg.get(
                "baseline_predictor.race.position_scaling.upper_scale", 0.3
            ),
            "position_scaling_mid_threshold": cfg.get(
                "baseline_predictor.race.position_scaling.mid_threshold", 12
            ),
            "position_scaling_mid_scale": cfg.get(
                "baseline_predictor.race.position_scaling.mid_scale", 0.6
            ),
            "position_scaling_back_scale": cfg.get(
                "baseline_predictor.race.position_scaling.back_scale", 1.0
            ),
        }
