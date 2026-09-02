"""Race scoring/parameter helpers for Baseline2026Predictor."""

from __future__ import annotations

from src.utils import config_loader


class BaselineRaceParamsMixin:
    """Race scoring and parameter-loading methods for Baseline2026Predictor."""

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
