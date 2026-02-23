"""Race prediction method for Baseline2026Predictor."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.types.prediction_types import PitStrategy, QualifyingGridEntry
from src.utils import config_loader
from src.utils.grid_validation import validate_qualifying_grid
from src.utils.lap_by_lap_simulator import (
    aggregate_simulation_results,
    simulate_race_lap_by_lap,
)
from src.utils.pit_strategy import generate_pit_strategy
from src.utils.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_race_distance_laps,
)
from src.utils.validation_helpers import validate_enum, validate_positive_int

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePredictionMixin:
    """Race prediction method implementation for Baseline2026Predictor."""

    @staticmethod
    def _enforce_non_increasing(values: list[float]) -> list[float]:
        """Apply isotonic-style smoothing so sequence is non-increasing."""
        if not values:
            return values

        # Pool adjacent violators for monotone decreasing fit.
        blocks: list[list[float]] = []  # [start, end, avg, count]
        for idx, value in enumerate(values):
            blocks.append([float(idx), float(idx), float(value), 1.0])
            while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
                right = blocks.pop()
                left = blocks.pop()
                merged_count = left[3] + right[3]
                merged_avg = ((left[2] * left[3]) + (right[2] * right[3])) / merged_count
                blocks.append([left[0], right[1], merged_avg, merged_count])

        smoothed = [0.0 for _ in values]
        for start, end, avg, _ in blocks:
            for idx in range(int(start), int(end) + 1):
                smoothed[idx] = float(avg)
        return smoothed

    def _get_learned_position_adjustment(
        self,
        *,
        team: str,
        driver: str,
        teammates: list[str],
        session: str = "race",
    ) -> float:
        """Return learned position adjustment from systematic calibration state."""
        calibration_system = getattr(self, "calibration_system", None)
        if calibration_system is None:
            return 0.0

        getter = getattr(calibration_system, "get_combined_position_adjustment", None)
        if not callable(getter):
            return 0.0

        cfg = getattr(self, "config", config_loader)
        min_samples = int(cfg.get("baseline_predictor.learning.min_samples", 1))
        driver_error_scale = float(cfg.get("baseline_predictor.learning.driver_error_scale", 0.18))
        teammate_gap_scale = float(cfg.get("baseline_predictor.learning.teammate_gap_scale", 0.10))
        max_adjustment = float(cfg.get("baseline_predictor.learning.max_adjustment", 2.5))

        try:
            return float(
                getter(
                    team=team,
                    driver=driver,
                    teammates=teammates,
                    session=session,
                    min_samples=max(1, min_samples),
                    driver_error_scale=driver_error_scale,
                    teammate_gap_scale=teammate_gap_scale,
                    max_adjustment=max_adjustment,
                )
            )
        except Exception as exc:
            logger.debug(f"Could not load learned race adjustment for {driver}: {exc}")
            return 0.0

    def predict_race(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
        is_sprint: bool = False,
        race_compound: str = "MEDIUM",
    ) -> dict[str, Any]:
        """Predict race result using lap-by-lap Monte Carlo simulation with tire deg and pit stops."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        cfg = getattr(self, "config", config_loader)
        validated_grid = validate_qualifying_grid(qualifying_grid)

        # Load track-specific parameters (pit loss, safety car prob, overtaking)
        track_params = load_track_specific_params(race_name)

        # Load base race parameters from config
        base_params = self._load_race_params()

        # Merge track-specific overrides into base params
        race_params = {**base_params, **track_params}
        race_params["track_name"] = race_name

        # Load additional params for lap-by-lap simulation
        race_params["fuel"] = {
            "initial_load_kg": cfg.get("baseline_predictor.race.fuel.initial_load_kg", 110.0),
            "effect_per_lap": cfg.get("baseline_predictor.race.fuel.effect_per_lap", 0.035),
            "burn_rate_kg_per_lap": cfg.get(
                "baseline_predictor.race.fuel.burn_rate_kg_per_lap", 1.5
            ),
        }

        race_params["lap_time"] = {
            "reference_base": cfg.get("baseline_predictor.race.lap_time.reference_base", 90.0),
            "team_pace_penalty_range": cfg.get(
                "baseline_predictor.race.lap_time.team_pace_penalty_range", 5.0
            ),
            "skill_improvement_max": cfg.get(
                "baseline_predictor.race.lap_time.skill_improvement_max", 0.5
            ),
            "bounds": cfg.get("baseline_predictor.race.lap_time.bounds", [70.0, 120.0]),
            "elite_skill_threshold": cfg.get(
                "baseline_predictor.race.lap_time.elite_skill_threshold", 0.88
            ),
            "elite_skill_lap_bonus_max": cfg.get(
                "baseline_predictor.race.lap_time.elite_skill_lap_bonus_max", 0.09
            ),
            "elite_skill_exponent": cfg.get(
                "baseline_predictor.race.lap_time.elite_skill_exponent", 1.3
            ),
        }
        race_params["team_strength_compression"] = cfg.get(
            "baseline_predictor.race.lap_time.team_strength_compression", 0.35
        )
        race_params["start_grid_gap_seconds"] = cfg.get(
            "baseline_predictor.race.start_grid_gap_seconds", 0.32
        )
        race_params["race_advantage_lap_impact"] = cfg.get(
            "baseline_predictor.race.race_advantage_lap_impact", 0.35
        )
        race_params["safety_car_trigger_lap"] = cfg.get(
            "baseline_predictor.race.safety_car_trigger_lap", 10
        )
        race_params["overtake_model"] = {
            "dirty_air_window_s": cfg.get(
                "baseline_predictor.race.overtake_model.dirty_air_window_s", 1.8
            ),
            "dirty_air_penalty_base": cfg.get(
                "baseline_predictor.race.overtake_model.dirty_air_penalty_base", 0.05
            ),
            "dirty_air_penalty_track_scale": cfg.get(
                "baseline_predictor.race.overtake_model.dirty_air_penalty_track_scale",
                0.12,
            ),
            "pass_window_s": cfg.get("baseline_predictor.race.overtake_model.pass_window_s", 1.2),
            "pass_threshold_base": cfg.get(
                "baseline_predictor.race.overtake_model.pass_threshold_base", 0.06
            ),
            "pass_threshold_track_scale": cfg.get(
                "baseline_predictor.race.overtake_model.pass_threshold_track_scale",
                0.16,
            ),
            "pass_probability_base": cfg.get(
                "baseline_predictor.race.overtake_model.pass_probability_base", 0.30
            ),
            "pass_probability_scale": cfg.get(
                "baseline_predictor.race.overtake_model.pass_probability_scale", 0.45
            ),
            "pass_time_bonus_range": cfg.get(
                "baseline_predictor.race.overtake_model.pass_time_bonus_range",
                [0.08, 0.35],
            ),
            "pace_diff_scale": cfg.get(
                "baseline_predictor.race.overtake_model.pace_diff_scale", 0.55
            ),
            "skill_scale": cfg.get("baseline_predictor.race.overtake_model.skill_scale", 0.25),
            "defense_scale": cfg.get("baseline_predictor.race.overtake_model.defense_scale", 0.28),
            "race_adv_scale": cfg.get(
                "baseline_predictor.race.overtake_model.race_adv_scale", 0.20
            ),
            "track_ease_scale": cfg.get(
                "baseline_predictor.race.overtake_model.track_ease_scale", 0.18
            ),
            "zone_front_threshold_boost": cfg.get(
                "baseline_predictor.race.overtake_model.zone_front_threshold_boost", 0.22
            ),
            "zone_upper_threshold_boost": cfg.get(
                "baseline_predictor.race.overtake_model.zone_upper_threshold_boost", 0.10
            ),
            "zone_mid_threshold_boost": cfg.get(
                "baseline_predictor.race.overtake_model.zone_mid_threshold_boost", 0.02
            ),
            "zone_back_threshold_boost": cfg.get(
                "baseline_predictor.race.overtake_model.zone_back_threshold_boost", -0.03
            ),
            "zone_front_probability_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_front_probability_scale", 0.55
            ),
            "zone_upper_probability_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_upper_probability_scale", 0.75
            ),
            "zone_mid_probability_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_mid_probability_scale", 0.92
            ),
            "zone_back_probability_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_back_probability_scale", 1.08
            ),
            "zone_front_bonus_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_front_bonus_scale", 0.55
            ),
            "zone_upper_bonus_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_upper_bonus_scale", 0.78
            ),
            "zone_mid_bonus_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_mid_bonus_scale", 0.93
            ),
            "zone_back_bonus_scale": cfg.get(
                "baseline_predictor.race.overtake_model.zone_back_bonus_scale", 1.05
            ),
        }

        # Prepare driver info with per-compound strengths
        driver_info_map, teams_with_long_profile = self._prepare_driver_info_with_compounds(
            validated_grid, race_name
        )

        # Determine race distance from FastF1 metadata, with safe fallback defaults.
        race_distance = resolve_race_distance_laps(
            year=2026,
            race_name=race_name,
            is_sprint=is_sprint,
        )

        # Get tire stress and available compounds
        tire_stress_score = get_tire_stress_score(race_name)
        available_compounds = get_available_compounds(race_name, weather=weather)
        # If no full-wet race is modeled, enforce FIA dry-compound diversity.
        # This keeps mixed forecasts from degenerating into illegal SOFT→SOFT plans.
        enforce_two_compound_rule = weather in {"dry", "mixed"}

        # Restructure race_params for lap_by_lap_simulator (expects nested dicts)
        race_params["base_chaos"] = {
            "dry": race_params.get("base_chaos_dry", 0.35),
            "wet": race_params.get("base_chaos_wet", 0.45),
        }
        race_params["lap1_chaos"] = {
            "front_row": race_params.get("lap1_front_row_chaos", 0.15),
            "upper_midfield": race_params.get("lap1_upper_midfield_chaos", 0.32),
            "midfield": race_params.get("lap1_midfield_chaos", 0.38),
            "back_field": race_params.get("lap1_back_field_chaos", 0.28),
        }
        if "track_overtaking" not in race_params:
            race_params["track_overtaking"] = cfg.get("track_defaults.overtaking_difficulty", 0.5)

        sc_weather_key = "sc_base_prob_wet" if weather in ["rain", "mixed"] else "sc_base_prob_dry"
        default_sc_probability = race_params.get(sc_weather_key, 0.45) + (
            race_params["track_overtaking"] * race_params.get("sc_track_modifier", 0.25)
        )
        race_params["sc_probability"] = race_params.get(
            "sc_probability", np.clip(default_sc_probability, 0.0, 1.0)
        )

        # Provide pit-stop defaults when track params do not override them.
        if "pit_stops" not in race_params:
            race_params["pit_stops"] = {
                "loss_duration": cfg.get("baseline_predictor.race.pit_stops.loss_duration", 22.0),
                "overtake_loss_range": cfg.get(
                    "baseline_predictor.race.pit_stops.overtake_loss_range",
                    [0, 3],
                ),
            }

        # Run lap-by-lap simulations
        simulation_results = []
        base_seed = int(getattr(self, "seed", 42))

        for sim_idx in range(n_simulations):
            rng = np.random.default_rng(base_seed + sim_idx)

            # Generate pit strategies for all drivers (Monte Carlo)
            strategies: dict[str, PitStrategy] = {}
            sprint_compound = (
                "SOFT"
                if "SOFT" in available_compounds
                else (available_compounds[0] if available_compounds else "MEDIUM")
            )
            for driver in driver_info_map.keys():
                if is_sprint:
                    # Sprint races run without scheduled pit stops in this model.
                    strategies[driver] = {
                        "num_stops": 0,
                        "pit_laps": [],
                        "compound_sequence": [sprint_compound],
                        "stint_lengths": [race_distance],
                    }
                else:
                    driver_info = driver_info_map.get(driver, {})
                    strategies[driver] = generate_pit_strategy(
                        race_distance=race_distance,
                        tire_stress_score=tire_stress_score,
                        available_compounds=available_compounds,
                        rng=rng,
                        enforce_two_compound_rule=enforce_two_compound_rule,
                        track_overtaking=race_params.get("track_overtaking"),
                        grid_position=driver_info.get("grid_pos"),
                        strategy_signal=driver_info.get("race_advantage", 0.0),
                    )

            # Simulate race lap-by-lap
            sim_result = simulate_race_lap_by_lap(
                driver_info_map=driver_info_map,
                strategies=strategies,
                race_params=race_params,
                race_distance=race_distance,
                weather=weather,
                rng=rng,
            )

            simulation_results.append(sim_result)

        # Aggregate results across all simulations
        aggregated = aggregate_simulation_results(simulation_results)

        # Blend race simulation output with grid anchoring based on overtaking difficulty.
        # Hard-to-pass tracks preserve more of qualifying order, while easy tracks let
        # pace and racecraft dominate more.
        track_overtaking = race_params.get("track_overtaking", 0.5)
        grid_anchor_weight = np.clip(
            cfg.get("baseline_predictor.race.grid_anchor.base", 0.30)
            + (track_overtaking * cfg.get("baseline_predictor.race.grid_anchor.track_scale", 0.35)),
            0.20,
            0.85,
        )
        grid_anchor_min = cfg.get("baseline_predictor.race.grid_anchor.min", 0.62)
        sprint_grid_anchor_min = cfg.get("baseline_predictor.race.grid_anchor.sprint_min", 0.78)
        grid_anchor_weight = max(
            grid_anchor_weight,
            sprint_grid_anchor_min if is_sprint else grid_anchor_min,
        )
        overtake_blend_scale = cfg.get(
            "baseline_predictor.race.final_blend.overtaking_skill_scale", 1.6
        )
        race_adv_blend_scale = cfg.get(
            "baseline_predictor.race.final_blend.race_advantage_scale", 1.3
        )
        skill_blend_scale = cfg.get("baseline_predictor.race.final_blend.driver_skill_scale", 1.1)
        elite_skill_threshold = cfg.get(
            "baseline_predictor.race.final_blend.elite_driver_skill_threshold", 0.88
        )
        elite_driver_scale = cfg.get("baseline_predictor.race.final_blend.elite_driver_scale", 0.80)
        elite_driver_exponent = cfg.get(
            "baseline_predictor.race.final_blend.elite_driver_exponent", 1.35
        )
        max_driver_adjustment = cfg.get(
            "baseline_predictor.race.final_blend.max_driver_adjustment_positions",
            0.9,
        )
        max_gain_base = cfg.get("baseline_predictor.race.final_blend.max_gain_base", 6.0)
        max_gain_track_scale = cfg.get(
            "baseline_predictor.race.final_blend.max_gain_track_scale",
            4.0,
        )
        skill_gain_scale = cfg.get(
            "baseline_predictor.race.final_blend.max_gain_overtaking_skill_scale", 2.0
        )
        race_adv_gain_scale = cfg.get(
            "baseline_predictor.race.final_blend.max_gain_race_advantage_scale", 2.5
        )
        max_gain_floor = cfg.get("baseline_predictor.race.final_blend.max_gain_floor", 4.0)
        max_gain_ceiling = cfg.get("baseline_predictor.race.final_blend.max_gain_ceiling", 11.0)
        confidence_floor = cfg.get("baseline_predictor.race.confidence.min", 40.0)
        confidence_cap = cfg.get("baseline_predictor.race.confidence.max", 60.0)
        weather_penalty = (
            cfg.get("baseline_predictor.race.confidence.weather_penalty_wet", 4.0)
            if weather in ("rain", "mixed")
            else 0.0
        )

        # Build finish order from blended position scores
        finish_order = []
        blended_samples_by_driver: dict[str, list[float]] = {}
        team_to_drivers: dict[str, list[str]] = {}
        for driver_code, info in driver_info_map.items():
            team_to_drivers.setdefault(info["team"], []).append(driver_code)
        learning_position_scale = float(
            cfg.get("baseline_predictor.race.learning.position_adjustment_scale", 0.70)
        )

        for driver_code, median_pos in aggregated["median_positions"].items():
            info = driver_info_map[driver_code]
            positions = aggregated["position_distributions"][driver_code]

            # Confidence based on consistency; wet/mixed runs receive an uncertainty penalty.
            position_std = np.std(positions)
            confidence = max(
                confidence_floor,
                min(confidence_cap, confidence_cap - (position_std * 3.0) - weather_penalty),
            )

            overtake_ease = 1.0 - track_overtaking
            racecraft_adjustment = (
                ((info["overtaking_skill"] - 0.5) * overtake_ease * overtake_blend_scale)
                + (info["race_advantage"] * race_adv_blend_scale)
                + ((info["skill"] - 0.5) * skill_blend_scale)
            )

            elite_denominator = max(1e-6, 1.0 - elite_skill_threshold)
            elite_driver_normalized = max(
                0.0, (info["skill"] - elite_skill_threshold) / elite_denominator
            )
            elite_driver_adjustment = (
                (elite_driver_normalized**elite_driver_exponent)
                * elite_driver_scale
                * (0.6 + (0.4 * overtake_ease))
            )
            racecraft_adjustment += elite_driver_adjustment

            is_elite_driver = info["skill"] >= elite_skill_threshold
            if info["grid_pos"] <= 3 and not is_elite_driver:
                adjustment_cap_negative = max_driver_adjustment * 0.5
                adjustment_cap_positive = max_driver_adjustment
                racecraft_adjustment = np.clip(
                    racecraft_adjustment,
                    -adjustment_cap_negative,
                    adjustment_cap_positive,
                )
            else:
                racecraft_adjustment = np.clip(
                    racecraft_adjustment,
                    -max_driver_adjustment,
                    max_driver_adjustment,
                )

            max_gain = (
                max_gain_base
                + (overtake_ease * max_gain_track_scale)
                + ((info["overtaking_skill"] - 0.5) * skill_gain_scale)
                + (max(0.0, info["race_advantage"]) * race_adv_gain_scale)
            )
            max_gain = np.clip(max_gain, max_gain_floor, max_gain_ceiling)
            min_position_score = max(1.0, info["grid_pos"] - max_gain)

            position_blend_score = (
                ((1.0 - grid_anchor_weight) * median_pos)
                + (grid_anchor_weight * info["grid_pos"])
                - racecraft_adjustment
            )
            learned_position_adjustment = self._get_learned_position_adjustment(
                team=info["team"],
                driver=driver_code,
                teammates=team_to_drivers.get(info["team"], []),
                session="race",
            )
            position_blend_score -= learned_position_adjustment * learning_position_scale
            position_blend_score = max(position_blend_score, min_position_score)

            blended_position_samples = [
                ((1.0 - grid_anchor_weight) * p)
                + (grid_anchor_weight * info["grid_pos"])
                - racecraft_adjustment
                for p in positions
            ]
            if learned_position_adjustment:
                blended_position_samples = [
                    sample - (learned_position_adjustment * learning_position_scale)
                    for sample in blended_position_samples
                ]
            blended_position_samples = [
                max(sample_position, min_position_score)
                for sample_position in blended_position_samples
            ]
            blended_samples_by_driver[driver_code] = blended_position_samples

            finish_order.append(
                {
                    "driver": driver_code,
                    "team": info["team"],
                    "median_position": median_pos,
                    "position_blend_score": round(position_blend_score, 4),
                    "p5": int(np.percentile(blended_position_samples, 5)),
                    "p95": int(np.percentile(blended_position_samples, 95)),
                    "confidence": round(confidence, 1),
                    # Filled after cross-driver sample ranking for consistency with final ordering.
                    "podium_probability": 0.0,
                    "dnf_probability": round(aggregated["dnf_rates"].get(driver_code, 0.0), 3),
                }
            )

        # Podium probability must come from ranked outcomes (relative order), not
        # absolute blended score thresholds, otherwise P2 can show as 0%.
        podium_prob_by_driver: dict[str, float] = {}
        rank_samples_by_driver: dict[str, list[int]] = {
            driver: [] for driver in blended_samples_by_driver.keys()
        }
        if blended_samples_by_driver:
            sample_lengths = [len(samples) for samples in blended_samples_by_driver.values()]
            sample_count = min(sample_lengths) if sample_lengths else 0
            if sample_count > 0:
                podium_counts: dict[str, int] = {driver: 0 for driver in blended_samples_by_driver}
                minimum_podium_samples = int(
                    cfg.get("baseline_predictor.race.podium_probability.min_sample_count", 250)
                )
                minimum_podium_samples = max(1, minimum_podium_samples)
                use_resampling = sample_count < minimum_podium_samples
                draw_count = minimum_podium_samples if use_resampling else sample_count

                resample_rng = None
                if use_resampling:
                    resample_seed_offset = int(
                        cfg.get(
                            "baseline_predictor.race.podium_probability.resample_seed_offset",
                            99173,
                        )
                    )
                    resample_rng = np.random.default_rng(base_seed + resample_seed_offset)

                for sample_idx in range(draw_count):
                    if resample_rng is None:
                        ranked = sorted(
                            blended_samples_by_driver.items(),
                            key=lambda item: (item[1][sample_idx], item[0]),
                        )
                    else:
                        ranked = sorted(
                            (
                                driver_code,
                                samples[int(resample_rng.integers(0, sample_count))],
                            )
                            for driver_code, samples in blended_samples_by_driver.items()
                        )
                        ranked = sorted(ranked, key=lambda item: (item[1], item[0]))
                    for rank_index, (driver_code, _) in enumerate(ranked, start=1):
                        rank_samples_by_driver[driver_code].append(rank_index)
                    for driver_code, _ in ranked[:3]:
                        podium_counts[driver_code] += 1

                podium_prob_by_driver = {
                    driver: (count / draw_count) * 100.0 for driver, count in podium_counts.items()
                }

        for row in finish_order:
            row["podium_probability"] = round(podium_prob_by_driver.get(row["driver"], 0.0), 1)
            rank_samples = rank_samples_by_driver.get(row["driver"], [])
            if rank_samples:
                row["position_blend_score"] = round(float(np.mean(rank_samples)), 4)
                row["median_position"] = int(np.median(rank_samples))
                row["p5"] = int(np.percentile(rank_samples, 5))
                row["p95"] = int(np.percentile(rank_samples, 95))

        # Sort by blended position score
        finish_order.sort(key=lambda x: x["position_blend_score"])

        # Assign final positions
        for i, item in enumerate(finish_order):
            item["position"] = i + 1

        if cfg.get("baseline_predictor.race.podium_probability.enforce_monotonic", True):
            raw_podium_values = [float(row.get("podium_probability", 0.0)) for row in finish_order]
            smoothed_values = self._enforce_non_increasing(raw_podium_values)
            for row, smoothed in zip(finish_order, smoothed_values, strict=True):
                row["podium_probability"] = round(float(np.clip(smoothed, 0.0, 100.0)), 1)

        return {
            "finish_order": finish_order,
            "characteristics_profile_used": "long_run",
            "teams_with_characteristics_profile": teams_with_long_profile,
            "compound_strategies": aggregated["compound_strategy_distribution"],
            "pit_lap_distribution": aggregated["pit_lap_distribution"],
        }
