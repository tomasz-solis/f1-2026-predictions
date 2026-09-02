"""Race prediction pipeline helper for baseline predictor mixin."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.simulation.pit_strategy import sample_sprint_compound as _sample_sprint_compound
from src.types.prediction_types import PitStrategy, QualifyingGridEntry, RaceSimulationResult

from .grid_uncertainty import (
    prepare_grid_uncertainty_profile,
    sample_probabilistic_grid_positions,
)
from .result_processing import (
    build_finish_order,
)
from .weather_context import (
    build_weather_feature_context,
    resolve_race_environment_context,
)


@dataclass(frozen=True)
class RaceSimulationDeps:
    """Bundle race-simulation callables into one readable dependency object."""

    load_race_params: Callable[[], dict[str, Any]]
    prepare_driver_info_with_compounds: Callable[
        [list[QualifyingGridEntry], str | None],
        tuple[dict[str, Any], int],
    ]
    get_learned_position_adjustment: Callable[..., float]
    get_learned_interval_radius: Callable[..., float]
    apply_race_residual_model: Callable[..., dict[str, float]] | None
    get_conformal_interval_radius: Callable[..., float] | None
    enforce_non_increasing: Callable[[list[float]], list[float]]
    load_track_specific_params: Callable[..., dict[str, Any]]
    get_tire_stress_score: Callable[..., float]
    get_available_compounds: Callable[..., list[str]]
    resolve_track_temperature_c: Callable[..., float | None] | None
    resolve_track_temperature_profile: Callable[..., dict[str, Any] | None] | None
    resolve_non_competitive_weather_features: Callable[..., dict[str, Any] | None] | None
    resolve_race_distance_laps: Callable[..., int]
    generate_pit_strategy: Callable[..., PitStrategy]
    simulate_race_lap_by_lap: Callable[..., RaceSimulationResult]
    aggregate_simulation_results: Callable[[list[RaceSimulationResult]], dict[str, Any]]


def _build_weather_feature_context(
    *,
    selected_weather: str,
    raw_features: dict[str, Any],
    cfg: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Translate non-competitive weather features into simulation modifiers."""
    return build_weather_feature_context(
        selected_weather=selected_weather,
        raw_features=raw_features,
        cfg=cfg,
    )


def _prepare_grid_uncertainty_profile(
    *,
    validated_grid: list[QualifyingGridEntry],
    input_confidence: float | None,
    cfg: Any,
) -> dict[str, dict[str, float]]:
    """Build per-driver grid-sampling settings from qualifying uncertainty fields."""
    return prepare_grid_uncertainty_profile(
        validated_grid=validated_grid,
        input_confidence=input_confidence,
        cfg=cfg,
    )


def _sample_probabilistic_grid_positions(
    *,
    validated_grid: list[QualifyingGridEntry],
    grid_uncertainty_profile: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Sample one coherent starting-grid permutation from qualifying uncertainty."""
    return sample_probabilistic_grid_positions(
        validated_grid=validated_grid,
        grid_uncertainty_profile=grid_uncertainty_profile,
        rng=rng,
    )


def predict_race_core(
    *,
    validated_grid: list[QualifyingGridEntry],
    weather: str,
    race_name: str | None,
    n_simulations: int,
    is_sprint: bool,
    race_compound: str,
    input_confidence: float | None,
    year: int,
    cfg: Any,
    base_seed: int,
    deps: RaceSimulationDeps,
    location: str | None = None,
) -> dict[str, Any]:
    """Run the full race prediction flow with injectable dependencies."""
    from src.models.conformal_calibration import resolve_race_data_regime
    from src.utils.validation_helpers import normalize_weather_key

    weather = normalize_weather_key(weather)
    _ = race_compound

    # ``location`` (schedule venue) makes circuit resolution authoritative; falls back to
    # name/year when absent or when a dep predates the kwarg.
    try:
        track_params = deps.load_track_specific_params(race_name, year=year, location=location)
    except TypeError:
        try:
            track_params = deps.load_track_specific_params(race_name, year=year)
        except TypeError:
            # Backward compatibility for patched/legacy callables without kwargs.
            track_params = deps.load_track_specific_params(race_name)
    base_params = deps.load_race_params()

    race_params = {**base_params, **track_params}
    race_params["track_name"] = race_name
    # Lets simulate_race_lap_by_lap resolve the season's measured team race-pace
    # artifact (data/processed/team_race_pace/) without threading a new parameter
    # through this whole call chain.
    race_params["year"] = year

    race_params["fuel"] = {
        "initial_load_kg": cfg.get("baseline_predictor.race.fuel.initial_load_kg", 110.0),
        "effect_per_lap": cfg.get("baseline_predictor.race.fuel.effect_per_lap", 0.035),
        "burn_rate_kg_per_lap": cfg.get("baseline_predictor.race.fuel.burn_rate_kg_per_lap", 1.5),
    }

    race_params["lap_time"] = {
        "reference_base": cfg.get("baseline_predictor.race.lap_time.reference_base", 90.0),
        "team_pace_penalty_range": cfg.get(
            "baseline_predictor.race.lap_time.team_pace_penalty_range", 5.0
        ),
        "skill_improvement_max": cfg.get(
            "baseline_predictor.race.lap_time.skill_improvement_max", 0.75
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
    race_params["wet_skill_lap_weight"] = cfg.get(
        "baseline_predictor.race.lap_time.wet_skill_lap_weight", 0.16
    )
    race_params["wet_skill_neutral"] = cfg.get(
        "baseline_predictor.race.lap_time.wet_skill_neutral", 0.70
    )
    race_params["mixed_wet_blend"] = cfg.get("baseline_predictor.mixed_wet_blend", 0.50)
    # Track-specific wet severity: derived from track_overtaking if not set directly.
    # Street circuits (high overtaking difficulty) amplify wet effects.
    if "track_wet_severity" not in race_params:
        track_ot = float(race_params.get("track_overtaking", 0.5))
        wet_sev_base = float(
            cfg.get("baseline_predictor.race.lap_time.track_wet_severity_base", 0.80)
        )
        wet_sev_scale = float(
            cfg.get("baseline_predictor.race.lap_time.track_wet_severity_scale", 0.40)
        )
        race_params["track_wet_severity"] = wet_sev_base + (track_ot * wet_sev_scale)
    race_params["start_grid_gap_seconds"] = cfg.get(
        "baseline_predictor.race.start_grid_gap_seconds", 0.32
    )
    race_params["race_advantage_lap_impact"] = cfg.get(
        "baseline_predictor.race.race_advantage_lap_impact", 0.35
    )
    race_params["safety_car_trigger_lap"] = cfg.get(
        "baseline_predictor.race.safety_car_trigger_lap", 10
    )
    # Compact 5-param overtake model (expanded to full set inside the simulator).
    race_params["overtake_model"] = {
        "dirty_air_window_s": cfg.get(
            "baseline_predictor.race.overtake_model.dirty_air_window_s", 1.8
        ),
        "pace_weight": cfg.get("baseline_predictor.race.overtake_model.pace_weight", 0.55),
        "racecraft_weight": cfg.get(
            "baseline_predictor.race.overtake_model.racecraft_weight", 0.25
        ),
        "track_factor": cfg.get("baseline_predictor.race.overtake_model.track_factor", 0.35),
        "pass_chance_base": cfg.get(
            "baseline_predictor.race.overtake_model.pass_chance_base", 0.30
        ),
    }

    driver_info_map, teams_with_long_profile = deps.prepare_driver_info_with_compounds(
        validated_grid, race_name
    )
    confidence_values = [
        float(row["confidence"]) / 100.0
        for row in validated_grid
        if isinstance(row.get("confidence"), int | float)
    ]
    mean_grid_confidence = (
        float(sum(confidence_values) / len(confidence_values)) if confidence_values else None
    )
    data_regime = resolve_race_data_regime(
        input_confidence=input_confidence,
        mean_grid_confidence=mean_grid_confidence,
    )
    race_residual_adjustments: dict[str, float] = {}
    if callable(deps.apply_race_residual_model):
        race_residual_adjustments = deps.apply_race_residual_model(
            driver_info_map=driver_info_map,
            qualifying_grid=validated_grid,
            race_name=race_name,
            weather=weather,
            input_confidence=input_confidence,
            is_sprint=is_sprint,
            year=year,
        )
    grid_uncertainty_profile = _prepare_grid_uncertainty_profile(
        validated_grid=validated_grid,
        input_confidence=input_confidence,
        cfg=cfg,
    )
    grid_position_samples_by_driver: dict[str, list[float]] = {
        driver: [] for driver in driver_info_map.keys()
    }

    try:
        race_distance = deps.resolve_race_distance_laps(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            location=location,
        )
    except TypeError:
        race_distance = deps.resolve_race_distance_laps(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
        )

    try:
        tire_stress_score = deps.get_tire_stress_score(race_name, year=year, location=location)
    except TypeError:
        try:
            tire_stress_score = deps.get_tire_stress_score(race_name, year=year)
        except TypeError:
            # Backward compatibility for patched/legacy callables without kwargs.
            tire_stress_score = deps.get_tire_stress_score(race_name)
    # Make stress score available to the lap-by-lap simulator for per-track cliff ages.
    race_params["tire_stress_score"] = float(tire_stress_score)
    available_compounds = deps.get_available_compounds(race_name, weather=weather)
    enforce_two_compound_rule = weather in {"dry", "mixed"}

    base_chaos_dry = float(race_params.get("base_chaos_dry", 0.35))
    base_chaos_wet = float(race_params.get("base_chaos_wet", 0.45))
    mixed_weather_chaos_blend = float(
        np.clip(
            race_params.get(
                "mixed_weather_chaos_blend",
                cfg.get("baseline_predictor.race.base_chaos.mixed_blend", 0.55),
            ),
            0.0,
            1.0,
        )
    )
    base_chaos_mixed = base_chaos_dry + (
        (base_chaos_wet - base_chaos_dry) * mixed_weather_chaos_blend
    )
    race_params["mixed_weather_chaos_blend"] = mixed_weather_chaos_blend
    race_params["base_chaos"] = {
        "dry": base_chaos_dry,
        "wet": base_chaos_wet,
        "mixed": base_chaos_mixed,
    }
    race_params["lap1_chaos"] = {
        "front_row": race_params.get("lap1_front_row_chaos", 0.15),
        "upper_midfield": race_params.get("lap1_upper_midfield_chaos", 0.32),
        "midfield": race_params.get("lap1_midfield_chaos", 0.38),
        "back_field": race_params.get("lap1_back_field_chaos", 0.28),
    }
    if "track_overtaking" not in race_params:
        race_params["track_overtaking"] = cfg.get("track_defaults.overtaking_difficulty", 0.5)

    track_temperature_context, weather_feature_modifiers, weather_feature_context = (
        resolve_race_environment_context(
            race_params=race_params,
            weather=weather,
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            cfg=cfg,
            resolve_track_temperature_c=deps.resolve_track_temperature_c,
            resolve_track_temperature_profile=deps.resolve_track_temperature_profile,
            resolve_non_competitive_weather_features=deps.resolve_non_competitive_weather_features,
        )
    )

    sc_weather_key = "sc_base_prob_wet" if weather in ["rain", "mixed"] else "sc_base_prob_dry"
    default_sc_probability = race_params.get(sc_weather_key, 0.45) + (
        race_params["track_overtaking"] * race_params.get("sc_track_modifier", 0.25)
    )
    race_params["sc_probability"] = race_params.get(
        "sc_probability", np.clip(default_sc_probability, 0.0, 1.0)
    )
    race_params["sc_pit_loss_reduction_s"] = float(
        race_params.get(
            "sc_pit_loss_reduction_s",
            cfg.get("baseline_predictor.race.sc_pit_loss_reduction_s", 12.0),
        )
    )
    race_params["vsc_pit_loss_reduction_s"] = float(
        race_params.get(
            "vsc_pit_loss_reduction_s",
            cfg.get("baseline_predictor.race.vsc_pit_loss_reduction_s", 5.0),
        )
    )
    race_params["sc_compression_gap_s"] = float(
        race_params.get(
            "sc_compression_gap_s",
            cfg.get("baseline_predictor.race.sc_compression_gap_s", 0.60),
        )
    )
    race_params["sc_tire_wear_fraction"] = float(
        race_params.get(
            "sc_tire_wear_fraction",
            cfg.get("baseline_predictor.race.sc_tire_wear_fraction", 0.65),
        )
    )

    if "pit_stops" not in race_params:
        race_params["pit_stops"] = {
            "loss_duration": cfg.get("baseline_predictor.race.pit_stops.loss_duration", 22.0),
            "overtake_loss_range": cfg.get(
                "baseline_predictor.race.pit_stops.overtake_loss_range",
                [0, 3],
            ),
        }

    simulation_results = []

    for sim_idx in range(n_simulations):
        rng = np.random.default_rng(base_seed + sim_idx)
        sampled_grid_positions = _sample_probabilistic_grid_positions(
            validated_grid=validated_grid,
            grid_uncertainty_profile=grid_uncertainty_profile,
            rng=rng,
        )
        simulation_driver_info_map = {}
        for driver_code, info in driver_info_map.items():
            sampled_grid_pos = int(sampled_grid_positions.get(driver_code, info["grid_pos"]))
            grid_position_samples_by_driver.setdefault(driver_code, []).append(
                float(sampled_grid_pos)
            )
            simulation_driver_info_map[driver_code] = {
                **info,
                "grid_pos": sampled_grid_pos,
            }

        strategies: dict[str, PitStrategy] = {}
        for driver in simulation_driver_info_map.keys():
            if is_sprint:
                driver_info = simulation_driver_info_map.get(driver, {})
                sprint_compound = _sample_sprint_compound(
                    available_compounds=available_compounds,
                    grid_position=driver_info.get("grid_pos"),
                    tire_stress_score=tire_stress_score,
                    rng=rng,
                )
                strategies[driver] = {
                    "num_stops": 0,
                    "pit_laps": [],
                    "compound_sequence": [sprint_compound],
                    "stint_lengths": [race_distance],
                }
            else:
                driver_info = simulation_driver_info_map.get(driver, {})
                strategies[driver] = deps.generate_pit_strategy(
                    race_distance=race_distance,
                    tire_stress_score=tire_stress_score,
                    available_compounds=available_compounds,
                    rng=rng,
                    enforce_two_compound_rule=enforce_two_compound_rule,
                    track_overtaking=race_params.get("track_overtaking"),
                    grid_position=driver_info.get("grid_pos"),
                    strategy_signal=driver_info.get("race_advantage", 0.0),
                )

        sim_result = deps.simulate_race_lap_by_lap(
            driver_info_map=simulation_driver_info_map,
            strategies=strategies,
            race_params=race_params,
            race_distance=race_distance,
            weather=weather,
            rng=rng,
        )

        simulation_results.append(sim_result)

    aggregated = deps.aggregate_simulation_results(simulation_results)

    finish_order = build_finish_order(
        aggregated=aggregated,
        driver_info_map=driver_info_map,
        grid_position_samples_by_driver=grid_position_samples_by_driver,
        field_size=max(1, len(validated_grid)),
        weather=weather,
        is_sprint=is_sprint,
        input_confidence=input_confidence,
        cfg=cfg,
        race_params=race_params,
        weather_feature_modifiers=weather_feature_modifiers,
        get_learned_position_adjustment=deps.get_learned_position_adjustment,
        learned_interval_radius=max(
            float(deps.get_learned_interval_radius(session="race")),
            float(
                deps.get_conformal_interval_radius(session="race", regime=data_regime)
                if callable(deps.get_conformal_interval_radius)
                else 0.0
            ),
        ),
        enforce_non_increasing=deps.enforce_non_increasing,
        base_seed=base_seed,
    )

    return {
        "finish_order": finish_order,
        "data_regime": data_regime,
        "characteristics_profile_used": "long_run",
        "teams_with_characteristics_profile": teams_with_long_profile,
        "compound_strategies": aggregated["compound_strategy_distribution"],
        "pit_lap_distribution": aggregated["pit_lap_distribution"],
        "track_temperature_context": track_temperature_context,
        "weather_feature_context": weather_feature_context,
        "race_residual_model_used": bool(race_residual_adjustments),
        "race_residual_mean_abs_adjustment": round(
            float(np.mean([abs(value) for value in race_residual_adjustments.values()]))
            if race_residual_adjustments
            else 0.0,
            4,
        ),
    }
