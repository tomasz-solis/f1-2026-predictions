"""Race prediction pipeline helper for baseline predictor mixin."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.types.prediction_types import PitStrategy, QualifyingGridEntry


def _coerce_optional_float(value: Any) -> float | None:
    """Convert value to float when possible; otherwise return None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _weather_bucket_mismatch_score(selected_weather: str, practice_weather: str) -> float:
    """Return mismatch score between selected and practice-derived weather buckets."""
    selected_key = str(selected_weather).strip().lower()
    practice_key = str(practice_weather).strip().lower()

    if not practice_key or practice_key == "unknown":
        return 0.25
    if selected_key == practice_key:
        return 0.0
    if selected_key == "mixed" or practice_key == "mixed":
        return 0.5
    return 1.0


def _build_weather_feature_context(
    *,
    selected_weather: str,
    raw_features: dict[str, Any],
    cfg: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Translate non-competitive weather features into simulation modifiers.

    The selector weather remains the primary scenario driver. Practice weather
    only modulates uncertainty and variance from dry/mixed/rain scenario
    alignment; wind/cold sensitivity is intentionally excluded.
    """
    source_session = raw_features.get("source_session")
    practice_weather_bucket = str(raw_features.get("practice_weather_bucket", "unknown"))
    wind_speed_kph = _coerce_optional_float(raw_features.get("wind_speed_kph"))
    track_temperature_c = _coerce_optional_float(raw_features.get("track_temperature_c"))
    air_temperature_c = _coerce_optional_float(raw_features.get("air_temperature_c"))
    humidity_pct = _coerce_optional_float(raw_features.get("humidity_pct"))
    rainfall_signal = _coerce_optional_float(raw_features.get("rainfall_signal"))

    mismatch_score = _weather_bucket_mismatch_score(selected_weather, practice_weather_bucket)

    mismatch_chaos_boost = float(
        cfg.get("baseline_predictor.race.weather_features.mismatch.chaos_boost", 0.18)
    )
    mismatch_variance_boost = float(
        cfg.get("baseline_predictor.race.weather_features.mismatch.variance_boost", 0.10)
    )
    mismatch_confidence_penalty = float(
        cfg.get("baseline_predictor.race.weather_features.mismatch.confidence_penalty", 2.0)
    )

    chaos_multiplier = float(np.clip(1.0 + (mismatch_score * mismatch_chaos_boost), 0.80, 1.40))
    teammate_variance_multiplier = float(
        np.clip(1.0 + (mismatch_score * mismatch_variance_boost), 0.80, 1.35)
    )
    confidence_adjustment = float(
        max(
            0.0,
            mismatch_score * mismatch_confidence_penalty,
        )
    )

    modifiers = {
        "chaos_multiplier": chaos_multiplier,
        "teammate_variance_multiplier": teammate_variance_multiplier,
        "confidence_adjustment": confidence_adjustment,
    }
    context = {
        "available": bool(raw_features.get("available", False)),
        "source_session": source_session,
        "selected_weather": str(selected_weather).strip().lower(),
        "practice_weather_bucket": practice_weather_bucket,
        "track_temperature_c": track_temperature_c,
        "air_temperature_c": air_temperature_c,
        "wind_speed_kph": wind_speed_kph,
        "humidity_pct": humidity_pct,
        "rainfall_signal": rainfall_signal,
        "weather_mismatch_score": mismatch_score,
        "chaos_multiplier": chaos_multiplier,
        "teammate_variance_multiplier": teammate_variance_multiplier,
        "confidence_adjustment": confidence_adjustment,
    }
    return modifiers, context


def _apply_low_confidence_interval_floor(
    *,
    finish_order: list[dict[str, Any]],
    input_confidence: float | None,
    cfg: Any,
    field_size: int,
) -> None:
    """Widen top-driver position ranges when run confidence is explicitly low.

    The race simulator can occasionally produce a point interval (for example
    ``P1-P1``) when one driver dominates all sampled outcomes. That can still
    happen in low-signal runs where inputs are incomplete and uncertainty should
    remain visible. This helper applies a small, bounded interval-width floor to
    top-ranked drivers only when the caller provided a low ``input_confidence``.
    """
    if input_confidence is None or field_size <= 1:
        return

    confidence = float(np.clip(input_confidence, 0.0, 1.0))
    threshold = float(
        np.clip(
            cfg.get(
                "baseline_predictor.race.position_interval_floor.apply_below_input_confidence",
                0.65,
            ),
            0.0,
            1.0,
        )
    )
    if threshold <= 0.0 or confidence >= threshold:
        return

    top_n = int(cfg.get("baseline_predictor.race.position_interval_floor.top_n", 3))
    top_n = max(1, min(top_n, field_size))

    min_width = int(cfg.get("baseline_predictor.race.position_interval_floor.min_width", 1))
    min_width = max(0, min(min_width, field_size - 1))
    max_extra_width = int(
        cfg.get("baseline_predictor.race.position_interval_floor.max_extra_width", 1)
    )
    max_extra_width = max(0, min(max_extra_width, field_size - 1))

    low_confidence_share = float((threshold - confidence) / max(threshold, 1e-6))
    target_width = min(
        field_size - 1,
        min_width + int(round(low_confidence_share * max_extra_width)),
    )
    if target_width <= 0:
        return

    for row in finish_order:
        try:
            position = int(row.get("position", field_size))
            p5 = int(row.get("p5", position))
            p95 = int(row.get("p95", position))
        except (TypeError, ValueError):
            continue

        if position > top_n:
            continue

        p5 = max(1, min(p5, field_size))
        p95 = max(p5, min(p95, field_size))
        current_width = p95 - p5
        if current_width >= target_width:
            row["p5"] = p5
            row["p95"] = p95
            continue

        deficit = target_width - current_width

        # Prefer widening upward for front-runners (P1 -> P2/P3) to keep the
        # optimistic tail realistic without implying impossible negative places.
        upper_room = field_size - p95
        add_upper = min(upper_room, deficit)
        deficit -= add_upper

        lower_room = p5 - 1
        add_lower = min(lower_room, deficit)
        deficit -= add_lower

        if deficit > 0 and upper_room > add_upper:
            extra_upper = min(upper_room - add_upper, deficit)
            add_upper += extra_upper
            deficit -= extra_upper
        if deficit > 0 and lower_room > add_lower:
            extra_lower = min(lower_room - add_lower, deficit)
            add_lower += extra_lower

        row["p5"] = p5 - add_lower
        row["p95"] = p95 + add_upper


def _normalize_confidence_to_unit_interval(value: Any) -> float:
    """Convert confidence values expressed as 0-1 or 0-100 into a 0-1 scale."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0

    if confidence > 1.0:
        confidence /= 100.0
    return float(np.clip(confidence, 0.0, 1.0))


def _coerce_grid_position_metric(row: QualifyingGridEntry, key: str, fallback: int) -> int:
    """Read an integer grid metric from a qualifying row with a safe fallback."""
    raw_value: Any = row.get(key, fallback)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(fallback)


def _prepare_grid_uncertainty_profile(
    *,
    validated_grid: list[QualifyingGridEntry],
    input_confidence: float | None,
    cfg: Any,
) -> dict[str, dict[str, float]]:
    """Build per-driver grid-sampling settings from qualifying uncertainty fields.

    Predicted qualifying results already include uncertainty signals (`median_position`,
    `p5`, `p95`, `confidence`), but a race simulation normally receives only one
    ranked grid order. This profile lets each race Monte Carlo run sample a coherent
    starting grid from those qualifying ranges instead of treating a forecast grid as
    fully known.
    """
    if not validated_grid:
        return {}

    field_size = max(1, len(validated_grid))
    input_uncertainty = 0.0
    if input_confidence is not None:
        input_uncertainty = float(1.0 - np.clip(input_confidence, 0.0, 1.0))

    base_std = float(cfg.get("baseline_predictor.race.grid_uncertainty.base_std", 0.35))
    interval_divisor = float(
        max(
            1e-6,
            cfg.get("baseline_predictor.race.grid_uncertainty.interval_divisor", 3.29),
        )
    )
    confidence_scale = float(
        cfg.get("baseline_predictor.race.grid_uncertainty.confidence_scale", 0.90)
    )
    input_confidence_scale = float(
        cfg.get("baseline_predictor.race.grid_uncertainty.input_confidence_scale", 0.60)
    )
    position_delta_scale = float(
        cfg.get("baseline_predictor.race.grid_uncertainty.position_delta_scale", 0.35)
    )
    max_std = float(
        cfg.get(
            "baseline_predictor.race.grid_uncertainty.max_std",
            max(1.0, field_size / 4.0),
        )
    )

    profile: dict[str, dict[str, float]] = {}
    has_probabilistic_signal = False

    for row in validated_grid:
        driver = str(row["driver"])
        base_position = int(row["position"])
        center_position = _coerce_grid_position_metric(row, "median_position", base_position)
        p5 = _coerce_grid_position_metric(row, "p5", center_position)
        p95 = _coerce_grid_position_metric(row, "p95", center_position)
        interval_width = max(0, p95 - p5)
        position_delta = abs(center_position - base_position)
        row_confidence = _normalize_confidence_to_unit_interval(row.get("confidence", 1.0))
        has_row_signal = any(key in row for key in ("median_position", "p5", "p95", "confidence"))

        if not has_row_signal:
            profile[driver] = {
                "center": float(base_position),
                "std": 0.0,
            }
            continue

        std = max(base_std, interval_width / interval_divisor)
        std += position_delta * position_delta_scale
        std *= (
            1.0
            + ((1.0 - row_confidence) * confidence_scale)
            + (input_uncertainty * input_confidence_scale)
        )
        std = float(np.clip(std, 0.0, max_std))
        has_probabilistic_signal = has_probabilistic_signal or std > 0.0
        profile[driver] = {
            "center": float(center_position),
            "std": std,
        }

    return profile if has_probabilistic_signal else {}


def _sample_probabilistic_grid_positions(
    *,
    validated_grid: list[QualifyingGridEntry],
    grid_uncertainty_profile: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Sample one coherent starting-grid permutation from qualifying uncertainty."""
    if not validated_grid:
        return {}

    if not grid_uncertainty_profile:
        return {str(row["driver"]): int(row["position"]) for row in validated_grid}

    latent_scores: list[tuple[str, float, int]] = []
    for row in validated_grid:
        driver = str(row["driver"])
        fallback_position = int(row["position"])
        uncertainty = grid_uncertainty_profile.get(driver, {})
        center = float(uncertainty.get("center", fallback_position))
        std = float(uncertainty.get("std", 0.0))
        latent_position = center if std <= 0.0 else float(rng.normal(center, std))
        latent_scores.append((driver, latent_position, fallback_position))

    ranked = sorted(latent_scores, key=lambda item: (item[1], item[2], item[0]))
    return {driver: index for index, (driver, _, _) in enumerate(ranked, start=1)}


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
    load_race_params: Any,
    prepare_driver_info_with_compounds: Any,
    get_learned_position_adjustment: Any,
    enforce_non_increasing: Any,
    load_track_specific_params: Any,
    get_tire_stress_score: Any,
    get_available_compounds: Any,
    resolve_track_temperature_c: Any | None,
    resolve_track_temperature_profile: Any | None,
    resolve_non_competitive_weather_features: Any | None,
    resolve_race_distance_laps: Any,
    generate_pit_strategy: Any,
    simulate_race_lap_by_lap: Any,
    aggregate_simulation_results: Any,
) -> dict[str, Any]:
    """Run the full race prediction flow with injectable dependencies."""
    _ = race_compound

    try:
        track_params = load_track_specific_params(race_name, year=year)
    except TypeError:
        # Backward compatibility for patched/legacy callables without year kwargs.
        track_params = load_track_specific_params(race_name)
    base_params = load_race_params()

    race_params = {**base_params, **track_params}
    race_params["track_name"] = race_name

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

    driver_info_map, teams_with_long_profile = prepare_driver_info_with_compounds(
        validated_grid, race_name
    )
    grid_uncertainty_profile = _prepare_grid_uncertainty_profile(
        validated_grid=validated_grid,
        input_confidence=input_confidence,
        cfg=cfg,
    )
    grid_position_samples_by_driver: dict[str, list[float]] = {
        driver: [] for driver in driver_info_map.keys()
    }

    race_distance = resolve_race_distance_laps(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
    )

    try:
        tire_stress_score = get_tire_stress_score(race_name, year=year)
    except TypeError:
        # Backward compatibility for patched/legacy callables without year kwargs.
        tire_stress_score = get_tire_stress_score(race_name)
    available_compounds = get_available_compounds(race_name, weather=weather)
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

    track_temperature_context: dict[str, Any] = {}
    if "track_temperature_c" not in race_params:
        resolved_temperature_profile: dict[str, Any] | None = None
        if callable(resolve_track_temperature_profile):
            try:
                candidate_profile = resolve_track_temperature_profile(
                    year=year,
                    race_name=race_name,
                    weather=weather,
                    is_sprint=is_sprint,
                )
            except TypeError:
                try:
                    candidate_profile = resolve_track_temperature_profile(
                        year,
                        race_name,
                        weather,
                        is_sprint,
                    )
                except Exception:
                    candidate_profile = None
            except Exception:
                candidate_profile = None

            if isinstance(candidate_profile, dict):
                resolved_temperature_profile = dict(candidate_profile)

        resolved_track_temp: float | None = None
        if resolved_temperature_profile is not None:
            try:
                resolved_track_temp = float(resolved_temperature_profile["track_temperature_c"])
            except (KeyError, TypeError, ValueError):
                resolved_track_temp = None

        if resolved_track_temp is None and callable(resolve_track_temperature_c):
            try:
                resolved_track_temp = float(
                    resolve_track_temperature_c(
                        year=year,
                        race_name=race_name,
                        weather=weather,
                        is_sprint=is_sprint,
                    )
                )
            except TypeError:
                try:
                    resolved_track_temp = float(
                        resolve_track_temperature_c(year, race_name, weather, is_sprint)
                    )
                except Exception:
                    resolved_track_temp = None
            except Exception:
                resolved_track_temp = None

        if resolved_track_temp is not None:
            race_params["track_temperature_c"] = resolved_track_temp
            if resolved_temperature_profile is not None:
                track_temperature_context = {
                    "track_temperature_c": float(resolved_track_temp),
                    "source": str(
                        resolved_temperature_profile.get("source", "session_or_fallback")
                    ),
                    "reason": str(resolved_temperature_profile.get("reason", "")),
                    "weather_bucket": str(
                        resolved_temperature_profile.get("weather_bucket", weather)
                    ),
                    "session_name": resolved_temperature_profile.get("session_name"),
                    "session_track_temperature_c": resolved_temperature_profile.get(
                        "session_track_temperature_c"
                    ),
                    "session_temperature_source": resolved_temperature_profile.get(
                        "session_temperature_source"
                    ),
                    "session_air_temperature_c": resolved_temperature_profile.get(
                        "session_air_temperature_c"
                    ),
                    "forecast_track_temperature_c": resolved_temperature_profile.get(
                        "forecast_track_temperature_c"
                    ),
                    "session_weight": resolved_temperature_profile.get("session_weight"),
                    "forecast_weight": resolved_temperature_profile.get("forecast_weight"),
                    "blend_enabled": bool(resolved_temperature_profile.get("blend_enabled", False)),
                }
            else:
                track_temperature_context = {
                    "track_temperature_c": float(resolved_track_temp),
                    "source": "legacy_scalar_resolver",
                    "reason": "legacy_temperature_resolver",
                    "weather_bucket": str(weather).strip().lower(),
                    "session_name": None,
                    "session_track_temperature_c": None,
                    "session_temperature_source": None,
                    "session_air_temperature_c": None,
                    "forecast_track_temperature_c": None,
                    "session_weight": None,
                    "forecast_weight": None,
                    "blend_enabled": False,
                }
        else:
            default_track_temp = {
                "dry": cfg.get("baseline_predictor.race.track_temperature.dry_c", 36.0),
                "mixed": cfg.get("baseline_predictor.race.track_temperature.mixed_c", 29.0),
                "rain": cfg.get("baseline_predictor.race.track_temperature.rain_c", 23.0),
            }
            fallback_track_temp = float(
                default_track_temp.get(str(weather).strip().lower(), default_track_temp["dry"])
            )
            race_params["track_temperature_c"] = fallback_track_temp
            track_temperature_context = {
                "track_temperature_c": fallback_track_temp,
                "source": "forecast_fallback",
                "reason": "no_temperature_signal",
                "weather_bucket": str(weather).strip().lower(),
                "session_name": None,
                "session_track_temperature_c": None,
                "session_temperature_source": None,
                "session_air_temperature_c": None,
                "forecast_track_temperature_c": fallback_track_temp,
                "session_weight": 0.0,
                "forecast_weight": 1.0,
                "blend_enabled": False,
            }
    else:
        existing_temp = float(race_params["track_temperature_c"])
        track_temperature_context = {
            "track_temperature_c": existing_temp,
            "source": "track_params_override",
            "reason": "track_params_override",
            "weather_bucket": str(weather).strip().lower(),
            "session_name": None,
            "session_track_temperature_c": None,
            "session_temperature_source": None,
            "session_air_temperature_c": None,
            "forecast_track_temperature_c": None,
            "session_weight": None,
            "forecast_weight": None,
            "blend_enabled": False,
        }

    weather_feature_modifiers: dict[str, float] = {
        "chaos_multiplier": 1.0,
        "teammate_variance_multiplier": 1.0,
        "confidence_adjustment": 0.0,
    }
    weather_feature_context: dict[str, Any] = {
        "available": False,
        "source_session": None,
        "selected_weather": str(weather).strip().lower(),
        "practice_weather_bucket": "unknown",
        "track_temperature_c": None,
        "air_temperature_c": None,
        "wind_speed_kph": None,
        "humidity_pct": None,
        "rainfall_signal": None,
        "weather_mismatch_score": 0.0,
        "chaos_multiplier": 1.0,
        "teammate_variance_multiplier": 1.0,
        "confidence_adjustment": 0.0,
    }
    if callable(resolve_non_competitive_weather_features):
        raw_weather_features: dict[str, Any] | None = None
        try:
            candidate_features = resolve_non_competitive_weather_features(
                year=year,
                race_name=race_name,
                is_sprint=is_sprint,
            )
        except TypeError:
            try:
                candidate_features = resolve_non_competitive_weather_features(
                    year,
                    race_name,
                    is_sprint,
                )
            except Exception:
                candidate_features = None
        except Exception:
            candidate_features = None

        if isinstance(candidate_features, dict):
            raw_weather_features = dict(candidate_features)

        if raw_weather_features and raw_weather_features.get("available"):
            weather_feature_modifiers, weather_feature_context = _build_weather_feature_context(
                selected_weather=weather,
                raw_features=raw_weather_features,
                cfg=cfg,
            )
        elif raw_weather_features:
            weather_feature_context = {
                **weather_feature_context,
                "available": False,
                "source_session": raw_weather_features.get("source_session"),
                "practice_weather_bucket": str(
                    raw_weather_features.get("practice_weather_bucket", "unknown")
                ),
            }

    race_params["weather_feature_modifiers"] = weather_feature_modifiers

    sc_weather_key = "sc_base_prob_wet" if weather in ["rain", "mixed"] else "sc_base_prob_dry"
    default_sc_probability = race_params.get(sc_weather_key, 0.45) + (
        race_params["track_overtaking"] * race_params.get("sc_track_modifier", 0.25)
    )
    race_params["sc_probability"] = race_params.get(
        "sc_probability", np.clip(default_sc_probability, 0.0, 1.0)
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
        sprint_compound = (
            "SOFT"
            if "SOFT" in available_compounds
            else (available_compounds[0] if available_compounds else "MEDIUM")
        )
        for driver in simulation_driver_info_map.keys():
            if is_sprint:
                strategies[driver] = {
                    "num_stops": 0,
                    "pit_laps": [],
                    "compound_sequence": [sprint_compound],
                    "stint_lengths": [race_distance],
                }
            else:
                driver_info = simulation_driver_info_map.get(driver, {})
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

        sim_result = simulate_race_lap_by_lap(
            driver_info_map=simulation_driver_info_map,
            strategies=strategies,
            race_params=race_params,
            race_distance=race_distance,
            weather=weather,
            rng=rng,
        )

        simulation_results.append(sim_result)

    aggregated = aggregate_simulation_results(simulation_results)

    confidence_floor = cfg.get("baseline_predictor.race.confidence.min", 40.0)
    confidence_cap = cfg.get("baseline_predictor.race.confidence.max", 60.0)
    context_confidence_scale = float(
        cfg.get("baseline_predictor.race.confidence.context_scale", 8.0)
    )
    context_confidence = (
        0.5 if input_confidence is None else float(np.clip(input_confidence, 0.0, 1.0))
    )
    low_confidence_share = float(np.clip(1.0 - context_confidence, 0.0, 1.0))
    context_confidence_adjustment = (context_confidence - 0.5) * context_confidence_scale
    weather_penalty = (
        cfg.get("baseline_predictor.race.confidence.weather_penalty_wet", 4.0)
        if weather in ("rain", "mixed")
        else 0.0
    )
    weather_penalty += float(weather_feature_modifiers.get("confidence_adjustment", 0.0))

    track_overtaking = race_params.get("track_overtaking", 0.5)
    grid_anchor_weight = np.clip(
        cfg.get("baseline_predictor.race.grid_anchor.base", 0.25)
        + (track_overtaking * cfg.get("baseline_predictor.race.grid_anchor.track_scale", 0.30)),
        0.10,
        0.80,
    )
    grid_anchor_min = cfg.get("baseline_predictor.race.grid_anchor.min", 0.35)
    main_grid_anchor_max = cfg.get("baseline_predictor.race.grid_anchor.main_max", 0.58)
    sprint_grid_anchor_min = cfg.get("baseline_predictor.race.grid_anchor.sprint_min", 0.78)
    if is_sprint:
        grid_anchor_weight = max(grid_anchor_weight, sprint_grid_anchor_min)
    else:
        grid_anchor_weight = max(grid_anchor_weight, grid_anchor_min)
        grid_anchor_weight = min(grid_anchor_weight, main_grid_anchor_max)
        # Keep low-signal runs closer to grid order to avoid overconfident reshuffles.
        grid_anchor_low_confidence_scale = float(
            cfg.get("baseline_predictor.race.grid_anchor.low_confidence_scale", 0.18)
        )
        grid_anchor_weight = float(
            np.clip(
                grid_anchor_weight + (low_confidence_share * grid_anchor_low_confidence_scale),
                grid_anchor_min,
                main_grid_anchor_max,
            )
        )
    overtake_blend_scale = cfg.get(
        "baseline_predictor.race.final_blend.overtaking_skill_scale", 1.6
    )
    race_adv_blend_scale = cfg.get("baseline_predictor.race.final_blend.race_advantage_scale", 1.3)
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
    low_confidence_racecraft_floor = float(
        cfg.get("baseline_predictor.race.final_blend.low_confidence_racecraft_floor", 0.72)
    )
    low_confidence_max_gain_floor = float(
        cfg.get("baseline_predictor.race.final_blend.low_confidence_max_gain_floor", 0.82)
    )
    racecraft_confidence_scale = float(
        np.clip(
            low_confidence_racecraft_floor
            + ((1.0 - low_confidence_racecraft_floor) * context_confidence),
            0.0,
            1.0,
        )
    )
    max_gain_confidence_scale = float(
        np.clip(
            low_confidence_max_gain_floor
            + ((1.0 - low_confidence_max_gain_floor) * context_confidence),
            0.0,
            1.0,
        )
    )

    finish_order = []
    blended_samples_by_driver: dict[str, list[float]] = {}
    team_to_drivers: dict[str, list[str]] = {}
    for driver_code, info in driver_info_map.items():
        team_to_drivers.setdefault(info["team"], []).append(driver_code)
    learning_position_scale = float(
        cfg.get("baseline_predictor.race.learning.position_adjustment_scale", 0.70)
    )
    grid_reference_positions = {
        driver_code: (
            float(np.mean(grid_position_samples_by_driver.get(driver_code, [])))
            if grid_position_samples_by_driver.get(driver_code)
            else float(info["grid_pos"])
        )
        for driver_code, info in driver_info_map.items()
    }

    for driver_code, median_pos in aggregated["median_positions"].items():
        info = driver_info_map[driver_code]
        positions = aggregated["position_distributions"][driver_code]
        reference_grid_pos = float(grid_reference_positions.get(driver_code, info["grid_pos"]))
        grid_position_samples = list(grid_position_samples_by_driver.get(driver_code, []))
        if len(grid_position_samples) != len(positions):
            grid_position_samples = [reference_grid_pos for _ in positions]

        position_std = np.std(positions)
        confidence = max(
            confidence_floor,
            min(
                confidence_cap,
                confidence_cap
                - (position_std * 3.0)
                - weather_penalty
                + context_confidence_adjustment,
            ),
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
        racecraft_adjustment *= racecraft_confidence_scale

        is_elite_driver = info["skill"] >= elite_skill_threshold
        if reference_grid_pos <= 3.0 and not is_elite_driver:
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
        max_gain *= max_gain_confidence_scale
        max_gain = np.clip(max_gain, max_gain_floor, max_gain_ceiling)
        min_position_score = max(1.0, reference_grid_pos - max_gain)

        position_blend_score = (
            ((1.0 - grid_anchor_weight) * median_pos)
            + (grid_anchor_weight * reference_grid_pos)
            - racecraft_adjustment
        )
        learned_position_adjustment = get_learned_position_adjustment(
            team=info["team"],
            driver=driver_code,
            teammates=team_to_drivers.get(info["team"], []),
            session="race",
        )
        position_blend_score -= learned_position_adjustment * learning_position_scale
        position_blend_score = max(position_blend_score, min_position_score)

        blended_position_samples = []
        for position_sample, grid_position_sample in zip(
            positions,
            grid_position_samples,
            strict=False,
        ):
            min_sample_position_score = max(1.0, float(grid_position_sample) - max_gain)
            blended_position_samples.append(
                max(
                    (
                        ((1.0 - grid_anchor_weight) * position_sample)
                        + (grid_anchor_weight * float(grid_position_sample))
                        - racecraft_adjustment
                    ),
                    min_sample_position_score,
                )
            )
        if learned_position_adjustment:
            blended_position_samples = [
                sample - (learned_position_adjustment * learning_position_scale)
                for sample in blended_position_samples
            ]
        blended_position_samples = [
            max(sample_position, min_position_score) for sample_position in blended_position_samples
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
                "podium_probability": 0.0,
                "dnf_probability": round(aggregated["dnf_rates"].get(driver_code, 0.0), 3),
            }
        )

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
                    ranked_scores = [
                        (driver_code, samples[sample_idx])
                        for driver_code, samples in blended_samples_by_driver.items()
                    ]
                else:
                    ranked_scores = [
                        (
                            driver_code,
                            samples[int(resample_rng.integers(0, sample_count))],
                        )
                        for driver_code, samples in blended_samples_by_driver.items()
                    ]

                ranked = sorted(ranked_scores, key=lambda item: (item[1], item[0]))
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

    finish_order.sort(key=lambda x: x["position_blend_score"])

    for i, item in enumerate(finish_order):
        item["position"] = i + 1

    if not is_sprint and blended_samples_by_driver:
        movement_floor = float(cfg.get("baseline_predictor.race.main_race_movement_floor", 1.0))
        movement_quantile = float(
            cfg.get("baseline_predictor.race.main_race_movement_quantile", 20.0)
        )
        movement_ceiling_base = float(
            cfg.get("baseline_predictor.race.main_race_movement_ceiling_base", 2.5)
        )
        movement_ceiling_track_scale = float(
            cfg.get("baseline_predictor.race.main_race_movement_ceiling_track_scale", 0.70)
        )
        movement_ceiling_min = float(
            cfg.get("baseline_predictor.race.main_race_movement_ceiling_min", movement_floor)
        )
        movement_ceiling = max(
            movement_ceiling_min,
            movement_ceiling_base
            - (float(np.clip(track_overtaking, 0.0, 1.0)) * movement_ceiling_track_scale),
        )

        def _avg_grid_change(rows: list[dict[str, Any]]) -> float:
            total_grid_change = 0.0
            total_drivers = 0
            for row in rows:
                info = driver_info_map.get(row["driver"])
                if info is None:
                    continue
                reference_grid_pos = float(
                    grid_reference_positions.get(row["driver"], info["grid_pos"])
                )
                total_grid_change += abs(float(row["position"]) - reference_grid_pos)
                total_drivers += 1
            return (total_grid_change / total_drivers) if total_drivers else 0.0

        def _apply_score_ranking(scores: dict[str, float]) -> None:
            finish_order.sort(
                key=lambda row: (
                    scores.get(row["driver"], float(row["position_blend_score"])),
                    float(row["position_blend_score"]),
                    row["driver"],
                )
            )
            for idx, row in enumerate(finish_order, start=1):
                row["position"] = idx
                if row["driver"] in scores:
                    row["position_blend_score"] = round(scores[row["driver"]], 4)

        def _avg_grid_change_for_scores(scores: dict[str, float]) -> float:
            total_grid_change = 0.0
            total_drivers = 0
            ranked_rows = sorted(
                finish_order,
                key=lambda row: (
                    scores.get(row["driver"], float(row["position_blend_score"])),
                    float(row["position_blend_score"]),
                    row["driver"],
                ),
            )
            for idx, row in enumerate(ranked_rows, start=1):
                info = driver_info_map.get(row["driver"])
                if info is None:
                    continue
                reference_grid_pos = float(
                    grid_reference_positions.get(row["driver"], info["grid_pos"])
                )
                total_grid_change += abs(float(idx) - reference_grid_pos)
                total_drivers += 1
            return (total_grid_change / total_drivers) if total_drivers else 0.0

        avg_grid_change = _avg_grid_change(finish_order)
        if avg_grid_change < movement_floor:
            quantile_candidates = [movement_quantile, 20.0, 10.0, 0.0]
            used_quantiles: set[float] = set()
            for quantile_candidate in quantile_candidates:
                quantile = float(np.clip(quantile_candidate, 0.0, 50.0))
                if quantile in used_quantiles:
                    continue
                used_quantiles.add(quantile)

                quantile_scores = {
                    driver: float(np.percentile(samples, quantile))
                    for driver, samples in blended_samples_by_driver.items()
                    if samples
                }
                _apply_score_ranking(quantile_scores)
                avg_grid_change = _avg_grid_change(finish_order)
                if avg_grid_change >= movement_floor:
                    break

            if avg_grid_change < movement_floor:
                sample_lengths = [
                    len(samples) for samples in blended_samples_by_driver.values() if samples
                ]
                sample_count = min(sample_lengths) if sample_lengths else 0
                if sample_count > 0:
                    selected_scores: dict[str, float] | None = None
                    selected_avg: float | None = None
                    strongest_scores: dict[str, float] | None = None
                    strongest_avg = -1.0

                    for sample_idx in range(sample_count):
                        candidate_scores = {
                            driver: float(samples[sample_idx])
                            for driver, samples in blended_samples_by_driver.items()
                            if len(samples) > sample_idx
                        }
                        candidate_avg = _avg_grid_change_for_scores(candidate_scores)
                        if candidate_avg > strongest_avg:
                            strongest_avg = candidate_avg
                            strongest_scores = candidate_scores
                        if candidate_avg >= movement_floor and (
                            selected_avg is None or candidate_avg < selected_avg
                        ):
                            selected_avg = candidate_avg
                            selected_scores = candidate_scores

                    final_scores = selected_scores or strongest_scores
                    if final_scores:
                        _apply_score_ranking(final_scores)

        avg_grid_change = _avg_grid_change(finish_order)
        if avg_grid_change > movement_ceiling:
            base_scores = {
                row["driver"]: float(row["position_blend_score"]) for row in finish_order
            }
            selected_ceiling_scores: dict[str, float] | None = None
            closest_scores: dict[str, float] | None = None
            closest_delta = float("inf")
            keep_factors = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30]

            for keep_factor in keep_factors:
                candidate_ceiling_scores: dict[str, float] = {}
                for row in finish_order:
                    driver_code = row["driver"]
                    info = driver_info_map.get(driver_code)
                    if info is None:
                        continue
                    reference_grid_pos = float(
                        grid_reference_positions.get(driver_code, info["grid_pos"])
                    )
                    candidate_ceiling_scores[driver_code] = (
                        keep_factor
                        * base_scores.get(driver_code, float(row["position_blend_score"]))
                    ) + ((1.0 - keep_factor) * reference_grid_pos)

                candidate_avg = _avg_grid_change_for_scores(candidate_ceiling_scores)
                if movement_floor <= candidate_avg <= movement_ceiling:
                    selected_ceiling_scores = candidate_ceiling_scores
                    break
                if movement_floor <= candidate_avg < avg_grid_change:
                    candidate_delta = abs(candidate_avg - movement_ceiling)
                    if candidate_delta < closest_delta:
                        closest_delta = candidate_delta
                        closest_scores = candidate_ceiling_scores

            final_scores = selected_ceiling_scores or closest_scores
            if final_scores:
                _apply_score_ranking(final_scores)

    if cfg.get("baseline_predictor.race.podium_probability.enforce_monotonic", True):
        raw_podium_values = [float(row.get("podium_probability", 0.0)) for row in finish_order]
        smoothed_values = enforce_non_increasing(raw_podium_values)
        for row, smoothed in zip(finish_order, smoothed_values, strict=True):
            row["podium_probability"] = round(float(np.clip(smoothed, 0.0, 100.0)), 1)

    _apply_low_confidence_interval_floor(
        finish_order=finish_order,
        input_confidence=input_confidence,
        cfg=cfg,
        field_size=max(1, len(validated_grid)),
    )

    return {
        "finish_order": finish_order,
        "characteristics_profile_used": "long_run",
        "teams_with_characteristics_profile": teams_with_long_profile,
        "compound_strategies": aggregated["compound_strategy_distribution"],
        "pit_lap_distribution": aggregated["pit_lap_distribution"],
        "track_temperature_context": track_temperature_context,
        "weather_feature_context": weather_feature_context,
    }
