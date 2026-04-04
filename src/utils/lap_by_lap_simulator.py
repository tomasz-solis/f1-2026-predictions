"""Lap-by-lap Monte Carlo race simulation for 2026-style F1 weekends.

Race pace in F1 is path-dependent. The same car can win comfortably in clean
air, get trapped behind traffic, or lose a race to one badly timed safety car.
That is why this model uses Monte Carlo rather than a neat analytical formula:
pit timing, lap-one variance, tire warm-up, safety car timing, and overtaking
windows all interact in ways that are hard to compress without losing the feel
of an actual Sunday.

The code intentionally keeps those moving parts explicit. Grid position matters,
but it is not destiny. Faster cars can pass if the active-aero window opens,
fresh tires create short-lived undercut opportunities, and a chaotic race
should look different from a straightforward one. The production predictor
aggregates many runs rather than trusting one simulated race, which is the same
reason teams run thousands of strategy scenarios before a grand prix.
"""

import logging
from typing import Any, cast

import numpy as np

from src.simulation.tire_degradation import (
    calculate_fuel_delta,
    calculate_tire_deg_delta,
    get_effective_tire_deg_slope,
    get_fresh_tire_advantage,
)
from src.simulation.traffic_model import (
    calculate_dirty_air_penalty,
    get_track_downforce_level,
)
from src.types.prediction_types import PitStrategy, RaceSimulationResult

logger = logging.getLogger(__name__)

# Internal ratios used to expand the compact overtake model.
_OVERTAKE_INTERNAL = {
    "pass_window_ratio": 0.67,
    "dirty_air_penalty_base": 0.05,
    "defense_ratio": 1.12,
    "race_adv_ratio": 0.80,
    "track_ease_ratio": 0.51,
    "dirty_air_track_ratio": 0.34,
    "pass_threshold_base": 0.06,
    "pass_threshold_track_ratio": 0.46,
    "pass_probability_sensitivity": 0.45,
    "pass_time_bonus_range": [0.08, 0.35],
}


def _expand_overtake_cfg(compact: dict[str, Any]) -> dict[str, Any]:
    """Expand 5 user-facing overtake params into the full internal set.

    The 5 exposed params and their defaults:
        dirty_air_window_s  (1.8)  – active aero / slipstream proximity window
        pace_weight         (0.55) – importance of raw pace delta
        racecraft_weight    (0.25) – combined attacker/defender skill weight
        track_factor        (0.35) – track influence on passing difficulty
        pass_chance_base    (0.30) – base pass probability when threshold met

    If callers still supply the old 11+ granular keys they are used directly
    for backward compatibility.
    """
    # If the legacy granular keys are present, pass through unchanged.
    if "pace_diff_scale" in compact or "skill_scale" in compact:
        return dict(compact)

    daw = compact.get("dirty_air_window_s", 1.8)
    pw = compact.get("pace_weight", 0.55)
    rw = compact.get("racecraft_weight", 0.25)
    tf = compact.get("track_factor", 0.35)
    pcb = compact.get("pass_chance_base", 0.30)

    c = _OVERTAKE_INTERNAL
    return {
        "dirty_air_window_s": daw,
        "dirty_air_penalty_base": c["dirty_air_penalty_base"],
        "dirty_air_penalty_track_scale": tf * c["dirty_air_track_ratio"],
        "pass_window_s": daw * c["pass_window_ratio"],
        "pace_diff_scale": pw,
        "skill_scale": rw,
        "defense_scale": rw * c["defense_ratio"],
        "race_adv_scale": rw * c["race_adv_ratio"],
        "track_ease_scale": tf * c["track_ease_ratio"],
        "pass_threshold_base": c["pass_threshold_base"],
        "pass_threshold_track_scale": tf * c["pass_threshold_track_ratio"],
        "pass_probability_base": pcb,
        "pass_probability_scale": c["pass_probability_sensitivity"],
        "pass_time_bonus_range": list(cast(list[float], c["pass_time_bonus_range"])),
        # Forward any zone-level overrides the caller may have set.
        **{k: v for k, v in compact.items() if k.startswith("zone_")},
    }


def _calculate_safety_car_lap_probability(
    sc_probability_race: float,
    eligible_laps: int,
) -> float:
    """Convert a race-level SC probability into a constant per-lap trigger chance."""
    probability = float(np.clip(sc_probability_race, 0.0, 1.0))
    if probability <= 0.0 or eligible_laps <= 0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return float(1.0 - (1.0 - probability) ** (1.0 / eligible_laps))


def simulate_race_lap_by_lap(
    driver_info_map: dict[str, dict[str, Any]],
    strategies: dict[str, PitStrategy],
    race_params: dict[str, Any],
    race_distance: int,
    weather: str,
    rng: np.random.Generator,
) -> RaceSimulationResult:
    """Simulate one race iteration lap-by-lap, return finish order and metadata.

    Returns dict with:
        - finish_order: List[str] (driver codes in finish order)
        - dnf_drivers: List[str] (drivers who did not finish)
        - strategies_used: Dict[str, Dict] (strategy per driver)
    """
    # Expand compact overtake config once before the lap loop.
    race_params = dict(race_params)
    race_params["overtake_model"] = _expand_overtake_cfg(race_params.get("overtake_model", {}))
    track_temperature_c = race_params.get("track_temperature_c")
    weather_feature_modifiers = race_params.get("weather_feature_modifiers", {})
    chaos_multiplier = float(
        np.clip(
            weather_feature_modifiers.get("chaos_multiplier", 1.0),
            0.80,
            1.40,
        )
    )
    teammate_variance_multiplier = float(
        np.clip(
            weather_feature_modifiers.get("teammate_variance_multiplier", 1.0),
            0.80,
            1.35,
        )
    )
    teammate_variance_std = max(0.0, float(race_params.get("teammate_variance_std", 0.15)))
    teammate_setup_offset_ratio = float(
        np.clip(race_params.get("teammate_setup_offset_ratio", 0.30), 0.0, 1.0)
    )
    teammate_lap_variance_ratio = float(
        np.clip(race_params.get("teammate_variance_lap_ratio", 0.45), 0.0, 1.0)
    )
    teammate_setup_offset_std = (
        teammate_variance_std * teammate_setup_offset_ratio * teammate_variance_multiplier
    )
    teammate_lap_variance_std = (
        teammate_variance_std * teammate_lap_variance_ratio * teammate_variance_multiplier
    )

    team_to_drivers: dict[str, list[str]] = {}
    for driver, info in driver_info_map.items():
        team_to_drivers.setdefault(str(info.get("team", "")), []).append(driver)

    persistent_setup_offset_by_driver: dict[str, float] = {}
    for teammates in team_to_drivers.values():
        if teammate_setup_offset_std <= 0.0 or len(teammates) <= 1:
            for driver in teammates:
                persistent_setup_offset_by_driver[driver] = 0.0
            continue

        raw_offsets = {
            driver: float(rng.normal(0.0, teammate_setup_offset_std)) for driver in teammates
        }
        team_mean_offset = float(np.mean(list(raw_offsets.values())))
        for driver, raw_offset in raw_offsets.items():
            persistent_setup_offset_by_driver[driver] = raw_offset - team_mean_offset

    # Initialize driver states
    start_grid_gap_seconds = race_params.get("start_grid_gap_seconds", 0.32)
    safety_car_trigger_lap = race_params.get("safety_car_trigger_lap", 10)
    sc_probability_race = float(np.clip(race_params.get("sc_probability", 0.0), 0.0, 1.0))
    eligible_sc_laps = max(0, race_distance - safety_car_trigger_lap)
    sc_lap_probability = _calculate_safety_car_lap_probability(
        sc_probability_race=sc_probability_race,
        eligible_laps=eligible_sc_laps,
    )
    driver_states = {}
    for driver, info in driver_info_map.items():
        driver_states[driver] = {
            "position": info["grid_pos"],
            # Preserve qualifying order at lights-out; pace then decides who can move.
            "cumulative_time": max(0.0, (info["grid_pos"] - 1) * start_grid_gap_seconds),
            "current_compound": strategies[driver]["compound_sequence"][0],
            "laps_on_tire": 0,
            "stint_number": 1,
            "fuel_load": race_params["fuel"]["initial_load_kg"],
            "has_dnf": False,
            "base_pace": 90.0,  # Will be calculated on first lap
            "teammate_setup_offset": persistent_setup_offset_by_driver.get(driver, 0.0),
        }

    # Pre-extract constant lap-time parameters (unchanged lap-to-lap).
    _lt_cfg = race_params.get("lap_time", {})
    _reference_base = _lt_cfg.get("reference_base", 90.0)
    _team_pace_penalty_range = _lt_cfg.get("team_pace_penalty_range", 5.0)
    _skill_improvement_max = _lt_cfg.get("skill_improvement_max", 0.75)
    _elite_skill_threshold = _lt_cfg.get("elite_skill_threshold", 0.88)
    _elite_skill_lap_bonus_max = _lt_cfg.get("elite_skill_lap_bonus_max", 0.09)
    _elite_skill_exponent = _lt_cfg.get("elite_skill_exponent", 1.3)
    _team_strength_compression = race_params.get("team_strength_compression", 0.35)
    _race_advantage_lap_impact = race_params.get("race_advantage_lap_impact", 0.35)
    _elite_denominator = max(1e-6, 1.0 - _elite_skill_threshold)
    _lap_time_bounds = _lt_cfg.get("bounds", [70.0, 120.0])

    # Lap-by-lap progression
    for lap_num in range(1, race_distance + 1):
        active_order = sorted(
            (
                (driver, state["position"])
                for driver, state in driver_states.items()
                if not state["has_dnf"]
            ),
            key=lambda item: item[1],
        )
        driver_ahead_map = {
            active_order[idx][0]: active_order[idx - 1][0] for idx in range(1, len(active_order))
        }
        sc_deployed_this_lap = (
            lap_num > safety_car_trigger_lap and rng.random() < sc_lap_probability
        )

        for driver in list(driver_states.keys()):
            state = driver_states[driver]
            info = driver_info_map[driver]

            # Skip DNF drivers
            if state["has_dnf"]:
                continue

            if rng.random() < info["dnf_probability"] / race_distance:
                state["has_dnf"] = True
                state["dnf_lap"] = lap_num
                logger.debug("%s DNF on lap %s", driver, lap_num)
                continue

            compound = state["current_compound"]
            laps_on_tire = state["laps_on_tire"]
            fuel_load = state["fuel_load"]

            team_strength = info["team_strength_by_compound"].get(compound, info["team_strength"])
            skill = info["skill"]

            # Base lap time from team strength (inverted: 1.0 = fastest, 0.0 = slowest)
            reference_base = _reference_base
            team_pace_penalty_range = _team_pace_penalty_range
            skill_improvement_max = _skill_improvement_max
            team_strength_compression = _team_strength_compression

            compressed_team_strength = 0.5 + ((team_strength - 0.5) * team_strength_compression)
            compressed_team_strength = np.clip(compressed_team_strength, 0.0, 1.0)

            team_pace_penalty = (1.0 - compressed_team_strength) * team_pace_penalty_range
            skill_improvement = skill * skill_improvement_max
            elite_skill_threshold = _elite_skill_threshold
            elite_skill_lap_bonus_max = _elite_skill_lap_bonus_max
            elite_skill_exponent = _elite_skill_exponent
            elite_denominator = _elite_denominator
            elite_skill_normalized = max(0.0, (skill - elite_skill_threshold) / elite_denominator)
            elite_skill_bonus = elite_skill_lap_bonus_max * (
                elite_skill_normalized**elite_skill_exponent
            )

            # Reference lap time (track-specific if available in race_params)
            race_advantage_lap_impact = _race_advantage_lap_impact
            race_advantage_delta = -info.get("race_advantage", 0.0) * race_advantage_lap_impact
            base_lap_time = (
                reference_base
                + team_pace_penalty
                - skill_improvement
                - elite_skill_bonus
                + race_advantage_delta
            )

            # Cache base pace (used for overtake opportunity modeling)
            state["base_pace"] = base_lap_time

            tire_deg_slope = info["tire_deg_by_compound"].get(compound, 0.15)

            # Adjust deg slope for traffic/dirty air
            effective_tire_deg_slope = get_effective_tire_deg_slope(
                base_tire_deg_slope=tire_deg_slope,
                traffic_position=state["position"],
                total_cars=len(driver_states),
            )

            tire_deg_delta = calculate_tire_deg_delta(
                tire_deg_slope=effective_tire_deg_slope,
                laps_on_tire=laps_on_tire,
                fuel_load_kg=fuel_load,
                initial_fuel_kg=race_params["fuel"]["initial_load_kg"],
                compound=compound,
                track_temp=track_temperature_c,
            )

            fresh_tire_bonus = get_fresh_tire_advantage(
                compound=compound,
                laps_on_tire=laps_on_tire,
                track_temp=track_temperature_c,
            )

            fuel_delta = calculate_fuel_delta(
                laps_remaining=(race_distance - lap_num),
                fuel_effect_per_lap=race_params["fuel"]["effect_per_lap"],
            )

            chaos = 0.0

            # Lap 1 chaos (incidents, battles) — with track-specific risk modifier
            if lap_num == 1:
                chaos += _get_lap1_chaos(state["position"], race_params, rng)

            # Base chaos (weather-dependent unpredictability)
            base_chaos_std = _resolve_base_chaos_std(race_params, weather)
            chaos += rng.normal(0, base_chaos_std * chaos_multiplier)

            # Track-specific chaos (overtaking difficulty)
            # Harder tracks = less chaos (positions more stable)
            if "track_overtaking" in race_params:
                track_chaos_factor = race_params.get("track_chaos_multiplier", 0.4)
                track_multiplier = 1.0 - (race_params["track_overtaking"] * track_chaos_factor)
                chaos *= track_multiplier

            sc_luck = 0.0
            if sc_deployed_this_lap:
                sc_luck_range = race_params.get("safety_car_luck_range", 0.25)
                sc_luck = rng.uniform(-sc_luck_range, sc_luck_range)

            teammate_variance = state.get("teammate_setup_offset", 0.0)
            if teammate_lap_variance_std > 0.0:
                teammate_variance += float(rng.normal(0.0, teammate_lap_variance_std))

            traffic_overtake_effect = _get_traffic_overtake_effect(
                driver=driver,
                driver_states=driver_states,
                driver_info_map=driver_info_map,
                driver_ahead_map=driver_ahead_map,
                race_params=race_params,
                rng=rng,
            )

            lap_time = (
                base_lap_time
                + tire_deg_delta
                - fresh_tire_bonus
                + fuel_delta
                + chaos
                + sc_luck
                + teammate_variance
                + traffic_overtake_effect
            )

            # Keep lap time within plausible bounds.
            lap_time_bounds = _lap_time_bounds
            lap_time = max(lap_time_bounds[0], min(lap_time_bounds[1], lap_time))

            # Update cumulative time and tire age
            state["cumulative_time"] += lap_time
            state["laps_on_tire"] += 1

            # Fuel burn (configurable)
            fuel_burn_rate = race_params.get("fuel", {}).get("burn_rate_kg_per_lap", 1.5)
            state["fuel_load"] = max(0.0, state["fuel_load"] - fuel_burn_rate)

            strategy = strategies[driver]
            if lap_num in strategy["pit_laps"]:
                _apply_pit_stop(state, strategy, race_params, rng)

        # Update positions based on cumulative time (after all drivers complete lap)
        _update_positions_from_times(driver_states)

    # Generate finish order and metadata
    return _generate_race_result(driver_states, strategies)


def _resolve_base_chaos_std(race_params: dict[str, Any], weather: str) -> float:
    """Resolve weather-specific chaos with explicit handling for mixed conditions."""
    base_chaos_cfg = race_params.get("base_chaos", {})
    dry_std = float(base_chaos_cfg.get("dry", 0.35))
    wet_std = float(base_chaos_cfg.get("wet", 0.45))

    mixed_std = base_chaos_cfg.get("mixed")
    if mixed_std is None:
        mixed_blend = float(np.clip(race_params.get("mixed_weather_chaos_blend", 0.55), 0.0, 1.0))
        mixed_std = dry_std + ((wet_std - dry_std) * mixed_blend)

    weather_key = str(weather).strip().lower()
    if weather_key in {"wet", "rain"}:
        return wet_std
    if weather_key == "mixed":
        return float(mixed_std)
    return dry_std


def _get_traffic_overtake_effect(
    driver: str,
    driver_states: dict[str, dict[str, Any]],
    driver_info_map: dict[str, dict[str, Any]],
    driver_ahead_map: dict[str, str],
    race_params: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """Return lap-time delta from traffic and overtake attempts.

    Positive values are time losses (dirty air), negative values are gains
    from successful overtakes.
    """
    ahead_driver = driver_ahead_map.get(driver)
    if ahead_driver is None:
        return 0.0  # Leader: clean air

    state = driver_states[driver]
    ahead_state = driver_states[ahead_driver]
    if ahead_state.get("has_dnf", False):
        return 0.0

    gap_to_ahead = max(0.0, state["cumulative_time"] - ahead_state["cumulative_time"])
    track_overtaking = race_params.get("track_overtaking", 0.5)
    overtake_cfg = race_params.get("overtake_model", {})

    dirty_air_window = overtake_cfg.get("dirty_air_window_s", 1.8)
    if gap_to_ahead > dirty_air_window:
        return 0.0

    info = driver_info_map[driver]
    ahead_info = driver_info_map.get(ahead_driver, {})
    dirty_air_penalty_base = overtake_cfg.get("dirty_air_penalty_base", 0.05)
    dirty_air_penalty_track_scale = overtake_cfg.get("dirty_air_penalty_track_scale", 0.12)
    dirty_air_cap = dirty_air_penalty_base + (track_overtaking * dirty_air_penalty_track_scale)

    if dirty_air_cap <= 0.0:
        dirty_air_penalty = 0.0
    else:
        track_name = race_params.get("track_name")
        track_downforce_level = get_track_downforce_level(
            track_name=track_name,
            track_overtaking=track_overtaking,
        )
        dirty_air_penalty = min(
            dirty_air_cap,
            calculate_dirty_air_penalty(
                gap_to_car_ahead_s=gap_to_ahead,
                track_downforce_level=track_downforce_level,
                dirty_air_window_s=dirty_air_window,
            ),
        )

    dirty_air_relief = np.clip(info.get("overtaking_skill", 0.5), 0.0, 1.0) * 0.5
    dirty_air_penalty *= 1.0 - dirty_air_relief

    effect = dirty_air_penalty

    pass_window = overtake_cfg.get("pass_window_s", 1.2)
    if gap_to_ahead > pass_window:
        return effect

    pace_diff_scale = overtake_cfg.get("pace_diff_scale", 0.55)
    skill_scale = overtake_cfg.get("skill_scale", 0.25)
    defense_scale = overtake_cfg.get("defense_scale", 0.28)
    race_adv_scale = overtake_cfg.get("race_adv_scale", 0.20)
    track_ease_scale = overtake_cfg.get("track_ease_scale", 0.18)
    defender_skill = np.clip(
        ahead_info.get("defensive_skill", ahead_info.get("skill", 0.5)),
        0.0,
        1.0,
    )

    pace_delta_to_ahead = ahead_state.get("base_pace", 90.0) - state.get("base_pace", 90.0)
    overtake_score = (
        (pace_delta_to_ahead * pace_diff_scale)
        + ((info.get("overtaking_skill", 0.5) - 0.5) * skill_scale)
        - ((defender_skill - 0.5) * defense_scale)
        + (info.get("race_advantage", 0.0) * race_adv_scale)
        + ((1.0 - track_overtaking) * track_ease_scale)
    )

    target_position = int(ahead_state.get("position", 22))
    (
        zone_threshold_boost,
        zone_probability_scale,
        zone_bonus_scale,
    ) = _get_overtake_zone_adjustments(
        target_position=target_position,
        overtake_cfg=overtake_cfg,
    )

    pass_threshold = overtake_cfg.get("pass_threshold_base", 0.06) + (
        track_overtaking * overtake_cfg.get("pass_threshold_track_scale", 0.16)
    )
    pass_threshold += zone_threshold_boost
    if overtake_score <= pass_threshold:
        return effect

    pass_probability = overtake_cfg.get("pass_probability_base", 0.30) + (
        (overtake_score - pass_threshold) * overtake_cfg.get("pass_probability_scale", 0.45)
    )
    pass_probability *= zone_probability_scale
    pass_probability = np.clip(pass_probability, 0.05, 0.95)

    if rng.random() < pass_probability:
        bonus_range = overtake_cfg.get("pass_time_bonus_range", [0.08, 0.35])
        if not isinstance(bonus_range, list) or len(bonus_range) != 2:
            bonus_range = [0.08, 0.35]
        pass_bonus = rng.uniform(bonus_range[0], bonus_range[1]) * zone_bonus_scale
        effect -= pass_bonus

    return effect


def _get_overtake_zone_adjustments(
    target_position: int, overtake_cfg: dict[str, Any]
) -> tuple[float, float, float]:
    """Scale overtake threshold/probability/benefit by target's position zone.

    Overtakes at the front are harder and lower reward; backfield passes are easier.
    """
    if target_position <= 3:
        return (
            overtake_cfg.get("zone_front_threshold_boost", 0.22),
            overtake_cfg.get("zone_front_probability_scale", 0.55),
            overtake_cfg.get("zone_front_bonus_scale", 0.55),
        )
    if target_position <= 10:
        return (
            overtake_cfg.get("zone_upper_threshold_boost", 0.10),
            overtake_cfg.get("zone_upper_probability_scale", 0.75),
            overtake_cfg.get("zone_upper_bonus_scale", 0.78),
        )
    if target_position <= 15:
        return (
            overtake_cfg.get("zone_mid_threshold_boost", 0.02),
            overtake_cfg.get("zone_mid_probability_scale", 0.92),
            overtake_cfg.get("zone_mid_bonus_scale", 0.93),
        )
    return (
        overtake_cfg.get("zone_back_threshold_boost", -0.03),
        overtake_cfg.get("zone_back_probability_scale", 1.08),
        overtake_cfg.get("zone_back_bonus_scale", 1.05),
    )


def _get_lap1_chaos(
    position: int,
    race_params: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """Calculate lap 1 chaos based on grid position and track-specific risk."""
    lap1_config = race_params.get("lap1_chaos", {})

    if position <= 3:
        std = lap1_config.get("front_row", 0.15)
    elif position <= 10:
        std = lap1_config.get("upper_midfield", 0.32)
    elif position <= 15:
        std = lap1_config.get("midfield", 0.38)
    else:
        std = lap1_config.get("back_field", 0.28)

    # Track-specific lap-1 risk modifier (street circuits, narrow tracks, etc.)
    lap1_risk_modifier = race_params.get("lap1_risk_modifier", 0.0)
    std *= 1.0 + lap1_risk_modifier

    return rng.normal(0, std)


def _apply_pit_stop(
    state: dict[str, Any],
    strategy: PitStrategy,
    race_params: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Apply pit stop time loss and compound change to driver state."""
    # Base pit loss
    pit_loss = race_params["pit_stops"]["loss_duration"]

    # Optional: overtake loss if unlucky timing
    overtake_loss_range = race_params["pit_stops"].get("overtake_loss_range", [0, 3])
    overtake_loss = rng.uniform(overtake_loss_range[0], overtake_loss_range[1])

    total_pit_loss = pit_loss + overtake_loss

    # Add pit loss to cumulative time
    state["cumulative_time"] += total_pit_loss

    # Change compound
    state["stint_number"] += 1
    stint_idx = state["stint_number"] - 1

    if stint_idx < len(strategy["compound_sequence"]):
        new_compound = strategy["compound_sequence"][stint_idx]
        state["current_compound"] = new_compound
        state["laps_on_tire"] = 0  # Fresh tires

        logger.debug(
            "Pit stop: %s → %s (+%ss)",
            state.get("driver", "unknown"),
            new_compound,
            format(total_pit_loss, ".2f"),
        )
    else:
        logger.warning(
            "Stint number %s exceeds compound sequence length %s",
            state["stint_number"],
            len(strategy["compound_sequence"]),
        )


def _update_positions_from_times(driver_states: dict[str, dict[str, Any]]) -> None:
    """Update positions based on cumulative race time.

    Drivers with lower cumulative time get better positions.
    DNF drivers are placed at the end.
    """
    # Separate active and DNF drivers
    active_drivers = []
    dnf_drivers = []

    for driver, state in driver_states.items():
        if state["has_dnf"]:
            dnf_drivers.append((driver, state.get("dnf_lap", 999)))
        else:
            active_drivers.append((driver, state["cumulative_time"]))

    # Sort active drivers by cumulative time (ascending)
    active_drivers.sort(key=lambda x: x[1])

    # Sort DNF drivers by lap they DNF'd (later DNF = better position).
    dnf_drivers.sort(key=lambda x: x[1], reverse=True)

    # Assign positions
    position = 1
    for driver, _ in active_drivers:
        driver_states[driver]["position"] = position
        position += 1

    for driver, _ in dnf_drivers:
        driver_states[driver]["position"] = position
        position += 1


def _generate_race_result(
    driver_states: dict[str, dict[str, Any]],
    strategies: dict[str, PitStrategy],
) -> RaceSimulationResult:
    """Generate final race result dict from driver states."""
    # Sort drivers by position
    sorted_drivers = sorted(driver_states.items(), key=lambda x: x[1]["position"])

    finish_order = [driver for driver, state in sorted_drivers]
    dnf_drivers = [driver for driver, state in sorted_drivers if state["has_dnf"]]

    return {
        "finish_order": finish_order,
        "dnf_drivers": dnf_drivers,
        "strategies_used": strategies,
    }


def aggregate_simulation_results(
    simulation_results: list[RaceSimulationResult],
) -> dict[str, Any]:
    """Aggregate results from multiple simulations.

    Returns dict with:
        - median_positions: Dict[str, int] (driver → median finish position)
        - position_distributions: Dict[str, List[int]] (driver → all positions)
        - dnf_rates: Dict[str, float] (driver → % of sims with DNF)
        - compound_strategy_distribution: Dict[str, float] (strategy → frequency)
        - pit_lap_distribution: Dict[str, int] (lap bin → count)
    """
    from collections import defaultdict

    position_data: defaultdict[str, list[int]] = defaultdict(list)
    dnf_counts: defaultdict[str, int] = defaultdict(int)
    strategy_counts: defaultdict[str, int] = defaultdict(int)
    pit_lap_counts: defaultdict[str, int] = defaultdict(int)

    total_simulations = len(simulation_results)

    for result in simulation_results:
        finish_order = result["finish_order"]
        dnf_drivers = result.get("dnf_drivers", [])
        strategies = result.get("strategies_used", {})

        # Collect position data
        for position, driver in enumerate(finish_order, start=1):
            position_data[driver].append(position)

        # Collect DNF data
        for driver in dnf_drivers:
            dnf_counts[driver] += 1

        # Collect strategy data
        for _driver, strategy in strategies.items():
            sequence = "→".join(strategy["compound_sequence"])
            strategy_counts[sequence] += 1

            # Collect pit lap data (binned into 5-lap windows)
            for pit_lap in strategy.get("pit_laps", []):
                bin_start = (pit_lap // 5) * 5
                bin_label = f"lap_{bin_start}-{bin_start + 5}"
                pit_lap_counts[bin_label] += 1

    # Calculate medians
    median_positions = {
        driver: int(np.median(positions)) for driver, positions in position_data.items()
    }

    # Calculate DNF rates
    dnf_rates = {driver: count / total_simulations for driver, count in dnf_counts.items()}

    # Convert strategy counts to percentages
    total_strategy_count = sum(strategy_counts.values())
    compound_strategy_distribution = (
        {strategy: count / total_strategy_count for strategy, count in strategy_counts.items()}
        if total_strategy_count > 0
        else {}
    )

    # Pit lap distribution
    pit_lap_distribution = dict(pit_lap_counts)

    return {
        "median_positions": median_positions,
        "position_distributions": dict(position_data),
        "dnf_rates": dnf_rates,
        "compound_strategy_distribution": compound_strategy_distribution,
        "pit_lap_distribution": pit_lap_distribution,
    }
