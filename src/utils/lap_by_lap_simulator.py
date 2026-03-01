"""Lap-by-lap race simulation engine with tire degradation and pit stops."""

import logging
from typing import Any

import numpy as np

from src.types.prediction_types import PitStrategy, RaceSimulationResult
from src.utils.tire_degradation import (
    calculate_fuel_delta,
    calculate_tire_deg_delta,
    get_effective_tire_deg_slope,
    get_fresh_tire_advantage,
)
from src.utils.traffic_model import (
    calculate_dirty_air_penalty,
    get_track_downforce_level,
)

logger = logging.getLogger(__name__)


def simulate_race_lap_by_lap(
    driver_info_map: dict[str, dict[str, Any]],
    strategies: dict[str, PitStrategy],
    race_params: dict,
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
    # Initialize driver states
    start_grid_gap_seconds = race_params.get("start_grid_gap_seconds", 0.32)
    safety_car_trigger_lap = race_params.get("safety_car_trigger_lap", 10)
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
        }

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
        # Treat safety-car probability as a race-level likelihood; convert to
        # per-lap trigger chance to avoid over-applying random swings.
        sc_probability_race = np.clip(race_params.get("sc_probability", 0.0), 0.0, 1.0)
        sc_laps_remaining = max(1, race_distance - safety_car_trigger_lap)
        sc_lap_probability = sc_probability_race / sc_laps_remaining
        sc_deployed_this_lap = (
            lap_num > safety_car_trigger_lap and rng.random() < sc_lap_probability
        )

        for driver in list(driver_states.keys()):
            state = driver_states[driver]
            info = driver_info_map[driver]

            # Skip DNF drivers
            if state["has_dnf"]:
                continue

            # 1. Check DNF probability (distributed across race)
            if rng.random() < info["dnf_probability"] / race_distance:
                state["has_dnf"] = True
                state["dnf_lap"] = lap_num
                logger.debug(f"{driver} DNF on lap {lap_num}")
                continue

            # 2. Get current compound and tire age
            compound = state["current_compound"]
            laps_on_tire = state["laps_on_tire"]
            fuel_load = state["fuel_load"]

            # 3. Calculate base pace (team strength for this compound + skill)
            team_strength = info["team_strength_by_compound"].get(compound, info["team_strength"])
            skill = info["skill"]

            # Base lap time from team strength (inverted: 1.0 = fastest, 0.0 = slowest)
            # Load lap time modeling config
            reference_base = race_params.get("lap_time", {}).get("reference_base", 90.0)
            team_pace_penalty_range = race_params.get("lap_time", {}).get(
                "team_pace_penalty_range", 5.0
            )
            skill_improvement_max = race_params.get("lap_time", {}).get(
                "skill_improvement_max", 0.5
            )
            team_strength_compression = race_params.get("team_strength_compression", 0.45)

            compressed_team_strength = 0.5 + ((team_strength - 0.5) * team_strength_compression)
            compressed_team_strength = np.clip(compressed_team_strength, 0.0, 1.0)

            team_pace_penalty = (1.0 - compressed_team_strength) * team_pace_penalty_range
            skill_improvement = skill * skill_improvement_max
            elite_skill_threshold = race_params.get("lap_time", {}).get(
                "elite_skill_threshold", 0.88
            )
            elite_skill_lap_bonus_max = race_params.get("lap_time", {}).get(
                "elite_skill_lap_bonus_max", 0.09
            )
            elite_skill_exponent = race_params.get("lap_time", {}).get("elite_skill_exponent", 1.3)
            elite_denominator = max(1e-6, 1.0 - elite_skill_threshold)
            elite_skill_normalized = max(0.0, (skill - elite_skill_threshold) / elite_denominator)
            elite_skill_bonus = elite_skill_lap_bonus_max * (
                elite_skill_normalized**elite_skill_exponent
            )

            # Reference lap time (track-specific if available in race_params)
            race_advantage_lap_impact = race_params.get("race_advantage_lap_impact", 0.35)
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

            # 4. Apply tire degradation
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
            )

            # 5. Apply fresh tire advantage (negative delta = faster)
            fresh_tire_bonus = get_fresh_tire_advantage(
                compound=compound,
                laps_on_tire=laps_on_tire,
            )

            # 6. Apply fuel effect
            fuel_delta = calculate_fuel_delta(
                laps_remaining=(race_distance - lap_num),
                fuel_effect_per_lap=race_params["fuel"]["effect_per_lap"],
            )

            # 7. Apply chaos factors
            chaos = 0.0

            # Lap 1 chaos (incidents, battles)
            if lap_num == 1:
                chaos += _get_lap1_chaos(state["position"], race_params, rng)

            # Base chaos (weather-dependent unpredictability)
            weather_key = "wet" if str(weather).lower() in {"wet", "rain", "mixed"} else "dry"
            base_chaos_std = race_params["base_chaos"][weather_key]
            chaos += rng.normal(0, base_chaos_std)

            # Track-specific chaos (overtaking difficulty)
            # Harder tracks = less chaos (positions more stable)
            if "track_overtaking" in race_params:
                track_chaos_factor = race_params.get("track_chaos_multiplier", 0.4)
                track_multiplier = 1.0 - (race_params["track_overtaking"] * track_chaos_factor)
                chaos *= track_multiplier

            # 8. Safety car luck (random position swing if SC deployed)
            sc_luck = 0.0
            if sc_deployed_this_lap:
                sc_luck_range = race_params.get("safety_car_luck_range", 0.25)
                sc_luck = rng.uniform(-sc_luck_range, sc_luck_range)

            # 9. Teammate variance (setup/strategy differences)
            teammate_variance = rng.normal(0, race_params.get("teammate_variance_std", 0.15))

            # 10.5 Traffic + overtaking effects
            traffic_overtake_effect = _get_traffic_overtake_effect(
                driver=driver,
                driver_states=driver_states,
                driver_info_map=driver_info_map,
                driver_ahead_map=driver_ahead_map,
                race_params=race_params,
                rng=rng,
            )

            # 11. Total lap time
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
            lap_time_bounds = race_params.get("lap_time", {}).get("bounds", [70.0, 120.0])
            lap_time = max(lap_time_bounds[0], min(lap_time_bounds[1], lap_time))

            # Update cumulative time and tire age
            state["cumulative_time"] += lap_time
            state["laps_on_tire"] += 1

            # Fuel burn (configurable)
            fuel_burn_rate = race_params.get("fuel", {}).get("burn_rate_kg_per_lap", 1.5)
            state["fuel_load"] = max(0.0, state["fuel_load"] - fuel_burn_rate)

            # 12. Pit stop handling
            strategy = strategies[driver]
            if lap_num in strategy["pit_laps"]:
                _apply_pit_stop(state, strategy, race_params, rng)

        # Update positions based on cumulative time (after all drivers complete lap)
        _update_positions_from_times(driver_states)

    # Generate finish order and metadata
    return _generate_race_result(driver_states, strategies)


def _get_traffic_overtake_effect(
    driver: str,
    driver_states: dict[str, dict],
    driver_info_map: dict[str, dict],
    driver_ahead_map: dict[str, str],
    race_params: dict,
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
    target_position: int, overtake_cfg: dict
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


def _get_lap1_chaos(position: int, race_params: dict, rng: np.random.Generator) -> float:
    """Calculate lap 1 chaos based on grid position."""
    lap1_config = race_params.get("lap1_chaos", {})

    if position <= 3:
        std = lap1_config.get("front_row", 0.15)
    elif position <= 10:
        std = lap1_config.get("upper_midfield", 0.32)
    elif position <= 15:
        std = lap1_config.get("midfield", 0.38)
    else:
        std = lap1_config.get("back_field", 0.28)

    return rng.normal(0, std)


def _apply_pit_stop(
    state: dict,
    strategy: PitStrategy,
    race_params: dict,
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
            f"Pit stop: {state.get('driver', 'unknown')} → {new_compound} (+{total_pit_loss:.2f}s)"
        )
    else:
        logger.warning(
            f"Stint number {state['stint_number']} exceeds compound sequence length "
            f"{len(strategy['compound_sequence'])}"
        )


def _update_positions_from_times(driver_states: dict[str, dict]) -> None:
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

    # Sort DNF drivers by lap they DNF'd (earlier DNF = worse position)
    dnf_drivers.sort(key=lambda x: x[1])

    # Assign positions
    position = 1
    for driver, _ in active_drivers:
        driver_states[driver]["position"] = position
        position += 1

    for driver, _ in dnf_drivers:
        driver_states[driver]["position"] = position
        position += 1


def _generate_race_result(
    driver_states: dict[str, dict],
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
) -> dict:
    """Aggregate results from multiple simulations.

    Returns dict with:
        - median_positions: Dict[str, int] (driver → median finish position)
        - position_distributions: Dict[str, List[int]] (driver → all positions)
        - dnf_rates: Dict[str, float] (driver → % of sims with DNF)
        - compound_strategy_distribution: Dict[str, float] (strategy → frequency)
        - pit_lap_distribution: Dict[str, int] (lap bin → count)
    """
    from collections import defaultdict

    position_data = defaultdict(list)
    dnf_counts = defaultdict(int)
    strategy_counts = defaultdict(int)
    pit_lap_counts = defaultdict(int)

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
