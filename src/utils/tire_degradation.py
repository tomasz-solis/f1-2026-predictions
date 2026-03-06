"""Tire degradation and fuel effect modeling for lap-by-lap race simulation."""

import math

from src.utils import config_loader


def _coerce_base_tire_deg_slope(base_tire_deg_slope: float | None) -> float:
    """Return a valid base tire degradation slope using configured fallback when needed."""
    if base_tire_deg_slope is not None:
        slope = float(base_tire_deg_slope)
        if math.isfinite(slope):
            return slope

    configured_default = config_loader.get(
        "baseline_predictor.race.tire_physics.default_deg_slope",
        0.15,
    )
    try:
        return float(configured_default)
    except (TypeError, ValueError):
        return 0.15


def _temperature_degradation_multiplier(track_temp: float | None) -> float:
    """Return degradation multiplier from track temperature."""
    if track_temp is None:
        return 1.0

    try:
        track_temp_c = float(track_temp)
    except (TypeError, ValueError):
        return 1.0

    reference_temp_c = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.degradation.reference_c", 35.0
    )
    sensitivity_per_c = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.degradation.sensitivity_per_c", 0.006
    )
    min_multiplier = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.degradation.min_multiplier", 0.88
    )
    max_multiplier = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.degradation.max_multiplier", 1.18
    )

    delta_c = track_temp_c - float(reference_temp_c)
    multiplier = 1.0 + (delta_c * float(sensitivity_per_c))
    return float(min(max_multiplier, max(min_multiplier, multiplier)))


def _temperature_fresh_tire_multiplier(track_temp: float | None) -> float:
    """Return fresh-tire advantage multiplier from distance to optimal temperature."""
    if track_temp is None:
        return 1.0

    try:
        track_temp_c = float(track_temp)
    except (TypeError, ValueError):
        return 1.0

    optimal_temp_c = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.fresh.optimal_c", 30.0
    )
    decay_per_c = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.fresh.decay_per_c", 0.01
    )
    min_multiplier = config_loader.get(
        "baseline_predictor.race.tire_physics.temperature.fresh.min_multiplier", 0.70
    )

    temp_delta_c = abs(track_temp_c - float(optimal_temp_c))
    multiplier = 1.0 - (temp_delta_c * float(decay_per_c))
    return float(max(min_multiplier, min(1.0, multiplier)))


def calculate_tire_deg_delta(
    tire_deg_slope: float,
    laps_on_tire: int,
    fuel_load_kg: float,
    initial_fuel_kg: float | None = None,
    compound: str | None = None,
    track_temp: float | None = None,
) -> float:
    """Calculate lap time penalty from tire wear.

    Degradation increases linearly with tire age up to compound max age,
    then accelerates sharply (cliff). Fuel load also increases wear.
    """
    if tire_deg_slope <= 0.0 or laps_on_tire <= 0:
        return 0.0

    # Load config
    if initial_fuel_kg is None:
        initial_fuel_kg = config_loader.get("baseline_predictor.race.fuel.initial_load_kg", 110.0)

    fuel_deg_multiplier = config_loader.get("baseline_predictor.race.fuel.deg_multiplier", 0.10)

    # Fuel load effect on degradation: heavier car = more tire stress
    # Multiplier: 1.0 (empty) to 1.1 (full tank)
    fuel_ratio = fuel_load_kg / initial_fuel_kg
    fuel_multiplier = 1.0 + (fuel_deg_multiplier * fuel_ratio)

    temp_multiplier = _temperature_degradation_multiplier(track_temp)

    # Tire cliff: beyond max age, degradation slope multiplies sharply
    if compound is not None:
        compound_max_ages = config_loader.get(
            "baseline_predictor.race.tire_physics.compound_max_age",
            {"SOFT": 24, "MEDIUM": 34, "HARD": 42},
        )
        cliff_multiplier = config_loader.get(
            "baseline_predictor.race.tire_physics.cliff_multiplier", 2.8
        )
        max_age = compound_max_ages.get(compound.upper(), 40)
        if laps_on_tire > max_age:
            laps_past_cliff = laps_on_tire - max_age
            linear_portion = tire_deg_slope * max_age * fuel_multiplier * temp_multiplier
            cliff_portion = (
                tire_deg_slope
                * cliff_multiplier
                * laps_past_cliff
                * fuel_multiplier
                * temp_multiplier
            )
            return float(max(0.0, linear_portion + cliff_portion))

    # Base degradation: slope × laps on tire × fuel effect
    degradation = tire_deg_slope * laps_on_tire * fuel_multiplier * temp_multiplier

    return float(max(0.0, degradation))


def calculate_fuel_delta(
    laps_remaining: int,
    fuel_effect_per_lap: float = 0.035,
) -> float:
    """Calculate lap time penalty from fuel weight.

    Cars are slower when heavy with fuel. Effect diminishes as fuel burns.
    """
    if laps_remaining <= 0:
        return 0.0

    # Fuel load estimate: ~1.5 kg per lap remaining
    fuel_load_kg = laps_remaining * 1.5

    # Convert to lap time penalty (per 10kg)
    fuel_penalty = (fuel_load_kg / 10.0) * fuel_effect_per_lap

    return float(max(0.0, fuel_penalty))


def get_fresh_tire_advantage(
    compound: str,
    laps_on_tire: int,
    track_temp: float | None = None,
) -> float:
    """Calculate lap time advantage for fresh tires.

    Fresh tires provide a pace advantage for the first few laps before
    reaching optimal operating window. Effect is larger for softer compounds.
    """
    compound_upper = compound.upper().strip()

    # Load fresh tire config
    fresh_tire_advantages = config_loader.get(
        "baseline_predictor.race.tire_physics.fresh_tire_advantage",
        {"SOFT": 0.5, "MEDIUM": 0.3, "HARD": 0.1},
    )
    fresh_tire_durations = config_loader.get(
        "baseline_predictor.race.tire_physics.fresh_tire_duration",
        {"SOFT": 3, "MEDIUM": 3, "HARD": 2},
    )

    if compound_upper not in fresh_tire_durations:
        return 0.0

    # Check if still in fresh tire window
    fresh_laps = fresh_tire_durations[compound_upper]
    if laps_on_tire >= fresh_laps:
        return 0.0

    # Base advantage
    base_advantage = fresh_tire_advantages.get(compound_upper, 0.0)

    # Linear decay: full advantage on lap 1, zero at fresh_laps
    # lap 1 → 1.0, lap 2 → 0.66, lap 3 → 0.33, lap 4 → 0.0 (for 3-lap window)
    decay_factor = 1.0 - (laps_on_tire / fresh_laps)

    advantage = base_advantage * decay_factor

    # Temperatures far from the optimal window reduce warm-up/freshness benefit.
    advantage *= _temperature_fresh_tire_multiplier(track_temp)

    return float(max(0.0, advantage))


def estimate_stint_pace_degradation(
    tire_deg_slope: float,
    stint_length: int,
    compound: str,
    fuel_load_start_kg: float = 110.0,
    track_temp: float | None = None,
) -> float:
    """Estimate total pace loss over a stint from tire degradation.

    Useful for strategy optimization before race simulation.
    """
    if tire_deg_slope <= 0.0 or stint_length <= 0:
        return 0.0

    total_deg = 0.0

    for lap_index in range(stint_length):
        # Keep indexing aligned with lap simulator (new stint starts at laps_on_tire=0).
        fuel_remaining = fuel_load_start_kg - (lap_index * 1.5)
        fuel_remaining = max(0.0, fuel_remaining)

        # Calculate degradation for this lap
        lap_deg = calculate_tire_deg_delta(
            tire_deg_slope=tire_deg_slope,
            laps_on_tire=lap_index,
            fuel_load_kg=fuel_remaining,
            initial_fuel_kg=fuel_load_start_kg,
            track_temp=track_temp,
        )

        # Subtract fresh tire advantage (first few laps)
        fresh_advantage = get_fresh_tire_advantage(
            compound,
            lap_index,
            track_temp=track_temp,
        )

        total_deg += lap_deg - fresh_advantage

    return float(max(0.0, total_deg))


def get_effective_tire_deg_slope(
    base_tire_deg_slope: float | None,
    traffic_position: int,
    total_cars: int = 20,
) -> float:
    """Adjust tire degradation based on traffic/dirty air.

    Cars running in dirty air experience more tire degradation.
    Leaders have cleaner air and better tire management.
    """
    resolved_base_slope = _coerce_base_tire_deg_slope(base_tire_deg_slope)

    if total_cars <= 0:
        return resolved_base_slope

    # Load config
    clean_air_bonus = config_loader.get(
        "baseline_predictor.race.tire_physics.clean_air_bonus", 0.05
    )
    traffic_deg_penalty = config_loader.get(
        "baseline_predictor.race.tire_physics.traffic_deg_penalty", 0.05
    )

    # Position effect: front runners (p1-p5) get slight advantage
    # Midfield (p6-p15) neutral
    # Back markers (p16+) get slight penalty

    if traffic_position <= 5:
        # Clean air advantage
        multiplier = 1.0 - clean_air_bonus
    elif traffic_position <= 15:
        # Midfield: neutral
        multiplier = 1.0
    else:
        # Dirty air penalty
        multiplier = 1.0 + traffic_deg_penalty

    return resolved_base_slope * multiplier
