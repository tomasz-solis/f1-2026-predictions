"""Tire degradation and fuel effect modeling for lap-by-lap race simulation."""

from src.utils import config_loader


def calculate_tire_deg_delta(
    tire_deg_slope: float,
    laps_on_tire: int,
    fuel_load_kg: float,
    initial_fuel_kg: float | None = None,
) -> float:
    """Calculate lap time penalty from tire wear.

    Degradation increases linearly with tire age, and is affected by fuel load.
    Heavier cars degrade tires faster.
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

    # Base degradation: slope × laps on tire × fuel effect
    degradation = tire_deg_slope * laps_on_tire * fuel_multiplier

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

    # Optional: track temp effect (hotter tracks reduce fresh tire advantage)
    # Intentionally omitted until a track-temperature model is added.

    return float(max(0.0, advantage))


def estimate_stint_pace_degradation(
    tire_deg_slope: float,
    stint_length: int,
    compound: str,
    fuel_load_start_kg: float = 110.0,
) -> float:
    """Estimate total pace loss over a stint from tire degradation.

    Useful for strategy optimization before race simulation.
    """
    if tire_deg_slope <= 0.0 or stint_length <= 0:
        return 0.0

    total_deg = 0.0

    for lap in range(1, stint_length + 1):
        # Approximate fuel burn
        fuel_remaining = fuel_load_start_kg - (lap * 1.5)
        fuel_remaining = max(0.0, fuel_remaining)

        # Calculate degradation for this lap
        lap_deg = calculate_tire_deg_delta(
            tire_deg_slope=tire_deg_slope,
            laps_on_tire=lap,
            fuel_load_kg=fuel_remaining,
            initial_fuel_kg=fuel_load_start_kg,
        )

        # Subtract fresh tire advantage (first few laps)
        fresh_advantage = get_fresh_tire_advantage(compound, lap)

        total_deg += lap_deg - fresh_advantage

    return float(max(0.0, total_deg))


def get_effective_tire_deg_slope(
    base_tire_deg_slope: float,
    traffic_position: int,
    total_cars: int = 20,
) -> float:
    """Adjust tire degradation based on traffic/dirty air.

    Cars running in dirty air experience more tire degradation.
    Leaders have cleaner air and better tire management.
    """
    if total_cars <= 0:
        return base_tire_deg_slope

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

    return base_tire_deg_slope * multiplier
