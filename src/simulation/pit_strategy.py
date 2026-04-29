"""Pit-stop strategy generation and validation for race simulation.

Dry grands prix still have the FIA two-compound mandate, so a legal strategy is
never just about the fastest single tire. The generator starts from that rule,
then picks between one-stop and two-stop shapes based on tire stress, with just
enough randomness to represent real race engineering tradeoffs. That is where
undercuts, overcuts, and conservative "cover the car behind" calls come from in
practice: the same race can support multiple valid plans even when one is
slightly faster on paper.
"""

import logging

import numpy as np

from src.simulation.strategy_optimizer import calculate_pit_timing_bias_laps
from src.types.prediction_types import PitStrategy
from src.utils import config_loader
from src.utils.prediction_context import get_config_value

logger = logging.getLogger(__name__)

# Canonical compound ordering used for filtering and fallback generation.
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def _ordered_available_compounds(available_compounds: list[str]) -> list[str]:
    """Return available compounds in canonical order, preserving unknown extras."""
    normalized = [str(compound).strip().upper() for compound in available_compounds if compound]
    ordered = [compound for compound in COMPOUND_ORDER if compound in normalized]
    extras = [compound for compound in normalized if compound not in ordered]
    return ordered + extras


def sample_sprint_compound(
    available_compounds: list[str],
    grid_position: int | None,
    tire_stress_score: float,
    rng: np.random.Generator,
) -> str:
    """Pick a sprint starting compound for one driver.

    Empirical 2024-2025 sprint data: MEDIUM is the modal compound (~88%).
    A small fraction of back-of-grid drivers gamble on SOFT for early track
    position. HARD appears occasionally at high-stress tracks.

    Returns a compound name that is always present in available_compounds.
    """
    base_distribution = get_config_value(
        "baseline_predictor.race.sprint_compound.base_distribution",
        {"MEDIUM": 0.88, "SOFT": 0.08, "HARD": 0.04},
        config=config_loader,
    )
    soft_bonus = float(
        get_config_value(
            "baseline_predictor.race.sprint_compound.soft_lower_grid_bonus",
            0.06,
            config=config_loader,
        )
    )
    high_stress_threshold = float(
        get_config_value(
            "baseline_predictor.race.sprint_compound.high_stress_threshold",
            4.0,
            config=config_loader,
        )
    )
    high_stress_shift = float(
        get_config_value(
            "baseline_predictor.race.sprint_compound.high_stress_hard_shift",
            0.08,
            config=config_loader,
        )
    )

    weights: dict[str, float] = {k: float(v) for k, v in base_distribution.items()}

    # Back-of-grid drivers more likely to gamble on SOFT for early position gain.
    if grid_position is not None and grid_position > 10:
        weights["SOFT"] = weights.get("SOFT", 0.0) + soft_bonus
        weights["MEDIUM"] = max(0.0, weights.get("MEDIUM", 0.0) - soft_bonus)

    # High-stress tracks shift mass toward HARD.
    if tire_stress_score > high_stress_threshold:
        shift = high_stress_shift
        weights["SOFT"] = max(0.0, weights.get("SOFT", 0.0) - shift / 2)
        weights["MEDIUM"] = max(0.0, weights.get("MEDIUM", 0.0) - shift / 2)
        weights["HARD"] = weights.get("HARD", 0.0) + shift

    available_set = set(_ordered_available_compounds(available_compounds))
    weights = {c: w for c, w in weights.items() if c in available_set and w > 0}

    if not weights:
        return available_compounds[0] if available_compounds else "MEDIUM"

    total = sum(weights.values())
    compounds = list(weights.keys())
    probs = [weights[c] / total for c in compounds]
    return str(rng.choice(compounds, p=probs))


def generate_pit_strategy(
    race_distance: int,
    tire_stress_score: float,
    available_compounds: list[str],
    rng: np.random.Generator,
    driver_risk_profile: float | None = None,
    enforce_two_compound_rule: bool = True,
    track_overtaking: float | None = None,
    grid_position: int | None = None,
    strategy_signal: float = 0.0,
) -> PitStrategy:
    """Generate Monte Carlo pit strategy for one driver in one simulation.

    Returns dict with:
        - num_stops: int (1, 2, or 3)
        - pit_laps: List[int] (which laps to pit)
        - compound_sequence: List[str] (starting compound + post-pit compounds)
        - stint_lengths: List[int] (laps per stint)
    """
    # Load config parameters
    high_stress_2stop_prob = get_config_value(
        "baseline_predictor.race.tire_strategy.stop_probability.high_stress_2stop",
        0.80,
        config=config_loader,
    )
    med_stress_1stop_prob = get_config_value(
        "baseline_predictor.race.tire_strategy.stop_probability.medium_stress_1stop",
        0.90,
        config=config_loader,
    )
    low_stress_1stop_prob = get_config_value(
        "baseline_predictor.race.tire_strategy.stop_probability.low_stress_1stop",
        0.95,
        config=config_loader,
    )

    high_stress_threshold = get_config_value(
        "baseline_predictor.compound_selection.high_stress_threshold",
        3.5,
        config=config_loader,
    )
    low_stress_threshold = get_config_value(
        "baseline_predictor.compound_selection.low_stress_threshold",
        2.5,
        config=config_loader,
    )

    # Decide number of stops based on tire stress
    if tire_stress_score > high_stress_threshold:
        # High stress: favor 2-stop
        num_stops = 2 if rng.random() < high_stress_2stop_prob else 1
    elif tire_stress_score < low_stress_threshold:
        # Low stress: favor 1-stop
        num_stops = 1 if rng.random() < low_stress_1stop_prob else 2
    else:
        # Medium stress: mostly 1-stop
        num_stops = 1 if rng.random() < med_stress_1stop_prob else 2

    # Optional: driver risk profile adjusts stop count
    # Aggressive drivers might attempt undercut (2-stop)
    if driver_risk_profile and driver_risk_profile > 0.8:
        # Aggressive: 10% chance to add extra stop
        if rng.random() < 0.10 and num_stops < 3:
            num_stops += 1

    # Generate pit laps based on number of stops
    pit_timing_bias_laps = calculate_pit_timing_bias_laps(
        track_overtaking=track_overtaking,
        grid_position=grid_position,
        race_distance=race_distance,
        strategy_signal=strategy_signal,
    )

    pit_laps = _sample_pit_laps(
        race_distance,
        num_stops,
        rng,
        timing_bias_laps=pit_timing_bias_laps,
    )

    # Generate compound sequence (dry races enforce >=2 compounds, wet does not).
    compound_sequence = _sample_compound_sequence(
        available_compounds,
        num_stops,
        tire_stress_score,
        rng,
        enforce_two_compound_rule=enforce_two_compound_rule,
    )

    # Calculate stint lengths
    stint_lengths = _calculate_stint_lengths(race_distance, pit_laps)

    strategy: PitStrategy = {
        "num_stops": num_stops,
        "pit_laps": pit_laps,
        "compound_sequence": compound_sequence,
        "stint_lengths": stint_lengths,
    }

    # Validate strategy
    if not validate_strategy(
        strategy,
        race_distance,
        available_compounds,
        enforce_two_compound_rule=enforce_two_compound_rule,
    ):
        logger.warning("Invalid strategy generated: %s. Falling back to default.", strategy)
        strategy = _get_default_strategy(
            race_distance,
            available_compounds,
            enforce_two_compound_rule=enforce_two_compound_rule,
        )

    return strategy


def _sample_pit_laps(
    race_distance: int,
    num_stops: int,
    rng: np.random.Generator,
    timing_bias_laps: float = 0.0,
) -> list[int]:
    """Sample pit lap numbers from realistic windows."""
    # Load pit windows from config
    one_stop_window = get_config_value(
        "baseline_predictor.race.tire_strategy.windows.one_stop",
        [23, 37],
        config=config_loader,
    )
    two_stop_first = get_config_value(
        "baseline_predictor.race.tire_strategy.windows.two_stop_first",
        [15, 25],
        config=config_loader,
    )
    two_stop_second = get_config_value(
        "baseline_predictor.race.tire_strategy.windows.two_stop_second",
        [35, 45],
        config=config_loader,
    )

    # Load variance config
    one_stop_variance = get_config_value(
        "baseline_predictor.race.strategy_constraints.pit_lap_variance.one_stop",
        3.0,
        config=config_loader,
    )
    two_stop_variance = get_config_value(
        "baseline_predictor.race.strategy_constraints.pit_lap_variance.two_stop",
        2.0,
        config=config_loader,
    )

    # Load safety margins
    min_pit_lap = get_config_value(
        "baseline_predictor.race.strategy_constraints.min_pit_lap",
        5,
        config=config_loader,
    )
    max_pit_lap_from_end = get_config_value(
        "baseline_predictor.race.strategy_constraints.max_pit_lap_from_end",
        5,
        config=config_loader,
    )
    min_laps_between_stops = get_config_value(
        "baseline_predictor.race.strategy_constraints.min_laps_between_stops",
        8,
        config=config_loader,
    )

    # Scale windows proportionally to race distance (default config assumes 60 laps)
    scale_factor = race_distance / 60.0

    def scale_window(window):
        """Scale a pit window from the 60-lap baseline to the actual race distance."""
        return [int(window[0] * scale_factor), int(window[1] * scale_factor)]

    one_stop_scaled = scale_window(one_stop_window)
    two_stop_first_scaled = scale_window(two_stop_first)
    two_stop_second_scaled = scale_window(two_stop_second)

    # Enforce safety margins
    max_pit_lap = race_distance - max_pit_lap_from_end

    pit_laps = []

    if num_stops == 1:
        # Single stop: sample from one_stop window
        lap = int(
            rng.normal(
                loc=((one_stop_scaled[0] + one_stop_scaled[1]) / 2.0) + timing_bias_laps,
                scale=one_stop_variance,
            )
        )
        lap = max(min_pit_lap, min(max_pit_lap, lap))
        pit_laps = [lap]

    elif num_stops == 2:
        # Two stops: sample from both windows
        lap1 = int(
            rng.normal(
                loc=((two_stop_first_scaled[0] + two_stop_first_scaled[1]) / 2.0)
                + (timing_bias_laps * 0.75),
                scale=two_stop_variance,
            )
        )
        lap1 = max(min_pit_lap, min(max_pit_lap, lap1))

        lap2 = int(
            rng.normal(
                loc=((two_stop_second_scaled[0] + two_stop_second_scaled[1]) / 2.0)
                + timing_bias_laps,
                scale=two_stop_variance,
            )
        )
        lap2 = max(lap1 + min_laps_between_stops, min(max_pit_lap, lap2))

        pit_laps = [lap1, lap2]

    elif num_stops == 3:
        # Three stops: divide race into quarters
        quarter = race_distance / 4.0
        lap1 = int(rng.normal(loc=quarter, scale=two_stop_variance))
        lap2 = int(rng.normal(loc=2 * quarter, scale=two_stop_variance))
        lap3 = int(rng.normal(loc=3 * quarter, scale=two_stop_variance))

        lap1 = max(min_pit_lap, min(max_pit_lap, lap1))
        lap2 = max(lap1 + min_laps_between_stops, min(max_pit_lap, lap2))
        lap3 = max(lap2 + min_laps_between_stops, min(max_pit_lap, lap3))

        pit_laps = [lap1, lap2, lap3]

    return sorted(pit_laps)


def _sample_compound_sequence(
    available_compounds: list[str],
    num_stops: int,
    tire_stress_score: float,
    rng: np.random.Generator,
    enforce_two_compound_rule: bool = True,
) -> list[str]:
    """Sample compound sequence (starting + post-pit compounds).

    Dry races must satisfy FIA rule: >=2 different compounds used.
    """
    num_compounds_needed = num_stops + 1  # Starting compound + 1 per stop

    # Load compound preferences from config
    # compound_prefs = config_loader.get(
    #    "baseline_predictor.race.tire_strategy.compound_preferences",
    #     {"SOFT": 1.0, "MEDIUM": 0.8, "HARD": 0.6},
    # )

    # Filter available compounds with wet/mixed options preserved.
    available = _ordered_available_compounds(available_compounds)

    if enforce_two_compound_rule and len(available) < 2:
        logger.warning("Insufficient compounds available: %s. Cannot satisfy FIA rule.", available)
        # Fallback: repeat available compound (will fail validation)
        available = available_compounds

    # Adjust preferences based on tire stress
    high_stress_threshold = get_config_value(
        "baseline_predictor.compound_selection.high_stress_threshold",
        3.5,
        config=config_loader,
    )
    low_stress_threshold = get_config_value(
        "baseline_predictor.compound_selection.low_stress_threshold",
        2.5,
        config=config_loader,
    )

    has_intermediate = "INTERMEDIATE" in available
    has_wet = "WET" in available
    wet_only = has_wet and not enforce_two_compound_rule and len(available) <= 2

    if wet_only:
        # Fully wet race: favor INTERMEDIATE unless stress strongly suggests full wets.
        if tire_stress_score > high_stress_threshold:
            preference_order = ["WET", "INTERMEDIATE"]
        else:
            preference_order = ["INTERMEDIATE", "WET"]
    elif has_intermediate:
        # Mixed weather: keep dry compounds in play but allow transition strategies.
        if tire_stress_score > high_stress_threshold:
            preference_order = ["HARD", "INTERMEDIATE", "MEDIUM", "SOFT"]
        elif tire_stress_score < low_stress_threshold:
            preference_order = ["SOFT", "INTERMEDIATE", "MEDIUM", "HARD"]
        else:
            preference_order = ["MEDIUM", "INTERMEDIATE", "SOFT", "HARD"]
    elif tire_stress_score > high_stress_threshold:
        # High stress: prefer HARD, avoid SOFT.
        preference_order = ["HARD", "MEDIUM", "SOFT"]
    elif tire_stress_score < low_stress_threshold:
        # Low stress: prefer SOFT.
        preference_order = ["SOFT", "MEDIUM", "HARD"]
    else:
        # Medium stress: prefer MEDIUM.
        preference_order = ["MEDIUM", "SOFT", "HARD"]

    # Filter preferences to available compounds
    ordered_compounds = [c for c in preference_order if c in available]

    if not ordered_compounds:
        ordered_compounds = list(available_compounds)

    # Wet races can legally run one compound throughout.
    if not enforce_two_compound_rule:
        if rng.random() < 0.75:
            return [ordered_compounds[0]] * num_compounds_needed

        shuffled = list(ordered_compounds)
        rng.shuffle(shuffled)
        sequence = shuffled[:num_compounds_needed]
        while len(sequence) < num_compounds_needed:
            sequence.append(sequence[-1])
        return sequence

    # Monte Carlo: configurable optimality ratio (for realism)
    optimality_ratio = get_config_value(
        "baseline_predictor.race.strategy_constraints.strategy_optimality",
        0.60,
        config=config_loader,
    )

    if rng.random() < optimality_ratio:
        # Optimal sequence: use preference order
        compound_sequence = ordered_compounds[:num_compounds_needed]
    else:
        # Suboptimal: shuffle or reverse order
        if rng.random() < 0.5:
            # Reversed order (e.g., HARD→SOFT instead of SOFT→HARD)
            compound_sequence = ordered_compounds[::-1][:num_compounds_needed]
        else:
            # Random shuffle
            shuffled = list(ordered_compounds)
            rng.shuffle(shuffled)
            compound_sequence = shuffled[:num_compounds_needed]

    # Pad if insufficient compounds (edge case)
    while len(compound_sequence) < num_compounds_needed:
        compound_sequence.append(compound_sequence[-1])

    return compound_sequence


def _calculate_stint_lengths(race_distance: int, pit_laps: list[int]) -> list[int]:
    """Calculate lap count per stint from pit laps."""
    if not pit_laps:
        # No stops: entire race is one stint
        return [race_distance]

    stint_lengths = []

    # First stint: laps 1 to first pit
    stint_lengths.append(pit_laps[0])

    # Middle stints: between pit stops
    for i in range(1, len(pit_laps)):
        stint_length = pit_laps[i] - pit_laps[i - 1]
        stint_lengths.append(stint_length)

    # Final stint: last pit to finish
    final_stint = race_distance - pit_laps[-1]
    stint_lengths.append(final_stint)

    return stint_lengths


def validate_strategy(
    strategy: PitStrategy,
    race_distance: int,
    available_compounds: list[str],
    enforce_two_compound_rule: bool = True,
) -> bool:
    """Validate strategy satisfies FIA rules and physical constraints."""
    # Load safety margins
    min_pit_lap = get_config_value(
        "baseline_predictor.race.strategy_constraints.min_pit_lap",
        5,
        config=config_loader,
    )
    max_pit_lap_from_end = get_config_value(
        "baseline_predictor.race.strategy_constraints.max_pit_lap_from_end",
        5,
        config=config_loader,
    )

    # Check required fields
    required_fields = ["num_stops", "pit_laps", "compound_sequence", "stint_lengths"]
    if not all(field in strategy for field in required_fields):
        logger.warning("Strategy missing required fields: %s", strategy)
        return False

    num_stops = strategy["num_stops"]
    pit_laps = strategy["pit_laps"]
    compound_sequence = strategy["compound_sequence"]
    stint_lengths = strategy["stint_lengths"]

    # Check: pit_laps length matches num_stops
    if len(pit_laps) != num_stops:
        logger.warning("Pit laps length (%s) != num_stops (%s)", len(pit_laps), num_stops)
        return False

    # Check: compound_sequence length = num_stops + 1
    if len(compound_sequence) != num_stops + 1:
        logger.warning(
            "Compound sequence length (%s) != num_stops + 1 (%s)",
            len(compound_sequence),
            num_stops + 1,
        )
        return False

    # Check: stint_lengths sum = race_distance
    if sum(stint_lengths) != race_distance:
        logger.warning(
            "Stint lengths sum (%s) != race_distance (%s)", sum(stint_lengths), race_distance
        )
        return False

    if enforce_two_compound_rule:
        # FIA dry-race rule: >=2 unique compounds.
        unique_compounds = set(compound_sequence)
        if len(unique_compounds) < 2:
            logger.warning("FIA rule violation: <2 unique compounds (%s)", unique_compounds)
            return False

    # Check: all compounds available
    for compound in compound_sequence:
        if compound not in available_compounds:
            logger.warning(
                "Compound %s not in available compounds: %s", compound, available_compounds
            )
            return False

    # Check: pit laps are sorted and within race bounds
    if pit_laps != sorted(pit_laps):
        logger.warning("Pit laps not sorted: %s", pit_laps)
        return False

    for lap in pit_laps:
        if lap < min_pit_lap or lap > (race_distance - max_pit_lap_from_end):
            logger.warning(
                "Pit lap %s outside valid range [%s, %s]",
                lap,
                min_pit_lap,
                race_distance - max_pit_lap_from_end,
            )
            return False

    return True


def _get_default_strategy(
    race_distance: int,
    available_compounds: list[str],
    enforce_two_compound_rule: bool = True,
) -> PitStrategy:
    """Return safe default 1-stop strategy as fallback."""
    # Default: 1-stop at ~50% race distance
    pit_lap = race_distance // 2

    # Use first two available compounds for dry-race fallback.
    available = _ordered_available_compounds(available_compounds)
    if len(available) < 2:
        available = available_compounds[:]

    if enforce_two_compound_rule:
        if len(available) < 2:
            available = (
                available_compounds[:2]
                if len(available_compounds) >= 2
                else available_compounds * 2
            )
        compound_sequence = available[:2]
    else:
        fallback_compound = available[0] if available else "INTERMEDIATE"
        compound_sequence = [fallback_compound, fallback_compound]

    stint_lengths = [pit_lap, race_distance - pit_lap]

    return {
        "num_stops": 1,
        "pit_laps": [pit_lap],
        "compound_sequence": compound_sequence,
        "stint_lengths": stint_lengths,
    }


def analyze_strategy_distribution(strategies: dict[str, PitStrategy]) -> dict:
    """Analyze compound strategy distribution across all drivers in simulation."""
    strategy_counts = {}

    for strategy in strategies.values():
        sequence = strategy["compound_sequence"]
        sequence_str = "→".join(sequence)

        if sequence_str not in strategy_counts:
            strategy_counts[sequence_str] = 0
        strategy_counts[sequence_str] += 1

    # Convert to percentages
    total = sum(strategy_counts.values())
    strategy_distribution = {seq: count / total for seq, count in strategy_counts.items()}

    return strategy_distribution


def analyze_pit_lap_distribution(strategies: dict[str, PitStrategy]) -> dict:
    """Analyze pit lap timing distribution across all drivers."""
    pit_lap_bins = {}

    for strategy in strategies.values():
        for pit_lap in strategy["pit_laps"]:
            # Bin into 5-lap windows
            bin_start = (pit_lap // 5) * 5
            bin_end = bin_start + 5
            bin_label = f"lap_{bin_start}-{bin_end}"

            if bin_label not in pit_lap_bins:
                pit_lap_bins[bin_label] = 0
            pit_lap_bins[bin_label] += 1

    # Convert to percentages
    total = sum(pit_lap_bins.values())
    if total > 0:
        pit_lap_distribution = {
            bin_label: count / total for bin_label, count in pit_lap_bins.items()
        }
    else:
        pit_lap_distribution = {}

    return pit_lap_distribution
