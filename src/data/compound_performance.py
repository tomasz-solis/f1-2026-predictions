"""
Compound Performance Utilities

Helper functions to adjust team performance based on tire compound characteristics
during race predictions.
"""

import logging

logger = logging.getLogger(__name__)

# Standard F1 tire compounds
COMPOUND_NAMES = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


def get_compound_performance_modifier(
    team_compound_chars: dict[str, dict[str, float]],
    compound: str,
    metric_weights: dict[str, float] | None = None,
) -> float:
    """Calculate performance modifier from compound characteristics (-0.05 to +0.05)."""
    if not team_compound_chars:
        return 0.0

    compound_upper = compound.upper().strip() if compound else ""
    compound_data = team_compound_chars.get(compound_upper)

    if not compound_data:
        return 0.0

    # Default weights: prioritize pace over tire deg
    if metric_weights is None:
        metric_weights = {
            "pace_performance": 0.7,
            "tire_deg_performance": 0.3,
        }

    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, weight in metric_weights.items():
        value = compound_data.get(metric_name)
        if value is not None:
            # Center around 0.5 (neutral performance)
            centered = float(value) - 0.5
            weighted_sum += centered * float(weight)
            total_weight += float(weight)

    if total_weight <= 0:
        return 0.0

    # Normalize and scale to a small modifier
    # Scale factor of 0.1 means max ±0.05 modifier (10% of the 0.5 range)
    normalized = weighted_sum / total_weight
    modifier = normalized * 0.1

    # Clip to reasonable bounds
    return float(max(-0.05, min(0.05, modifier)))


def get_compound_tire_deg_factor(
    team_compound_chars: dict[str, dict[str, float]],
    compound: str,
) -> float:
    """Get tire degradation factor (0.0 = best, 1.0 = worst, default 0.5)."""
    if not team_compound_chars:
        return 0.5

    compound_upper = compound.upper().strip() if compound else ""
    compound_data = team_compound_chars.get(compound_upper)

    if not compound_data:
        return 0.5

    # Use tire_deg_performance: 1.0 = best tire life, 0.0 = worst
    tire_deg_perf = compound_data.get("tire_deg_performance")
    if tire_deg_perf is None:
        return 0.5

    # Invert the metric so lower degradation maps to a lower factor.
    return 1.0 - float(tire_deg_perf)


def should_use_compound_adjustments(
    team_compound_chars: dict[str, dict[str, float]],
    min_laps_threshold: int = 10,
) -> bool:
    """Check if compound data has sufficient laps for reliable use."""
    if not team_compound_chars:
        return False

    total_laps = 0
    compounds_with_data = 0

    for compound_data in team_compound_chars.values():
        laps = int(compound_data.get("laps_sampled", 0))
        total_laps += laps
        if laps >= 3:  # At least 3 laps per compound
            compounds_with_data += 1

    # Need at least 2 compounds with data and sufficient total laps
    return compounds_with_data >= 2 and total_laps >= min_laps_threshold


def get_team_compound_advantage(
    team_compound_chars: dict[str, dict[str, float]],
    race_compounds: list[str],
) -> float:
    """Calculate average performance modifier across multiple race compounds."""
    if not race_compounds or not team_compound_chars:
        return 0.0

    modifiers = []
    for compound in race_compounds:
        modifier = get_compound_performance_modifier(team_compound_chars, compound)
        modifiers.append(modifier)

    if not modifiers:
        return 0.0

    return sum(modifiers) / len(modifiers)
