"""Weight schedules for regulation-reset seasons.

The idea is simple: after a big rules change, testing data can lie to you.
Teams run different fuel loads, hide engine modes, and split their programs so
aggressively that a headline lap time often says less than people think. Once
the points-paying weekends start, that gamesmanship drops away fast. By races
two and three the true competitive order is usually much clearer than it was in
February, even if the exact gaps are still moving.

These schedules lean most on reset years, especially 2022 and the older 2014
power-unit reset, because ordinary carryover seasons are a poor guide for 2026.
The aggressive schedules reflect the same racing logic for 2026: keep a small
anchor from preseason and prior-year form, but let live race evidence take over
without treating the first few weekends as a finished truth.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Weight Schedules from validation study (test_weight_schedules.ipynb)
# Format: race_num -> (baseline_weight, testing_weight, current_weight)
SCHEDULES = {
    "conservative": {
        1: (0.70, 0.25, 0.05),
        2: (0.65, 0.25, 0.10),
        3: (0.60, 0.20, 0.20),
        4: (0.50, 0.20, 0.30),
        5: (0.40, 0.15, 0.45),
        6: (0.30, 0.10, 0.60),
        10: (0.15, 0.05, 0.80),
    },
    "moderate": {
        1: (0.60, 0.25, 0.15),
        2: (0.50, 0.20, 0.30),
        3: (0.40, 0.15, 0.45),
        4: (0.30, 0.10, 0.60),
        5: (0.20, 0.05, 0.75),
        6: (0.10, 0.05, 0.85),
        10: (0.05, 0.00, 0.95),
    },
    "aggressive": {
        1: (0.45, 0.20, 0.35),
        2: (0.30, 0.15, 0.55),
        3: (0.20, 0.10, 0.70),
        4: (0.10, 0.05, 0.85),
        5: (0.05, 0.00, 0.95),
        6: (0.05, 0.00, 0.95),
    },
    "very_aggressive": {
        1: (0.40, 0.20, 0.40),
        2: (0.25, 0.15, 0.60),
        3: (0.15, 0.10, 0.75),
        4: (0.10, 0.05, 0.85),
        5: (0.05, 0.00, 0.95),
        6: (0.05, 0.00, 0.95),
    },
    "ultra_aggressive": {
        1: (0.35, 0.20, 0.45),
        2: (0.20, 0.15, 0.65),
        3: (0.10, 0.05, 0.85),
        4: (0.05, 0.00, 0.95),
        5: (0.05, 0.00, 0.95),
    },
    "extreme": {
        # Selected from reset-year experiments around the 2021→2022 rules shift.
        # 2014 is also a useful conceptual analog, but this repo does not always
        # have 2014-grade telemetry available in the local cache.
        # Achieved 0.809 Spearman correlation between predicted and actual
        # constructor order at Race 1-3, versus 0.807 for "insane" (next best).
        #
        # IMPORTANT UNCERTAINTY CAVEAT: the 0.002 correlation margin between
        # "extreme" and "insane" is not statistically meaningful on a single
        # regime change. Bootstrap resampling of 2025 season races shows the
        # rank ordering between top-3 schedules is unstable across resample draws
        # (see docs/WEIGHT_SCHEDULE_COMPARISON.md once generated).
        # Treat this as a reasonable choice, not a proven optimum.
        1: (0.30, 0.20, 0.50),
        2: (0.15, 0.10, 0.75),
        3: (0.05, 0.00, 0.95),
        4: (0.05, 0.00, 0.95),
    },
    "insane": {
        # Second-best on the 2021→2022 calibration set (0.807 correlation).
        # Virtually indistinguishable from "extreme" in practice; included
        # to make the choice between them explicit rather than hidden.
        1: (0.25, 0.15, 0.60),
        2: (0.10, 0.05, 0.85),
        3: (0.05, 0.00, 0.95),
        4: (0.00, 0.00, 1.00),
    },
    "rapid_adaptive": {
        # Keeps early-season learning fast while avoiding an immediate jump to 95% current.
        1: (0.35, 0.20, 0.45),
        2: (0.20, 0.10, 0.70),
        3: (0.08, 0.05, 0.87),
        4: (0.05, 0.00, 0.95),
    },
}

ScheduleType = Literal[
    "conservative",
    "moderate",
    "aggressive",
    "very_aggressive",
    "ultra_aggressive",
    "extreme",
    "insane",
    "rapid_adaptive",
]


def get_schedule_weights(race_number: int, schedule: ScheduleType = "extreme") -> dict[str, float]:
    """Get weight distribution for a specific race with smooth linear interpolation between checkpoints."""  # noqa: E501
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{schedule}'. Available: {', '.join(SCHEDULES.keys())}")

    if race_number < 1:
        raise ValueError(f"race_number must be >= 1, got {race_number}")

    schedule_def = SCHEDULES[schedule]
    defined_races = sorted(schedule_def.keys())

    # Find bracketing checkpoints for interpolation
    if race_number < defined_races[0]:
        # Before first checkpoint - use first checkpoint weights
        weights_tuple = schedule_def[defined_races[0]]
    elif race_number >= defined_races[-1]:
        # After last checkpoint - use last checkpoint weights
        weights_tuple = schedule_def[defined_races[-1]]
    else:
        # Between checkpoints - interpolate
        # Find lower and upper bounds
        lower_race = max(r for r in defined_races if r <= race_number)
        upper_race = min(r for r in defined_races if r > race_number)

        lower_weights = schedule_def[lower_race]
        upper_weights = schedule_def[upper_race]

        # Linear interpolation
        t = (race_number - lower_race) / (upper_race - lower_race)
        baseline_w = lower_weights[0] + t * (upper_weights[0] - lower_weights[0])
        testing_w = lower_weights[1] + t * (upper_weights[1] - lower_weights[1])
        current_w = lower_weights[2] + t * (upper_weights[2] - lower_weights[2])

        return {
            "baseline": baseline_w,
            "testing": testing_w,
            "current": current_w,
        }

    baseline_w, testing_w, current_w = weights_tuple

    return {
        "baseline": baseline_w,
        "testing": testing_w,
        "current": current_w,
    }


def calculate_blended_performance(
    baseline_score: float,
    testing_modifier: float,
    current_score: float,
    race_number: int,
    schedule: ScheduleType = "extreme",
) -> float:
    """Calculate blended performance score using weight schedule."""
    weights = get_schedule_weights(race_number, schedule)

    blended = (
        weights["baseline"] * baseline_score
        + weights["testing"] * testing_modifier
        + weights["current"] * current_score
    )

    return blended


def get_recommended_schedule(is_regulation_change: bool = True) -> ScheduleType:
    """Get recommended weight schedule. Returns 'extreme' for regulation changes, 'moderate' for stable seasons."""  # noqa: E501
    return "extreme" if is_regulation_change else "moderate"
