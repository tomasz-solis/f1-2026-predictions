"""Type definitions for prediction data structures."""

from typing import NotRequired, TypedDict


class QualifyingGridEntry(TypedDict):
    """Single driver's qualifying result."""

    driver: str
    team: str
    position: int
    median_position: NotRequired[int]
    p5: NotRequired[int]
    p95: NotRequired[int]
    confidence: NotRequired[float]
    # Calibrated probability (0-100) that the entrant finishes within the configured
    # tolerance of its predicted position. Supplements the legacy ``confidence`` heuristic
    # and is what the dashboard surfaces as "Order Confidence %".
    order_confidence: NotRequired[float | None]
    # True for race-session results where the driver did not finish / was not classified.
    # Used by finisher-only MAE and DNF calibration; absent for qualifying and predictions.
    dnf: NotRequired[bool]
    # How the entrant actually took the start ("grid", or a pit-lane start). Present only
    # on official starting grids, never on a prediction: a predicted grid cannot know that
    # a car will be sent to the pit lane. Replay consumers that reconstruct a real race
    # require it on every row and fail closed without it.
    start_type: NotRequired[str]


class DriverRaceInfo(TypedDict):
    """Driver information for race simulation."""

    driver: str
    team: str
    grid_pos: int
    team_strength: float
    team_strength_score: NotRequired[float]
    team_strength_seconds: NotRequired[float]
    team_strength_seconds_delta: NotRequired[float]
    race_rating_mu_s: NotRequired[float]
    team_uncertainty: NotRequired[float]
    team_strength_by_compound: NotRequired[dict[str, float]]
    team_strength_seconds_by_compound: NotRequired[dict[str, float]]
    team_strength_seconds_delta_by_compound: NotRequired[dict[str, float]]
    tire_deg_by_compound: NotRequired[dict[str, float]]
    skill: float
    race_advantage: float
    race_residual_adjustment: NotRequired[float]
    overtaking_skill: float
    defensive_skill: float
    dnf_probability: float
    season_races_completed: NotRequired[int]
    wet_skill: NotRequired[float]
    current_lineup_team: NotRequired[str]
    raw_skill: NotRequired[float]
    portable_skill: NotRequired[float]
    is_hypothetical_team_assignment: NotRequired[bool]


class PitStrategy(TypedDict):
    """Pit stop strategy for one driver."""

    num_stops: int
    pit_laps: list[int]
    compound_sequence: list[str]
    stint_lengths: list[int]


class RaceSimulationResult(TypedDict):
    """Result from a single Monte Carlo race simulation."""

    finish_order: list[str]
    dnf_drivers: list[str]
    strategies_used: dict[str, PitStrategy]
