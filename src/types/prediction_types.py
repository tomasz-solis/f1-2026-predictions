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


class DriverRaceInfo(TypedDict):
    """Driver information for race simulation."""

    driver: str
    team: str
    grid_pos: int
    team_strength: float
    team_strength_by_compound: NotRequired[dict[str, float]]
    tire_deg_by_compound: NotRequired[dict[str, float]]
    skill: float
    race_advantage: float
    overtaking_skill: float
    defensive_skill: float
    dnf_probability: float
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
