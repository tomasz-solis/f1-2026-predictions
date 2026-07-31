"""Coherent qualifying-grid scenarios shared by qualifying and race prediction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256

import numpy as np

GRID_SOURCE_DETAILS = frozenset(
    {
        "predicted_joint",
        "predicted_marginal_fallback",
        "actual_qualifying",
        "actual_starting_grid",
    }
)


def validate_grid_scenarios(
    grid_scenarios: Sequence[Sequence[str]],
    *,
    expected_drivers: Sequence[str],
) -> list[list[str]]:
    """Validate complete driver permutations without changing scenario frequency."""
    expected = [str(driver).strip() for driver in expected_drivers]
    if not expected or any(not driver for driver in expected):
        raise ValueError("Expected grid-scenario drivers cannot be empty")
    if len(set(expected)) != len(expected):
        raise ValueError("Expected grid-scenario drivers must be unique")
    if not grid_scenarios:
        raise ValueError("grid_scenarios cannot be empty when provided")

    expected_set = set(expected)
    validated: list[list[str]] = []
    for scenario_index, raw_scenario in enumerate(grid_scenarios):
        if isinstance(raw_scenario, str | bytes):
            raise ValueError(f"Grid scenario {scenario_index} must be a driver sequence")
        scenario = [str(driver).strip() for driver in raw_scenario]
        if any(not driver for driver in scenario):
            raise ValueError(f"Grid scenario {scenario_index} contains an empty driver")
        if len(scenario) != len(expected):
            raise ValueError(
                f"Grid scenario {scenario_index} must contain {len(expected)} drivers, "
                f"got {len(scenario)}"
            )
        if len(set(scenario)) != len(scenario):
            raise ValueError(f"Grid scenario {scenario_index} contains duplicate drivers")
        if set(scenario) != expected_set:
            missing = sorted(expected_set - set(scenario))
            unexpected = sorted(set(scenario) - expected_set)
            raise ValueError(
                f"Grid scenario {scenario_index} driver set does not match the central grid "
                f"(missing={missing}, unexpected={unexpected})"
            )
        validated.append(scenario)
    return validated


def build_grid_scenarios(
    *,
    position_records: Mapping[str, Sequence[int]],
    expected_drivers: Sequence[str],
) -> list[list[str]]:
    """Reconstruct coherent grids by reading every driver's shared simulation index."""
    drivers = [str(driver).strip() for driver in expected_drivers]
    if not drivers:
        return []
    missing_records = [driver for driver in drivers if driver not in position_records]
    if missing_records:
        raise ValueError(f"Position records missing drivers: {sorted(missing_records)}")

    sample_counts = {len(position_records[driver]) for driver in drivers}
    if len(sample_counts) != 1:
        raise ValueError("Position records must share one simulation count across all drivers")
    sample_count = next(iter(sample_counts), 0)
    if sample_count <= 0:
        raise ValueError("Position records cannot be empty when building grid scenarios")

    field_size = len(drivers)
    expected_positions = set(range(1, field_size + 1))
    scenarios: list[list[str]] = []
    for simulation_index in range(sample_count):
        positions: dict[str, int] = {}
        for driver in drivers:
            try:
                positions[driver] = int(position_records[driver][simulation_index])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid position for {driver} at simulation {simulation_index}"
                ) from exc
        if set(positions.values()) != expected_positions:
            raise ValueError(
                f"Simulation {simulation_index} positions must be a complete 1-{field_size} "
                "permutation"
            )
        scenarios.append(sorted(drivers, key=positions.__getitem__))

    return validate_grid_scenarios(scenarios, expected_drivers=drivers)


def grid_scenario_digest(grid_scenarios: Sequence[Sequence[str]]) -> str:
    """Return a stable digest for persisted scenario provenance."""
    canonical = json.dumps(grid_scenarios, ensure_ascii=True, separators=(",", ":"))
    return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


def build_grid_scenario_schedule(
    *,
    scenario_count: int,
    simulation_count: int,
    base_seed: int,
) -> list[int]:
    """Schedule scenarios deterministically with usage counts differing by at most one."""
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    if simulation_count <= 0:
        raise ValueError("simulation_count must be positive")

    seed_material = f"joint-grid-schedule:{int(base_seed)}:{scenario_count}:{simulation_count}"
    schedule_seed = int(sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    schedule_rng = np.random.default_rng(schedule_seed)
    full_cycles, remainder = divmod(simulation_count, scenario_count)
    repeated = np.tile(np.arange(scenario_count, dtype=int), full_cycles)
    if remainder:
        remainder_indices = schedule_rng.permutation(scenario_count)[:remainder]
        repeated = np.concatenate((repeated, remainder_indices))
    schedule_rng.shuffle(repeated)
    return [int(index) for index in repeated]


def grid_scenario_positions(scenario: Sequence[str]) -> dict[str, int]:
    """Convert one ordered driver permutation into starting positions."""
    return {str(driver): position for position, driver in enumerate(scenario, start=1)}


def resolve_grid_source_detail(
    requested: str | None,
    *,
    has_joint_scenarios: bool,
    has_probabilistic_grid_fields: bool,
) -> str:
    """Resolve detailed grid provenance while retaining the legacy grid_source field."""
    if requested is None or not str(requested).strip():
        if has_joint_scenarios:
            return "predicted_joint"
        if has_probabilistic_grid_fields:
            return "predicted_marginal_fallback"
        return "actual_qualifying"

    normalized = str(requested).strip().lower()
    if normalized not in GRID_SOURCE_DETAILS:
        raise ValueError(
            f"grid_source_detail must be one of {sorted(GRID_SOURCE_DETAILS)}, got {requested!r}"
        )
    if has_joint_scenarios and normalized != "predicted_joint":
        raise ValueError("Joint grid scenarios require grid_source_detail='predicted_joint'")
    if not has_joint_scenarios and normalized == "predicted_joint":
        raise ValueError("grid_source_detail='predicted_joint' requires grid_scenarios")
    return normalized
