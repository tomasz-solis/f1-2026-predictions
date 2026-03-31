"""Regression tests that prove seeded Bayesian form reaches the prediction path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.predictors.baseline.qualifying_preparation import resolve_bayesian_skill_score
from src.predictors.baseline_2026 import Baseline2026Predictor

_DRIVER_CHARACTERISTICS_PATH = Path("data/processed/driver_characteristics.json")
_BAYESIAN_IMPACT_RACE = "Japanese Grand Prix"


def _load_driver_characteristics() -> dict[str, Any]:
    """Load committed driver characteristics or skip when the artifact is absent."""
    if not _DRIVER_CHARACTERISTICS_PATH.exists():
        pytest.skip("Driver characteristics artifact not present")
    return json.loads(_DRIVER_CHARACTERISTICS_PATH.read_text())


def test_bayesian_state_influences_qualifying_predictions() -> None:
    """Removing seeded Bayesian state should measurably change later-race grids."""
    predictor_with_bayesian = Baseline2026Predictor(seed=42)
    races_completed = predictor_with_bayesian._get_contextual_races_completed(_BAYESIAN_IMPACT_RACE)
    if races_completed <= 0:
        pytest.skip(
            f"{_BAYESIAN_IMPACT_RACE} still has zero completed races in committed data, "
            "so Bayesian pace blending is correctly inactive."
        )

    result_with = predictor_with_bayesian.predict_qualifying(
        2026,
        _BAYESIAN_IMPACT_RACE,
        n_simulations=80,
    )
    grid_with = {entry["driver"]: entry["position"] for entry in result_with["grid"]}

    predictor_without_bayesian = Baseline2026Predictor(seed=42)
    for driver_data in predictor_without_bayesian.drivers.values():
        if isinstance(driver_data, dict):
            driver_data.pop("bayesian", None)

    result_without = predictor_without_bayesian.predict_qualifying(
        2026,
        _BAYESIAN_IMPACT_RACE,
        n_simulations=80,
    )
    grid_without = {entry["driver"]: entry["position"] for entry in result_without["grid"]}

    drivers_changed = sum(
        1
        for driver_code, position in grid_with.items()
        if position != grid_without.get(driver_code)
    )
    assert drivers_changed >= 3, (
        f"Only {drivers_changed} drivers changed position with Bayesian state removed "
        f"for {_BAYESIAN_IMPACT_RACE} after {races_completed} completed race(s)."
    )


def test_bayesian_skill_score_resolves_from_seeded_state() -> None:
    """Every committed driver should expose a normalized Bayesian skill score."""
    characteristics = _load_driver_characteristics()
    drivers = characteristics.get("drivers", {})
    assert isinstance(drivers, dict) and drivers

    grid_size = len(drivers)
    resolved_count = 0

    for driver_code, driver_info in drivers.items():
        assert isinstance(driver_info, dict), f"{driver_code} payload should be a dict"
        score = resolve_bayesian_skill_score(driver_info, grid_size=grid_size)
        if score is not None:
            resolved_count += 1
            assert 0.0 <= score <= 1.0, f"{driver_code} Bayesian score out of range: {score}"

    assert resolved_count == len(drivers), (
        f"Only {resolved_count}/{len(drivers)} drivers resolved Bayesian score from "
        "committed seeded state."
    )


def test_bayesian_ratings_reflect_driver_form_not_just_car() -> None:
    """Teammate-relative updates should elevate the stronger driver signal across teams."""
    from src.models.bayesian import BayesianDriverRanking, DriverPrior

    priors = {
        "FAST_A": DriverPrior("1", "FAST_A", "TopTeam", "top", mu=16.0, sigma=2.0),
        "FAST_B": DriverPrior("2", "FAST_B", "TopTeam", "top", mu=14.0, sigma=2.0),
        "SLOW_A": DriverPrior("3", "SLOW_A", "BackTeam", "backmarker", mu=8.0, sigma=2.0),
        "SLOW_B": DriverPrior("4", "SLOW_B", "BackTeam", "backmarker", mu=6.0, sigma=2.0),
    }
    lineups = {
        "TopTeam": ["FAST_A", "FAST_B"],
        "BackTeam": ["SLOW_A", "SLOW_B"],
    }

    ranker = BayesianDriverRanking(priors, grid_size=22)

    for race_num in range(5):
        ranker.update_teammate_relative(
            observations={
                "FAST_A": 1,
                "FAST_B": 6,
                "SLOW_A": 12,
                "SLOW_B": 17,
            },
            session_name=f"race_{race_num}",
            lineups=lineups,
            confidence=1.0,
        )

    slow_a_mu = ranker.ratings["SLOW_A"][0]
    fast_b_mu = ranker.ratings["FAST_B"][0]

    assert slow_a_mu > fast_b_mu, (
        f"SLOW_A (mu={slow_a_mu:.2f}) should rate higher than FAST_B (mu={fast_b_mu:.2f}) "
        "because SLOW_A consistently dominates their teammate while FAST_B underperforms."
    )
