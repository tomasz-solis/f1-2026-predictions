from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.qualifying_practice_challenger import (
    FittedQualifyingPracticeModel,
    calibrate_temperature,
    fit_bradley_terry_model,
    load_qualifying_practice_model,
    save_qualifying_practice_model,
    simulate_plackett_luce_grids,
)


def _training_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_id, offset in (("e1", 0.0), ("e2", 0.2), ("e3", -0.1)):
        for driver, pace, position in (
            ("AAA", 90.0 + offset, 1),
            ("BBB", 90.4 + offset, 2),
            ("CCC", 91.0 + offset, 3),
        ):
            rows.append(
                {
                    "event_id": event_id,
                    "driver": driver,
                    "actual_position": position,
                    "pace": pace,
                }
            )
    return pd.DataFrame(rows)


def test_fit_bradley_terry_model_learns_symmetric_pace_utility() -> None:
    model = fit_bradley_terry_model(
        _training_rows(),
        checkpoint="FP2",
        feature_columns=("pace",),
    )
    utilities = model.utilities(
        pd.DataFrame(
            [
                {"driver": "AAA", "pace": 90.0},
                {"driver": "BBB", "pace": 90.5},
            ]
        )
    )

    assert model.training_events == 3
    assert model.coefficients[0] < 0
    assert utilities["AAA"] > utilities["BBB"]


def test_plackett_luce_samples_are_coherent_and_reproducible() -> None:
    utilities = {"AAA": 1.0, "BBB": 0.5, "CCC": 0.0}
    first = simulate_plackett_luce_grids(
        utilities=utilities,
        n_simulations=50,
        rng=np.random.default_rng(17),
        temperature=0.5,
    )
    second = simulate_plackett_luce_grids(
        utilities=utilities,
        n_simulations=50,
        rng=np.random.default_rng(17),
        temperature=0.5,
    )

    assert first == second
    records, scenarios = first
    assert all(sorted(scenario) == ["AAA", "BBB", "CCC"] for scenario in scenarios)
    for sample_index, scenario in enumerate(scenarios):
        for position, driver in enumerate(scenario, start=1):
            assert records[driver][sample_index] == position

    explicit_empty = simulate_plackett_luce_grids(
        utilities=utilities,
        n_simulations=50,
        rng=np.random.default_rng(17),
        temperature=0.5,
        utility_candidates_by_driver={},
    )
    assert explicit_empty == first


def test_small_noisy_gap_does_not_create_extreme_head_to_head() -> None:
    records, _ = simulate_plackett_luce_grids(
        utilities={"HAD": 0.02, "VER": 0.0},
        n_simulations=20_000,
        rng=np.random.default_rng(42),
        temperature=0.10,
        utility_sigma_by_driver={"HAD": 0.40, "VER": 0.40},
    )

    had_ahead = np.mean(
        np.asarray(records["HAD"], dtype=int) < np.asarray(records["VER"], dtype=int)
    )
    assert 0.45 <= had_ahead <= 0.58


def test_plackett_luce_bootstraps_run_utilities_before_execution_noise() -> None:
    arguments = {
        "utilities": {"AAA": 0.0, "BBB": 0.0},
        "n_simulations": 200,
        "temperature": 1e-9,
        "utility_candidates_by_driver": {
            "AAA": [2.0, -2.0],
            "BBB": [0.0],
        },
    }

    first = simulate_plackett_luce_grids(rng=np.random.default_rng(91), **arguments)
    second = simulate_plackett_luce_grids(rng=np.random.default_rng(91), **arguments)

    assert first == second
    aaa_positions = set(first[0]["AAA"])
    assert aaa_positions == {1, 2}


def test_plackett_luce_rejects_nonfinite_run_utility_candidate() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        simulate_plackett_luce_grids(
            utilities={"AAA": 0.0, "BBB": 0.0},
            n_simulations=10,
            rng=np.random.default_rng(17),
            temperature=0.5,
            utility_candidates_by_driver={"AAA": [float("nan")]},
        )


def test_temperature_is_chosen_by_validation_log_loss() -> None:
    temperature = calibrate_temperature(
        utility_differences=np.asarray([0.2, 0.1, -0.2, -0.1]),
        outcomes=np.asarray([1.0, 1.0, 0.0, 0.0]),
        candidates=(0.25, 1.0, 3.0),
    )

    assert temperature == 0.25


def test_json_artifact_round_trip(tmp_path) -> None:
    model = fit_bradley_terry_model(
        _training_rows(),
        checkpoint="FP2",
        feature_columns=("pace",),
    )
    path = save_qualifying_practice_model(model, tmp_path / "model.json")
    loaded = load_qualifying_practice_model(path)

    assert isinstance(loaded, FittedQualifyingPracticeModel)
    assert loaded.to_dict() == model.to_dict()


def test_model_rejects_missing_inference_feature() -> None:
    model = fit_bradley_terry_model(
        _training_rows(),
        checkpoint="FP2",
        feature_columns=("pace",),
    )

    with pytest.raises(ValueError, match="Missing qualifying practice inference columns"):
        model.utilities(pd.DataFrame([{"driver": "AAA"}]))


def test_lap_level_features_are_event_relative_and_missing_is_neutral() -> None:
    model = FittedQualifyingPracticeModel(
        checkpoint="FP2",
        feature_columns=("best_adjusted_lap_s",),
        coefficients=(-1.0,),
        feature_medians=(0.0,),
        feature_scales=(1.0,),
        temperature=1.0,
        training_events=30,
        generated_at="2026-01-01T00:00:00Z",
    )

    utilities = model.utilities(
        pd.DataFrame(
            [
                {"driver": "AAA", "best_adjusted_lap_s": 100.0},
                {"driver": "BBB", "best_adjusted_lap_s": 101.0},
                {"driver": "CCC", "best_adjusted_lap_s": None},
            ]
        )
    )

    assert utilities == pytest.approx({"AAA": 0.5, "BBB": -0.5, "CCC": 0.0})
