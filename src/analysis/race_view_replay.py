"""Research-only dual-view race replay helpers.

This module keeps race-model evaluation separate from serving and persistence.
It runs the same candidate under two controlled grid conditions:

``conditional_actual_grid``
    Uses the official starting grid and no grid scenarios.  This isolates race
    physics from qualifying prediction quality.

``end_to_end_predicted_grid``
    Uses the predicted qualifying output. R1 candidates pass complete joint grid
    scenarios; qualifying-only candidates retain the champion marginal-grid path.
    This measures the full qualifying-to-race pipeline without changing the
    candidate's registered ablation scope.

Callers provide a predictor factory rather than a live predictor instance.  A
fresh predictor is created for each view at every seed so mutable runtime state
cannot leak between paired runs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from typing import Any, cast

from src.analysis.challenger_governance import DEFAULT_REPLAY_SEEDS
from src.analysis.model_evaluation import (
    compute_dnf_calibration,
    compute_prediction_accuracy,
)
from src.types.prediction_types import QualifyingGridEntry
from src.utils.grid_scenarios import grid_scenario_digest, validate_grid_scenarios
from src.utils.grid_validation import validate_qualifying_grid

DUAL_RACE_REPLAY_SCHEMA_VERSION = 1
CONDITIONAL_ACTUAL_GRID = "conditional_actual_grid"
END_TO_END_PREDICTED_GRID = "end_to_end_predicted_grid"
RACE_VIEW_NAMES = (CONDITIONAL_ACTUAL_GRID, END_TO_END_PREDICTED_GRID)

_RESERVED_RACE_KWARGS = frozenset({"qualifying_grid", "grid_scenarios", "grid_source_detail"})


def _normalise_seeds(seeds: Sequence[int]) -> list[int]:
    normalised = [int(seed) for seed in seeds]
    if not normalised:
        raise ValueError("seeds cannot be empty")
    if len(set(normalised)) != len(normalised):
        raise ValueError("seeds must be unique")
    return normalised


def _validate_complete_grid(
    grid: Sequence[QualifyingGridEntry | Mapping[str, Any]],
    *,
    field_name: str,
) -> list[QualifyingGridEntry]:
    try:
        validated = validate_qualifying_grid(
            grid,
            require_sequential_positions=True,
            max_position=len(grid),
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} is invalid: {exc}") from exc
    return sorted(validated, key=lambda row: int(row["position"]))


def _starting_grid_without_uncertainty(
    grid: Sequence[QualifyingGridEntry],
) -> list[QualifyingGridEntry]:
    """Strip prediction uncertainty while preserving official start metadata."""

    observed: list[QualifyingGridEntry] = []
    for row in grid:
        entry: QualifyingGridEntry = {
            "driver": str(row["driver"]),
            "team": str(row["team"]),
            "position": int(row["position"]),
        }
        if "start_type" in row:
            entry["start_type"] = row["start_type"]
        observed.append(entry)
    return observed


def _validate_prediction_result(
    result: Any,
    *,
    expected_drivers: Sequence[str],
    expected_grid_source_detail: str,
    expected_scenario_count: int,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("predict_race must return a dictionary")

    returned_detail = str(result.get("grid_source_detail", "")).strip()
    if returned_detail != expected_grid_source_detail:
        raise ValueError(
            "predict_race returned the wrong grid_source_detail "
            f"(expected={expected_grid_source_detail!r}, got={returned_detail!r})"
        )
    try:
        returned_scenario_count = int(cast(Any, result.get("grid_scenario_count")))
    except (TypeError, ValueError) as exc:
        raise ValueError("predict_race must report an integer grid_scenario_count") from exc
    if returned_scenario_count != expected_scenario_count:
        raise ValueError(
            "predict_race returned the wrong grid_scenario_count "
            f"(expected={expected_scenario_count}, got={returned_scenario_count})"
        )

    finish_order = result.get("finish_order")
    if not isinstance(finish_order, Sequence) or isinstance(finish_order, str | bytes):
        raise ValueError("predict_race result must contain a finish_order sequence")
    expected = {str(driver) for driver in expected_drivers}
    returned_drivers: list[str] = []
    returned_positions: list[int] = []
    for index, row in enumerate(finish_order):
        if not isinstance(row, Mapping):
            raise ValueError(f"finish_order row {index} must be a mapping")
        driver = str(row.get("driver", "")).strip()
        if not driver:
            raise ValueError(f"finish_order row {index} has no driver")
        try:
            position = int(cast(Any, row.get("position")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"finish_order row {index} has an invalid position") from exc
        returned_drivers.append(driver)
        returned_positions.append(position)

    if len(returned_drivers) != len(expected):
        raise ValueError(
            f"finish_order must contain {len(expected)} drivers, got {len(returned_drivers)}"
        )
    if len(set(returned_drivers)) != len(returned_drivers):
        raise ValueError("finish_order contains duplicate drivers")
    if set(returned_drivers) != expected:
        raise ValueError("finish_order driver set does not match the replay grid")
    if sorted(returned_positions) != list(range(1, len(expected) + 1)):
        raise ValueError("finish_order positions must be a complete sequential permutation")
    return result


def _check_factory_seed(predictor: Any, seed: int) -> None:
    """Verify a declared predictor seed when the implementation exposes one."""
    if not hasattr(predictor, "seed"):
        return
    try:
        predictor_seed = int(predictor.seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("predictor factory returned a predictor with an invalid seed") from exc
    if predictor_seed != seed:
        raise ValueError(
            f"predictor factory requested seed {seed}, but returned seed {predictor_seed}"
        )


def run_dual_race_replay(
    *,
    predictor_factory: Callable[[int], Any],
    actual_starting_grid: Sequence[QualifyingGridEntry | Mapping[str, Any]],
    predicted_qualifying_grid: Sequence[QualifyingGridEntry | Mapping[str, Any]],
    predicted_grid_scenarios: Sequence[Sequence[str]] | None,
    race_kwargs: Mapping[str, Any],
    seeds: Sequence[int] = DEFAULT_REPLAY_SEEDS,
) -> dict[str, Any]:
    """Run seed-matched actual-grid and predicted-grid race views.

    ``predictor_factory(seed)`` must return a fresh predictor configured for the
    candidate being evaluated.  Runtime model selection remains the caller's
    responsibility; this helper never changes configuration or artifacts.
    """
    normalised_seeds = _normalise_seeds(seeds)
    reserved = sorted(_RESERVED_RACE_KWARGS.intersection(race_kwargs))
    if reserved:
        raise ValueError(
            "race_kwargs cannot override controlled dual-view inputs: " + ", ".join(reserved)
        )

    actual_grid = _validate_complete_grid(
        actual_starting_grid,
        field_name="actual_starting_grid",
    )
    if not all("start_type" in row for row in actual_grid):
        raise ValueError("actual_starting_grid must identify every row as a grid or pit-lane start")
    predicted_grid = _validate_complete_grid(
        predicted_qualifying_grid,
        field_name="predicted_qualifying_grid",
    )
    actual_drivers = [str(row["driver"]) for row in actual_grid]
    predicted_drivers = [str(row["driver"]) for row in predicted_grid]
    if set(actual_drivers) != set(predicted_drivers):
        raise ValueError(
            "actual_starting_grid and predicted_qualifying_grid must contain the same drivers"
        )

    joint_scenarios = (
        validate_grid_scenarios(
            predicted_grid_scenarios,
            expected_drivers=predicted_drivers,
        )
        if predicted_grid_scenarios is not None
        else None
    )
    predicted_source_detail = (
        "predicted_joint" if joint_scenarios is not None else "predicted_marginal_fallback"
    )
    predicted_scenario_count = len(joint_scenarios or [])
    observed_grid = _starting_grid_without_uncertainty(actual_grid)
    common_kwargs = dict(race_kwargs)
    conditional_runs: list[dict[str, Any]] = []
    end_to_end_runs: list[dict[str, Any]] = []
    predictor_references: list[Any] = []

    for seed in normalised_seeds:
        conditional_predictor = predictor_factory(seed)
        end_to_end_predictor = predictor_factory(seed)
        if conditional_predictor is end_to_end_predictor:
            raise ValueError(
                "predictor_factory must return independent predictors for paired views"
            )
        predictor_references.extend((conditional_predictor, end_to_end_predictor))
        _check_factory_seed(conditional_predictor, seed)
        _check_factory_seed(end_to_end_predictor, seed)

        conditional_prediction = conditional_predictor.predict_race(
            qualifying_grid=observed_grid,
            grid_scenarios=None,
            grid_source_detail="actual_starting_grid",
            **common_kwargs,
        )
        conditional_prediction = _validate_prediction_result(
            conditional_prediction,
            expected_drivers=actual_drivers,
            expected_grid_source_detail="actual_starting_grid",
            expected_scenario_count=0,
        )

        end_to_end_prediction = end_to_end_predictor.predict_race(
            qualifying_grid=predicted_grid,
            grid_scenarios=joint_scenarios,
            grid_source_detail=predicted_source_detail,
            **common_kwargs,
        )
        end_to_end_prediction = _validate_prediction_result(
            end_to_end_prediction,
            expected_drivers=predicted_drivers,
            expected_grid_source_detail=predicted_source_detail,
            expected_scenario_count=predicted_scenario_count,
        )

        conditional_runs.append({"seed": seed, "prediction": conditional_prediction})
        end_to_end_runs.append({"seed": seed, "prediction": end_to_end_prediction})

    # Keep references alive until all runs finish so factories cannot accidentally
    # pass an identity check through Python object-id reuse.
    _ = predictor_references
    return {
        "artifact_type": "dual_race_replay",
        "schema_version": DUAL_RACE_REPLAY_SCHEMA_VERSION,
        "seeds": normalised_seeds,
        "views": {
            CONDITIONAL_ACTUAL_GRID: {
                "grid_source_detail": "actual_starting_grid",
                "grid_scenario_count": 0,
                "runs": conditional_runs,
            },
            END_TO_END_PREDICTED_GRID: {
                "grid_source_detail": predicted_source_detail,
                "grid_scenario_count": predicted_scenario_count,
                "grid_scenario_digest": (
                    grid_scenario_digest(joint_scenarios) if joint_scenarios is not None else None
                ),
                "runs": end_to_end_runs,
            },
        },
    }


def _mean_finite(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(fmean(finite)) if finite else None


def score_dual_race_replay(
    replay: Mapping[str, Any],
    *,
    actual_finish_order: Sequence[QualifyingGridEntry | Mapping[str, Any]],
) -> dict[str, Any]:
    """Score both replay views against one immutable race result.

    Metrics are reported per seed and as simple seed means.  ``finisher_mae``
    is included explicitly for the promotion gates, while DNF Brier remains
    ``None`` when the prediction payload has no DNF probabilities.
    """
    actual = _validate_complete_grid(actual_finish_order, field_name="actual_finish_order")
    actual_rows: list[dict[str, Any]] = [dict(row) for row in actual]
    actual_drivers = {str(row["driver"]) for row in actual}
    replay_seeds = _normalise_seeds(replay.get("seeds", []))
    raw_views = replay.get("views")
    if not isinstance(raw_views, Mapping):
        raise ValueError("replay must contain a views mapping")

    scored_views: dict[str, Any] = {}
    for view_name in RACE_VIEW_NAMES:
        raw_view = raw_views.get(view_name)
        if not isinstance(raw_view, Mapping):
            raise ValueError(f"replay is missing view {view_name!r}")
        raw_runs = raw_view.get("runs")
        if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, str | bytes):
            raise ValueError(f"replay view {view_name!r} must contain a runs sequence")

        per_seed: list[dict[str, Any]] = []
        seen_seeds: list[int] = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, Mapping):
                raise ValueError(f"replay view {view_name!r} contains an invalid run")
            seed = int(cast(Any, raw_run.get("seed")))
            prediction = raw_run.get("prediction")
            if not isinstance(prediction, Mapping):
                raise ValueError(f"replay view {view_name!r} run {seed} has no prediction")
            finish_order = prediction.get("finish_order")
            if not isinstance(finish_order, Sequence) or isinstance(finish_order, str | bytes):
                raise ValueError(f"replay view {view_name!r} run {seed} has no finish_order")
            predicted_rows: list[dict[str, Any]] = [
                dict(row) for row in finish_order if isinstance(row, Mapping)
            ]
            predicted_drivers = {str(row.get("driver", "")) for row in predicted_rows}
            if predicted_drivers != actual_drivers or len(predicted_rows) != len(actual):
                raise ValueError(
                    f"replay view {view_name!r} run {seed} driver set does not match actuals"
                )

            accuracy = compute_prediction_accuracy(predicted_rows, actual_rows)
            dnf = compute_dnf_calibration(predicted_rows, actual_rows)
            per_seed.append(
                {
                    "seed": seed,
                    "mae": float(accuracy["mae"]),
                    "finisher_mae": float(accuracy["finisher_mae"]),
                    "weighted_mae": float(accuracy["weighted_mae"]),
                    # Secondary top-heavy diagnostic (actual P1-3=3, P4-10=2, P11+=1);
                    # see src.analysis.model_evaluation.compute_prediction_accuracy.
                    "top_heavy_weighted_mae": float(accuracy["top_heavy_weighted_mae"]),
                    "winner_correct": float(accuracy["winner_correct"]),
                    "top3_accuracy_percent": float(accuracy["top_3_pct"]),
                    "top10_accuracy_percent": float(accuracy["top_10_pct"]),
                    "dnf_brier": (
                        float(dnf["brier_score"])
                        if math.isfinite(float(dnf["brier_score"]))
                        else None
                    ),
                }
            )
            seen_seeds.append(seed)

        if seen_seeds != replay_seeds:
            raise ValueError(f"replay view {view_name!r} seeds do not match the replay seed order")
        scored_views[view_name] = {
            "runs": per_seed,
            "mean": {
                "mae": _mean_finite([row["mae"] for row in per_seed]),
                "finisher_mae": _mean_finite([row["finisher_mae"] for row in per_seed]),
                "weighted_mae": _mean_finite([row["weighted_mae"] for row in per_seed]),
                "top_heavy_weighted_mae": _mean_finite(
                    [row["top_heavy_weighted_mae"] for row in per_seed]
                ),
                "winner_accuracy_percent": _mean_finite(
                    [row["winner_correct"] * 100.0 for row in per_seed]
                ),
                "top3_accuracy_percent": _mean_finite(
                    [row["top3_accuracy_percent"] for row in per_seed]
                ),
                "top10_accuracy_percent": _mean_finite(
                    [row["top10_accuracy_percent"] for row in per_seed]
                ),
                "dnf_brier": _mean_finite(
                    [row["dnf_brier"] for row in per_seed if row["dnf_brier"] is not None]
                ),
            },
        }

    return {
        "artifact_type": "dual_race_replay_metrics",
        "schema_version": DUAL_RACE_REPLAY_SCHEMA_VERSION,
        "seeds": replay_seeds,
        "views": scored_views,
    }


def extract_race_view_metric(
    report: Mapping[str, Any],
    *,
    metric: str = "finisher_mae",
) -> dict[str, float]:
    """Extract one finite aggregate metric for governance/bootstrap inputs."""
    raw_views = report.get("views")
    if not isinstance(raw_views, Mapping):
        raise ValueError("report must contain a views mapping")
    extracted: dict[str, float] = {}
    for view_name in RACE_VIEW_NAMES:
        raw_view = raw_views.get(view_name)
        if not isinstance(raw_view, Mapping) or not isinstance(raw_view.get("mean"), Mapping):
            raise ValueError(f"report is missing aggregate metrics for {view_name!r}")
        raw_value = raw_view["mean"].get(metric)
        if raw_value is None or not math.isfinite(float(raw_value)):
            raise ValueError(f"metric {metric!r} is not finite for {view_name!r}")
        extracted[view_name] = float(raw_value)
    return extracted
