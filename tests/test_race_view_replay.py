"""Tests for research-only dual race evaluation views."""

from __future__ import annotations

from typing import Any

import pytest

from src.analysis.race_view_replay import (
    CONDITIONAL_ACTUAL_GRID,
    END_TO_END_PREDICTED_GRID,
    extract_race_view_metric,
    run_dual_race_replay,
    score_dual_race_replay,
)
from src.utils.grid_scenarios import grid_scenario_digest

ACTUAL_GRID = [
    {
        "driver": "A",
        "team": "One",
        "position": 1,
        "p5": 1,
        "p95": 2,
        "start_type": "grid",
    },
    {
        "driver": "B",
        "team": "Two",
        "position": 2,
        "p5": 1,
        "p95": 3,
        "start_type": "grid",
    },
    {
        "driver": "C",
        "team": "Three",
        "position": 3,
        "p5": 2,
        "p95": 3,
        "start_type": "grid",
    },
]
PREDICTED_GRID = [
    {"driver": "B", "team": "Two", "position": 1, "p5": 1, "p95": 2},
    {"driver": "A", "team": "One", "position": 2, "p5": 1, "p95": 3},
    {"driver": "C", "team": "Three", "position": 3, "p5": 2, "p95": 3},
]
SCENARIOS = [["B", "A", "C"], ["A", "C", "B"]]


class _StubPredictor:
    def __init__(self, seed: int, calls: list[dict[str, Any]]) -> None:
        self.seed = seed
        self.calls = calls

    def predict_race(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"predictor_id": id(self), "seed": self.seed, **kwargs})
        grid = kwargs["qualifying_grid"]
        detail = kwargs["grid_source_detail"]
        scenarios = kwargs["grid_scenarios"]
        return {
            "finish_order": [
                {
                    "driver": row["driver"],
                    "team": row["team"],
                    "position": position,
                    "dnf_probability": 0.1,
                }
                for position, row in enumerate(grid, start=1)
            ],
            "grid_source_detail": detail,
            "grid_scenario_count": len(scenarios or []),
        }


def test_run_dual_race_replay_uses_matched_seeds_and_controlled_grid_inputs():
    calls: list[dict[str, Any]] = []

    replay = run_dual_race_replay(
        predictor_factory=lambda seed: _StubPredictor(seed, calls),
        actual_starting_grid=ACTUAL_GRID,
        predicted_qualifying_grid=PREDICTED_GRID,
        predicted_grid_scenarios=SCENARIOS,
        race_kwargs={"race_name": "Test GP", "year": 2026, "n_simulations": 25},
    )

    assert replay["seeds"] == [17, 42, 91]
    assert [run["seed"] for run in replay["views"][CONDITIONAL_ACTUAL_GRID]["runs"]] == [
        17,
        42,
        91,
    ]
    assert [run["seed"] for run in replay["views"][END_TO_END_PREDICTED_GRID]["runs"]] == [
        17,
        42,
        91,
    ]
    assert replay["views"][END_TO_END_PREDICTED_GRID]["grid_scenario_count"] == 2
    assert replay["views"][END_TO_END_PREDICTED_GRID][
        "grid_scenario_digest"
    ] == grid_scenario_digest(SCENARIOS)

    assert len(calls) == 6
    for offset, seed in enumerate((17, 42, 91)):
        conditional_call = calls[offset * 2]
        end_to_end_call = calls[offset * 2 + 1]
        assert conditional_call["seed"] == end_to_end_call["seed"] == seed
        assert conditional_call["predictor_id"] != end_to_end_call["predictor_id"]
        assert conditional_call["grid_source_detail"] == "actual_starting_grid"
        assert conditional_call["grid_scenarios"] is None
        assert all("p5" not in row for row in conditional_call["qualifying_grid"])
        assert end_to_end_call["grid_source_detail"] == "predicted_joint"
        assert end_to_end_call["grid_scenarios"] == SCENARIOS
        assert end_to_end_call["qualifying_grid"] == PREDICTED_GRID
        assert conditional_call["n_simulations"] == end_to_end_call["n_simulations"] == 25


def test_conditional_view_preserves_official_pit_lane_start_metadata() -> None:
    calls: list[dict[str, Any]] = []
    official_grid = [
        {**ACTUAL_GRID[0], "start_type": "grid"},
        {**ACTUAL_GRID[1], "start_type": "grid"},
        {**ACTUAL_GRID[2], "start_type": "pit_lane"},
    ]

    run_dual_race_replay(
        predictor_factory=lambda seed: _StubPredictor(seed, calls),
        actual_starting_grid=official_grid,
        predicted_qualifying_grid=PREDICTED_GRID,
        predicted_grid_scenarios=None,
        race_kwargs={"n_simulations": 5},
        seeds=[17],
    )

    conditional_grid = calls[0]["qualifying_grid"]
    assert [row["start_type"] for row in conditional_grid] == [
        "grid",
        "grid",
        "pit_lane",
    ]
    assert all("p5" not in row and "p95" not in row for row in conditional_grid)


def test_conditional_view_rejects_qualifying_classification_mislabeled_as_starting_grid() -> None:
    classification = [
        {key: value for key, value in row.items() if key != "start_type"} for row in ACTUAL_GRID
    ]

    with pytest.raises(ValueError, match="must identify every row"):
        run_dual_race_replay(
            predictor_factory=lambda seed: _StubPredictor(seed, []),
            actual_starting_grid=classification,
            predicted_qualifying_grid=PREDICTED_GRID,
            predicted_grid_scenarios=None,
            race_kwargs={"n_simulations": 5},
            seeds=[17],
        )


def test_run_dual_race_replay_supports_qualifying_only_marginal_grid_path():
    calls: list[dict[str, Any]] = []

    replay = run_dual_race_replay(
        predictor_factory=lambda seed: _StubPredictor(seed, calls),
        actual_starting_grid=ACTUAL_GRID,
        predicted_qualifying_grid=PREDICTED_GRID,
        predicted_grid_scenarios=None,
        race_kwargs={"n_simulations": 5},
        seeds=[17],
    )

    predicted_view = replay["views"][END_TO_END_PREDICTED_GRID]
    assert predicted_view["grid_source_detail"] == "predicted_marginal_fallback"
    assert predicted_view["grid_scenario_count"] == 0
    assert predicted_view["grid_scenario_digest"] is None
    assert calls[1]["grid_source_detail"] == "predicted_marginal_fallback"
    assert calls[1]["grid_scenarios"] is None


@pytest.mark.parametrize(
    ("actual_grid", "predicted_grid", "scenarios", "match"),
    [
        (
            [ACTUAL_GRID[0], {**ACTUAL_GRID[1], "position": 3}, ACTUAL_GRID[2]],
            PREDICTED_GRID,
            SCENARIOS,
            "actual_starting_grid is invalid",
        ),
        (
            ACTUAL_GRID,
            [PREDICTED_GRID[0], PREDICTED_GRID[1], {**PREDICTED_GRID[2], "driver": "D"}],
            [["B", "A", "D"]],
            "must contain the same drivers",
        ),
        (
            ACTUAL_GRID,
            PREDICTED_GRID,
            [["B", "A", "A"]],
            "duplicate drivers",
        ),
    ],
)
def test_run_dual_race_replay_rejects_invalid_grids_and_scenarios(
    actual_grid: list[dict[str, Any]],
    predicted_grid: list[dict[str, Any]],
    scenarios: list[list[str]],
    match: str,
):
    with pytest.raises(ValueError, match=match):
        run_dual_race_replay(
            predictor_factory=lambda seed: _StubPredictor(seed, []),
            actual_starting_grid=actual_grid,
            predicted_qualifying_grid=predicted_grid,
            predicted_grid_scenarios=scenarios,
            race_kwargs={"n_simulations": 5},
        )


def test_run_dual_race_replay_rejects_controlled_kwargs_override():
    with pytest.raises(ValueError, match="cannot override"):
        run_dual_race_replay(
            predictor_factory=lambda seed: _StubPredictor(seed, []),
            actual_starting_grid=ACTUAL_GRID,
            predicted_qualifying_grid=PREDICTED_GRID,
            predicted_grid_scenarios=SCENARIOS,
            race_kwargs={"grid_source_detail": "actual_qualifying"},
        )


class _WrongMetadataPredictor(_StubPredictor):
    def predict_race(self, **kwargs: Any) -> dict[str, Any]:
        result = super().predict_race(**kwargs)
        result["grid_source_detail"] = "actual_qualifying"
        return result


def test_run_dual_race_replay_rejects_prediction_provenance_mismatch():
    with pytest.raises(ValueError, match="wrong grid_source_detail"):
        run_dual_race_replay(
            predictor_factory=lambda seed: _WrongMetadataPredictor(seed, []),
            actual_starting_grid=ACTUAL_GRID,
            predicted_qualifying_grid=PREDICTED_GRID,
            predicted_grid_scenarios=SCENARIOS,
            race_kwargs={"n_simulations": 5},
            seeds=[17],
        )


def test_score_dual_race_replay_reports_both_views_and_extracts_finisher_mae():
    calls: list[dict[str, Any]] = []
    replay = run_dual_race_replay(
        predictor_factory=lambda seed: _StubPredictor(seed, calls),
        actual_starting_grid=ACTUAL_GRID,
        predicted_qualifying_grid=PREDICTED_GRID,
        predicted_grid_scenarios=SCENARIOS,
        race_kwargs={"n_simulations": 5},
        seeds=[17, 42],
    )
    actual_finish = [
        {"driver": "A", "team": "One", "position": 1},
        {"driver": "B", "team": "Two", "position": 2},
        {"driver": "C", "team": "Three", "position": 3, "dnf": True},
    ]

    report = score_dual_race_replay(replay, actual_finish_order=actual_finish)

    conditional = report["views"][CONDITIONAL_ACTUAL_GRID]["mean"]
    end_to_end = report["views"][END_TO_END_PREDICTED_GRID]["mean"]
    assert conditional["mae"] == pytest.approx(0.0)
    assert conditional["finisher_mae"] == pytest.approx(0.0)
    assert conditional["winner_accuracy_percent"] == pytest.approx(100.0)
    assert conditional["top3_accuracy_percent"] == pytest.approx(100.0)
    assert conditional["dnf_brier"] == pytest.approx(0.2766666667)
    assert end_to_end["mae"] == pytest.approx(2.0 / 3.0)
    assert end_to_end["finisher_mae"] == pytest.approx(1.0)
    assert end_to_end["winner_accuracy_percent"] == pytest.approx(0.0)
    assert end_to_end["top3_accuracy_percent"] == pytest.approx(100.0)
    assert extract_race_view_metric(report) == {
        CONDITIONAL_ACTUAL_GRID: pytest.approx(0.0),
        END_TO_END_PREDICTED_GRID: pytest.approx(1.0),
    }


def test_score_dual_race_replay_rejects_unmatched_view_seeds():
    replay = run_dual_race_replay(
        predictor_factory=lambda seed: _StubPredictor(seed, []),
        actual_starting_grid=ACTUAL_GRID,
        predicted_qualifying_grid=PREDICTED_GRID,
        predicted_grid_scenarios=SCENARIOS,
        race_kwargs={"n_simulations": 5},
        seeds=[17, 42],
    )
    replay["views"][END_TO_END_PREDICTED_GRID]["runs"].reverse()

    with pytest.raises(ValueError, match="seeds do not match"):
        score_dual_race_replay(replay, actual_finish_order=ACTUAL_GRID)
