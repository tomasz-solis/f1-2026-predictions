"""Tests for backtesting and ablation helpers."""

from __future__ import annotations

import pytest

from src.utils.backtesting import (
    aggregate_race_metrics,
    apply_config_overrides,
    parse_experiment_spec,
    rank_experiments_for_generalization,
    run_previous_race_naive_backtest,
    run_single_race_backtest,
    summarize_generalization,
)


class _StubPredictor:
    def predict_qualifying(self, year: int, race_name: str, n_simulations: int) -> dict:
        return {
            "grid": [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ]
        }

    def predict_race(
        self,
        qualifying_grid: list[dict],
        weather: str,
        race_name: str,
        n_simulations: int,
    ) -> dict:
        return {
            "finish_order": [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ]
        }


def test_parse_experiment_spec_parses_typed_overrides():
    name, overrides = parse_experiment_spec(
        'candidate:alpha=1,beta=0.5,enabled=true,tags=["a","b"],label=baseline'
    )

    assert name == "candidate"
    assert overrides["alpha"] == 1
    assert overrides["beta"] == pytest.approx(0.5)
    assert overrides["enabled"] is True
    assert overrides["tags"] == ["a", "b"]
    assert overrides["label"] == "baseline"


def test_parse_experiment_spec_requires_key_value_assignments():
    with pytest.raises(ValueError, match="key=value"):
        parse_experiment_spec("invalid:no_equals_here")


def test_apply_config_overrides_sets_nested_values():
    base = {
        "baseline_predictor": {
            "race": {
                "grid_anchor": {"base": 0.40},
            }
        }
    }

    merged = apply_config_overrides(
        base,
        {
            "baseline_predictor.race.grid_anchor.base": 0.45,
            "baseline_predictor.race.safety_car_luck_range": 0.20,
        },
    )

    # Original should remain unchanged.
    assert base["baseline_predictor"]["race"]["grid_anchor"]["base"] == pytest.approx(0.40)

    assert merged["baseline_predictor"]["race"]["grid_anchor"]["base"] == pytest.approx(0.45)
    assert merged["baseline_predictor"]["race"]["safety_car_luck_range"] == pytest.approx(0.20)


def test_run_single_race_backtest_returns_metrics_for_successful_race():
    def _fetcher(year: int, race_name: str, session_name: str):
        assert year == 2025
        assert race_name == "Bahrain Grand Prix"
        if session_name == "Q":
            return [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ]
        if session_name == "R":
            return [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ]
        raise AssertionError("Unexpected session requested")

    result = run_single_race_backtest(
        predictor=_StubPredictor(),
        year=2025,
        race_name="Bahrain Grand Prix",
        weather="dry",
        qualifying_simulations=10,
        race_simulations=10,
        results_fetcher=_fetcher,
    )

    assert result["status"] == "ok"
    assert result["qualifying_mae"] == pytest.approx(0.0)
    assert result["race_mae"] == pytest.approx(0.0)
    assert result["top3_accuracy"] == pytest.approx(100.0)
    assert result["winner_correct"] is True


def test_run_single_race_backtest_marks_missing_actuals_as_skipped():
    def _fetcher(_year: int, _race_name: str, _session_name: str):
        return None

    result = run_single_race_backtest(
        predictor=_StubPredictor(),
        year=2025,
        race_name="Bahrain Grand Prix",
        weather="dry",
        qualifying_simulations=10,
        race_simulations=10,
        results_fetcher=_fetcher,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_actual_results"


def test_run_previous_race_naive_backtest_uses_prior_actual_results():
    actuals = {
        ("Australian Grand Prix", "Q"): [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
        ("Australian Grand Prix", "R"): [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
        ("Chinese Grand Prix", "Q"): [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
        ("Chinese Grand Prix", "R"): [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
    }

    def _fetcher(_year: int, race_name: str, session_name: str):
        return actuals.get((race_name, session_name))

    result = run_previous_race_naive_backtest(
        year=2025,
        race_names=["Australian Grand Prix", "Chinese Grand Prix"],
        results_fetcher=_fetcher,
    )

    assert result["name"] == "previous_race_classification"
    assert result["summary"]["races_total"] == 2
    assert result["summary"]["races_evaluated"] == 1
    assert result["summary"]["races_skipped"] == 1
    assert result["summary"]["qualifying_mae_mean"] == pytest.approx(0.0)
    assert result["summary"]["race_mae_mean"] == pytest.approx(0.0)

    skipped, evaluated = result["race_results"]
    assert skipped["reason"] == "missing_previous_race_results"
    assert evaluated["status"] == "ok"
    assert evaluated["predicted_from_race"] == "Australian Grand Prix"
    assert evaluated["top3_accuracy"] == pytest.approx(100.0)


def test_run_previous_race_naive_backtest_resets_after_missing_actuals():
    actuals = {
        ("Australian Grand Prix", "Q"): [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
        ("Australian Grand Prix", "R"): [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
        ("Japanese Grand Prix", "Q"): [
            {"driver": "NOR", "team": "McLaren", "position": 1},
            {"driver": "VER", "team": "Red Bull Racing", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
        ("Japanese Grand Prix", "R"): [
            {"driver": "NOR", "team": "McLaren", "position": 1},
            {"driver": "VER", "team": "Red Bull Racing", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ],
    }

    def _fetcher(_year: int, race_name: str, session_name: str):
        return actuals.get((race_name, session_name))

    result = run_previous_race_naive_backtest(
        year=2025,
        race_names=[
            "Australian Grand Prix",
            "Chinese Grand Prix",
            "Japanese Grand Prix",
        ],
        results_fetcher=_fetcher,
    )

    assert [row["reason"] for row in result["race_results"] if row["status"] == "skipped"] == [
        "missing_previous_race_results",
        "missing_actual_results",
        "missing_previous_race_results",
    ]
    assert result["summary"]["races_evaluated"] == 0


def test_rank_experiments_recommends_only_generalizing_improvements():
    baseline_races = [
        {
            "race_name": "A",
            "status": "ok",
            "race_mae": 2.0,
            "top3_accuracy": 60.0,
            "winner_correct": True,
        },
        {
            "race_name": "B",
            "status": "ok",
            "race_mae": 2.4,
            "top3_accuracy": 55.0,
            "winner_correct": False,
        },
        {
            "race_name": "C",
            "status": "ok",
            "race_mae": 2.2,
            "top3_accuracy": 50.0,
            "winner_correct": False,
        },
        {
            "race_name": "D",
            "status": "ok",
            "race_mae": 2.3,
            "top3_accuracy": 58.0,
            "winner_correct": True,
        },
    ]
    better_races = [
        {
            "race_name": "A",
            "status": "ok",
            "race_mae": 1.8,
            "top3_accuracy": 66.0,
            "winner_correct": True,
        },
        {
            "race_name": "B",
            "status": "ok",
            "race_mae": 2.0,
            "top3_accuracy": 60.0,
            "winner_correct": False,
        },
        {
            "race_name": "C",
            "status": "ok",
            "race_mae": 2.0,
            "top3_accuracy": 55.0,
            "winner_correct": False,
        },
        {
            "race_name": "D",
            "status": "ok",
            "race_mae": 2.1,
            "top3_accuracy": 62.0,
            "winner_correct": True,
        },
    ]
    overfit_races = [
        {
            "race_name": "A",
            "status": "ok",
            "race_mae": 1.1,
            "top3_accuracy": 70.0,
            "winner_correct": True,
        },
        {
            "race_name": "B",
            "status": "ok",
            "race_mae": 1.2,
            "top3_accuracy": 68.0,
            "winner_correct": True,
        },
        {
            "race_name": "C",
            "status": "ok",
            "race_mae": 3.0,
            "top3_accuracy": 45.0,
            "winner_correct": False,
        },
        {
            "race_name": "D",
            "status": "ok",
            "race_mae": 3.2,
            "top3_accuracy": 42.0,
            "winner_correct": False,
        },
    ]

    reports = [
        {
            "name": "baseline",
            "overrides": {},
            "summary": aggregate_race_metrics(baseline_races),
            "generalization": summarize_generalization(baseline_races, train_fraction=0.5, seed=7),
            "race_results": baseline_races,
        },
        {
            "name": "better",
            "overrides": {"baseline_predictor.race.grid_anchor.base": 0.45},
            "summary": aggregate_race_metrics(better_races),
            "generalization": summarize_generalization(better_races, train_fraction=0.5, seed=7),
            "race_results": better_races,
        },
        {
            "name": "overfit",
            "overrides": {"baseline_predictor.race.grid_anchor.base": 0.30},
            "summary": aggregate_race_metrics(overfit_races),
            "generalization": summarize_generalization(overfit_races, train_fraction=0.5, seed=7),
            "race_results": overfit_races,
        },
    ]

    ranked = rank_experiments_for_generalization(
        reports,
        min_test_race_mae_improvement=0.1,
        max_generalization_gap=0.35,
    )

    better = next(row for row in ranked if row["name"] == "better")
    overfit = next(row for row in ranked if row["name"] == "overfit")

    assert better["recommended"] is True
    assert overfit["recommended"] is False
