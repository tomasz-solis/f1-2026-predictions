"""Tests for backtesting and ablation helpers."""

from __future__ import annotations

import pytest

from src.utils.backtesting import (
    _normalize_ranked_entries,
    aggregate_race_metrics,
    apply_config_overrides,
    build_checked_backtest_summary,
    build_error_analysis,
    build_overlap_comparison,
    build_segment_breakdown,
    get_races_for_year,
    parse_experiment_spec,
    rank_experiments_for_generalization,
    run_previous_race_naive_backtest,
    run_single_race_backtest,
    summarize_generalization,
    warm_fastf1_results_cache,
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


def test_normalize_ranked_entries_preserves_adjustment_diagnostics():
    """Prediction payload snapshots should keep model adjustment diagnostics."""
    rows = [
        {
            "driver": "NOR",
            "team": "McLaren",
            "position": 2,
            "median_position": 2,
            "qualifying_residual_adjustment": 0.5,
            "race_residual_adjustment": "-0.25",
            "learned_position_adjustment": 1,
        }
    ]

    normalized = _normalize_ranked_entries(rows, preserve_interval_fields=True)

    assert normalized == [
        {
            "driver": "NOR",
            "team": "McLaren",
            "position": 2,
            "median_position": 2,
            "qualifying_residual_adjustment": 0.5,
            "race_residual_adjustment": -0.25,
            "learned_position_adjustment": 1.0,
        }
    ]


class _LearningSystemStub:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def update_from_prediction_record(self, prediction_record: dict) -> dict:
        self.records.append(prediction_record)
        return {
            "sessions_updated": 2,
            "driver_updates": 6,
            "pair_updates": 2,
            "details": [],
            "skipped": False,
            "skip_reason": None,
        }


class _ContextAwarePredictor:
    def __init__(self) -> None:
        self.seed = 11
        self.qualifying_contexts: list[object] = []
        self.race_contexts: list[object] = []
        self.calibration_system = _LearningSystemStub()
        self.replayed_weekends: list[dict[str, object]] = []

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int,
        prediction_context=None,
    ) -> dict:
        self.qualifying_contexts.append(prediction_context)
        base = _StubPredictor().predict_qualifying(year, race_name, n_simulations)
        return {
            "grid": [
                {
                    **row,
                    "median_position": row["position"],
                    "p5": max(1, row["position"] - 1),
                    "p95": row["position"] + 1,
                }
                for row in base["grid"]
            ]
        }

    def predict_race(
        self,
        qualifying_grid: list[dict],
        weather: str,
        race_name: str,
        n_simulations: int,
        year: int | None = None,
        prediction_context=None,
    ) -> dict:
        _ = qualifying_grid
        _ = weather
        _ = year
        self.race_contexts.append(prediction_context)
        base = _StubPredictor().predict_race([], weather, race_name, n_simulations)
        return {
            "finish_order": [
                {
                    **row,
                    "median_position": row["position"],
                    "p5": max(1, row["position"] - 1),
                    "p95": row["position"] + 1,
                }
                for row in base["finish_order"]
            ]
        }

    def record_completed_weekend_actuals(
        self,
        *,
        year: int,
        race_name: str,
        qualifying_actual: list[dict[str, object]],
        race_actual: list[dict[str, object]],
    ) -> dict[str, object]:
        self.replayed_weekends.append(
            {
                "year": year,
                "race_name": race_name,
                "qualifying_actual": qualifying_actual,
                "race_actual": race_actual,
            }
        )
        return {"teams_updated": 3, "races_recorded": len(self.replayed_weekends)}


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


def test_get_races_for_year_uses_season_calendar_when_fastf1_fails(patcher):
    """Historical fallback calendars should not leak modern race names into 2022."""
    patcher.setattr(
        "src.utils.backtesting.fastf1.get_event_schedule",
        lambda _year: (_ for _ in ()).throw(RuntimeError("rate limited")),
    )

    races = get_races_for_year(2022)

    assert races[:4] == [
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Australian Grand Prix",
        "Emilia Romagna Grand Prix",
    ]
    assert "French Grand Prix" in races
    assert "São Paulo Grand Prix" in races
    assert "Chinese Grand Prix" not in races
    assert "Las Vegas Grand Prix" not in races
    assert "Qatar Grand Prix" not in races


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
    assert result["qualifying_predicted_top10"][0]["driver"] == "VER"
    assert result["race_actual_top10"][0]["driver"] == "VER"


def test_run_single_race_backtest_replays_historical_context_and_learning_updates():
    def _fetcher(_year: int, _race_name: str, session_name: str):
        rows = [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ]
        if session_name in {"Q", "R"}:
            return rows
        raise AssertionError("Unexpected session requested")

    predictor = _ContextAwarePredictor()
    result = run_single_race_backtest(
        predictor=predictor,
        year=2025,
        race_name="Bahrain Grand Prix",
        weather="dry",
        qualifying_simulations=10,
        race_simulations=10,
        evaluation_mode="historical",
        learning_mode="adaptive",
        results_fetcher=_fetcher,
    )

    assert result["status"] == "ok"
    assert predictor.qualifying_contexts
    assert predictor.race_contexts
    assert predictor.qualifying_contexts[0].mode == "historical"
    assert predictor.race_contexts[0].mode == "historical"
    assert predictor.qualifying_contexts[0].season_year == 2025
    assert predictor.race_contexts[0].season_year == 2025
    assert result["adaptive_learning"]["sessions_updated"] == 2
    assert len(predictor.calibration_system.records) == 1
    assert predictor.replayed_weekends == [
        {
            "year": 2025,
            "race_name": "Bahrain Grand Prix",
            "qualifying_actual": [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ],
            "race_actual": [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ],
        }
    ]
    assert predictor.calibration_system.records[0]["metadata"]["source"] == "backtest"
    assert result["qualifying_interval_count"] == 3
    assert result["qualifying_interval_hits"] == 3
    assert result["qualifying_interval_empirical_coverage"] == pytest.approx(1.0)
    assert result["race_interval_count"] == 3
    assert result["race_interval_hits"] == 3
    assert result["race_interval_empirical_coverage"] == pytest.approx(1.0)
    qualifying_row = predictor.calibration_system.records[0]["qualifying"]["predicted_grid"][0]
    race_row = predictor.calibration_system.records[0]["race"]["predicted_results"][0]
    assert qualifying_row["median_position"] == 1
    assert qualifying_row["p5"] == 1
    assert qualifying_row["p95"] == 2
    assert race_row["median_position"] == 1
    assert race_row["p5"] == 1
    assert race_row["p95"] == 2


def test_run_single_race_backtest_keeps_static_mode_read_only():
    def _fetcher(_year: int, _race_name: str, session_name: str):
        rows = [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1},
            {"driver": "NOR", "team": "McLaren", "position": 2},
            {"driver": "LEC", "team": "Ferrari", "position": 3},
        ]
        if session_name in {"Q", "R"}:
            return rows
        raise AssertionError("Unexpected session requested")

    predictor = _ContextAwarePredictor()
    result = run_single_race_backtest(
        predictor=predictor,
        year=2025,
        race_name="Bahrain Grand Prix",
        weather="dry",
        qualifying_simulations=10,
        race_simulations=10,
        evaluation_mode="historical",
        learning_mode="static",
        results_fetcher=_fetcher,
    )

    assert result["status"] == "ok"
    assert result["adaptive_learning"] is None
    assert predictor.calibration_system.records == []
    assert predictor.replayed_weekends == [
        {
            "year": 2025,
            "race_name": "Bahrain Grand Prix",
            "qualifying_actual": [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ],
            "race_actual": [
                {"driver": "VER", "team": "Red Bull Racing", "position": 1},
                {"driver": "NOR", "team": "McLaren", "position": 2},
                {"driver": "LEC", "team": "Ferrari", "position": 3},
            ],
        }
    ]


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


def test_warm_fastf1_results_cache_collects_success_and_failure_rows(patcher):
    class _Session:
        def __init__(self, rows: int):
            self.results = [object()] * rows

        def load(self, **kwargs):
            assert kwargs == {
                "laps": False,
                "telemetry": False,
                "weather": False,
                "messages": False,
            }

    def _get_session(year: int, race_name: str, session_name: str):
        assert year == 2025
        if race_name == "Bad Race":
            raise RuntimeError("cache miss")
        return _Session(20 if session_name == "Q" else 22)

    from src.utils import backtesting

    patcher.setattr(backtesting.fastf1, "get_session", _get_session)

    report = warm_fastf1_results_cache(
        year=2025,
        race_names=["Good Race", "Bad Race"],
        session_names=("Q",),
    )

    assert report[0]["status"] == "ok"
    assert report[0]["rows_loaded"] == 20
    assert report[1]["status"] == "error"
    assert "cache miss" in report[1]["reason"]


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
    assert evaluated["qualifying_predicted_top10"][0]["driver"] == "VER"
    assert evaluated["race_actual_top10"][0]["driver"] == "VER"


def test_aggregate_race_metrics_aggregates_interval_calibration_weighted_by_count():
    summary = aggregate_race_metrics(
        [
            {
                "race_name": "Race 1",
                "status": "ok",
                "qualifying_mae": 2.0,
                "qualifying_exact_accuracy": 10.0,
                "race_mae": 3.0,
                "race_exact_accuracy": 5.0,
                "race_within_3": 50.0,
                "top3_accuracy": 33.3,
                "winner_correct": False,
                "qualifying_interval_count": 10,
                "qualifying_interval_hits": 8,
                "qualifying_interval_width_mean": 4.0,
                "qualifying_interval_average_miss_distance": 1.5,
                "race_interval_count": 10,
                "race_interval_hits": 7,
                "race_interval_width_mean": 5.0,
                "race_interval_average_miss_distance": 2.0,
            },
            {
                "race_name": "Race 2",
                "status": "ok",
                "qualifying_mae": 4.0,
                "qualifying_exact_accuracy": 15.0,
                "race_mae": 5.0,
                "race_exact_accuracy": 10.0,
                "race_within_3": 60.0,
                "top3_accuracy": 66.7,
                "winner_correct": True,
                "qualifying_interval_count": 20,
                "qualifying_interval_hits": 19,
                "qualifying_interval_width_mean": 6.0,
                "qualifying_interval_average_miss_distance": 0.5,
                "race_interval_count": 20,
                "race_interval_hits": 16,
                "race_interval_width_mean": 7.0,
                "race_interval_average_miss_distance": 1.0,
            },
        ]
    )

    assert summary["qualifying_interval_races"] == 2
    assert summary["qualifying_interval_count"] == 30
    assert summary["qualifying_interval_empirical_coverage"] == pytest.approx(0.9)
    assert summary["qualifying_interval_calibration_error"] == pytest.approx(0.0)
    assert summary["qualifying_interval_width_mean"] == pytest.approx((10 * 4.0 + 20 * 6.0) / 30)
    assert summary["race_interval_count"] == 30
    assert summary["race_interval_empirical_coverage"] == pytest.approx(23 / 30)
    assert summary["race_interval_width_mean"] == pytest.approx((10 * 5.0 + 20 * 7.0) / 30)


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


def test_build_overlap_comparison_uses_only_shared_successful_races():
    model_race_results = [
        {
            "race_name": "Australian Grand Prix",
            "status": "ok",
            "qualifying_mae": 3.0,
            "race_mae": 4.0,
            "top3_accuracy": 60.0,
            "winner_correct": True,
        },
        {
            "race_name": "Chinese Grand Prix",
            "status": "ok",
            "qualifying_mae": 2.0,
            "race_mae": 3.0,
            "top3_accuracy": 70.0,
            "winner_correct": False,
        },
        {
            "race_name": "Japanese Grand Prix",
            "status": "skipped",
            "reason": "missing_actual_results",
        },
    ]
    naive_race_results = [
        {
            "race_name": "Australian Grand Prix",
            "status": "ok",
            "qualifying_mae": 4.0,
            "race_mae": 5.0,
            "top3_accuracy": 50.0,
            "winner_correct": False,
        },
        {
            "race_name": "Japanese Grand Prix",
            "status": "ok",
            "qualifying_mae": 2.5,
            "race_mae": 3.5,
            "top3_accuracy": 55.0,
            "winner_correct": False,
        },
    ]

    comparison = build_overlap_comparison(
        model_race_results=model_race_results,
        naive_race_results=naive_race_results,
    )

    assert comparison["races_evaluated"] == 1
    assert comparison["shared_races"] == ["Australian Grand Prix"]
    assert comparison["model"]["race_mae_mean"] == pytest.approx(4.0)
    assert comparison["naive"]["race_mae_mean"] == pytest.approx(5.0)
    assert comparison["qualifying_mae_improvement"] == pytest.approx(1.0)
    assert comparison["race_mae_improvement"] == pytest.approx(1.0)


def test_build_segment_breakdown_groups_successful_rows_by_metadata():
    race_results = [
        {
            "race_name": "Australian Grand Prix",
            "status": "ok",
            "weekend_format": "normal",
            "track_type": "permanent",
            "weather": "dry",
            "qualifying_mae": 1.0,
            "qualifying_exact_accuracy": 20.0,
            "race_mae": 2.0,
            "race_exact_accuracy": 15.0,
            "race_within_3": 80.0,
            "top3_accuracy": 70.0,
            "winner_correct": True,
        },
        {
            "race_name": "Chinese Grand Prix",
            "status": "ok",
            "weekend_format": "sprint",
            "track_type": "permanent",
            "weather": "dry",
            "qualifying_mae": 1.5,
            "qualifying_exact_accuracy": 18.0,
            "race_mae": 3.0,
            "race_exact_accuracy": 10.0,
            "race_within_3": 75.0,
            "top3_accuracy": 60.0,
            "winner_correct": False,
        },
        {
            "race_name": "Monaco Grand Prix",
            "status": "ok",
            "weekend_format": "normal",
            "track_type": "street",
            "weather": "mixed",
            "qualifying_mae": 2.0,
            "qualifying_exact_accuracy": 12.0,
            "race_mae": 5.0,
            "race_exact_accuracy": 5.0,
            "race_within_3": 55.0,
            "top3_accuracy": 30.0,
            "winner_correct": False,
        },
        {
            "race_name": "Missing Grand Prix",
            "status": "skipped",
            "weekend_format": "normal",
            "track_type": "unknown",
            "weather": "dry",
            "reason": "missing_actual_results",
        },
    ]

    breakdown = build_segment_breakdown(race_results)

    assert breakdown["track_type"]["permanent"]["events"] == 2
    assert breakdown["track_type"]["permanent"]["race_mae_mean"] == pytest.approx(2.5)
    assert breakdown["track_type"]["street"]["events"] == 1
    assert breakdown["weather"]["mixed"]["race_mae_mean"] == pytest.approx(5.0)
    assert breakdown["weekend_format"]["normal"]["winner_accuracy_percent"] == pytest.approx(50.0)


def test_build_error_analysis_highlights_worst_weekends_and_winner_misses():
    race_results = [
        {
            "race_name": "Australian Grand Prix",
            "status": "ok",
            "weekend_format": "normal",
            "track_type": "permanent",
            "weather": "dry",
            "qualifying_mae": 1.0,
            "race_mae": 2.0,
            "top3_accuracy": 70.0,
            "winner_correct": True,
        },
        {
            "race_name": "Monaco Grand Prix",
            "status": "ok",
            "weekend_format": "normal",
            "track_type": "street",
            "weather": "mixed",
            "qualifying_mae": 2.5,
            "race_mae": 6.0,
            "top3_accuracy": 20.0,
            "winner_correct": False,
        },
        {
            "race_name": "Chinese Grand Prix",
            "status": "ok",
            "weekend_format": "sprint",
            "track_type": "permanent",
            "weather": "dry",
            "qualifying_mae": 3.0,
            "race_mae": 4.5,
            "top3_accuracy": 35.0,
            "winner_correct": False,
        },
    ]

    analysis = build_error_analysis(race_results)

    assert analysis["races_evaluated"] == 3
    assert analysis["worst_race_events"][0]["race_name"] == "Monaco Grand Prix"
    assert analysis["worst_qualifying_events"][0]["race_name"] == "Chinese Grand Prix"
    assert analysis["winner_miss_events"][0]["race_name"] == "Monaco Grand Prix"
    assert analysis["worst_race_track_types"][0] == {"label": "permanent", "count": 2}


def test_build_checked_backtest_summary_keeps_race_level_detail():
    baseline_report = {
        "name": "baseline",
        "summary": {"races_evaluated": 1, "race_mae_mean": 3.4},
        "generalization": {"test": {"race_mae_mean": 3.8}},
        "race_results": [
            {
                "race_name": "Australian Grand Prix",
                "status": "ok",
                "qualifying_predicted_top10": [
                    {"driver": "VER", "position": 1, "team": "Red Bull"}
                ],
                "race_actual_top10": [{"driver": "NOR", "position": 1, "team": "McLaren"}],
            }
        ],
    }
    naive_report = {
        "name": "previous_race_classification",
        "summary": {"races_evaluated": 1, "race_mae_mean": 4.2},
        "race_results": [
            {
                "race_name": "Australian Grand Prix",
                "status": "ok",
                "race_predicted_top10": [{"driver": "VER", "position": 1, "team": "Red Bull"}],
                "race_actual_top10": [{"driver": "NOR", "position": 1, "team": "McLaren"}],
            }
        ],
    }
    overlap_comparison = {
        "model": {"race_mae_mean": 3.4},
        "naive": {"race_mae_mean": 4.2},
        "race_mae_improvement": 0.8,
        "races_evaluated": 1,
        "shared_races": ["Australian Grand Prix"],
    }

    summary = build_checked_backtest_summary(
        year=2025,
        baseline_report=baseline_report,
        naive_report=naive_report,
        overlap_comparison=overlap_comparison,
        reports_dir="reports/backtest_2025",
    )

    assert summary["season"] == 2025
    assert summary["model"]["name"] == "baseline"
    assert summary["model"]["race_mae_mean"] == pytest.approx(3.4)
    assert (
        summary["baseline_report"]["race_results"][0]["qualifying_predicted_top10"][0]["driver"]
        == "VER"
    )
    assert (
        summary["naive_previous_race_baseline"]["race_results"][0]["race_actual_top10"][0]["driver"]
        == "NOR"
    )
    assert summary["overlap_comparison"]["shared_races"] == ["Australian Grand Prix"]


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


# ---------------------------------------------------------------------------
# Temporal split tests
# ---------------------------------------------------------------------------

from src.utils.backtesting import split_train_test_results  # noqa: E402


def _make_race_results(n: int, mae_offset: float = 0.0) -> list[dict]:
    """Return n sequential race results ordered as they would arrive from the harness."""
    return [
        {
            "race_name": f"Race_{i + 1:02d}",
            "status": "ok",
            "race_mae": 3.0 + mae_offset,
            "qualifying_mae": 2.5,
            "qualifying_exact_accuracy": 15.0,
            "race_exact_accuracy": 10.0,
            "race_within_3": 60.0,
            "top3_accuracy": 40.0,
            "winner_correct": i % 3 == 0,
        }
        for i in range(n)
    ]


def test_temporal_split_preserves_calendar_order():
    """Training set must contain the earliest races; test set the latest ones.

    This is the correctness requirement: a predictor that learns from completed
    races should never have future-race outcomes in its training fold.
    """
    results = _make_race_results(10)
    train, test = split_train_test_results(results, train_fraction=0.7, strategy="temporal")

    assert len(train) == 7
    assert len(test) == 3

    train_names = [r["race_name"] for r in train]
    test_names = [r["race_name"] for r in test]

    # First 7 races are training, last 3 are test — no shuffling
    assert train_names == [f"Race_{i:02d}" for i in range(1, 8)]
    assert test_names == [f"Race_{i:02d}" for i in range(8, 11)]


def test_temporal_split_is_deterministic():
    """Same input always produces same split regardless of call count."""
    results = _make_race_results(8)
    train_a, test_a = split_train_test_results(results, strategy="temporal")
    train_b, test_b = split_train_test_results(results, strategy="temporal")

    assert [r["race_name"] for r in train_a] == [r["race_name"] for r in train_b]
    assert [r["race_name"] for r in test_a] == [r["race_name"] for r in test_b]


def test_random_split_is_not_necessarily_ordered():
    """Random strategy should sometimes differ from temporal order (probabilistic).

    This test seeds the RNG to a value known to produce a shuffle; it verifies
    that the random strategy is distinct from temporal — not that it's wrong,
    but that the two strategies are actually different.
    """
    results = _make_race_results(10)
    train_temporal, _ = split_train_test_results(results, strategy="temporal", seed=99)
    train_random, _ = split_train_test_results(results, strategy="random", seed=99)

    temporal_names = [r["race_name"] for r in train_temporal]
    random_names = [r["race_name"] for r in train_random]

    # Random split with seed=99 should not be identical to the ordered temporal split
    assert temporal_names != random_names, (
        "random split coincidentally matches temporal order with this seed — "
        "change the seed in the test"
    )


def test_random_split_is_reproducible_with_same_seed():
    """Random strategy with the same seed must produce the same split every time."""
    results = _make_race_results(10)
    train_a, _ = split_train_test_results(results, strategy="random", seed=42)
    train_b, _ = split_train_test_results(results, strategy="random", seed=42)

    assert [r["race_name"] for r in train_a] == [r["race_name"] for r in train_b]


def test_split_excludes_skipped_races():
    """Rows with status != 'ok' must not appear in either split."""
    results = _make_race_results(6)
    results[2]["status"] = "skipped"
    results[2]["reason"] = "missing_actual_results"

    train, test = split_train_test_results(results, train_fraction=0.7, strategy="temporal")

    all_names = {r["race_name"] for r in train + test}
    assert "Race_03" not in all_names
    assert len(train) + len(test) == 5  # 6 total minus 1 skipped


def test_summarize_generalization_records_split_strategy():
    """The returned dict must include the strategy used so callers can audit it."""
    results = _make_race_results(8)
    summary = summarize_generalization(results, strategy="temporal")

    assert summary["split_strategy"] == "temporal"


def test_temporal_split_default_in_summarize_generalization():
    """summarize_generalization must default to temporal, not random."""
    results = _make_race_results(10)

    # Calling without explicit strategy should behave like strategy="temporal"
    default_summary = summarize_generalization(results, train_fraction=0.7)
    temporal_summary = summarize_generalization(results, train_fraction=0.7, strategy="temporal")

    assert default_summary["split_strategy"] == "temporal"
    assert (
        default_summary["train"]["races_evaluated"] == temporal_summary["train"]["races_evaluated"]
    )


def test_invalid_split_strategy_raises():
    """Unknown strategy names must raise ValueError immediately."""
    results = _make_race_results(5)
    with pytest.raises(ValueError, match="Unknown split strategy"):
        split_train_test_results(results, strategy="xgboost")  # type: ignore[arg-type]
