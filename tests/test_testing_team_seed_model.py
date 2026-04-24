"""Tests for the testing-derived preseason team seed model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models import testing_team_seed


def _feature_row(
    *,
    season_year: int,
    team_name: str,
    source_kind: str = "preseason",
    sample_weight: float = 1.0,
    target_team_strength: float = 0.5,
    base_value: float = 0.5,
) -> dict[str, object]:
    """Build one deterministic training row covering the full feature surface."""
    row: dict[str, object] = {
        "season_year": season_year,
        "team_name": team_name,
        "source_kind": source_kind,
        "source_events": ["Pre-Season Testing"],
        "sample_weight": sample_weight,
        "target_team_strength": target_team_strength,
    }
    for feature_name in testing_team_seed.FEATURE_COLUMNS:
        row[feature_name] = base_value
    return row


def _team_payload(*, sessions_used: int, circuit_count: int) -> dict[str, object]:
    """Build one minimal team payload with the testing fields the model expects."""
    circuits = [f"Testing {index}" for index in range(1, circuit_count + 1)]
    profile = {
        "run_profile": "balanced",
        "overall_pace": 0.72,
        "slow_corner_performance": 0.69,
        "medium_corner_performance": 0.71,
        "fast_corner_performance": 0.74,
        "braking_performance": 0.67,
        "top_speed": 0.73,
        "consistency": 0.66,
        "tire_deg_performance": 0.62,
        "tire_deg_slope": 0.05,
        "sessions_used": sessions_used,
        "sessions_blended": sessions_used,
        "session_aggregation": "laps_weighted",
        "last_updated": "2026-04-21T00:00:00+00:00",
        "circuits_observed": circuits,
    }
    return {
        "directionality": {
            "max_speed": 0.02,
            "slow_corner_speed": 0.01,
            "medium_corner_speed": 0.00,
            "high_corner_speed": 0.03,
        },
        "testing_characteristics": profile,
        "testing_characteristics_profiles": {
            "balanced": profile,
            "short_run": {
                **profile,
                "run_profile": "short_run",
                "overall_pace": 0.75,
                "top_speed": 0.77,
            },
            "long_run": {
                **profile,
                "run_profile": "long_run",
                "overall_pace": 0.68,
                "consistency": 0.78,
            },
        },
    }


def _full_ranking_payload() -> dict[str, dict[str, float]]:
    """Return a full prior-year ranking map aligned with the champion baseline."""
    return {
        "McLaren": {"overall_performance": 0.99},
        "Mercedes": {"overall_performance": 0.95},
        "Red Bull Racing": {"overall_performance": 0.93},
        "Ferrari": {"overall_performance": 0.90},
        "Williams": {"overall_performance": 0.82},
        "RB": {"overall_performance": 0.76},
        "Aston Martin": {"overall_performance": 0.71},
        "Haas F1 Team": {"overall_performance": 0.66},
        "Alpine": {"overall_performance": 0.61},
        "Audi": {"overall_performance": 0.55},
        "Cadillac F1": {"overall_performance": 0.40},
    }


def test_discover_preseason_event_names_handles_track_session_and_testing(tmp_path):
    """Cache discovery should accept both preseason naming conventions."""
    cache_root = tmp_path / "cache"
    (cache_root / "2022" / "2022-02-25_Pre-Season_Track_Session" / "Practice_1").mkdir(parents=True)
    (cache_root / "2025" / "2025-02-28_Pre-Season_Testing" / "Practice_1").mkdir(parents=True)

    season_2022 = testing_team_seed.discover_preseason_event_names(
        2022,
        cache_dirs=(cache_root,),
    )
    season_2025 = testing_team_seed.discover_preseason_event_names(
        2025,
        cache_dirs=(cache_root,),
    )

    assert season_2022 == ["Pre-Season Track Session"]
    assert season_2025 == ["Pre-Season Testing"]


def test_discover_auxiliary_race_events_limits_first_two_weekends(monkeypatch):
    """Auxiliary FP rows should stop at the first two race weekends."""
    monkeypatch.setattr(
        testing_team_seed,
        "get_schedule_rows",
        lambda year: (
            ("Australian Grand Prix", "conventional"),
            ("Chinese Grand Prix", "sprint_qualifying"),
            ("Japanese Grand Prix", "conventional"),
        ),
    )

    assert testing_team_seed.discover_auxiliary_race_events(2025, max_events=2) == [
        "Australian Grand Prix",
        "Chinese Grand Prix",
    ]


def test_build_target_label_map_canonicalizes_shared_team_keys(monkeypatch):
    """Early-season labels should collapse old aliases onto current team keys."""
    monkeypatch.setattr(
        testing_team_seed,
        "calculate_team_performance_from_races",
        lambda year, max_races=None: {
            "Kick Sauber": {"overall_performance": 0.97},
            "AlphaTauri": {"overall_performance": 0.95},
            "McLaren": {"overall_performance": 0.99},
        },
    )

    labels = testing_team_seed.build_target_label_map(2025, race_limit=3)

    assert set(labels) == {"Audi", "RB", "McLaren"}
    assert labels["McLaren"] > labels["Audi"]


def test_fit_and_loso_validation_are_deterministic():
    """Model fit and leave-one-season-out validation should be reproducible."""
    dataset = pd.DataFrame(
        [
            _feature_row(
                season_year=2022, team_name="McLaren", target_team_strength=0.9, base_value=0.9
            ),
            _feature_row(
                season_year=2022, team_name="Audi", target_team_strength=0.1, base_value=0.1
            ),
            _feature_row(
                season_year=2023, team_name="McLaren", target_team_strength=0.88, base_value=0.88
            ),
            _feature_row(
                season_year=2023, team_name="Audi", target_team_strength=0.12, base_value=0.12
            ),
            _feature_row(
                season_year=2024, team_name="McLaren", target_team_strength=0.85, base_value=0.85
            ),
            _feature_row(
                season_year=2024, team_name="Audi", target_team_strength=0.15, base_value=0.15
            ),
            _feature_row(
                season_year=2025, team_name="McLaren", target_team_strength=0.83, base_value=0.83
            ),
            _feature_row(
                season_year=2025, team_name="Audi", target_team_strength=0.17, base_value=0.17
            ),
        ]
    )

    first_model = testing_team_seed.fit_testing_team_seed_model(dataset)
    second_model = testing_team_seed.fit_testing_team_seed_model(dataset)
    validation_a = testing_team_seed.run_leave_one_season_out_validation(dataset)
    validation_b = testing_team_seed.run_leave_one_season_out_validation(dataset)

    first_predictions = first_model.predict_rows(dataset)
    second_predictions = second_model.predict_rows(dataset)

    assert first_model.regressor.alpha_ == second_model.regressor.alpha_
    assert validation_a.mae == validation_b.mae
    assert validation_a.rmse == validation_b.rmse
    assert first_predictions.tolist() == second_predictions.tolist()


def test_compute_target_year_transfer_weight_prioritizes_reset_year_similarity():
    """A 2026 reset-year target should not treat mature 2025 as representative."""
    weights = {
        year: testing_team_seed.compute_target_year_transfer_weight(
            source_year=year,
            target_year=2026,
        )
        for year in (2014, 2022, 2023, 2024, 2025)
    }

    assert weights[2022] > weights[2023] > weights[2024] > weights[2025]
    assert weights[2022] > weights[2014] > weights[2025]
    assert weights[2025] <= 0.45
    assert all(0.35 <= value <= 1.0 for value in weights.values())


def test_apply_target_year_transfer_weights_adds_effective_weight_columns():
    """Target-year transfer weighting should preserve base weights and add effective weights."""
    dataset = pd.DataFrame(
        [
            _feature_row(season_year=2022, team_name="McLaren", sample_weight=1.0),
            _feature_row(season_year=2025, team_name="Audi", sample_weight=0.35),
        ]
    )

    weighted = testing_team_seed.apply_target_year_transfer_weights(
        dataset,
        target_year=2026,
    )

    assert weighted["base_sample_weight"].tolist() == [1.0, 0.35]
    assert weighted["transfer_weight"].iloc[0] > weighted["transfer_weight"].iloc[1]
    assert weighted["effective_sample_weight"].iloc[0] > weighted["effective_sample_weight"].iloc[1]


def test_fit_testing_team_seed_model_uses_effective_sample_weight_when_present(monkeypatch):
    """Effective sample weights should override the base sample-weight column when available."""
    captured: dict[str, object] = {}

    class _DummyRidgeCV:
        def __init__(self, alphas):
            self.alpha_ = float(tuple(alphas)[0])

        def fit(self, features, target, sample_weight=None):
            captured["sample_weight"] = None if sample_weight is None else sample_weight.tolist()
            captured["features_shape"] = tuple(features.shape)
            captured["target"] = target.tolist()
            return self

        def predict(self, rows):
            return np.zeros(len(rows), dtype=float)

    monkeypatch.setattr(testing_team_seed, "RidgeCV", _DummyRidgeCV)

    dataset = pd.DataFrame(
        [
            {
                **_feature_row(
                    season_year=2022,
                    team_name="McLaren",
                    target_team_strength=0.9,
                    sample_weight=1.0,
                    base_value=0.9,
                ),
                "effective_sample_weight": 0.88,
            },
            {
                **_feature_row(
                    season_year=2025,
                    team_name="Audi",
                    target_team_strength=0.1,
                    sample_weight=1.0,
                    base_value=0.1,
                ),
                "effective_sample_weight": 0.63,
            },
        ]
    )

    model = testing_team_seed.fit_testing_team_seed_model(dataset)

    assert model.regressor.alpha_ == testing_team_seed._MODEL_ALPHA_GRID[0]
    assert captured["sample_weight"] == [0.88, 0.63]


def test_apply_bounded_testing_delta_to_champion_prior_penalizes_high_signal_disagreement():
    """High signal disagreement should collapse the testing delta back toward champion."""
    low_disagreement = testing_team_seed._apply_bounded_testing_delta_to_champion_prior(
        champion_prior_overall_performance=0.85,
        testing_glimpse_overall_performance=0.58,
        pace_glimpse_confidence=0.78,
        signal_disagreement=0.10,
        coverage_penalty=0.0,
    )
    high_disagreement = testing_team_seed._apply_bounded_testing_delta_to_champion_prior(
        champion_prior_overall_performance=0.85,
        testing_glimpse_overall_performance=0.58,
        pace_glimpse_confidence=0.78,
        signal_disagreement=1.0,
        coverage_penalty=0.0,
    )

    assert abs(high_disagreement[1]) < abs(low_disagreement[1])
    assert high_disagreement[3] < low_disagreement[3]
    assert abs(low_disagreement[2]) <= testing_team_seed._MAX_TESTING_DELTA_FROM_CHAMPION


def test_build_training_dataset_uses_preseason_and_first_two_fp_events_only(monkeypatch):
    """Training rows should include preseason plus only the first two FP windows."""

    def _fake_snapshot(year, *, event_names, session_candidates, source_kind, **_kwargs):
        return testing_team_seed.TestingSeedSnapshot(
            year=year,
            event_names=tuple(event_names),
            source_kind=source_kind,
            loaded_sessions=tuple(
                f"{event_name}::{session}"
                for event_name in event_names
                for session in session_candidates[:1]
            ),
            updated_teams=("McLaren",),
            teams={"McLaren": _team_payload(sessions_used=3, circuit_count=1)},
            selected_lap_weights={"McLaren": 20.0},
            team_session_counts={"McLaren": 3},
        )

    monkeypatch.setattr(
        testing_team_seed,
        "build_target_label_map",
        lambda year, race_limit=3: {"McLaren": 0.9},
    )
    monkeypatch.setattr(
        testing_team_seed,
        "discover_preseason_event_names",
        lambda year, cache_dirs=None: ["Pre-Season Testing"],
    )
    monkeypatch.setattr(
        testing_team_seed,
        "discover_auxiliary_race_events",
        lambda year, max_events=2: [
            "Australian Grand Prix",
            "Chinese Grand Prix",
            "Japanese Grand Prix",
        ][:max_events],
    )
    monkeypatch.setattr(testing_team_seed, "collect_team_testing_snapshot", _fake_snapshot)

    dataset = testing_team_seed.build_training_dataset([2025])

    assert dataset["source_kind"].tolist() == ["preseason", "auxiliary_fp", "auxiliary_fp"]
    assert dataset["source_events"].tolist() == [
        ["Pre-Season Testing"],
        ["Australian Grand Prix"],
        ["Chinese Grand Prix"],
    ]


def test_build_training_dataset_reuses_cached_season_rows(tmp_path, monkeypatch):
    """Repeated dataset builds should reuse cached rows for unchanged seasons."""
    calls: list[tuple[int, str]] = []

    def _fake_snapshot(year, *, event_names, session_candidates, source_kind, **_kwargs):
        calls.append((year, source_kind))
        return testing_team_seed.TestingSeedSnapshot(
            year=year,
            event_names=tuple(event_names),
            source_kind=source_kind,
            loaded_sessions=tuple(
                f"{event_name}::{session}"
                for event_name in event_names
                for session in session_candidates[:1]
            ),
            updated_teams=("McLaren",),
            teams={"McLaren": _team_payload(sessions_used=3, circuit_count=1)},
            selected_lap_weights={"McLaren": 20.0},
            team_session_counts={"McLaren": 3},
        )

    monkeypatch.setattr(
        testing_team_seed,
        "build_target_label_map",
        lambda year, race_limit=3: {"McLaren": 0.9},
    )
    monkeypatch.setattr(
        testing_team_seed,
        "discover_preseason_event_names",
        lambda year, cache_dirs=None: ["Pre-Season Testing"],
    )
    monkeypatch.setattr(
        testing_team_seed,
        "discover_auxiliary_race_events",
        lambda year, max_events=2: ["Australian Grand Prix"][:max_events],
    )
    monkeypatch.setattr(testing_team_seed, "collect_team_testing_snapshot", _fake_snapshot)

    first = testing_team_seed.build_training_dataset(
        [2025],
        cache_dir=tmp_path / "fastf1-cache",
        feature_cache_dir=tmp_path / "feature-cache",
    )
    second = testing_team_seed.build_training_dataset(
        [2025],
        cache_dir=tmp_path / "fastf1-cache",
        feature_cache_dir=tmp_path / "feature-cache",
    )

    assert calls == [(2025, "preseason"), (2025, "auxiliary_fp")]
    assert first.to_dict(orient="records") == second.to_dict(orient="records")


def test_build_testing_model_team_payload_shrinks_pace_glimpse_and_penalizes_sparse_coverage(
    monkeypatch,
):
    """Generated team seeds should stay close to champion and punish weaker testing evidence."""

    class _DummyRegressor:
        alpha_ = 1.0

    class _DummyModel:
        regressor = _DummyRegressor()

        def predict_rows(self, rows: pd.DataFrame):
            return pd.Series([5.0, -2.0], dtype=float).to_numpy()

    validation = testing_team_seed.LeaveOneSeasonOutSummary(
        rows=[],
        seasons=[{"season_year": 2024, "mae": 0.08, "rmse": 0.10}],
        mae=0.08,
        rmse=0.10,
    )
    snapshot = testing_team_seed.TestingSeedSnapshot(
        year=2026,
        event_names=("Pre-Season Testing",),
        source_kind="preseason",
        loaded_sessions=("Pre-Season Testing::Testing 1 Day 1",),
        updated_teams=("McLaren", "Audi"),
        teams={
            "McLaren": _team_payload(sessions_used=6, circuit_count=1),
            "Audi": _team_payload(sessions_used=1, circuit_count=0),
        },
        selected_lap_weights={"McLaren": 30.0, "Audi": 4.0},
        team_session_counts={"McLaren": 6, "Audi": 1},
    )

    monkeypatch.setattr(
        testing_team_seed,
        "build_training_dataset",
        lambda training_years, cache_dir=testing_team_seed._DEFAULT_CACHE_DIR: pd.DataFrame(
            [
                _feature_row(season_year=2022, team_name="McLaren", target_team_strength=0.90),
                _feature_row(season_year=2023, team_name="McLaren", target_team_strength=0.82),
                _feature_row(season_year=2024, team_name="McLaren", target_team_strength=0.74),
                _feature_row(season_year=2025, team_name="McLaren", target_team_strength=0.66),
            ]
        ),
    )
    monkeypatch.setattr(
        testing_team_seed,
        "run_leave_one_season_out_validation",
        lambda dataset, **kwargs: validation,
    )
    monkeypatch.setattr(
        testing_team_seed, "fit_testing_team_seed_model", lambda dataset: _DummyModel()
    )
    monkeypatch.setattr(
        testing_team_seed,
        "discover_preseason_event_names",
        lambda year, cache_dirs=None: ["Pre-Season Testing"],
    )
    monkeypatch.setattr(
        testing_team_seed,
        "collect_team_testing_snapshot",
        lambda year, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        testing_team_seed,
        "calculate_team_performance_from_races",
        lambda year, max_races=None: _full_ranking_payload(),
    )

    payload = testing_team_seed.build_testing_model_team_payload(
        target_year=2026,
        training_years=[2022, 2023, 2024, 2025],
    )

    assert 0.80 < payload["teams"]["McLaren"]["overall_performance"] < 0.85
    assert 0.38 <= payload["teams"]["Audi"]["overall_performance"] < 0.41
    assert payload["teams"]["Audi"]["uncertainty"] > payload["teams"]["McLaren"]["uncertainty"]
    training_year_relevance = payload["directionality_meta"]["training_year_relevance"]
    weights_by_year = {
        row["season_year"]: row["transfer_weight_mean"] for row in training_year_relevance
    }
    assert weights_by_year[2022] > weights_by_year[2025]
    diagnostics = payload["directionality_meta"]["team_modeling_diagnostics"]
    assert (
        diagnostics["McLaren"]["pace_glimpse_confidence"]
        > diagnostics["Audi"]["pace_glimpse_confidence"]
    )
    assert diagnostics["McLaren"]["delta_multiplier"] > diagnostics["Audi"]["delta_multiplier"]
    assert abs(
        diagnostics["McLaren"]["conservative_overall_performance"]
        - diagnostics["McLaren"]["champion_prior_overall_performance"]
    ) < abs(
        diagnostics["McLaren"]["testing_glimpse_overall_performance"]
        - diagnostics["McLaren"]["champion_prior_overall_performance"]
    )
    assert payload["directionality_meta"]["blend_strategy"] == "bounded_champion_delta"


def test_build_testing_model_team_payload_falls_back_when_loso_is_impossible(monkeypatch):
    """Single-season smoke runs should use conservative fallback validation instead of failing."""

    class _DummyRegressor:
        alpha_ = 1.0

    class _DummyModel:
        regressor = _DummyRegressor()

        def predict_rows(self, rows: pd.DataFrame):
            return pd.Series([0.4], dtype=float).to_numpy()

    snapshot = testing_team_seed.TestingSeedSnapshot(
        year=2026,
        event_names=("Pre-Season Testing",),
        source_kind="preseason",
        loaded_sessions=("Pre-Season Testing::Testing 1 Day 1",),
        updated_teams=("McLaren",),
        teams={"McLaren": _team_payload(sessions_used=4, circuit_count=1)},
        selected_lap_weights={"McLaren": 20.0},
        team_session_counts={"McLaren": 4},
    )

    monkeypatch.setattr(
        testing_team_seed,
        "build_training_dataset",
        lambda training_years, cache_dir=testing_team_seed._DEFAULT_CACHE_DIR: pd.DataFrame(
            [_feature_row(season_year=2024, team_name="McLaren", target_team_strength=0.9)]
        ),
    )
    monkeypatch.setattr(
        testing_team_seed, "fit_testing_team_seed_model", lambda dataset: _DummyModel()
    )
    monkeypatch.setattr(
        testing_team_seed,
        "discover_preseason_event_names",
        lambda year, cache_dirs=None: ["Pre-Season Testing"],
    )
    monkeypatch.setattr(
        testing_team_seed,
        "collect_team_testing_snapshot",
        lambda year, **kwargs: snapshot,
    )

    payload = testing_team_seed.build_testing_model_team_payload(
        target_year=2026,
        training_years=[2024],
    )

    meta = payload["directionality_meta"]
    assert meta["validation"]["season_summaries"][0]["method"] == "fallback_single_season"
    assert payload["teams"]["McLaren"]["uncertainty"] >= 0.26


def test_coverage_penalty_uses_lap_weight_separately_from_session_count():
    """Thin lap coverage should raise uncertainty even when session counts match."""
    team_payload = _team_payload(sessions_used=4, circuit_count=1)

    dense_penalty = testing_team_seed._coverage_penalty(
        team_payload,
        selected_lap_weight=20.0,
    )
    sparse_penalty = testing_team_seed._coverage_penalty(
        team_payload,
        selected_lap_weight=4.0,
    )

    assert dense_penalty == 0.0
    assert sparse_penalty == 0.02


def test_build_prior_year_ranking_seed_payload_preserves_testing_fields(monkeypatch):
    """Comparison payloads should keep target-year testing features intact."""
    snapshot = testing_team_seed.TestingSeedSnapshot(
        year=2025,
        event_names=("Pre-Season Testing",),
        source_kind="preseason",
        loaded_sessions=("Pre-Season Testing::Testing 1 Day 1",),
        updated_teams=("McLaren", "Audi"),
        teams={
            "McLaren": _team_payload(sessions_used=5, circuit_count=1),
            "Audi": _team_payload(sessions_used=5, circuit_count=1),
        },
        selected_lap_weights={"McLaren": 24.0, "Audi": 18.0},
        team_session_counts={"McLaren": 5, "Audi": 5},
    )

    monkeypatch.setattr(
        testing_team_seed,
        "discover_preseason_event_names",
        lambda year, cache_dirs=None: ["Pre-Season Testing"],
    )
    monkeypatch.setattr(
        testing_team_seed,
        "collect_team_testing_snapshot",
        lambda year, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        testing_team_seed,
        "calculate_team_performance_from_races",
        lambda year, max_races=None: {
            "McLaren": {"overall_performance": 0.99},
            "Kick Sauber": {"overall_performance": 0.94},
        },
    )

    payload = testing_team_seed.build_prior_year_ranking_seed_payload(
        target_year=2025,
        source_year=2024,
    )

    assert (
        payload["teams"]["McLaren"]["overall_performance"]
        > payload["teams"]["Audi"]["overall_performance"]
    )
    assert "testing_characteristics" in payload["teams"]["Audi"]
    assert payload["directionality_meta"]["seed_mode"] == "prior_year_ranking"


def test_write_validation_report_outputs_compact_json(tmp_path):
    """Validation report writer should emit a small machine-readable summary."""
    payload = {
        "year": 2026,
        "generated_at": "2026-04-21T00:00:00+00:00",
        "directionality_meta": {"seed_mode": "testing_model"},
        "teams": {
            "McLaren": {"overall_performance": 0.8, "uncertainty": 0.14},
            "Audi": {"overall_performance": 0.4, "uncertainty": 0.22},
        },
    }

    output_path = testing_team_seed.write_validation_report(
        payload=payload,
        output_path=tmp_path / "report.json",
    )
    written = json.loads(output_path.read_text())

    assert written["seed_mode"] == "testing_model"
    assert written["teams_ranked"][0]["team_name"] == "McLaren"
