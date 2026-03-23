"""Tests for dashboard prediction orchestration."""

from unittest.mock import MagicMock

import pytest

from src.dashboard import prediction_flow


def test_run_prediction_executes_on_repeated_calls(patcher):
    """
    Prediction orchestration must execute every call.

    This guards against stale cached results in the Generate Prediction flow.
    """
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction("Bahrain Grand Prix", "dry", artifact_versions, is_sprint=False)
    prediction_flow.run_prediction("Bahrain Grand Prix", "dry", artifact_versions, is_sprint=False)

    assert mock_predictor.predict_qualifying.call_count == 2
    assert mock_predictor.predict_race.call_count == 2


def test_run_prediction_sprint_path_refreshes_both_competitive_grids(patcher):
    """Sprint flow should fetch both SQ and Q grids when available."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.side_effect = [
        {"grid": [{"driver": "NOR", "team": "McLaren", "position": 1}]},
        {"grid": [{"driver": "NOR", "team": "McLaren", "position": 2}]},
    ]
    mock_predictor.predict_sprint_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )

    grid_sessions: list[str] = []

    def _fetch_grid(year: int, race_name: str, session_name: str, predicted_grid: list):
        grid_sessions.append(session_name)
        return predicted_grid, "ACTUAL"

    patcher.setattr(prediction_flow, "fetch_grid_if_available", _fetch_grid)

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    result = prediction_flow.run_prediction(
        "Chinese Grand Prix",
        "dry",
        artifact_versions,
        is_sprint=True,
    )

    assert grid_sessions == ["SQ", "Q"]
    assert result["sprint_quali"]["grid_source"] == "ACTUAL"
    assert result["main_quali"]["grid_source"] == "ACTUAL"
    assert mock_predictor.predict_sprint_race.call_count == 1
    assert mock_predictor.predict_race.call_count == 1


def test_run_prediction_uses_explicit_year_for_fastf1_refresh(patcher):
    """The orchestration layer should pass the requested season to all refresh calls."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }

    predictor_years: list[int] = []

    def _get_predictor(_versions, year=2026):
        predictor_years.append(year)
        return mock_predictor

    patcher.setattr(prediction_flow, "get_predictor", _get_predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )

    years_seen: list[int] = []

    def _fetch_grid(year: int, race_name: str, session_name: str, predicted_grid: list):
        years_seen.append(year)
        return predicted_grid, "PREDICTED"

    patcher.setattr(prediction_flow, "fetch_grid_if_available", _fetch_grid)

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction(
        race_name="Bahrain Grand Prix",
        weather="dry",
        _artifact_versions=artifact_versions,
        is_sprint=False,
        year=2027,
    )

    assert years_seen == [2027]
    assert predictor_years == [2027]
    assert mock_predictor.predict_qualifying.call_args.kwargs["year"] == 2027


def test_run_prediction_uses_stored_checkpoint_profiles_for_qualifying(patcher):
    """Dashboard qualifying should use the stored checkpoint profile path explicitly."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "_resolve_current_checkpoint_session",
        lambda year, race_name, is_sprint: "FP2",
    )
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction(
        "Australian Grand Prix", "dry", artifact_versions, is_sprint=False
    )

    assert mock_predictor.predict_qualifying.call_args.kwargs["practice_signal_mode"] == (
        "stored_profiles"
    )
    assert mock_predictor.predict_qualifying.call_args.kwargs["checkpoint_session_name"] == "FP2"


def test_run_prediction_builds_checkpoint_overlay_predictor(patcher):
    """Prediction orchestration should use the checkpoint-aware predictor when available."""
    base_predictor = MagicMock()
    overlay_predictor = MagicMock()
    overlay_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }
    overlay_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }
    overlay_calls: list[dict[str, object]] = []

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions, year=2026: base_predictor)
    patcher.setattr(
        prediction_flow,
        "build_checkpoint_overlay_predictor",
        lambda **kwargs: overlay_calls.append(dict(kwargs)) or overlay_predictor,
    )
    patcher.setattr(
        prediction_flow,
        "_resolve_current_checkpoint_session",
        lambda year, race_name, is_sprint: "FP2",
    )
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction(
        "Australian Grand Prix",
        "dry",
        artifact_versions,
        is_sprint=False,
    )

    assert overlay_calls == [
        {
            "base_predictor": base_predictor,
            "year": 2026,
            "race_name": "Australian Grand Prix",
            "checkpoint_session": "FP2",
            "is_sprint": False,
        }
    ]
    assert overlay_predictor.predict_qualifying.call_count == 1
    assert base_predictor.predict_qualifying.call_count == 0


def test_run_prediction_keeps_checkpoint_profile_confidence_penalty(patcher):
    """Checkpoint profile blends should still carry the reduced race-input confidence."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}],
        "data_confidence_score": 0.85,
        "data_source": (
            "FP2 checkpoint profile blend (latest stored snapshot: Australian Grand Prix / FP2)"
        ),
        "testing_fallback_used": True,
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "build_checkpoint_overlay_predictor",
        lambda **kwargs: mock_predictor,
    )
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )
    patcher.setattr(
        prediction_flow,
        "_resolve_current_checkpoint_session",
        lambda year, race_name, is_sprint: "FP2",
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction("Bahrain Grand Prix", "dry", artifact_versions, is_sprint=False)

    assert mock_predictor.predict_race.call_args.kwargs["input_confidence"] == pytest.approx(0.8)


def test_run_prediction_passes_race_input_confidence_from_quali_context(patcher):
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}],
        "data_confidence_score": 0.85,
        "data_source": "FP3 short-stint",
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "ACTUAL"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction("Bahrain Grand Prix", "dry", artifact_versions, is_sprint=False)

    called_kwargs = mock_predictor.predict_race.call_args.kwargs
    assert called_kwargs["input_confidence"] == pytest.approx(1.0)


def test_run_prediction_uses_configured_dashboard_simulation_counts(patcher):
    """Dashboard orchestration should respect configured simulation counts."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}],
        "data_confidence_score": 0.85,
        "data_source": "FP3 short-stint",
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "get_prediction_precompute_config",
        lambda: {"qualifying_n_simulations": 120, "race_n_simulations": 140},
    )
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction("Bahrain Grand Prix", "dry", artifact_versions, is_sprint=False)

    assert mock_predictor.predict_qualifying.call_args.kwargs["n_simulations"] == 120
    assert mock_predictor.predict_race.call_args.kwargs["n_simulations"] == 140


def test_run_prediction_passes_sprint_race_input_confidence_from_sq_context(patcher):
    """Sprint race predictions should receive the derived qualifying confidence."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.side_effect = [
        {
            "grid": [{"driver": "NOR", "team": "McLaren", "position": 1}],
            "data_confidence_score": 0.62,
            "data_source": "FP1 short-stint",
        },
        {
            "grid": [{"driver": "NOR", "team": "McLaren", "position": 2}],
            "data_confidence_score": 0.88,
            "data_source": "FP3 short-stint",
        },
    ]
    mock_predictor.predict_sprint_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )
    patcher.setattr(
        prediction_flow,
        "_resolve_current_checkpoint_session",
        lambda year, race_name, is_sprint: "SQ",
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction(
        "Chinese Grand Prix",
        "dry",
        artifact_versions,
        is_sprint=True,
    )

    called_kwargs = mock_predictor.predict_sprint_race.call_args.kwargs
    assert called_kwargs["input_confidence"] == pytest.approx(0.62)


def test_run_prediction_caps_main_race_input_confidence_by_sprint_checkpoint(patcher):
    """Sprint weekends should cap Sunday race confidence more aggressively right after SQ."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.side_effect = [
        {
            "grid": [{"driver": "NOR", "team": "McLaren", "position": 1}],
            "data_confidence_score": 0.62,
            "data_source": "FP1 short-stint",
            "qualifying_stage": "sprint",
        },
        {
            "grid": [{"driver": "NOR", "team": "McLaren", "position": 2}],
            "data_confidence_score": 0.88,
            "data_source": "SQ short-stint",
            "qualifying_stage": "main",
        },
    ]
    mock_predictor.predict_sprint_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )
    patcher.setattr(
        prediction_flow,
        "_resolve_current_checkpoint_session",
        lambda year, race_name, is_sprint: "SQ",
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    prediction_flow.run_prediction(
        "Chinese Grand Prix",
        "dry",
        artifact_versions,
        is_sprint=True,
    )

    called_kwargs = mock_predictor.predict_race.call_args.kwargs
    assert called_kwargs["input_confidence"] == pytest.approx(0.5)


def test_run_prediction_falls_back_when_predict_race_signature_is_legacy(patcher):
    class _LegacyPredictor:
        def __init__(self):
            self.calls = 0

        def predict_qualifying(self, **kwargs):
            return {
                "grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}],
                "data_confidence_score": 0.60,
                "data_source": "FP2 short-stint",
            }

        def predict_race(self, **kwargs):
            self.calls += 1
            if "input_confidence" in kwargs:
                raise TypeError("unexpected keyword argument: input_confidence")
            return {"finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]}

    predictor = _LegacyPredictor()
    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    result = prediction_flow.run_prediction(
        "Bahrain Grand Prix", "dry", artifact_versions, is_sprint=False
    )

    assert predictor.calls == 2
    assert result["race"]["finish_order"][0]["driver"] == "VER"


def test_run_prediction_accepts_real_baseline_predictor_signatures(patcher):
    """Guard against signature drift between dashboard orchestration and predictor facade."""
    from src.predictors.baseline_2026 import Baseline2026Predictor

    predictor = Baseline2026Predictor.__new__(Baseline2026Predictor)
    calls: dict[str, dict] = {}

    class _QualifyingEngine:
        def predict(self, **kwargs):
            calls["qualifying"] = kwargs
            return {"grid": [{"driver": "NOR", "team": "McLaren", "position": 1}]}

    class _RaceEngine:
        def predict(self, **kwargs):
            calls["race"] = kwargs
            return {"finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]}

    predictor.qualifying_engine = _QualifyingEngine()
    predictor.race_engine = _RaceEngine()

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions, year=2026: predictor)
    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2027::car_characteristics": (1, "ts")}
    result = prediction_flow.run_prediction(
        race_name="Bahrain Grand Prix",
        weather="dry",
        _artifact_versions=artifact_versions,
        is_sprint=False,
        year=2027,
    )

    assert result["race"]["finish_order"][0]["driver"] == "NOR"
    assert calls["qualifying"]["year"] == 2027
    assert calls["race"]["year"] == 2027


def test_run_prediction_uses_actual_qualifying_section_without_predicting_when_q_completed(patcher):
    """Completed qualifying should render as an actual result and feed the race model directly."""
    mock_predictor = MagicMock()
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)

    def _fetch_actual(year: int, race_name: str, session_name: str):
        if session_name == "Q":
            return ([{"driver": "RUS", "team": "Mercedes", "position": 1}], "ACTUAL")
        return None, "INCOMPLETE"

    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        _fetch_actual,
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    result = prediction_flow.run_prediction(
        "Australian Grand Prix",
        "dry",
        artifact_versions,
        is_sprint=False,
    )

    assert mock_predictor.predict_qualifying.call_count == 0
    assert result["qualifying"]["result_mode"] == "ACTUAL"
    assert result["qualifying"]["grid"][0]["driver"] == "RUS"
    assert result["race"]["grid_source"] == "ACTUAL"
    assert "no penalties applied" in result["race"]["starting_grid_note"].lower()
    assert mock_predictor.predict_race.call_args.kwargs["qualifying_grid"] == [
        {"driver": "RUS", "team": "Mercedes", "position": 1}
    ]
    assert mock_predictor.predict_race.call_args.kwargs["input_confidence"] == pytest.approx(1.0)


def test_run_prediction_uses_actual_sprint_race_section_after_completion(patcher):
    """Completed sprint sessions should not be regenerated as predictions."""
    mock_predictor = MagicMock()
    mock_predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }
    mock_predictor.predict_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)

    def _fetch_actual(year: int, race_name: str, session_name: str):
        if session_name == "SQ":
            return ([{"driver": "RUS", "team": "Mercedes", "position": 1}], "ACTUAL")
        if session_name == "Sprint":
            return ([{"driver": "RUS", "team": "Mercedes", "position": 1}], "ACTUAL")
        return None, "INCOMPLETE"

    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        _fetch_actual,
    )
    patcher.setattr(
        prediction_flow,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    result = prediction_flow.run_prediction(
        "Chinese Grand Prix",
        "dry",
        artifact_versions,
        is_sprint=True,
    )

    assert mock_predictor.predict_sprint_race.call_count == 0
    assert result["sprint_quali"]["result_mode"] == "ACTUAL"
    assert result["sprint_race"]["result_mode"] == "ACTUAL"
    assert result["main_quali"]["grid_source"] == "PREDICTED"
    assert mock_predictor.predict_qualifying.call_count == 1
    assert mock_predictor.predict_race.call_count == 1


def test_fetch_grid_if_available_uses_actual_grid_for_completed_session(patcher):
    from src.utils import actual_results_fetcher

    patcher.setattr(
        actual_results_fetcher,
        "get_competitive_session_completion_state",
        lambda year, race_name, session_name: "completed",
    )
    patcher.setattr(
        actual_results_fetcher,
        "fetch_actual_session_results",
        lambda year, race_name, session_name: [
            {"driver": "VER", "team": "Red Bull Racing", "position": 1}
        ],
    )

    grid, source = prediction_flow.fetch_grid_if_available(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="Q",
        predicted_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
    )

    assert source == "ACTUAL"
    assert grid[0]["driver"] == "VER"


def test_fetch_grid_if_available_fails_closed_when_completed_results_missing(patcher):
    from src.utils import actual_results_fetcher

    patcher.setattr(
        actual_results_fetcher,
        "get_competitive_session_completion_state",
        lambda year, race_name, session_name: "completed",
    )
    patcher.setattr(
        actual_results_fetcher,
        "fetch_actual_session_results",
        lambda year, race_name, session_name: None,
    )

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        prediction_flow.fetch_grid_if_available(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="Q",
            predicted_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        )


def test_fetch_grid_if_available_raises_when_completion_state_is_unknown(patcher):
    from src.utils import actual_results_fetcher

    patcher.setattr(
        actual_results_fetcher,
        "get_competitive_session_completion_state",
        lambda year, race_name, session_name: "unknown",
    )

    with pytest.raises(
        prediction_flow.CompetitiveSessionStatusUnavailableError,
        match="Could not verify completion state",
    ):
        prediction_flow.fetch_grid_if_available(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="Q",
            predicted_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        )
