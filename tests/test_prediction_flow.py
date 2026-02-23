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

    patcher.setattr(prediction_flow, "get_predictor", lambda _versions: mock_predictor)

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
    assert mock_predictor.predict_qualifying.call_args.kwargs["year"] == 2027


def test_fetch_grid_if_available_uses_actual_grid_for_completed_session(patcher):
    from src.utils import actual_results_fetcher

    patcher.setattr(
        actual_results_fetcher,
        "is_competitive_session_completed",
        lambda year, race_name, session_name: True,
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
        "is_competitive_session_completed",
        lambda year, race_name, session_name: True,
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
