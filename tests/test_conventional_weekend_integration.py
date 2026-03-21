"""Integration tests for the conventional weekend checkpoint cascade."""

from unittest.mock import MagicMock

import pytest

from src.dashboard import prediction_flow
from src.utils.accuracy_targets import (
    TARGET_GRAND_PRIX_RACE,
    TARGET_MAIN_QUALIFYING,
    eligible_target_keys,
    target_checkpoint_sequence,
)


def _build_normal_weekend_predictor() -> MagicMock:
    """Return a stable predictor stub for conventional-weekend flow tests."""
    predictor = MagicMock()
    predictor.predict_qualifying.return_value = {
        "grid": [{"driver": "NOR", "team": "McLaren", "position": 1}],
        "data_confidence_score": 0.73,
        "data_source": "FP3 short-stint",
    }
    predictor.predict_race.return_value = {
        "finish_order": [{"driver": "NOR", "team": "McLaren", "position": 1}]
    }
    return predictor


@pytest.mark.parametrize(
    (
        "checkpoint_session",
        "expected_target_keys",
        "expected_qualifying_mode",
        "expected_grid_source",
        "expect_predicted_qualifying",
    ),
    [
        pytest.param(
            "PRE",
            (TARGET_MAIN_QUALIFYING, TARGET_GRAND_PRIX_RACE),
            "PREDICTED",
            "PREDICTED",
            True,
            id="pre",
        ),
        pytest.param(
            "FP1",
            (TARGET_MAIN_QUALIFYING, TARGET_GRAND_PRIX_RACE),
            "PREDICTED",
            "PREDICTED",
            True,
            id="fp1",
        ),
        pytest.param(
            "FP2",
            (TARGET_MAIN_QUALIFYING, TARGET_GRAND_PRIX_RACE),
            "PREDICTED",
            "PREDICTED",
            True,
            id="fp2",
        ),
        pytest.param(
            "FP3",
            (TARGET_MAIN_QUALIFYING, TARGET_GRAND_PRIX_RACE),
            "PREDICTED",
            "PREDICTED",
            True,
            id="fp3",
        ),
        pytest.param(
            "Q",
            (TARGET_GRAND_PRIX_RACE,),
            "ACTUAL",
            "ACTUAL",
            False,
            id="q-completed",
        ),
    ],
)
def test_conventional_weekend_checkpoint_matrix_keeps_targets_and_dashboard_flow_aligned(
    patcher,
    checkpoint_session: str,
    expected_target_keys: tuple[str, ...],
    expected_qualifying_mode: str,
    expected_grid_source: str,
    expect_predicted_qualifying: bool,
):
    """Conventional checkpoints should keep scoring targets and request flow in sync."""
    predictor = _build_normal_weekend_predictor()
    actual_qualifying_grid = [{"driver": "RUS", "team": "Mercedes", "position": 1}]
    grid_refresh_sessions: list[str] = []

    patcher.setattr(
        prediction_flow,
        "get_predictor",
        lambda _artifact_versions, year=2026: predictor,
    )
    patcher.setattr(
        prediction_flow,
        "_resolve_current_checkpoint_session",
        lambda year, race_name, is_sprint: checkpoint_session,
    )

    def _fetch_actual_results(year: int, race_name: str, session_name: str):
        del year, race_name
        if checkpoint_session == "Q" and session_name == "Q":
            return actual_qualifying_grid, "ACTUAL"
        return None, "INCOMPLETE"

    def _fetch_grid(year: int, race_name: str, session_name: str, predicted_grid: list):
        del year, race_name
        grid_refresh_sessions.append(session_name)
        return predicted_grid, "PREDICTED"

    patcher.setattr(
        prediction_flow,
        "fetch_actual_competitive_results_if_completed",
        _fetch_actual_results,
    )
    patcher.setattr(prediction_flow, "fetch_grid_if_available", _fetch_grid)

    artifact_versions = {"car_characteristics::2026::car_characteristics": (1, "ts")}
    result = prediction_flow.run_prediction(
        race_name="Australian Grand Prix",
        weather="dry",
        _artifact_versions=artifact_versions,
        is_sprint=False,
    )

    assert set(result) == {"qualifying", "race"}
    assert eligible_target_keys(checkpoint_session, is_sprint=False) == expected_target_keys
    assert checkpoint_session in target_checkpoint_sequence(TARGET_GRAND_PRIX_RACE, "normal")
    assert (checkpoint_session in target_checkpoint_sequence(TARGET_MAIN_QUALIFYING, "normal")) is (
        TARGET_MAIN_QUALIFYING in expected_target_keys
    )
    assert (
        str(result["qualifying"].get("result_mode", "PREDICTED")).upper()
        == expected_qualifying_mode
    )
    assert result["race"]["grid_source"] == expected_grid_source

    if expect_predicted_qualifying:
        assert predictor.predict_qualifying.call_count == 1
        assert grid_refresh_sessions == ["Q"]
        assert result["qualifying"]["grid_source"] == "PREDICTED"
        assert predictor.predict_qualifying.call_args.kwargs["checkpoint_session_name"] == (
            checkpoint_session
        )
        assert predictor.predict_race.call_args.kwargs["qualifying_grid"] == [
            {"driver": "NOR", "team": "McLaren", "position": 1}
        ]
        assert predictor.predict_race.call_args.kwargs["input_confidence"] == pytest.approx(0.73)
        return

    assert predictor.predict_qualifying.call_count == 0
    assert grid_refresh_sessions == []
    assert result["qualifying"]["grid"] == actual_qualifying_grid
    assert result["qualifying"]["grid_source"] == "ACTUAL"
    assert predictor.predict_race.call_args.kwargs["qualifying_grid"] == actual_qualifying_grid
    assert predictor.predict_race.call_args.kwargs["input_confidence"] == pytest.approx(1.0)
    assert "no penalties applied" in result["race"]["starting_grid_note"].lower()
