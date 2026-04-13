"""Tests for the extracted warmup prediction builder helpers."""

import pytest

from src.dashboard import warmup_prediction_builders


def test_compute_base_features_clamps_completed_q_to_fp3_inputs():
    """Base features should ignore completed Q data and stay on the FP3 prediction boundary."""

    class _Predictor:
        """Capture the checkpoint passed to qualifying prediction."""

        def __init__(self) -> None:
            self.qualifying_calls: list[dict[str, object]] = []

        def predict_qualifying(self, **kwargs):
            self.qualifying_calls.append(dict(kwargs))
            return {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

    predictor = _Predictor()

    result = warmup_prediction_builders.compute_base_features(
        2026,
        "Australian Grand Prix",
        "Q",
        "artifact_hash",
        "boundary_signature",
        predictor=predictor,
        is_sprint=False,
        get_prediction_precompute_config_fn=lambda: {},
        fetch_actual_competitive_results_if_completed_fn=lambda year, race_name, session_name: (
            ([{"position": 1, "driver": "RUS", "team": "Mercedes"}], "ACTUAL")
            if session_name == "Q"
            else (None, "INCOMPLETE")
        ),
        build_actual_qualifying_section_fn=lambda grid, session_name: {
            "result_mode": "ACTUAL",
            "session_name": session_name,
            "grid": grid,
        },
        fetch_grid_if_available_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fetch_grid_if_available should not run past the FP3 boundary")
        ),
        derive_race_input_confidence_fn=lambda qualifying_payload, grid_source: (
            1.0 if grid_source == "ACTUAL" else 0.45
        ),
        cap_predicted_main_race_input_confidence_fn=lambda *args, **kwargs: 0.0,
    )

    assert predictor.qualifying_calls[0]["checkpoint_session_name"] == "FP3"
    assert result["qualifying"]["grid_source"] == "PREDICTED"
    assert result["qualifying"]["grid"][0]["driver"] == "NOR"
    assert result["race_input_confidence"] == 0.45


def test_compute_base_features_keeps_sprint_checkpoint_profile_calls():
    """Sprint warmup base features should pin both qualifying calls to the checkpoint state."""

    class _Predictor:
        """Capture qualifying kwargs for both sprint and main branches."""

        def __init__(self) -> None:
            self.qualifying_calls: list[dict[str, object]] = []

        def predict_qualifying(self, **kwargs):
            self.qualifying_calls.append(dict(kwargs))
            return {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

    predictor = _Predictor()
    result = warmup_prediction_builders.compute_base_features(
        2026,
        "Chinese Grand Prix",
        "FP1",
        "artifact_hash",
        "boundary_signature",
        predictor=predictor,
        is_sprint=True,
        get_prediction_precompute_config_fn=lambda: {"qualifying_n_simulations": 50},
        fetch_actual_competitive_results_if_completed_fn=lambda year, race_name, session_name: (
            None,
            "INCOMPLETE",
        ),
        build_actual_qualifying_section_fn=lambda grid, session_name: {
            "result_mode": "ACTUAL",
            "session_name": session_name,
            "grid": grid,
        },
        fetch_grid_if_available_fn=lambda year, race_name, session_name, predicted_grid: (
            predicted_grid,
            "PREDICTED",
        ),
        derive_race_input_confidence_fn=lambda qualifying_payload, grid_source: 0.55,
        cap_predicted_main_race_input_confidence_fn=lambda confidence, **kwargs: confidence + 0.2,
    )

    assert [call["qualifying_stage"] for call in predictor.qualifying_calls] == ["sprint", "main"]
    assert [call["checkpoint_session_name"] for call in predictor.qualifying_calls] == [
        "FP1",
        "FP1",
    ]
    assert result["is_sprint"] is True
    assert result["main_race_input_confidence"] == pytest.approx(0.75)


def test_compute_base_features_uses_actual_sq_but_not_actual_q():
    """Sprint base features may use SQ results but must keep main qualifying on the SQ boundary."""

    class _Predictor:
        """Capture whether main qualifying still runs from the SQ checkpoint."""

        def __init__(self) -> None:
            self.qualifying_calls: list[dict[str, object]] = []

        def predict_qualifying(self, **kwargs):
            self.qualifying_calls.append(dict(kwargs))
            return {"grid": [{"position": 1, "driver": "PIA", "team": "McLaren"}]}

    predictor = _Predictor()

    def _actual_results(year, race_name, session_name):
        del year, race_name
        if session_name == "SQ":
            return ([{"position": 1, "driver": "NOR", "team": "McLaren"}], "ACTUAL")
        if session_name == "Q":
            return ([{"position": 1, "driver": "RUS", "team": "Mercedes"}], "ACTUAL")
        return (None, "INCOMPLETE")

    result = warmup_prediction_builders.compute_base_features(
        2026,
        "Chinese Grand Prix",
        "Q",
        "artifact_hash",
        "boundary_signature",
        predictor=predictor,
        is_sprint=True,
        get_prediction_precompute_config_fn=lambda: {},
        fetch_actual_competitive_results_if_completed_fn=_actual_results,
        build_actual_qualifying_section_fn=lambda grid, session_name: {
            "result_mode": "ACTUAL",
            "session_name": session_name,
            "grid": grid,
        },
        fetch_grid_if_available_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fetch_grid_if_available should not run for completed qualifying")
        ),
        derive_race_input_confidence_fn=lambda qualifying_payload, grid_source: (
            1.0 if grid_source == "ACTUAL" else 0.0
        ),
        cap_predicted_main_race_input_confidence_fn=lambda confidence, **kwargs: confidence,
    )

    assert result["sprint_quali"]["result_mode"] == "ACTUAL"
    assert result["sprint_quali"]["grid"][0]["driver"] == "NOR"
    assert [call["qualifying_stage"] for call in predictor.qualifying_calls] == ["main"]
    assert predictor.qualifying_calls[0]["checkpoint_session_name"] == "SQ"
    assert result["main_quali"]["grid_source"] == "PREDICTED"
    assert result["main_quali"]["grid"][0]["driver"] == "PIA"
    assert result["sprint_race_input_confidence"] == 1.0
    assert result["main_race_input_confidence"] == 0.0


def test_compute_base_features_does_not_refresh_q_grid_after_fp3():
    """Normal-weekend base features should not pull in Q results once the boundary is FP3."""

    class _Predictor:
        """Return one predicted grid that should remain untouched."""

        def predict_qualifying(self, **kwargs):
            del kwargs
            return {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

    result = warmup_prediction_builders.compute_base_features(
        2026,
        "Australian Grand Prix",
        "FP3",
        "artifact_hash",
        "boundary_signature",
        predictor=_Predictor(),
        is_sprint=False,
        get_prediction_precompute_config_fn=lambda: {},
        fetch_actual_competitive_results_if_completed_fn=lambda year, race_name, session_name: (
            None,
            "INCOMPLETE",
        ),
        build_actual_qualifying_section_fn=lambda grid, session_name: {
            "result_mode": "ACTUAL",
            "session_name": session_name,
            "grid": grid,
        },
        fetch_grid_if_available_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fetch_grid_if_available should not run after FP3")
        ),
        derive_race_input_confidence_fn=lambda qualifying_payload, grid_source: (
            1.0 if grid_source == "ACTUAL" else 0.35
        ),
        cap_predicted_main_race_input_confidence_fn=lambda *args, **kwargs: 0.0,
    )

    assert result["qualifying"]["grid_source"] == "PREDICTED"
    assert result["qualifying"]["grid"][0]["driver"] == "NOR"
    assert result["race_input_confidence"] == 0.35


def test_compute_weather_predictions_rejects_unknown_weather():
    """Weather overlays should fail fast when scenario selection is invalid."""
    with pytest.raises(ValueError, match="Invalid weather scenario"):
        warmup_prediction_builders.compute_weather_predictions(
            {
                "is_sprint": False,
                "qualifying": {"grid_source": "PREDICTED"},
                "qualifying_grid_for_race": [],
                "race_input_confidence": 0.4,
                "timing": {"qualifying": 0.1},
            },
            "storm",
            predictor=object(),
            year=2026,
            target_race="Bahrain Grand Prix",
            valid_weather_scenarios={"dry", "mixed", "rain"},
            fetch_actual_competitive_results_if_completed_fn=lambda *args, **kwargs: (
                None,
                "INCOMPLETE",
            ),
            build_actual_race_section_fn=lambda *args, **kwargs: {},
            predict_sprint_race_with_optional_confidence_fn=lambda *args, **kwargs: {},
            predict_race_with_optional_confidence_fn=lambda *args, **kwargs: {},
            build_starting_grid_note_fn=lambda session_name: session_name,
        )


def test_compute_weather_predictions_uses_actual_main_race_for_sprint_weekend():
    """Sprint weather overlays should reuse completed main-race results and keep the Q note."""

    def _actual_results(year, race_name, session_name):
        del year, race_name
        if session_name == "R":
            return ([{"position": 1, "driver": "VER", "team": "Red Bull"}], "ACTUAL")
        return (None, "INCOMPLETE")

    result = warmup_prediction_builders.compute_weather_predictions(
        {
            "is_sprint": True,
            "sprint_quali": {"grid_source": "PREDICTED"},
            "sprint_grid_for_race": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "sprint_race_input_confidence": 0.58,
            "main_quali": {"grid_source": "ACTUAL"},
            "main_grid_for_race": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "main_race_input_confidence": 1.0,
            "timing": {"sprint_quali": 0.1, "main_quali": 0.2},
        },
        "dry",
        predictor=object(),
        year=2026,
        target_race="Chinese Grand Prix",
        valid_weather_scenarios={"dry", "mixed", "rain"},
        fetch_actual_competitive_results_if_completed_fn=_actual_results,
        build_actual_race_section_fn=lambda finish_order, session_name: {
            "result_mode": "ACTUAL",
            "session_name": session_name,
            "finish_order": finish_order,
        },
        predict_sprint_race_with_optional_confidence_fn=lambda predictor, **kwargs: {
            "finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "result_mode": "PREDICTED",
        },
        predict_race_with_optional_confidence_fn=lambda predictor, **kwargs: (_ for _ in ()).throw(
            AssertionError("predict_race should not run for completed main race")
        ),
        build_starting_grid_note_fn=lambda session_name: f"note {session_name}",
    )

    assert result["main_race"]["result_mode"] == "ACTUAL"
    assert result["main_race"]["grid_source"] == "ACTUAL"
    assert result["main_race"]["starting_grid_note"] == "note Q"
    assert "input_confidence" not in result["main_race"]


def test_compute_weather_predictions_uses_actual_race_for_normal_weekend():
    """Normal-weekend weather overlays should reuse completed race results."""

    result = warmup_prediction_builders.compute_weather_predictions(
        {
            "is_sprint": False,
            "qualifying": {"grid_source": "ACTUAL", "grid": []},
            "qualifying_grid_for_race": [{"position": 1, "driver": "RUS", "team": "Mercedes"}],
            "race_input_confidence": 1.0,
            "timing": {"qualifying": 0.15},
        },
        "dry",
        predictor=object(),
        year=2026,
        target_race="Australian Grand Prix",
        valid_weather_scenarios={"dry", "mixed", "rain"},
        fetch_actual_competitive_results_if_completed_fn=lambda year, race_name, session_name: (
            ([{"position": 1, "driver": "RUS", "team": "Mercedes"}], "ACTUAL")
            if session_name == "R"
            else (None, "INCOMPLETE")
        ),
        build_actual_race_section_fn=lambda finish_order, session_name: {
            "result_mode": "ACTUAL",
            "session_name": session_name,
            "finish_order": finish_order,
        },
        predict_sprint_race_with_optional_confidence_fn=lambda *args, **kwargs: {},
        predict_race_with_optional_confidence_fn=lambda predictor, **kwargs: (_ for _ in ()).throw(
            AssertionError("predict_race should not run for completed race")
        ),
        build_starting_grid_note_fn=lambda session_name: f"note {session_name}",
    )

    assert result["race"]["result_mode"] == "ACTUAL"
    assert result["race"]["grid_source"] == "ACTUAL"
    assert result["race"]["starting_grid_note"] == "note Q"
    assert "input_confidence" not in result["race"]
