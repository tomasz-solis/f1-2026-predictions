"""Tests for extracted checkpoint-save helpers."""

from src.dashboard import prediction_checkpointing


def test_prediction_payload_for_session_uses_main_outputs_after_sprint_weekend_q():
    prediction_results = {
        "sprint_quali": {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "sprint_race": {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "main_quali": {"grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "main_race": {"finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
    }

    qualifying_grid, race_finish, fp_blend_info = (
        prediction_checkpointing.prediction_payload_for_session(
            prediction_results=prediction_results,
            is_sprint=True,
            session_name="Q",
        )
    )

    assert qualifying_grid == prediction_results["main_quali"]["grid"]
    assert race_finish == prediction_results["main_race"]["finish_order"]
    assert fp_blend_info == {}


def test_prediction_payload_for_session_clamps_sprint_race_to_sq_outputs():
    prediction_results = {
        "sprint_quali": {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "sprint_race": {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "main_quali": {"grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "main_race": {"finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
    }

    qualifying_grid, race_finish, fp_blend_info = (
        prediction_checkpointing.prediction_payload_for_session(
            prediction_results=prediction_results,
            is_sprint=True,
            session_name="Sprint",
        )
    )

    assert qualifying_grid == prediction_results["sprint_quali"]["grid"]
    assert race_finish == prediction_results["sprint_race"]["finish_order"]
    assert fp_blend_info == {}


def test_prediction_targets_for_checkpoint_excludes_actual_targets():
    prediction_results = {
        "qualifying": {
            "grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "result_mode": "ACTUAL",
        },
        "race": {
            "finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "result_mode": "PREDICTED",
        },
    }

    targets = prediction_checkpointing.prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=False,
        session_name="Q",
    )

    assert set(targets) == {"grand_prix_race"}


def test_prediction_model_diagnostics_for_sections_keeps_bounded_signal_metadata():
    diagnostics = prediction_checkpointing.prediction_model_diagnostics_for_sections(
        qualifying_section={
            "data_source": "FP2",
            "data_regime": "practice",
            "fp_blend_weight_used": 0.72,
            "characteristics_profile_used": "short_run",
            "teams_with_characteristics_profile": 11,
            "qualifying_residual_model_used": False,
            "teammate_head_to_head": [{"large": "payload"}],
        },
        race_section={
            "data_regime": "predicted_grid",
            "input_confidence": 0.66,
            "characteristics_profile_used": "long_run",
            "teams_with_characteristics_profile": 10,
            "race_residual_model_used": False,
            "track_temperature_context": {"source": "session_blend", "track_c": 38.0},
            "compound_strategies": {"VER": {"one_stop": 0.6}},
        },
    )

    assert diagnostics["model_diagnostics_schema_version"] == 1
    assert diagnostics["qualifying_model_diagnostics"] == {
        "data_source": "FP2",
        "data_regime": "practice",
        "fp_blend_weight_used": 0.72,
        "characteristics_profile_used": "short_run",
        "teams_with_characteristics_profile": 11,
        "qualifying_residual_model_used": False,
    }
    race_diagnostics = diagnostics["race_model_diagnostics"]
    assert race_diagnostics["characteristics_profile_used"] == "long_run"
    assert race_diagnostics["track_temperature_context"]["source"] == "session_blend"
    assert race_diagnostics["compound_strategy_count"] == 1


def test_save_prediction_if_enabled_core_prefers_post_quali_boundary_over_override():
    class _Logger:
        def __init__(self) -> None:
            self.saved_session_name: str | None = None
            self.saved_kwargs: dict | None = None
            self.artifact_store = None

        def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
            return False

        def save_prediction(self, **kwargs):
            self.saved_session_name = str(kwargs["session_name"])
            self.saved_kwargs = dict(kwargs)

    class _Streamlit:
        @staticmethod
        def info(message: str) -> None:
            del message

        @staticmethod
        def warning(message: str) -> None:
            raise AssertionError(message)

    logger_instance = _Logger()
    prediction_results = {
        "sprint_quali": {
            "grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "result_mode": "ACTUAL",
        },
        "sprint_race": {
            "finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "result_mode": "ACTUAL",
        },
        "main_quali": {
            "grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "result_mode": "ACTUAL",
            "grid_source": "ACTUAL",
        },
        "main_race": {
            "finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "result_mode": "PREDICTED",
            "grid_source": "ACTUAL",
        },
        "_prediction_context": {"boundary_session_name": "Q"},
    }

    prediction_checkpointing.save_prediction_if_enabled_core(
        enable_logging=True,
        prediction_results=prediction_results,
        is_sprint=True,
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
        detector_factory=lambda: (_ for _ in ()).throw(AssertionError("detector should not run")),
        prediction_logger_factory=lambda: logger_instance,
        st_module=_Streamlit(),
        checkpoint_session_override="SPRINT",
    )

    assert logger_instance.saved_session_name == "Q"
    assert logger_instance.saved_kwargs is not None
    metadata = logger_instance.saved_kwargs["metadata"]
    assert metadata["model_diagnostics_schema_version"] == 1
    assert "qualifying_model_diagnostics" in metadata
    assert "race_model_diagnostics" in metadata
