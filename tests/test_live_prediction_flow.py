"""Focused tests for live prediction flow helpers."""

from src.dashboard.live_prediction_flow import (
    prediction_payload_for_session,
    prediction_targets_for_checkpoint,
    save_prediction_if_enabled_core,
)


def test_prediction_payload_for_session_uses_sprint_phase_outputs_for_sq():
    """Sprint early-phase checkpoints should persist sprint cascade outputs."""
    prediction_results = {
        "sprint_quali": {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "sprint_race": {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "main_quali": {"grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "main_race": {"finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
    }

    qualifying_grid, race_finish, fp_blend_info = prediction_payload_for_session(
        prediction_results=prediction_results,
        is_sprint=True,
        session_name="SQ",
    )

    assert qualifying_grid == prediction_results["sprint_quali"]["grid"]
    assert race_finish == prediction_results["sprint_race"]["finish_order"]
    assert fp_blend_info == {}


def test_prediction_payload_for_session_uses_main_outputs_for_main_sessions():
    """Sprint late-phase checkpoints should persist main qualifying/race outputs."""
    prediction_results = {
        "sprint_quali": {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "sprint_race": {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "main_quali": {"grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "main_race": {"finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
    }

    qualifying_grid, race_finish, fp_blend_info = prediction_payload_for_session(
        prediction_results=prediction_results,
        is_sprint=True,
        session_name="Q",
    )

    assert qualifying_grid == prediction_results["main_quali"]["grid"]
    assert race_finish == prediction_results["main_race"]["finish_order"]
    assert fp_blend_info == {}


def test_prediction_targets_for_checkpoint_keeps_multiple_sprint_targets():
    """Early sprint checkpoints should retain both sprint and main targets."""
    prediction_results = {
        "sprint_quali": {"grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "sprint_race": {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "main_quali": {"grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "main_race": {"finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
    }

    targets = prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=True,
        session_name="FP1",
    )

    assert set(targets) == {
        "sprint_qualifying",
        "sprint_race",
        "main_qualifying",
        "grand_prix_race",
    }


def test_prediction_targets_for_checkpoint_excludes_actual_targets():
    """Completed-session targets should not be treated as saved forecasts."""
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

    targets = prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=False,
        session_name="Q",
    )

    assert set(targets) == {"grand_prix_race"}


def test_save_prediction_prefers_prediction_context_boundary_over_override():
    """Persisted predictions should keep the boundary they were actually generated from."""

    class _Logger:
        """Capture save calls without touching real storage."""

        def __init__(self) -> None:
            self.saved_session_name: str | None = None
            self.artifact_store = None

        def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
            return False

        def save_prediction(self, **kwargs):
            self.saved_session_name = str(kwargs["session_name"])

    class _Streamlit:
        """Minimal Streamlit stub for save helper tests."""

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

    save_prediction_if_enabled_core(
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
