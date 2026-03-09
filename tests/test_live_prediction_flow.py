"""Focused tests for live prediction flow helpers."""

from src.dashboard.live_prediction_flow import (
    prediction_payload_for_session,
    prediction_targets_for_checkpoint,
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
