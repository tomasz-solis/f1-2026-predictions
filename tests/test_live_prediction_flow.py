"""Focused tests for live prediction flow helpers."""

from src.dashboard.live_prediction_flow import prediction_payload_for_session


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
