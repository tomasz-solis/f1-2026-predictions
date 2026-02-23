from __future__ import annotations

from src.systems.systematic_learning import SystematicLearningSystem


def _sample_prediction_record() -> dict:
    return {
        "metadata": {
            "race_name": "Bahrain Grand Prix",
            "session_name": "FP3",
            "run_id": "test-run-1",
        },
        "qualifying": {
            "predicted_grid": [
                {"position": 1, "driver": "HAD", "team": "Red Bull Racing"},
                {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 3, "driver": "NOR", "team": "McLaren"},
                {"position": 4, "driver": "PIA", "team": "McLaren"},
            ]
        },
        "race": {
            "predicted_results": [
                {"position": 1, "driver": "NOR", "team": "McLaren"},
                {"position": 2, "driver": "PIA", "team": "McLaren"},
                {"position": 3, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 4, "driver": "HAD", "team": "Red Bull Racing"},
            ]
        },
        "actuals": {
            "qualifying": [
                {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 2, "driver": "NOR", "team": "McLaren"},
                {"position": 3, "driver": "HAD", "team": "Red Bull Racing"},
                {"position": 4, "driver": "PIA", "team": "McLaren"},
            ],
            "race": [
                {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
                {"position": 2, "driver": "NOR", "team": "McLaren"},
                {"position": 3, "driver": "PIA", "team": "McLaren"},
                {"position": 4, "driver": "HAD", "team": "Red Bull Racing"},
            ],
        },
    }


def test_update_from_prediction_record_learns_driver_and_teammate_biases(tmp_path):
    state_file = tmp_path / "learning_state.json"
    system = SystematicLearningSystem(state_file=state_file)

    summary = system.update_from_prediction_record(_sample_prediction_record())

    assert summary["sessions_updated"] == 2
    assert summary["driver_updates"] == 8
    assert summary["pair_updates"] == 4

    ver_quali_adjustment = system.get_combined_position_adjustment(
        team="Red Bull Racing",
        driver="VER",
        teammates=["VER", "HAD"],
        session="qualifying",
        min_samples=1,
    )
    had_quali_adjustment = system.get_combined_position_adjustment(
        team="Red Bull Racing",
        driver="HAD",
        teammates=["VER", "HAD"],
        session="qualifying",
        min_samples=1,
    )

    assert ver_quali_adjustment > 0
    assert had_quali_adjustment < 0


def test_update_from_prediction_record_no_actuals_no_updates(tmp_path):
    state_file = tmp_path / "learning_state.json"
    system = SystematicLearningSystem(state_file=state_file)

    prediction_without_actuals = {
        "metadata": {"race_name": "Monaco Grand Prix", "session_name": "FP2"},
        "qualifying": {"predicted_grid": []},
        "race": {"predicted_results": []},
        "actuals": {"qualifying": None, "race": None},
    }

    summary = system.update_from_prediction_record(prediction_without_actuals)

    assert summary["sessions_updated"] == 0
    assert summary["driver_updates"] == 0
    assert summary["pair_updates"] == 0
