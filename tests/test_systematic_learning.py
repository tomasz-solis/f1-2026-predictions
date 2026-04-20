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
                {
                    "position": 1,
                    "median_position": 1,
                    "p5": 1,
                    "p95": 1,
                    "driver": "HAD",
                    "team": "Red Bull Racing",
                },
                {
                    "position": 2,
                    "median_position": 2,
                    "p5": 2,
                    "p95": 2,
                    "driver": "VER",
                    "team": "Red Bull Racing",
                },
                {
                    "position": 3,
                    "median_position": 3,
                    "p5": 3,
                    "p95": 3,
                    "driver": "NOR",
                    "team": "McLaren",
                },
                {
                    "position": 4,
                    "median_position": 4,
                    "p5": 4,
                    "p95": 4,
                    "driver": "PIA",
                    "team": "McLaren",
                },
            ]
        },
        "race": {
            "predicted_results": [
                {
                    "position": 1,
                    "median_position": 1,
                    "p5": 1,
                    "p95": 1,
                    "driver": "NOR",
                    "team": "McLaren",
                },
                {
                    "position": 2,
                    "median_position": 2,
                    "p5": 2,
                    "p95": 2,
                    "driver": "PIA",
                    "team": "McLaren",
                },
                {
                    "position": 3,
                    "median_position": 3,
                    "p5": 3,
                    "p95": 3,
                    "driver": "VER",
                    "team": "Red Bull Racing",
                },
                {
                    "position": 4,
                    "median_position": 4,
                    "p5": 4,
                    "p95": 4,
                    "driver": "HAD",
                    "team": "Red Bull Racing",
                },
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


def test_update_from_prediction_record_learns_interval_radius_from_residuals(tmp_path):
    state_file = tmp_path / "learning_state.json"
    system = SystematicLearningSystem(state_file=state_file)

    summary = system.update_from_prediction_record(_sample_prediction_record())

    qualifying_summary = system.get_interval_calibration_summary(
        "qualifying",
        min_samples=1,
        target_coverage=0.90,
        max_adjustment=5.0,
    )
    race_summary = system.get_interval_calibration_summary(
        "race",
        min_samples=1,
        target_coverage=0.90,
        max_adjustment=5.0,
    )

    assert summary["interval_samples"] == 8
    assert qualifying_summary["sample_count"] == 4.0
    assert qualifying_summary["empirical_coverage"] == 0.25
    assert qualifying_summary["learned_radius"] >= 1.0
    assert race_summary["sample_count"] == 4.0
    assert race_summary["learned_radius"] >= 1.0


def test_update_from_prediction_record_skips_retrospective_predictions(tmp_path):
    state_file = tmp_path / "learning_state.json"
    system = SystematicLearningSystem(state_file=state_file)

    prediction = _sample_prediction_record()
    prediction["metadata"].update(
        {
            "source": "checkpoint_reconstruction",
            "predicted_at": "2026-03-17T10:26:25+00:00",
            "information_cutoff_at": "2026-03-06T04:59:59+00:00",
        }
    )

    summary = system.update_from_prediction_record(prediction)

    assert summary["skipped"] is True
    assert summary["skip_reason"] == "retrospective_prediction"
    assert summary["sessions_updated"] == 0
    assert (
        system.get_combined_position_adjustment(
            team="Red Bull Racing",
            driver="VER",
            teammates=["VER", "HAD"],
            session="qualifying",
            min_samples=1,
        )
        == 0.0
    )


def test_update_from_prediction_record_skips_duplicate_run_id(tmp_path):
    state_file = tmp_path / "learning_state.json"
    system = SystematicLearningSystem(state_file=state_file)

    prediction = _sample_prediction_record()
    first = system.update_from_prediction_record(prediction)
    second = system.update_from_prediction_record(prediction)

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["skip_reason"] == "duplicate_run_id"
