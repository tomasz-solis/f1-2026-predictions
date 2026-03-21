import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_baseline_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_2026_baseline.py"
    spec = importlib.util.spec_from_file_location("generate_2026_baseline_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_team_file(output_dir: Path) -> dict:
    team_file = output_dir / "car_characteristics" / "2026_car_characteristics.json"
    return json.loads(team_file.read_text())


def test_generate_team_characteristics_defaults_to_ranked_seed(tmp_path):
    module = _load_baseline_module()

    module.generate_team_characteristics(tmp_path)
    payload = _load_team_file(tmp_path)
    teams = payload["teams"]

    assert teams["McLaren"]["overall_performance"] > teams["Williams"]["overall_performance"]
    assert teams["Williams"]["overall_performance"] > teams["Cadillac F1"]["overall_performance"]
    assert teams["McLaren"]["overall_performance"] != teams["Ferrari"]["overall_performance"]


def test_generate_team_characteristics_neutral_mode_sets_all_equal(tmp_path):
    module = _load_baseline_module()

    module.generate_team_characteristics(tmp_path, neutral_start=True)
    payload = _load_team_file(tmp_path)
    values = {team["overall_performance"] for team in payload["teams"].values()}

    assert values == {0.5}


def test_generate_team_characteristics_preserves_enriched_existing_file_by_default(tmp_path):
    module = _load_baseline_module()
    output_file = tmp_path / "car_characteristics" / "2026_car_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    enriched_payload = {
        "year": 2026,
        "generated_at": "2026-03-03T23:33:59.825095",
        "data_freshness": "TESTING_ENRICHED",
        "teams": {
            "McLaren": {
                "overall_performance": 0.85,
                "uncertainty": 0.3,
                "note": "existing enriched payload",
                "last_updated": "2026-03-03T23:33:59.825095",
                "races_completed": 0,
                "directionality": {"max_speed": -0.0098},
                "testing_characteristics": {"overall_pace": 0.4879},
            }
        },
    }
    output_file.write_text(json.dumps(enriched_payload, indent=2))

    module.generate_team_characteristics(tmp_path)

    assert _load_team_file(tmp_path) == enriched_payload


def test_generate_team_characteristics_force_reset_overwrites_enriched_file(tmp_path):
    module = _load_baseline_module()
    output_file = tmp_path / "car_characteristics" / "2026_car_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    enriched_payload = {
        "year": 2026,
        "generated_at": "2026-03-03T23:33:59.825095",
        "data_freshness": "TESTING_ENRICHED",
        "teams": {
            "McLaren": {
                "overall_performance": 0.85,
                "uncertainty": 0.3,
                "note": "existing enriched payload",
                "last_updated": "2026-03-03T23:33:59.825095",
                "races_completed": 0,
                "directionality": {"max_speed": -0.0098},
                "testing_characteristics": {"overall_pace": 0.4879},
            }
        },
    }
    output_file.write_text(json.dumps(enriched_payload, indent=2))

    module.generate_team_characteristics(tmp_path, force_reset=True)
    payload = _load_team_file(tmp_path)

    assert payload["data_freshness"] == "BASELINE_PRESEASON"
    assert payload["teams"]["McLaren"]["last_updated"] is None
    assert "directionality" not in payload["teams"]["McLaren"]


def test_estimate_pit_losses_from_pit_timestamps():
    module = _load_baseline_module()
    laps = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "BBB", "BBB"],
            "PitInTime": [
                pd.to_timedelta("00:10:00"),
                pd.NaT,
                pd.to_timedelta("00:20:00"),
                pd.NaT,
            ],
            "PitOutTime": [
                pd.NaT,
                pd.to_timedelta("00:10:22"),
                pd.NaT,
                pd.to_timedelta("00:20:28"),
            ],
            "LapTime": [
                pd.to_timedelta("00:01:40"),
                pd.to_timedelta("00:01:43"),
                pd.to_timedelta("00:01:42"),
                pd.to_timedelta("00:01:44"),
            ],
        }
    )

    losses = module._estimate_pit_losses_from_laps(laps)
    assert sorted(round(v, 1) for v in losses) == [22.0, 28.0]


def test_estimate_pit_losses_falls_back_to_lap_delta():
    module = _load_baseline_module()
    laps = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA", "AAA", "AAA"],
            "PitInTime": [pd.NaT, pd.NaT, pd.to_timedelta("00:10:00"), pd.NaT],
            "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            "LapTime": [
                pd.to_timedelta("00:01:30"),
                pd.to_timedelta("00:01:31"),
                pd.to_timedelta("00:01:52"),
                pd.to_timedelta("00:01:29"),
            ],
        }
    )

    losses = module._estimate_pit_losses_from_laps(laps)
    assert len(losses) == 1
    assert 20.0 <= losses[0] <= 23.0


def test_filter_outlier_pit_losses_removes_extremes():
    module = _load_baseline_module()
    losses = [21.8, 22.0, 22.1, 22.2, 22.3, 22.4, 29.9]

    filtered = module._filter_outlier_pit_losses(losses)
    assert 29.9 not in filtered
    assert filtered == [21.8, 22.0, 22.1, 22.2, 22.3, 22.4]


def test_filter_outlier_pit_losses_keeps_small_samples():
    module = _load_baseline_module()
    losses = [21.0, 22.0, 29.5]

    filtered = module._filter_outlier_pit_losses(losses)
    assert filtered == losses


def test_estimate_overtaking_changes_per_lap_uses_lap_by_lap_positions():
    module = _load_baseline_module()
    laps = pd.DataFrame(
        {
            "LapNumber": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            "Driver": ["A", "B", "C", "D", "E"] * 3,
            "Position": [1, 2, 3, 4, 5, 2, 1, 3, 4, 5, 2, 1, 4, 3, 5],
            "PitOutTime": [pd.NaT] * 15,
        }
    )

    changes = module._estimate_overtaking_changes_per_lap(laps)
    assert changes == 2.0


def test_changes_per_lap_to_overtaking_difficulty_is_calibrated():
    module = _load_baseline_module()

    assert module._changes_per_lap_to_overtaking_difficulty(None) == 0.5
    assert module._changes_per_lap_to_overtaking_difficulty(1.0) == 0.95
    assert module._changes_per_lap_to_overtaking_difficulty(5.0) == pytest.approx(0.3846, abs=1e-4)


def test_resolve_track_overtaking_difficulty_uses_track_prior_when_history_missing():
    module = _load_baseline_module()

    assert module._resolve_track_overtaking_difficulty("Monaco Grand Prix", None) == 0.95
    assert module._resolve_track_overtaking_difficulty("Bahrain Grand Prix", None) == 0.40
