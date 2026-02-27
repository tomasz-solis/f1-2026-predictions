import importlib.util
import json
from pathlib import Path

import pandas as pd


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
