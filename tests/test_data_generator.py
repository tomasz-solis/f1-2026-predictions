from __future__ import annotations

import json
from pathlib import Path

import src.data.data_generator as data_generator


def _write_baseline_artifacts(data_dir: Path, team_freshness: str, track_freshness: str) -> None:
    car_dir = data_dir / "car_characteristics"
    track_dir = data_dir / "track_characteristics"
    car_dir.mkdir(parents=True, exist_ok=True)
    track_dir.mkdir(parents=True, exist_ok=True)
    (car_dir / "2026_car_characteristics.json").write_text(
        json.dumps({"data_freshness": team_freshness})
    )
    (track_dir / "2026_track_characteristics.json").write_text(
        json.dumps({"data_freshness": track_freshness})
    )


def test_create_baseline_if_missing_triggers_generation_when_files_missing(tmp_path, patcher):
    data_dir = tmp_path / "processed"
    calls: list[Path] = []
    patcher.setattr(
        data_generator,
        "generate_quick_baseline",
        lambda path: calls.append(Path(path)),
    )

    data_generator.create_baseline_if_missing(data_dir)

    assert calls == [data_dir]


def test_create_baseline_if_missing_triggers_generation_for_unknown_freshness(tmp_path, patcher):
    data_dir = tmp_path / "processed"
    _write_baseline_artifacts(
        data_dir, team_freshness="UNKNOWN", track_freshness="BASELINE_PRESEASON"
    )

    calls: list[Path] = []
    patcher.setattr(
        data_generator,
        "generate_quick_baseline",
        lambda path: calls.append(Path(path)),
    )

    data_generator.create_baseline_if_missing(data_dir)

    assert calls == [data_dir]


def test_create_baseline_if_missing_skips_generation_when_metadata_is_fresh(tmp_path, patcher):
    data_dir = tmp_path / "processed"
    _write_baseline_artifacts(
        data_dir,
        team_freshness="BASELINE_PRESEASON",
        track_freshness="BASELINE_PRESEASON",
    )
    calls: list[Path] = []
    patcher.setattr(
        data_generator,
        "generate_quick_baseline",
        lambda path: calls.append(Path(path)),
    )

    data_generator.create_baseline_if_missing(data_dir)

    assert calls == []


def test_generate_quick_baseline_runs_all_generation_steps_in_order(tmp_path, patcher):
    calls: list[str] = []
    patcher.setattr(
        data_generator,
        "generate_neutral_team_characteristics",
        lambda _data_dir: calls.append("teams"),
    )
    patcher.setattr(
        data_generator,
        "generate_default_track_characteristics",
        lambda _data_dir: calls.append("tracks"),
    )
    patcher.setattr(
        data_generator,
        "create_driver_characteristics_if_missing",
        lambda _data_dir: calls.append("drivers"),
    )
    patcher.setattr(data_generator, "reset_learning_state", lambda: calls.append("learning"))

    data_generator.generate_quick_baseline(tmp_path / "processed")

    assert calls == ["teams", "tracks", "drivers", "learning"]


def test_generate_neutral_team_characteristics_writes_expected_payload(tmp_path):
    data_dir = tmp_path / "processed"
    data_generator.generate_neutral_team_characteristics(data_dir)

    payload = json.loads(
        (data_dir / "car_characteristics" / "2026_car_characteristics.json").read_text()
    )

    assert payload["year"] == 2026
    assert payload["data_freshness"] == "BASELINE_PRESEASON"
    assert len(payload["teams"]) == 11
    assert payload["teams"]["McLaren"]["overall_performance"] == 0.85
    assert payload["teams"]["Cadillac F1"]["overall_performance"] == 0.35


def test_generate_default_track_characteristics_writes_calendar_baseline(tmp_path):
    data_dir = tmp_path / "processed"
    data_generator.generate_default_track_characteristics(data_dir)

    payload = json.loads(
        (data_dir / "track_characteristics" / "2026_track_characteristics.json").read_text()
    )

    assert payload["year"] == 2026
    assert payload["data_freshness"] == "BASELINE_PRESEASON"
    assert "Bahrain Grand Prix" in payload["tracks"]
    assert payload["tracks"]["Monaco Grand Prix"]["overtaking_difficulty"] == 0.95
    assert payload["tracks"]["Chinese Grand Prix"]["has_sprint"] is True


def test_create_driver_characteristics_if_missing_adds_metadata_when_missing(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    driver_file = data_dir / "driver_characteristics.json"
    driver_file.write_text(json.dumps({"drivers": {"NOR": {}}}))

    data_generator.create_driver_characteristics_if_missing(data_dir)

    payload = json.loads(driver_file.read_text())
    assert payload["carried_over_from"] == 2025
    assert payload["note"].startswith("Driver characteristics carried over")
    assert "last_updated" in payload


def test_create_driver_characteristics_if_missing_keeps_existing_metadata(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    driver_file = data_dir / "driver_characteristics.json"
    original = {
        "drivers": {"NOR": {}},
        "data_freshness": "CARRIED_FORWARD",
        "note": "keep me",
    }
    driver_file.write_text(json.dumps(original))

    data_generator.create_driver_characteristics_if_missing(data_dir)

    payload = json.loads(driver_file.read_text())
    assert payload["data_freshness"] == "CARRIED_FORWARD"
    assert payload["note"] == "keep me"


def test_reset_learning_state_creates_default_payload(tmp_path, patcher):
    patcher.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    data_generator.reset_learning_state()

    payload = json.loads((tmp_path / "data" / "learning_state.json").read_text())
    assert payload["season"] == 2026
    assert payload["races_completed"] == 0
    assert payload["recommended_method"] == "blend"


def test_reset_learning_state_preserves_existing_2026_progress(tmp_path, patcher):
    patcher.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    learning_file = data_dir / "learning_state.json"
    existing = {
        "season": 2026,
        "races_completed": 5,
        "recommended_method": "custom",
    }
    learning_file.write_text(json.dumps(existing))

    data_generator.reset_learning_state()

    assert json.loads(learning_file.read_text()) == existing
