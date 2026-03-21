import importlib.util
import json
from pathlib import Path


def _load_validator_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_characteristics.py"
    spec = importlib.util.spec_from_file_location("validate_characteristics_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_valid_driver_payload() -> dict:
    drivers = {
        "HAM": {
            "racecraft": {"skill_score": 0.83},
            "pace": {"quali_pace": 0.83, "race_pace": 0.82},
            "dnf_risk": {"dnf_rate": 0.05},
        },
        "VER": {
            "racecraft": {"skill_score": 0.91},
            "pace": {"quali_pace": 0.92, "race_pace": 0.90},
            "dnf_risk": {"dnf_rate": 0.02},
        },
    }
    for idx in range(16):
        skill = 0.32 + (idx * 0.02)
        drivers[f"D{idx:02d}"] = {
            "racecraft": {"skill_score": round(skill, 3)},
            "pace": {"quali_pace": round(skill, 3), "race_pace": round(skill, 3)},
            "dnf_risk": {"dnf_rate": 0.05},
        }

    return {"drivers": drivers}


def _build_valid_team_payload() -> dict:
    teams = {f"Team {idx}": {"overall_performance": 0.5, "uncertainty": 0.3} for idx in range(10)}
    return {
        "year": 2026,
        "data_freshness": "BASELINE_PRESEASON",
        "teams": teams,
    }


def _build_valid_track_payload() -> dict:
    return {
        "year": 2026,
        "tracks": {
            "Bahrain Grand Prix": {
                "pit_stop_loss": 21.5,
                "safety_car_prob": 0.45,
                "overtaking_difficulty": 0.55,
            }
        },
    }


def test_team_validation_accepts_preseason_neutral_baseline(tmp_path):
    validator = _load_validator_module()
    team_file = tmp_path / "car_characteristics" / "2026_car_characteristics.json"
    teams = {f"Team {idx}": {"overall_performance": 0.5, "uncertainty": 0.3} for idx in range(10)}
    _write_json(
        team_file,
        {
            "year": 2026,
            "data_freshness": "BASELINE_PRESEASON",
            "teams": teams,
        },
    )

    is_valid, errors, warnings = validator.validate_team_characteristics(team_file)

    assert is_valid
    assert errors == []
    assert warnings == []


def test_team_expectations_warn_by_default_and_fail_when_enforced(tmp_path):
    validator = _load_validator_module()
    team_file = tmp_path / "car_characteristics" / "2026_car_characteristics.json"
    _write_json(
        team_file,
        {
            "year": 2026,
            "data_freshness": "IN_SEASON",
            "teams": {
                "McLaren": {"overall_performance": 0.50},
                "Mercedes": {"overall_performance": 0.64},
                "Red Bull Racing": {"overall_performance": 0.66},
                "Ferrari": {"overall_performance": 0.60},
                "Williams": {"overall_performance": 0.58},
                "RB": {"overall_performance": 0.52},
                "Aston Martin": {"overall_performance": 0.48},
                "Haas F1 Team": {"overall_performance": 0.46},
                "Alpine": {"overall_performance": 0.44},
                "Sauber": {"overall_performance": 0.40},
            },
        },
    )

    is_valid, errors, warnings = validator.validate_team_characteristics(team_file)
    assert is_valid
    assert errors == []
    assert any("McLaren" in warning for warning in warnings)

    is_valid_enforced, errors_enforced, warnings_enforced = validator.validate_team_characteristics(
        team_file, enforce_expectations=True
    )
    assert not is_valid_enforced
    assert any("McLaren" in error for error in errors_enforced)
    assert warnings_enforced == []


def test_driver_expectations_warn_by_default(tmp_path):
    validator = _load_validator_module()
    driver_file = tmp_path / "driver_characteristics.json"
    drivers = {
        "HAM": {
            "racecraft": {"skill_score": 0.60},
            "pace": {"quali_pace": 0.60, "race_pace": 0.60},
            "dnf_risk": {"dnf_rate": 0.05},
        },
        "VER": {
            "racecraft": {"skill_score": 0.95},
            "pace": {"quali_pace": 0.95, "race_pace": 0.95},
            "dnf_risk": {"dnf_rate": 0.01},
        },
    }
    for idx in range(16):
        skill = 0.30 + (idx * 0.02)
        drivers[f"D{idx:02d}"] = {
            "racecraft": {"skill_score": round(skill, 3)},
            "pace": {"quali_pace": round(skill, 3), "race_pace": round(skill, 3)},
            "dnf_risk": {"dnf_rate": 0.05},
        }

    _write_json(driver_file, {"drivers": drivers})

    is_valid, errors, warnings = validator.validate_driver_characteristics(driver_file)

    assert is_valid
    assert errors == []
    assert any("HAM" in warning for warning in warnings)


def test_resolve_driver_characteristics_file_falls_back_to_legacy_layout(tmp_path):
    validator = _load_validator_module()
    legacy_file = tmp_path / "driver_characteristics.json"
    _write_json(legacy_file, {"drivers": {}})

    resolved = validator._resolve_driver_characteristics_file(tmp_path, 2026)

    assert resolved == legacy_file


def test_main_accepts_season_scoped_repo_layout(tmp_path, monkeypatch, capsys):
    validator = _load_validator_module()
    data_dir = tmp_path / "processed"
    _write_json(
        data_dir / "driver_characteristics" / "2026_driver_characteristics.json",
        _build_valid_driver_payload(),
    )
    _write_json(
        data_dir / "car_characteristics" / "2026_car_characteristics.json",
        _build_valid_team_payload(),
    )
    _write_json(
        data_dir / "track_characteristics" / "2026_track_characteristics.json",
        _build_valid_track_payload(),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["validate_characteristics.py", "--data-dir", str(data_dir)],
    )

    exit_code = validator.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "2026_driver_characteristics.json" in output


def test_track_validation_rejects_collapsed_overtaking_distribution(tmp_path):
    validator = _load_validator_module()
    track_file = tmp_path / "track_characteristics" / "2026_track_characteristics.json"
    collapsed_tracks = {
        f"Track {idx}": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": [0.0, 0.03, 0.05][idx % 3],
        }
        for idx in range(12)
    }
    _write_json(
        track_file,
        {
            "year": 2026,
            "tracks": collapsed_tracks,
        },
    )

    is_valid, errors, warnings = validator.validate_track_characteristics(track_file)

    assert not is_valid
    assert any("distribution is collapsed" in error for error in errors)
    assert warnings == []
