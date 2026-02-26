from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.predictors.baseline.data_mixin as data_mixin_module
from src.predictors.baseline.data_mixin import BaselineDataMixin


class DummyPredictor(BaselineDataMixin):
    def __init__(self, data_dir: Path, artifact_store=None):
        self.data_dir = Path(data_dir)
        self.artifact_store = artifact_store
        super().__init__()


class StubStore:
    def __init__(self, payloads=None, storage_mode: str = "file_only"):
        self.payloads = payloads or {}
        self.storage_mode = storage_mode
        self.saved = []

    def load_artifact(self, artifact_type: str, artifact_key: str):
        return self.payloads.get((artifact_type, artifact_key))

    def save_artifact(self, artifact_type: str, artifact_key: str, data):
        self.saved.append((artifact_type, artifact_key, data))


@pytest.fixture
def sample_payloads() -> tuple[dict, dict, dict]:
    car = {
        "year": 2026,
        "version": 2,
        "races_completed": 3,
        "data_freshness": "LIVE_UPDATED",
        "teams": {
            "McLaren": {
                "overall_performance": 0.8,
                "directionality": {
                    "max_speed": 0.10,
                    "slow_corner_speed": 0.00,
                    "medium_corner_speed": -0.02,
                    "high_corner_speed": 0.03,
                },
                "current_season_performance": [0.70, 0.72],
                "testing_characteristics": {"run_profile": "balanced", "overall_pace": 0.62},
                "compound_characteristics": {},
            }
        },
    }
    drivers = {
        "drivers": {
            "NOR": {
                "racecraft": {"skill_score": 0.70},
                "pace": {"quali_pace": 0.71, "race_pace": 0.70},
                "dnf_risk": {"dnf_rate": 0.10},
            }
        }
    }
    tracks = {
        "tracks": {
            "Bahrain Grand Prix": {
                "straights_pct": 30,
                "slow_corners_pct": 25,
                "medium_corners_pct": 25,
                "high_corners_pct": 20,
            }
        }
    }
    return car, drivers, tracks


def _write_baseline_files(
    base_dir: Path,
    car: dict,
    drivers: dict,
    tracks: dict | None,
    *,
    year: int = 2026,
    write_legacy_driver_file: bool = True,
) -> None:
    (base_dir / "car_characteristics").mkdir(parents=True, exist_ok=True)
    (base_dir / "track_characteristics").mkdir(parents=True, exist_ok=True)

    (base_dir / "car_characteristics" / f"{year}_car_characteristics.json").write_text(
        json.dumps(car)
    )
    if write_legacy_driver_file:
        (base_dir / "driver_characteristics.json").write_text(json.dumps(drivers))
    else:
        (base_dir / "driver_characteristics").mkdir(parents=True, exist_ok=True)
        (base_dir / "driver_characteristics" / f"{year}_driver_characteristics.json").write_text(
            json.dumps(drivers)
        )

    if tracks is not None:
        (base_dir / "track_characteristics" / f"{year}_track_characteristics.json").write_text(
            json.dumps(tracks)
        )


def test_load_data_falls_back_to_files(tmp_path, patcher, sample_payloads):
    car, drivers, tracks = sample_payloads
    data_dir = tmp_path / "processed"
    _write_baseline_files(data_dir, car, drivers, tracks)

    predictor = DummyPredictor(data_dir=data_dir, artifact_store=StubStore(payloads={}))

    patcher.setattr(data_mixin_module, "validate_team_characteristics", lambda payload: None)
    patcher.setattr(data_mixin_module, "validate_driver_characteristics", lambda payload: None)
    patcher.setattr(
        "src.utils.driver_validation.validate_driver_data",
        lambda drivers_payload: ["sample warning"],
    )

    predictor.load_data()

    assert "McLaren" in predictor.teams
    assert "NOR" in predictor.drivers
    assert "Bahrain Grand Prix" in predictor.tracks
    assert predictor.races_completed == 3
    assert predictor.year == 2026


def test_load_data_missing_track_file_sets_empty_tracks(tmp_path, patcher, sample_payloads):
    car, drivers, _tracks = sample_payloads
    data_dir = tmp_path / "processed"
    _write_baseline_files(data_dir, car, drivers, tracks=None)

    predictor = DummyPredictor(data_dir=data_dir, artifact_store=StubStore(payloads={}))
    patcher.setattr(data_mixin_module, "validate_team_characteristics", lambda payload: None)
    patcher.setattr(data_mixin_module, "validate_driver_characteristics", lambda payload: None)
    patcher.setattr("src.utils.driver_validation.validate_driver_data", lambda payload: [])

    predictor.load_data()

    assert predictor.tracks == {}


def test_load_data_uses_season_scoped_driver_fallback_file(tmp_path, patcher, sample_payloads):
    car, drivers, tracks = sample_payloads
    car_2027 = {**car, "year": 2027}
    data_dir = tmp_path / "processed"
    _write_baseline_files(
        data_dir,
        car_2027,
        drivers,
        tracks,
        year=2027,
        write_legacy_driver_file=False,
    )

    predictor = DummyPredictor(data_dir=data_dir, artifact_store=StubStore(payloads={}))
    predictor.season_year = 2027
    patcher.setattr(data_mixin_module, "validate_team_characteristics", lambda payload: None)
    patcher.setattr(data_mixin_module, "validate_driver_characteristics", lambda payload: None)
    patcher.setattr("src.utils.driver_validation.validate_driver_data", lambda payload: [])

    predictor.load_data()

    assert "NOR" in predictor.drivers
    assert predictor.year == 2027


def test_load_data_raises_for_invalid_team_schema(tmp_path, patcher, sample_payloads):
    car, drivers, tracks = sample_payloads
    data_dir = tmp_path / "processed"
    _write_baseline_files(data_dir, car, drivers, tracks)

    store = StubStore(
        payloads={
            ("car_characteristics", "2026::car_characteristics"): car,
            ("driver_characteristics", "2026::driver_characteristics"): drivers,
            ("track_characteristics", "2026::track_characteristics"): tracks,
        }
    )
    predictor = DummyPredictor(data_dir=data_dir, artifact_store=store)

    patcher.setattr(
        data_mixin_module,
        "validate_team_characteristics",
        lambda payload: (_ for _ in ()).throw(ValueError("bad team payload")),
    )

    with pytest.raises(ValueError, match="bad team payload"):
        predictor.load_data()


def test_calculate_track_suitability_variants(tmp_path):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {
        "McLaren": {
            "directionality": {
                "max_speed": 0.10,
                "slow_corner_speed": 0.00,
                "medium_corner_speed": -0.10,
                "high_corner_speed": 0.05,
            }
        },
        "NoData": {},
    }
    predictor.tracks = {
        "Bahrain Grand Prix": {
            "straights_pct": 40,
            "slow_corners_pct": 20,
            "medium_corners_pct": 20,
            "high_corners_pct": 20,
        },
        "Unknown": {
            "straights_pct": 0,
            "slow_corners_pct": 0,
            "medium_corners_pct": 0,
            "high_corners_pct": 0,
        },
    }

    assert predictor.calculate_track_suitability("NoData", "Bahrain Grand Prix") == 0.0
    assert predictor.calculate_track_suitability("McLaren", "Missing") == 0.0
    assert predictor.calculate_track_suitability("McLaren", "Unknown") == 0.0
    assert predictor.calculate_track_suitability("McLaren", "Bahrain Grand Prix") == pytest.approx(
        0.03
    )


def test_get_blended_team_strength_uses_current_fallback(tmp_path, patcher):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {"McLaren": {"overall_performance": 0.82, "current_season_performance": []}}
    predictor.races_completed = 4

    patcher.setattr(predictor, "calculate_track_suitability", lambda team, race_name: 0.02)

    captured = {}

    def _fake_blend(**kwargs):
        captured.update(kwargs)
        return 0.77

    patcher.setattr(
        data_mixin_module, "get_recommended_schedule", lambda is_regulation_change: {"ok": True}
    )
    patcher.setattr(data_mixin_module, "calculate_blended_performance", _fake_blend)

    result = predictor.get_blended_team_strength("McLaren", "Bahrain Grand Prix")

    assert result == 0.77
    assert captured["baseline_score"] == 0.82
    assert captured["current_score"] == 0.82
    assert captured["race_number"] == 5


@pytest.mark.parametrize(
    ("stress", "expected"),
    [
        ({"traction": 4.1, "braking": 4.0, "lateral": 4.0, "asphalt_abrasion": 4.0}, "HARD"),
        ({"traction": 2.0, "braking": 2.1, "lateral": 2.2, "asphalt_abrasion": 2.3}, "SOFT"),
        ({"traction": 3.0, "braking": 3.0, "lateral": 3.0, "asphalt_abrasion": 3.0}, "MEDIUM"),
    ],
)
def test_select_race_compound_thresholds(tmp_path, patcher, stress, expected):
    predictor = DummyPredictor(data_dir=tmp_path)
    patcher.chdir(tmp_path)

    (tmp_path / "data").mkdir()
    payload = {"bahrain_grand_prix": {"tyre_stress": stress}}
    (tmp_path / "data" / "2026_pirelli_info.json").write_text(json.dumps(payload))

    def _get_config(key: str, default):
        if key.endswith("high_stress_threshold"):
            return 3.5
        if key.endswith("low_stress_threshold"):
            return 2.5
        return default

    patcher.setattr(data_mixin_module.config_loader, "get", _get_config)

    assert predictor._select_race_compound("Bahrain Grand Prix") == expected


def test_select_race_compound_defaults_for_missing_or_invalid_file(tmp_path, patcher):
    predictor = DummyPredictor(data_dir=tmp_path)
    patcher.chdir(tmp_path)
    assert predictor._select_race_compound("Bahrain Grand Prix") == "MEDIUM"

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "2026_pirelli_info.json").write_text("{bad json")
    assert predictor._select_race_compound("Bahrain Grand Prix") == "MEDIUM"


def test_get_compound_adjusted_team_strength_branches(tmp_path, patcher):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {"McLaren": {"compound_characteristics": {"SOFT": {"laps_sampled": 20}}}}
    patcher.setattr(predictor, "get_blended_team_strength", lambda team, race_name: 0.9)

    patcher.setattr(
        data_mixin_module,
        "should_use_compound_adjustments",
        lambda payload, min_laps_threshold: False,
    )
    assert predictor.get_compound_adjusted_team_strength("McLaren", "Bahrain", "SOFT") == 0.9

    patcher.setattr(
        data_mixin_module,
        "should_use_compound_adjustments",
        lambda payload, min_laps_threshold: True,
    )
    patcher.setattr(
        data_mixin_module, "get_compound_performance_modifier", lambda payload, compound: 0.3
    )
    assert predictor.get_compound_adjusted_team_strength("McLaren", "Bahrain", "SOFT") == 1.0


def test_testing_characteristics_profile_fallbacks(tmp_path):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {
        "McLaren": {
            "testing_characteristics_profiles": {"short_run": {"overall_pace": 0.8}},
            "testing_characteristics": {"run_profile": "long_run", "overall_pace": 0.6},
        },
        "Ferrari": {"testing_characteristics": {"overall_pace": 0.55}},
        "RB": {"testing_characteristics": "invalid"},
    }

    assert predictor._get_testing_characteristics_for_profile("McLaren", "short_run") == {
        "overall_pace": 0.8
    }
    assert predictor._get_testing_characteristics_for_profile("McLaren", "long_run") == {
        "run_profile": "long_run",
        "overall_pace": 0.6,
    }
    assert predictor._get_testing_characteristics_for_profile("Ferrari", "balanced") == {
        "overall_pace": 0.55
    }
    assert predictor._get_testing_characteristics_for_profile("RB", "balanced") == {}


def test_compute_testing_profile_modifier_branches(tmp_path, patcher):
    predictor = DummyPredictor(data_dir=tmp_path)

    patcher.setattr(
        predictor,
        "_get_testing_characteristics_for_profile",
        lambda team, profile: {"overall_pace": 0.9, "consistency": 0.4},
    )
    modifier, has_data = predictor._compute_testing_profile_modifier(
        "McLaren",
        "balanced",
        metric_weights={"overall_pace": 2.0, "consistency": 1.0, "missing": 10.0},
        scale=0.5,
    )

    assert has_data is True
    assert modifier == pytest.approx(0.04)

    patcher.setattr(predictor, "_get_testing_characteristics_for_profile", lambda team, profile: {})
    modifier, has_data = predictor._compute_testing_profile_modifier(
        "McLaren",
        "balanced",
        metric_weights={"overall_pace": 1.0},
        scale=1.0,
    )
    assert modifier == 0.0
    assert has_data is False


def test_update_compound_characteristics_uses_cache(tmp_path):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {"McLaren": {"compound_characteristics": {}}}

    session_laps = pd.DataFrame({"Team": ["McLaren", "McLaren"], "LapTime": [1, 2]})
    cache_key = ("Bahrain Grand Prix", 2026, len(session_laps))
    predictor._compound_cache[cache_key] = {"McLaren": {"SOFT": {"laps_sampled": 8}}}

    predictor._update_compound_characteristics_from_session(
        session_laps=session_laps,
        race_name="Bahrain Grand Prix",
        year=2026,
        is_sprint=False,
    )

    assert predictor.teams["McLaren"]["compound_characteristics"]["SOFT"]["laps_sampled"] == 8


def test_update_compound_characteristics_extracts_and_persists(tmp_path, patcher):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {
        "McLaren": {
            "compound_characteristics": {},
        }
    }
    store = StubStore(
        payloads={
            (
                "car_characteristics",
                "2026::car_characteristics",
            ): {"teams": {"McLaren": {"compound_characteristics": {}}}}
        },
        storage_mode="dual_write",
    )
    predictor.artifact_store = store

    session_laps = pd.DataFrame(
        {
            "Team": ["McLaren", "McLaren"],
            "LapTime": [pd.to_timedelta("0:01:30"), pd.to_timedelta("0:01:31")],
        }
    )

    patcher.setattr(
        "src.utils.team_mapping.map_team_to_characteristics",
        lambda raw_team, known_teams: "McLaren",
    )
    patcher.setattr(
        "src.systems.compound_analyzer.extract_compound_metrics",
        lambda team_laps, canonical_team, race_name: {"SOFT": {"laps_sampled": 12}},
    )
    patcher.setattr(
        "src.systems.compound_analyzer.normalize_compound_metrics_across_teams",
        lambda metrics, race_name: metrics,
    )
    patcher.setattr(
        "src.systems.compound_analyzer.aggregate_compound_samples",
        lambda existing, new, blend_weight, race_name: new,
    )
    patcher.setattr(data_mixin_module.config_loader, "get", lambda key, default: 0.5)

    predictor._update_compound_characteristics_from_session(
        session_laps=session_laps,
        race_name="Bahrain Grand Prix",
        year=2026,
        is_sprint=True,
    )

    assert predictor.teams["McLaren"]["compound_characteristics"]["SOFT"]["laps_sampled"] == 12
    assert store.saved, "Expected compound updates to be persisted when DB storage is enabled"


def test_update_compound_characteristics_handles_empty_extraction(tmp_path, patcher):
    predictor = DummyPredictor(data_dir=tmp_path)
    predictor.teams = {"McLaren": {"compound_characteristics": {"SOFT": {"laps_sampled": 5}}}}

    session_laps = pd.DataFrame(
        {
            "Team": ["McLaren"],
            "LapTime": [pd.to_timedelta("0:01:30")],
        }
    )

    patcher.setattr(
        "src.utils.team_mapping.map_team_to_characteristics",
        lambda raw_team, known_teams: "McLaren",
    )
    patcher.setattr(
        "src.systems.compound_analyzer.extract_compound_metrics",
        lambda team_laps, canonical_team, race_name: {},
    )

    predictor._update_compound_characteristics_from_session(
        session_laps=session_laps,
        race_name="Bahrain Grand Prix",
        year=2026,
        is_sprint=False,
    )

    assert predictor.teams["McLaren"]["compound_characteristics"]["SOFT"]["laps_sampled"] == 5
