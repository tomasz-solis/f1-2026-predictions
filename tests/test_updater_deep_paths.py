from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.systems import updater


def _timedelta_seconds(value: float) -> pd.Timedelta:
    return pd.to_timedelta(value, unit="s")


def _write_characteristics_file(path: Path) -> None:
    payload = {
        "year": 2026,
        "version": 1,
        "races_completed": 0,
        "teams": {
            "Ferrari": {
                "overall_performance": 0.8,
                "directionality": {
                    "max_speed": 0.0,
                    "slow_corner_speed": 0.0,
                    "medium_corner_speed": 0.0,
                    "high_corner_speed": 0.0,
                },
                "uncertainty": 0.3,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_load_race_session_enriches_results_and_uses_cache(patcher, tmp_path):
    patcher.chdir(tmp_path)

    results = pd.DataFrame({"Abbreviation": ["LEC"], "Position": [1]})
    session = MagicMock()
    session.results = results

    enable_cache = MagicMock()
    patcher.setattr(updater.fastf1.Cache, "enable_cache", enable_cache)
    patcher.setattr(
        updater.fastf1,
        "get_session",
        lambda year, race_name, session_name: session,
    )

    loaded_results, loaded_session = updater.load_race_session(2026, "Bahrain Grand Prix")

    assert loaded_session is session
    assert list(loaded_results["race_name"].unique()) == ["Bahrain Grand Prix"]
    assert list(loaded_results["year"].unique()) == [2026]
    enable_cache.assert_called_once_with("data/raw/.fastf1_cache")
    session.load.assert_called_once_with(laps=True, telemetry=False, weather=False)


def test_extract_team_performance_missing_laps_or_team_column():
    session_no_laps = SimpleNamespace(laps=pd.DataFrame())
    assert updater.extract_team_performance_from_telemetry(session_no_laps, ["Ferrari"]) == {}

    session_no_team_column = SimpleNamespace(
        laps=pd.DataFrame(
            {
                "LapTime": [_timedelta_seconds(90), _timedelta_seconds(91)],
                "LapNumber": [2, 3],
            }
        )
    )
    assert (
        updater.extract_team_performance_from_telemetry(session_no_team_column, ["Ferrari"]) == {}
    )


def test_extract_team_performance_handles_unloaded_lap_property():
    class _SessionWithBrokenLaps:
        @property
        def laps(self):
            raise RuntimeError("laps not loaded")

    assert (
        updater.extract_team_performance_from_telemetry(_SessionWithBrokenLaps(), ["Ferrari"]) == {}
    )


def test_extract_team_performance_equal_pace_and_missing_team(patcher):
    rows = []
    for team in ("Ferrari", "McLaren"):
        for lap in range(2, 9):
            rows.append(
                {
                    "Team": team,
                    "LapTime": _timedelta_seconds(90),
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "LapNumber": lap,
                }
            )

    session = SimpleNamespace(laps=pd.DataFrame(rows))
    patcher.setattr(updater, "map_team_to_characteristics", lambda raw, known_teams: str(raw))

    result = updater.extract_team_performance_from_telemetry(
        session,
        ["Ferrari", "McLaren", "Red Bull"],
    )

    assert result["Ferrari"] == 0.5
    assert result["McLaren"] == 0.5
    assert "Red Bull" not in result


def test_extract_team_performance_skips_insufficient_valid_laps(patcher):
    laps = pd.DataFrame(
        {
            "Team": ["Ferrari"] * 4,
            "LapTime": [
                _timedelta_seconds(90),
                _timedelta_seconds(91),
                _timedelta_seconds(92),
                _timedelta_seconds(93),
            ],
            "PitOutTime": [pd.NaT] * 4,
            "PitInTime": [pd.NaT] * 4,
            "LapNumber": [2, 3, 4, 5],
        }
    )
    session = SimpleNamespace(laps=laps)
    patcher.setattr(updater, "map_team_to_characteristics", lambda raw, known_teams: str(raw))

    result = updater.extract_team_performance_from_telemetry(session, ["Ferrari"])
    assert result == {}


def test_update_team_characteristics_position_fallback_and_file_save(patcher, tmp_path):
    characteristics_file = (
        tmp_path / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    )
    _write_characteristics_file(characteristics_file)

    class FailingStore:
        def __init__(self, data_root):
            self.data_root = data_root

        def load_artifact(self, artifact_type, artifact_key):
            return None

        def save_artifact(self, artifact_type, artifact_key, data, version):
            raise RuntimeError("db save failed")

    patcher.setattr(updater, "ArtifactStore", FailingStore)
    patcher.setattr(
        updater, "extract_team_performance_from_telemetry", lambda session, team_names: {}
    )
    patcher.setattr(updater, "map_team_to_characteristics", lambda raw, known_teams: str(raw))

    race_results = pd.DataFrame({"TeamName": ["Ferrari"], "Position": [2]})
    session = SimpleNamespace(event=None, name="Race Session", laps=pd.DataFrame())

    updater.update_team_characteristics(race_results, session, characteristics_file)

    saved = json.loads(characteristics_file.read_text())
    ferrari = saved["teams"]["Ferrari"]
    assert ferrari["current_season_performance"]
    assert ferrari["races_completed"] == 1
    assert saved["version"] == 2
    assert saved["data_freshness"] == "LIVE_UPDATED"
    assert Path(str(characteristics_file) + ".backup").exists()


def test_update_team_characteristics_uses_full_position_fallback_for_partial_telemetry(
    patcher, tmp_path
):
    characteristics_file = (
        tmp_path / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    )
    payload = {
        "year": 2026,
        "version": 1,
        "races_completed": 0,
        "teams": {
            "Ferrari": {
                "overall_performance": 0.8,
                "directionality": {
                    "max_speed": 0.0,
                    "slow_corner_speed": 0.0,
                    "medium_corner_speed": 0.0,
                    "high_corner_speed": 0.0,
                },
                "uncertainty": 0.3,
            },
            "McLaren": {
                "overall_performance": 0.82,
                "directionality": {
                    "max_speed": 0.0,
                    "slow_corner_speed": 0.0,
                    "medium_corner_speed": 0.0,
                    "high_corner_speed": 0.0,
                },
                "uncertainty": 0.3,
            },
        },
    }
    characteristics_file.parent.mkdir(parents=True, exist_ok=True)
    characteristics_file.write_text(json.dumps(payload))

    class Store:
        def __init__(self, data_root):
            self.data_root = data_root

        def load_artifact(self, artifact_type, artifact_key):
            return None

        def save_artifact(self, artifact_type, artifact_key, data, version):
            raise RuntimeError("db save failed")

    patcher.setattr(updater, "ArtifactStore", Store)
    patcher.setattr(
        updater,
        "extract_team_performance_from_telemetry",
        lambda session, team_names: {"Ferrari": 1.0},
    )
    patcher.setattr(updater, "map_team_to_characteristics", lambda raw, known_teams: str(raw))

    race_results = pd.DataFrame(
        {
            "TeamName": ["Ferrari", "McLaren"],
            "Position": [1, 2],
        }
    )
    session = SimpleNamespace(event=None, name="Race Session", laps=pd.DataFrame())

    updater.update_team_characteristics(race_results, session, characteristics_file)

    saved = json.loads(characteristics_file.read_text())
    ferrari = saved["teams"]["Ferrari"]
    mclaren = saved["teams"]["McLaren"]
    assert ferrari["current_season_performance"] == [1.0]
    assert mclaren["current_season_performance"] == [0.0]
    assert saved["races_completed"] == 1


def test_update_team_characteristics_handles_compound_extraction_failure(patcher, tmp_path):
    characteristics_file = (
        tmp_path / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    )
    _write_characteristics_file(characteristics_file)

    payload = json.loads(characteristics_file.read_text())

    class Store:
        def __init__(self, data_root):
            self.saved = False

        def load_artifact(self, artifact_type, artifact_key):
            return payload

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved = True

    store = Store(data_root=tmp_path)
    patcher.setattr(updater, "ArtifactStore", lambda data_root: store)
    patcher.setattr(
        updater,
        "extract_team_performance_from_telemetry",
        lambda session, team_names: {"Ferrari": 0.8},
    )
    patcher.setattr(updater, "map_team_to_characteristics", lambda raw, known_teams: "Ferrari")
    patcher.setattr(
        updater,
        "extract_compound_metrics",
        lambda team_laps, canonical_team, race_name: (_ for _ in ()).throw(
            RuntimeError("bad compound")
        ),
    )

    session = SimpleNamespace(
        event={"EventName": "Bahrain Grand Prix"},
        laps=pd.DataFrame(
            {
                "Team": ["Ferrari"],
                "LapTime": [_timedelta_seconds(90)],
            }
        ),
    )

    updater.update_team_characteristics(
        pd.DataFrame({"TeamName": ["Ferrari"], "Position": [1]}), session, characteristics_file
    )

    assert store.saved is True


def test_update_bayesian_driver_ratings_skips_when_no_valid_positions(patcher):
    race_results = pd.DataFrame({"Abbreviation": ["LEC"], "Position": [pd.NA]})

    bayesian_cls = MagicMock()
    patcher.setattr(updater, "BayesianDriverRanking", bayesian_cls)
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: {})

    updater.update_bayesian_driver_ratings(race_results)

    bayesian_cls.return_value.update.assert_not_called()


def test_update_bayesian_driver_ratings_prefers_teammate_relative_updates(patcher, tmp_path):
    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "HAM"],
            "Position": [1, 4],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2026, 2026],
        }
    )

    bayesian_cls = MagicMock()
    patcher.setattr(updater, "BayesianDriverRanking", bayesian_cls)
    patcher.setattr(
        "src.models.priors_factory.PriorsFactory.create_priors",
        lambda self: {"LEC": object(), "HAM": object()},
    )
    patcher.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda config_path="data/current_lineups.json": {"Ferrari": ["LEC", "HAM"]},
    )

    class _Store:
        def __init__(self, data_root):
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {"racecraft": {"skill_score": 0.55}},
                        "HAM": {"racecraft": {"skill_score": 0.50}},
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved.append((artifact_type, artifact_key, data, version))

    patcher.setattr(updater, "ArtifactStore", _Store)
    patcher.setattr(updater.config_loader, "get", lambda key, default=None: default)
    bayesian_cls.return_value.ratings = {"LEC": (17.0, 1.8), "HAM": (15.0, 1.9)}

    updater.update_bayesian_driver_ratings(race_results)

    bayesian_cls.return_value.update_teammate_relative.assert_called_once()
    assert bayesian_cls.return_value.update_teammate_relative.call_args.kwargs["confidence"] == (
        pytest.approx(0.35)
    )
    bayesian_cls.return_value.update.assert_not_called()


def test_update_bayesian_driver_ratings_persists_driver_characteristics_updates(patcher, tmp_path):
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "NOR"],
            "Position": [1, 2],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2026, 2026],
        }
    )

    priors = {
        "LEC": DriverPrior(
            driver_number="16",
            driver_code="LEC",
            team="Ferrari",
            team_tier="top",
            mu=16.0,
            sigma=2.0,
        ),
        "NOR": DriverPrior(
            driver_number="4",
            driver_code="NOR",
            team="McLaren",
            team_tier="top",
            mu=15.0,
            sigma=2.1,
        ),
    }

    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)

    class _Store:
        def __init__(self, data_root):
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {
                            "racecraft": {"skill_score": 0.55, "overtaking_skill": 0.58},
                            "pace": {"quali_pace": 0.62, "race_pace": 0.60},
                            "dnf_risk": {"dnf_rate": 0.08},
                        },
                        "NOR": {
                            "racecraft": {"skill_score": 0.60, "overtaking_skill": 0.61},
                            "pace": {"quali_pace": 0.66, "race_pace": 0.64},
                            "dnf_risk": {"dnf_rate": 0.07},
                        },
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved.append((artifact_type, artifact_key, data, version))

    store = _Store("data")
    patcher.setattr(updater, "ArtifactStore", lambda data_root: store)
    patcher.setattr(updater.config_loader, "get", lambda key, default=None: default)

    updater.update_bayesian_driver_ratings(race_results)

    assert len(store.saved) == 1
    artifact_type, artifact_key, payload, version = store.saved[0]
    assert artifact_type == "driver_characteristics"
    assert artifact_key == "2026::driver_characteristics"
    assert version == payload["version"]
    assert payload["version"] >= 2
    assert payload["drivers"]["LEC"]["racecraft"]["skill_score"] != 0.55
    assert "bayesian" in payload["drivers"]["LEC"]


def test_update_bayesian_driver_ratings_reuses_saved_posteriors_and_cleans_stale_fields(
    patcher, tmp_path
):
    """Persisted Bayesian state should carry forward, and deprecated fields should be stripped."""
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "HAM"],
            "Position": [1, 4],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2026, 2026],
        }
    )

    priors = {
        "LEC": DriverPrior(
            driver_number="16",
            driver_code="LEC",
            team="Ferrari",
            team_tier="top",
            mu=5.0,
            sigma=3.0,
        ),
        "HAM": DriverPrior(
            driver_number="44",
            driver_code="HAM",
            team="Ferrari",
            team_tier="top",
            mu=4.0,
            sigma=3.1,
        ),
    }
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)
    patcher.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda config_path="data/current_lineups.json": {"Ferrari": ["LEC", "HAM"]},
    )

    seen_initial_ratings: dict[str, tuple[float, float]] = {}

    class _Bayesian:
        def __init__(self, seeded_priors, grid_size=22):
            self.ratings = {
                driver_code: (prior.mu, prior.sigma) for driver_code, prior in seeded_priors.items()
            }

        def update_teammate_relative(self, observations, session_name, lineups, confidence=1.0):
            seen_initial_ratings.update(self.ratings)
            self.ratings["LEC"] = (18.4, 1.05)
            self.ratings["HAM"] = (14.2, 1.25)

        def update(self, observations, session_name, confidence=1.0):
            pytest.fail("Expected teammate-relative Bayesian updates for paired teammates")

    class _Store:
        def __init__(self, data_root):
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {
                            "racecraft": {"skill_score": 0.55},
                            "pace": {"quali_pace": 0.62},
                            "bayesian": {
                                "rating_mu": 17.8,
                                "rating_sigma": 1.1,
                                "normalized_skill_score": 0.81,
                            },
                        },
                        "HAM": {
                            "racecraft": {"skill_score": 0.52},
                            "pace": {"quali_pace": 0.60},
                            "bayesian": {
                                "rating_mu": 13.5,
                                "rating_sigma": 1.4,
                                "normalized_skill_score": 0.60,
                            },
                        },
                        "RIC": {
                            "bayesian": {
                                "rating_mu": 12.0,
                                "rating_sigma": 2.5,
                                "normalized_skill_score": 0.52,
                            }
                        },
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved.append((artifact_type, artifact_key, data, version))

    store = _Store("data")
    patcher.setattr(updater, "BayesianDriverRanking", _Bayesian)
    patcher.setattr(updater, "ArtifactStore", lambda data_root: store)
    patcher.setattr(updater.config_loader, "get", lambda key, default=None: default)

    updater.update_bayesian_driver_ratings(race_results)

    assert seen_initial_ratings["LEC"] == (17.8, 1.1)
    assert seen_initial_ratings["HAM"] == (13.5, 1.4)
    assert len(store.saved) == 1
    payload = store.saved[0][2]
    assert "normalized_skill_score" not in payload["drivers"]["LEC"]["bayesian"]
    assert "normalized_skill_score" not in payload["drivers"]["HAM"]["bayesian"]
    assert "normalized_skill_score" not in payload["drivers"]["RIC"]["bayesian"]


def test_update_bayesian_driver_ratings_also_writes_year_scoped_fallback_on_store_success(
    patcher, tmp_path
):
    """A successful store write should still refresh the file fallback used locally."""
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "NOR"],
            "Position": [1, 2],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2027, 2027],
        }
    )

    priors = {
        "LEC": DriverPrior(
            driver_number="16",
            driver_code="LEC",
            team="Ferrari",
            team_tier="top",
            mu=16.0,
            sigma=2.0,
        ),
        "NOR": DriverPrior(
            driver_number="4",
            driver_code="NOR",
            team="McLaren",
            team_tier="top",
            mu=15.0,
            sigma=2.1,
        ),
    }
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)
    patcher.setattr(updater.config_loader, "get", lambda key, default=None: default)

    class _Store:
        def __init__(self, data_root):
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {"racecraft": {"skill_score": 0.55}, "pace": {"quali_pace": 0.60}},
                        "NOR": {"racecraft": {"skill_score": 0.60}, "pace": {"quali_pace": 0.58}},
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved.append((artifact_type, artifact_key, data, version))

    patcher.setattr(updater, "ArtifactStore", _Store)

    updater.update_bayesian_driver_ratings(race_results)

    fallback_file = Path("data/processed/driver_characteristics/2027_driver_characteristics.json")
    assert fallback_file.exists()
    persisted = json.loads(fallback_file.read_text())
    assert "bayesian" in persisted["drivers"]["LEC"]


def test_update_bayesian_driver_ratings_refreshes_quali_pace_from_qualifying_results(
    patcher, tmp_path
):
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "NOR"],
            "Position": [1, 2],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2026, 2026],
        }
    )
    qualifying_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "NOR"],
            "Position": [2, 8],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2026, 2026],
        }
    )

    priors = {
        "LEC": DriverPrior(
            driver_number="16",
            driver_code="LEC",
            team="Ferrari",
            team_tier="top",
            mu=16.0,
            sigma=2.0,
        ),
        "NOR": DriverPrior(
            driver_number="4",
            driver_code="NOR",
            team="McLaren",
            team_tier="top",
            mu=15.0,
            sigma=2.1,
        ),
    }

    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)

    class _Store:
        def __init__(self, data_root):
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {
                            "racecraft": {"skill_score": 0.55, "overtaking_skill": 0.58},
                            "pace": {"quali_pace": 0.20, "race_pace": 0.60},
                            "dnf_risk": {"dnf_rate": 0.08},
                        },
                        "NOR": {
                            "racecraft": {"skill_score": 0.60, "overtaking_skill": 0.61},
                            "pace": {"quali_pace": 0.66, "race_pace": 0.64},
                            "dnf_risk": {"dnf_rate": 0.07},
                        },
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved.append((artifact_type, artifact_key, data, version))

    store = _Store("data")
    patcher.setattr(updater, "ArtifactStore", lambda data_root: store)

    def _config_get(key, default=None):
        if key == "grid.size":
            return 22
        if key == "baseline_predictor.driver_form.quali_pace_update_blend":
            return 0.5
        return default

    patcher.setattr(updater.config_loader, "get", _config_get)

    updater.update_bayesian_driver_ratings(race_results, qualifying_results=qualifying_results)

    payload = store.saved[0][2]
    assert payload["drivers"]["LEC"]["pace"]["quali_pace"] == pytest.approx(0.576, abs=1e-3)


def test_load_driver_characteristics_payload_prefers_year_scoped_fallback(tmp_path, patcher):
    patcher.chdir(tmp_path)

    legacy_path = Path("data/processed/driver_characteristics.json")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps({"source": "legacy"}))

    year_path = Path("data/processed/driver_characteristics/2027_driver_characteristics.json")
    year_path.parent.mkdir(parents=True, exist_ok=True)
    year_path.write_text(json.dumps({"source": "year_scoped"}))

    class _Store:
        def load_artifact(self, artifact_type, artifact_key):
            return None

    payload = updater._load_driver_characteristics_payload(_Store(), year=2027)

    assert payload == {"source": "year_scoped"}


def test_update_bayesian_driver_ratings_writes_year_scoped_fallback_on_store_save_failure(
    patcher, tmp_path
):
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "NOR"],
            "Position": [1, 2],
            "race_name": ["Bahrain Grand Prix", "Bahrain Grand Prix"],
            "year": [2027, 2027],
        }
    )

    priors = {
        "LEC": DriverPrior(
            driver_number="16",
            driver_code="LEC",
            team="Ferrari",
            team_tier="top",
            mu=16.0,
            sigma=2.0,
        ),
        "NOR": DriverPrior(
            driver_number="4",
            driver_code="NOR",
            team="McLaren",
            team_tier="top",
            mu=15.0,
            sigma=2.1,
        ),
    }
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)
    patcher.setattr(updater.config_loader, "get", lambda key, default=None: default)

    class _Store:
        def __init__(self, data_root):
            self.data_root = data_root

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {"racecraft": {"skill_score": 0.55}},
                        "NOR": {"racecraft": {"skill_score": 0.60}},
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            raise RuntimeError("store unavailable")

    patcher.setattr(updater, "ArtifactStore", _Store)

    updater.update_bayesian_driver_ratings(race_results)

    year_fallback = Path("data/processed/driver_characteristics/2027_driver_characteristics.json")
    legacy_fallback = Path("data/processed/driver_characteristics.json")
    assert year_fallback.exists()
    assert not legacy_fallback.exists()


def test_update_from_race_skips_team_update_when_characteristics_missing(patcher, tmp_path):
    data_dir = tmp_path / "processed"
    (data_dir / "car_characteristics").mkdir(parents=True)

    race_results = pd.DataFrame({"Abbreviation": ["LEC"], "Position": [1]})
    qualifying_results = pd.DataFrame({"Abbreviation": ["LEC"], "Position": [2]})
    session = SimpleNamespace(name="Race")

    patcher.setattr(updater, "load_race_session", lambda year, race_name: (race_results, session))
    patcher.setattr(
        updater,
        "load_qualifying_session",
        lambda year, race_name: (qualifying_results, session),
    )
    team_update = MagicMock()
    bayesian_update = MagicMock()
    patcher.setattr(updater, "update_team_characteristics", team_update)
    patcher.setattr(updater, "update_bayesian_driver_ratings", bayesian_update)

    updater.update_from_race(2026, "Bahrain Grand Prix", str(data_dir))

    team_update.assert_not_called()
    bayesian_update.assert_called_once_with(race_results, qualifying_results=qualifying_results)


def test_update_from_race_reraises_load_errors(patcher):
    patcher.setattr(
        updater,
        "load_race_session",
        lambda year, race_name: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        updater.update_from_race(2026, "Bahrain Grand Prix")
