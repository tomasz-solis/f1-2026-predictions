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
                "note": "2025 P4 seed, updated with 0 race(s) of 2026 data",
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
    enable_cache.assert_called_once()
    assert Path(enable_cache.call_args.args[0]).parts == ("data", "raw", ".fastf1_cache")
    session.load.assert_called_once_with(laps=True, telemetry=False, weather=True)


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
    assert ferrari["note"] == "2025 P4 seed, updated with 1 race(s) of 2026 data"
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
            self.data_root = data_root
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
            self.data_root = data_root
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
    assert payload["drivers"]["LEC"]["racecraft"]["skill_score"] == 0.55
    assert "rating_mu" in payload["drivers"]["LEC"]["bayesian"]
    assert "rating_sigma" in payload["drivers"]["LEC"]["bayesian"]


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
            self.data_root = data_root
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
            self.data_root = data_root
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
            self.data_root = data_root
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


def test_update_bayesian_driver_ratings_persists_post_qualifying_session_state(patcher, tmp_path):
    """Race+qualifying updates should both be counted in the stored posterior."""
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
            mu=12.0,
            sigma=2.0,
        ),
        "NOR": DriverPrior(
            driver_number="4",
            driver_code="NOR",
            team="McLaren",
            team_tier="top",
            mu=12.0,
            sigma=2.0,
        ),
    }
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)
    patcher.setattr("src.utils.lineups.load_current_lineups", lambda: None)

    class _Store:
        def __init__(self, data_root):
            self.data_root = data_root
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {
                            "racecraft": {"skill_score": 0.55, "overtaking_skill": 0.58},
                            "pace": {"quali_pace": 0.60, "race_pace": 0.60},
                            "dnf_risk": {"dnf_rate": 0.08},
                            "bayesian": {
                                "rating_mu": 12.0,
                                "rating_sigma": 2.0,
                                "sessions_observed": 3,
                            },
                        },
                        "NOR": {
                            "racecraft": {"skill_score": 0.60, "overtaking_skill": 0.61},
                            "pace": {"quali_pace": 0.58, "race_pace": 0.64},
                            "dnf_risk": {"dnf_rate": 0.07},
                            "bayesian": {"rating_mu": 12.0, "rating_sigma": 2.0},
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
        return default

    patcher.setattr(updater.config_loader, "get", _config_get)

    updater.update_bayesian_driver_ratings(race_results, qualifying_results=qualifying_results)

    payload = store.saved[0][2]
    assert payload["drivers"]["LEC"]["bayesian"]["sessions_observed"] == 5
    assert payload["drivers"]["NOR"]["bayesian"]["sessions_observed"] == 2
    assert payload["drivers"]["LEC"]["bayesian"]["last_session"] == "Qualifying_Bahrain Grand Prix"
    assert payload["drivers"]["LEC"]["bayesian"]["rating_mu"] > 12.0


def test_update_bayesian_driver_ratings_updates_dnf_rate_from_statuses(patcher, tmp_path):
    """Finished and retired statuses should update DNF risk without hand tuning."""
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    race_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "NOR"],
            "Position": [1, 22],
            "Status": ["Finished", "Retired"],
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
    patcher.setattr("src.utils.lineups.load_current_lineups", lambda: None)

    class _Store:
        def __init__(self, data_root):
            self.data_root = data_root
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        "LEC": {
                            "racecraft": {"skill_score": 0.55, "overtaking_skill": 0.58},
                            "pace": {"quali_pace": 0.60, "race_pace": 0.60},
                            "dnf_risk": {"dnf_rate": 0.20},
                        },
                        "NOR": {
                            "racecraft": {"skill_score": 0.60, "overtaking_skill": 0.61},
                            "pace": {"quali_pace": 0.58, "race_pace": 0.64},
                            "dnf_risk": {"dnf_rate": 0.10},
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
        if key == "baseline_predictor.driver_form.dnf_rate_update_blend":
            return 0.5
        if key == "baseline_predictor.driver_form.dnf_rate_cap":
            return 0.9
        return default

    patcher.setattr(updater.config_loader, "get", _config_get)

    updater.update_bayesian_driver_ratings(race_results)

    payload = store.saved[0][2]
    assert payload["drivers"]["LEC"]["dnf_risk"]["dnf_rate"] == pytest.approx(0.10)
    assert payload["drivers"]["NOR"]["dnf_risk"]["dnf_rate"] == pytest.approx(0.55)


def test_load_driver_characteristics_payload_prefers_year_scoped_fallback(tmp_path, patcher):
    patcher.chdir(tmp_path)

    legacy_path = Path("data/processed/driver_characteristics.json")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps({"source": "legacy"}))

    year_path = Path("data/processed/driver_characteristics/2027_driver_characteristics.json")
    year_path.parent.mkdir(parents=True, exist_ok=True)
    year_path.write_text(json.dumps({"source": "year_scoped"}))

    class _Store:
        data_root = Path("data")

        def load_artifact(self, artifact_type, artifact_key):
            return None

    payload = updater._load_driver_characteristics_payload(_Store(), year=2027)

    assert payload == {"source": "year_scoped"}


def test_persist_driver_characteristics_writes_under_store_root_not_repo(tmp_path, patcher):
    # Regression: the fallback file must land under the store's own data root so
    # sidecar tools (historical replay) cannot overwrite the repo's live
    # data/processed driver characteristics. chdir to tmp_path so a regression to
    # the old hardcoded "data/processed" would surface as a distinct leaked path.
    patcher.chdir(tmp_path)
    sidecar_root = tmp_path / "sidecar"

    class _Store:
        def __init__(self, data_root):
            self.data_root = data_root

        def get_latest_version(self, artifact_type, artifact_key):
            return 0

        def save_artifact(self, artifact_type, artifact_key, data, version):
            return None

    updater._persist_driver_characteristics_payload(
        _Store(sidecar_root), {"drivers": {"VER": {}}}, 2026
    )

    written = (
        sidecar_root / "processed" / "driver_characteristics" / "2026_driver_characteristics.json"
    )
    leaked = (
        tmp_path
        / "data"
        / "processed"
        / "driver_characteristics"
        / "2026_driver_characteristics.json"
    )
    assert written.exists(), "fallback must be written under the store's data root"
    assert not leaked.exists(), "fallback must not leak into the repo-relative data/processed tree"


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
    """Driver updates should inherit the replay data root when team data is absent."""
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
    bayesian_update.assert_called_once()
    call_args, call_kwargs = bayesian_update.call_args
    assert call_args[0] is race_results
    assert call_kwargs["qualifying_results"] is qualifying_results
    assert call_kwargs["data_root"] == tmp_path
    assert call_kwargs["weather"] == "dry"
    assert call_kwargs["qualifying_weather"] == "dry"
    assert call_kwargs["race_matched_lap_aggregates"].empty
    assert call_kwargs["qualifying_matched_lap_aggregates"].empty


def test_update_from_race_reraises_load_errors(patcher):
    patcher.setattr(
        updater,
        "load_race_session",
        lambda year, race_name: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        updater.update_from_race(2026, "Bahrain Grand Prix")


def test_sprint_race_updates_race_pace_ema(patcher, tmp_path):
    from src.models.bayesian import DriverPrior

    patcher.chdir(tmp_path)

    sprint_results = pd.DataFrame(
        {
            "Abbreviation": ["VER", "NOR", "LEC", "HAM"],
            "Position": [1, 2, 3, 4],
            "Status": ["Finished", "Finished", "Finished", "Finished"],
            "race_name": ["China Grand Prix"] * 4,
            "year": [2026] * 4,
        }
    )
    sprint_session = MagicMock()

    patcher.setattr(
        "src.utils.weekend.is_sprint_weekend",
        lambda year, race_name: True,
    )
    patcher.setattr(
        updater,
        "load_competitive_session",
        lambda year, race_name, session_name, load_laps=False: (sprint_results, sprint_session),
    )
    patcher.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda config_path="data/current_lineups.json": {
            "Red Bull": ["VER", "PER"],
            "McLaren": ["NOR", "PIA"],
            "Ferrari": ["LEC", "HAM"],
        },
    )

    priors = {
        "VER": DriverPrior("1", "VER", "Red Bull", "top", mu=18.0, sigma=2.0),
        "NOR": DriverPrior("4", "NOR", "McLaren", "top", mu=17.0, sigma=2.1),
        "LEC": DriverPrior("16", "LEC", "Ferrari", "top", mu=16.5, sigma=2.0),
        "HAM": DriverPrior("44", "HAM", "Ferrari", "top", mu=16.0, sigma=2.2),
    }
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)

    initial_race_pace = {"VER": 0.60, "NOR": 0.58, "LEC": 0.55, "HAM": 0.52}

    class _Store:
        def __init__(self, data_root):
            self.data_root = data_root
            self.saved = []

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return {
                    "version": 1,
                    "drivers": {
                        dc: {
                            "racecraft": {"skill_score": 0.60},
                            "pace": {"race_pace": initial_race_pace[dc]},
                        }
                        for dc in initial_race_pace
                    },
                }
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            self.saved.append((artifact_type, artifact_key, data, version))

    patcher.setattr(updater, "ArtifactStore", _Store)

    configured_blend = 0.25

    def _config_get(key, default=None):
        if key == "baseline_predictor.driver_form.race_pace_update_blend":
            return configured_blend
        return default

    patcher.setattr(updater.config_loader, "get", _config_get)

    updater.update_from_sprint_race(2026, "China Grand Prix", data_root=str(tmp_path / "data"))

    # Read the year-scoped fallback file that _persist_driver_characteristics_payload writes
    # under <data_root>/processed, mirroring production's data_root="data".
    fallback = (
        tmp_path
        / "data"
        / "processed"
        / "driver_characteristics"
        / "2026_driver_characteristics.json"
    )
    assert fallback.exists(), "Sprint update should persist driver characteristics"
    saved = json.loads(fallback.read_text())
    drivers = saved["drivers"]

    # Every driver's race_pace must have moved from its initial value.
    for dc in initial_race_pace:
        updated_pace = drivers[dc]["pace"]["race_pace"]
        assert updated_pace != initial_race_pace[dc], f"{dc} race_pace was not updated"

    # Verify blend weight is half of normal race_pace_update_blend.
    # Re-derive expected pace for VER (P1) using the same EMA formula.
    expected_blend = configured_blend * 0.5  # 0.125
    grid_size = max(22, len(priors))
    ver_raw_pace = 1.0 - ((1 - 1) / (grid_size - 1))  # P1 -> 1.0
    # VER has no teammate in the sprint observations, so team-mean correction
    # uses field mean fallback -> observed_pace is clipped raw_pace.
    ver_old = initial_race_pace["VER"]
    # Allow for teammate-relative adjustments; just check the blend magnitude.
    ver_updated = drivers["VER"]["pace"]["race_pace"]
    # The movement should be proportional to expected_blend (0.125), not
    # the full blend (0.25). Verify approximate half-weight by checking
    # that the absolute shift is closer to half-blend than full-blend.
    full_blend_shift = abs(configured_blend * (ver_raw_pace - ver_old))
    half_blend_shift = abs(expected_blend * (ver_raw_pace - ver_old))
    actual_shift = abs(ver_updated - ver_old)
    assert actual_shift == pytest.approx(half_blend_shift, abs=0.03), (
        f"Sprint pace blend should be ~half of normal: "
        f"actual_shift={actual_shift:.4f}, expected_half={half_blend_shift:.4f}, "
        f"full_would_be={full_blend_shift:.4f}"
    )


def test_sprint_update_moves_race_and_qualifying_seconds_with_half_evidence(patcher, tmp_path):
    """Sprint and SQ aggregates should update isolated seconds paths at half precision."""
    from src.models.bayesian import DriverPrior
    from src.models.driver_seconds_state import read_driver_seconds_state

    patcher.chdir(tmp_path)
    sprint_results = pd.DataFrame(
        {
            "Abbreviation": ["AAA", "BBB"],
            "Position": [1, 2],
            "Status": ["Finished", "Finished"],
            "race_name": ["China Grand Prix", "China Grand Prix"],
            "year": [2026, 2026],
        }
    )
    sprint_session = SimpleNamespace(
        weather_data=pd.DataFrame({"Rainfall": [False, False]}),
    )
    sprint_qualifying_session = SimpleNamespace(
        weather_data=pd.DataFrame({"Rainfall": [False, False]}),
    )

    patcher.setattr("src.utils.weekend.is_sprint_weekend", lambda year, race_name: True)
    patcher.setattr(
        updater,
        "load_competitive_session",
        lambda year, race_name, session_name, load_laps=False: (
            sprint_results,
            sprint_session if session_name == "Sprint" else sprint_qualifying_session,
        ),
    )
    patcher.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda config_path="data/current_lineups.json": {"Example": ["AAA", "BBB"]},
    )
    patcher.setattr(
        "src.models.priors_factory.PriorsFactory.create_priors",
        lambda self: {
            "AAA": DriverPrior("1", "AAA", "Example", "top", mu=18.0, sigma=2.0),
            "BBB": DriverPrior("2", "BBB", "Example", "top", mu=17.0, sigma=2.0),
        },
    )

    def _aggregate(session_kind: str, gap_s: float) -> pd.DataFrame:
        """Build one usable sprint aggregate row for the requested state path."""
        return pd.DataFrame(
            [
                {
                    "reference_driver_code": "AAA",
                    "comparison_driver_code": "BBB",
                    "session_kind": session_kind,
                    "matched_gap_median_s": gap_s,
                    "matched_gap_se_s": 0.10,
                    "n_matched_pairs": 4,
                    "weather_bucket": "dry",
                    "skip_reason": pd.NA,
                }
            ]
        )

    patcher.setattr(
        updater,
        "_extract_driver_seconds_aggregates",
        lambda session, session_kind, weather: _aggregate(
            session_kind,
            0.40 if session_kind == "race" else -0.30,
        ),
    )

    driver_payload = {
        "version": 1,
        "drivers": {
            code: {
                "racecraft": {"skill_score": 0.60},
                "pace": {"race_pace": 0.50},
                "bayesian": {
                    "race_rating_mu_s": 0.0,
                    "race_rating_sigma_s": 0.30,
                    "quali_rating_mu_s": 0.0,
                    "quali_rating_sigma_s": 0.30,
                },
            }
            for code in ("AAA", "BBB")
        },
    }

    class _Store:
        """Artifact store stub for a sprint driver-seconds update."""

        def __init__(self, data_root):
            self.data_root = data_root

        def load_artifact(self, artifact_type, artifact_key):
            return driver_payload if artifact_type == "driver_characteristics" else None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            """Accept the saved sprint driver artifact."""

    patcher.setattr(updater, "ArtifactStore", _Store)

    evidence_scales: list[float] = []
    original_seconds_update = updater._update_dry_driver_seconds_path

    def _record_seconds_update(**kwargs):
        """Record evidence scale before applying the real seconds update."""
        evidence_scales.append(float(kwargs["evidence_scale"]))
        return original_seconds_update(**kwargs)

    patcher.setattr(updater, "_update_dry_driver_seconds_path", _record_seconds_update)

    updater.update_from_sprint_race(2026, "China Grand Prix", data_root=str(tmp_path))

    aaa = read_driver_seconds_state(driver_payload["drivers"]["AAA"])
    bbb = read_driver_seconds_state(driver_payload["drivers"]["BBB"])
    assert aaa is not None
    assert bbb is not None
    assert aaa.race_rating_mu_s > 0.0
    assert bbb.race_rating_mu_s < 0.0
    assert aaa.quali_rating_mu_s < 0.0
    assert bbb.quali_rating_mu_s > 0.0
    assert driver_payload["drivers"]["AAA"]["bayesian"]["race_rating_observations"] == 1
    assert driver_payload["drivers"]["AAA"]["bayesian"]["quali_rating_observations"] == 1
    assert evidence_scales == [0.5, 0.5]
