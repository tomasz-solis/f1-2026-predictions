"""Focused tests for retrospective checkpoint reconstruction helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.dashboard.accuracy import AccuracyPipeline
from src.utils.car_snapshot_history import build_car_characteristics_snapshot_payload
from src.utils.checkpoint_reconstruction import (
    SnapshotOverlayArtifactStore,
    build_reconstructed_prediction_results,
    build_snapshot_overlay_car_characteristics,
    load_checkpoint_snapshot_payload,
    reconstruct_checkpoint_prediction,
)
from src.utils.model_version import get_model_version


def test_build_snapshot_overlay_car_characteristics_preserves_priors_and_replaces_profiles():
    """Snapshot overlays should keep season priors while replacing practice-profile fields."""
    base_payload = {
        "year": 2026,
        "teams": {
            "McLaren": {
                "overall_performance": 0.81,
                "testing_characteristics": {"overall_pace": 0.42},
                "testing_characteristics_profiles": {
                    "balanced": {"overall_pace": 0.42},
                },
            }
        },
    }
    snapshot_payload = {
        "event_name": "Australian Grand Prix",
        "session_name": "FP1",
        "source": "snapshot_history_backfill",
        "captured_at": "2026-03-14T23:32:08+00:00",
        "session_started_at": "2026-03-06T01:30:00+00:00",
        "teams": {
            "McLaren": {
                "profiles": {
                    "balanced": {"overall_pace": 0.91},
                    "short_run": {"overall_pace": 0.93},
                },
                "driver_deltas_seconds": {
                    "short_run": {"NOR": -0.12, "PIA": 0.12},
                },
            }
        },
    }

    merged = build_snapshot_overlay_car_characteristics(
        base_car_payload=base_payload,
        snapshot_payload=snapshot_payload,
    )

    assert merged["teams"]["McLaren"]["overall_performance"] == 0.81
    assert merged["teams"]["McLaren"]["testing_characteristics"]["overall_pace"] == 0.91
    assert (
        merged["teams"]["McLaren"]["testing_characteristics_profiles"]["short_run"]["overall_pace"]
        == 0.93
    )
    assert merged["teams"]["McLaren"]["checkpoint_driver_deltas_seconds"]["short_run"]["NOR"] == (
        pytest.approx(-0.12)
    )
    assert merged["checkpoint_snapshot"]["session_name"] == "FP1"


def test_build_snapshot_overlay_car_characteristics_rejects_snapshots_without_team_profiles():
    """Checkpoint overlays should return empty when a snapshot has no usable team profiles."""
    base_payload = {
        "year": 2026,
        "teams": {
            "McLaren": {
                "overall_performance": 0.81,
                "testing_characteristics": {"overall_pace": 0.42},
            }
        },
    }
    snapshot_payload = {
        "event_name": "Australian Grand Prix",
        "session_name": "FP2",
        "teams": {
            "McLaren": {
                "driver_deltas_seconds": {
                    "short_run": {"NOR": -0.12, "PIA": 0.12},
                }
            }
        },
    }

    with pytest.raises(ValueError, match="valid team profiles"):
        build_snapshot_overlay_car_characteristics(
            base_car_payload=base_payload,
            snapshot_payload=snapshot_payload,
        )


def test_build_car_characteristics_snapshot_payload_ignores_delta_only_teams():
    """Snapshot payloads should only persist teams that have real profile data."""
    payload = build_car_characteristics_snapshot_payload(
        year=2026,
        event_name="Australian Grand Prix",
        session_name="FP2",
        team_profiles={
            "McLaren": {
                "balanced": {"overall_pace": 0.91},
            }
        },
        team_driver_deltas_seconds={
            "McLaren": {"short_run": {"NOR": -0.12, "PIA": 0.12}},
            "Ferrari": {"short_run": {"LEC": -0.03, "HAM": 0.03}},
        },
        source="snapshot_history_backfill",
    )

    assert set(payload["teams"]) == {"McLaren"}
    assert payload["teams"]["McLaren"]["testing_characteristics"]["overall_pace"] == pytest.approx(
        0.91
    )
    assert payload["teams"]["McLaren"]["testing_characteristics_profiles"]["balanced"][
        "overall_pace"
    ] == pytest.approx(0.91)
    assert payload["teams"]["McLaren"]["driver_deltas_seconds"]["short_run"]["NOR"] == (
        pytest.approx(-0.12)
    )


def test_accuracy_pipeline_uses_information_cutoff_for_reconstructed_predictions(patcher):
    """Retrospective checkpoints should be judged by their clean information cutoff."""

    class _Logger:
        def reconcile_completed_prediction_actuals(self, year: int) -> int:
            del year
            return 0

        def get_all_predictions(self, year: int):
            del year
            return [
                {
                    "metadata": {
                        "year": 2026,
                        "race_name": "Australian Grand Prix",
                        "session_name": "FP1",
                        "weekend_format": "normal",
                        "predicted_at": "2026-03-17T10:00:00+00:00",
                        "information_cutoff_at": "2026-03-06T04:59:59+00:00",
                    },
                    "qualifying": {"predicted_grid": []},
                    "race": {"predicted_results": []},
                    "targets": {
                        "main_qualifying": {
                            "target_session": "Q",
                            "predicted_order": [
                                {"position": 1, "driver": "NOR", "team": "McLaren"}
                            ],
                            "eligible_at_save": True,
                        }
                    },
                    "actuals": {
                        "qualifying": None,
                        "race": None,
                        "targets": {
                            "main_qualifying": [{"position": 1, "driver": "NOR", "team": "McLaren"}]
                        },
                    },
                }
            ]

    class _Metrics:
        def calculate_prediction_target_metrics(self, prediction_data, *, is_sprint):
            del prediction_data, is_sprint
            return {"main_qualifying": {"overall_mae": 1.0, "top_3_pct": 100.0}}

    class _Store:
        def __init__(self, data_root: str = "data"):
            del data_root

        def list_artifacts(self, artifact_type: str, key_prefix=None, limit: int = 100):
            del artifact_type, key_prefix, limit
            return []

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)
    patcher.setattr("src.persistence.artifact_store.ArtifactStore", _Store)

    summary = AccuracyPipeline(year=2026).build_summary()

    assert summary.n_excluded_targets == 0
    assert summary.targets["main_qualifying"].aggregate["overall_mae"]["mean"] == 1.0


def test_build_reconstructed_prediction_results_caps_sprint_main_race_confidence(patcher):
    """Sprint checkpoint reconstruction should cap Sunday race confidence for predicted grids."""

    class _Predictor:
        def predict_qualifying(self, **kwargs):
            qualifying_stage = kwargs["qualifying_stage"]
            return {
                "grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "grid_source": "PREDICTED",
                "qualifying_stage": qualifying_stage,
                "data_confidence_score": 0.9,
                "data_source": f"{qualifying_stage} checkpoint profile blend",
            }

        def predict_sprint_race(self, **kwargs):
            del kwargs
            return {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

        def predict_race(self, **kwargs):
            return {
                "finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "captured_input_confidence": kwargs["input_confidence"],
            }

    patcher.setattr(
        "src.utils.checkpoint_reconstruction.Baseline2026Predictor",
        lambda *args, **kwargs: _Predictor(),
    )
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.is_sprint_weekend",
        lambda year, race_name: True,
    )

    artifact_store = MagicMock(spec=SnapshotOverlayArtifactStore)
    artifact_store.data_root = "data"
    artifact_store.load_artifact.side_effect = [
        {
            "year": 2026,
            "teams": {"McLaren": {"testing_characteristics_profiles": {}}},
        },
        {
            "event_name": "Chinese Grand Prix",
            "session_name": "SQ",
            "captured_at": "2026-03-15T03:00:00+00:00",
            "session_started_at": "2026-03-15T02:30:00+00:00",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {"overall_pace": 0.9},
                        "short_run": {"overall_pace": 0.92},
                    }
                }
            },
        },
    ]

    results, reconstructed_is_sprint = build_reconstructed_prediction_results(
        year=2026,
        race_name="Chinese Grand Prix",
        checkpoint_session="SQ",
        weather="dry",
        artifact_store=artifact_store,
    )

    assert reconstructed_is_sprint is True
    assert results["main_race"]["input_confidence"] == pytest.approx(0.5)


def test_build_reconstructed_prediction_results_penalizes_checkpoint_profile_blend(patcher):
    """Normal-weekend reconstruction should keep the checkpoint-blend confidence penalty."""

    class _Predictor:
        def predict_qualifying(self, **kwargs):
            qualifying_stage = kwargs["qualifying_stage"]
            return {
                "grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "grid_source": "PREDICTED",
                "qualifying_stage": qualifying_stage,
                "data_confidence_score": 0.9,
                "data_source": (
                    "FP2 checkpoint profile blend "
                    "(latest stored snapshot: Australian Grand Prix / FP2)"
                ),
                "testing_fallback_used": False,
            }

        def predict_race(self, **kwargs):
            return {
                "finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "captured_input_confidence": kwargs["input_confidence"],
            }

    patcher.setattr(
        "src.utils.checkpoint_reconstruction.Baseline2026Predictor",
        lambda *args, **kwargs: _Predictor(),
    )
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.is_sprint_weekend",
        lambda year, race_name: False,
    )

    artifact_store = MagicMock(spec=SnapshotOverlayArtifactStore)
    artifact_store.data_root = "data"
    artifact_store.load_artifact.side_effect = [
        {
            "year": 2026,
            "teams": {"McLaren": {"testing_characteristics_profiles": {}}},
        },
        {
            "event_name": "Australian Grand Prix",
            "session_name": "FP2",
            "captured_at": "2026-03-14T23:32:08+00:00",
            "session_started_at": "2026-03-14T22:30:00+00:00",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {"overall_pace": 0.9},
                        "short_run": {"overall_pace": 0.92},
                    }
                }
            },
        },
    ]

    results, reconstructed_is_sprint = build_reconstructed_prediction_results(
        year=2026,
        race_name="Australian Grand Prix",
        checkpoint_session="FP2",
        weather="dry",
        artifact_store=artifact_store,
    )

    assert reconstructed_is_sprint is False
    assert results["race"]["input_confidence"] == pytest.approx(0.85)


def test_load_checkpoint_snapshot_payload_falls_back_to_latest_prior_snapshot_for_pre(patcher):
    """PRE reconstruction should use the newest snapshot before the weekend begins."""
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.is_sprint_weekend",
        lambda year, race_name: False,
    )
    patcher.setattr(
        "src.utils.accuracy_targets._scheduled_session_start",
        lambda *, year, race_name, session_name: (
            datetime.fromisoformat("2026-03-06T05:00:00+00:00")
            if (year, race_name, session_name) == (2026, "Australian Grand Prix", "FP1")
            else None
        ),
    )

    store = MagicMock()
    store.load_artifact.return_value = None
    store.list_artifacts.return_value = [
        {
            "artifact_key": "2026::Testing 2::Testing 2 Day 2",
            "data": {
                "event_name": "Testing 2",
                "session_name": "Testing 2 Day 2",
                "captured_at": "2026-02-28T18:00:00+00:00",
                "session_started_at": "2026-02-28T08:00:00+00:00",
                "teams": {"McLaren": {"profiles": {"balanced": {"overall_pace": 0.7}}}},
            },
        },
        {
            "artifact_key": "2026::Testing 2::Testing 2 Day 3",
            "data": {
                "event_name": "Testing 2",
                "session_name": "Testing 2 Day 3",
                "captured_at": "2026-03-01T18:00:00+00:00",
                "session_started_at": "2026-03-01T08:00:00+00:00",
                "teams": {"McLaren": {"profiles": {"balanced": {"overall_pace": 0.8}}}},
            },
        },
        {
            "artifact_key": "2026::Australian Grand Prix::FP1",
            "data": {
                "event_name": "Australian Grand Prix",
                "session_name": "FP1",
                "captured_at": "2026-03-06T06:00:00+00:00",
                "session_started_at": "2026-03-06T05:00:00+00:00",
                "teams": {"McLaren": {"profiles": {"balanced": {"overall_pace": 0.9}}}},
            },
        },
    ]

    payload = load_checkpoint_snapshot_payload(
        store=store,
        year=2026,
        race_name="Australian Grand Prix",
        checkpoint_session="PRE",
    )

    assert payload["event_name"] == "Testing 2"
    assert payload["session_name"] == "Testing 2 Day 3"


def test_load_checkpoint_snapshot_payload_skips_prior_sprint_only_snapshot_for_pre(patcher):
    """PRE reconstruction should keep sprint-only snapshots out of full-weekend profiles."""
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.is_sprint_weekend",
        lambda year, race_name: True,
    )
    patcher.setattr(
        "src.utils.accuracy_targets._scheduled_session_start",
        lambda *, year, race_name, session_name: (
            datetime.fromisoformat("2026-05-22T16:30:00+00:00")
            if (year, race_name, session_name) == (2026, "Canadian Grand Prix", "FP1")
            else None
        ),
    )

    store = MagicMock()
    store.load_artifact.return_value = None
    store.list_artifacts.return_value = [
        {
            "data": {
                "event_name": "Miami Grand Prix",
                "session_name": "R",
                "captured_at": "2026-05-03T22:00:00+00:00",
                "session_started_at": "2026-05-03T20:00:00+00:00",
                "teams": {"Ferrari": {"profiles": {"balanced": {"overall_pace": 0.4}}}},
            },
        },
        {
            "data": {
                "event_name": "Miami Grand Prix",
                "session_name": "Sprint",
                "captured_at": "2026-05-03T23:00:00+00:00",
                "session_started_at": "2026-05-03T21:00:00+00:00",
                "teams": {"Ferrari": {"profiles": {"balanced": {"overall_pace": 0.9}}}},
            },
        },
    ]

    payload = load_checkpoint_snapshot_payload(
        store=store,
        year=2026,
        race_name="Canadian Grand Prix",
        checkpoint_session="PRE",
        is_sprint=True,
    )

    assert payload["event_name"] == "Miami Grand Prix"
    assert payload["session_name"] == "R"


def test_load_checkpoint_snapshot_payload_uses_prior_round_when_pre_deadline_missing(patcher):
    """PRE reconstruction should fall back to the latest earlier round when times are absent."""
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.is_sprint_weekend",
        lambda year, race_name: False,
    )
    patcher.setattr(
        "src.utils.accuracy_targets._scheduled_session_start",
        lambda *, year, race_name, session_name: None,
    )
    patcher.setattr(
        "src.utils.weekend.get_schedule_rows",
        lambda year: (
            ("Australian Grand Prix", "conventional"),
            ("Chinese Grand Prix", "sprint_qualifying"),
            ("Japanese Grand Prix", "conventional"),
            ("Miami Grand Prix", "sprint_qualifying"),
            ("Canadian Grand Prix", "sprint_qualifying"),
            ("Monaco Grand Prix", "conventional"),
            ("Barcelona Grand Prix", "conventional"),
        ),
    )

    store = MagicMock()
    store.load_artifact.return_value = None
    store.list_artifacts.return_value = [
        {
            "data": {
                "event_name": "Miami Grand Prix",
                "session_name": "R",
                "round_number": 4,
                "captured_at": "2026-05-03T22:00:00+00:00",
                "session_started_at": "2026-05-03T20:00:00+00:00",
                "teams": {"Ferrari": {"profiles": {"balanced": {"overall_pace": 0.4}}}},
            },
        },
        {
            "data": {
                "event_name": "Canadian Grand Prix",
                "session_name": "R",
                "round_number": 5,
                "captured_at": "2026-05-24T22:00:00+00:00",
                "session_started_at": "2026-05-24T20:00:00+00:00",
                "teams": {"Mercedes": {"profiles": {"balanced": {"overall_pace": 0.8}}}},
            },
        },
        {
            "data": {
                "event_name": "Canadian Grand Prix",
                "session_name": "Sprint",
                "round_number": 5,
                "captured_at": "2026-05-24T23:00:00+00:00",
                "session_started_at": "2026-05-24T21:00:00+00:00",
                "teams": {"Mercedes": {"profiles": {"balanced": {"overall_pace": 0.99}}}},
            },
        },
        {
            "data": {
                "event_name": "Barcelona Grand Prix",
                "session_name": "FP1",
                "round_number": 7,
                "captured_at": "2026-06-05T13:30:00+00:00",
                "session_started_at": "2026-06-05T12:30:00+00:00",
                "teams": {"McLaren": {"profiles": {"balanced": {"overall_pace": 0.9}}}},
            },
        },
    ]

    payload = load_checkpoint_snapshot_payload(
        store=store,
        year=2026,
        race_name="Monaco Grand Prix",
        checkpoint_session="PRE",
    )

    assert payload["event_name"] == "Canadian Grand Prix"
    assert payload["session_name"] == "R"


def test_reconstruct_checkpoint_prediction_allows_missing_actuals_for_upcoming_pre(
    patcher,
    tmp_path,
):
    """Upcoming PRE checkpoints should save cleanly even when no actuals exist yet."""

    class _ArtifactStore:
        def __init__(self, data_root: str | Path = "data"):
            self.data_root = Path(data_root)
            self.saved_artifacts: list[dict[str, object]] = []

        def save_artifact(self, **kwargs):
            if kwargs["artifact_type"] == "accuracy_snapshot":
                raise AssertionError("Unscored checkpoints should not write accuracy snapshots")
            self.saved_artifacts.append(kwargs)
            return kwargs

    class _PredictionLogger:
        instances: list["_PredictionLogger"] = []

        def __init__(self, predictions_dir: str = "data/predictions"):
            self.predictions_dir = Path(predictions_dir)
            self.artifact_store = _ArtifactStore(data_root=self.predictions_dir.parent)
            self.update_actuals_called = False
            _PredictionLogger.instances.append(self)

        def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
            del year, race_name, session_name
            return False

        def get_all_predictions(self, year: int):
            del year
            return []

        def save_prediction(self, **kwargs) -> Path:
            del kwargs
            return self.predictions_dir / "2026" / "miami_grand_prix" / "miami_grand_prix_pre.json"

        def update_actuals(self, **kwargs) -> bool:
            del kwargs
            self.update_actuals_called = True
            return True

        def load_prediction(self, year: int, race_name: str, session_name: str):
            del year, race_name, session_name
            return {
                "metadata": {
                    "year": 2026,
                    "race_name": "Miami Grand Prix",
                    "session_name": "PRE",
                    "predicted_at": "2026-05-01T12:00:00+00:00",
                    "weather": "dry",
                    "weekend_format": "normal",
                },
                "qualifying": {
                    "predicted_grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]
                },
                "race": {
                    "predicted_results": [{"position": 1, "driver": "NOR", "team": "McLaren"}]
                },
                "targets": {
                    "main_qualifying": {
                        "target_session": "Q",
                        "predicted_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                        "eligible_at_save": True,
                    },
                    "grand_prix_race": {
                        "target_session": "R",
                        "predicted_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                        "eligible_at_save": True,
                    },
                },
                "actuals": {
                    "qualifying": None,
                    "race": None,
                    "targets": {
                        "main_qualifying": None,
                        "grand_prix_race": None,
                    },
                },
            }

    patcher.setattr(
        "src.utils.checkpoint_reconstruction.ArtifactStore",
        _ArtifactStore,
    )
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.PredictionLogger",
        _PredictionLogger,
    )
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.build_reconstructed_prediction_results",
        lambda **kwargs: (
            {
                "qualifying": {
                    "grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                    "grid_source": "PREDICTED",
                    "result_mode": "PREDICTED",
                    "fp_blend_info": {},
                },
                "race": {
                    "finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                    "grid_source": "PREDICTED",
                    "result_mode": "PREDICTED",
                },
            },
            False,
        ),
    )
    patcher.setattr(
        "src.dashboard.prediction_checkpointing.prediction_targets_for_checkpoint",
        lambda **kwargs: {
            "main_qualifying": {
                "target_session": "Q",
                "predicted_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "eligible_at_save": True,
            },
            "grand_prix_race": {
                "target_session": "R",
                "predicted_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "eligible_at_save": True,
            },
        },
    )

    def _raise_missing_actuals(**kwargs):
        del kwargs
        raise FileNotFoundError("no saved actuals")

    def _raise_snapshot_build(**kwargs):
        del kwargs
        raise AssertionError("Missing-actuals checkpoints should not build accuracy snapshots")

    patcher.setattr(
        "src.utils.checkpoint_reconstruction.collect_saved_target_actuals",
        _raise_missing_actuals,
    )
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.compute_information_cutoff_at",
        lambda **kwargs: "2026-05-01T13:29:59+00:00",
    )
    patcher.setattr(
        "src.utils.checkpoint_reconstruction.build_accuracy_snapshot_records",
        _raise_snapshot_build,
    )

    summary = reconstruct_checkpoint_prediction(
        year=2026,
        race_name="Miami Grand Prix",
        checkpoint_session="PRE",
        data_root=tmp_path,
        overwrite=True,
    )

    assert summary.actuals_source == "unavailable"
    assert summary.snapshot_records_written == 0
    assert _PredictionLogger.instances[0].update_actuals_called is False
    assert len(_PredictionLogger.instances[0].artifact_store.saved_artifacts) == 1
    saved_artifact = _PredictionLogger.instances[0].artifact_store.saved_artifacts[0]
    assert saved_artifact["artifact_type"] == "prediction_checkpoint"
    assert saved_artifact["artifact_key"] == "2026::Miami Grand Prix::PRE"
    assert saved_artifact["version"] == 1
    assert isinstance(saved_artifact["data"], dict)
    assert saved_artifact["data"]["metadata"]["model_version"] == get_model_version()
