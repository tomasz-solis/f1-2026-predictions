"""Focused tests for retrospective checkpoint reconstruction helpers."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.dashboard.accuracy import AccuracyPipeline
from src.utils.checkpoint_reconstruction import (
    SnapshotOverlayArtifactStore,
    build_reconstructed_prediction_results,
    build_snapshot_overlay_car_characteristics,
    load_checkpoint_snapshot_payload,
)


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
                }
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
    assert merged["checkpoint_snapshot"]["session_name"] == "FP1"


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
