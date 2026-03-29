"""Focused tests for the target-aware accuracy pipeline."""

from src.dashboard.accuracy import (
    AccuracyPipeline,
    CheckpointAccuracyPoint,
    CheckpointStatusPoint,
    TargetAccuracySummary,
)
from src.dashboard.accuracy_view import (
    build_progression_checkpoint_state,
    build_progression_line_series,
    build_progression_series,
    build_target_metric_cards,
)
from src.utils.accuracy_snapshots import accuracy_snapshot_artifact_key


def test_accuracy_snapshot_artifact_key_normalizes_checkpoint_identity():
    """Snapshot keys should collapse whitespace and normalize session/target casing."""
    artifact_key = accuracy_snapshot_artifact_key(
        year=2026,
        race_name="  Chinese   Grand Prix ",
        checkpoint_session=" fp1 ",
        target_key=" Grand_Prix_Race ",
    )

    assert artifact_key == "2026::Chinese Grand Prix::FP1::grand_prix_race"


def test_accuracy_pipeline_prefers_persisted_snapshots(patcher):
    """Snapshot artifacts should win over raw recomputation when both exist."""

    class _Logger:
        def reconcile_completed_prediction_actuals(self, year: int) -> int:
            assert year == 2026
            return 0

        def get_all_predictions(self, year: int):
            assert year == 2026
            return [
                {
                    "metadata": {
                        "year": 2026,
                        "race_name": "Bahrain Grand Prix",
                        "session_name": "FP3",
                        "weekend_format": "normal",
                    },
                    "qualifying": {
                        "predicted_grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]
                    },
                    "race": {"predicted_results": []},
                    "targets": {
                        "main_qualifying": {
                            "target_session": "Q",
                            "predicted_order": [
                                {"position": 1, "driver": "VER", "team": "Red Bull"}
                            ],
                            "eligible_at_save": True,
                        }
                    },
                    "actuals": {
                        "qualifying": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
                        "race": None,
                        "targets": {
                            "main_qualifying": [
                                {"position": 1, "driver": "VER", "team": "Red Bull"}
                            ]
                        },
                    },
                }
            ]

    class _Metrics:
        def calculate_prediction_target_metrics(self, prediction_data, *, is_sprint):
            del prediction_data, is_sprint
            return {"main_qualifying": {"overall_mae": 9.0, "top_3_pct": 0.0}}

    class _Store:
        def __init__(self, data_root: str = "data"):
            del data_root

        def list_artifacts(self, artifact_type: str, key_prefix=None, limit: int = 100):
            assert artifact_type == "accuracy_snapshot"
            del key_prefix, limit
            return [
                {
                    "artifact_key": "2026::Bahrain Grand Prix::FP3::main_qualifying",
                    "data": {
                        "metrics": {
                            "overall_mae": 1.5,
                            "top_3_pct": 66.0,
                            "top_10_pct": 100.0,
                        }
                    },
                }
            ]

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)
    patcher.setattr("src.persistence.artifact_store.ArtifactStore", _Store)

    summary = AccuracyPipeline(year=2026).build_summary()

    assert summary.targets["main_qualifying"].aggregate["overall_mae"]["mean"] == 1.5
    assert summary.targets["main_qualifying"].aggregate["top_3_pct"]["mean"] == 66.0


def test_accuracy_pipeline_skips_reconciliation_on_initial_load_by_default(patcher):
    """Dashboard reads should stay lightweight unless the caller opts into refresh work."""
    reconcile_calls: list[int] = []

    class _Logger:
        def reconcile_completed_prediction_actuals(self, year: int) -> int:
            reconcile_calls.append(year)
            return 1

        def get_all_predictions(self, year: int):
            assert year == 2026
            return []

    class _Metrics:
        pass

    class _Store:
        def __init__(self, data_root: str = "data"):
            del data_root

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)
    patcher.setattr("src.persistence.artifact_store.ArtifactStore", _Store)

    pipeline = AccuracyPipeline(year=2026)
    summary = pipeline.build_summary()

    assert summary.n_predictions == 0
    assert pipeline.actuals_reconciled == 0
    assert reconcile_calls == []


def test_accuracy_pipeline_excludes_ineligible_target_but_keeps_race_target(patcher):
    """A contaminated qualifying target should not block a valid race target from scoring."""

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
                        "session_name": "Q",
                        "weekend_format": "normal",
                    },
                    "qualifying": {
                        "predicted_grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]
                    },
                    "race": {
                        "predicted_results": [{"position": 1, "driver": "VER", "team": "Red Bull"}]
                    },
                    "targets": {
                        "main_qualifying": {
                            "target_session": "Q",
                            "predicted_order": [
                                {"position": 1, "driver": "VER", "team": "Red Bull"}
                            ],
                            "eligible_at_save": False,
                        },
                        "grand_prix_race": {
                            "target_session": "R",
                            "predicted_order": [
                                {"position": 1, "driver": "VER", "team": "Red Bull"}
                            ],
                            "eligible_at_save": True,
                        },
                    },
                    "actuals": {
                        "qualifying": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
                        "race": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
                        "targets": {
                            "main_qualifying": [
                                {"position": 1, "driver": "VER", "team": "Red Bull"}
                            ],
                            "grand_prix_race": [
                                {"position": 1, "driver": "VER", "team": "Red Bull"}
                            ],
                        },
                    },
                }
            ]

    class _Metrics:
        def calculate_prediction_target_metrics(self, prediction_data, *, is_sprint):
            del is_sprint
            return {
                target_key: {
                    "overall_mae": 0.0,
                    "top_3_pct": 100.0,
                    "top_10_pct": 100.0,
                    "exact_accuracy": 100.0,
                    "within_1": 100.0,
                    "within_3": 100.0,
                    "correlation": 1.0,
                    "field_size": 1.0,
                    "top_3_hits": 1.0,
                    "top_10_hits": 1.0,
                }
                for target_key in prediction_data.get("targets", {})
            }

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

    assert "grand_prix_race" in summary.targets
    assert "main_qualifying" not in summary.targets
    assert summary.n_excluded_targets == 1


def test_accuracy_pipeline_excludes_target_saved_after_next_checkpoint_start(patcher):
    """Timestamp checks should drop targets that were saved after newer data became available."""

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
                        "race_name": "Chinese Grand Prix",
                        "session_name": "SPRINT",
                        "weekend_format": "sprint",
                        "predicted_at": "2026-03-14T11:00:08+00:00",
                    },
                    "qualifying": {"predicted_grid": []},
                    "race": {"predicted_results": []},
                    "targets": {
                        "grand_prix_race": {
                            "target_session": "R",
                            "predicted_order": [
                                {"position": 1, "driver": "RUS", "team": "Mercedes"}
                            ],
                            "eligible_at_save": True,
                        }
                    },
                    "actuals": {
                        "qualifying": None,
                        "race": None,
                        "targets": {
                            "grand_prix_race": [
                                {"position": 1, "driver": "RUS", "team": "Mercedes"}
                            ]
                        },
                    },
                }
            ]

    class _Metrics:
        def calculate_prediction_target_metrics(self, prediction_data, *, is_sprint):
            del prediction_data, is_sprint
            return {
                "grand_prix_race": {
                    "overall_mae": 0.0,
                    "top_3_pct": 100.0,
                    "top_10_pct": 100.0,
                    "exact_accuracy": 100.0,
                    "within_1": 100.0,
                    "within_3": 100.0,
                    "correlation": 1.0,
                    "field_size": 1.0,
                    "top_3_hits": 1.0,
                    "top_10_hits": 1.0,
                }
            }

    class _Store:
        def __init__(self, data_root: str = "data"):
            del data_root

        def list_artifacts(self, artifact_type: str, key_prefix=None, limit: int = 100):
            del artifact_type, key_prefix, limit
            return []

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)
    patcher.setattr("src.persistence.artifact_store.ArtifactStore", _Store)
    patcher.setattr(
        "src.utils.accuracy_targets._load_event_boundary_state",
        lambda: {
            "races": {
                "2026::Chinese Grand Prix": {
                    "session_schedule": {
                        "FP1": "2026-03-13T03:30:00+00:00",
                        "SQ": "2026-03-13T07:30:00+00:00",
                        "Sprint": "2026-03-14T03:00:00+00:00",
                        "Q": "2026-03-14T07:00:00+00:00",
                        "R": "2026-03-15T07:00:00+00:00",
                    }
                }
            }
        },
    )

    summary = AccuracyPipeline(year=2026).build_summary()

    assert "grand_prix_race" not in summary.targets
    assert summary.n_excluded_targets == 1


def test_accuracy_pipeline_reconcile_actuals_rebuilds_snapshots(patcher):
    """Manual refresh should rewrite accuracy snapshots so KPI cards stay current."""
    saved_snapshots: list[tuple[str, str]] = []

    prediction_record = {
        "metadata": {
            "year": 2026,
            "race_name": "Bahrain Grand Prix",
            "session_name": "FP3",
            "run_id": "run-123",
            "weekend_format": "normal",
        },
        "qualifying": {"predicted_grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "race": {"predicted_results": []},
        "targets": {
            "main_qualifying": {
                "target_session": "Q",
                "predicted_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
                "eligible_at_save": True,
            }
        },
        "actuals": {
            "qualifying": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "race": None,
            "targets": {"main_qualifying": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        },
    }

    class _Logger:
        def reconcile_completed_prediction_actuals(self, year: int) -> int:
            assert year == 2026
            return 1

        def get_all_predictions(self, year: int):
            assert year == 2026
            return [prediction_record]

    class _Metrics:
        def calculate_prediction_target_metrics(self, prediction_data, *, is_sprint):
            del prediction_data, is_sprint
            return {"main_qualifying": {"overall_mae": 1.0, "top_3_pct": 100.0}}

    class _Store:
        def __init__(self, data_root: str = "data"):
            del data_root

        def save_artifact(
            self,
            artifact_type: str,
            artifact_key: str,
            data,
            version: int | None = None,
            run_id: str | None = None,
        ):
            del data, version, run_id
            saved_snapshots.append((artifact_type, artifact_key))
            return {}

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)
    patcher.setattr("src.persistence.artifact_store.ArtifactStore", _Store)

    pipeline = AccuracyPipeline(year=2026)
    reconciled = pipeline.reconcile_actuals()

    assert reconciled == 1
    assert pipeline.snapshots_written == 1
    assert saved_snapshots == [
        ("accuracy_snapshot", "2026::Bahrain Grand Prix::FP3::main_qualifying")
    ]


def test_build_progression_series_keeps_missing_expected_checkpoints_visible():
    """Weekend progression should preserve the full checkpoint axis with gaps."""
    target_summary = TargetAccuracySummary(
        target_key="main_qualifying",
        label="Main Qualifying",
        checkpoint_progression=[
            CheckpointAccuracyPoint(
                target_key="main_qualifying",
                weekend_format="normal",
                checkpoint_session="FP2",
                checkpoint_index=2,
                metrics={"overall_mae": 4.73},
                race_count=1,
            ),
            CheckpointAccuracyPoint(
                target_key="main_qualifying",
                weekend_format="normal",
                checkpoint_session="FP3",
                checkpoint_index=3,
                metrics={"overall_mae": 4.82},
                race_count=1,
            ),
        ],
    )

    checkpoint_labels, metric_values, race_counts, missing_checkpoints = build_progression_series(
        target_summary=target_summary,
        metric_name="overall_mae",
        weekend_format="normal",
    )

    assert checkpoint_labels == ["PRE", "FP1", "FP2", "FP3"]
    assert metric_values == [None, None, 4.73, 4.82]
    assert race_counts == [0, 0, 1, 1]
    assert missing_checkpoints == ["PRE", "FP1"]


def test_build_progression_line_series_skips_missing_checkpoints_in_trace():
    """The chart trace should connect observed checkpoints while keeping the full axis labels."""
    labels, values, counts = build_progression_line_series(
        checkpoint_labels=["PRE", "FP1", "SQ", "SPRINT"],
        metric_values=[2.8, None, 4.2, None],
        race_counts=[1, 0, 1, 0],
    )

    assert labels == ["PRE", "SQ"]
    assert values == [2.8, 4.2]
    assert counts == [1, 1]


def test_build_progression_series_does_not_mark_excluded_checkpoint_as_missing():
    """Excluded checkpoints should be surfaced separately from truly missing saves."""
    target_summary = TargetAccuracySummary(
        target_key="main_qualifying",
        label="Main Qualifying",
        checkpoint_progression=[
            CheckpointAccuracyPoint(
                target_key="main_qualifying",
                weekend_format="normal",
                checkpoint_session="FP2",
                checkpoint_index=2,
                metrics={"overall_mae": 4.73},
                race_count=1,
            ),
        ],
        checkpoint_status=[
            CheckpointStatusPoint(
                target_key="main_qualifying",
                weekend_format="normal",
                checkpoint_session="FP3",
                checkpoint_index=3,
                excluded_count=1,
            )
        ],
    )

    checkpoint_labels, metric_values, race_counts, missing_checkpoints = build_progression_series(
        target_summary=target_summary,
        metric_name="overall_mae",
        weekend_format="normal",
    )

    assert checkpoint_labels == ["PRE", "FP1", "FP2", "FP3"]
    assert metric_values == [None, None, 4.73, None]
    assert race_counts == [0, 0, 1, 0]
    assert missing_checkpoints == ["PRE", "FP1"]


def test_build_progression_checkpoint_state_returns_excluded_checkpoints():
    """Checkpoint-state helper should classify excluded checkpoints separately."""
    target_summary = TargetAccuracySummary(
        target_key="grand_prix_race",
        label="Grand Prix Race",
        checkpoint_status=[
            CheckpointStatusPoint(
                target_key="grand_prix_race",
                weekend_format="normal",
                checkpoint_session="FP2",
                checkpoint_index=2,
                scored_count=1,
            ),
            CheckpointStatusPoint(
                target_key="grand_prix_race",
                weekend_format="normal",
                checkpoint_session="FP3",
                checkpoint_index=3,
                excluded_count=1,
            ),
        ],
    )

    result = build_progression_checkpoint_state(
        target_summary=target_summary,
        weekend_format="normal",
    )

    assert result["scored_checkpoints"] == ["FP2"]
    assert result["excluded_checkpoints"] == ["FP3"]
    assert result["pending_checkpoints"] == []


def test_build_target_metric_cards_surfaces_qualifying_summary_values():
    """Selected-target cards should expose qualifying accuracy metrics directly."""
    target_summary = TargetAccuracySummary(
        target_key="main_qualifying",
        label="Main Qualifying",
        aggregate={
            "overall_mae": {"mean": 2.41},
            "exact_accuracy": {"mean": 18.0},
            "within_1": {"mean": 44.0},
            "within_3": {"mean": 71.0},
            "correlation": {"mean": 0.82},
        },
    )

    cards = build_target_metric_cards(target_summary)

    assert [card["label"] for card in cards] == [
        "MAE",
        "Exact",
        "Within 1",
        "Within 3",
        "Correlation",
    ]
    assert [card["value"] for card in cards] == [2.41, 18.0, 44.0, 71.0, 0.82]
