"""Tests for storage-path resolution helpers used by local replay workflows."""

from __future__ import annotations

from pathlib import Path

from src.utils.data_paths import get_project_data_root, resolve_data_root, resolve_repo_data_path
from src.utils.prediction_logger import PredictionLogger


def test_resolve_data_root_uses_env_override_for_default_data_root(monkeypatch):
    """The default ``data`` root should follow ``F1_DATA_DIR`` when it is set."""
    monkeypatch.setenv("F1_DATA_DIR", "data/historical_replay")

    assert get_project_data_root() == Path("data/historical_replay")
    assert resolve_data_root("data") == Path("data/historical_replay")


def test_resolve_repo_data_path_only_rewrites_repo_data_prefix(monkeypatch):
    """Only repo-relative ``data/...`` paths should move to the replay root."""
    monkeypatch.setenv("F1_DATA_DIR", "data/historical_replay")

    assert resolve_repo_data_path("data/predictions") == Path("data/historical_replay/predictions")
    assert resolve_repo_data_path("config/default.yaml") == Path("config/default.yaml")


def test_prediction_logger_default_predictions_dir_uses_env_root(monkeypatch, tmp_path):
    """PredictionLogger should follow the active replay root when using its default path."""
    replay_root = tmp_path / "historical_replay"
    monkeypatch.setenv("F1_DATA_DIR", str(replay_root))

    logger = PredictionLogger()

    assert logger.predictions_dir == replay_root / "predictions"
    assert logger.artifact_store.data_root == replay_root
