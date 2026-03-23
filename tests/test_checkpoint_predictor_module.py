"""Tests for checkpoint-aware predictor overlay helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.dashboard import checkpoint_predictor


class _ArtifactStore:
    """Tiny artifact store stub for checkpoint overlay tests."""

    def __init__(self, base_payload: dict):
        """Keep one base season payload available for overlay reads."""
        self.base_payload = deepcopy(base_payload)
        self.data_root = Path("data")
        self.storage_mode = "file_only"

    def load_artifact(
        self,
        artifact_type: str,
        artifact_key: str,
        version: str | int = "latest",
        run_id: str | None = None,
    ) -> dict | None:
        """Return the stored season car payload for matching artifact requests."""
        del version, run_id
        if artifact_type == "car_characteristics" and artifact_key == "2026::car_characteristics":
            return deepcopy(self.base_payload)
        return None

    def list_artifacts(self, artifact_type: str, key_prefix: str | None = None, limit: int = 100):
        """Return an empty artifact listing for this focused test stub."""
        del artifact_type, key_prefix, limit
        return []

    def get_latest_version(self, artifact_type: str, artifact_key: str) -> int:
        """Return a stable dummy version for the wrapped base payload."""
        del artifact_type, artifact_key
        return 1


def test_build_checkpoint_overlay_predictor_uses_snapshot_profiles(patcher):
    """Overlay predictor should replace stored profiles with the selected checkpoint snapshot."""
    captured: dict[str, object] = {}
    base_payload = {
        "year": 2026,
        "teams": {
            "McLaren": {
                "overall_performance": 0.81,
                "testing_characteristics": {"overall_pace": 0.41},
                "testing_characteristics_profiles": {
                    "balanced": {"overall_pace": 0.41},
                },
            }
        },
    }
    snapshot_payload = {
        "event_name": "Australian Grand Prix",
        "session_name": "FP1",
        "teams": {
            "McLaren": {
                "profiles": {
                    "balanced": {"overall_pace": 0.92},
                    "short_run": {"overall_pace": 0.95},
                }
            }
        },
    }

    class _OverlayPredictor:
        """Capture the overlaid artifact store passed into predictor bootstrap."""

        def __init__(
            self,
            *,
            data_dir: str = "data/processed",
            seed: int = 42,
            season_year: int = 2026,
            config=None,
            artifact_store=None,
        ):
            del data_dir, seed, season_year, config
            captured["artifact_store"] = artifact_store
            self.artifact_store = artifact_store
            self.car_characteristics = artifact_store.load_artifact(
                "car_characteristics",
                "2026::car_characteristics",
            )

    patcher.setattr(
        checkpoint_predictor,
        "load_checkpoint_snapshot_payload",
        lambda **kwargs: (
            captured.__setitem__("loader_kwargs", dict(kwargs)) or deepcopy(snapshot_payload)
        ),
    )
    patcher.setattr(checkpoint_predictor, "Baseline2026Predictor", _OverlayPredictor)

    base_predictor = type(
        "_BasePredictor",
        (),
        {
            "artifact_store": _ArtifactStore(base_payload),
            "data_dir": Path("data/processed"),
            "seed": 11,
            "config": object(),
        },
    )()

    result = checkpoint_predictor.build_checkpoint_overlay_predictor(
        base_predictor=base_predictor,
        year=2026,
        race_name="Australian Grand Prix",
        checkpoint_session="FP1",
        is_sprint=False,
    )

    assert result is not base_predictor
    assert captured["loader_kwargs"]["checkpoint_session"] == "FP1"
    assert captured["loader_kwargs"]["is_sprint"] is False
    assert result.car_characteristics["teams"]["McLaren"]["overall_performance"] == 0.81
    assert (
        result.car_characteristics["teams"]["McLaren"]["testing_characteristics_profiles"][
            "short_run"
        ]["overall_pace"]
        == 0.95
    )
    assert result.car_characteristics["checkpoint_snapshot"]["session_name"] == "FP1"


def test_build_checkpoint_overlay_predictor_returns_base_predictor_when_snapshot_missing(patcher):
    """Missing stored snapshots should fall back to the already-loaded predictor."""
    base_predictor = type(
        "_BasePredictor",
        (),
        {
            "artifact_store": _ArtifactStore({"year": 2026, "teams": {}}),
            "data_dir": Path("data/processed"),
            "seed": 11,
            "config": object(),
        },
    )()

    patcher.setattr(
        checkpoint_predictor,
        "load_checkpoint_snapshot_payload",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing snapshot")),
    )

    result = checkpoint_predictor.build_checkpoint_overlay_predictor(
        base_predictor=base_predictor,
        year=2026,
        race_name="Australian Grand Prix",
        checkpoint_session="PRE",
        is_sprint=False,
    )

    assert result is base_predictor
