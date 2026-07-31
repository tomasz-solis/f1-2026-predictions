"""Strict challenger-governance fixtures shared by focused tests."""

from __future__ import annotations

from typing import Any

from src.analysis.challenger_governance import (
    DEFAULT_REPLAY_CHECKPOINTS,
    DEFAULT_REPLAY_SEEDS,
    ReplayProvenance,
    stable_json_sha256,
)
from src.models.challenger_variants import VARIANT_COMPONENTS


def strict_manifest(
    variant: str = "q1_qualifying_practice",
    *,
    candidate_id: str | None = None,
    simulation_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a complete digest-valid manifest without touching git or config files."""

    config_files = [
        {"path": "config/default.yaml", "sha256": "d" * 64},
        {"path": "config/production_config.json", "sha256": "e" * 64},
    ]
    feature_schema = {"version": "qualifying-practice-v2", "columns": ["driver"]}
    manifest: dict[str, Any] = {
        "artifact_type": "prediction_challenger_manifest",
        "schema_version": 1,
        "candidate_id": candidate_id or f"candidate_{variant}",
        "variant_id": variant,
        "created_at": "2026-07-18T11:00:00Z",
        "cutoff_at": "2026-07-18T10:00:00Z",
        "default_variant": "champion",
        "runtime_activation_allowed": False,
        "variants": {
            "champion": {"role": "champion", "default": True, "components": []},
            variant: {
                "role": "challenger",
                "default": False,
                "components": sorted(VARIANT_COMPONENTS[variant]),
            },
        },
        "provenance": {
            "git": {
                "source_sha": "a" * 40,
                "is_dirty": False,
                "dirty_diff_sha256": "b" * 64,
                "dirty_status_sha256": "c" * 64,
                "untracked_file_count": 0,
            },
            "configuration": {
                "files": config_files,
                "effective_bundle_sha256": stable_json_sha256(config_files),
            },
            "feature_schema": feature_schema,
            "feature_schema_sha256": stable_json_sha256(feature_schema),
            "input_snapshot_ids": ["2026::example::FP3"],
            "seeds": list(DEFAULT_REPLAY_SEEDS),
            "checkpoints": list(DEFAULT_REPLAY_CHECKPOINTS),
            "dry_only": True,
            "simulation_counts": simulation_counts or {"qualifying": 5000, "race": 3000},
        },
        "metadata": {},
    }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    return manifest


def strict_replay_provenance(
    *,
    event_count: int = 30,
    simulation_counts: dict[str, int] | None = None,
) -> ReplayProvenance:
    return ReplayProvenance(
        seeds=DEFAULT_REPLAY_SEEDS,
        simulation_counts=simulation_counts or {"qualifying": 5000, "race": 3000},
        dry_only=True,
        checkpoint_event_counts={
            checkpoint: event_count for checkpoint in DEFAULT_REPLAY_CHECKPOINTS
        },
        replay_sha256="f" * 64,
    )
