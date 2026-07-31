"""Tests for immutable challenger research sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.challenger_governance import stable_json_sha256
from src.persistence.research_sidecar import ResearchSidecarStore
from tests.challenger_test_helpers import strict_manifest


def _manifest() -> dict:
    """Return a minimal valid research manifest."""
    return strict_manifest(candidate_id="q1_practice")


def test_sidecar_is_immutable_idempotent_and_manifest_linked(tmp_path: Path) -> None:
    """Identical retries are safe, while changed payloads cannot overwrite a run."""
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    manifest = _manifest()

    manifest_path = store.write_manifest(manifest)
    assert store.write_manifest(manifest) == manifest_path
    evidence_path = store.write_artifact(
        manifest=manifest,
        artifact_kind="qualifying_practice_evidence",
        payload={"drivers": {"VER": {"clean_laps": 4}}},
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["manifest_sha256"] == manifest["manifest_sha256"]
    assert evidence["artifact_kind"] == "qualifying_practice_evidence"
    assert len(evidence["artifact_sha256"]) == 64
    with pytest.raises(FileExistsError, match="immutable research artifact"):
        store.write_artifact(
            manifest=manifest,
            artifact_kind="qualifying_practice_evidence",
            payload={"drivers": {"VER": {"clean_laps": 5}}},
        )


def test_sidecar_rejects_live_repo_data_roots(tmp_path: Path) -> None:
    """The research writer cannot point at processed or prediction state."""
    repo = tmp_path / "repo"
    (repo / "data" / "processed").mkdir(parents=True)

    with pytest.raises(ValueError, match="challenger_research"):
        ResearchSidecarStore(repo / "data" / "processed", repo_root=repo)


def test_sidecar_rejects_runtime_active_or_tampered_manifest(tmp_path: Path) -> None:
    """Only intact champion-default manifests may start a research run."""
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    manifest = _manifest()
    manifest["runtime_activation_allowed"] = True

    with pytest.raises(ValueError, match="manifest_sha256"):
        store.write_manifest(manifest)

    manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="cannot allow runtime activation"):
        store.write_manifest(manifest)


def test_sidecar_rejects_digest_valid_but_incomplete_manifest(tmp_path: Path) -> None:
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    manifest = _manifest()
    manifest["provenance"]["input_snapshot_ids"] = []
    manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )

    with pytest.raises(ValueError, match="input_snapshot_ids"):
        store.write_manifest(manifest)
