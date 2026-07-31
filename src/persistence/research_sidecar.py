"""Immutable, research-only persistence for challenger artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.analysis.challenger_governance import (
    stable_json_sha256,
    validate_challenger_manifest,
)

DEFAULT_RESEARCH_SIDECAR_ROOT = Path("data/model_diagnostics/challenger_research")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether a resolved path is inside or equal to its parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_segment(value: str, *, field_name: str) -> str:
    """Validate one caller-controlled path segment."""
    candidate = str(value).strip()
    if not _SAFE_SEGMENT.fullmatch(candidate):
        raise ValueError(f"{field_name} must be a filesystem-safe identifier")
    return candidate


class ResearchSidecarStore:
    """Store immutable challenger evidence outside all live artifact paths.

    Any root inside the repository's ``data`` directory must live below the
    dedicated challenger-research directory.  Explicit temporary/external roots
    remain supported for tests and isolated replay workers.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_RESEARCH_SIDECAR_ROOT,
        *,
        repo_root: str | Path = ".",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        configured_root = Path(root)
        self.root = (
            configured_root if configured_root.is_absolute() else self.repo_root / configured_root
        ).resolve()
        repo_data_root = (self.repo_root / "data").resolve()
        allowed_research_root = (self.repo_root / DEFAULT_RESEARCH_SIDECAR_ROOT).resolve()
        if _is_within(self.root, repo_data_root) and not _is_within(
            self.root,
            allowed_research_root,
        ):
            raise ValueError(
                "research sidecars inside repo data/ must use "
                f"{DEFAULT_RESEARCH_SIDECAR_ROOT.as_posix()}"
            )

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        """Persist one champion-default manifest without overwriting an existing run."""
        payload = dict(manifest)
        candidate_id, manifest_digest = self._validate_manifest(payload)
        run_id = manifest_digest[:16]
        return self._write_immutable(candidate_id, run_id, "manifest", payload)

    def write_artifact(
        self,
        *,
        manifest: Mapping[str, Any],
        artifact_kind: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Persist one manifest-linked research artifact in an immutable envelope."""
        candidate_id, manifest_digest = self._validate_manifest(manifest)
        normalised_kind = _safe_segment(artifact_kind, field_name="artifact_kind")
        envelope = {
            "artifact_type": "challenger_research_sidecar",
            "artifact_kind": normalised_kind,
            "candidate_id": candidate_id,
            "manifest_sha256": manifest_digest,
            "payload": dict(payload),
        }
        envelope["artifact_sha256"] = stable_json_sha256(envelope)
        return self._write_immutable(
            candidate_id,
            manifest_digest[:16],
            normalised_kind,
            envelope,
        )

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, str]:
        """Validate the complete shared manifest contract before persistence."""

        validated = validate_challenger_manifest(manifest)
        candidate_id = _safe_segment(validated.candidate_id, field_name="candidate_id")
        return candidate_id, validated.manifest_sha256

    def _write_immutable(
        self,
        candidate_id: str,
        run_id: str,
        artifact_kind: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Write stable JSON exactly once; permit only byte-identical retries."""
        target = self.root / candidate_id / run_id / f"{artifact_kind}.json"
        if not _is_within(target.resolve(), self.root):
            raise ValueError("research sidecar target escaped its configured root")
        encoded = (
            json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x", encoding="utf-8", newline="\n") as file_handle:
                file_handle.write(encoded)
        except FileExistsError as exc:
            if target.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(
                    f"immutable research artifact already exists: {target}"
                ) from exc
        return target
