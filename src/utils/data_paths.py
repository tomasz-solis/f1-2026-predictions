"""Helpers for resolving project storage roots for local data-backed workflows."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_ROOT = Path("data")
_DATA_ROOT_ENV_VAR = "F1_DATA_DIR"


def get_project_data_root() -> Path:
    """Return the active project data root.

    When ``F1_DATA_DIR`` is set, local storage-backed components should read and
    write under that root. Otherwise they fall back to the repository default
    ``data/`` tree.
    """
    configured_root = os.getenv(_DATA_ROOT_ENV_VAR)
    if configured_root:
        return Path(configured_root)
    return _DEFAULT_DATA_ROOT


def resolve_data_root(candidate: str | Path = _DEFAULT_DATA_ROOT) -> Path:
    """Resolve a storage root, honoring ``F1_DATA_DIR`` for the default ``data`` root.

    Explicit non-default roots are preserved exactly so sidecar tools can still
    target arbitrary directories without being rewritten.
    """
    candidate_path = Path(candidate)
    if candidate_path == _DEFAULT_DATA_ROOT:
        return get_project_data_root()
    return candidate_path


def resolve_repo_data_path(path_like: str | Path) -> Path:
    """Rewrite repo-relative ``data/...`` paths to the active data root.

    Paths outside the repository ``data`` tree are returned unchanged.
    """
    candidate_path = Path(path_like)
    if not candidate_path.parts:
        return get_project_data_root()
    if candidate_path.parts[0] != _DEFAULT_DATA_ROOT.name:
        return candidate_path
    return get_project_data_root().joinpath(*candidate_path.parts[1:])
