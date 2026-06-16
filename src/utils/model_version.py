"""Helpers for resolving and stamping the active model version."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.utils import config_loader

_DEFAULT_MODEL_VERSION = "2.3"


def get_model_version() -> str:
    """Return the normalized model version from config."""
    raw_value = config_loader.get("model.version", _DEFAULT_MODEL_VERSION)
    normalized = " ".join(str(raw_value).split()).strip()
    return normalized or _DEFAULT_MODEL_VERSION


def format_model_version_label(version: str | None = None) -> str:
    """Return a UI-friendly model version label."""
    resolved_version = " ".join(str(version or get_model_version()).split()).strip()
    if not resolved_version:
        resolved_version = _DEFAULT_MODEL_VERSION
    return resolved_version if resolved_version.lower().startswith("v") else f"v{resolved_version}"


def resolve_model_version(metadata: Mapping[str, Any] | None = None) -> str:
    """Resolve the model version from artifact metadata or current config."""
    if isinstance(metadata, Mapping):
        candidate = " ".join(str(metadata.get("model_version", "")).split()).strip()
        if candidate:
            return candidate
    return get_model_version()


def stamp_model_metadata(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return metadata with a stable ``model_version`` field attached."""
    payload = dict(metadata) if isinstance(metadata, Mapping) else {}
    payload["model_version"] = resolve_model_version(payload)
    return payload
