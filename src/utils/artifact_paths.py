"""Safe path helpers for locally persisted artifacts."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_SAFE_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_race_name(value: object) -> str:
    """Normalize display race names while preserving human-readable metadata."""
    return " ".join(str(value).split()).strip()


def normalize_session_name(value: object) -> str:
    """Normalize checkpoint session identifiers for metadata and artifact keys."""
    return str(value).strip().upper()


def _reject_unsafe_raw_token(raw: str, *, field_name: str) -> None:
    if not raw:
        raise ValueError(f"{field_name} cannot be empty")
    if raw in {".", ".."}:
        raise ValueError(f"{field_name} cannot be a path segment")
    if _CONTROL_RE.search(raw):
        raise ValueError(f"{field_name} contains control characters")
    if any(separator in raw for separator in ("/", "\\")):
        raise ValueError(f"{field_name} cannot contain path separators")
    if ":" in raw:
        raise ValueError(f"{field_name} cannot contain ':'")


def safe_slug(value: object, *, field_name: str, uppercase_input: bool = False) -> str:
    """Return a lowercase filesystem slug or raise for unsafe artifact identity input."""
    raw_input = str(value)
    if _CONTROL_RE.search(raw_input):
        raise ValueError(f"{field_name} contains control characters")
    raw = normalize_session_name(raw_input) if uppercase_input else normalize_race_name(raw_input)
    _reject_unsafe_raw_token(raw, field_name=field_name)
    ascii_raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    slug = ascii_raw.lower().replace("'", "").replace(" ", "_")
    if not slug or slug in {".", ".."}:
        raise ValueError(f"{field_name} cannot be empty after slugging")
    if not _SAFE_SLUG_RE.fullmatch(slug):
        raise ValueError(f"{field_name} slug must match [a-z0-9_]+")
    return slug


def safe_race_slug(race_name: object) -> str:
    """Return the canonical prediction race directory slug."""
    return safe_slug(race_name, field_name="race_name")


def safe_session_slug(session_name: object) -> str:
    """Return the canonical prediction session filename slug."""
    return safe_slug(session_name, field_name="session_name", uppercase_input=True)


def safe_artifact_key_parts(artifact_key: str) -> list[str]:
    """Split a generic artifact key into safe filesystem path components."""
    parts = str(artifact_key).split("::")
    if not parts or any(part == "" for part in parts):
        raise ValueError("artifact_key cannot contain empty path components")
    return [safe_slug(part, field_name="artifact_key") for part in parts]


def ensure_path_under_root(root: Path, path: Path) -> Path:
    """Resolve a candidate artifact path and assert it stays under root."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Resolved artifact path escapes data root: {path}") from exc
    return resolved_path
