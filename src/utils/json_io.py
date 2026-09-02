"""Shared JSON reading helpers for analysis modules and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``.

    Raises:
        ValueError: If the file's top-level JSON value is not an object.
    """
    with path.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
