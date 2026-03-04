"""Persistent storage helpers for precomputed dashboard predictions."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

from src.persistence.config import should_read_db_first, should_write_to_db, should_write_to_file
from src.persistence.runtime_state_store import RuntimeStateStore
from src.utils import config_loader

logger = logging.getLogger(__name__)

_PRECOMPUTED_PREDICTIONS_FILE = Path("data/systems/precomputed_predictions.json")
_STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS = "precomputed_predictions"
_DEFAULT_MAX_FILE_ENTRIES = 96
_DEFAULT_WEATHER_SCENARIOS = ("dry", "mixed", "rain")


def get_prediction_precompute_config() -> dict[str, Any]:
    """
    Read precompute behavior from config with safe defaults.

    Returns:
        dict with:
            enabled: bool
            include_next_weekend: bool
            weather_scenarios: list[str]
            max_file_entries: int
    """
    enabled = bool(config_loader.get("dashboard.prediction_precompute.enabled", True))
    include_next_weekend = bool(
        config_loader.get("dashboard.prediction_precompute.include_next_weekend", False)
    )
    raw_weather = config_loader.get(
        "dashboard.prediction_precompute.weather_scenarios",
        list(_DEFAULT_WEATHER_SCENARIOS),
    )
    valid_weather = {"dry", "mixed", "rain"}
    weather_scenarios: list[str] = []
    if isinstance(raw_weather, list):
        for item in raw_weather:
            normalized = str(item).strip().lower()
            if normalized in valid_weather and normalized not in weather_scenarios:
                weather_scenarios.append(normalized)
    if not weather_scenarios:
        weather_scenarios = list(_DEFAULT_WEATHER_SCENARIOS)

    raw_max_entries = config_loader.get(
        "dashboard.prediction_precompute.max_file_entries",
        _DEFAULT_MAX_FILE_ENTRIES,
    )
    try:
        max_file_entries = max(16, int(raw_max_entries))
    except (TypeError, ValueError):
        max_file_entries = _DEFAULT_MAX_FILE_ENTRIES

    return {
        "enabled": enabled,
        "include_next_weekend": include_next_weekend,
        "weather_scenarios": weather_scenarios,
        "max_file_entries": max_file_entries,
    }


def compute_artifact_hash(artifact_versions: dict[str, tuple[int, str]]) -> str:
    """Build a stable hash from artifact versions for precompute cache keys."""
    normalized = {
        str(key): [int(value[0]), str(value[1])]
        for key, value in sorted(artifact_versions.items())
        if isinstance(value, tuple) and len(value) >= 2
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()


def build_precomputed_prediction_key(
    *,
    year: int,
    race_name: str,
    weather: str,
    artifact_hash: str,
    boundary_signature: str,
) -> str:
    """Build deterministic key for precomputed prediction storage."""
    payload = {
        "year": int(year),
        "race_name": str(race_name),
        "weather": str(weather).strip().lower(),
        "artifact_hash": str(artifact_hash),
        "boundary_signature": str(boundary_signature),
    }
    return sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_precomputed_prediction(
    *,
    year: int,
    race_name: str,
    weather: str,
    artifact_hash: str,
    boundary_signature: str,
) -> dict[str, Any] | None:
    """Load precomputed prediction payload from DB/file backends."""
    state_key = build_precomputed_prediction_key(
        year=year,
        race_name=race_name,
        weather=weather,
        artifact_hash=artifact_hash,
        boundary_signature=boundary_signature,
    )

    if should_read_db_first():
        try:
            store = RuntimeStateStore()
            db_payload = store.get_record(_STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS, state_key)
            validated = _extract_prediction_results(db_payload)
            if validated is not None:
                return validated
        except Exception as exc:
            logger.warning("Could not load precomputed prediction from DB: %s", exc)

    file_state = _load_file_state()
    file_entries = file_state.get("entries", {})
    if not isinstance(file_entries, dict):
        return None
    file_payload = file_entries.get(state_key)
    return _extract_prediction_results(file_payload)


def save_precomputed_prediction(
    *,
    year: int,
    race_name: str,
    weather: str,
    artifact_hash: str,
    boundary_signature: str,
    is_sprint: bool,
    prediction_results: dict[str, Any],
    max_file_entries: int | None = None,
) -> None:
    """Persist precomputed prediction payload to DB/file backends."""
    state_key = build_precomputed_prediction_key(
        year=year,
        race_name=race_name,
        weather=weather,
        artifact_hash=artifact_hash,
        boundary_signature=boundary_signature,
    )
    now_iso = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "year": int(year),
        "race_name": str(race_name),
        "weather": str(weather).strip().lower(),
        "artifact_hash": str(artifact_hash),
        "boundary_signature": str(boundary_signature),
        "is_sprint": bool(is_sprint),
        "updated_at": now_iso,
        "prediction_results": prediction_results,
    }

    if should_write_to_db():
        try:
            RuntimeStateStore().upsert_record(
                _STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS,
                state_key,
                payload,
            )
        except Exception as exc:
            logger.warning("Could not save precomputed prediction to DB: %s", exc)

    if not should_write_to_file():
        return

    try:
        file_state = _load_file_state()
        entries = file_state.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        entries[state_key] = payload

        if max_file_entries is None:
            max_entries = _DEFAULT_MAX_FILE_ENTRIES
        else:
            max_entries = max(16, int(max_file_entries))
        _prune_entries(entries, max_entries=max_entries)

        file_state["entries"] = entries
        file_state["updated_at"] = now_iso
        _write_file_state(file_state)
    except Exception as exc:
        logger.warning("Could not save precomputed prediction to file cache: %s", exc)


def _extract_prediction_results(payload: Any) -> dict[str, Any] | None:
    """Extract prediction results when payload has the expected shape."""
    if not isinstance(payload, dict):
        return None
    raw_results = payload.get("prediction_results")
    if not isinstance(raw_results, dict):
        return None
    return raw_results


def _load_file_state() -> dict[str, Any]:
    """Load file-backed precompute state, returning empty state on corruption."""
    if not _PRECOMPUTED_PREDICTIONS_FILE.exists():
        return {"entries": {}}

    try:
        with open(_PRECOMPUTED_PREDICTIONS_FILE) as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"entries": {}}

    if not isinstance(loaded, dict):
        return {"entries": {}}
    if not isinstance(loaded.get("entries"), dict):
        loaded["entries"] = {}
    return loaded


def _write_file_state(state: dict[str, Any]) -> None:
    """Persist file-backed precompute state atomically."""
    _PRECOMPUTED_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRECOMPUTED_PREDICTIONS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(state, handle, indent=2)
    tmp_path.replace(_PRECOMPUTED_PREDICTIONS_FILE)


def _prune_entries(entries: dict[str, Any], *, max_entries: int) -> None:
    """Prune oldest entries in-place by `updated_at` timestamp."""
    if len(entries) <= max_entries:
        return

    sortable: list[tuple[str, str]] = []
    for key, value in entries.items():
        if not isinstance(value, dict):
            sortable.append((key, ""))
            continue
        sortable.append((key, str(value.get("updated_at", ""))))

    sortable.sort(key=lambda item: item[1])
    overflow = len(entries) - max_entries
    for key, _ in sortable[:overflow]:
        entries.pop(key, None)
