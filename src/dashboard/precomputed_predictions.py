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
_PRECOMPUTED_BASE_FEATURES_FILE = Path("data/systems/precomputed_base_features.json")
_STATE_NAMESPACE_PRECOMPUTED_BASE_FEATURES = "precomputed_prediction_base_features"
_PRECOMPUTE_HORIZON_INDEX_FILE = Path("data/systems/precompute_horizon_index.json")
_STATE_NAMESPACE_PRECOMPUTE_HORIZON_INDEX = "prediction_precompute_horizon_index"
_DEFAULT_MAX_FILE_ENTRIES = 2048
_DEFAULT_PRECOMPUTE_HORIZON_RACES = 3
_DEFAULT_WEATHER_SCENARIOS = ("dry", "mixed", "rain")
_STATE_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _resolve_max_entries(max_file_entries: int | None) -> int:
    """Clamp the configured entry limit."""
    if max_file_entries is None:
        return _DEFAULT_MAX_FILE_ENTRIES
    try:
        return max(16, int(max_file_entries))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_FILE_ENTRIES


def _resolve_simulation_count(value: Any, *, default: int) -> int:
    """Clamp the configured simulation count."""
    try:
        return max(10, int(value))
    except (TypeError, ValueError):
        return int(default)


def _is_db_only_mode() -> bool:
    """Return True when DB is the only configured writable backend."""
    return should_write_to_db() and not should_write_to_file()


def _parse_updated_at(value: Any) -> datetime:
    """Parse entry updated-at timestamp; return oldest sentinel when unavailable."""
    if not isinstance(value, str):
        return datetime.min.replace(tzinfo=UTC)
    candidate = value.strip()
    if not candidate:
        return datetime.min.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _prune_db_namespace_entries(
    namespace: str,
    *,
    max_entries: int,
    store: RuntimeStateStore | None = None,
) -> None:
    """
    Prune runtime-state namespace rows in DB to bounded size by updated-at.

    Entries with missing or invalid ``updated_at`` are treated as oldest.
    """
    runtime_store = store or RuntimeStateStore()
    overflow = 0
    if hasattr(runtime_store, "count_records") and callable(
        getattr(runtime_store, "count_records", None)
    ):
        namespace_count = runtime_store.count_records(namespace)
        overflow = max(0, int(namespace_count) - int(max_entries))
    if overflow <= 0:
        return

    stale_keys: list[str] = []
    if hasattr(runtime_store, "list_oldest_state_keys") and callable(
        getattr(runtime_store, "list_oldest_state_keys", None)
    ):
        stale_keys = runtime_store.list_oldest_state_keys(namespace, limit=overflow)

    if not stale_keys:
        entries = runtime_store.load_namespace(namespace)
        if not isinstance(entries, dict) or len(entries) <= max_entries:
            return

        sortable: list[tuple[str, datetime]] = []
        for state_key, payload in entries.items():
            if not isinstance(state_key, str):
                continue
            updated_at = _parse_updated_at(
                payload.get("updated_at") if isinstance(payload, dict) else ""
            )
            sortable.append((state_key, updated_at))
        if len(sortable) <= max_entries:
            return
        sortable.sort(key=lambda item: item[1])
        overflow = len(sortable) - max_entries
        stale_keys = [state_key for state_key, _ in sortable[:overflow]]

    if stale_keys:
        runtime_store.delete_records(namespace, stale_keys)


def get_prediction_precompute_config() -> dict[str, Any]:
    """Read precompute settings from config."""
    enabled = bool(config_loader.get("dashboard.prediction_precompute.enabled", True))
    raw_horizon_races = config_loader.get(
        "dashboard.prediction_precompute.horizon_races",
        _DEFAULT_PRECOMPUTE_HORIZON_RACES,
    )
    try:
        horizon_races = max(1, int(raw_horizon_races))
    except (TypeError, ValueError):
        horizon_races = _DEFAULT_PRECOMPUTE_HORIZON_RACES
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
        "horizon_races": horizon_races,
        "weather_scenarios": weather_scenarios,
        "max_file_entries": max_file_entries,
        "qualifying_n_simulations": _resolve_simulation_count(
            config_loader.get("dashboard.prediction_precompute.qualifying_n_simulations", 100),
            default=100,
        ),
        "race_n_simulations": _resolve_simulation_count(
            config_loader.get("dashboard.prediction_precompute.race_n_simulations", 100),
            default=100,
        ),
    }


def compute_artifact_hash(artifact_versions: dict[str, tuple[int, str]]) -> str:
    """Hash artifact versions for cache keys."""
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
    """Build the cache key for a precomputed prediction."""
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
    """Load a cached prediction."""
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
        except _STATE_ERRORS as exc:
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
    metadata: dict[str, Any] | None = None,
    max_file_entries: int | None = None,
) -> None:
    """Persist precomputed prediction payload to DB/file backends."""
    max_entries = _resolve_max_entries(max_file_entries)
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
        "metadata": metadata or {},
    }

    if should_write_to_db():
        try:
            runtime_store = RuntimeStateStore()
            runtime_store.upsert_record(
                _STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS,
                state_key,
                payload,
            )
            _prune_db_namespace_entries(
                _STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS,
                max_entries=max_entries,
                store=runtime_store,
            )
        except _STATE_ERRORS as exc:
            logger.warning("Could not save precomputed prediction to DB: %s", exc)
            if _is_db_only_mode():
                raise RuntimeError(
                    "Could not save precomputed prediction in db_only mode."
                ) from exc

    if not should_write_to_file():
        return

    try:
        file_state = _load_file_state()
        entries = file_state.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        entries[state_key] = payload

        _prune_entries(entries, max_entries=max_entries)

        file_state["entries"] = entries
        file_state["updated_at"] = now_iso
        _write_file_state(file_state)
    except _STATE_ERRORS as exc:
        logger.warning("Could not save precomputed prediction to file cache: %s", exc)


def build_precomputed_base_features_key(
    *,
    year: int,
    race_name: str,
    checkpoint: str,
    artifact_hash: str,
    boundary_signature: str,
) -> str:
    """Build the cache key for base features."""
    payload = {
        "year": int(year),
        "race_name": str(race_name).strip(),
        "checkpoint": str(checkpoint).strip().upper(),
        "artifact_hash": str(artifact_hash),
        "boundary_signature": str(boundary_signature),
    }
    return sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_precomputed_base_features(
    *,
    year: int,
    race_name: str,
    checkpoint: str,
    artifact_hash: str,
    boundary_signature: str,
) -> dict[str, Any] | None:
    """Load cached base features."""
    state_key = build_precomputed_base_features_key(
        year=year,
        race_name=race_name,
        checkpoint=checkpoint,
        artifact_hash=artifact_hash,
        boundary_signature=boundary_signature,
    )

    if should_read_db_first():
        try:
            store = RuntimeStateStore()
            db_payload = store.get_record(_STATE_NAMESPACE_PRECOMPUTED_BASE_FEATURES, state_key)
            validated = _extract_base_features(db_payload)
            if validated is not None:
                return validated
        except _STATE_ERRORS as exc:
            logger.warning("Could not load precomputed base features from DB: %s", exc)

    file_state = _load_base_features_file_state()
    file_entries = file_state.get("entries", {})
    if not isinstance(file_entries, dict):
        return None
    file_payload = file_entries.get(state_key)
    return _extract_base_features(file_payload)


def save_precomputed_base_features(
    *,
    year: int,
    race_name: str,
    checkpoint: str,
    artifact_hash: str,
    boundary_signature: str,
    is_sprint: bool,
    base_features: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    max_file_entries: int | None = None,
) -> None:
    """Persist precomputed base-feature payload to DB/file backends."""
    max_entries = _resolve_max_entries(max_file_entries)
    state_key = build_precomputed_base_features_key(
        year=year,
        race_name=race_name,
        checkpoint=checkpoint,
        artifact_hash=artifact_hash,
        boundary_signature=boundary_signature,
    )
    now_iso = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "year": int(year),
        "race_name": str(race_name),
        "checkpoint": str(checkpoint).strip().upper(),
        "artifact_hash": str(artifact_hash),
        "boundary_signature": str(boundary_signature),
        "is_sprint": bool(is_sprint),
        "updated_at": now_iso,
        "base_features": base_features,
        "metadata": metadata or {},
    }

    if should_write_to_db():
        try:
            runtime_store = RuntimeStateStore()
            runtime_store.upsert_record(
                _STATE_NAMESPACE_PRECOMPUTED_BASE_FEATURES,
                state_key,
                payload,
            )
            _prune_db_namespace_entries(
                _STATE_NAMESPACE_PRECOMPUTED_BASE_FEATURES,
                max_entries=max_entries,
                store=runtime_store,
            )
        except _STATE_ERRORS as exc:
            logger.warning("Could not save precomputed base features to DB: %s", exc)
            if _is_db_only_mode():
                raise RuntimeError(
                    "Could not save precomputed base features in db_only mode."
                ) from exc

    if not should_write_to_file():
        return

    try:
        file_state = _load_base_features_file_state()
        entries = file_state.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        entries[state_key] = payload

        _prune_entries(entries, max_entries=max_entries)

        file_state["entries"] = entries
        file_state["updated_at"] = now_iso
        _write_base_features_file_state(file_state)
    except _STATE_ERRORS as exc:
        logger.warning("Could not save precomputed base features to file cache: %s", exc)


def _extract_prediction_results(payload: Any) -> dict[str, Any] | None:
    """
    Extract prediction results when payload has the expected shape.

    The returned mapping keeps a small context footer so downstream callers can
    tell which boundary session produced the cached prediction and when the
    persisted payload was last updated. That matters when a served prediction is
    later persisted for accuracy tracking or refreshed in memory.
    """
    if not isinstance(payload, dict):
        return None
    raw_results = payload.get("prediction_results")
    if not isinstance(raw_results, dict):
        return None

    extracted_results = dict(raw_results)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        boundary_session_name = str(metadata.get("boundary_session_name", "")).strip().upper()
        if boundary_session_name:
            existing_context = extracted_results.get("_prediction_context")
            context = dict(existing_context) if isinstance(existing_context, dict) else {}
            context["boundary_session_name"] = boundary_session_name
            extracted_results["_prediction_context"] = context

    updated_at = str(payload.get("updated_at", "")).strip()
    if updated_at:
        existing_context = extracted_results.get("_prediction_context")
        context = dict(existing_context) if isinstance(existing_context, dict) else {}
        context["persisted_updated_at"] = updated_at
        extracted_results["_prediction_context"] = context

    return extracted_results


def _extract_base_features(payload: Any) -> dict[str, Any] | None:
    """Extract base-feature payload when it has the expected shape."""
    if not isinstance(payload, dict):
        return None
    raw_features = payload.get("base_features")
    if not isinstance(raw_features, dict):
        return None
    return raw_features


def list_precomputed_race_names(
    *,
    year: int,
    artifact_hash: str,
    boundary_signature: str | None = None,
) -> list[str]:
    """
    List race names that already have cached predictions.

    Args:
        year: Season year used in prediction keys.
        artifact_hash: Artifact hash currently active in predictor bootstrap.
        boundary_signature: Optional boundary signature filter.

    Returns:
        Sorted race names with at least one persisted weather scenario.
    """
    race_names: set[str] = set()
    expected_year = int(year)
    expected_hash = str(artifact_hash).strip()
    expected_boundary = None if boundary_signature is None else str(boundary_signature).strip()

    def _collect(entries: dict[str, Any]) -> None:
        for payload in entries.values():
            if not isinstance(payload, dict):
                continue
            raw_payload_year = payload.get("year", "")
            if not isinstance(raw_payload_year, int | float | str):
                continue
            try:
                payload_year = int(raw_payload_year)
            except (TypeError, ValueError):
                continue
            if payload_year != expected_year:
                continue
            if str(payload.get("artifact_hash", "")).strip() != expected_hash:
                continue
            if expected_boundary is not None and (
                str(payload.get("boundary_signature", "")).strip() != expected_boundary
            ):
                continue
            race_name = str(payload.get("race_name", "")).strip()
            if race_name:
                race_names.add(race_name)

    if should_read_db_first():
        try:
            db_entries = RuntimeStateStore().load_namespace(
                _STATE_NAMESPACE_PRECOMPUTED_PREDICTIONS
            )
            if isinstance(db_entries, dict):
                _collect(db_entries)
        except _STATE_ERRORS as exc:
            logger.warning("Could not list precomputed predictions from DB: %s", exc)

    if should_write_to_file():
        file_state = _load_file_state()
        file_entries = file_state.get("entries", {})
        if isinstance(file_entries, dict):
            _collect(file_entries)

    return sorted(race_names)


def load_precompute_horizon_index(*, year: int, artifact_hash: str) -> dict[str, Any] | None:
    """Load the saved warmup horizon for one season and artifact state."""
    state_key = f"{int(year)}::{str(artifact_hash).strip()}"

    if should_read_db_first():
        try:
            payload = RuntimeStateStore().get_record(
                _STATE_NAMESPACE_PRECOMPUTE_HORIZON_INDEX, state_key
            )
            if isinstance(payload, dict):
                return payload
        except _STATE_ERRORS as exc:
            logger.warning("Could not load precompute horizon index from DB: %s", exc)

    file_state = _load_horizon_index_state()
    entries = file_state.get("entries", {})
    if not isinstance(entries, dict):
        return None
    payload = entries.get(state_key)
    return payload if isinstance(payload, dict) else None


def has_precompute_horizon_for_year(
    *,
    year: int,
    exclude_artifact_hash: str | None = None,
) -> bool:
    """Return True when the season has any saved horizon metadata."""
    year_prefix = f"{int(year)}::"
    excluded_key = ""
    if exclude_artifact_hash is not None:
        excluded_key = f"{int(year)}::{str(exclude_artifact_hash).strip()}"

    def _matches(entries: Any) -> bool:
        if not isinstance(entries, dict):
            return False
        for state_key, payload in entries.items():
            normalized_key = str(state_key).strip()
            if not normalized_key.startswith(year_prefix):
                continue
            if excluded_key and normalized_key == excluded_key:
                continue
            if isinstance(payload, dict):
                return True
        return False

    if should_read_db_first():
        try:
            db_entries = RuntimeStateStore().load_namespace(
                _STATE_NAMESPACE_PRECOMPUTE_HORIZON_INDEX
            )
            if _matches(db_entries):
                return True
        except _STATE_ERRORS as exc:
            logger.warning("Could not inspect precompute horizon index in DB: %s", exc)

    file_state = _load_horizon_index_state()
    file_entries = file_state.get("entries", {})
    return _matches(file_entries)


def save_precompute_horizon_index(
    *,
    year: int,
    artifact_hash: str,
    boundary_signature: str,
    anchor_race_name: str,
    anchor_session_name: str,
    expected_targets: list[str],
    ready_races: list[str],
    weather_scenarios: list[str],
    race_boundaries: dict[str, str] | None = None,
) -> None:
    """Save the current warmup horizon."""
    state_key = f"{int(year)}::{str(artifact_hash).strip()}"
    now_iso = datetime.now(UTC).isoformat()
    payload = {
        "year": int(year),
        "artifact_hash": str(artifact_hash).strip(),
        "boundary_signature": str(boundary_signature).strip(),
        "anchor_race_name": str(anchor_race_name).strip(),
        "anchor_session_name": str(anchor_session_name).strip().upper(),
        "expected_targets": [str(race).strip() for race in expected_targets if str(race).strip()],
        "ready_races": [str(race).strip() for race in ready_races if str(race).strip()],
        "weather_scenarios": [
            str(weather).strip().lower() for weather in weather_scenarios if str(weather).strip()
        ],
        "race_boundaries": {
            str(race).strip(): str(signature).strip()
            for race, signature in (race_boundaries or {}).items()
            if str(race).strip()
        },
        "updated_at": now_iso,
    }

    if should_write_to_db():
        try:
            RuntimeStateStore().upsert_record(
                _STATE_NAMESPACE_PRECOMPUTE_HORIZON_INDEX,
                state_key,
                payload,
            )
        except _STATE_ERRORS as exc:
            logger.warning("Could not save precompute horizon index to DB: %s", exc)
            if _is_db_only_mode():
                raise RuntimeError(
                    "Could not save precompute horizon index in db_only mode."
                ) from exc

    if not should_write_to_file():
        return

    try:
        state = _load_horizon_index_state()
        entries = state.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        entries[state_key] = payload
        state["entries"] = entries
        state["updated_at"] = now_iso
        _write_horizon_index_state(state)
    except _STATE_ERRORS as exc:
        logger.warning("Could not save precompute horizon index to file: %s", exc)


def _load_file_state() -> dict[str, Any]:
    """Load prediction cache state from disk."""
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
    """Write prediction cache state to disk."""
    _PRECOMPUTED_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRECOMPUTED_PREDICTIONS_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(state, handle, indent=2)
    tmp_path.replace(_PRECOMPUTED_PREDICTIONS_FILE)


def _load_horizon_index_state() -> dict[str, Any]:
    """Load horizon index state from disk."""
    if not _PRECOMPUTE_HORIZON_INDEX_FILE.exists():
        return {"entries": {}}

    try:
        with open(_PRECOMPUTE_HORIZON_INDEX_FILE) as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"entries": {}}

    if not isinstance(loaded, dict):
        return {"entries": {}}
    if not isinstance(loaded.get("entries"), dict):
        loaded["entries"] = {}
    return loaded


def _load_base_features_file_state() -> dict[str, Any]:
    """Load base-feature cache state from disk."""
    if not _PRECOMPUTED_BASE_FEATURES_FILE.exists():
        return {"entries": {}}

    try:
        with open(_PRECOMPUTED_BASE_FEATURES_FILE) as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"entries": {}}

    if not isinstance(loaded, dict):
        return {"entries": {}}
    if not isinstance(loaded.get("entries"), dict):
        loaded["entries"] = {}
    return loaded


def _write_base_features_file_state(state: dict[str, Any]) -> None:
    """Persist file-backed base-feature state atomically."""
    _PRECOMPUTED_BASE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRECOMPUTED_BASE_FEATURES_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(state, handle, indent=2)
    tmp_path.replace(_PRECOMPUTED_BASE_FEATURES_FILE)


def _write_horizon_index_state(state: dict[str, Any]) -> None:
    """Persist file-backed horizon-index state atomically."""
    _PRECOMPUTE_HORIZON_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRECOMPUTE_HORIZON_INDEX_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(state, handle, indent=2)
    tmp_path.replace(_PRECOMPUTE_HORIZON_INDEX_FILE)


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
