"""
Artifact Store: Abstraction layer for file and database persistence.

Supports multiple storage modes:
- file_only: Read/write only to local files (development)
- db_only: Read/write only to Supabase (production)
- fallback: Read DB first, fall back to file if not found
- dual_write: Write to both DB and file (migration phase)
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from postgrest.exceptions import APIError

from .config import (
    get_storage_mode,
    should_read_db_first,
    should_write_to_db,
    should_write_to_file,
)
from .db import get_supabase_client

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Unified interface for artifact persistence.

    Handles both file-based and database-backed storage with
    configurable modes for zero-downtime migration.
    """

    def __init__(self, data_root: str | Path = "data"):
        """Initialize artifact store with data_root for file-based storage."""
        self.data_root = Path(data_root)
        self.storage_mode = get_storage_mode()
        logger.info(f"ArtifactStore initialized with mode: {self.storage_mode}")

    def save_artifact(
        self,
        artifact_type: str,
        artifact_key: str,
        data: dict[str, Any],
        version: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Save artifact to configured storage backend(s).

        Raises RuntimeError if all writes fail. Returns saved artifact metadata.
        """
        start_time = time.time()
        file_success = db_success = False
        last_error = None

        # Determine version
        if version is None:
            # Auto-increment: get latest version + 1
            try:
                latest_version = self.get_latest_version(artifact_type, artifact_key)
                version = latest_version + 1
            except (APIError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                version = 1  # First version

        # Prepare run_id (convert to UUID if needed)
        run_uuid = None
        if run_id:
            try:
                run_uuid = uuid.UUID(run_id) if isinstance(run_id, str) else run_id
            except ValueError:
                logger.warning(f"Invalid run_id format: {run_id}, ignoring")

        # Write to file
        if should_write_to_file():
            try:
                self._write_file(artifact_type, artifact_key, data)
                file_success = True
                logger.debug(f"File write successful: {artifact_type}::{artifact_key} v{version}")
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                last_error = e
                logger.error(f"File write failed: {e}")

        # Write to DB
        if should_write_to_db():
            try:
                result = self._write_db(artifact_type, artifact_key, data, version, run_uuid)
                db_success = True
                elapsed = time.time() - start_time
                logger.info(
                    f"DB write successful: {artifact_type}::{artifact_key} v{version} ({elapsed:.3f}s)"
                )
                # Return DB result (includes id, timestamps)
                return result
            except (APIError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                last_error = e
                logger.error(f"DB write failed: {e}")

        # Check if any write succeeded
        if not file_success and not db_success:
            raise RuntimeError(f"All writes failed. Last error: {last_error}")

        # Return minimal metadata if only file write succeeded
        elapsed = time.time() - start_time
        return {
            "artifact_type": artifact_type,
            "artifact_key": artifact_key,
            "version": version,
            "run_id": str(run_uuid) if run_uuid else None,
            "storage": "file_only",
            "elapsed_ms": elapsed * 1000,
        }

    def load_artifact(
        self,
        artifact_type: str,
        artifact_key: str,
        version: str | int = "latest",
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load artifact from configured storage backend(s). Returns data or None if not found."""
        start_time = time.time()

        # Try DB first if configured
        if should_read_db_first():
            try:
                result = self._read_db(artifact_type, artifact_key, version, run_id)
                if result:
                    elapsed = time.time() - start_time
                    logger.debug(
                        f"DB read successful: {artifact_type}::{artifact_key} ({elapsed:.3f}s)"
                    )
                    return result
            except (APIError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"DB read failed, trying file fallback: {e}")

        # Try file (either as primary or fallback)
        try:
            result = self._read_file(artifact_type, artifact_key)
            if result:
                elapsed = time.time() - start_time
                logger.debug(
                    f"File read successful: {artifact_type}::{artifact_key} ({elapsed:.3f}s)"
                )
                return result
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"File read failed: {e}")

        return None

    def list_artifacts(
        self,
        artifact_type: str,
        key_prefix: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List artifacts of given type. Returns artifact metadata sorted by created_at DESC."""
        # Try DB first if configured
        if should_read_db_first():
            try:
                db_rows = self._list_db(artifact_type, key_prefix, limit)
                if db_rows or self.storage_mode == "db_only":
                    return db_rows
            except (APIError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"DB list failed: {e}")

        # Fallback to file listing (less efficient)
        return self._list_files(artifact_type, key_prefix, limit)

    def get_latest_version(self, artifact_type: str, artifact_key: str) -> int:
        """Get the latest version number for an artifact. Returns 0 if artifact doesn't exist."""
        if should_read_db_first():
            try:
                supabase = get_supabase_client()
                result = (
                    supabase.table("artifacts")
                    .select("version")
                    .eq("artifact_type", artifact_type)
                    .eq("artifact_key", artifact_key)
                    .order("version", desc=True)
                    .limit(1)
                    .execute()
                )
                if result.data:
                    row = cast(list[dict[str, Any]], result.data)[0]
                    return int(row.get("version", 0))
            except (APIError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"DB version check failed: {e}")

        # Fallback: check file (version stored in data if available)
        try:
            data = self._read_file(artifact_type, artifact_key)
            if data and "version" in data:
                return data["version"]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

        return 0

    # Private methods: File operations

    @staticmethod
    def _extract_year_from_artifact_key(artifact_key: str) -> str | None:
        """Extract season-year token from `YYYY::...` artifact keys."""
        if "::" not in artifact_key:
            return None
        year_token = artifact_key.split("::", 1)[0].strip()
        return year_token if year_token.isdigit() else None

    def _get_file_path(self, artifact_type: str, artifact_key: str) -> Path:
        """Map artifact type/key to file path."""
        # Map to existing file paths
        if artifact_type == "car_characteristics":
            year = artifact_key.split("::")[0] if "::" in artifact_key else "2026"
            return (
                self.data_root
                / "processed"
                / "car_characteristics"
                / f"{year}_car_characteristics.json"
            )
        elif artifact_type == "driver_characteristics":
            season_year = self._extract_year_from_artifact_key(artifact_key)
            if season_year:
                return (
                    self.data_root
                    / "processed"
                    / "driver_characteristics"
                    / f"{season_year}_driver_characteristics.json"
                )
            return self.data_root / "processed" / "driver_characteristics.json"
        elif artifact_type == "driver_debuts":
            return self.data_root / "driver_debuts.json"
        elif artifact_type == "track_characteristics":
            year = artifact_key.split("::")[0] if "::" in artifact_key else "2026"
            return (
                self.data_root
                / "processed"
                / "track_characteristics"
                / f"{year}_track_characteristics.json"
            )
        elif artifact_type == "learning_state":
            learning_year = self._extract_year_from_artifact_key(artifact_key)
            if learning_year:
                return self.data_root / "learning_state" / f"{learning_year}_learning_state.json"
            return self.data_root / "learning_state.json"
        elif artifact_type == "practice_state":
            return self.data_root / "systems" / "practice_characteristics_state.json"
        elif artifact_type == "prediction":
            # Parse key: '2026::Bahrain Grand Prix::qualifying'
            parts = artifact_key.split("::")
            if len(parts) == 3:
                year, race_name, session_name = parts
                safe_race = race_name.lower().replace(" ", "_").replace("'", "")
                return (
                    self.data_root
                    / "predictions"
                    / year
                    / safe_race
                    / f"{safe_race}_{session_name.lower()}.json"
                )

        # Default: use type/key structure
        safe_key = artifact_key.replace("::", "/")
        return self.data_root / artifact_type / f"{safe_key}.json"

    def _write_file(self, artifact_type: str, artifact_key: str, data: dict[str, Any]) -> None:
        """Write artifact to file."""
        file_path = self._get_file_path(artifact_type, artifact_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _read_file(self, artifact_type: str, artifact_key: str) -> dict[str, Any] | None:
        """Read artifact from file."""
        file_path = self._get_file_path(artifact_type, artifact_key)

        if not file_path.exists():
            return None

        with open(file_path) as f:
            return json.load(f)

    def _list_files(
        self, artifact_type: str, key_prefix: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """List files (slow, fallback only)."""
        base_path = self._get_file_listing_base_path(artifact_type)
        candidates = self._iter_candidate_files(artifact_type, base_path)

        rows: list[tuple[float, dict[str, Any]]] = []
        for file_path in candidates:
            if not file_path.exists() or not file_path.is_file():
                continue

            artifact_key = self._artifact_key_from_file_path(artifact_type, file_path, base_path)
            if key_prefix and not artifact_key.startswith(key_prefix):
                continue

            try:
                data = json.loads(file_path.read_text())
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                logger.warning(f"Skipping unreadable artifact file {file_path}: {exc}")
                continue

            mtime = file_path.stat().st_mtime
            rows.append(
                (
                    mtime,
                    {
                        "artifact_type": artifact_type,
                        "artifact_key": artifact_key,
                        "version": data.get("version") if isinstance(data, dict) else None,
                        "run_id": data.get("run_id") if isinstance(data, dict) else None,
                        "created_at": datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
                        "data": data,
                    },
                )
            )

        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows[:limit]]

    def _get_file_listing_base_path(self, artifact_type: str) -> Path:
        """Get base directory for file-based artifact listing."""
        if artifact_type == "car_characteristics":
            return self.data_root / "processed" / "car_characteristics"
        if artifact_type == "driver_characteristics":
            return self.data_root / "processed"
        if artifact_type == "driver_debuts":
            return self.data_root
        if artifact_type == "track_characteristics":
            return self.data_root / "processed" / "track_characteristics"
        if artifact_type == "learning_state":
            return self.data_root
        if artifact_type == "practice_state":
            return self.data_root / "systems"
        if artifact_type == "prediction":
            return self.data_root / "predictions"
        return self.data_root / artifact_type

    def _iter_candidate_files(self, artifact_type: str, base_path: Path) -> list[Path]:
        """Return candidate files for fallback listing."""
        if artifact_type == "driver_characteristics":
            season_root = base_path / "driver_characteristics"
            season_files: list[Path] = (
                list(season_root.glob("*_driver_characteristics.json"))
                if season_root.exists()
                else []
            )
            return [*season_files, base_path / "driver_characteristics.json"]
        if artifact_type == "driver_debuts":
            return [base_path / "driver_debuts.json"]
        if artifact_type == "learning_state":
            season_root = base_path / "learning_state"
            learning_season_files: list[Path] = (
                list(season_root.glob("*_learning_state.json")) if season_root.exists() else []
            )
            return [*learning_season_files, base_path / "learning_state.json"]
        if artifact_type == "practice_state":
            return [base_path / "practice_characteristics_state.json"]

        if not base_path.exists():
            return []

        pattern = "*.json"
        if artifact_type == "car_characteristics":
            pattern = "*_car_characteristics.json"
        elif artifact_type == "track_characteristics":
            pattern = "*_track_characteristics.json"

        return list(base_path.rglob(pattern))

    def _artifact_key_from_file_path(
        self, artifact_type: str, file_path: Path, base_path: Path
    ) -> str:
        """Derive artifact key from file path for file-based listing."""
        if artifact_type == "car_characteristics":
            year = file_path.stem.split("_")[0]
            return f"{year}::car_characteristics"

        if artifact_type == "driver_characteristics":
            stem = file_path.stem
            if stem.endswith("_driver_characteristics"):
                year = stem.split("_", 1)[0]
                if year.isdigit():
                    return f"{year}::driver_characteristics"
            return "driver_characteristics"

        if artifact_type == "driver_debuts":
            return "driver_debuts"

        if artifact_type == "track_characteristics":
            year = file_path.stem.split("_")[0]
            return f"{year}::track_characteristics"

        if artifact_type == "learning_state":
            stem = file_path.stem
            if stem.endswith("_learning_state"):
                year = stem.split("_", 1)[0]
                if year.isdigit():
                    return f"{year}::learning_state"
            return "learning_state"

        if artifact_type == "practice_state":
            return "practice_state"

        if artifact_type == "prediction":
            parts = file_path.relative_to(base_path).parts
            if len(parts) >= 3:
                year = parts[0]
                race_slug = parts[-2]
                session_fragment = Path(parts[-1]).stem
                prefix = f"{race_slug}_"
                if session_fragment.startswith(prefix):
                    session_fragment = session_fragment[len(prefix) :]
                return f"{year}::{race_slug}::{session_fragment}"

        relative_stem = file_path.relative_to(base_path).with_suffix("")
        return "::".join(relative_stem.parts)

    # Private methods: Database operations

    def _write_db(
        self,
        artifact_type: str,
        artifact_key: str,
        data: dict[str, Any],
        version: int,
        run_uuid: uuid.UUID | None,
    ) -> dict[str, Any]:
        """Write artifact to Supabase."""
        supabase = get_supabase_client()

        row = {
            "artifact_type": artifact_type,
            "artifact_key": artifact_key,
            "version": version,
            "data": data,
        }

        if run_uuid:
            row["run_id"] = str(run_uuid)

        # Insert or update (upsert on conflict)
        result = (
            supabase.table("artifacts")
            .upsert(cast(Any, row), on_conflict="artifact_type,artifact_key,version")
            .execute()
        )

        if not result.data:
            raise RuntimeError("DB insert returned no data")

        return cast(list[dict[str, Any]], result.data)[0]

    def _read_db(
        self,
        artifact_type: str,
        artifact_key: str,
        version: str | int,
        run_id: str | None,
    ) -> dict[str, Any] | None:
        """Read artifact from Supabase."""
        supabase = get_supabase_client()

        query = supabase.table("artifacts").select("data")

        # Filter by type and key
        query = query.eq("artifact_type", artifact_type)
        query = query.eq("artifact_key", artifact_key)

        # Filter by run_id if provided
        if run_id:
            query = query.eq("run_id", run_id)

        # Filter by version
        if version == "latest":
            query = query.order("version", desc=True).limit(1)
        else:
            query = query.eq("version", version)

        result = query.execute()

        if result.data:
            row = cast(list[dict[str, Any]], result.data)[0]
            payload = row.get("data")
            return payload if isinstance(payload, dict) else None

        return None

    def _list_db(
        self, artifact_type: str, key_prefix: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """List artifacts from Supabase."""
        supabase = get_supabase_client()

        query = supabase.table("artifacts").select(
            "artifact_key, version, run_id, created_at, data"
        )
        query = query.eq("artifact_type", artifact_type)

        if key_prefix:
            query = query.like("artifact_key", f"{key_prefix}%")

        query = query.order("created_at", desc=True).limit(limit)

        result = query.execute()
        return cast(list[dict[str, Any]], result.data) if result.data else []
