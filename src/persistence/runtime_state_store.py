"""Supabase-backed runtime state and processing-lock store."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from .config import get_storage_mode, should_read_db_first, should_write_to_db
from .db import get_supabase_client

logger = logging.getLogger(__name__)

_RUNTIME_STATE_TABLE = "runtime_state"
_RUNTIME_LOCK_TABLE = "runtime_processing_locks"


class RuntimeStateStore:
    """Persists small runtime state blobs and lock rows in Supabase."""

    def __init__(self) -> None:
        self.storage_mode = get_storage_mode()
        self.db_reads_enabled = should_read_db_first()
        self.db_writes_enabled = should_write_to_db()

    def load_namespace(self, namespace: str) -> dict[str, dict[str, Any]]:
        """Load namespace state as `{state_key: payload}`."""
        if not self.db_reads_enabled:
            return {}
        try:
            client = get_supabase_client()
            result = (
                client.table(_RUNTIME_STATE_TABLE)
                .select("state_key,state")
                .eq("namespace", str(namespace))
                .execute()
            )
            rows = cast(list[dict[str, Any]], result.data or [])
            loaded: dict[str, dict[str, Any]] = {}
            for row in rows:
                key = str(row.get("state_key", "")).strip()
                payload = row.get("state")
                if key and isinstance(payload, dict):
                    loaded[key] = payload
            return loaded
        except Exception as exc:
            if self.db_writes_enabled:
                raise RuntimeError(
                    f"Supabase runtime state read failed for namespace={namespace}: {exc}"
                ) from exc
            logger.warning("Could not read runtime state from Supabase: %s", exc)
            return {}

    def get_record(self, namespace: str, state_key: str) -> dict[str, Any] | None:
        """Load single runtime state record."""
        if not self.db_reads_enabled:
            return None
        try:
            client = get_supabase_client()
            result = (
                client.table(_RUNTIME_STATE_TABLE)
                .select("state")
                .eq("namespace", str(namespace))
                .eq("state_key", str(state_key))
                .limit(1)
                .execute()
            )
            rows = cast(list[dict[str, Any]], result.data or [])
            if not rows:
                return None
            payload = rows[0].get("state")
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            if self.db_writes_enabled:
                raise RuntimeError(
                    f"Supabase runtime state read failed for {namespace}:{state_key}: {exc}"
                ) from exc
            logger.warning("Could not read runtime state record from Supabase: %s", exc)
            return None

    def upsert_record(self, namespace: str, state_key: str, payload: dict[str, Any]) -> None:
        """Upsert runtime state record in Supabase when DB writes are enabled."""
        if not self.db_writes_enabled:
            return
        client = get_supabase_client()
        row = {
            "namespace": str(namespace),
            "state_key": str(state_key),
            "state": payload,
        }
        client.table(_RUNTIME_STATE_TABLE).upsert(
            cast(Any, row),
            on_conflict="namespace,state_key",
        ).execute()

    def upsert_many(self, namespace: str, records: dict[str, dict[str, Any]]) -> None:
        """Bulk-upsert namespace records."""
        if not self.db_writes_enabled:
            return
        if not records:
            return
        client = get_supabase_client()
        rows = [
            {"namespace": str(namespace), "state_key": str(key), "state": value}
            for key, value in records.items()
            if isinstance(value, dict)
        ]
        if not rows:
            return
        client.table(_RUNTIME_STATE_TABLE).upsert(
            cast(Any, rows),
            on_conflict="namespace,state_key",
        ).execute()

    def acquire_lock(self, lock_key: str, owner_id: str, ttl_seconds: int = 900) -> bool:
        """Acquire lock row in Supabase, returning False when held by another worker."""
        if not self.db_writes_enabled:
            return True

        lock_key_str = str(lock_key)
        owner_id_str = str(owner_id)
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=max(30, int(ttl_seconds)))
        now_iso = now.isoformat()

        client = get_supabase_client()

        # Delete stale lock before trying to acquire current lease.
        client.table(_RUNTIME_LOCK_TABLE).delete().eq("lock_key", lock_key_str).lt(
            "expires_at", now_iso
        ).execute()

        try:
            client.table(_RUNTIME_LOCK_TABLE).insert(
                cast(
                    Any,
                    {
                        "lock_key": lock_key_str,
                        "owner_id": owner_id_str,
                        "expires_at": expires_at.isoformat(),
                    },
                )
            ).execute()
            return True
        except Exception as exc:
            error_text = str(exc).lower()
            if "duplicate" in error_text or "unique" in error_text or "conflict" in error_text:
                return False
            raise RuntimeError(f"Could not acquire Supabase lock {lock_key}: {exc}") from exc

    def release_lock(self, lock_key: str, owner_id: str) -> None:
        """Release lock row owned by `owner_id`."""
        if not self.db_writes_enabled:
            return
        client = get_supabase_client()
        client.table(_RUNTIME_LOCK_TABLE).delete().eq("lock_key", str(lock_key)).eq(
            "owner_id", str(owner_id)
        ).execute()

    def delete_records(self, namespace: str, state_keys: list[str]) -> None:
        """Delete runtime state records by namespace/state-key list."""
        if not self.db_writes_enabled:
            return
        normalized_keys = [str(key).strip() for key in state_keys if str(key).strip()]
        if not normalized_keys:
            return
        client = get_supabase_client()
        client.table(_RUNTIME_STATE_TABLE).delete().eq("namespace", str(namespace)).in_(
            "state_key", normalized_keys
        ).execute()
