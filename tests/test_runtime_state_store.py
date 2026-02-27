from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.persistence.runtime_state_store import RuntimeStateStore


def test_load_namespace_returns_empty_when_db_read_disabled(patcher):
    patcher.setattr("src.persistence.runtime_state_store.should_read_db_first", lambda: False)
    patcher.setattr("src.persistence.runtime_state_store.should_write_to_db", lambda: False)
    patcher.setattr("src.persistence.runtime_state_store.get_storage_mode", lambda: "file_only")

    store = RuntimeStateStore()
    assert store.load_namespace("event_boundary_refresh") == {}


def test_load_namespace_reads_supabase_rows(patcher):
    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
    query.execute.return_value = SimpleNamespace(
        data=[
            {"state_key": "2026::Bahrain Grand Prix", "state": {"sessions": ["FP1"]}},
            {"state_key": "2026::Chinese Grand Prix", "state": {"sessions": ["FP1", "SQ"]}},
        ]
    )
    client = MagicMock()
    client.table.return_value = query

    patcher.setattr("src.persistence.runtime_state_store.should_read_db_first", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.should_write_to_db", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.get_storage_mode", lambda: "db_only")
    patcher.setattr("src.persistence.runtime_state_store.get_supabase_client", lambda: client)

    store = RuntimeStateStore()
    loaded = store.load_namespace("practice_characteristics")

    assert loaded["2026::Bahrain Grand Prix"]["sessions"] == ["FP1"]
    assert loaded["2026::Chinese Grand Prix"]["sessions"] == ["FP1", "SQ"]


def test_upsert_record_writes_to_supabase_when_db_enabled(patcher):
    query = MagicMock()
    query.upsert.return_value = query
    query.execute.return_value = SimpleNamespace(data=[{"ok": True}])
    client = MagicMock()
    client.table.return_value = query

    patcher.setattr("src.persistence.runtime_state_store.should_read_db_first", lambda: False)
    patcher.setattr("src.persistence.runtime_state_store.should_write_to_db", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.get_storage_mode", lambda: "db_only")
    patcher.setattr("src.persistence.runtime_state_store.get_supabase_client", lambda: client)

    store = RuntimeStateStore()
    store.upsert_record("event_boundary_refresh", "2026::Bahrain Grand Prix", {"foo": "bar"})

    client.table.assert_called_with("runtime_state")
    query.upsert.assert_called_once()


def test_acquire_lock_returns_false_on_conflict(patcher):
    delete_query = MagicMock()
    delete_query.eq.return_value = delete_query
    delete_query.lt.return_value = delete_query
    delete_query.execute.return_value = SimpleNamespace(data=[])

    insert_query = MagicMock()
    insert_query.insert.return_value = insert_query
    insert_query.execute.side_effect = RuntimeError(
        "duplicate key value violates unique constraint"
    )

    client = MagicMock()
    client.table.side_effect = [delete_query, insert_query]

    patcher.setattr("src.persistence.runtime_state_store.should_read_db_first", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.should_write_to_db", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.get_storage_mode", lambda: "db_only")
    patcher.setattr("src.persistence.runtime_state_store.get_supabase_client", lambda: client)

    store = RuntimeStateStore()
    assert store.acquire_lock("practice_backlog::2026::Bahrain Grand Prix", "owner-1") is False


def test_acquire_lock_raises_on_non_conflict_error(patcher):
    delete_query = MagicMock()
    delete_query.eq.return_value = delete_query
    delete_query.lt.return_value = delete_query
    delete_query.execute.return_value = SimpleNamespace(data=[])

    insert_query = MagicMock()
    insert_query.insert.return_value = insert_query
    insert_query.execute.side_effect = RuntimeError("network closed")

    client = MagicMock()
    client.table.side_effect = [delete_query, insert_query]

    patcher.setattr("src.persistence.runtime_state_store.should_read_db_first", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.should_write_to_db", lambda: True)
    patcher.setattr("src.persistence.runtime_state_store.get_storage_mode", lambda: "db_only")
    patcher.setattr("src.persistence.runtime_state_store.get_supabase_client", lambda: client)

    store = RuntimeStateStore()
    with pytest.raises(RuntimeError, match="Could not acquire Supabase lock"):
        store.acquire_lock("practice_backlog::2026::Bahrain Grand Prix", "owner-1")
