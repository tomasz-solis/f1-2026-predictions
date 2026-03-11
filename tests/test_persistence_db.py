from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.persistence.db as db_module


def test_get_supabase_client_raises_when_db_disabled(patcher):
    patcher.setattr(db_module, "_supabase_client", None)
    patcher.setattr(db_module, "is_db_enabled", lambda: False)

    with pytest.raises(RuntimeError, match="not enabled"):
        db_module.get_supabase_client()


def test_get_supabase_client_raises_when_credentials_missing(patcher):
    patcher.setattr(db_module, "_supabase_client", None)
    patcher.setattr(db_module, "is_db_enabled", lambda: True)
    patcher.setattr(db_module, "get_supabase_url", lambda: None)
    patcher.setattr(db_module, "get_supabase_key", lambda: None)

    with pytest.raises(RuntimeError, match="environment variables must be set"):
        db_module.get_supabase_client()


def test_get_supabase_client_initializes_once(patcher):
    created = object()
    create_client = MagicMock(return_value=created)

    patcher.setattr(db_module, "_supabase_client", None)
    patcher.setattr(db_module, "is_db_enabled", lambda: True)
    patcher.setattr(db_module, "get_supabase_url", lambda: "https://example.supabase.co")
    patcher.setattr(db_module, "get_supabase_key", lambda: "secret")
    patcher.setattr(db_module, "create_client", create_client)

    first = db_module.get_supabase_client()
    second = db_module.get_supabase_client()

    assert first is created
    assert second is created
    create_client.assert_called_once_with("https://example.supabase.co", "secret")


def test_get_supabase_client_wraps_creation_errors(patcher):
    patcher.setattr(db_module, "_supabase_client", None)
    patcher.setattr(db_module, "is_db_enabled", lambda: True)
    patcher.setattr(db_module, "get_supabase_url", lambda: "https://example.supabase.co")
    patcher.setattr(db_module, "get_supabase_key", lambda: "secret")
    patcher.setattr(
        db_module,
        "create_client",
        lambda url, key: (_ for _ in ()).throw(RuntimeError("connect fail")),
    )

    with pytest.raises(RuntimeError, match="Failed to connect to Supabase"):
        db_module.get_supabase_client()


def test_check_connection_success(patcher):
    query = MagicMock()
    query.select.return_value = query
    query.limit.return_value = query
    query.execute.return_value = SimpleNamespace(data=[{"id": 1}])

    client = MagicMock()
    client.table.return_value = query

    patcher.setattr(db_module, "get_supabase_client", lambda: client)

    message = db_module.check_connection()
    assert "healthy" in message


def test_check_connection_failure(patcher):
    patcher.setattr(
        db_module,
        "get_supabase_client",
        lambda: (_ for _ in ()).throw(RuntimeError("auth failed")),
    )

    with pytest.raises(RuntimeError, match="auth failed"):
        db_module.check_connection()


def test_close_client_resets_singleton(patcher):
    patcher.setattr(db_module, "_supabase_client", object())
    db_module.close_client()

    assert db_module._supabase_client is None
