from __future__ import annotations

import importlib
import sys

import pytest

MODULE_PATH = "src.persistence.config"


def _load_config_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    storage_mode: str | None = None,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
):
    """Import config module under a controlled env snapshot."""
    if storage_mode is None:
        monkeypatch.delenv("USE_DB_STORAGE", raising=False)
    else:
        monkeypatch.setenv("USE_DB_STORAGE", storage_mode)

    if supabase_url is None:
        monkeypatch.delenv("SUPABASE_URL", raising=False)
    else:
        monkeypatch.setenv("SUPABASE_URL", supabase_url)

    if supabase_key is None:
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
    else:
        monkeypatch.setenv("SUPABASE_KEY", supabase_key)

    sys.modules.pop(MODULE_PATH, None)
    return importlib.import_module(MODULE_PATH)


def test_defaults_to_file_only(monkeypatch):
    config = _load_config_module(monkeypatch)
    assert config.USE_DB_STORAGE == "file_only"
    assert config.is_db_enabled() is False


def test_storage_mode_is_normalized(monkeypatch):
    config = _load_config_module(
        monkeypatch,
        storage_mode="  DB_ONLY  ",
        supabase_url="https://example.supabase.co",
        supabase_key="secret",
    )
    assert config.USE_DB_STORAGE == "db_only"


def test_invalid_storage_mode_raises(monkeypatch):
    with pytest.raises(ValueError, match="Invalid USE_DB_STORAGE value"):
        _load_config_module(monkeypatch, storage_mode="wrong_mode")


def test_db_mode_requires_supabase_url(monkeypatch):
    with pytest.raises(ValueError, match="SUPABASE_URL environment variable is required"):
        _load_config_module(monkeypatch, storage_mode="fallback", supabase_key="secret")


def test_db_mode_rejects_non_https_url(monkeypatch):
    with pytest.raises(ValueError, match="must start with 'https://'"):
        _load_config_module(
            monkeypatch,
            storage_mode="db_only",
            supabase_url="ttps://example.supabase.co",
            supabase_key="secret",
        )


def test_db_mode_rejects_missing_hostname(monkeypatch):
    with pytest.raises(ValueError, match="missing a hostname"):
        _load_config_module(
            monkeypatch,
            storage_mode="db_only",
            supabase_url="https://",
            supabase_key="secret",
        )


def test_db_mode_requires_supabase_key(monkeypatch):
    with pytest.raises(ValueError, match="SUPABASE_KEY environment variable is required"):
        _load_config_module(
            monkeypatch,
            storage_mode="db_only",
            supabase_url="https://example.supabase.co",
        )


def test_file_only_logs_when_supabase_creds_are_present(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        _load_config_module(
            monkeypatch,
            storage_mode="file_only",
            supabase_url="https://example.supabase.co",
            supabase_key="secret",
        )
    assert "credentials are ignored" in caplog.text


def test_file_only_logs_invalid_supabase_url(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        _load_config_module(
            monkeypatch,
            storage_mode="file_only",
            supabase_url="ttps://example.supabase.co",
            supabase_key="secret",
        )
    assert "looks invalid" in caplog.text


@pytest.mark.parametrize(
    ("storage_mode", "expected"),
    [
        ("file_only", False),
        ("fallback", True),
        ("db_only", True),
        ("dual_write", True),
    ],
)
def test_should_read_db_first_by_storage_mode(monkeypatch, storage_mode, expected):
    config = _load_config_module(
        monkeypatch,
        storage_mode=storage_mode,
        supabase_url="https://example.supabase.co" if storage_mode != "file_only" else None,
        supabase_key="secret" if storage_mode != "file_only" else None,
    )
    assert config.should_read_db_first() is expected
