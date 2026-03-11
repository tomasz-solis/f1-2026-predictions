"""
Configuration for persistence layer.

Environment variables:
- USE_DB_STORAGE: 'file_only', 'db_only', 'fallback', 'dual_write'
- SUPABASE_URL: Supabase project URL (https://xxx.supabase.co)
- SUPABASE_KEY: Supabase service-role key (required for write-capable modes)
"""

import logging
import os
from typing import Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

StorageMode = Literal["file_only", "db_only", "fallback", "dual_write"]

# Validate storage mode
VALID_MODES: set[str] = {"file_only", "db_only", "fallback", "dual_write"}


def _validate_supabase_url(url: str) -> None:
    """Validate that SUPABASE_URL is a valid HTTPS URL."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        hint = ""
        if url.startswith("ttps://"):
            hint = " Did you mean 'https://...'? "
        elif not parsed.scheme:
            hint = " Include the scheme, e.g. 'https://...'."
        raise ValueError(
            f"SUPABASE_URL must start with 'https://'. Received: {url!r}.{hint}".strip()
        )
    if not parsed.netloc:
        raise ValueError(f"SUPABASE_URL is missing a hostname: {url!r}")


def _resolve_storage_mode() -> str:
    """Read and validate the active storage mode from the current environment."""
    storage_mode = (os.getenv("USE_DB_STORAGE") or "file_only").strip().lower()
    if storage_mode not in VALID_MODES:
        raise ValueError(
            "Invalid USE_DB_STORAGE value: "
            f"{storage_mode}. Must be one of: {', '.join(sorted(VALID_MODES))}"
        )
    return storage_mode


def get_supabase_url() -> str | None:
    """Return the configured Supabase URL from the current environment."""
    return (os.getenv("SUPABASE_URL") or "").strip() or None


def get_supabase_key() -> str | None:
    """Return the configured Supabase service key from the current environment."""
    return (os.getenv("SUPABASE_KEY") or "").strip() or None


def _validate_current_environment() -> None:
    """Validate persistence-related environment variables once at import time."""
    storage_mode = _resolve_storage_mode()
    supabase_url = get_supabase_url()
    supabase_key = get_supabase_key()

    if storage_mode != "file_only":
        if not supabase_url:
            raise ValueError(
                f"SUPABASE_URL environment variable is required when USE_DB_STORAGE={storage_mode}"
            )
        _validate_supabase_url(supabase_url)
        if not supabase_key:
            raise ValueError(
                f"SUPABASE_KEY environment variable is required when USE_DB_STORAGE={storage_mode}"
            )
    elif supabase_url:
        try:
            _validate_supabase_url(supabase_url)
        except ValueError as exc:
            logger.warning(
                "USE_DB_STORAGE=file_only and SUPABASE_URL looks invalid: %s "
                "Set USE_DB_STORAGE to db_only/fallback/dual_write after fixing env vars.",
                exc,
            )
        else:
            logger.warning(
                "USE_DB_STORAGE=file_only, so Supabase credentials are ignored. "
                "Set USE_DB_STORAGE to db_only/fallback/dual_write to enable DB persistence."
            )


_validate_current_environment()

# Backward-compatible snapshots of the env at import time. Runtime code should
# prefer the getter helpers above so tests and tools can override env safely.
USE_DB_STORAGE = _resolve_storage_mode()
SUPABASE_URL = get_supabase_url()
SUPABASE_KEY = get_supabase_key()


def get_storage_mode() -> str:
    """Get current storage mode."""
    return _resolve_storage_mode()


def is_db_enabled() -> bool:
    """Check if database storage is enabled."""
    return get_storage_mode() != "file_only"


def is_file_enabled() -> bool:
    """Check if file storage is enabled."""
    return get_storage_mode() != "db_only"


def should_write_to_db() -> bool:
    """Check if writes should go to database."""
    return get_storage_mode() in ("db_only", "fallback", "dual_write")


def should_write_to_file() -> bool:
    """Check if writes should go to files."""
    return get_storage_mode() in ("file_only", "dual_write")


def should_read_db_first() -> bool:
    """Check if reads should try database first."""
    return get_storage_mode() in ("db_only", "fallback", "dual_write")
