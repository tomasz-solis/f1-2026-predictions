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

# Storage mode configuration
USE_DB_STORAGE = (os.getenv("USE_DB_STORAGE") or "file_only").strip().lower()
StorageMode = Literal["file_only", "db_only", "fallback", "dual_write"]

# Validate storage mode
VALID_MODES: set[str] = {"file_only", "db_only", "fallback", "dual_write"}
if USE_DB_STORAGE not in VALID_MODES:
    raise ValueError(
        f"Invalid USE_DB_STORAGE value: {USE_DB_STORAGE}. Must be one of: {', '.join(sorted(VALID_MODES))}"
    )

# Supabase credentials
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip() or None
SUPABASE_KEY = (os.getenv("SUPABASE_KEY") or "").strip() or None


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


# Validate credentials if DB storage is enabled
if USE_DB_STORAGE != "file_only":
    if not SUPABASE_URL:
        raise ValueError(
            f"SUPABASE_URL environment variable is required when USE_DB_STORAGE={USE_DB_STORAGE}"
        )
    _validate_supabase_url(SUPABASE_URL)
    if not SUPABASE_KEY:
        raise ValueError(
            f"SUPABASE_KEY environment variable is required when USE_DB_STORAGE={USE_DB_STORAGE}"
        )
elif SUPABASE_URL:
    # This catches accidental mode drift in deployment where DB creds exist but mode was left file_only.
    try:
        _validate_supabase_url(SUPABASE_URL)
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


def get_storage_mode() -> str:
    """Get current storage mode."""
    return USE_DB_STORAGE


def is_db_enabled() -> bool:
    """Check if database storage is enabled."""
    return USE_DB_STORAGE != "file_only"


def is_file_enabled() -> bool:
    """Check if file storage is enabled."""
    return USE_DB_STORAGE != "db_only"


def should_write_to_db() -> bool:
    """Check if writes should go to database."""
    return USE_DB_STORAGE in ("db_only", "fallback", "dual_write")


def should_write_to_file() -> bool:
    """Check if writes should go to files."""
    return USE_DB_STORAGE in ("file_only", "dual_write")


def should_read_db_first() -> bool:
    """Check if reads should try database first."""
    return USE_DB_STORAGE in ("db_only", "fallback", "dual_write")
