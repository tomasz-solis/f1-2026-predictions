"""Supabase client access for database-backed storage."""

import logging

from postgrest.exceptions import APIError
from supabase import Client, create_client

from .config import get_supabase_key, get_supabase_url, is_db_enabled

logger = logging.getLogger(__name__)

_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """Return the shared Supabase client for this process."""
    global _supabase_client

    if not is_db_enabled():
        raise RuntimeError(
            "Database storage is not enabled. Set USE_DB_STORAGE to 'db_only', 'fallback', or 'dual_write'"
        )

    if _supabase_client is None:
        supabase_url = get_supabase_url()
        supabase_key = get_supabase_key()
        if not supabase_url or not supabase_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

        try:
            _supabase_client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized: %s", supabase_url)
        except (APIError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error("Failed to initialize Supabase client: %s", e)
            raise RuntimeError(f"Failed to connect to Supabase: {e}") from e

    return _supabase_client


def check_connection() -> str:
    """Run a simple read to confirm the Supabase connection works."""
    try:
        client = get_supabase_client()
        result = client.table("artifacts").select("id").limit(1).execute()
        return f"Supabase connection healthy ({len(result.data)} row(s) accessible)"
    except (APIError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
        raise RuntimeError(f"Supabase connection failed: {e}") from e


def close_client() -> None:
    """Clear the cached client so tests or shutdown paths can start fresh."""
    global _supabase_client
    if _supabase_client is not None:
        _supabase_client = None
        logger.info("Supabase client closed")
