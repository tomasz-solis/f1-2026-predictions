"""Server-side event logging to Supabase.

Events are best-effort and never raise into the UI. Session IDs are short
hashed UUIDs scoped to the Streamlit session_state; they reset when the
browser tab closes. No IP, no email, and no raw user-agent is stored.

UTM parameters (utm_source, utm_medium, utm_campaign, utm_content,
utm_term) are captured once on the first event of a session and pinned
to every subsequent event in that session, giving per-session attribution.
"""

import logging
import uuid
from typing import Any

import streamlit as st

from src.persistence.config import is_db_enabled
from src.persistence.db import get_supabase_client
from src.utils.model_version import format_model_version_label

logger = logging.getLogger(__name__)

_SESSION_KEY = "_analytics_session_id"
_LAST_PAGE_KEY = "_analytics_last_page"
_CLIENT_CHANNEL_KEY = "_analytics_client_channel"
_UTM_KEY = "_analytics_utm"
_TABLE = "app_events"

_UTM_FIELDS = ("utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term")
_MAX_PAYLOAD_KEYS = 32
_MAX_PAYLOAD_KEY_LENGTH = 64
_MAX_PAYLOAD_STRING_LENGTH = 512
_MAX_PAYLOAD_LIST_ITEMS = 20


def _session_id() -> str:
    """Return a stable per-tab session identifier, generating one if needed."""
    sid = st.session_state.get(_SESSION_KEY)
    if not sid:
        sid = uuid.uuid4().hex[:16]
        st.session_state[_SESSION_KEY] = sid
    return sid


def _classify_user_agent(user_agent: str | None) -> str | None:
    """Classify a raw user-agent into a coarse non-identifying channel."""
    if not user_agent:
        return None
    lowered = user_agent.lower()
    if "instagram" in lowered:
        return "instagram_in_app"
    if "threads" in lowered:
        return "threads_in_app"
    if "fbav" in lowered or "fban" in lowered or "facebook" in lowered:
        return "facebook_in_app"
    return "browser"


def _client_channel() -> str | None:
    """Best-effort coarse client-channel capture from Streamlit's request context.

    Cached per session because st.context is read-only and cheap, but no
    point hitting it on every event. The raw user-agent never leaves this
    function.
    """
    cached = st.session_state.get(_CLIENT_CHANNEL_KEY)
    if cached is not None:
        return cached or None

    ua: str | None = None
    try:
        headers = getattr(st.context, "headers", None)
        if headers is not None:
            ua = headers.get("User-Agent") or headers.get("user-agent")
    except Exception:  # st.context not available or behaves unexpectedly
        ua = None

    channel = _classify_user_agent(ua)
    st.session_state[_CLIENT_CHANNEL_KEY] = channel or ""
    return channel


def _sanitize_payload_value(value: Any, *, depth: int = 0) -> Any:
    """Bound telemetry payload values to simple JSON-compatible shapes."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_PAYLOAD_STRING_LENGTH]
    if depth >= 2:
        return str(value)[:_MAX_PAYLOAD_STRING_LENGTH]
    if isinstance(value, dict):
        return {
            str(key)[:_MAX_PAYLOAD_KEY_LENGTH]: _sanitize_payload_value(
                nested_value,
                depth=depth + 1,
            )
            for key, nested_value in list(value.items())[:_MAX_PAYLOAD_KEYS]
        }
    if isinstance(value, list | tuple):
        return [
            _sanitize_payload_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_PAYLOAD_LIST_ITEMS]
        ]
    return str(value)[:_MAX_PAYLOAD_STRING_LENGTH]


def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded payload suitable for analytics storage."""
    return {
        str(key)[:_MAX_PAYLOAD_KEY_LENGTH]: _sanitize_payload_value(value)
        for key, value in list(payload.items())[:_MAX_PAYLOAD_KEYS]
    }


def _capture_utm() -> dict[str, str]:
    """Read UTM params from the URL once and pin them to session_state.

    Subsequent calls return the cached dict. Empty dict if no UTMs were
    present on the first hit (direct visit, refresh without query string,
    etc.). We freeze on first read so a user navigating internally can't
    overwrite their original attribution.
    """
    cached = st.session_state.get(_UTM_KEY)
    if cached is not None:
        return cached

    captured: dict[str, str] = {}
    try:
        params = st.query_params
        for field in _UTM_FIELDS:
            value = params.get(field)
            if isinstance(value, list):
                value = value[0] if value else None
            if value:
                # Cap length defensively; UTMs are short by convention.
                captured[field] = str(value)[:128]
    except Exception as exc:
        logger.warning("UTM capture failed: %s", exc)

    st.session_state[_UTM_KEY] = captured
    return captured


def track_event(event_type: str, **payload: Any) -> None:
    """Insert one row into app_events. Silent on failure.

    Special kwargs: ``page`` (str) is promoted to its own column.
    Everything else lands in the JSONB ``payload`` column. UTM params
    captured at session start are merged into the payload automatically.
    """
    if not is_db_enabled():
        return

    page = payload.pop("page", None)
    utm = _capture_utm()
    merged_payload = _sanitize_payload({**utm, **payload})

    record = {
        "session_id": _session_id(),
        "event_type": event_type,
        "page": page,
        "payload": merged_payload,
        "model_version": format_model_version_label(),
        "user_agent": _client_channel(),
    }

    try:
        get_supabase_client().table(_TABLE).insert(record).execute()
    except Exception as exc:  # never break the UI for telemetry
        logger.warning("track_event(%s) failed: %s", event_type, exc)


def track_page_view(page: str) -> None:
    """Log a page_view, but only when the active page actually changes.

    Streamlit reruns on every interaction, so without this dedup we'd log
    a page_view per widget click. Dedup is per-tab via session_state.
    """
    if st.session_state.get(_LAST_PAGE_KEY) == page:
        return
    st.session_state[_LAST_PAGE_KEY] = page
    track_event("page_view", page=page)
