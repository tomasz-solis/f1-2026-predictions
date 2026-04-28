"""Server-side event logging to Supabase.

Events are best-effort and never raise into the UI. Session IDs are short
hashed UUIDs scoped to the Streamlit session_state; they reset when the
browser tab closes. No IP, no email, no other PII is stored.

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
_USER_AGENT_KEY = "_analytics_user_agent"
_UTM_KEY = "_analytics_utm"
_TABLE = "app_events"

_UTM_FIELDS = ("utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term")


def _session_id() -> str:
    """Return a stable per-tab session identifier, generating one if needed."""
    sid = st.session_state.get(_SESSION_KEY)
    if not sid:
        sid = uuid.uuid4().hex[:16]
        st.session_state[_SESSION_KEY] = sid
    return sid


def _user_agent() -> str | None:
    """Best-effort user-agent capture from Streamlit's request context.

    Cached per session because st.context is read-only and cheap, but no
    point hitting it on every event. Returns None on older Streamlit
    versions where st.context is unavailable.
    """
    cached = st.session_state.get(_USER_AGENT_KEY)
    if cached is not None:
        return cached or None

    ua: str | None = None
    try:
        headers = getattr(st.context, "headers", None)
        if headers is not None:
            ua = headers.get("User-Agent") or headers.get("user-agent")
    except Exception:  # st.context not available or behaves unexpectedly
        ua = None

    st.session_state[_USER_AGENT_KEY] = ua or ""
    return ua


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
    merged_payload: dict[str, Any] = {**utm, **payload}

    record = {
        "session_id": _session_id(),
        "event_type": event_type,
        "page": page,
        "payload": merged_payload,
        "model_version": format_model_version_label(),
        "user_agent": _user_agent(),
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
