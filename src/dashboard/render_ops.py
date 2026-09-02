"""Render platform calls the operator panel needs: precompute now, restart the service.

Every deploy that touches a file in ``_PREDICTION_CODE_FINGERPRINT_FILES`` moves the
artifact hash, so the warmed predictions no longer match and the dashboard has nothing to
serve until the scheduled ``preheat`` cron next runs. Recovering meant opening the Render
dashboard and clicking two buttons; these two calls put them on the admin page instead.

The precompute deliberately does *not* run in this process. The web service is on the free
plan and the cron on starter — loading the predictor and simulating a weekend here is how
the web process runs out of memory. The cron instance does the work; this only starts it.

Nothing raises. A missing environment variable or an unreachable API returns
``(False, message)``, because the admin page must still render on a laptop with no Render
credentials at all.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

RENDER_API_BASE = "https://api.render.com/v1"
_API_KEY_VAR = "RENDER_API_KEY"
_CRON_ID_VAR = "RENDER_PRECOMPUTE_CRON_ID"
_SERVICE_ID_VAR = "RENDER_WEB_SERVICE_ID"
_REQUIRED_VARS = (_API_KEY_VAR, _CRON_ID_VAR, _SERVICE_ID_VAR)
_TIMEOUT_SECONDS = 15


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return env if env is not None else os.environ


def missing_settings(env: Mapping[str, str] | None = None) -> list[str]:
    """Return the names of the Render variables that are not set."""
    source = _env(env)
    return [name for name in _REQUIRED_VARS if not str(source.get(name, "")).strip()]


def render_ops_configured(env: Mapping[str, str] | None = None) -> bool:
    """Return whether this service can talk to the Render API."""
    return not missing_settings(env)


def trigger_precompute_run(env: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Start an unscheduled run of the precompute cron job.

    Render runs one instance of a cron job at a time, so this cancels an in-flight run
    before starting the new one.
    """
    source = _env(env)
    cron_id = str(source.get(_CRON_ID_VAR, "")).strip()
    return _post(
        f"{RENDER_API_BASE}/cron-jobs/{cron_id}/runs",
        env=env,
        needs=(_API_KEY_VAR, _CRON_ID_VAR),
        success_message="Precompute run started. It takes a few minutes; reload this page to see the new horizon.",
    )


def restart_web_service(env: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Restart the dashboard web service."""
    source = _env(env)
    service_id = str(source.get(_SERVICE_ID_VAR, "")).strip()
    return _post(
        f"{RENDER_API_BASE}/services/{service_id}/restart",
        env=env,
        needs=(_API_KEY_VAR, _SERVICE_ID_VAR),
        success_message="Restart requested. The dashboard drops for about a minute while it comes back.",
    )


# A cron job is a service in Render's API, so its runs are read from the shared events
# feed rather than from a cron-specific endpoint (there is no GET for cron runs).
_CRON_EVENT_TYPES = ("cron_job_run_started", "cron_job_run_ended")
_WEB_EVENT_TYPES = ("server_restarted", "deploy_started", "deploy_ended")


def precompute_run_events(
    env: Mapping[str, str] | None = None, *, limit: int = 5
) -> tuple[list[dict[str, str]], str]:
    """Return the precompute cron's recent starts and finishes, newest first."""
    source = _env(env)
    return _events(
        str(source.get(_CRON_ID_VAR, "")).strip(),
        env=env,
        needs=(_API_KEY_VAR, _CRON_ID_VAR),
        types=_CRON_EVENT_TYPES,
        limit=limit,
    )


def web_service_events(
    env: Mapping[str, str] | None = None, *, limit: int = 5
) -> tuple[list[dict[str, str]], str]:
    """Return the dashboard service's recent restarts and deploys, newest first."""
    source = _env(env)
    return _events(
        str(source.get(_SERVICE_ID_VAR, "")).strip(),
        env=env,
        needs=(_API_KEY_VAR, _SERVICE_ID_VAR),
        types=_WEB_EVENT_TYPES,
        limit=limit,
    )


def _post(
    url: str,
    *,
    env: Mapping[str, str] | None,
    needs: tuple[str, ...],
    success_message: str,
) -> tuple[bool, str]:
    """POST to the Render API, reporting every failure as a message instead of raising."""
    source = _env(env)
    absent = _absent(source, needs)
    if absent:
        return False, f"Not configured on this service: {', '.join(absent)}."

    requests = _requests_module()
    if requests is None:
        return False, "The requests package is unavailable, so the Render API cannot be called."

    try:
        response = requests.post(url, headers=_headers(source), timeout=_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - a network failure must not break the panel
        logger.warning("Render API call to %s failed: %s", url, exc)
        return False, f"Could not reach the Render API: {exc}"

    status_code = int(getattr(response, "status_code", 0) or 0)
    if 200 <= status_code < 300:
        return True, success_message

    detail = str(getattr(response, "text", "") or "").strip()[:300]
    logger.warning("Render API call to %s returned %s: %s", url, status_code, detail)
    return False, f"Render API returned {status_code}. {detail}".strip()


def _events(
    service_id: str,
    *,
    env: Mapping[str, str] | None,
    needs: tuple[str, ...],
    types: tuple[str, ...],
    limit: int,
) -> tuple[list[dict[str, str]], str]:
    """GET one service's recent events, returning ``(rows, error_message)``."""
    source = _env(env)
    absent = _absent(source, needs)
    if absent:
        return [], f"Not configured on this service: {', '.join(absent)}."

    requests = _requests_module()
    if requests is None:
        return [], "The requests package is unavailable, so the Render API cannot be called."

    try:
        response = requests.get(
            f"{RENDER_API_BASE}/services/{service_id}/events",
            headers=_headers(source),
            params={"type": list(types), "limit": max(1, int(limit))},
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - a network failure must not break the panel
        logger.warning("Render event lookup for %s failed: %s", service_id, exc)
        return [], f"Could not reach the Render API: {exc}"

    status_code = int(getattr(response, "status_code", 0) or 0)
    if not 200 <= status_code < 300:
        detail = str(getattr(response, "text", "") or "").strip()[:200]
        logger.warning("Render event lookup returned %s: %s", status_code, detail)
        return [], f"Render API returned {status_code}. {detail}".strip()

    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - a malformed body is not worth a traceback
        logger.warning("Render event lookup returned unreadable JSON: %s", exc)
        return [], "The Render API returned a response this page could not read."

    return [_event_row(item) for item in _unwrap_events(payload)], ""


def _unwrap_events(payload: Any) -> list[dict[str, Any]]:
    """Pull event dicts out of Render's list envelope, tolerating a bare list."""
    if not isinstance(payload, list):
        return []
    events: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        event = item.get("event", item)
        if isinstance(event, dict):
            events.append(event)
    return events


def _event_row(event: dict[str, Any]) -> dict[str, str]:
    """Flatten one event into the timestamp, label, and outcome the panel shows."""
    details = event.get("details")
    details = details if isinstance(details, dict) else {}
    status = str(details.get("status", "") or "").strip()

    reason = details.get("reason")
    causes = [name for name, value in reason.items() if value] if isinstance(reason, dict) else []

    return {
        "timestamp": str(event.get("timestamp", "") or "").replace("T", " ")[:16],
        "type": str(event.get("type", "") or "").replace("_", " ").strip(),
        "outcome": ", ".join(part for part in [status, ", ".join(sorted(causes))] if part),
    }


def _headers(source: Mapping[str, str]) -> dict[str, str]:
    """Build the Render API auth headers."""
    return {
        "Authorization": f"Bearer {str(source.get(_API_KEY_VAR, '')).strip()}",
        "Accept": "application/json",
    }


def _absent(source: Mapping[str, str], needs: tuple[str, ...]) -> list[str]:
    """Return the required variables that carry no value."""
    return [name for name in needs if not str(source.get(name, "")).strip()]


def _requests_module() -> Any:
    """Import requests lazily so a public page load never pays for it."""
    try:
        # types-requests is not a dependency here; the module is used through Any anyway.
        import requests  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - requests is a pinned dependency
        logger.warning("requests is not importable: %s", exc)
        return None
    return requests
