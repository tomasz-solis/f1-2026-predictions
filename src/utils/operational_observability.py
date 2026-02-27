"""Operational counters and alerts for dashboard/runtime health signals."""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter, deque
from datetime import UTC, datetime
from threading import RLock
from typing import Any, cast

from src.persistence.config import should_write_to_db
from src.persistence.db import get_supabase_client

logger = logging.getLogger(__name__)

_EVENTS_TABLE = "operational_events"
_DB_FAILURE_COOLDOWN_SECONDS = 30.0
_MAX_RECENT_ALERTS = 50

_lock = RLock()
_counters: Counter[str] = Counter()
_recent_alerts: deque[dict[str, Any]] = deque(maxlen=_MAX_RECENT_ALERTS)
_db_writes_disabled_until = 0.0


def _normalized_labels(labels: dict[str, Any] | None) -> dict[str, str]:
    if not labels:
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def _write_event(
    *,
    event_type: str,
    event_name: str,
    severity: str,
    message: str,
    labels: dict[str, str],
) -> None:
    global _db_writes_disabled_until
    if not should_write_to_db():
        return

    now_monotonic = time.monotonic()
    if now_monotonic < _db_writes_disabled_until:
        return

    payload = {
        "id": str(uuid.uuid4()),
        "event_type": str(event_type),
        "event_name": str(event_name),
        "severity": str(severity),
        "message": str(message),
        "labels": labels,
        "created_at": datetime.now(UTC).isoformat(),
    }

    try:
        supabase = get_supabase_client()
        supabase.table(_EVENTS_TABLE).insert(cast(Any, payload)).execute()
    except Exception as exc:  # pragma: no cover - network/db edge behavior
        _db_writes_disabled_until = now_monotonic + _DB_FAILURE_COOLDOWN_SECONDS
        logger.warning(
            "Could not write operational event to Supabase table %s: %s",
            _EVENTS_TABLE,
            exc,
        )


def record_counter(
    metric_name: str,
    amount: int = 1,
    labels: dict[str, Any] | None = None,
) -> None:
    """Increment process-level counter and emit best-effort operational event."""
    normalized_labels = _normalized_labels(labels)
    with _lock:
        _counters[str(metric_name)] += int(amount)

    _write_event(
        event_type="counter",
        event_name=str(metric_name),
        severity="info",
        message=f"counter incremented by {amount}",
        labels=normalized_labels,
    )


def record_alert(
    alert_name: str,
    message: str,
    *,
    severity: str = "warning",
    labels: dict[str, Any] | None = None,
) -> None:
    """Record alert for UI visibility and persistence to operational event stream."""
    normalized_labels = _normalized_labels(labels)
    alert_payload = {
        "name": str(alert_name),
        "severity": str(severity),
        "message": str(message),
        "labels": normalized_labels,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with _lock:
        _recent_alerts.append(alert_payload)

    log_message = f"[{alert_payload['name']}] {alert_payload['message']}"
    if severity.lower() == "error":
        logger.error(log_message)
    else:
        logger.warning(log_message)

    _write_event(
        event_type="alert",
        event_name=str(alert_name),
        severity=str(severity),
        message=str(message),
        labels=normalized_labels,
    )


def snapshot_counters() -> dict[str, int]:
    """Return a copy of current process-level counters."""
    with _lock:
        return {key: int(value) for key, value in _counters.items()}


def drain_recent_alerts(limit: int = 20) -> list[dict[str, Any]]:
    """Drain and return recent alerts for one-shot UI rendering."""
    with _lock:
        drained = list(_recent_alerts)[-max(1, int(limit)) :]
        _recent_alerts.clear()
    return drained
