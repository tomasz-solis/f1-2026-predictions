from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import src.utils.operational_observability as observability


def test_record_counter_updates_snapshot_when_db_disabled(patcher):
    patcher.setattr(observability, "should_write_to_db", lambda: False)
    observability._counters.clear()

    observability.record_counter("fastf1_retry_attempt_total")

    snapshot = observability.snapshot_counters()
    assert snapshot["fastf1_retry_attempt_total"] == 1


def test_record_alert_is_drained_for_ui(patcher):
    patcher.setattr(observability, "should_write_to_db", lambda: False)
    observability._recent_alerts.clear()

    observability.record_alert("fastf1_circuit_trip", "Circuit opened")
    alerts = observability.drain_recent_alerts()

    assert len(alerts) == 1
    assert alerts[0]["name"] == "fastf1_circuit_trip"
    assert observability.drain_recent_alerts() == []


def test_record_counter_writes_event_to_supabase_when_enabled(patcher):
    query = MagicMock()
    query.insert.return_value = query
    query.execute.return_value = SimpleNamespace(data=[{"id": "row"}])
    client = MagicMock()
    client.table.return_value = query

    patcher.setattr(observability, "should_write_to_db", lambda: True)
    patcher.setattr(observability, "get_supabase_client", lambda: client)
    patcher.setattr(observability, "_db_writes_disabled_until", 0.0)

    observability.record_counter("fastf1_call_failure_total", labels={"operation": "get_event"})

    client.table.assert_called_with("operational_events")
    query.insert.assert_called_once()
