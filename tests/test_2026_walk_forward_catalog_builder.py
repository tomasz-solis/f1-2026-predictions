"""Tests for the 2026 walk-forward event catalog builder.

These tests avoid live FastF1 session loads: the sprint-detection helper is pure, the
explicit-exclusion path returns before any session load, and the fail-closed paths are
exercised by monkeypatching the fetch helpers the script calls.
"""

from __future__ import annotations

from typing import Any

import scripts.build_2026_walk_forward_catalog as catalog_builder


def test_detect_session_kind_flags_sprint_qualifying_as_sprint() -> None:
    assert catalog_builder.detect_session_kind("sprint_qualifying") == "sprint"
    assert catalog_builder.detect_session_kind("SPRINT_SHOOTOUT") == "sprint"


def test_detect_session_kind_flags_conventional_as_main() -> None:
    assert catalog_builder.detect_session_kind("conventional") == "main"
    assert catalog_builder.detect_session_kind("") == "main"


def test_explicitly_excluded_event_is_excluded_before_any_session_load() -> None:
    """The in-progress weekend must be excluded without touching FastF1 at all."""
    row: dict[str, Any] = {
        "EventName": "Belgian Grand Prix",
        "RoundNumber": 10,
        "EventFormat": "conventional",
    }
    event_row, diagnostics = catalog_builder._build_event_row(row)
    assert event_row is None
    assert diagnostics["status"] == "excluded"
    assert diagnostics["reason"] == "in_progress_weekend_excluded_by_run_date"


def test_missing_qualifying_results_fail_closed_never_fabricated(monkeypatch) -> None:
    """A row that cannot source qualifying classification must be skipped, not invented."""

    class _FakeEvent:
        def get_session_date(self, _label: str, utc: bool = True):
            from datetime import UTC, datetime

            return datetime(2026, 5, 1, tzinfo=UTC)

    monkeypatch.setattr(catalog_builder.fastf1, "get_event", lambda year, race_name: _FakeEvent())
    monkeypatch.setattr(catalog_builder, "fetch_actual_session_results", lambda *a, **k: None)

    row: dict[str, Any] = {
        "EventName": "Test Grand Prix",
        "RoundNumber": 99,
        "EventFormat": "conventional",
    }
    event_row, diagnostics = catalog_builder._build_event_row(row)
    assert event_row is None
    assert diagnostics["status"] == "skipped"
    assert diagnostics["reason"] == "missing_or_incomplete_qualifying_results"


def test_missing_race_results_fail_closed_after_qualifying_succeeds(monkeypatch) -> None:
    """Qualifying can succeed while the race is still incomplete; must still fail closed."""

    class _FakeEvent:
        def get_session_date(self, _label: str, utc: bool = True):
            from datetime import UTC, datetime

            return datetime(2026, 5, 1, tzinfo=UTC)

    grid = [{"driver": "D1", "team": "T1", "position": 1}]

    def fake_fetch(year, race_name, session_name):
        return grid if session_name == "Q" else None

    monkeypatch.setattr(catalog_builder.fastf1, "get_event", lambda year, race_name: _FakeEvent())
    monkeypatch.setattr(catalog_builder, "fetch_actual_session_results", fake_fetch)

    row: dict[str, Any] = {
        "EventName": "Test Grand Prix",
        "RoundNumber": 99,
        "EventFormat": "conventional",
    }
    event_row, diagnostics = catalog_builder._build_event_row(row)
    assert event_row is None
    assert diagnostics["status"] == "skipped"
    assert diagnostics["reason"] == "missing_or_incomplete_race_results"
