"""Tests for dashboard prediction-horizon helpers."""

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.dashboard import prediction_horizon


def test_parse_refresh_timestamp_normalizes_naive_and_aware_values():
    naive = prediction_horizon.parse_refresh_timestamp("2026-03-11T09:16:00")
    aware = prediction_horizon.parse_refresh_timestamp("2026-03-11T09:16:00+02:00")

    assert naive == datetime(2026, 3, 11, 9, 16, tzinfo=UTC)
    assert aware == datetime(2026, 3, 11, 7, 16, tzinfo=UTC)


def test_resolve_dashboard_race_horizon_uses_next_competitive_window():
    now_utc = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    schedule_rows = (
        ("Pre-Season Testing", "testing", "2026-02-20T10:00:00+00:00"),
        ("Australian Grand Prix", "conventional", (now_utc - timedelta(days=4)).isoformat()),
        ("Chinese Grand Prix", "sprint", (now_utc + timedelta(days=3)).isoformat()),
        ("Japanese Grand Prix", "conventional", (now_utc + timedelta(days=10)).isoformat()),
        ("Miami Grand Prix", "sprint", (now_utc + timedelta(days=17)).isoformat()),
    )

    planned_races = prediction_horizon.resolve_dashboard_race_horizon(
        schedule_rows=schedule_rows,
        horizon_races=2,
        now_utc=now_utc,
    )

    assert planned_races == [
        "Chinese Grand Prix",
        "Japanese Grand Prix",
    ]


def test_load_schedule_event_rows_prefers_race_session_cutoff_over_midnight_event_date():
    race_start = datetime(2026, 6, 14, 13, 0, tzinfo=UTC)
    schedule = pd.DataFrame(
        {
            "EventName": ["Barcelona Grand Prix"],
            "EventFormat": ["conventional"],
            "EventDate": [datetime(2026, 6, 14, 0, 0, tzinfo=UTC)],
            "Session5DateUtc": [race_start],
        }
    )

    rows = prediction_horizon.load_schedule_event_rows(
        2026,
        get_event_schedule_fn=lambda year: schedule,
        fallback_schedule_rows_fn=lambda year: (),
        logger=logging.getLogger(__name__),
    )

    assert rows == (
        (
            "Barcelona Grand Prix",
            "conventional",
            (race_start + timedelta(hours=4)).isoformat(),
        ),
    )


def test_resolve_dashboard_race_horizon_keeps_current_gp_until_race_window_closes():
    now_utc = datetime(2026, 6, 14, 10, 0, tzinfo=UTC)
    schedule_rows = (
        (
            "Barcelona Grand Prix",
            "conventional",
            datetime(2026, 6, 14, 17, 0, tzinfo=UTC).isoformat(),
        ),
        (
            "Austrian Grand Prix",
            "conventional",
            datetime(2026, 6, 28, 17, 0, tzinfo=UTC).isoformat(),
        ),
    )

    planned_races = prediction_horizon.resolve_dashboard_race_horizon(
        schedule_rows=schedule_rows,
        horizon_races=2,
        now_utc=now_utc,
    )

    assert planned_races == ["Barcelona Grand Prix", "Austrian Grand Prix"]


def test_prediction_action_state_keeps_selected_race_enabled_during_boundary_lag():
    state = prediction_horizon.prediction_action_state(
        {
            "applied": True,
            "fallback_boundary_active": True,
            "stale_reason": "boundary_mismatch",
        }
    )

    assert state["disabled"] is False
    assert "still being prepared" in state["pending_message"]


def test_prediction_action_state_reports_missing_current_artifact_warmup():
    state = prediction_horizon.prediction_action_state(
        {
            "applied": False,
            "scope_applied": True,
            "stale_reason": "artifact_hash_mismatch",
        }
    )

    assert state["disabled"] is True
    assert "refreshed for the latest model version" in state["pending_message"]
