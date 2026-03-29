"""Tests for dashboard prediction-horizon helpers."""

from datetime import UTC, datetime, timedelta

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


def test_prediction_action_state_keeps_selected_race_enabled_during_boundary_lag():
    state = prediction_horizon.prediction_action_state(
        {
            "applied": True,
            "fallback_boundary_active": True,
            "stale_reason": "boundary_mismatch",
        }
    )

    assert state["disabled"] is False
    assert "latest warmed persisted checkpoint" in state["pending_message"]


def test_prediction_action_state_reports_missing_current_artifact_warmup():
    state = prediction_horizon.prediction_action_state(
        {
            "applied": False,
            "scope_applied": True,
            "stale_reason": "artifact_hash_mismatch",
        }
    )

    assert state["disabled"] is True
    assert "older artifact set" in state["pending_message"]
