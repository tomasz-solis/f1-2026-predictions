"""Tests for driver update trace snapshots and row formatting."""

from __future__ import annotations

import pytest

from src.systems.driver_update_trace import (
    build_driver_update_trace_rows,
    snapshot_driver_update_state,
)


def test_driver_update_trace_keeps_legacy_and_seconds_fields_separate() -> None:
    """Trace snapshots should expose legacy rating units beside seconds fields."""
    before = snapshot_driver_update_state(
        {
            "ANT": {
                "bayesian": {
                    "rating_mu": 10.0,
                    "rating_sigma": 2.0,
                    "race_rating_mu_s": 0.2,
                    "race_rating_sigma_s": 0.3,
                    "quali_rating_mu_s": 0.1,
                    "quali_rating_sigma_s": 0.4,
                },
                "wet_skill": 0.75,
            }
        }
    )
    after = snapshot_driver_update_state(
        {
            "ANT": {
                "bayesian": {
                    "rating_mu": 10.4,
                    "rating_sigma": 1.9,
                    "race_rating_mu_s": 0.25,
                    "race_rating_sigma_s": 0.29,
                    "quali_rating_mu_s": 0.1,
                    "quali_rating_sigma_s": 0.4,
                },
                "wet_skill": 0.78,
            }
        }
    )

    rows = build_driver_update_trace_rows(
        year=2026,
        event_name="Test Grand Prix",
        session_name="Race",
        session_kind="race",
        weather_route="mixed",
        driver_codes=["ANT"],
        before=before,
        after=after,
        dry_race_update_applied=True,
        dry_quali_update_applied=False,
        wet_update_drivers={"ANT"},
    )

    row = rows[0]
    assert row["legacy_rating_mu_delta"] == pytest.approx(0.4)
    assert row["race_rating_mu_s_delta"] == pytest.approx(0.05)
    assert row["quali_rating_mu_s_delta"] == 0.0
    assert row["wet_skill_delta"] == pytest.approx(0.03)
    assert row["dry_race_update_applied"] is True
    assert row["dry_quali_update_applied"] is False
    assert row["wet_update_applied"] is True
