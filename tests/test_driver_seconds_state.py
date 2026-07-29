"""Tests for seconds-native driver state artifact helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.models.driver_seconds_state import (
    DriverSecondsState,
    center_rating_mu_by_team,
    preserve_driver_seconds_fields,
    read_driver_seconds_state,
    update_driver_seconds_from_teammate_aggregates,
    write_driver_seconds_state,
)
from src.systems.updater import _persist_bayesian_ratings_to_drivers


def test_driver_seconds_state_reads_and_writes_without_legacy_unit_conversion() -> None:
    """Seconds helpers should keep the seconds fields separate from rating_mu."""
    driver_entry = {"bayesian": {"rating_mu": 14.2, "rating_sigma": 1.8}}
    state = DriverSecondsState(
        race_rating_mu_s=0.12,
        race_rating_sigma_s=0.21,
        quali_rating_mu_s=-0.08,
        quali_rating_sigma_s=0.17,
    )

    write_driver_seconds_state(driver_entry, state)

    assert driver_entry["bayesian"]["rating_mu"] == 14.2
    assert read_driver_seconds_state(driver_entry) == state


def test_legacy_bayesian_write_preserves_valid_seconds_fields() -> None:
    """Legacy posterior persistence should not erase migrated seconds fields."""
    merged = preserve_driver_seconds_fields(
        previous_bayesian={
            "rating_mu": 14.2,
            "race_rating_mu_s": 0.12,
            "race_rating_sigma_s": 0.21,
            "race_rating_observations": 2,
            "quali_rating_mu_s": -0.08,
            "quali_rating_sigma_s": 0.17,
            "quali_rating_observations": 1,
        },
        updated_bayesian={"rating_mu": 14.5, "rating_sigma": 1.7},
    )

    assert merged == {
        "rating_mu": 14.5,
        "rating_sigma": 1.7,
        "race_rating_mu_s": 0.12,
        "race_rating_sigma_s": 0.21,
        "race_rating_observations": 2,
        "quali_rating_mu_s": -0.08,
        "quali_rating_sigma_s": 0.17,
        "quali_rating_observations": 1,
    }


def test_driver_posterior_persistence_keeps_seconds_state() -> None:
    """Updater persistence should keep migrated seconds fields during legacy writes."""
    drivers = {
        "ANT": {
            "bayesian": {
                "rating_mu": 14.2,
                "rating_sigma": 1.8,
                "race_rating_mu_s": 0.12,
                "race_rating_sigma_s": 0.21,
                "quali_rating_mu_s": -0.08,
                "quali_rating_sigma_s": 0.17,
            }
        }
    }
    bayesian = SimpleNamespace(ratings={"ANT": (14.5, 1.7)}, history=[])

    _persist_bayesian_ratings_to_drivers(
        bayesian=bayesian,
        drivers_payload=drivers,
        season_year=2026,
        fallback_session_name="Race",
        updated_at="2026-05-21T00:00:00",
    )

    assert read_driver_seconds_state(drivers["ANT"]) == DriverSecondsState(
        race_rating_mu_s=0.12,
        race_rating_sigma_s=0.21,
        quali_rating_mu_s=-0.08,
        quali_rating_sigma_s=0.17,
    )


def _driver_entry(
    *,
    race_mu_s: float,
    quali_mu_s: float,
    sigma_s: float = 0.30,
) -> dict:
    """Build one complete seconds-native driver entry for update tests."""
    return {
        "bayesian": {
            "race_rating_mu_s": race_mu_s,
            "race_rating_sigma_s": sigma_s,
            "quali_rating_mu_s": quali_mu_s,
            "quali_rating_sigma_s": sigma_s,
        }
    }


def _aggregate_row(*, session_kind: str, gap_s: float) -> pd.DataFrame:
    """Build one usable canonical aggregate row."""
    return pd.DataFrame(
        [
            {
                "reference_driver_code": "AAA",
                "comparison_driver_code": "BBB",
                "team": "Example",
                "year": 2026,
                "race_name": "Test Grand Prix",
                "session_name": "Race" if session_kind == "race" else "Qualifying",
                "session_kind": session_kind,
                "matched_gap_median_s": gap_s,
                "matched_gap_se_s": 0.10,
                "n_matched_pairs": 8,
                "weather_bucket": "dry",
                "skip_reason": pd.NA,
            }
        ]
    )


def test_race_seconds_update_moves_only_race_path() -> None:
    """A dry race teammate gap should not alter qualifying seconds state."""
    drivers = {
        "AAA": _driver_entry(race_mu_s=0.0, quali_mu_s=0.22),
        "BBB": _driver_entry(race_mu_s=0.0, quali_mu_s=-0.18),
    }

    summary = update_driver_seconds_from_teammate_aggregates(
        drivers_payload=drivers,
        aggregate_rows=_aggregate_row(session_kind="race", gap_s=0.40),
        session_kind="race",
    )

    aaa = read_driver_seconds_state(drivers["AAA"])
    bbb = read_driver_seconds_state(drivers["BBB"])
    assert aaa is not None
    assert bbb is not None
    assert aaa.race_rating_mu_s > 0.0
    assert bbb.race_rating_mu_s < 0.0
    assert aaa.quali_rating_mu_s == pytest.approx(0.22)
    assert bbb.quali_rating_mu_s == pytest.approx(-0.18)
    assert drivers["AAA"]["bayesian"]["race_rating_observations"] == 1
    assert "quali_rating_observations" not in drivers["AAA"]["bayesian"]
    assert summary.observations_applied == 1
    assert summary.drivers_touched == 2


def test_qualifying_seconds_update_moves_only_qualifying_path() -> None:
    """A dry qualifying teammate gap should not alter race seconds state."""
    drivers = {
        "AAA": _driver_entry(race_mu_s=0.31, quali_mu_s=0.0),
        "BBB": _driver_entry(race_mu_s=-0.26, quali_mu_s=0.0),
    }

    update_driver_seconds_from_teammate_aggregates(
        drivers_payload=drivers,
        aggregate_rows=_aggregate_row(session_kind="qualifying", gap_s=-0.30),
        session_kind="qualifying",
    )

    aaa = read_driver_seconds_state(drivers["AAA"])
    bbb = read_driver_seconds_state(drivers["BBB"])
    assert aaa is not None
    assert bbb is not None
    assert aaa.race_rating_mu_s == pytest.approx(0.31)
    assert bbb.race_rating_mu_s == pytest.approx(-0.26)
    assert aaa.quali_rating_mu_s < 0.0
    assert bbb.quali_rating_mu_s > 0.0
    assert drivers["BBB"]["bayesian"]["quali_rating_observations"] == 1


def test_seconds_update_evidence_scale_reduces_sprint_like_movement() -> None:
    """Lower evidence precision should move seconds state less for the same gap."""
    full_evidence = {
        "AAA": _driver_entry(race_mu_s=0.0, quali_mu_s=0.0),
        "BBB": _driver_entry(race_mu_s=0.0, quali_mu_s=0.0),
    }
    half_evidence = {
        "AAA": _driver_entry(race_mu_s=0.0, quali_mu_s=0.0),
        "BBB": _driver_entry(race_mu_s=0.0, quali_mu_s=0.0),
    }
    rows = _aggregate_row(session_kind="race", gap_s=0.40)

    update_driver_seconds_from_teammate_aggregates(
        drivers_payload=full_evidence,
        aggregate_rows=rows,
        session_kind="race",
    )
    update_driver_seconds_from_teammate_aggregates(
        drivers_payload=half_evidence,
        aggregate_rows=rows,
        session_kind="race",
        evidence_scale=0.5,
    )

    full_state = read_driver_seconds_state(full_evidence["AAA"])
    half_state = read_driver_seconds_state(half_evidence["AAA"])
    assert full_state is not None
    assert half_state is not None
    assert 0.0 < half_state.race_rating_mu_s < full_state.race_rating_mu_s
    assert half_state.race_rating_sigma_s > full_state.race_rating_sigma_s


def test_seconds_update_ignores_wet_and_skipped_rows() -> None:
    """Only valid dry aggregate rows are evidence for dry seconds state."""
    drivers = {
        "AAA": _driver_entry(race_mu_s=0.0, quali_mu_s=0.0),
        "BBB": _driver_entry(race_mu_s=0.0, quali_mu_s=0.0),
    }
    wet = _aggregate_row(session_kind="race", gap_s=0.50)
    wet.loc[0, "weather_bucket"] = "wet"
    skipped = _aggregate_row(session_kind="race", gap_s=0.50)
    skipped.loc[0, "skip_reason"] = "insufficient_matched_pairs"

    summary = update_driver_seconds_from_teammate_aggregates(
        drivers_payload=drivers,
        aggregate_rows=pd.concat([wet, skipped], ignore_index=True),
        session_kind="race",
    )

    assert read_driver_seconds_state(drivers["AAA"]) == DriverSecondsState(
        race_rating_mu_s=0.0,
        race_rating_sigma_s=0.30,
        quali_rating_mu_s=0.0,
        quali_rating_sigma_s=0.30,
    )
    assert summary.observations_applied == 0


def test_center_rating_mu_by_team_removes_team_mean_and_keeps_within_team_gap() -> None:
    """Team mean should vanish while the teammate-relative gap survives."""
    records = [
        {"driver": "AAA", "team": "Ferrari", "quali_rating_mu_s": 0.30},
        {"driver": "BBB", "team": "Ferrari", "quali_rating_mu_s": -0.10},
        {"driver": "CCC", "team": "Audi", "quali_rating_mu_s": -0.40},
    ]

    center_rating_mu_by_team(records, field="quali_rating_mu_s")

    ferrari_gap = records[0]["quali_rating_mu_s"] - records[1]["quali_rating_mu_s"]
    assert ferrari_gap == pytest.approx(0.40)
    assert records[0]["quali_rating_mu_s"] + records[1]["quali_rating_mu_s"] == pytest.approx(0.0)
    # Audi has only one rated driver, so it is left untouched rather than zeroed.
    assert records[2]["quali_rating_mu_s"] == pytest.approx(-0.40)


def test_center_rating_mu_by_team_skips_missing_field_without_error() -> None:
    """A record without the target field should be skipped, not raise."""
    records = [
        {"driver": "AAA", "team": "Williams", "quali_rating_mu_s": 0.20},
        {"driver": "BBB", "team": "Williams"},
    ]

    center_rating_mu_by_team(records, field="quali_rating_mu_s")

    # Only one finite value present for Williams, so the team is untouched.
    assert records[0]["quali_rating_mu_s"] == pytest.approx(0.20)
    assert "quali_rating_mu_s" not in records[1]
