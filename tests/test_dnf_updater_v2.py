"""Unit tests for the DNF technical/collision split's season-state update
functions (docs/DNF_CALIBRATION_BRIEF.md v2): retirement-REASON classification
from real FastF1 ``Status`` text, per-team mechanical-rate EMA, and per-driver
cumulative collision track record.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.systems.updater import (
    _classify_dnf_status_reason,
    _dnf_status_reasons,
    _update_driver_collision_track_record,
    _update_team_technical_dnf_rate_ema,
)


def _session_results(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# -- classifier -------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        ("Engine", "mechanical"),
        ("Gearbox", "mechanical"),
        ("Hydraulics", "mechanical"),
        ("Power Unit", "mechanical"),
        ("Collision", "collision"),
        ("Accident", "collision"),
        ("Spun off", "collision"),
        ("Collision damage", "collision"),  # collision keyword wins over "damage" ambiguity
        ("Retired", "other"),
        ("Disqualified", "other"),
        ("Withdrawn", "other"),
        ("Did not start", "other"),
        ("Finished", "other"),  # not a DNF classification question here, just the raw text
    ],
)
def test_classify_dnf_status_reason(status, expected):
    assert _classify_dnf_status_reason(status) == expected


def test_dnf_status_reasons_only_covers_actual_dnf_rows():
    results = _session_results(
        [
            {"Abbreviation": "AAA", "Status": "Finished"},
            {"Abbreviation": "BBB", "Status": "+1 Lap"},
            {"Abbreviation": "CCC", "Status": "Engine"},
            {"Abbreviation": "DDD", "Status": "Collision"},
            {"Abbreviation": "EEE", "Status": "Retired"},
        ]
    )
    reasons = _dnf_status_reasons(results)
    assert reasons == {"CCC": "mechanical", "DDD": "collision", "EEE": "other"}
    assert "AAA" not in reasons and "BBB" not in reasons


# -- per-team technical EMA --------------------------------------------------


def test_team_technical_ema_new_team_starts_from_neutral_default():
    """A team with no prior stored record starts from the function's own
    neutral default (0.10) and blends toward this race's observed rate."""
    results = _session_results(
        [
            {"Abbreviation": "A1", "Status": "Engine"},
            {"Abbreviation": "A2", "Status": "Finished"},
        ]
    )
    drivers_payload = {"A1": {}, "A2": {}}
    driver_to_team = {"A1": "Cadillac", "A2": "Cadillac"}

    touched = _update_team_technical_dnf_rate_ema(
        session_results=results,
        drivers_payload=drivers_payload,
        driver_to_team=driver_to_team,
        blend_weight=0.35,
        floor=0.02,
        cap=0.35,
    )
    assert touched == 2
    # observed_rate = 1 mechanical / 2 observed cars = 0.5; blended from a 0.10
    # default: (1-0.35)*0.10 + 0.35*0.5 = 0.065 + 0.175 = 0.24
    for code in ("A1", "A2"):
        assert drivers_payload[code]["team_technical_dnf_risk"]["dnf_rate"] == pytest.approx(0.24)
        assert drivers_payload[code]["team_technical_dnf_risk"]["races_observed"] == 1


def test_team_technical_ema_is_shared_identically_by_both_teammates():
    """Team-level -- both teammates must ALWAYS carry the identical value,
    even when only one of them actually retired."""
    results = _session_results(
        [
            {"Abbreviation": "A1", "Status": "Gearbox"},
            {"Abbreviation": "A2", "Status": "Finished"},
        ]
    )
    drivers_payload = {"A1": {}, "A2": {}}
    driver_to_team = {"A1": "TeamX", "A2": "TeamX"}
    _update_team_technical_dnf_rate_ema(
        session_results=results,
        drivers_payload=drivers_payload,
        driver_to_team=driver_to_team,
        blend_weight=0.35,
        floor=0.02,
        cap=0.35,
    )
    assert (
        drivers_payload["A1"]["team_technical_dnf_risk"]
        == drivers_payload["A2"]["team_technical_dnf_risk"]
    )


def test_team_technical_ema_follows_reliability_improving_across_races():
    """A team's stored rate must trend DOWN across consecutive clean races
    (the reg-reset-teething-fades-away dynamic the brief requires)."""
    drivers_payload = {"A1": {}, "A2": {}}
    driver_to_team = {"A1": "TeamX", "A2": "TeamX"}
    clean_race = _session_results(
        [
            {"Abbreviation": "A1", "Status": "Finished"},
            {"Abbreviation": "A2", "Status": "Finished"},
        ]
    )
    rates = []
    for _ in range(5):
        _update_team_technical_dnf_rate_ema(
            session_results=clean_race,
            drivers_payload=drivers_payload,
            driver_to_team=driver_to_team,
            blend_weight=0.35,
            floor=0.02,
            cap=0.35,
        )
        rates.append(drivers_payload["A1"]["team_technical_dnf_risk"]["dnf_rate"])
    assert rates == sorted(rates, reverse=True)  # monotonically non-increasing
    assert rates[-1] < rates[0]


def test_team_technical_ema_ignores_collision_retirements():
    """A collision-only retirement must NOT move the technical rate -- proves
    the reason classifier actually gates the aggregation, not just any DNF."""
    drivers_payload = {"A1": {}, "A2": {}}
    driver_to_team = {"A1": "TeamX", "A2": "TeamX"}
    crash_race = _session_results(
        [
            {"Abbreviation": "A1", "Status": "Collision"},
            {"Abbreviation": "A2", "Status": "Finished"},
        ]
    )
    _update_team_technical_dnf_rate_ema(
        session_results=crash_race,
        drivers_payload=drivers_payload,
        driver_to_team=driver_to_team,
        blend_weight=0.35,
        floor=0.02,
        cap=0.35,
    )
    # observed mechanical rate = 0 (the retirement was a collision, not mechanical)
    assert drivers_payload["A1"]["team_technical_dnf_risk"]["dnf_rate"] == pytest.approx(
        (1 - 0.35) * 0.10
    )


def test_team_technical_ema_no_observed_drivers_is_a_no_op():
    empty_results = _session_results([])
    drivers_payload = {"A1": {}}
    touched = _update_team_technical_dnf_rate_ema(
        session_results=empty_results,
        drivers_payload=drivers_payload,
        driver_to_team={"A1": "TeamX"},
        blend_weight=0.35,
        floor=0.02,
        cap=0.35,
    )
    assert touched == 0
    assert "team_technical_dnf_risk" not in drivers_payload["A1"]


# -- per-driver collision track record ---------------------------------------


def test_driver_collision_track_record_accumulates_races_and_collisions():
    drivers_payload = {"D1": {}}
    race1 = _session_results([{"Abbreviation": "D1", "Status": "Collision"}])
    race2 = _session_results([{"Abbreviation": "D1", "Status": "Finished"}])

    _update_driver_collision_track_record(session_results=race1, drivers_payload=drivers_payload)
    assert drivers_payload["D1"]["collision_dnf_track_record"] == {
        "races_observed": 1,
        "collisions_observed": 1,
    }

    _update_driver_collision_track_record(session_results=race2, drivers_payload=drivers_payload)
    assert drivers_payload["D1"]["collision_dnf_track_record"] == {
        "races_observed": 2,
        "collisions_observed": 1,
    }


def test_driver_collision_track_record_mechanical_dnf_does_not_count_as_collision():
    drivers_payload = {"D1": {}}
    race = _session_results([{"Abbreviation": "D1", "Status": "Engine"}])
    _update_driver_collision_track_record(session_results=race, drivers_payload=drivers_payload)
    assert drivers_payload["D1"]["collision_dnf_track_record"] == {
        "races_observed": 1,
        "collisions_observed": 0,
    }


def test_driver_collision_track_record_leakage_only_accumulates_forward():
    """Simulate 3 leakage-safe sequential race-state updates -- the stored
    record after race N must reflect exactly races 1..N, never race N+1."""
    drivers_payload = {"D1": {}}
    races = [
        _session_results([{"Abbreviation": "D1", "Status": "Collision"}]),
        _session_results([{"Abbreviation": "D1", "Status": "Finished"}]),
        _session_results([{"Abbreviation": "D1", "Status": "Collision"}]),
    ]
    snapshots = []
    for race in races:
        _update_driver_collision_track_record(session_results=race, drivers_payload=drivers_payload)
        snapshots.append(dict(drivers_payload["D1"]["collision_dnf_track_record"]))

    assert snapshots[0] == {"races_observed": 1, "collisions_observed": 1}
    assert snapshots[1] == {"races_observed": 2, "collisions_observed": 1}
    assert snapshots[2] == {"races_observed": 3, "collisions_observed": 2}
