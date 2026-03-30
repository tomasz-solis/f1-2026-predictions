"""Integrity checks for committed driver characteristic artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_LINEUP_FILE = Path("data/current_lineups.json")
_DRIVER_FILE = Path("data/processed/driver_characteristics.json")


def _load_driver_payload() -> dict:
    """Load the committed driver artifact or skip when it is unavailable."""
    if not _DRIVER_FILE.exists():
        pytest.skip("Driver characteristics artifact not present")
    with open(_DRIVER_FILE) as handle:
        return json.load(handle)


def test_all_lineup_drivers_have_characteristics():
    """Every driver in the current lineup should exist in the artifact."""
    if not _LINEUP_FILE.exists():
        pytest.skip("Current lineup artifact not present")

    with open(_LINEUP_FILE) as handle:
        lineups = json.load(handle).get("current_lineups", {})
    drivers = _load_driver_payload().get("drivers", {})

    lineup_drivers = {driver for team_drivers in lineups.values() for driver in team_drivers}
    missing = lineup_drivers - set(drivers)
    assert not missing, (
        f"Drivers in current_lineups.json but missing from characteristics: {missing}"
    )


def test_bayesian_state_present_in_driver_characteristics():
    """Every driver should carry an initial Bayesian prior."""
    drivers = _load_driver_payload().get("drivers", {})

    for code, info in drivers.items():
        bayesian = info.get("bayesian", {})
        assert "rating_mu" in bayesian, f"{code} missing bayesian.rating_mu"
        assert "rating_sigma" in bayesian, f"{code} missing bayesian.rating_sigma"


def test_dnf_rates_have_reasonable_floor():
    """Modern reliability is good, but nobody should have a literal zero DNF rate."""
    drivers = _load_driver_payload().get("drivers", {})

    for code, info in drivers.items():
        dnf_rate = info.get("dnf_risk", {}).get("dnf_rate", 0.0)
        assert dnf_rate >= 0.03, f"{code} has unrealistic DNF floor: {dnf_rate}"


def test_driver_names_use_real_names():
    """Placeholder names like 'Driver XYZ' should never ship in committed data."""
    drivers = _load_driver_payload().get("drivers", {})

    for code, info in drivers.items():
        name = str(info.get("name", ""))
        assert not name.startswith("Driver "), f"{code} still uses placeholder name {name!r}"


def test_antonelli_quali_pace_stays_above_backmarker_floor():
    """Antonelli's qualifying pace should not collapse below midfield territory."""
    drivers = _load_driver_payload().get("drivers", {})
    antonelli = drivers.get("ANT")
    if antonelli is None:
        pytest.skip("ANT not present in driver artifact")

    quali_pace = antonelli.get("pace", {}).get("quali_pace")
    assert quali_pace is not None
    assert quali_pace > 0.45, f"ANT quali pace still looks implausibly low: {quali_pace}"
