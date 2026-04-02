"""Tests for in-season race_pace blending in the updater."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.systems.updater import update_bayesian_driver_ratings


def _make_race_results(
    driver_positions: dict[str, int],
    statuses: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a minimal race results DataFrame the updater can consume."""
    rows = []
    for code, pos in driver_positions.items():
        row: dict = {"Abbreviation": code, "Position": pos, "race_name": "Test GP"}
        if statuses is not None:
            row["Status"] = statuses.get(code, "Finished")
        rows.append(row)
    return pd.DataFrame(rows)


def _make_quali_results(driver_positions: dict[str, int]) -> pd.DataFrame:
    """Build a minimal qualifying results DataFrame."""
    rows = [{"driver_code": code, "position": pos} for code, pos in driver_positions.items()]
    return pd.DataFrame(rows)


def _make_driver_payload(quali_pace: float, race_pace: float, skill: float) -> dict:
    """Build a minimal driver characteristics entry."""
    return {
        "pace": {"quali_pace": quali_pace, "race_pace": race_pace},
        "racecraft": {"skill_score": skill},
        "bayesian": {"rating_mu": 12.0, "rating_sigma": 2.5},
    }


def _run_update(
    driver_entries: dict[str, dict],
    race_positions: dict[str, int],
    quali_positions: dict[str, int],
    statuses: dict[str, str] | None = None,
) -> dict:
    """
    Run one update cycle and return the mutated drivers dict.

    Patches out all I/O and the Bayesian model so the test stays unit-level.
    The Bayesian mock returns the same mu/sigma already in the payload so the
    skill-score blend doesn't interfere with what we're measuring.
    """
    driver_payload = {"drivers": driver_entries}

    mock_bayesian = MagicMock()
    mock_bayesian.ratings = {
        code: (entry["bayesian"]["rating_mu"], entry["bayesian"]["rating_sigma"])
        for code, entry in driver_entries.items()
    }

    race_results = _make_race_results(race_positions, statuses=statuses)
    quali_results = _make_quali_results(quali_positions)

    with (
        patch("src.models.priors_factory.PriorsFactory.create_priors", return_value={}),
        patch("src.systems.updater.BayesianDriverRanking", return_value=mock_bayesian),
        patch("src.utils.lineups.load_current_lineups", return_value=None),
        patch(
            "src.systems.updater._load_driver_characteristics_payload", return_value=driver_payload
        ),
        patch("src.systems.updater._persist_driver_characteristics_payload"),
    ):
        update_bayesian_driver_ratings(race_results, quali_results)

    return driver_payload["drivers"]


def test_race_pace_increases_after_strong_finish():
    """Winning a race should pull race_pace upward from its prior value."""
    drivers = {"VER": _make_driver_payload(0.75, 0.75, 0.75)}
    result = _run_update(drivers, race_positions={"VER": 1}, quali_positions={"VER": 1})
    assert result["VER"]["pace"]["race_pace"] > 0.75


def test_race_pace_decreases_after_poor_finish():
    """Finishing near the back should drag race_pace below its prior."""
    drivers = {"ALO": _make_driver_payload(0.50, 0.80, 0.65)}
    result = _run_update(drivers, race_positions={"ALO": 18}, quali_positions={"ALO": 16})
    assert result["ALO"]["pace"]["race_pace"] < 0.80


def test_race_pace_stays_bounded():
    """race_pace must stay within [0.05, 0.99] regardless of finish position."""
    drivers = {
        "HIGH": _make_driver_payload(0.99, 0.99, 0.90),
        "LOW": _make_driver_payload(0.05, 0.05, 0.30),
    }
    result = _run_update(
        drivers,
        race_positions={"HIGH": 1, "LOW": 22},
        quali_positions={"HIGH": 1, "LOW": 22},
    )
    assert 0.05 <= result["HIGH"]["pace"]["race_pace"] <= 0.99
    assert 0.05 <= result["LOW"]["pace"]["race_pace"] <= 0.99


def test_race_pace_blend_is_partial():
    """
    A single race result should only partially move race_pace, not replace it.

    With blend=0.25 and a P1 finish, a driver starting at 0.50 should land
    around 0.625 — not jump all the way to 1.0.
    """
    drivers = {"NOR": _make_driver_payload(0.50, 0.50, 0.60)}
    result = _run_update(drivers, race_positions={"NOR": 1}, quali_positions={"NOR": 2})
    updated = result["NOR"]["pace"]["race_pace"]
    assert 0.51 < updated < 0.90, (
        f"Expected a partial blend toward 1.0 but got {updated:.3f}. "
        "Check that the blend weight isn't being treated as a hard replace."
    )


def test_dnf_driver_race_pace_unchanged():
    """A mechanical retirement should not penalize race_pace."""
    drivers = {"RIC": _make_driver_payload(0.60, 0.60, 0.55)}
    result = _run_update(
        drivers,
        race_positions={"RIC": 18},
        quali_positions={"RIC": 10},
        statuses={"RIC": "Retired"},
    )
    assert result["RIC"]["pace"]["race_pace"] == 0.60, (
        f"DNF'd driver should keep prior race_pace, got {result['RIC']['pace']['race_pace']}"
    )


def test_finished_driver_still_updated_with_status_column():
    """When Status column is present, finished drivers should still get updated."""
    drivers = {"NOR": _make_driver_payload(0.50, 0.50, 0.60)}
    result = _run_update(
        drivers,
        race_positions={"NOR": 1},
        quali_positions={"NOR": 2},
        statuses={"NOR": "Finished"},
    )
    assert result["NOR"]["pace"]["race_pace"] > 0.50, (
        "A P1 finisher with Status=Finished should still get race_pace updated"
    )
