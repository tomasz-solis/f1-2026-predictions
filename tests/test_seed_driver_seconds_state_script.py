"""Tests for the driver seconds-state seed migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.seed_driver_seconds_state import (
    active_lineup_drivers,
    seed_driver_seconds_file,
    seed_driver_seconds_payload,
)

from src.models.driver_seconds_state import DriverSecondsState, read_driver_seconds_state


def _driver_entry() -> dict[str, Any]:
    """Return the smallest driver entry accepted by artifact validation."""
    return {
        "racecraft": {"skill_score": 0.5, "overtaking_skill": 0.5},
        "pace": {"quali_pace": 0.5, "race_pace": 0.5},
        "dnf_risk": {"dnf_rate": 0.05},
        "bayesian": {"rating_mu": 12.0, "rating_sigma": 2.0, "season_year": 2026},
    }


def _driver_payload(*driver_codes: str) -> dict[str, Any]:
    """Return a valid season-scoped driver artifact payload."""
    return {"year": 2026, "version": 7, "drivers": {code: _driver_entry() for code in driver_codes}}


def _prior_payload(*driver_codes: str) -> dict[str, Any]:
    """Return race and qualifying seconds priors for the supplied drivers."""
    return {
        "built_at": "2026-05-18T05:27:03+00:00",
        "race_network": {
            "drivers": {
                code: {"mu_s": index / 10, "sigma_s": 0.2 + index / 100}
                for index, code in enumerate(driver_codes, start=1)
            }
        },
        "quali_network": {
            "drivers": {
                code: {"mu_s": -index / 20, "sigma_s": 0.3 + index / 100}
                for index, code in enumerate(driver_codes, start=1)
            }
        },
    }


def _rookie_fallback_payload() -> dict[str, Any]:
    """Return a generated-style debut-season rookie fallback payload."""
    return {
        "built_at": "2026-05-21T00:00:00+00:00",
        "race": {"mu_s": -0.19, "sigma_s": 0.53},
        "qualifying": {"mu_s": -0.03, "sigma_s": 0.54},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON for one migration fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_active_lineup_drivers_returns_sorted_unique_codes() -> None:
    """Coverage gates should use every current lineup driver once."""
    drivers = active_lineup_drivers(
        {"current_lineups": {"Team A": ["ant", "RUS"], "Team B": ["RUS", "GAS"]}}
    )

    assert drivers == ["ANT", "GAS", "RUS"]


def test_seed_payload_uses_prior_seconds_without_converting_legacy_rating_mu() -> None:
    """Seconds seeding should keep legacy Bayesian rating units untouched."""
    driver_payload = _driver_payload("ANT", "RUS", "OLD")
    updated, details = seed_driver_seconds_payload(
        driver_payload=driver_payload,
        prior_payload=_prior_payload("ANT", "RUS"),
        active_drivers=["ANT", "RUS"],
    )

    assert details["seeded_active_drivers"] == ["ANT", "RUS"]
    assert details["missing_prior_drivers"] == ["OLD"]
    assert updated["drivers"]["ANT"]["bayesian"]["rating_mu"] == 12.0
    ant_state = read_driver_seconds_state(updated["drivers"]["ANT"])
    assert ant_state is not None
    assert ant_state.race_rating_mu_s == pytest.approx(0.1)
    assert ant_state.race_rating_sigma_s == pytest.approx(0.21)
    assert ant_state.quali_rating_mu_s == pytest.approx(-0.05)
    assert ant_state.quali_rating_sigma_s == pytest.approx(0.31)
    assert read_driver_seconds_state(updated["drivers"]["OLD"]) is None


def test_seed_payload_strips_rejected_legacy_bayesian_cache_fields() -> None:
    """Old baseline artifacts should be cleaned before seeded output validates."""
    driver_payload = _driver_payload("ANT")
    driver_payload["drivers"]["ANT"]["bayesian"]["normalized_skill_score"] = 0.7

    updated, details = seed_driver_seconds_payload(
        driver_payload=driver_payload,
        prior_payload=_prior_payload("ANT"),
        active_drivers=["ANT"],
    )

    assert "normalized_skill_score" not in updated["drivers"]["ANT"]["bayesian"]
    assert details["stripped_legacy_bayesian_fields"] == 1


def test_seed_payload_retains_complete_seconds_state() -> None:
    """Rerunning the seed should not reset a complete seconds-native state."""
    driver_payload = _driver_payload("ANT")
    driver_payload["drivers"]["ANT"]["bayesian"].update(
        {
            "race_rating_mu_s": 0.91,
            "race_rating_sigma_s": 0.22,
            "quali_rating_mu_s": 0.81,
            "quali_rating_sigma_s": 0.23,
        }
    )

    updated, details = seed_driver_seconds_payload(
        driver_payload=driver_payload,
        prior_payload=_prior_payload("ANT"),
        active_drivers=["ANT"],
    )

    assert details["seeded_drivers"] == []
    assert details["retained_active_drivers"] == ["ANT"]
    assert read_driver_seconds_state(updated["drivers"]["ANT"]) == DriverSecondsState(
        race_rating_mu_s=0.91,
        race_rating_sigma_s=0.22,
        quali_rating_mu_s=0.81,
        quali_rating_sigma_s=0.23,
    )


def test_seed_payload_rejects_missing_active_prior_coverage() -> None:
    """Active lineup drivers must be present in both prior networks."""
    driver_payload = _driver_payload("ANT", "RUS")
    prior_payload = _prior_payload("ANT", "RUS")
    del prior_payload["quali_network"]["drivers"]["RUS"]

    with pytest.raises(ValueError, match="missing active driver seconds coverage: RUS"):
        seed_driver_seconds_payload(
            driver_payload=driver_payload,
            prior_payload=prior_payload,
            active_drivers=["ANT", "RUS"],
        )


def test_seed_payload_uses_rookie_fallback_for_active_debut_driver_without_prior() -> None:
    """Unseen debut drivers should use the generated rookie seconds fallback."""
    driver_payload = _driver_payload("LIN", "LAW")
    driver_payload["drivers"]["LIN"]["experience"] = {
        "tier": "rookie",
        "debut_year": 2026,
        "years_of_experience": 0,
    }

    updated, details = seed_driver_seconds_payload(
        driver_payload=driver_payload,
        prior_payload=_prior_payload("LAW"),
        active_drivers=["LIN", "LAW"],
        rookie_fallback_payload=_rookie_fallback_payload(),
        year=2026,
    )

    assert details["rookie_fallback_seeded_drivers"] == ["LIN"]
    assert read_driver_seconds_state(updated["drivers"]["LIN"]) == DriverSecondsState(
        race_rating_mu_s=-0.19,
        race_rating_sigma_s=0.53,
        quali_rating_mu_s=-0.03,
        quali_rating_sigma_s=0.54,
    )


def test_seed_file_snapshots_artifact_and_writes_report(tmp_path: Path) -> None:
    """A local write should keep a pre-migration snapshot and audit report."""
    driver_file = _write_json(tmp_path / "drivers.json", _driver_payload("ANT", "RUS"))
    prior_file = _write_json(tmp_path / "prior.json", _prior_payload("ANT", "RUS"))
    fallback_file = _write_json(tmp_path / "rookie_fallback.json", _rookie_fallback_payload())
    lineup_file = _write_json(
        tmp_path / "lineups.json",
        {"current_lineups": {"Mercedes": ["ANT", "RUS"]}},
    )
    report_file = tmp_path / "reports" / "driver_seconds_seed.json"
    backup_dir = tmp_path / "backups"

    report = seed_driver_seconds_file(
        driver_file=driver_file,
        prior_file=prior_file,
        rookie_fallback_file=fallback_file,
        lineup_file=lineup_file,
        report_file=report_file,
        backup_dir=backup_dir,
        year=2026,
        seeded_at="2026-05-21T10:11:12+00:00",
    )

    written_payload = json.loads(driver_file.read_text())
    written_report = json.loads(report_file.read_text())
    backup_path = Path(report["driver_artifact"]["backup_path"])
    assert read_driver_seconds_state(written_payload["drivers"]["ANT"]) is not None
    assert backup_path.exists()
    assert read_driver_seconds_state(json.loads(backup_path.read_text())["drivers"]["ANT"]) is None
    assert report["counts"]["seeded_active_drivers"] == 2
    assert written_report == report
