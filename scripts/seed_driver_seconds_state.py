"""Seed seconds-native driver state from the teammate-network prior artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.driver_seconds_state import (  # noqa: E402
    DriverSecondsState,
    read_driver_seconds_state,
    write_driver_seconds_state,
)
from src.utils.schema_validation import (  # noqa: E402
    strip_legacy_bayesian_fields,
    validate_driver_characteristics,
)

DEFAULT_PRIOR_FILE = Path("data/processed/teammate_network_prior/latest.json")
DEFAULT_ROOKIE_FALLBACK_FILE = Path("data/processed/driver_seconds_rookie_fallback/latest.json")
DEFAULT_LINEUP_FILE = Path("data/current_lineups.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface for the seeding migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Season driver artifact to seed.")
    parser.add_argument(
        "--driver-file",
        type=Path,
        default=None,
        help="Driver artifact to update. Defaults to the season-scoped processed artifact.",
    )
    parser.add_argument(
        "--prior-file",
        type=Path,
        default=DEFAULT_PRIOR_FILE,
        help="Teammate-network prior JSON containing race and qualifying driver seconds.",
    )
    parser.add_argument(
        "--rookie-fallback-file",
        type=Path,
        default=DEFAULT_ROOKIE_FALLBACK_FILE,
        help="Generated debut-season fallback used for active rookies without prior nodes.",
    )
    parser.add_argument(
        "--lineup-file",
        type=Path,
        default=DEFAULT_LINEUP_FILE,
        help="Current lineup file used for the active-driver coverage gate.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Migration report path. Defaults to data/processed/driver_seconds_state_seed/latest.json.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Snapshot directory. Defaults to a migration_backups directory beside the driver file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the migration report without writing artifact, backup, or report files.",
    )
    return parser


def active_lineup_drivers(lineup_payload: Mapping[str, Any]) -> list[str]:
    """Return sorted active driver codes from a current-lineups payload."""
    lineups = lineup_payload.get("current_lineups")
    if not isinstance(lineups, Mapping):
        raise ValueError("Lineup payload is missing a 'current_lineups' mapping")

    drivers = {
        str(driver_code).strip().upper()
        for team_drivers in lineups.values()
        if isinstance(team_drivers, list)
        for driver_code in team_drivers
        if str(driver_code).strip()
    }
    if not drivers:
        raise ValueError("Lineup payload contains no active drivers")
    return sorted(drivers)


def seed_driver_seconds_payload(
    *,
    driver_payload: Mapping[str, Any],
    prior_payload: Mapping[str, Any],
    active_drivers: list[str],
    rookie_fallback_payload: Mapping[str, Any] | None = None,
    year: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a seeded driver payload and migration details.

    Existing complete seconds-native state is retained. Missing or incomplete
    seconds state is seeded from prior `mu_s` and `sigma_s`; legacy
    `bayesian.rating_mu` values are never interpreted as seconds.
    """
    updated_payload = deepcopy(dict(driver_payload))
    drivers = updated_payload.get("drivers")
    if not isinstance(drivers, dict):
        raise ValueError("Driver artifact is missing a 'drivers' mapping")

    normalized_active = sorted({str(code).strip().upper() for code in active_drivers if code})
    if not normalized_active:
        raise ValueError("Active-driver coverage gate received no drivers")

    missing_artifact_drivers = [code for code in normalized_active if code not in drivers]
    if missing_artifact_drivers:
        raise ValueError(
            "Driver artifact is missing active lineup drivers: "
            + ", ".join(missing_artifact_drivers)
        )

    season_year = int(year if year is not None else driver_payload.get("year", 0))
    active_seed_states: dict[str, DriverSecondsState] = {}
    active_state_sources: dict[str, str] = {}
    missing_active_prior: list[str] = []
    for driver_code in normalized_active:
        entry = drivers[driver_code]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Driver entry for {driver_code} is not a JSON object")

        existing_state = read_driver_seconds_state(entry)
        if existing_state is not None:
            active_seed_states[driver_code] = existing_state
            active_state_sources[driver_code] = "existing_seconds_state"
            continue

        prior_state = _driver_seconds_state_from_prior(prior_payload, driver_code)
        if prior_state is not None:
            active_seed_states[driver_code] = prior_state
            active_state_sources[driver_code] = "teammate_network_prior"
            continue

        fallback_state = _debut_season_rookie_fallback_state(
            driver_entry=entry,
            rookie_fallback_payload=rookie_fallback_payload,
            year=season_year,
        )
        if fallback_state is not None:
            active_seed_states[driver_code] = fallback_state
            active_state_sources[driver_code] = "debut_season_rookie_fallback"
            continue

        missing_active_prior.append(driver_code)

    if missing_active_prior:
        raise ValueError(
            "Teammate-network prior is missing active driver seconds coverage: "
            + ", ".join(missing_active_prior)
        )

    seeded_drivers: list[str] = []
    prior_seeded_drivers: list[str] = []
    rookie_fallback_seeded_drivers: list[str] = []
    retained_existing_drivers: list[str] = []
    missing_prior_drivers: list[str] = []
    for driver_code in sorted(drivers):
        entry = drivers[driver_code]
        if not isinstance(entry, dict):
            raise ValueError(f"Driver entry for {driver_code} is not a JSON object")

        if read_driver_seconds_state(entry) is not None:
            retained_existing_drivers.append(driver_code)
            continue

        state = active_seed_states.get(driver_code)
        source = active_state_sources.get(driver_code)
        if state is None:
            state = _driver_seconds_state_from_prior(prior_payload, driver_code)
            source = "teammate_network_prior" if state is not None else None
        if state is None:
            missing_prior_drivers.append(driver_code)
            continue

        write_driver_seconds_state(entry, state)
        seeded_drivers.append(driver_code)
        if source == "debut_season_rookie_fallback":
            rookie_fallback_seeded_drivers.append(driver_code)
        elif source == "teammate_network_prior":
            prior_seeded_drivers.append(driver_code)

    seeded_active_drivers = [code for code in normalized_active if code in seeded_drivers]
    retained_active_drivers = [
        code for code in normalized_active if code in retained_existing_drivers
    ]
    stripped_legacy_bayesian_fields = strip_legacy_bayesian_fields(drivers)
    return updated_payload, {
        "active_drivers": normalized_active,
        "seeded_drivers": seeded_drivers,
        "prior_seeded_drivers": prior_seeded_drivers,
        "rookie_fallback_seeded_drivers": rookie_fallback_seeded_drivers,
        "seeded_active_drivers": seeded_active_drivers,
        "retained_existing_drivers": retained_existing_drivers,
        "retained_active_drivers": retained_active_drivers,
        "missing_prior_drivers": missing_prior_drivers,
        "stripped_legacy_bayesian_fields": stripped_legacy_bayesian_fields,
    }


def seed_driver_seconds_file(
    *,
    driver_file: Path,
    prior_file: Path,
    rookie_fallback_file: Path,
    lineup_file: Path,
    report_file: Path,
    backup_dir: Path,
    year: int,
    dry_run: bool = False,
    seeded_at: str | None = None,
) -> dict[str, Any]:
    """Seed one local driver artifact and return the migration report."""
    current_payload = _read_json_object(driver_file)
    prior_payload = _read_json_object(prior_file)
    rookie_fallback_payload = _read_json_object(rookie_fallback_file)
    lineup_payload = _read_json_object(lineup_file)
    active_drivers = active_lineup_drivers(lineup_payload)
    updated_payload, details = seed_driver_seconds_payload(
        driver_payload=current_payload,
        prior_payload=prior_payload,
        active_drivers=active_drivers,
        rookie_fallback_payload=rookie_fallback_payload,
        year=year,
    )
    validate_driver_characteristics(updated_payload, expected_year=year)

    migration_time = seeded_at or datetime.now(UTC).isoformat()
    artifact_changed = bool(details["seeded_drivers"])
    backup_file: Path | None = None
    if artifact_changed and not dry_run:
        backup_file = snapshot_driver_file(
            driver_file,
            backup_dir=backup_dir,
            snapshot_time=_snapshot_token(migration_time),
        )
        _write_json_object(driver_file, updated_payload)

    report = build_seed_report(
        seeded_at=migration_time,
        year=year,
        driver_file=driver_file,
        backup_file=backup_file,
        prior_file=prior_file,
        prior_payload=prior_payload,
        rookie_fallback_file=rookie_fallback_file,
        rookie_fallback_payload=rookie_fallback_payload,
        lineup_file=lineup_file,
        details=details,
        artifact_changed=artifact_changed,
        dry_run=dry_run,
    )
    if not dry_run:
        _write_json_object(report_file, report)
    return report


def snapshot_driver_file(driver_file: Path, *, backup_dir: Path, snapshot_time: str) -> Path:
    """Copy the current driver artifact before a migration write."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"{driver_file.name}.driver_seconds_seed.{snapshot_time}.backup"
    shutil.copy2(driver_file, backup_file)
    return backup_file


def build_seed_report(
    *,
    seeded_at: str,
    year: int,
    driver_file: Path,
    backup_file: Path | None,
    prior_file: Path,
    prior_payload: Mapping[str, Any],
    rookie_fallback_file: Path,
    rookie_fallback_payload: Mapping[str, Any],
    lineup_file: Path,
    details: Mapping[str, Any],
    artifact_changed: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Build the persisted migration report."""
    active_drivers = list(details["active_drivers"])
    seeded_drivers = list(details["seeded_drivers"])
    retained_existing_drivers = list(details["retained_existing_drivers"])
    missing_prior_drivers = list(details["missing_prior_drivers"])
    return {
        "schema_version": 1,
        "migration": "driver_seconds_state_seed",
        "seeded_at": seeded_at,
        "year": int(year),
        "dry_run": bool(dry_run),
        "artifact_changed": bool(artifact_changed),
        "driver_artifact": {
            "path": str(driver_file),
            "backup_path": str(backup_file) if backup_file is not None else None,
        },
        "teammate_network_prior": {
            "path": str(prior_file),
            "built_at": prior_payload.get("built_at"),
        },
        "rookie_fallback": {
            "path": str(rookie_fallback_file),
            "built_at": rookie_fallback_payload.get("built_at"),
        },
        "lineup_file": str(lineup_file),
        "counts": {
            "active_drivers": len(active_drivers),
            "seeded_drivers": len(seeded_drivers),
            "prior_seeded_drivers": len(details["prior_seeded_drivers"]),
            "rookie_fallback_seeded_drivers": len(details["rookie_fallback_seeded_drivers"]),
            "seeded_active_drivers": len(details["seeded_active_drivers"]),
            "retained_existing_drivers": len(retained_existing_drivers),
            "retained_active_drivers": len(details["retained_active_drivers"]),
            "missing_prior_drivers": len(missing_prior_drivers),
            "stripped_legacy_bayesian_fields": int(details["stripped_legacy_bayesian_fields"]),
        },
        **dict(details),
    }


def main() -> None:
    """Run the local driver-seconds seed migration."""
    args = build_parser().parse_args()
    year = int(args.year)
    driver_file = args.driver_file or Path(
        f"data/processed/driver_characteristics/{year}_driver_characteristics.json"
    )
    report_file = args.report_file or Path("data/processed/driver_seconds_state_seed/latest.json")
    backup_dir = args.backup_dir or driver_file.parent / "migration_backups"
    report = seed_driver_seconds_file(
        driver_file=Path(driver_file),
        prior_file=Path(args.prior_file),
        rookie_fallback_file=Path(args.rookie_fallback_file),
        lineup_file=Path(args.lineup_file),
        report_file=Path(report_file),
        backup_dir=Path(backup_dir),
        year=year,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2))


def _driver_seconds_state_from_prior(
    prior_payload: Mapping[str, Any],
    driver_code: str,
) -> DriverSecondsState | None:
    """Read complete race and qualifying seconds state from one prior payload."""
    race_state = _network_driver_state(prior_payload, "race_network", driver_code)
    quali_state = _network_driver_state(prior_payload, "quali_network", driver_code)
    if race_state is None or quali_state is None:
        return None
    race_mu, race_sigma = race_state
    quali_mu, quali_sigma = quali_state
    return DriverSecondsState(
        race_rating_mu_s=race_mu,
        race_rating_sigma_s=race_sigma,
        quali_rating_mu_s=quali_mu,
        quali_rating_sigma_s=quali_sigma,
    )


def _debut_season_rookie_fallback_state(
    *,
    driver_entry: Mapping[str, Any],
    rookie_fallback_payload: Mapping[str, Any] | None,
    year: int,
) -> DriverSecondsState | None:
    """Return fallback state only for a debut-season rookie with no prior node."""
    if rookie_fallback_payload is None or not _is_debut_season_rookie(driver_entry, year=year):
        return None

    race_state = _fallback_session_state(rookie_fallback_payload, "race")
    qualifying_state = _fallback_session_state(rookie_fallback_payload, "qualifying")
    if race_state is None or qualifying_state is None:
        return None
    race_mu, race_sigma = race_state
    qualifying_mu, qualifying_sigma = qualifying_state
    return DriverSecondsState(
        race_rating_mu_s=race_mu,
        race_rating_sigma_s=race_sigma,
        quali_rating_mu_s=qualifying_mu,
        quali_rating_sigma_s=qualifying_sigma,
    )


def _is_debut_season_rookie(driver_entry: Mapping[str, Any], *, year: int) -> bool:
    """Return True when artifact metadata marks a driver as a debut-season rookie."""
    experience = driver_entry.get("experience")
    if not isinstance(experience, Mapping):
        return False
    debut_year_raw = experience.get("debut_year")
    if debut_year_raw is None:
        return False
    try:
        debut_year = int(debut_year_raw)
    except (TypeError, ValueError):
        return False
    tier = str(experience.get("tier", "")).strip().lower()
    return year > 0 and debut_year == year and tier == "rookie"


def _fallback_session_state(
    fallback_payload: Mapping[str, Any],
    session_key: str,
) -> tuple[float, float] | None:
    """Read finite seconds mean and non-negative sigma from one fallback session."""
    state = fallback_payload.get(session_key)
    if not isinstance(state, Mapping):
        return None
    mu_s = _finite_float(state.get("mu_s"))
    sigma_s = _finite_float(state.get("sigma_s"))
    if mu_s is None or sigma_s is None or sigma_s < 0.0:
        return None
    return mu_s, sigma_s


def _network_driver_state(
    prior_payload: Mapping[str, Any],
    network_key: str,
    driver_code: str,
) -> tuple[float, float] | None:
    """Read finite `mu_s` and non-negative `sigma_s` from one prior network."""
    network = prior_payload.get(network_key)
    if not isinstance(network, Mapping):
        return None
    drivers = network.get("drivers")
    if not isinstance(drivers, Mapping):
        return None
    state = drivers.get(driver_code)
    if not isinstance(state, Mapping):
        return None

    mu_s = _finite_float(state.get("mu_s"))
    sigma_s = _finite_float(state.get("sigma_s"))
    if mu_s is None or sigma_s is None or sigma_s < 0.0:
        return None
    return mu_s, sigma_s


def _finite_float(value: Any) -> float | None:
    """Coerce a finite float without accepting booleans."""
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric == float("inf") or numeric == float("-inf") or numeric != numeric:
        return None
    return numeric


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    with open(path, encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json_object(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object with stable local formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _snapshot_token(timestamp: str) -> str:
    """Return a filename-safe snapshot token from an ISO-like timestamp."""
    return timestamp.replace(":", "").replace("-", "").replace("+", "_").replace("T", "T")


if __name__ == "__main__":
    main()
