"""Rebuild 2026 race artifacts from preseason seeds plus cached race data.

This is the safe offline regeneration path for the 2026 in-season files when
the model logic changes but the committed JSON artifacts are still stuck on an
older scoring scheme. The script:

1. Backs up the current 2026 car and driver files.
2. Resets the 2026 car file to the seeded preseason baseline.
3. Restores the 2026 driver file from the carried-over baseline file.
4. Replays completed races through ``update_from_race`` using cached FastF1 data.
5. Refreshes team note strings so they describe the rebuilt state honestly.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

_DEFAULT_RACES = (
    "Australian Grand Prix",
    "Chinese Grand Prix",
    "Japanese Grand Prix",
)
_TEAM_2026_SEEDS = {
    "McLaren": {"position": 1, "performance": 0.85},
    "Mercedes": {"position": 2, "performance": 0.75},
    "Red Bull Racing": {"position": 3, "performance": 0.74},
    "Ferrari": {"position": 4, "performance": 0.70},
    "Williams": {"position": 5, "performance": 0.55},
    "RB": {"position": 6, "performance": 0.48},
    "Aston Martin": {"position": 7, "performance": 0.47},
    "Haas F1 Team": {"position": 8, "performance": 0.43},
    "Alpine": {"position": 9, "performance": 0.40},
    "Audi": {"position": 10, "performance": 0.38},
    "Cadillac F1": {"position": 11, "performance": 0.35},
}


def _parse_args() -> argparse.Namespace:
    """Parse the small CLI surface for the rebuild flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Processed data directory containing the 2026 artifacts.",
    )
    parser.add_argument(
        "--races",
        nargs="*",
        default=list(_DEFAULT_RACES),
        help="Completed 2026 races to replay in order.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Season year to rebuild.",
    )
    return parser.parse_args()


def _load_update_from_race() -> Any:
    """Import the updater lazily after the project root is on ``sys.path``."""
    from src.systems.updater import update_from_race

    return update_from_race


def _read_json(path: Path) -> dict[str, Any]:
    """Load one JSON payload from disk."""
    with open(path) as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not deserialize to a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_handle:
        json.dump(payload, file_handle, indent=2)
        file_handle.write("\n")


def _backup_file(path: Path) -> Path:
    """Create one timestamp-free rebuild backup next to the target file."""
    backup_path = path.with_suffix(path.suffix + ".rebuild_backup")
    if backup_path.exists():
        return backup_path
    shutil.copy2(path, backup_path)
    return backup_path


def _reset_car_artifact(car_file: Path, *, year: int) -> None:
    """Reset the 2026 car file to seeded preseason values."""
    payload = _read_json(car_file)
    teams = payload.get("teams")
    if not isinstance(teams, dict):
        raise ValueError(f"{car_file} is missing the 'teams' payload")

    payload["year"] = int(year)
    payload["version"] = 1
    payload["races_completed"] = 0
    payload["data_freshness"] = "BASELINE_PRESEASON"

    for team_name, team_data in teams.items():
        if not isinstance(team_data, dict):
            continue
        seed = _TEAM_2026_SEEDS.get(team_name)
        if seed is None:
            raise KeyError(f"Missing 2026 baseline seed for team '{team_name}'")

        team_data["overall_performance"] = float(seed["performance"])
        team_data["uncertainty"] = 0.30
        team_data["note"] = (
            f"2025 P{seed['position']} seed with high uncertainty for 2026 regulation reset"
        )
        team_data["last_updated"] = None
        team_data["races_completed"] = 0
        team_data["current_season_performance"] = []
        team_data.pop("compound_characteristics", None)

    _write_json(car_file, payload)


def _reset_driver_artifact(
    driver_file: Path,
    *,
    baseline_driver_file: Path,
    year: int,
) -> None:
    """Restore the 2026 driver file from the carried-over baseline profile."""
    baseline_payload = _read_json(baseline_driver_file)
    baseline_drivers = baseline_payload.get("drivers")
    if not isinstance(baseline_drivers, dict):
        raise ValueError(f"{baseline_driver_file} is missing the 'drivers' payload")

    target_payload: dict[str, Any] = {
        "note": f"{year} driver baseline carried over from 2025 before in-season updates",
        "years": baseline_payload.get("years", [2025]),
        "method": baseline_payload.get("method", "historical_carry_over"),
        "drivers": deepcopy(baseline_drivers),
        "version": 1,
        "last_updated": baseline_payload.get("extraction_date"),
        "extraction_date": baseline_payload.get("extraction_date"),
        "carried_over_from": 2025,
    }
    _write_json(driver_file, target_payload)


def _refresh_team_notes(car_file: Path) -> None:
    """Rewrite team note strings so they reflect the rebuilt race count."""
    payload = _read_json(car_file)
    teams = payload.get("teams")
    if not isinstance(teams, dict):
        raise ValueError(f"{car_file} is missing the 'teams' payload")

    for team_name, team_data in teams.items():
        if not isinstance(team_data, dict):
            continue
        seed = _TEAM_2026_SEEDS.get(team_name)
        if seed is None:
            continue
        race_count = len(team_data.get("current_season_performance", []))
        team_data["note"] = (
            f"2025 P{seed['position']} seed, updated with {race_count} race(s) of 2026 data"
        )

    _write_json(car_file, payload)


def main() -> None:
    """Reset the 2026 seed artifacts and replay completed race weekends."""
    args = _parse_args()
    update_from_race = _load_update_from_race()
    data_dir = Path(args.data_dir)
    year = int(args.year)
    races = [str(race).strip() for race in args.races if str(race).strip()]
    if not races:
        raise ValueError("At least one completed race is required to rebuild the 2026 artifacts")

    car_file = data_dir / "car_characteristics" / f"{year}_car_characteristics.json"
    driver_file = data_dir / "driver_characteristics" / f"{year}_driver_characteristics.json"
    baseline_driver_file = data_dir / "driver_characteristics.json"

    if not car_file.exists():
        raise FileNotFoundError(f"Missing car artifact: {car_file}")
    if not baseline_driver_file.exists():
        raise FileNotFoundError(f"Missing baseline driver artifact: {baseline_driver_file}")
    if not driver_file.exists():
        raise FileNotFoundError(f"Missing 2026 driver artifact: {driver_file}")

    car_backup = _backup_file(car_file)
    driver_backup = _backup_file(driver_file)
    logger.info("Backed up %s -> %s", car_file, car_backup)
    logger.info("Backed up %s -> %s", driver_file, driver_backup)

    _reset_car_artifact(car_file, year=year)
    _reset_driver_artifact(
        driver_file,
        baseline_driver_file=baseline_driver_file,
        year=year,
    )

    for race_name in races:
        logger.info("Replaying %s %s", year, race_name)
        update_from_race(year, race_name, str(data_dir))

    _refresh_team_notes(car_file)
    logger.info("Rebuilt %s race artifacts through %s", year, races[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
