"""Rebuild 2026 race artifacts from preseason seeds plus cached race data.

This is the safe offline regeneration path for the 2026 in-season files when
the model logic changes but the committed JSON artifacts are still stuck on an
older scoring scheme. The script:

1. Backs up the current 2026 car and driver files.
2. Resets the 2026 car file to the seeded preseason baseline.
3. Restores the 2026 driver file from a clean preseason driver baseline.
4. Replays completed practice, sprint, and race sessions through the updaters using cached
   FastF1 data.
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
    "Miami Grand Prix",
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
    parser.add_argument(
        "--car-baseline-file",
        default=None,
        help=(
            "Clean car baseline to restore before replay. Defaults to the "
            "*.pre_sync_backup file when present."
        ),
    )
    parser.add_argument(
        "--driver-baseline-file",
        default=None,
        help=(
            "Clean driver baseline to restore before replay. Defaults to the current "
            "season-scoped driver file, read before it is overwritten."
        ),
    )
    parser.add_argument(
        "--skip-sprint-updates",
        action="store_true",
        help="Replay only grand-prix qualifying/race updates.",
    )
    parser.add_argument(
        "--skip-practice-updates",
        action="store_true",
        help="Do not replay FP/testing-derived car directionality updates.",
    )
    return parser.parse_args()


def _load_update_from_race() -> Any:
    """Import the updater lazily after the project root is on ``sys.path``."""
    from src.systems.updater import update_from_race

    return update_from_race


def _load_update_from_sprint_race() -> Any:
    """Import the sprint updater lazily after the project root is on ``sys.path``."""
    from src.systems.updater import update_from_sprint_race

    return update_from_sprint_race


def _load_update_from_testing_sessions() -> Any:
    """Import the practice updater lazily after the project root is on ``sys.path``."""
    from src.systems.testing_updater import update_from_testing_sessions

    return update_from_testing_sessions


def _load_is_sprint_weekend() -> Any:
    """Import the sprint-weekend resolver lazily after the project root is on ``sys.path``."""
    from src.utils.weekend import is_sprint_weekend

    return is_sprint_weekend


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


def _reset_artifact_from_payload(path: Path, payload: dict[str, Any], *, year: int) -> None:
    """Restore an artifact from an already-loaded clean baseline payload."""
    reset_payload = deepcopy(payload)
    reset_payload["year"] = int(year)
    reset_payload["version"] = 1
    _write_json(path, reset_payload)


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

        preseason_performance = float(seed["performance"])
        team_data["overall_performance"] = preseason_performance
        team_data["preseason_overall_performance"] = preseason_performance
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
    baseline_payload: dict[str, Any],
    year: int,
) -> None:
    """Restore the 2026 driver file from a clean preseason profile."""
    baseline_drivers = baseline_payload.get("drivers")
    if not isinstance(baseline_drivers, dict):
        raise ValueError("Driver baseline payload is missing the 'drivers' payload")

    target_payload: dict[str, Any] = deepcopy(baseline_payload)
    target_payload["year"] = int(year)
    target_payload["version"] = 1
    target_payload["bayesian_last_updated_year"] = int(year)
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


def _practice_sessions_for_weekend(is_sprint: bool) -> list[str]:
    """Return practice sessions that exist before competitive running."""
    return ["FP1"] if is_sprint else ["FP1", "FP2", "FP3"]


def main() -> None:
    """Reset the 2026 seed artifacts and replay completed race weekends."""
    args = _parse_args()
    update_from_race = _load_update_from_race()
    update_from_sprint_race = _load_update_from_sprint_race()
    update_from_testing_sessions = _load_update_from_testing_sessions()
    is_sprint_weekend = _load_is_sprint_weekend()
    data_dir = Path(args.data_dir)
    year = int(args.year)
    races = [str(race).strip() for race in args.races if str(race).strip()]
    if not races:
        raise ValueError("At least one completed race is required to rebuild the 2026 artifacts")

    car_file = data_dir / "car_characteristics" / f"{year}_car_characteristics.json"
    driver_file = data_dir / "driver_characteristics" / f"{year}_driver_characteristics.json"
    default_car_baseline = car_file.with_suffix(car_file.suffix + ".pre_sync_backup")
    car_baseline_file = (
        Path(args.car_baseline_file)
        if args.car_baseline_file
        else (default_car_baseline if default_car_baseline.exists() else None)
    )
    driver_baseline_file = (
        Path(args.driver_baseline_file) if args.driver_baseline_file else driver_file
    )

    if not car_file.exists():
        raise FileNotFoundError(f"Missing car artifact: {car_file}")
    if not driver_file.exists():
        raise FileNotFoundError(f"Missing 2026 driver artifact: {driver_file}")
    if not driver_baseline_file.exists():
        raise FileNotFoundError(f"Missing driver baseline artifact: {driver_baseline_file}")

    driver_baseline_payload = _read_json(driver_baseline_file)
    car_baseline_payload = _read_json(car_baseline_file) if car_baseline_file is not None else None

    car_backup = _backup_file(car_file)
    driver_backup = _backup_file(driver_file)
    logger.info("Backed up %s -> %s", car_file, car_backup)
    logger.info("Backed up %s -> %s", driver_file, driver_backup)

    if car_baseline_payload is not None:
        logger.info("Resetting car artifact from %s", car_baseline_file)
        _reset_artifact_from_payload(car_file, car_baseline_payload, year=year)
    else:
        logger.info("Resetting car artifact from embedded 2026 seed table")
        _reset_car_artifact(car_file, year=year)
    _reset_driver_artifact(
        driver_file,
        baseline_payload=driver_baseline_payload,
        year=year,
    )

    data_root = data_dir.parent if data_dir.name == "processed" else data_dir
    for race_name in races:
        logger.info("Replaying %s %s", year, race_name)
        sprint_weekend = bool(is_sprint_weekend(year, race_name))
        if not args.skip_practice_updates:
            update_from_testing_sessions(
                year=year,
                characteristics_year=year,
                events=[race_name],
                data_dir=str(data_dir),
                sessions=_practice_sessions_for_weekend(sprint_weekend),
                testing_backend="auto",
                cache_dir="data/raw/.fastf1_cache",
                force_renew_cache=False,
                new_weight=0.7,
                directionality_scale=0.10,
                session_aggregation="laps_weighted",
                run_profile="balanced",
                dry_run=False,
            )
        if not args.skip_sprint_updates and sprint_weekend:
            update_from_sprint_race(year, race_name, str(data_root))
        update_from_race(year, race_name, str(data_dir))

    _refresh_team_notes(car_file)
    logger.info("Rebuilt %s race artifacts through %s", year, races[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
