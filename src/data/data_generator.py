"""Generate baseline artifacts when preseason data is missing or stale."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_baseline_if_missing(data_dir: Path) -> None:
    """Create only the baseline artifacts that are missing or stale."""
    logger.info("Checking baseline data status...")

    # Check all required files
    team_file = data_dir / "car_characteristics" / "2026_car_characteristics.json"
    track_file = data_dir / "track_characteristics" / "2026_track_characteristics.json"
    data_dir / "driver_characteristics.json"

    needs_team_generation = False
    needs_track_generation = False

    # Check team characteristics
    if not team_file.exists():
        logger.warning("Team characteristics missing - will generate baseline")
        needs_team_generation = True
    else:
        with open(team_file) as f:
            data = json.load(f)
            freshness = data.get("data_freshness", "UNKNOWN")
            if freshness == "UNKNOWN":
                logger.warning(
                    "Team characteristics have unknown freshness - regenerating with proper metadata"
                )
                needs_team_generation = True

    # Check track characteristics
    if not track_file.exists():
        logger.warning("Track characteristics missing - will generate baseline")
        needs_track_generation = True
    else:
        with open(track_file) as f:
            data = json.load(f)
            freshness = data.get("data_freshness", "UNKNOWN")
            if freshness == "UNKNOWN":
                logger.warning(
                    "Track characteristics have unknown freshness - regenerating with proper metadata"
                )
                needs_track_generation = True

    if not needs_team_generation and not needs_track_generation:
        return

    logger.info("Generating missing baseline data automatically...")
    if needs_team_generation:
        logger.info("Generating neutral team baseline...")
        generate_neutral_team_characteristics(data_dir)
    if needs_track_generation:
        logger.info("Generating track baseline from F1 historical averages...")
        generate_default_track_characteristics(data_dir)

    logger.info("Checking driver characteristics...")
    create_driver_characteristics_if_missing(data_dir)

    logger.info("Resetting learning state...")
    reset_learning_state()
    logger.info("Baseline data generated successfully.")


def generate_quick_baseline(data_dir: Path) -> None:
    """Generate a lightweight baseline without expensive historical analysis."""
    logger.info("Generating neutral team baseline...")
    generate_neutral_team_characteristics(data_dir)

    logger.info("Generating track baseline from F1 historical averages...")
    generate_default_track_characteristics(data_dir)

    logger.info("Checking driver characteristics...")
    create_driver_characteristics_if_missing(data_dir)

    logger.info("Resetting learning state...")
    reset_learning_state()


def generate_neutral_team_characteristics(data_dir: Path) -> None:
    """Generate 2026 team characteristics from 2025 constructor standings."""
    # Use the 2025 standings as a preseason anchor with deliberately high uncertainty.
    team_2025_standings = {
        "McLaren": {"position": 1, "performance": 0.85},  # Champions
        "Mercedes": {"position": 2, "performance": 0.75},
        "Red Bull Racing": {"position": 3, "performance": 0.74},
        "Ferrari": {"position": 4, "performance": 0.70},
        "Williams": {"position": 5, "performance": 0.55},
        "RB": {"position": 6, "performance": 0.48},
        "Aston Martin": {"position": 7, "performance": 0.47},
        "Haas F1 Team": {"position": 8, "performance": 0.43},
        "Alpine": {"position": 9, "performance": 0.40},
        "Audi": {"position": 10, "performance": 0.38},
        "Cadillac F1": {
            "position": 11,
            "performance": 0.35,
        },  # New team - lowest estimate
    }

    team_characteristics: dict[str, Any] = {
        "year": 2026,
        "version": 1,
        "note": "Initialized from 2025 standings with high uncertainty for the regulation reset. Teams are re-ranked as 2026 races complete.",
        "generated_at": datetime.now().isoformat(),
        "data_freshness": "BASELINE_PRESEASON",
        "races_completed": 0,
        "last_updated": datetime.now().isoformat(),
        "teams": {},
    }

    for team, data in team_2025_standings.items():
        preseason_performance = float(data["performance"])
        team_characteristics["teams"][team] = {
            "overall_performance": preseason_performance,
            "preseason_overall_performance": preseason_performance,
            "uncertainty": 0.30,  # HIGH - regulations changed!
            "note": f"2025 P{data['position']} - starting estimate with high uncertainty",
            "last_updated": None,
            "races_completed": 0,
        }

    output_file = data_dir / "car_characteristics" / "2026_car_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(team_characteristics, f, indent=2)

    logger.info("  Generated %s team ratings from 2025 standings", len(team_2025_standings))


def generate_default_track_characteristics(data_dir: Path) -> None:
    """Generate track characteristics with reasonable F1 defaults."""
    # Baseline 2026 track catalog used for circuit priors and fallback metadata.
    # Runtime schedule resolution filters canceled 2026 Bahrain/Saudi race weekends
    # separately, so keeping these circuit profiles does not imply they remain on
    # the active race calendar.
    tracks = {
        "Bahrain Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Saudi Arabian Grand Prix": {
            "pit_stop_loss": 21.5,
            "safety_car_prob": 0.5,
            "overtaking_difficulty": 0.6,
            "type": "street",
        },
        "Australian Grand Prix": {
            "pit_stop_loss": 23.0,
            "safety_car_prob": 0.6,
            "overtaking_difficulty": 0.5,
            "type": "street",
        },
        "Japanese Grand Prix": {
            "pit_stop_loss": 21.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.5,
            "type": "permanent",
        },
        "Chinese Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.3,
            "type": "permanent",
            "has_sprint": True,
        },
        "Miami Grand Prix": {
            "pit_stop_loss": 22.5,
            "safety_car_prob": 0.6,
            "overtaking_difficulty": 0.5,
            "type": "street",
            "has_sprint": True,
        },
        "Monaco Grand Prix": {
            "pit_stop_loss": 19.0,
            "safety_car_prob": 0.7,
            "overtaking_difficulty": 0.95,
            "type": "street",
        },
        "Spanish Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.2,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Canadian Grand Prix": {
            "pit_stop_loss": 20.0,
            "safety_car_prob": 0.6,
            "overtaking_difficulty": 0.5,
            "type": "street",
            "has_sprint": True,
        },
        "Austrian Grand Prix": {
            "pit_stop_loss": 20.5,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "British Grand Prix": {
            "pit_stop_loss": 21.5,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
            "has_sprint": True,
        },
        "Belgian Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.4,
            "overtaking_difficulty": 0.3,
            "type": "permanent",
        },
        "Dutch Grand Prix": {
            "pit_stop_loss": 21.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.5,
            "type": "permanent",
            "has_sprint": True,
        },
        "Italian Grand Prix": {
            "pit_stop_loss": 21.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.2,
            "type": "permanent",
        },
        "Singapore Grand Prix": {
            "pit_stop_loss": 24.0,
            "safety_car_prob": 0.7,
            "overtaking_difficulty": 0.8,
            "type": "street",
            "has_sprint": True,
        },
        "United States Grand Prix": {
            "pit_stop_loss": 22.5,
            "safety_car_prob": 0.4,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Mexico City Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.4,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Brazilian Grand Prix": {
            "pit_stop_loss": 21.5,
            "safety_car_prob": 0.5,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Las Vegas Grand Prix": {
            "pit_stop_loss": 21.0,
            "safety_car_prob": 0.5,
            "overtaking_difficulty": 0.3,
            "type": "street",
        },
        "Qatar Grand Prix": {
            "pit_stop_loss": 21.5,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Abu Dhabi Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.4,
            "overtaking_difficulty": 0.5,
            "type": "permanent",
        },
    }

    track_characteristics = {
        "year": 2026,
        "version": 1,
        "note": "Initialized from default F1 assumptions. Run scripts/generate_2026_baseline.py for historical averages.",
        "generated_at": datetime.now().isoformat(),
        "data_freshness": "BASELINE_PRESEASON",
        "last_updated": datetime.now().isoformat(),
        "tracks": tracks,
    }

    output_file = data_dir / "track_characteristics" / "2026_track_characteristics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(track_characteristics, f, indent=2)

    logger.info("  Generated %s track characteristics", len(tracks))


def create_driver_characteristics_if_missing(data_dir: Path) -> None:
    """Add driver-characteristics metadata when it is missing."""
    driver_file = data_dir / "driver_characteristics.json"

    if not driver_file.exists():
        logger.warning(
            "Driver characteristics missing! Run: python scripts/extract_driver_characteristics.py --years 2023,2024,2025,2026"
        )
        return

    with open(driver_file) as f:
        data = json.load(f)

    # Add metadata if missing
    if "data_freshness" not in data:
        data["carried_over_from"] = 2025
        data["last_updated"] = datetime.now().isoformat()
        data["note"] = (
            "Driver characteristics carried over from 2025. Skills persist across regulation changes."
        )

        with open(driver_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("  Updated driver characteristics metadata")
    else:
        logger.info("  Driver characteristics already include metadata")


def reset_learning_state() -> None:
    """Reset learning state for 2026 season if needed."""
    learning_file = Path("data/learning_state.json")

    if learning_file.exists():
        with open(learning_file) as f:
            state = json.load(f)
            if state.get("races_completed", 0) > 0 and state.get("season") == 2026:
                logger.info("  Learning state already has race data; keeping it")
                return

    learning_state = {
        "season": 2026,
        "races_completed": 0,
        "last_checkpoint": 0,
        "last_updated": datetime.now().isoformat(),
        "method_performance": {
            "blend_50_50": {"maes": [], "avg": None},
            "blend_70_30": {"maes": [], "avg": None},
            "blend_90_10": {"maes": [], "avg": None},
            "session_order": {"maes": [], "avg": None},
        },
        "recommended_method": "blend",
        "recommended_split": "70/30",
        "overtaking_factors": {},
        "pace_model_weights": {
            "pace_weight": 0.4,
            "grid_weight": 0.3,
            "overtaking_weight": 0.2,
            "tire_deg_weight": 0.1,
        },
        "insights": [],
    }

    with open(learning_file, "w") as f:
        json.dump(learning_state, f, indent=2)

    logger.info("  Reset learning state to clean 2026 baseline")
