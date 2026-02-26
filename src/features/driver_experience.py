"""
Driver experience calculation and categorization.

Detects first F1 season, calculates experience, and assigns tiers:
- Rookie: First season (0 years prior)
- Developing: 1-3 complete seasons
- Established: 4-6 complete seasons
- Veteran: 7+ complete seasons
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_driver_debuts_from_csv(csv_path: Path) -> dict[str, int]:
    """Load driver debuts from CSV file, returning updated debuts dictionary."""
    name_to_abbr = {
        "Fernando Alonso": "ALO",
        "Lewis Hamilton": "HAM",
        "Nico Hülkenberg": "HUL",
        "Sergio Pérez": "PER",
        "Daniel Ricciardo": "RIC",
        "Valtteri Bottas": "BOT",
        "Kevin Magnussen": "MAG",
        "Max Verstappen": "VER",
        "Carlos Sainz": "SAI",
        "Esteban Ocon": "OCO",
        "Pierre Gasly": "GAS",
        "Lance Stroll": "STR",
        "Charles Leclerc": "LEC",
        "Alexander Albon": "ALB",
        "Lando Norris": "NOR",
        "George Russell": "RUS",
        "Yuki Tsunoda": "TSU",
        "Zhou Guanyu": "ZHO",
        "Oscar Piastri": "PIA",
        "Logan Sargeant": "SAR",
        "Franco Colapinto": "COL",
        "Oliver Bearman": "BEA",
        "Isack Hadjar": "HAD",
        "Andrea Kimi Antonelli": "ANT",
        "Gabriel Bortoleto": "BOR",
        "Jack Doohan": "DOO",
        "Arvid Lindblad": "LIN",
        "Liam Lawson": "LAW",
    }

    debuts = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver_name = row["Driver"]
            debut_year = int(row["First F1 season"])

            if driver_name in name_to_abbr:
                abbr = name_to_abbr[driver_name]
                debuts[abbr] = debut_year

    return debuts


def detect_first_season(driver_data: dict[str, Any]) -> int | None:
    """Detect driver's first F1 season from their by_year data."""
    if "by_year" not in driver_data:
        return None

    years = [int(year) for year in driver_data["by_year"].keys()]
    return min(years) if years else None


def calculate_experience(
    driver_abbr: str,
    driver_data: dict[str, Any],
    current_year: int = 2025,
    driver_debuts: dict[str, int] | None = None,
) -> int | None:
    """Calculate years of F1 experience using debut year from provided dict or auto-detection."""
    debut_year = None

    # Try to get from debuts dict first
    if driver_debuts and driver_abbr in driver_debuts:
        debut_year = driver_debuts[driver_abbr]
    else:
        # Auto-detect from data
        debut_year = detect_first_season(driver_data)

    if debut_year is None:
        return None

    # Years of complete experience = current_year - debut_year
    return current_year - debut_year


def assign_experience_tier(years_experience: int | None) -> str:
    """Assign experience tier (rookie/developing/established/veteran) based on years in F1."""
    if years_experience is None:
        return "unknown"

    if years_experience == 0:
        return "rookie"
    elif 1 <= years_experience <= 3:
        return "developing"
    elif 4 <= years_experience <= 6:
        return "established"
    else:  # 7+
        return "veteran"


def calculate_pace_delta(quali_ratio: float, race_ratio: float) -> float:
    """Calculate pace delta between qualifying and race pace relative to teammates."""
    return quali_ratio - race_ratio


def determine_confidence_flag(
    driver_data: dict, experience_tier: str, pace_delta: float, min_sessions: int = 10
) -> str:
    """Determine confidence flag (high/gathering_info/low) based on data quality and patterns."""
    sessions = driver_data.get("sessions", 0)

    # Low confidence: insufficient data
    if sessions < min_sessions:
        return "low"

    # Gathering info: unusual patterns that need monitoring
    unusual_conditions = [
        # Rookie with negative pace delta (better at races than quali)
        experience_tier == "rookie" and pace_delta < -0.005,
        # High variance (inconsistent performance)
        driver_data.get("std_ratio", 0) > 0.025,
        # Multiple team changes (sample contamination)
        len(driver_data.get("teams", [])) > 2,
    ]

    if any(unusual_conditions):
        return "gathering_info"

    # High confidence: sufficient data, normal pattern
    return "high"


def enrich_driver_characteristics(
    quali_data: dict[str, Any],
    race_data: dict[str, Any],
    current_year: int = 2025,
    debuts_csv_path: str | None = None,
) -> dict[str, Any]:
    """Add experience metadata and confidence flags to driver characteristics data."""
    # Load debuts from CSV if provided
    driver_debuts = {}
    if debuts_csv_path and Path(debuts_csv_path).exists():
        driver_debuts = load_driver_debuts_from_csv(Path(debuts_csv_path))
        logger.info("Loaded %s driver debuts from CSV", len(driver_debuts))
    else:
        logger.info("No debuts CSV provided - auto-detecting debuts from data")

    enriched_drivers = {}

    for driver_abbr in quali_data["drivers"].keys():
        quali = quali_data["drivers"][driver_abbr]
        race = race_data["drivers"].get(driver_abbr, {})

        # Calculate experience using CSV debuts or auto-detection
        first_season = None
        if driver_abbr in driver_debuts:
            first_season = driver_debuts[driver_abbr]
        else:
            first_season = detect_first_season(quali)

        years_exp = calculate_experience(driver_abbr, quali, current_year, driver_debuts)
        tier = assign_experience_tier(years_exp)

        # Calculate pace delta
        quali_ratio = quali.get("avg_ratio", 1.0)
        race_ratio = race.get("avg_ratio", 1.0) if race else None
        pace_delta = calculate_pace_delta(quali_ratio, race_ratio) if race_ratio else None

        # Determine confidence
        confidence = determine_confidence_flag(quali, tier, pace_delta or 0)

        # Build enriched profile
        enriched_drivers[driver_abbr] = {
            "driver": driver_abbr,
            "experience": {
                "first_season": first_season,
                "years_experience": years_exp,
                "tier": tier,
            },
            "quali_pace": quali_ratio,
            "race_pace": race_ratio,
            "pace_delta": pace_delta,
            "quali_sessions": quali.get("sessions", 0),
            "race_sessions": race.get("sessions", 0) if race else 0,
            "quali_std": quali.get("std_ratio", 0),
            "race_std": race.get("std_ratio", 0) if race else 0,
            "teammates": quali.get("teammates", []),
            "teams": quali.get("teams", []),
            "confidence": confidence,
            "rookie_flag": tier == "rookie",  # Keep for backwards compatibility
        }

    return {
        "extracted_at": quali_data["extracted_at"],
        "seasons": quali_data["seasons"],
        "current_year": current_year,
        "total_drivers": len(enriched_drivers),
        "drivers": enriched_drivers,
    }


def analyze_experience_distribution(enriched_data: dict) -> dict:
    """Analyze distribution of drivers across experience tiers with summary statistics."""
    tiers = {"rookie": [], "developing": [], "established": [], "veteran": []}

    for driver_abbr, driver in enriched_data["drivers"].items():
        tier = driver["experience"]["tier"]
        if tier in tiers:
            tiers[tier].append(driver_abbr)

    return {
        "by_tier": {
            tier: {"count": len(drivers), "drivers": sorted(drivers)}
            for tier, drivers in tiers.items()
        },
        "confidence_distribution": {
            "high": sum(1 for d in enriched_data["drivers"].values() if d["confidence"] == "high"),
            "gathering_info": sum(
                1 for d in enriched_data["drivers"].values() if d["confidence"] == "gathering_info"
            ),
            "low": sum(1 for d in enriched_data["drivers"].values() if d["confidence"] == "low"),
        },
    }


if __name__ == "__main__":
    # Test with the extracted data
    import sys

    quali_path = Path("../data/processed/driver_characteristics/driver_quali_characteristics.json")
    race_path = Path("../data/processed/driver_characteristics/driver_race_characteristics.json")
    debuts_csv = Path("../data/driver_debuts.csv")

    if not quali_path.exists() or not race_path.exists():
        logger.error("Driver characteristics files not found")
        sys.exit(1)

    with open(quali_path) as f:
        quali_data = json.load(f)

    with open(race_path) as f:
        race_data = json.load(f)

    # Enrich data (with CSV if available)
    debuts_csv_str = str(debuts_csv) if debuts_csv.exists() else None
    enriched = enrich_driver_characteristics(
        quali_data, race_data, current_year=2025, debuts_csv_path=debuts_csv_str
    )

    # Analyze distribution
    analysis = analyze_experience_distribution(enriched)

    logger.info("=== EXPERIENCE TIER DISTRIBUTION ===")
    for tier, data in analysis["by_tier"].items():
        logger.info("%s: %s drivers", tier.upper(), data["count"])
        logger.info("  %s", ", ".join(data["drivers"]))

    logger.info("=== CONFIDENCE DISTRIBUTION ===")
    for conf, count in analysis["confidence_distribution"].items():
        logger.info("%s: %s drivers", conf, count)

    logger.info("=== GATHERING INFO FLAGS (Unusual Patterns) ===")
    for driver_abbr, driver in enriched["drivers"].items():
        if driver["confidence"] == "gathering_info":
            logger.info("%s:", driver_abbr)
            logger.info("  Tier: %s", driver["experience"]["tier"])
            if driver["pace_delta"] is not None:
                logger.info("  Pace delta: %.4f", driver["pace_delta"])
            else:
                logger.info("  No race data")
            logger.info("  Teams: %s", ", ".join(driver["teams"]))
            logger.info("  Quali std: %.4f", driver["quali_std"])

    # Save enriched data
    output_path = Path(
        "../data/processed/driver_characteristics/driver_characteristics_enriched.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    logger.info("Saved enriched data to %s", output_path)
