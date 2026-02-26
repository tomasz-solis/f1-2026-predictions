"""
Characteristics Validation Script

Validates driver, team, and track characteristics for sanity and correctness.
Catches obviously wrong values before they cause prediction issues.

USAGE:
    python scripts/validate_characteristics.py --data-dir data/processed
"""

import argparse
import json
import sys
from pathlib import Path

# Known driver skill ranges (from F1 community consensus + championships)
DRIVER_VALIDATION_RULES = {
    # Elite (World Champions, consistent top performers)
    "VER": {"min": 0.85, "max": 0.99, "tier": "elite"},
    "HAM": {"min": 0.82, "max": 0.95, "tier": "elite"},
    "ALO": {"min": 0.80, "max": 0.92, "tier": "elite"},
    "NOR": {"min": 0.78, "max": 0.90, "tier": "elite"},
    # Strong (Regular podium contenders)
    "LEC": {"min": 0.75, "max": 0.88, "tier": "strong"},
    "SAI": {"min": 0.72, "max": 0.82, "tier": "strong"},
    "PIA": {"min": 0.72, "max": 0.85, "tier": "strong"},
    "RUS": {"min": 0.73, "max": 0.84, "tier": "strong"},
    # Solid midfield
    "GAS": {"min": 0.65, "max": 0.78, "tier": "solid"},
    "OCO": {"min": 0.65, "max": 0.78, "tier": "solid"},
    "ALB": {"min": 0.67, "max": 0.78, "tier": "solid"},
    "HUL": {"min": 0.65, "max": 0.76, "tier": "solid"},
    # Pay drivers / Weaker
    "STR": {"min": 0.45, "max": 0.65, "tier": "weak"},
    "ZHO": {"min": 0.45, "max": 0.63, "tier": "weak"},
    "LAW": {"min": 0.40, "max": 0.65, "tier": "rookie"},
    "BOR": {"min": 0.40, "max": 0.70, "tier": "rookie"},
}

# Team performance rules (based on championship positions)
TEAM_VALIDATION_RULES = {
    # 2025 top teams
    "McLaren": {"min": 0.80, "max": 0.95},  # Champions
    "Mercedes": {"min": 0.70, "max": 0.85},  # P2
    "Red Bull Racing": {"min": 0.70, "max": 0.85},  # P3
    "Ferrari": {"min": 0.65, "max": 0.80},  # P4
    # Midfield
    "Williams": {"min": 0.50, "max": 0.65},
    "Aston Martin": {"min": 0.40, "max": 0.55},
    "Haas F1 Team": {"min": 0.35, "max": 0.50},
    "Alpine": {"min": 0.35, "max": 0.48},
    # Back markers
    "Sauber": {"min": 0.30, "max": 0.45},
    "Audi": {"min": 0.30, "max": 0.50},  # New team, uncertain
    "Cadillac F1": {"min": 0.25, "max": 0.45},  # New team
}


def validate_driver_characteristics(driver_file: Path) -> tuple[bool, list[str]]:
    """Validate driver characteristics file, returning validation status and errors."""
    errors = []

    try:
        with open(driver_file) as f:
            data = json.load(f)

        drivers = data.get("drivers", {})

        for driver_code, driver_data in drivers.items():
            # Check required fields
            if "racecraft" not in driver_data:
                errors.append(f"{driver_code}: Missing 'racecraft' field")
                continue

            if "skill_score" not in driver_data["racecraft"]:
                errors.append(f"{driver_code}: Missing 'skill_score'")
                continue

            skill = driver_data["racecraft"]["skill_score"]

            # Range check
            if skill < 0.1 or skill > 0.99:
                errors.append(f"{driver_code}: Skill {skill:.3f} out of valid range [0.1, 0.99]")

            # Known driver validation
            if driver_code in DRIVER_VALIDATION_RULES:
                rules = DRIVER_VALIDATION_RULES[driver_code]

                if skill < rules["min"]:
                    errors.append(
                        f"{driver_code}: Skill {skill:.3f} below minimum {rules['min']:.3f} for {rules['tier']} driver"
                    )

                if skill > rules["max"]:
                    errors.append(
                        f"{driver_code}: Skill {skill:.3f} above maximum {rules['max']:.3f} for {rules['tier']} driver"
                    )

            # Pace consistency check
            if "pace" in driver_data:
                quali_pace = driver_data["pace"].get("quali_pace", 0)
                race_pace = driver_data["pace"].get("race_pace", 0)

                # Race and quali pace should be similar (within 20%)
                if abs(quali_pace - race_pace) > 0.20:
                    errors.append(
                        f"{driver_code}: Large pace gap between quali ({quali_pace:.3f}) and race ({race_pace:.3f})"
                    )

            # DNF rate sanity check
            if "dnf_risk" in driver_data:
                dnf_rate = driver_data["dnf_risk"].get("dnf_rate", 0)
                if dnf_rate > 0.40:
                    errors.append(
                        f"{driver_code}: DNF rate {dnf_rate:.3f} unrealistically high (>40%)"
                    )

    except FileNotFoundError:
        errors.append(f"File not found: {driver_file}")
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in {driver_file}")
    except Exception as e:
        errors.append(f"Error reading {driver_file}: {e}")

    return len(errors) == 0, errors


def validate_team_characteristics(team_file: Path) -> tuple[bool, list[str]]:
    """Validate team/car characteristics file, returning validation status and errors."""
    errors = []

    try:
        with open(team_file) as f:
            data = json.load(f)

        teams = data.get("teams", {})

        for team_name, team_data in teams.items():
            if "overall_performance" not in team_data:
                errors.append(f"{team_name}: Missing 'overall_performance'")
                continue

            performance = team_data["overall_performance"]

            # Range check
            if performance < 0.1 or performance > 0.99:
                errors.append(
                    f"{team_name}: Performance {performance:.3f} out of valid range [0.1, 0.99]"
                )

            # Known team validation
            if team_name in TEAM_VALIDATION_RULES:
                rules = TEAM_VALIDATION_RULES[team_name]

                if performance < rules["min"]:
                    errors.append(
                        f"{team_name}: Performance {performance:.3f} below expected minimum {rules['min']:.3f}"
                    )

                if performance > rules["max"]:
                    errors.append(
                        f"{team_name}: Performance {performance:.3f} above expected maximum {rules['max']:.3f}"
                    )

    except FileNotFoundError:
        errors.append(f"File not found: {team_file}")
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in {team_file}")
    except Exception as e:
        errors.append(f"Error reading {team_file}: {e}")

    return len(errors) == 0, errors


def validate_track_characteristics(track_file: Path) -> tuple[bool, list[str]]:
    """Validate track characteristics file, returning validation status and errors."""
    errors = []

    try:
        with open(track_file) as f:
            data = json.load(f)

        tracks = data.get("tracks", {})

        for track_name, track_data in tracks.items():
            # Check required fields
            required_fields = [
                "pit_stop_loss",
                "safety_car_prob",
                "overtaking_difficulty",
            ]

            for field in required_fields:
                if field not in track_data:
                    errors.append(f"{track_name}: Missing '{field}'")

            # Range checks
            if "pit_stop_loss" in track_data:
                pit_loss = track_data["pit_stop_loss"]
                if pit_loss < 15.0 or pit_loss > 30.0:
                    errors.append(
                        f"{track_name}: Pit stop loss {pit_loss:.1f}s outside reasonable range [15-30s]"
                    )

            if "safety_car_prob" in track_data:
                sc_prob = track_data["safety_car_prob"]
                if sc_prob < 0.0 or sc_prob > 1.0:
                    errors.append(
                        f"{track_name}: Safety car probability {sc_prob:.2f} outside [0.0-1.0]"
                    )

            if "overtaking_difficulty" in track_data:
                ot_diff = track_data["overtaking_difficulty"]
                if ot_diff < 0.0 or ot_diff > 1.0:
                    errors.append(
                        f"{track_name}: Overtaking difficulty {ot_diff:.2f} outside [0.0-1.0]"
                    )

    except FileNotFoundError:
        errors.append(f"File not found: {track_file}")
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in {track_file}")
    except Exception as e:
        errors.append(f"Error reading {track_file}: {e}")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate characteristics files for sanity")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing characteristics files",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Characteristics Validation")
    print("=" * 60)
    print()

    all_valid = True
    all_errors = []

    # Validate driver characteristics
    print("1. Validating driver characteristics...")
    driver_file = data_dir / "driver_characteristics.json"
    driver_valid, driver_errors = validate_driver_characteristics(driver_file)

    if driver_valid:
        print("   [OK] Driver characteristics VALID")
    else:
        print(f"   [ERROR] Found {len(driver_errors)} errors:")
        for error in driver_errors[:10]:  # Show first 10
            print(f"      - {error}")
        if len(driver_errors) > 10:
            print(f"      ... and {len(driver_errors) - 10} more")
        all_valid = False
        all_errors.extend(driver_errors)

    print()

    # Validate team characteristics
    print("2. Validating team characteristics...")
    team_file = data_dir / "car_characteristics" / "2026_car_characteristics.json"
    team_valid, team_errors = validate_team_characteristics(team_file)

    if team_valid:
        print("   [OK] Team characteristics VALID")
    else:
        print(f"   [ERROR] Found {len(team_errors)} errors:")
        for error in team_errors:
            print(f"      - {error}")
        all_valid = False
        all_errors.extend(team_errors)

    print()

    # Validate track characteristics
    print("3. Validating track characteristics...")
    track_file = data_dir / "track_characteristics" / "2026_track_characteristics.json"
    track_valid, track_errors = validate_track_characteristics(track_file)

    if track_valid:
        print("   [OK] Track characteristics VALID")
    else:
        print(f"   [ERROR] Found {len(track_errors)} errors:")
        for error in track_errors:
            print(f"      - {error}")
        all_valid = False
        all_errors.extend(track_errors)

    print()
    print("=" * 60)

    if all_valid:
        print("[OK] All characteristics files are VALID!")
        print("=" * 60)
        return 0
    else:
        print(f"[ERROR] Validation FAILED with {len(all_errors)} total errors")
        print("=" * 60)
        print()
        print("[WARN]  DO NOT USE these characteristics for predictions!")
        print("   Run extraction scripts with --fix flag to correct issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
