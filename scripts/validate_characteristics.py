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

DEFAULT_SEASON_YEAR = 2026

# Known driver skill expectations (historical guidance only; can drift as extraction logic evolves)
DRIVER_EXPECTATION_RULES = {
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

# Team performance expectations (historical guidance only; not used for preseason neutral baseline)
TEAM_EXPECTATION_RULES = {
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


def _find_latest_season_file(
    data_dir: Path,
    subdir: str,
    filename_suffix: str,
) -> Path | None:
    """Return the highest-year season file for one characteristics artifact."""
    season_dir = data_dir / subdir
    matches: list[tuple[int, Path]] = []

    for candidate in season_dir.glob(f"*_{filename_suffix}.json"):
        year_token = candidate.name.split("_", 1)[0]
        if year_token.isdigit():
            matches.append((int(year_token), candidate))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def _resolve_season_scoped_file(
    data_dir: Path,
    subdir: str,
    filename_suffix: str,
    season_year: int | None,
) -> Path:
    """Resolve a season-scoped artifact path, defaulting to the latest available season."""
    if season_year is not None:
        return data_dir / subdir / f"{season_year}_{filename_suffix}.json"

    latest_file = _find_latest_season_file(data_dir, subdir, filename_suffix)
    if latest_file is not None:
        return latest_file

    return data_dir / subdir / f"{DEFAULT_SEASON_YEAR}_{filename_suffix}.json"


def _resolve_driver_characteristics_file(data_dir: Path, season_year: int | None) -> Path:
    """Prefer season-scoped driver data, then fall back to the legacy flat file."""
    season_file = _resolve_season_scoped_file(
        data_dir=data_dir,
        subdir="driver_characteristics",
        filename_suffix="driver_characteristics",
        season_year=season_year,
    )
    if season_file.exists():
        return season_file

    legacy_file = data_dir / "driver_characteristics.json"
    if legacy_file.exists():
        return legacy_file

    return season_file


def _record_expectation_violation(
    errors: list[str], warnings: list[str], message: str, *, enforce_expectations: bool
) -> None:
    """Store expectation violations as warnings by default, optionally as errors."""
    if enforce_expectations:
        errors.append(message)
    else:
        warnings.append(message)


def _is_preseason_team_baseline(data: dict) -> bool:
    """
    Detect neutral preseason baseline payloads for regulation resets.

    These intentionally keep teams near 0.5 with high uncertainty and should not
    be validated against historical championship ordering expectations.
    """
    freshness = str(data.get("data_freshness", "")).upper()
    note = str(data.get("note", "")).upper()
    if freshness == "BASELINE_PRESEASON":
        return True
    if "REGULATION RESET" in note:
        return True

    teams = data.get("teams", {})
    if not isinstance(teams, dict) or not teams:
        return False

    perf_values = [
        team_data.get("overall_performance")
        for team_data in teams.values()
        if isinstance(team_data, dict)
    ]
    if not perf_values:
        return False

    return all(
        isinstance(value, (int | float)) and abs(float(value) - 0.5) <= 0.05
        for value in perf_values
    )


def validate_driver_characteristics(
    driver_file: Path, *, enforce_expectations: bool = False
) -> tuple[bool, list[str], list[str]]:
    """Validate driver characteristics file, returning status, errors, and warnings."""
    errors = []
    warnings = []
    skill_values: list[float] = []

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
            if isinstance(skill, (int | float)):
                skill_values.append(float(skill))

            # Range check
            if skill < 0.1 or skill > 0.99:
                errors.append(f"{driver_code}: Skill {skill:.3f} out of valid range [0.1, 0.99]")

            # Historical driver expectations (warning by default)
            if driver_code in DRIVER_EXPECTATION_RULES:
                rules = DRIVER_EXPECTATION_RULES[driver_code]

                if skill < rules["min"]:
                    _record_expectation_violation(
                        errors,
                        warnings,
                        (
                            f"{driver_code}: Skill {skill:.3f} below expected minimum "
                            f"{rules['min']:.3f} for {rules['tier']} driver"
                        ),
                        enforce_expectations=enforce_expectations,
                    )

                if skill > rules["max"]:
                    _record_expectation_violation(
                        errors,
                        warnings,
                        (
                            f"{driver_code}: Skill {skill:.3f} above expected maximum "
                            f"{rules['max']:.3f} for {rules['tier']} driver"
                        ),
                        enforce_expectations=enforce_expectations,
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

        # Distribution sanity: ensure extracted skills are not collapsed or missing.
        if len(skill_values) < 18:
            errors.append(
                f"Only {len(skill_values)} drivers with valid skill scores found (expected >=18)"
            )
        if skill_values:
            spread = max(skill_values) - min(skill_values)
            if spread < 0.10:
                errors.append(
                    f"Driver skill distribution too narrow (spread={spread:.3f}); extraction likely failed"
                )

    except FileNotFoundError:
        errors.append(f"File not found: {driver_file}")
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in {driver_file}")
    except Exception as e:
        errors.append(f"Error reading {driver_file}: {e}")

    return len(errors) == 0, errors, warnings


def validate_team_characteristics(
    team_file: Path, *, enforce_expectations: bool = False
) -> tuple[bool, list[str], list[str]]:
    """Validate team/car characteristics file, returning status, errors, and warnings."""
    errors = []
    warnings = []
    performance_values: list[float] = []

    try:
        with open(team_file) as f:
            data = json.load(f)

        teams = data.get("teams", {})
        is_preseason_baseline = _is_preseason_team_baseline(data)

        for team_name, team_data in teams.items():
            if "overall_performance" not in team_data:
                errors.append(f"{team_name}: Missing 'overall_performance'")
                continue

            performance = team_data["overall_performance"]
            if isinstance(performance, (int | float)):
                performance_values.append(float(performance))

            # Range check
            if performance < 0.1 or performance > 0.99:
                errors.append(
                    f"{team_name}: Performance {performance:.3f} out of valid range [0.1, 0.99]"
                )

            if is_preseason_baseline:
                # Preseason payloads should include uncertainty metadata.
                uncertainty = team_data.get("uncertainty")
                if uncertainty is None:
                    errors.append(
                        f"{team_name}: Missing 'uncertainty' in preseason baseline payload"
                    )
                elif not isinstance(uncertainty, (int | float)):
                    errors.append(
                        f"{team_name}: Uncertainty must be numeric, got {type(uncertainty).__name__}"
                    )
                elif uncertainty < 0 or uncertainty > 1:
                    errors.append(
                        f"{team_name}: Uncertainty {uncertainty:.3f} out of valid range [0.0, 1.0]"
                    )
            elif team_name in TEAM_EXPECTATION_RULES:
                # Historical team expectations (warning by default)
                rules = TEAM_EXPECTATION_RULES[team_name]

                if performance < rules["min"]:
                    _record_expectation_violation(
                        errors,
                        warnings,
                        (
                            f"{team_name}: Performance {performance:.3f} below expected minimum "
                            f"{rules['min']:.3f}"
                        ),
                        enforce_expectations=enforce_expectations,
                    )

                if performance > rules["max"]:
                    _record_expectation_violation(
                        errors,
                        warnings,
                        (
                            f"{team_name}: Performance {performance:.3f} above expected maximum "
                            f"{rules['max']:.3f}"
                        ),
                        enforce_expectations=enforce_expectations,
                    )

        if len(performance_values) < 10:
            errors.append(
                f"Only {len(performance_values)} teams with valid performance values found (expected >=10)"
            )
        if performance_values:
            spread = max(performance_values) - min(performance_values)
            if not is_preseason_baseline and spread < 0.05:
                errors.append(
                    "Team performance distribution too narrow for in-season data "
                    f"(spread={spread:.3f})"
                )

    except FileNotFoundError:
        errors.append(f"File not found: {team_file}")
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in {team_file}")
    except Exception as e:
        errors.append(f"Error reading {team_file}: {e}")

    return len(errors) == 0, errors, warnings


def validate_track_characteristics(track_file: Path) -> tuple[bool, list[str], list[str]]:
    """Validate track characteristics file, returning status, errors, and warnings."""
    errors = []
    warnings = []

    try:
        with open(track_file) as f:
            data = json.load(f)

        tracks = data.get("tracks", {})

        overtaking_values: list[float] = []

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
                else:
                    overtaking_values.append(float(ot_diff))

        if len(overtaking_values) >= 10:
            rounded_values = sorted({round(value, 2) for value in overtaking_values})
            spread = max(overtaking_values) - min(overtaking_values)
            if len(rounded_values) <= 3 and spread <= 0.10:
                errors.append(
                    "Track overtaking difficulty distribution is collapsed "
                    f"(unique_values={rounded_values}, spread={spread:.2f}). "
                    "Rebuild the track dataset instead of shipping placeholder-like values."
                )

    except FileNotFoundError:
        errors.append(f"File not found: {track_file}")
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON in {track_file}")
    except Exception as e:
        errors.append(f"Error reading {track_file}: {e}")

    return len(errors) == 0, errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate characteristics files for sanity")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing characteristics files",
    )
    parser.add_argument(
        "--enforce-expectations",
        action="store_true",
        help=(
            "Treat historical driver/team expectation deviations as hard errors. "
            "By default these are warnings so preseason neutral baselines can pass."
        ),
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help=(
            "Season year to validate. Defaults to the latest season-scoped files "
            "found in each characteristics directory."
        ),
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Characteristics Validation")
    print("=" * 60)
    print()

    all_valid = True
    all_errors = []
    all_warnings = []

    # Validate driver characteristics
    print("1. Validating driver characteristics...")
    driver_file = _resolve_driver_characteristics_file(data_dir, args.season_year)
    print(f"   Using {driver_file}")
    driver_valid, driver_errors, driver_warnings = validate_driver_characteristics(
        driver_file, enforce_expectations=args.enforce_expectations
    )

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
    if driver_warnings:
        print(f"   [WARN] Found {len(driver_warnings)} expectation warnings:")
        for warning in driver_warnings[:10]:
            print(f"      - {warning}")
        if len(driver_warnings) > 10:
            print(f"      ... and {len(driver_warnings) - 10} more")
        all_warnings.extend(driver_warnings)

    print()

    # Validate team characteristics
    print("2. Validating team characteristics...")
    team_file = _resolve_season_scoped_file(
        data_dir=data_dir,
        subdir="car_characteristics",
        filename_suffix="car_characteristics",
        season_year=args.season_year,
    )
    print(f"   Using {team_file}")
    team_valid, team_errors, team_warnings = validate_team_characteristics(
        team_file, enforce_expectations=args.enforce_expectations
    )

    if team_valid:
        print("   [OK] Team characteristics VALID")
    else:
        print(f"   [ERROR] Found {len(team_errors)} errors:")
        for error in team_errors:
            print(f"      - {error}")
        all_valid = False
        all_errors.extend(team_errors)
    if team_warnings:
        print(f"   [WARN] Found {len(team_warnings)} expectation warnings:")
        for warning in team_warnings:
            print(f"      - {warning}")
        all_warnings.extend(team_warnings)

    print()

    # Validate track characteristics
    print("3. Validating track characteristics...")
    track_file = _resolve_season_scoped_file(
        data_dir=data_dir,
        subdir="track_characteristics",
        filename_suffix="track_characteristics",
        season_year=args.season_year,
    )
    print(f"   Using {track_file}")
    track_valid, track_errors, track_warnings = validate_track_characteristics(track_file)

    if track_valid:
        print("   [OK] Track characteristics VALID")
    else:
        print(f"   [ERROR] Found {len(track_errors)} errors:")
        for error in track_errors:
            print(f"      - {error}")
        all_valid = False
        all_errors.extend(track_errors)
    if track_warnings:
        print(f"   [WARN] Found {len(track_warnings)} warnings:")
        for warning in track_warnings:
            print(f"      - {warning}")
        all_warnings.extend(track_warnings)

    print()
    print("=" * 60)

    if all_valid:
        print("[OK] All characteristics files are VALID!")
        if all_warnings:
            print(f"[WARN] Validation passed with {len(all_warnings)} expectation warnings")
        print("=" * 60)
        return 0
    else:
        print(f"[ERROR] Validation FAILED with {len(all_errors)} total errors")
        print("=" * 60)
        print()
        print("[WARN] Data has blocking validation errors and should not be used for predictions.")
        print("   To regenerate characteristics, run:")
        print("   1) python scripts/extract_driver_characteristics.py --years 2023,2024,2025")
        print(
            "      Validator accepts either season-scoped driver output under "
            "data/processed/driver_characteristics/<year>_driver_characteristics.json"
        )
        print("      or the legacy fallback data/processed/driver_characteristics.json")
        print(
            "   2) python scripts/generate_2026_baseline.py --years 2023,2024,2025 --output data/processed"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
