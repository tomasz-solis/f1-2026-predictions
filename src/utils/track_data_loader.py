"""Track-specific data loader for race simulation parameters."""

import json
import logging
from functools import lru_cache
from pathlib import Path

import fastf1

from src.utils import config_loader

logger = logging.getLogger(__name__)

KNOWN_MAIN_RACE_LAPS: dict[str, int] = {
    "Bahrain Grand Prix": 57,
    "Saudi Arabian Grand Prix": 50,
    "Australian Grand Prix": 58,
    "Japanese Grand Prix": 53,
    "Chinese Grand Prix": 56,
    "Miami Grand Prix": 57,
    "Monaco Grand Prix": 78,
    "Spanish Grand Prix": 66,
    "Canadian Grand Prix": 70,
    "Austrian Grand Prix": 71,
    "British Grand Prix": 52,
    "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,
    "Singapore Grand Prix": 62,
    "United States Grand Prix": 56,
    "Mexico City Grand Prix": 71,
    "Brazilian Grand Prix": 71,
    "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57,
    "Abu Dhabi Grand Prix": 58,
}

KNOWN_SPRINT_LAPS: dict[str, int] = {
    "Bahrain Grand Prix": 19,
    "Saudi Arabian Grand Prix": 19,
    "Australian Grand Prix": 19,
    "Japanese Grand Prix": 17,
    "Chinese Grand Prix": 19,
    "Miami Grand Prix": 19,
    "Monaco Grand Prix": 26,
    "Spanish Grand Prix": 22,
    "Canadian Grand Prix": 18,
    "Austrian Grand Prix": 24,
    "British Grand Prix": 17,
    "Belgian Grand Prix": 15,
    "Dutch Grand Prix": 24,
    "Italian Grand Prix": 18,
    "Singapore Grand Prix": 20,
    "United States Grand Prix": 19,
    "Mexico City Grand Prix": 24,
    "Brazilian Grand Prix": 24,
    "Las Vegas Grand Prix": 17,
    "Qatar Grand Prix": 19,
    "Abu Dhabi Grand Prix": 20,
}

# Conservative overtaking-difficulty priors used when extracted files are clearly
# under-scaled (e.g., all tracks compressed around 0.00-0.05).
_TRACK_OVERTAKING_BASELINES: dict[str, float] = {
    "Bahrain Grand Prix": 0.40,
    "Saudi Arabian Grand Prix": 0.60,
    "Australian Grand Prix": 0.50,
    "Japanese Grand Prix": 0.50,
    "Chinese Grand Prix": 0.30,
    "Miami Grand Prix": 0.50,
    "Monaco Grand Prix": 0.95,
    "Spanish Grand Prix": 0.40,
    "Canadian Grand Prix": 0.50,
    "Austrian Grand Prix": 0.40,
    "British Grand Prix": 0.40,
    "Hungarian Grand Prix": 0.80,
    "Belgian Grand Prix": 0.30,
    "Dutch Grand Prix": 0.50,
    "Italian Grand Prix": 0.20,
    "Singapore Grand Prix": 0.80,
    "United States Grand Prix": 0.40,
    "Mexico City Grand Prix": 0.40,
    "Brazilian Grand Prix": 0.40,
    "Las Vegas Grand Prix": 0.30,
    "Qatar Grand Prix": 0.40,
    "Abu Dhabi Grand Prix": 0.50,
}

_UNDERSCALED_OVERTAKING_THRESHOLD = 0.10


def _pirelli_candidate_years(year: int) -> list[int]:
    """Return candidate Pirelli-data seasons in priority order."""
    candidates = [int(year)]
    if year > 0:
        candidates.append(int(year) - 1)
    if 2025 not in candidates:
        candidates.append(2025)
    return list(dict.fromkeys(candidates))


def _resolve_track_characteristics_path(year: int) -> Path | None:
    """Resolve season-aware track characteristics file with conservative fallback."""
    processed_root = Path(config_loader.get("paths.processed", "data/processed"))
    candidates = [int(year)]
    if year > 0:
        candidates.append(int(year) - 1)
    if 2026 not in candidates:
        candidates.append(2026)

    for candidate_year in dict.fromkeys(candidates):
        candidate_path = (
            processed_root
            / "track_characteristics"
            / f"{int(candidate_year)}_track_characteristics.json"
        )
        if candidate_path.exists():
            return candidate_path
    return None


def _resolve_pirelli_path(year: int) -> Path | None:
    """Resolve season-aware Pirelli file with fallback ordering."""
    for candidate_year in _pirelli_candidate_years(year):
        candidate_path = Path("data") / f"{candidate_year}_pirelli_info.json"
        if candidate_path.exists():
            return candidate_path
    return None


def _normalize_overtaking_difficulty(race_name: str, raw_value: object) -> float | None:
    """Normalize overtaking difficulty to a bounded 0..1 scale."""
    if raw_value is None:
        return None
    if not isinstance(raw_value, (int | float | str)):
        return None
    try:
        overtaking = float(raw_value)
    except (TypeError, ValueError):
        return None

    overtaking = float(max(0.0, min(1.0, overtaking)))

    # Some generated files compress values to ~0.00-0.05 for most tracks, which
    # unrealistically treats nearly every circuit as easy to overtake.
    if overtaking <= _UNDERSCALED_OVERTAKING_THRESHOLD and race_name in _TRACK_OVERTAKING_BASELINES:
        baseline = _TRACK_OVERTAKING_BASELINES[race_name]
        logger.info(
            "Overtaking difficulty for %s appears under-scaled (%.3f); using baseline %.2f",
            race_name,
            overtaking,
            baseline,
        )
        return baseline

    return overtaking


def load_track_specific_params(race_name: str | None = None, year: int = 2026) -> dict:
    """Load track-specific parameters from track_characteristics.

    Returns dict with track-specific overrides for race simulation:
        - pit_stops.loss_duration: seconds (from track_characteristics)
        - sc_probability: safety car probability (from track_characteristics)
        - track_overtaking: overtaking difficulty (from track_characteristics)

    Falls back to config defaults if track_name not found or data missing.
    """
    track_params: dict[str, float | dict[str, float]] = {}

    if race_name:
        track_chars_path = _resolve_track_characteristics_path(year)
        if track_chars_path is None:
            logger.warning(
                "Track characteristics file not found for year %s (or fallbacks). "
                "Using config defaults.",
                year,
            )
            return track_params

        try:
            with open(track_chars_path) as f:
                track_data = json.load(f)

            tracks = track_data.get("tracks", {})
            track_info = tracks.get(race_name)

            if track_info:
                # Extract track-specific pit stop loss
                pit_loss = track_info.get("pit_stop_loss")
                if pit_loss is not None:
                    track_params["pit_stops"] = {"loss_duration": float(pit_loss)}
                    logger.info(f"Loaded track-specific pit stop loss for {race_name}: {pit_loss}s")

                # Extract safety car probability
                sc_prob = track_info.get("safety_car_prob")
                if sc_prob is not None:
                    track_params["sc_probability"] = float(sc_prob)

                # Extract overtaking difficulty
                overtaking = _normalize_overtaking_difficulty(
                    race_name,
                    track_info.get("overtaking_difficulty"),
                )
                if overtaking is not None:
                    track_params["track_overtaking"] = overtaking

            else:
                logger.warning(
                    f"Track '{race_name}' not found in track_characteristics. "
                    "Using config defaults."
                )

        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse track characteristics JSON at {track_chars_path}. "
                "Using config defaults."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error loading track characteristics: {e}. Using config defaults."
            )

    return track_params


def get_tire_stress_score(race_name: str | None = None, year: int = 2026) -> float:
    """Get tire stress score for race from Pirelli data.

    Returns average of traction + braking + lateral + abrasion.
    Defaults to 3.0 (medium stress) if data missing.
    """
    if not race_name:
        return config_loader.get(
            "baseline_predictor.compound_selection.default_stress_fallback", 3.0
        )

    pirelli_path = _resolve_pirelli_path(year)
    if pirelli_path is None:
        logger.warning(
            "Pirelli data file not found for year %s (or fallbacks). Using default stress (3.0).",
            year,
        )
        return config_loader.get(
            "baseline_predictor.compound_selection.default_stress_fallback", 3.0
        )

    try:
        with open(pirelli_path) as f:
            pirelli_data = json.load(f)

        # Normalize race name (lowercase, underscores)
        race_key = race_name.lower().replace(" ", "_")
        race_info = pirelli_data.get(race_key)

        if race_info and "tyre_stress" in race_info:
            tyre_stress = race_info["tyre_stress"]

            # Calculate average stress from key metrics
            stress_score = (
                tyre_stress.get("traction", 3.0)
                + tyre_stress.get("braking", 3.0)
                + tyre_stress.get("lateral", 3.0)
                + tyre_stress.get("asphalt_abrasion", 3.0)
            ) / 4.0

            return float(stress_score)
        else:
            logger.warning(f"Tire stress data not found for {race_name}. Using default (3.0).")

    except Exception as e:
        logger.error(f"Error loading Pirelli data: {e}. Using default stress (3.0).")

    # Fallback to config default
    return config_loader.get("baseline_predictor.compound_selection.default_stress_fallback", 3.0)


def get_available_compounds(race_name: str | None = None, weather: str = "dry") -> list[str]:
    """Get list of available tire compounds for race.

    Weather-aware approximation:
    - dry: dry compounds only
    - rain: wet compounds only
    - mixed: dry compounds + intermediate
    """
    weather_key = (weather or "dry").strip().lower()
    if weather_key == "rain":
        return ["INTERMEDIATE", "WET"]
    if weather_key == "mixed":
        return ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]

    return ["SOFT", "MEDIUM", "HARD"]


@lru_cache(maxsize=128)
def resolve_race_distance_laps(year: int, race_name: str | None, is_sprint: bool) -> int:
    """
    Resolve race distance in laps from FastF1 session metadata.

    Falls back to conservative defaults when metadata is unavailable.
    """
    default_distance = 20 if is_sprint else 60
    if not race_name:
        return default_distance

    known_laps = (KNOWN_SPRINT_LAPS if is_sprint else KNOWN_MAIN_RACE_LAPS).get(race_name)
    if known_laps:
        return known_laps

    session_name = "S" if is_sprint else "R"
    try:
        session = fastf1.get_session(year, race_name, session_name)
        if session is None:
            return default_distance

        total_laps = getattr(session, "total_laps", None)
        if total_laps:
            return max(1, int(total_laps))

        # Metadata load is enough for total_laps; telemetry/laps are unnecessary here.
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        total_laps = getattr(session, "total_laps", None)
        if total_laps:
            return max(1, int(total_laps))
    except Exception as exc:
        logger.warning(
            f"Could not resolve race distance for {race_name} ({year}, {session_name}): {exc}. "
            f"Using fallback {default_distance} laps."
        )

    return default_distance
