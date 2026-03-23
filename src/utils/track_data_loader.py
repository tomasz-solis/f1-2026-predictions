"""Track-specific data loader for race simulation parameters."""

import json
import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import fastf1
import numpy as np
from fastf1.exceptions import DataNotLoadedError

from src.utils import config_loader
from src.utils.track_overtaking import get_track_overtaking_baseline

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

_OVERTAKING_DIFFICULTY_LABELS: dict[str, float] = {
    "very_hard": 0.95,
    "hard": 0.75,
    "moderate": 0.55,
    "easy": 0.35,
    "very_easy": 0.20,
}

_UNDERSCALED_OVERTAKING_THRESHOLD = 0.10
_CONVENTIONAL_TEMP_PRIORITY = ("R", "Q", "FP3", "FP2", "FP1")
_SPRINT_TEMP_PRIORITY = ("R", "Q", "Sprint", "SQ", "FP1")
_CONVENTIONAL_NON_COMPETITIVE_PRIORITY = ("FP3", "FP2", "FP1")
_SPRINT_NON_COMPETITIVE_PRIORITY = ("FP1",)
_SESSION_DURATION_HOURS = {
    "FP1": 1.5,
    "FP2": 1.5,
    "FP3": 1.5,
    "SQ": 1.5,
    "Sprint": 1.0,
    "Q": 1.5,
    "R": 2.5,
}
_FINAL_STATUS_TOKENS = ("FINISHED", "FINALISED", "FINALIZED", "ENDED", "ABORTED")
_ACTIVE_STATUS_TOKENS = ("STARTED", "GREEN", "RUNNING", "RESTART", "SUSPENDED")
_DEFAULT_TEMPERATURE_SESSION_BLEND_WEIGHT = 0.70
_SESSION_TEMPERATURE_BLEND_WEIGHTS = {
    "R": 0.90,
    "Q": 0.80,
    "Sprint": 0.80,
    "SQ": 0.75,
    "FP3": 0.70,
    "FP2": 0.65,
    "FP1": 0.60,
}


def _coerce_float(value: object) -> float | None:
    """Best-effort float coercion for mypy-friendly dict[str, object] payloads."""
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_unit_interval_value(value: object) -> float | None:
    """Coerce value to a 0..1 float; supports percentages and common labels."""
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_")
        if not normalized:
            return None
        if normalized in _OVERTAKING_DIFFICULTY_LABELS:
            return _OVERTAKING_DIFFICULTY_LABELS[normalized]
        try:
            numeric_value = float(normalized)
        except ValueError:
            return None
    elif isinstance(value, int | float):
        numeric_value = float(value)
    else:
        return None

    if numeric_value > 1.0 and numeric_value <= 100.0:
        numeric_value /= 100.0
    return float(max(0.0, min(1.0, numeric_value)))


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


def _default_track_temperature_c(weather: str) -> float:
    """Return fallback track temperature from config by weather bucket."""
    weather_key = str(weather or "dry").strip().lower()
    if weather_key == "rain":
        return float(config_loader.get("baseline_predictor.race.track_temperature.rain_c", 23.0))
    if weather_key == "mixed":
        return float(config_loader.get("baseline_predictor.race.track_temperature.mixed_c", 29.0))
    return float(config_loader.get("baseline_predictor.race.track_temperature.dry_c", 36.0))


def _coerce_utc_datetime(value: object) -> datetime | None:
    """Coerce datetime-like objects to timezone-aware UTC datetimes."""
    if value is None:
        return None

    if hasattr(value, "to_pydatetime"):
        try:
            value = value.to_pydatetime()
        except Exception:
            return None

    if not isinstance(value, datetime):
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _session_status_completed(session: object) -> bool | None:
    """Return completion status from FastF1 session status feed when available."""
    try:
        status_feed = session.session_status
    except (AttributeError, DataNotLoadedError):
        return None

    if status_feed is None or getattr(status_feed, "empty", False):
        return None

    columns = getattr(status_feed, "columns", [])
    status_values = None
    for column in ("Status", "SessionStatus", "Message"):
        if column in columns:
            status_values = status_feed[column]
            break
    if status_values is None:
        return None

    try:
        cleaned = status_values.dropna().astype(str)
    except Exception:
        return None
    if cleaned.empty:
        return None

    latest = cleaned.iloc[-1].upper()
    if any(token in latest for token in _FINAL_STATUS_TOKENS):
        return True
    if any(token in latest for token in _ACTIVE_STATUS_TOKENS):
        return False
    return None


def _normalize_weather_column_name(name: object) -> str:
    """Normalize weather column names for resilient matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _weather_metric_median(weather_data: object, metric_token: str) -> float | None:
    """Return median weather metric value for the matching column."""
    df = cast(Any, weather_data)  # FastF1 provides a pandas DataFrame

    columns = getattr(df, "columns", [])
    selected_column = None
    for column in columns:
        normalized = _normalize_weather_column_name(column)
        if metric_token in normalized:
            selected_column = column
            break

    if selected_column is None:
        return None

    try:
        values = df[selected_column].dropna().astype(float)
    except Exception:
        return None
    if values.empty:
        return None

    return float(values.median())


def _clamp_track_temperature_c(track_temp_c: float) -> float:
    """Clamp track temperature to a realistic range."""
    min_temp_c = float(config_loader.get("baseline_predictor.race.track_temperature.min_c", 5.0))
    max_temp_c = float(config_loader.get("baseline_predictor.race.track_temperature.max_c", 65.0))
    return float(max(min_temp_c, min(max_temp_c, float(track_temp_c))))


def _resolve_session_temperature_blend_weight(session_name: str) -> float:
    """Resolve blend weight for session signal vs race-weather fallback baseline."""
    configured_default = config_loader.get(
        "baseline_predictor.race.track_temperature.blend.session_weight",
        _DEFAULT_TEMPERATURE_SESSION_BLEND_WEIGHT,
    )
    try:
        default_weight = float(configured_default)
    except (TypeError, ValueError):
        default_weight = _DEFAULT_TEMPERATURE_SESSION_BLEND_WEIGHT

    configured_by_session = config_loader.get(
        "baseline_predictor.race.track_temperature.blend.session_weight_by_session",
        {},
    )
    if isinstance(configured_by_session, dict):
        raw_weight = configured_by_session.get(
            session_name,
            _SESSION_TEMPERATURE_BLEND_WEIGHTS.get(session_name, default_weight),
        )
    else:
        raw_weight = _SESSION_TEMPERATURE_BLEND_WEIGHTS.get(session_name, default_weight)

    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        weight = default_weight
    return float(max(0.0, min(1.0, weight)))


def _session_scheduled_end_passed(event: object, session_name: str, now_utc: datetime) -> bool:
    """Return whether scheduled session end has passed based on FastF1 event metadata."""
    try:
        raw_start = event.get_session_date(session_name)
    except Exception:
        return True

    session_start = _coerce_utc_datetime(raw_start)
    if session_start is None:
        return True

    duration_hours = float(_SESSION_DURATION_HOURS.get(session_name, 2.0))
    return now_utc >= (session_start + timedelta(hours=duration_hours))


def _load_session_track_temperature_c(
    *,
    year: int,
    race_name: str,
    session_name: str,
) -> float | None:
    """Load representative track temperature from a specific FastF1 session."""
    signal = _load_session_temperature_signal(
        year=year,
        race_name=race_name,
        session_name=session_name,
    )
    if signal is None:
        return None

    temp = _coerce_float(signal.get("session_track_temperature_c"))
    if temp is None:
        return None
    return temp


def _load_session_temperature_signal(
    *,
    year: int,
    race_name: str,
    session_name: str,
) -> dict[str, object] | None:
    """Load session temperature signal with provenance metadata."""
    try:
        session = fastf1.get_session(year, race_name, session_name)
    except Exception as exc:
        logger.debug(
            "Could not create FastF1 session for %s %s %s while resolving track temperature: %s",
            year,
            race_name,
            session_name,
            exc,
        )
        return None

    if session is None:
        return None

    try:
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        logger.debug(
            "Could not load FastF1 weather for %s %s %s: %s",
            year,
            race_name,
            session_name,
            exc,
        )
        return None

    status_completed = _session_status_completed(session)
    if status_completed is False:
        return None

    weather_data = getattr(session, "weather_data", None)
    if weather_data is None or getattr(weather_data, "empty", False):
        return None

    track_temp_c = _weather_metric_median(weather_data, "tracktemp")
    if track_temp_c is not None:
        return {
            "session_track_temperature_c": _clamp_track_temperature_c(track_temp_c),
            "session_temperature_source": "track_temp",
            "session_air_temperature_c": None,
        }

    # Fallback inference: estimate track temp from air temp when TrackTemp is missing.
    air_temp_c = _weather_metric_median(weather_data, "airtemp")
    if air_temp_c is None:
        return None
    offset_c = float(
        config_loader.get("baseline_predictor.race.track_temperature.air_to_track_offset_c", 9.0)
    )
    return {
        "session_track_temperature_c": _clamp_track_temperature_c(air_temp_c + offset_c),
        "session_temperature_source": "air_temp_inferred",
        "session_air_temperature_c": float(air_temp_c),
    }


def _fallback_temperature_profile(
    *,
    fallback_temp_c: float,
    weather: str,
    reason: str,
) -> dict[str, object]:
    """Build forecast-only track-temperature profile for downstream transparency."""
    return {
        "track_temperature_c": float(fallback_temp_c),
        "source": "forecast_fallback",
        "reason": reason,
        "weather_bucket": weather,
        "session_name": None,
        "session_track_temperature_c": None,
        "session_temperature_source": None,
        "session_air_temperature_c": None,
        "forecast_track_temperature_c": float(fallback_temp_c),
        "session_weight": 0.0,
        "forecast_weight": 1.0,
        "blend_enabled": False,
    }


def _normalize_rainfall_signal(rainfall_value: float | None) -> float | None:
    """Normalize rainfall to a 0..1 wetness signal."""
    if rainfall_value is None:
        return None
    try:
        value = float(rainfall_value)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return 0.0
    return float(min(1.0, value))


def _infer_weather_bucket_from_rainfall_signal(rainfall_signal: float | None) -> str:
    """Infer dry/mixed/rain weather bucket from normalized rainfall signal."""
    if rainfall_signal is None:
        return "unknown"
    if rainfall_signal >= 0.60:
        return "rain"
    if rainfall_signal >= 0.15:
        return "mixed"
    return "dry"


def _load_session_weather_features(
    *,
    year: int,
    race_name: str,
    session_name: str,
) -> dict[str, object] | None:
    """Load weather summary features from a specific FastF1 session."""
    try:
        session = fastf1.get_session(year, race_name, session_name)
    except Exception as exc:
        logger.debug(
            "Could not create FastF1 session for %s %s %s while resolving weather features: %s",
            year,
            race_name,
            session_name,
            exc,
        )
        return None

    if session is None:
        return None

    try:
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        logger.debug(
            "Could not load FastF1 weather for %s %s %s while resolving weather features: %s",
            year,
            race_name,
            session_name,
            exc,
        )
        return None

    status_completed = _session_status_completed(session)
    if status_completed is False:
        return None

    weather_data = getattr(session, "weather_data", None)
    if weather_data is None or getattr(weather_data, "empty", False):
        return None

    track_temp_c = _weather_metric_median(weather_data, "tracktemp")
    air_temp_c = _weather_metric_median(weather_data, "airtemp")
    if track_temp_c is None and air_temp_c is not None:
        offset_c = float(
            config_loader.get(
                "baseline_predictor.race.track_temperature.air_to_track_offset_c", 9.0
            )
        )
        track_temp_c = air_temp_c + offset_c

    wind_speed_kph = _weather_metric_median(weather_data, "windspeed")
    humidity_pct = _weather_metric_median(weather_data, "humidity")
    rainfall_signal = _normalize_rainfall_signal(_weather_metric_median(weather_data, "rainfall"))

    has_feature = any(
        value is not None
        for value in (track_temp_c, air_temp_c, wind_speed_kph, humidity_pct, rainfall_signal)
    )
    if not has_feature:
        return None

    return {
        "track_temperature_c": _clamp_track_temperature_c(track_temp_c)
        if track_temp_c is not None
        else None,
        "air_temperature_c": float(air_temp_c) if air_temp_c is not None else None,
        "wind_speed_kph": float(wind_speed_kph) if wind_speed_kph is not None else None,
        "humidity_pct": float(humidity_pct) if humidity_pct is not None else None,
        "rainfall_signal": rainfall_signal,
    }


def _normalize_overtaking_difficulty(
    race_name: str,
    raw_value: object,
    *,
    raw_likelihood: object | None = None,
) -> float | None:
    """Normalize overtaking difficulty to a bounded 0..1 scale.

    Supports three encodings commonly found in generated files:
    - numeric difficulty in ``[0, 1]`` (canonical)
    - numeric percentages in ``[0, 100]``
    - categorical labels like ``hard`` or ``very_easy``

    If the resolved value still looks under-scaled, we first try to infer
    difficulty from ``overtaking_likelihood`` (difficulty ~= ``1 - likelihood``)
    before falling back to track priors.
    """
    overtaking = _coerce_unit_interval_value(raw_value)
    likelihood = _coerce_unit_interval_value(raw_likelihood)

    if overtaking is None and likelihood is not None:
        overtaking = float(max(0.0, min(1.0, 1.0 - likelihood)))
        return overtaking

    if overtaking is None:
        return None

    # Some generated files compress values to ~0.00-0.05 for most tracks, which
    # unrealistically treats nearly every circuit as easy to overtake.
    if overtaking <= _UNDERSCALED_OVERTAKING_THRESHOLD:
        if likelihood is not None:
            inferred = float(max(0.0, min(1.0, 1.0 - likelihood)))
            if inferred > _UNDERSCALED_OVERTAKING_THRESHOLD:
                logger.info(
                    "Overtaking difficulty for %s appears under-scaled (%.3f); inferred %.2f from likelihood",
                    race_name,
                    overtaking,
                    inferred,
                )
                return inferred

        baseline = get_track_overtaking_baseline(
            race_name,
            default=float(config_loader.get("track_defaults.overtaking_difficulty", 0.5)),
        )
        logger.info(
            "Overtaking difficulty for %s appears under-scaled (%.3f); using baseline %.2f",
            race_name,
            overtaking,
            baseline,
        )
        return baseline

    return overtaking


def _blend_overtaking_with_transition_prior(
    race_name: str,
    observed_overtaking: float,
    *,
    observed_races: int | None,
) -> float:
    """Blend observed overtaking with track prior for gradual regulation adaptation.

    When a track payload includes ``overtaking_observed_races``, this applies a
    bounded transition from historical prior to observed value. Early races
    nudge the prior; larger evidence allows stronger movement.
    """
    if observed_races is None:
        return observed_overtaking
    if observed_races <= 0:
        return observed_overtaking

    prior = get_track_overtaking_baseline(
        race_name,
        default=float(config_loader.get("track_defaults.overtaking_difficulty", 0.5)),
    )

    min_weight = float(
        config_loader.get(
            "baseline_predictor.race.overtaking_transition.min_observed_weight",
            0.12,
        )
    )
    max_weight = float(
        config_loader.get(
            "baseline_predictor.race.overtaking_transition.max_observed_weight",
            0.65,
        )
    )
    races_to_full = int(
        config_loader.get(
            "baseline_predictor.race.overtaking_transition.races_to_full_weight",
            8,
        )
    )
    max_delta = float(
        config_loader.get(
            "baseline_predictor.race.overtaking_transition.max_delta_from_prior",
            0.25,
        )
    )

    races_to_full = max(1, races_to_full)
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    max_weight = float(np.clip(max_weight, min_weight, 1.0))
    max_delta = float(max(0.0, min(1.0, max_delta)))

    evidence_ratio = float(np.clip(float(observed_races) / float(races_to_full), 0.0, 1.0))
    observed_weight = min_weight + ((max_weight - min_weight) * evidence_ratio)

    capped_delta = float(np.clip(observed_overtaking - prior, -max_delta, max_delta))
    blended = float(np.clip(prior + (capped_delta * observed_weight), 0.0, 1.0))

    logger.info(
        "Blended overtaking difficulty for %s using transition prior: prior=%.2f observed=%.2f races=%s weight=%.2f result=%.2f",
        race_name,
        prior,
        observed_overtaking,
        observed_races,
        observed_weight,
        blended,
    )
    return blended


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
                    raw_likelihood=track_info.get("overtaking_likelihood"),
                )
                if overtaking is not None:
                    observed_races_raw = _coerce_float(track_info.get("overtaking_observed_races"))
                    observed_races = (
                        int(observed_races_raw) if observed_races_raw is not None else None
                    )
                    overtaking = _blend_overtaking_with_transition_prior(
                        race_name,
                        overtaking,
                        observed_races=observed_races,
                    )
                    track_params["track_overtaking"] = overtaking

                # Extract lap 1 track-specific risk modifier
                lap1_risk = track_info.get("lap1_risk_modifier")
                if lap1_risk is not None:
                    track_params["lap1_risk_modifier"] = float(lap1_risk)

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


@lru_cache(maxsize=256)
def resolve_track_temperature_profile(
    year: int,
    race_name: str | None,
    weather: str = "dry",
    is_sprint: bool = False,
) -> dict[str, object]:
    """
    Resolve track temperature profile from session weather and forecast weighting.

    The resolver checks latest completed sessions in reverse weekend order. When
    a session signal is available, it is blended with race-weather fallback
    temperature using session-specific weights. Returned metadata is used in UI.
    """
    weather_key = str(weather or "dry").strip().lower()
    fallback_temp_c = _default_track_temperature_c(weather_key)
    if not race_name:
        return _fallback_temperature_profile(
            fallback_temp_c=fallback_temp_c,
            weather=weather_key,
            reason="missing_race_name",
        )

    try:
        event = fastf1.get_event(year, race_name)
    except Exception as exc:
        logger.info(
            "Could not load FastF1 event while resolving track temperature for %s %s: %s. "
            "Using fallback %.1fC.",
            race_name,
            year,
            exc,
            fallback_temp_c,
        )
        return _fallback_temperature_profile(
            fallback_temp_c=fallback_temp_c,
            weather=weather_key,
            reason="event_load_failed",
        )

    session_priority = _SPRINT_TEMP_PRIORITY if is_sprint else _CONVENTIONAL_TEMP_PRIORITY
    now_utc = datetime.now(UTC)
    blend_enabled = bool(
        config_loader.get("baseline_predictor.race.track_temperature.blend.enabled", True)
    )

    for session_name in session_priority:
        if not _session_scheduled_end_passed(event, session_name, now_utc):
            continue

        session_signal = _load_session_temperature_signal(
            year=year,
            race_name=race_name,
            session_name=session_name,
        )
        if session_signal is not None:
            session_track_temp_c = _coerce_float(session_signal.get("session_track_temperature_c"))
            if session_track_temp_c is None:
                continue
            session_weight = 1.0
            forecast_weight = 0.0
            final_track_temp_c = session_track_temp_c
            source = "session_weather"
            if blend_enabled:
                session_weight = _resolve_session_temperature_blend_weight(session_name)
                forecast_weight = float(max(0.0, min(1.0, 1.0 - session_weight)))
                final_track_temp_c = _clamp_track_temperature_c(
                    (session_track_temp_c * session_weight) + (fallback_temp_c * forecast_weight)
                )
                source = "session_weather_blend"

            logger.info(
                (
                    "Resolved track temperature %.1fC for %s %s using %s session "
                    "signal from %s (session=%.2f, forecast=%.2f)."
                ),
                final_track_temp_c,
                year,
                race_name,
                session_name,
                str(session_signal.get("session_temperature_source", "")),
                session_weight,
                forecast_weight,
            )
            return {
                "track_temperature_c": float(final_track_temp_c),
                "source": source,
                "reason": "session_signal_available",
                "weather_bucket": weather_key,
                "session_name": session_name,
                "session_track_temperature_c": session_track_temp_c,
                "session_temperature_source": str(
                    session_signal.get("session_temperature_source", "")
                ),
                "session_air_temperature_c": session_signal.get("session_air_temperature_c"),
                "forecast_track_temperature_c": float(fallback_temp_c),
                "session_weight": float(session_weight),
                "forecast_weight": float(forecast_weight),
                "blend_enabled": bool(blend_enabled),
            }

    logger.info(
        "No reliable FastF1 weather data found for %s %s; using fallback %.1fC.",
        race_name,
        year,
        fallback_temp_c,
    )
    return _fallback_temperature_profile(
        fallback_temp_c=fallback_temp_c,
        weather=weather_key,
        reason="no_completed_session_weather",
    )


@lru_cache(maxsize=256)
def resolve_track_temperature_c(
    year: int,
    race_name: str | None,
    weather: str = "dry",
    is_sprint: bool = False,
) -> float:
    """
    Resolve scalar track temperature for simulation backward compatibility.

    This wrapper delegates to the profile resolver and returns only the final
    temperature consumed by lap-by-lap simulation.
    """
    profile = resolve_track_temperature_profile(
        year=year,
        race_name=race_name,
        weather=weather,
        is_sprint=is_sprint,
    )
    try:
        value = _coerce_float(profile.get("track_temperature_c"))
        if value is not None:
            return value
        return _default_track_temperature_c(weather)
    except Exception:
        return _default_track_temperature_c(weather)


@lru_cache(maxsize=256)
def resolve_non_competitive_weather_features(
    year: int,
    race_name: str | None,
    is_sprint: bool = False,
) -> dict[str, object]:
    """
    Resolve weather features from latest completed non-competitive session.

    Non-competitive sessions are practice sessions (FP1/FP2/FP3). Sprint and
    qualifying sessions are intentionally excluded so this signal reflects
    weekend preparation context instead of competitive execution context.
    """
    fallback: dict[str, object] = {
        "available": False,
        "source_session": None,
        "reason": "missing_race_name",
        "practice_weather_bucket": "unknown",
        "track_temperature_c": None,
        "air_temperature_c": None,
        "wind_speed_kph": None,
        "humidity_pct": None,
        "rainfall_signal": None,
    }
    if not race_name:
        return fallback

    try:
        event = fastf1.get_event(year, race_name)
    except Exception as exc:
        logger.info(
            "Could not load FastF1 event while resolving non-competitive weather features "
            "for %s %s: %s.",
            race_name,
            year,
            exc,
        )
        return {**fallback, "reason": "event_load_failed"}

    session_priority = (
        _SPRINT_NON_COMPETITIVE_PRIORITY if is_sprint else _CONVENTIONAL_NON_COMPETITIVE_PRIORITY
    )
    now_utc = datetime.now(UTC)
    for session_name in session_priority:
        if not _session_scheduled_end_passed(event, session_name, now_utc):
            continue

        weather_features = _load_session_weather_features(
            year=year,
            race_name=race_name,
            session_name=session_name,
        )
        if weather_features is None:
            continue

        rainfall_signal = _coerce_float(weather_features.get("rainfall_signal"))
        practice_weather_bucket = _infer_weather_bucket_from_rainfall_signal(rainfall_signal)
        return {
            "available": True,
            "source_session": session_name,
            "reason": "session_weather_available",
            "practice_weather_bucket": practice_weather_bucket,
            **weather_features,
        }

    return {**fallback, "reason": "no_completed_non_competitive_weather"}


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
