"""Adaptive FP blend weight calculation based on session quality."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def calculate_adaptive_blend_weight(
    session_data: dict[str, Any],
    predicted_race_weather: str,
    track_name: str,
) -> float:
    """
    Calculate adaptive FP blend weight based on session quality indicators.

    Args:
        session_data: Session metadata including weather, laps, etc.
        predicted_race_weather: Expected race weather ('dry', 'rain', 'mixed')
        track_name: Name of the circuit

    Returns:
        Blend weight between 0.5-0.85 (higher = trust FP data more)

    Quality Factors:
    - Weather consistency: Penalize if FP weather differs from race forecast
    - Representative running: Penalize truncated sessions (<30 laps)
    - Track evolution: FP3 (Saturday) more representative than FP1 (Friday)
    - Track limits: Penalize excessive violations (session validity)
    """
    base_weight = 0.70

    # Factor 1: Weather consistency
    fp_weather = session_data.get("weather", "unknown")
    if fp_weather != predicted_race_weather and fp_weather != "unknown":
        weather_penalty = 0.15
        base_weight -= weather_penalty
        logger.debug(
            "Weather mismatch (FP=%s, Race=%s): -%s blend weight",
            fp_weather,
            predicted_race_weather,
            weather_penalty,
        )

    # Factor 2: Representative running
    total_laps = session_data.get("total_laps", 0)
    if total_laps < 30:
        # Red flag or truncated session
        representativeness_penalty = 0.10
        base_weight -= representativeness_penalty
        logger.debug(
            "Truncated session (%s laps): -%s blend weight", total_laps, representativeness_penalty
        )

    # Factor 3: Track evolution
    # FP3 (Saturday) is more representative than FP1 (Friday)
    session_type = session_data.get("session_type", "unknown")
    if session_type == "FP1":
        evolution_penalty = 0.08
        base_weight -= evolution_penalty
        logger.debug("FP1 track evolution: -%s blend weight", evolution_penalty)
    elif session_type == "FP3":
        evolution_bonus = 0.05
        base_weight += evolution_bonus
        logger.debug("FP3 track evolution: +%s blend weight", evolution_bonus)

    # Factor 4: Track limits violations
    # High violation count suggests drivers struggling with limits = less representative
    track_limits_count = session_data.get("track_limits_violations", 0)
    if track_limits_count > 50:  # Unusually high
        validity_penalty = 0.05
        base_weight -= validity_penalty
        logger.debug(
            "High track limits violations (%s): -%s blend weight",
            track_limits_count,
            validity_penalty,
        )

    # Factor 5: Circuit-specific adjustments
    # Street circuits have less representative FP (more track evolution)
    street_circuits = [
        "Monaco Grand Prix",
        "Singapore Grand Prix",
        "Las Vegas Grand Prix",
        "Azerbaijan Grand Prix",
    ]
    if track_name in street_circuits:
        street_penalty = 0.05
        base_weight -= street_penalty
        logger.debug("Street circuit: -%s blend weight", street_penalty)

    final_weight = max(0.50, min(0.85, base_weight))
    logger.info(
        "Adaptive FP blend weight for %s: %s (base=%s, adjusted=%s)",
        track_name,
        format(final_weight, ".2f"),
        format(0.70, ".2f"),
        format(base_weight, ".2f"),
    )

    return final_weight


def get_session_quality_metadata(year: int, race_name: str, session_name: str) -> dict[str, Any]:
    """
    Extract quality metadata from a practice session.

    Args:
        year: Season year
        race_name: Race name
        session_name: Session name (FP1, FP2, FP3)

    Returns:
        Dictionary with session quality indicators
    """
    try:
        import fastf1

        session = fastf1.get_session(year, race_name, session_name)
        session.load(laps=True, telemetry=False, weather=True, messages=False)

        # Extract metadata
        laps = getattr(session, "laps", None)
        total_laps = len(laps) if laps is not None and not laps.empty else 0

        # Determine weather (simplified: check if wet tires used)
        weather = "dry"
        if laps is not None and not laps.empty:
            compounds = laps["Compound"].dropna().unique()
            if "INTERMEDIATE" in compounds or "WET" in compounds:
                weather = "rain"

        # Track limits violations (if available in session data)
        track_limits = 0
        try:
            # This may not be available in all FastF1 versions
            messages = getattr(session, "messages", None)
            if messages is not None:
                track_limits = len(
                    messages[messages["Message"].str.contains("TRACK LIMITS", case=False, na=False)]
                )
        except Exception as exc:
            logger.debug("Track limits metadata extraction failed: %s", exc)

        return {
            "session_type": session_name,
            "total_laps": total_laps,
            "weather": weather,
            "track_limits_violations": track_limits,
        }

    except Exception as e:
        logger.warning("Could not extract session quality metadata: %s", e)
        return {
            "session_type": session_name,
            "total_laps": 0,
            "weather": "unknown",
            "track_limits_violations": 0,
        }
