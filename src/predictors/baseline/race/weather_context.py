"""Weather and temperature context helpers for race prediction flow."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _safe_call_resolver(
    resolver: Any,
    kwargs: dict[str, Any],
    resolver_name: str = "resolver",
) -> Any | None:
    """Call resolver with kwargs, falling back to positional args on TypeError.

    Logs failures at debug level so internal resolver bugs are traceable
    rather than silently swallowed.
    """
    if not callable(resolver):
        logger.debug(
            "%s is not callable (type=%s), returning None",
            resolver_name,
            type(resolver).__name__,
        )
        return None
    try:
        return resolver(**kwargs)
    except TypeError:
        try:
            return resolver(*kwargs.values())
        except (TypeError, ValueError) as exc:
            logger.debug(
                "%s positional fallback failed: %s: %s",
                resolver_name,
                type(exc).__name__,
                exc,
            )
            return None
    except (ValueError, KeyError, AttributeError) as exc:
        logger.debug(
            "%s failed: %s: %s",
            resolver_name,
            type(exc).__name__,
            exc,
        )
        return None


def coerce_optional_float(value: Any) -> float | None:
    """Convert value to float when possible; otherwise return None."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def weather_bucket_mismatch_score(selected_weather: str, practice_weather: str) -> float:
    """Return mismatch score between selected and practice-derived weather buckets."""
    selected_key = str(selected_weather).strip().lower()
    practice_key = str(practice_weather).strip().lower()

    if not practice_key or practice_key == "unknown":
        return 0.25
    if selected_key == practice_key:
        return 0.0
    if selected_key == "mixed" or practice_key == "mixed":
        return 0.5
    return 1.0


def build_weather_feature_context(
    *,
    selected_weather: str,
    raw_features: dict[str, Any],
    cfg: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Translate non-competitive weather features into race simulation modifiers."""
    source_session = raw_features.get("source_session")
    practice_weather_bucket = str(raw_features.get("practice_weather_bucket", "unknown"))
    wind_speed_kph = coerce_optional_float(raw_features.get("wind_speed_kph"))
    track_temperature_c = coerce_optional_float(raw_features.get("track_temperature_c"))
    air_temperature_c = coerce_optional_float(raw_features.get("air_temperature_c"))
    humidity_pct = coerce_optional_float(raw_features.get("humidity_pct"))
    rainfall_signal = coerce_optional_float(raw_features.get("rainfall_signal"))

    mismatch_score = weather_bucket_mismatch_score(selected_weather, practice_weather_bucket)
    mismatch_chaos_boost = float(
        cfg.get("baseline_predictor.race.weather_features.mismatch.chaos_boost", 0.18)
    )
    mismatch_variance_boost = float(
        cfg.get("baseline_predictor.race.weather_features.mismatch.variance_boost", 0.10)
    )
    mismatch_confidence_penalty = float(
        cfg.get("baseline_predictor.race.weather_features.mismatch.confidence_penalty", 2.0)
    )

    chaos_multiplier = float(np.clip(1.0 + (mismatch_score * mismatch_chaos_boost), 0.80, 1.40))
    teammate_variance_multiplier = float(
        np.clip(1.0 + (mismatch_score * mismatch_variance_boost), 0.80, 1.35)
    )
    confidence_adjustment = float(max(0.0, mismatch_score * mismatch_confidence_penalty))

    modifiers = {
        "chaos_multiplier": chaos_multiplier,
        "teammate_variance_multiplier": teammate_variance_multiplier,
        "confidence_adjustment": confidence_adjustment,
    }
    context = {
        "available": bool(raw_features.get("available", False)),
        "source_session": source_session,
        "selected_weather": str(selected_weather).strip().lower(),
        "practice_weather_bucket": practice_weather_bucket,
        "track_temperature_c": track_temperature_c,
        "air_temperature_c": air_temperature_c,
        "wind_speed_kph": wind_speed_kph,
        "humidity_pct": humidity_pct,
        "rainfall_signal": rainfall_signal,
        "weather_mismatch_score": mismatch_score,
        "chaos_multiplier": chaos_multiplier,
        "teammate_variance_multiplier": teammate_variance_multiplier,
        "confidence_adjustment": confidence_adjustment,
    }
    return modifiers, context


def resolve_race_environment_context(
    *,
    race_params: dict[str, Any],
    weather: str,
    year: int,
    race_name: str | None,
    is_sprint: bool,
    cfg: Any,
    resolve_track_temperature_c: Any | None,
    resolve_track_temperature_profile: Any | None,
    resolve_non_competitive_weather_features: Any | None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    """Resolve track-temperature and practice-weather context for one race run."""
    track_temperature_context: dict[str, Any] = {}
    if "track_temperature_c" not in race_params:
        resolved_temperature_profile: dict[str, Any] | None = None
        if callable(resolve_track_temperature_profile):
            candidate_profile = _safe_call_resolver(
                resolve_track_temperature_profile,
                {"year": year, "race_name": race_name, "weather": weather, "is_sprint": is_sprint},
                resolver_name="resolve_track_temperature_profile",
            )

            if isinstance(candidate_profile, dict):
                resolved_temperature_profile = dict(candidate_profile)

        resolved_track_temp: float | None = None
        if resolved_temperature_profile is not None:
            try:
                resolved_track_temp = float(resolved_temperature_profile["track_temperature_c"])
            except (KeyError, TypeError, ValueError):
                resolved_track_temp = None

        if resolved_track_temp is None and callable(resolve_track_temperature_c):
            resolved_track_temp_candidate = _safe_call_resolver(
                resolve_track_temperature_c,
                {"year": year, "race_name": race_name, "weather": weather, "is_sprint": is_sprint},
                resolver_name="resolve_track_temperature_c",
            )
            if resolved_track_temp_candidate is not None:
                try:
                    resolved_track_temp = float(resolved_track_temp_candidate)
                except (TypeError, ValueError):
                    resolved_track_temp = None

        if resolved_track_temp is not None:
            race_params["track_temperature_c"] = resolved_track_temp
            if resolved_temperature_profile is not None:
                track_temperature_context = {
                    "track_temperature_c": float(resolved_track_temp),
                    "source": str(
                        resolved_temperature_profile.get("source", "session_or_fallback")
                    ),
                    "reason": str(resolved_temperature_profile.get("reason", "")),
                    "weather_bucket": str(
                        resolved_temperature_profile.get("weather_bucket", weather)
                    ),
                    "session_name": resolved_temperature_profile.get("session_name"),
                    "session_track_temperature_c": resolved_temperature_profile.get(
                        "session_track_temperature_c"
                    ),
                    "session_temperature_source": resolved_temperature_profile.get(
                        "session_temperature_source"
                    ),
                    "session_air_temperature_c": resolved_temperature_profile.get(
                        "session_air_temperature_c"
                    ),
                    "forecast_track_temperature_c": resolved_temperature_profile.get(
                        "forecast_track_temperature_c"
                    ),
                    "session_weight": resolved_temperature_profile.get("session_weight"),
                    "forecast_weight": resolved_temperature_profile.get("forecast_weight"),
                    "blend_enabled": bool(resolved_temperature_profile.get("blend_enabled", False)),
                }
            else:
                track_temperature_context = {
                    "track_temperature_c": float(resolved_track_temp),
                    "source": "legacy_scalar_resolver",
                    "reason": "legacy_temperature_resolver",
                    "weather_bucket": str(weather).strip().lower(),
                    "session_name": None,
                    "session_track_temperature_c": None,
                    "session_temperature_source": None,
                    "session_air_temperature_c": None,
                    "forecast_track_temperature_c": None,
                    "session_weight": None,
                    "forecast_weight": None,
                    "blend_enabled": False,
                }
        else:
            default_track_temp = {
                "dry": cfg.get("baseline_predictor.race.track_temperature.dry_c", 36.0),
                "mixed": cfg.get("baseline_predictor.race.track_temperature.mixed_c", 29.0),
                "rain": cfg.get("baseline_predictor.race.track_temperature.rain_c", 23.0),
            }
            fallback_track_temp = float(
                default_track_temp.get(str(weather).strip().lower(), default_track_temp["dry"])
            )
            race_params["track_temperature_c"] = fallback_track_temp
            track_temperature_context = {
                "track_temperature_c": fallback_track_temp,
                "source": "forecast_fallback",
                "reason": "no_temperature_signal",
                "weather_bucket": str(weather).strip().lower(),
                "session_name": None,
                "session_track_temperature_c": None,
                "session_temperature_source": None,
                "session_air_temperature_c": None,
                "forecast_track_temperature_c": fallback_track_temp,
                "session_weight": 0.0,
                "forecast_weight": 1.0,
                "blend_enabled": False,
            }
    else:
        existing_temp = float(race_params["track_temperature_c"])
        track_temperature_context = {
            "track_temperature_c": existing_temp,
            "source": "track_params_override",
            "reason": "track_params_override",
            "weather_bucket": str(weather).strip().lower(),
            "session_name": None,
            "session_track_temperature_c": None,
            "session_temperature_source": None,
            "session_air_temperature_c": None,
            "forecast_track_temperature_c": None,
            "session_weight": None,
            "forecast_weight": None,
            "blend_enabled": False,
        }

    weather_feature_modifiers: dict[str, float] = {
        "chaos_multiplier": 1.0,
        "teammate_variance_multiplier": 1.0,
        "confidence_adjustment": 0.0,
    }
    weather_feature_context: dict[str, Any] = {
        "available": False,
        "source_session": None,
        "selected_weather": str(weather).strip().lower(),
        "practice_weather_bucket": "unknown",
        "track_temperature_c": None,
        "air_temperature_c": None,
        "wind_speed_kph": None,
        "humidity_pct": None,
        "rainfall_signal": None,
        "weather_mismatch_score": 0.0,
        "chaos_multiplier": 1.0,
        "teammate_variance_multiplier": 1.0,
        "confidence_adjustment": 0.0,
    }
    if callable(resolve_non_competitive_weather_features):
        raw_weather_features: dict[str, Any] | None = None
        candidate_features = _safe_call_resolver(
            resolve_non_competitive_weather_features,
            {"year": year, "race_name": race_name, "is_sprint": is_sprint},
            resolver_name="resolve_non_competitive_weather_features",
        )

        if isinstance(candidate_features, dict):
            raw_weather_features = dict(candidate_features)

        if raw_weather_features and raw_weather_features.get("available"):
            weather_feature_modifiers, weather_feature_context = build_weather_feature_context(
                selected_weather=weather,
                raw_features=raw_weather_features,
                cfg=cfg,
            )
        elif raw_weather_features:
            weather_feature_context = {
                **weather_feature_context,
                "available": False,
                "source_session": raw_weather_features.get("source_session"),
                "practice_weather_bucket": str(
                    raw_weather_features.get("practice_weather_bucket", "unknown")
                ),
            }

    race_params["weather_feature_modifiers"] = weather_feature_modifiers
    return track_temperature_context, weather_feature_modifiers, weather_feature_context
