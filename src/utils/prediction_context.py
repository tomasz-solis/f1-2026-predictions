"""Shared runtime context for prediction-time config and historical evaluation."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, runtime_checkable

import fastf1

logger = logging.getLogger(__name__)

_MAX_HISTORICAL_REFERENCE = datetime.max.replace(tzinfo=UTC)
_ACTIVE_PREDICTION_CONTEXT: ContextVar[PredictionContext | None] = ContextVar(
    "prediction_context",
    default=None,
)
_ACTIVE_CONFIG: ContextVar[ConfigProvider | None] = ContextVar(
    "prediction_config",
    default=None,
)
_SESSION_NAME_ALIASES = {
    "SQ": "Sprint Qualifying",
    "SPRINT": "Sprint",
    "Q": "Q",
    "R": "R",
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
}


@runtime_checkable
class ConfigProvider(Protocol):
    """Read-only config interface used inside prediction-time helpers."""

    def get(self, key: str, default: Any = None) -> Any:
        """Return config value for one dotted key or the supplied default."""


def normalize_utc_datetime(value: datetime | None) -> datetime | None:
    """Normalize one datetime-like value to timezone-aware UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class PredictionContext:
    """Runtime metadata for one prediction or historical evaluation run."""

    mode: Literal["live", "historical"] = "live"
    as_of_datetime: datetime | None = None
    target_session_datetime: datetime | None = None
    seed: int | None = None

    def normalized(self) -> PredictionContext:
        """Return a UTC-normalized copy of this context."""
        normalized_mode: Literal["live", "historical"] = (
            "historical" if str(self.mode).strip().lower() == "historical" else "live"
        )
        return PredictionContext(
            mode=normalized_mode,
            as_of_datetime=normalize_utc_datetime(self.as_of_datetime),
            target_session_datetime=normalize_utc_datetime(self.target_session_datetime),
            seed=self.seed,
        )

    @property
    def is_historical(self) -> bool:
        """Return whether this context represents historical replay mode."""
        return self.mode == "historical"

    def reference_now(self) -> datetime:
        """Return the effective "current time" for this prediction run."""
        if self.target_session_datetime is not None:
            return self.target_session_datetime
        if self.as_of_datetime is not None:
            return self.as_of_datetime
        if self.is_historical:
            return _MAX_HISTORICAL_REFERENCE
        return datetime.now(UTC)

    def freshness_reference(self, session_datetime: datetime | None = None) -> datetime:
        """Return the timestamp used for stale-session checks."""
        if self.target_session_datetime is not None:
            return self.target_session_datetime
        if self.as_of_datetime is not None:
            return self.as_of_datetime
        if self.is_historical and session_datetime is not None:
            normalized_session = normalize_utc_datetime(session_datetime)
            if normalized_session is not None:
                return normalized_session
        return self.reference_now()


def get_active_prediction_context() -> PredictionContext | None:
    """Return the currently active prediction context, if any."""
    context = _ACTIVE_PREDICTION_CONTEXT.get()
    if context is None:
        return None
    return context.normalized()


def get_active_config(config: ConfigProvider | None = None) -> ConfigProvider:
    """Return the config provider active for the current prediction run."""
    active_config = _ACTIVE_CONFIG.get()
    if active_config is not None:
        return active_config
    if config is not None:
        return config
    from src.utils import config_loader

    return config_loader


def get_config_value(key: str, default: Any = None, *, config: ConfigProvider | None = None) -> Any:
    """Read one config value from the injected provider or the global fallback."""
    return get_active_config(config).get(key, default)


def get_prediction_reference_now() -> datetime:
    """Return the effective current time for the active prediction context."""
    active_context = get_active_prediction_context()
    if active_context is None:
        return datetime.now(UTC)
    return active_context.reference_now()


def get_session_freshness_age(session_datetime: datetime | None) -> timedelta:
    """Return practice-session age relative to the active prediction context."""
    normalized_session = normalize_utc_datetime(session_datetime)
    if normalized_session is None:
        return timedelta(0)

    active_context = get_active_prediction_context()
    if active_context is None:
        reference = datetime.now(UTC)
    else:
        reference = active_context.freshness_reference(normalized_session)

    if reference <= normalized_session:
        return timedelta(0)
    return reference - normalized_session


def _clear_prediction_runtime_caches() -> None:
    """Clear caches that depend on active config or historical timing."""
    try:
        from src.data import track_data_loader

        track_data_loader.resolve_non_competitive_weather_features.cache_clear()
        track_data_loader.resolve_track_temperature_profile.cache_clear()
        track_data_loader.resolve_track_temperature_c.cache_clear()
    except Exception as exc:
        logger.debug("Could not clear prediction-runtime caches: %s", exc)


@contextmanager
def activate_prediction_runtime(
    *,
    config: ConfigProvider | None = None,
    prediction_context: PredictionContext | None = None,
):
    """Activate config and historical-evaluation context for one prediction run."""
    normalized_context = prediction_context.normalized() if prediction_context is not None else None
    context_token = _ACTIVE_PREDICTION_CONTEXT.set(normalized_context)
    config_token = _ACTIVE_CONFIG.set(config)
    _clear_prediction_runtime_caches()
    try:
        yield normalized_context
    finally:
        _ACTIVE_CONFIG.reset(config_token)
        _ACTIVE_PREDICTION_CONTEXT.reset(context_token)


def build_historical_prediction_context(
    *,
    year: int,
    race_name: str,
    target_session_name: str,
    seed: int | None = None,
    as_of_offset: timedelta = timedelta(0),
) -> PredictionContext:
    """Build historical replay context anchored to one scheduled session."""
    session_name = _SESSION_NAME_ALIASES.get(
        str(target_session_name).strip().upper(),
        str(target_session_name).strip(),
    )
    try:
        event = fastf1.get_event(year, race_name)
        raw_session_datetime = event.get_session_date(session_name)
    except Exception as exc:
        logger.warning(
            "Could not resolve %s session date for %s %s; using historical fallback context: %s",
            session_name,
            year,
            race_name,
            exc,
        )
        return PredictionContext(
            mode="historical",
            as_of_datetime=None,
            target_session_datetime=None,
            seed=seed,
        )

    target_session_datetime = normalize_utc_datetime(raw_session_datetime)
    as_of_datetime = None
    if target_session_datetime is not None:
        as_of_datetime = target_session_datetime + as_of_offset

    return PredictionContext(
        mode="historical",
        as_of_datetime=as_of_datetime,
        target_session_datetime=target_session_datetime,
        seed=seed,
    )
