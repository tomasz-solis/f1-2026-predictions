"""Session-loading helpers for testing updater flows."""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
from pandas.errors import SettingWithCopyWarning

logger = logging.getLogger(__name__)


def normalize_name(value: str) -> str:
    """Normalize names for fuzzy matching."""
    return "".join(char for char in value.lower() if char.isalnum())


def is_testing_event(event_name: str) -> bool:
    """Best-effort detection of testing events from user-provided name."""
    normalized = normalize_name(event_name)
    return "test" in normalized


def extract_testing_day(session_name: str) -> int | None:
    """Map session label to a testing day number (1..3) if possible."""
    normalized = normalize_name(session_name)
    for day in (1, 2, 3):
        if str(day) in normalized:
            return day
    return None


def extract_testing_number(event_name: str) -> int | None:
    """Parse explicit test number from event name (e.g., 'Testing 2')."""
    normalized = normalize_name(event_name)
    for number in (1, 2, 3):
        if f"test{number}" in normalized or f"testing{number}" in normalized:
            return number
    return None


def resolve_testing_backends(
    preferred_backend: str | None = "auto",
    default_backends: tuple[str | None, ...] = ("f1timing", "fastf1", None),
) -> tuple[str | None, ...]:
    """Resolve backend preference into an ordered list of backends to try."""
    if preferred_backend in (None, "auto"):
        return default_backends
    if preferred_backend in ("fastf1", "f1timing"):
        return (preferred_backend,)

    raise ValueError("Invalid testing backend. Use one of: auto, fastf1, f1timing.")


def resolve_testing_cache_dir(
    cache_dir: str | None = None,
    default_cache_dir: Path = Path("data/raw/.fastf1_cache_testing"),
    cache_root: Path = Path("data/raw"),
) -> Path:
    """
    Resolve testing cache location.

    Relative paths are kept under data/raw to avoid repository root clutter.
    """
    if not cache_dir:
        return default_cache_dir

    candidate = Path(cache_dir).expanduser()
    if candidate.is_absolute():
        return candidate

    cleaned_parts = tuple(part for part in candidate.parts if part not in ("", "."))
    if not cleaned_parts:
        return default_cache_dir

    relative_candidate = Path(*cleaned_parts)
    if relative_candidate.parts[:2] == ("data", "raw"):
        return relative_candidate

    return cache_root / relative_candidate


def coerce_utc_datetime(value: Any) -> datetime | None:
    """Convert FastF1 event datetime values to UTC-aware datetime."""
    if value is None or pd.isna(value):
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.to_pydatetime()


def testing_session_has_started(
    event: fastf1.events.Event,
    day_number: int,
    now_utc: datetime | None = None,
    coerce_utc_datetime_fn=coerce_utc_datetime,
) -> bool:
    """Check whether a testing day has started based on UTC session timestamp."""
    session_dt_utc = coerce_utc_datetime_fn(event.get(f"Session{day_number}DateUtc"))
    if session_dt_utc is None:
        return True

    now = now_utc or datetime.now(UTC)
    # Keep a small tolerance for clock skew between systems.
    return session_dt_utc <= (now + timedelta(minutes=15))


def get_testing_event_with_backends(
    year: int,
    test_number: int,
    testing_backends: tuple[str | None, ...],
    error_messages: list[str] | None = None,
    fastf1_get_testing_event=fastf1.get_testing_event,
    logger_obj: Any = logger,
) -> fastf1.events.Event | None:
    """Load a testing event, trying explicit backends before auto mode."""
    for backend in testing_backends:
        kwargs = {"backend": backend} if backend is not None else {}
        backend_label = backend or "auto"
        try:
            return fastf1_get_testing_event(year, test_number, **kwargs)
        except Exception as exc:
            logger_obj.debug(
                "Unable to load testing event %s/%s via backend %s: %s",
                year,
                test_number,
                backend_label,
                exc,
            )
            if error_messages is not None:
                error_messages.append(
                    f"testing_event#{test_number} backend={backend_label} -> "
                    f"{type(exc).__name__}: {exc}"
                )

    return None


def normalize_testing_event_sessions(event: fastf1.events.Event) -> None:
    """
    Normalize testing session labels to FastF1-compatible names.

    Some schedules expose "Day 1/2/3". FastF1 Session initialization expects
    canonical names like "Practice 1/2/3".
    """
    for day_number in (1, 2, 3):
        key = f"Session{day_number}"
        value = event.get(key)
        if not isinstance(value, str):
            continue
        if normalize_name(value) == f"day{day_number}":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SettingWithCopyWarning)
                event[key] = f"Practice {day_number}"


def load_testing_session_with_backends(
    year: int,
    test_number: int,
    day_number: int,
    testing_backends: tuple[str | None, ...],
    error_messages: list[str] | None = None,
    fastf1_get_testing_event=fastf1.get_testing_event,
    normalize_testing_event_sessions_fn=normalize_testing_event_sessions,
    logger_obj: Any = logger,
) -> fastf1.core.Session | None:
    """
    Load a testing session and verify laps are actually accessible.

    This avoids reporting sessions as discovered when `session.laps` would still
    raise DataNotLoadedError after `load()`.
    """
    for backend in testing_backends:
        kwargs = {"backend": backend} if backend is not None else {}
        backend_label = backend or "auto"
        try:
            event = fastf1_get_testing_event(year, test_number, **kwargs)
            normalize_testing_event_sessions_fn(event)
            session = event.get_session(day_number)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps
            if laps is None:
                raise ValueError("laps are None after session.load()")
            # Access row count to force DataNotLoadedError if load is incomplete.
            _ = len(laps)
            return session
        except Exception as exc:
            logger_obj.debug(
                "Unable to load testing session %s/%s day %s via backend %s: %s",
                year,
                test_number,
                day_number,
                backend_label,
                exc,
            )
            if error_messages is not None:
                error_messages.append(
                    f"testing#{test_number}/day{day_number} backend={backend_label} -> "
                    f"{type(exc).__name__}: {exc}"
                )

    return None


def load_sessions_for_event(
    year: int,
    event_name: str,
    session_candidates: list[str],
    testing_backends: tuple[str | None, ...] = ("f1timing", "fastf1", None),
    error_messages: list[str] | None = None,
    is_testing_event_fn=is_testing_event,
    extract_testing_number_fn=extract_testing_number,
    extract_testing_day_fn=extract_testing_day,
    get_testing_event_with_backends_fn=get_testing_event_with_backends,
    testing_session_has_started_fn=testing_session_has_started,
    load_testing_session_with_backends_fn=load_testing_session_with_backends,
    fastf1_get_session=fastf1.get_session,
    logger_obj: Any = logger,
) -> list[tuple[str, fastf1.core.Session]]:
    """
    Load available sessions for an event.

    Strategy:
    1) For non-testing events: use regular `get_session(event_name, session_name)`.
    2) For testing events: use `get_testing_event` + `get_testing_session`.
    """
    loaded: list[tuple[str, fastf1.core.Session]] = []

    if not is_testing_event_fn(event_name):
        for session_name in session_candidates:
            try:
                session = fastf1_get_session(year, event_name, session_name)
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = session.laps
                if laps is None:
                    raise ValueError("laps are None after session.load()")
                # Access row count to force DataNotLoadedError if load is incomplete.
                _ = len(laps)
                loaded.append((session_name, session))
            except Exception as exc:
                logger_obj.debug(
                    f"Skipping unavailable session {year} {event_name} {session_name}: {exc}"
                )
                if error_messages is not None:
                    error_messages.append(
                        f"{event_name}::{session_name} -> {type(exc).__name__}: {exc}"
                    )
        return loaded

    explicit_test_number = extract_testing_number_fn(event_name)
    test_numbers = [explicit_test_number] if explicit_test_number else [1, 2, 3]

    day_candidates = []
    for session_name in session_candidates:
        maybe_day = extract_testing_day_fn(session_name)
        if maybe_day is not None and maybe_day not in day_candidates:
            day_candidates.append(maybe_day)
    if not day_candidates:
        day_candidates = [1, 2, 3]

    now_utc = datetime.now(UTC)

    for test_number in test_numbers:
        event = get_testing_event_with_backends_fn(
            year=year,
            test_number=test_number,
            testing_backends=testing_backends,
            error_messages=error_messages,
        )
        if event is None:
            continue

        for day_number in day_candidates:
            if not testing_session_has_started_fn(event, day_number, now_utc=now_utc):
                if error_messages is not None:
                    error_messages.append(
                        f"testing#{test_number}/day{day_number} -> session has not started yet"
                    )
                continue

            session = load_testing_session_with_backends_fn(
                year=year,
                test_number=test_number,
                day_number=day_number,
                testing_backends=testing_backends,
                error_messages=error_messages,
            )
            if session is None:
                continue

            label = f"Testing {test_number} Day {day_number}"
            loaded.append((label, session))

    return loaded
