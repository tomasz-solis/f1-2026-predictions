"""FastF1 session loader for Phase 2 smoke-session inspection.

This module is the only place Phase 2 imports ``fastf1``. Keeping the
boundary here lets the inspector module stay testable without FastF1
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_session(
    year: int,
    event_name: str,
    session_kind: str,
    *,
    cache_dir: Path,
) -> Any:
    """Load a FastF1 session with laps, results, and weather data.

    Telemetry is intentionally not loaded; the inspector does not need
    it and skipping it makes the load substantially faster.

    Parameters
    ----------
    year:
        Season year, e.g. ``2024``.
    event_name:
        Event name FastF1 accepts, e.g. ``"Bahrain Grand Prix"``.
    session_kind:
        ``"race"`` (loads ``"R"``) or ``"qualifying"`` (loads ``"Q"``).
    cache_dir:
        FastF1 cache directory. Created if missing.

    Returns
    -------
    fastf1.core.Session
        Loaded session ready for inspection.

    Raises
    ------
    ImportError
        If ``fastf1`` is not installed in the environment.
    ValueError
        If ``session_kind`` is not one of the accepted values.
    """
    try:
        import fastf1
    except ImportError as exc:
        raise ImportError(
            "fastf1 is required for load_session(); install via 'pip install fastf1'"
        ) from exc

    kind_map = {"race": "R", "qualifying": "Q"}
    if session_kind not in kind_map:
        raise ValueError(f"session_kind must be one of {sorted(kind_map)}, got {session_kind!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, event_name, kind_map[session_kind])
    session.load(laps=True, weather=True, telemetry=False)
    return session
