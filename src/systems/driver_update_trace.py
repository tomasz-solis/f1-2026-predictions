"""Trace dry driver-state and wet-skill changes at updater boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

DRIVER_UPDATE_TRACE_SCHEMA_VERSION = 1

_TRACE_STATE_FIELDS = (
    "legacy_rating_mu",
    "legacy_rating_sigma",
    "race_rating_mu_s",
    "race_rating_sigma_s",
    "quali_rating_mu_s",
    "quali_rating_sigma_s",
    "wet_skill",
)


def snapshot_driver_update_state(
    drivers_payload: Mapping[str, Any],
    *,
    legacy_ratings: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Read traceable driver fields from artifacts and optional live ratings."""
    state: dict[str, dict[str, float | None]] = {}
    ratings = legacy_ratings or {}
    for driver_code, driver_payload in drivers_payload.items():
        if not isinstance(driver_payload, Mapping):
            continue

        bayesian = driver_payload.get("bayesian", {})
        bayesian_payload = bayesian if isinstance(bayesian, Mapping) else {}
        rating_state = ratings.get(str(driver_code))
        state[str(driver_code)] = {
            "legacy_rating_mu": (
                _coerce_float(rating_state[0])
                if rating_state is not None
                else _coerce_float(bayesian_payload.get("rating_mu"))
            ),
            "legacy_rating_sigma": (
                _coerce_float(rating_state[1])
                if rating_state is not None
                else _coerce_float(bayesian_payload.get("rating_sigma"))
            ),
            "race_rating_mu_s": _coerce_float(bayesian_payload.get("race_rating_mu_s")),
            "race_rating_sigma_s": _coerce_float(bayesian_payload.get("race_rating_sigma_s")),
            "quali_rating_mu_s": _coerce_float(bayesian_payload.get("quali_rating_mu_s")),
            "quali_rating_sigma_s": _coerce_float(bayesian_payload.get("quali_rating_sigma_s")),
            "wet_skill": _coerce_float(driver_payload.get("wet_skill")),
        }
    return state


def legacy_ratings_from_trace_state(
    state: Mapping[str, Mapping[str, float | None]],
) -> dict[str, tuple[float, float]]:
    """Return saved legacy `(mu, sigma)` pairs from one trace state snapshot."""
    ratings: dict[str, tuple[float, float]] = {}
    for driver_code, driver_state in state.items():
        mu = driver_state.get("legacy_rating_mu")
        sigma = driver_state.get("legacy_rating_sigma")
        if mu is None or sigma is None:
            continue
        ratings[str(driver_code)] = (float(mu), float(sigma))
    return ratings


def build_driver_update_trace_rows(
    *,
    year: int,
    event_name: str,
    session_name: str,
    session_kind: str,
    weather_route: str,
    driver_codes: Iterable[str],
    before: Mapping[str, Mapping[str, float | None]],
    after: Mapping[str, Mapping[str, float | None]],
    dry_race_update_applied: bool,
    dry_quali_update_applied: bool,
    wet_update_drivers: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic per-driver updater trace rows for one session route."""
    wet_drivers = wet_update_drivers or frozenset()
    rows: list[dict[str, Any]] = []
    for driver_code in sorted({str(code) for code in driver_codes if str(code)}):
        row: dict[str, Any] = {
            "schema_version": DRIVER_UPDATE_TRACE_SCHEMA_VERSION,
            "year": int(year),
            "event_name": str(event_name),
            "session_name": str(session_name),
            "session_kind": str(session_kind),
            "weather_route": str(weather_route),
            "driver_code": driver_code,
            "dry_race_update_applied": bool(dry_race_update_applied),
            "dry_quali_update_applied": bool(dry_quali_update_applied),
            "wet_update_applied": driver_code in wet_drivers,
        }
        before_state = before.get(driver_code, {})
        after_state = after.get(driver_code, {})
        for field in _TRACE_STATE_FIELDS:
            before_value = _trace_value(before_state, field)
            after_value = _trace_value(after_state, field)
            row[f"{field}_before"] = before_value
            row[f"{field}_after"] = after_value
            if field.endswith("_mu") or field.endswith("_mu_s") or field == "wet_skill":
                row[f"{field}_delta"] = _delta(before_value, after_value)
        rows.append(row)
    return rows


def _trace_value(state: Mapping[str, float | None], field: str) -> float | None:
    """Return one normalized trace value from a driver state snapshot."""
    return _coerce_float(state.get(field))


def _delta(before: float | None, after: float | None) -> float | None:
    """Return a state delta when both trace endpoints are numeric."""
    if before is None or after is None:
        return None
    return float(after - before)


def _coerce_float(value: Any) -> float | None:
    """Convert finite numeric values for JSON-safe update traces."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
