"""Helpers for storing and sorting session-level car characteristic snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

SNAPSHOT_ARTIFACT_TYPE = "car_characteristics_snapshot"

SESSION_ORDER: dict[str, int] = {
    "DAY1": 1,
    "DAY2": 2,
    "DAY3": 3,
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "SQ": 4,
    "SPRINT": 5,
    "S": 5,
    "Q": 6,
    "R": 7,
}


def snapshot_artifact_key(year: int, event_name: str, session_name: str) -> str:
    """Return the canonical artifact key for one session snapshot."""
    return f"{int(year)}::{str(event_name).strip()}::{str(session_name).strip()}"


def session_order_index(session_name: str) -> int:
    """Return a stable intra-weekend order for a session label."""
    normalized = "".join(ch for ch in str(session_name or "").strip().upper() if ch.isalnum())
    if normalized in SESSION_ORDER:
        return SESSION_ORDER[normalized]
    for token, order in SESSION_ORDER.items():
        if token in normalized:
            return order
    return 99


def merge_snapshot_team_metrics(
    performance_by_team: dict[str, dict[str, float]],
    tire_by_team: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Merge per-team performance and tire metrics into one snapshot payload."""
    merged: dict[str, dict[str, float]] = {}

    for source in (performance_by_team, tire_by_team):
        for team_name, metrics in source.items():
            if not isinstance(metrics, dict):
                continue
            team_payload = merged.setdefault(str(team_name), {})
            for metric_name, value in metrics.items():
                if not isinstance(value, int | float):
                    continue
                value_float = float(value)
                if value_float != value_float:
                    continue
                team_payload[str(metric_name)] = round(value_float, 4)

    return {team_name: metrics for team_name, metrics in merged.items() if metrics}


def resolve_session_snapshot_metadata(session: Any, session_name: str) -> dict[str, Any]:
    """Extract sorting metadata from a FastF1 session object when available."""
    event = getattr(session, "event", None)

    round_number = _coerce_optional_int(getattr(session, "round_number", None))
    if round_number is None:
        round_number = _coerce_optional_int(_event_round_number_candidate(event))

    session_started_at = _coerce_iso_datetime(getattr(session, "date", None))
    if session_started_at is None:
        session_started_at = _coerce_iso_datetime(
            _event_session_date_candidate(event, session_name)
        )

    return {
        "round_number": round_number,
        "session_order": session_order_index(session_name),
        "session_started_at": session_started_at,
    }


def build_car_characteristics_snapshot_payload(
    *,
    year: int,
    event_name: str,
    session_name: str,
    team_profiles: dict[str, dict[str, dict[str, float]]],
    team_driver_deltas_seconds: dict[str, dict[str, dict[str, float]]] | None = None,
    source: str,
    captured_at: str | None = None,
    round_number: int | None = None,
    session_started_at: str | None = None,
    season_characteristics_version: int | None = None,
) -> dict[str, Any]:
    """Build a persisted payload for one session-level car profile snapshot.

    Snapshot artifacts now mirror the season artifact naming where practical.
    ``testing_characteristics_profiles`` and ``testing_characteristics`` match the
    live team schema, while ``profiles`` stays as a compatibility alias for
    older readers and already-saved snapshot consumers.
    """
    normalized_team_driver_deltas = team_driver_deltas_seconds or {}
    payload: dict[str, Any] = {
        "year": int(year),
        "event_name": str(event_name).strip(),
        "session_name": str(session_name).strip(),
        "session_order": session_order_index(session_name),
        "source": str(source).strip(),
        "captured_at": str(captured_at or datetime.now(UTC).isoformat()),
        "teams": {},
        "version": 1,
    }

    team_names = {
        str(team_name)
        for team_name in team_profiles.keys()
        if isinstance(team_name, str) and str(team_name).strip()
    }
    for team_name in sorted(team_names):
        team_entry: dict[str, Any] = {}

        profiles = team_profiles.get(team_name)
        if isinstance(profiles, dict) and profiles:
            balanced_profile = profiles.get("balanced")
            team_entry["profiles"] = profiles
            team_entry["testing_characteristics_profiles"] = profiles
            if isinstance(balanced_profile, dict) and balanced_profile:
                team_entry["testing_characteristics"] = balanced_profile

        driver_deltas_seconds = normalized_team_driver_deltas.get(team_name)
        if isinstance(driver_deltas_seconds, dict) and driver_deltas_seconds:
            team_entry["driver_deltas_seconds"] = driver_deltas_seconds

        if team_entry:
            payload["teams"][team_name] = team_entry

    if round_number is not None:
        payload["round_number"] = int(round_number)
    if session_started_at:
        payload["session_started_at"] = str(session_started_at)
    if season_characteristics_version is not None:
        payload["season_characteristics_version"] = int(season_characteristics_version)

    return payload


def sort_snapshot_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort snapshots in season order, then by intra-weekend session order."""
    return sorted(payloads, key=_snapshot_sort_key)


def snapshot_sort_timestamp(payload: dict[str, Any]) -> datetime:
    """Return the best available timestamp for chronological snapshot ordering."""
    for key in ("session_started_at", "captured_at"):
        parsed = _coerce_sort_datetime(payload.get(key))
        if parsed is not None:
            return parsed
    return datetime.max.replace(tzinfo=UTC)


def _coerce_iso_datetime(value: Any) -> str | None:
    """Convert datetime-like values into UTC ISO strings when possible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_value = value
    else:
        try:
            dt_value = datetime.fromisoformat(str(value))
        except ValueError:
            return None

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=UTC)
    else:
        dt_value = dt_value.astimezone(UTC)
    return dt_value.isoformat()


def _coerce_sort_datetime(value: Any) -> datetime | None:
    """Parse a datetime-like value for ordering, preserving full timestamp precision."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_value = value
    else:
        try:
            dt_value = datetime.fromisoformat(str(value))
        except ValueError:
            return None

    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(UTC)


def _snapshot_sort_key(payload: dict[str, Any]) -> tuple[datetime, int, int, str]:
    """Build a stable sort key for one stored snapshot payload."""
    round_number = payload.get("round_number")
    try:
        round_key = int(round_number) if round_number is not None else 0
    except (TypeError, ValueError):
        round_key = 0

    session_order = payload.get("session_order")
    try:
        session_key = int(session_order) if session_order is not None else 99
    except (TypeError, ValueError):
        session_key = 99

    timestamp = snapshot_sort_timestamp(payload)
    event_name = str(payload.get("event_name") or "")
    return (timestamp, round_key, session_key, event_name)


def _event_round_number_candidate(event: object) -> object | None:
    """Extract a round number candidate from a FastF1 event-like object."""
    if event is None:
        return None

    get_value = getattr(event, "get", None)
    if callable(get_value):
        try:
            return get_value("RoundNumber")
        except Exception:
            return None

    return getattr(event, "RoundNumber", None)


def _event_session_date_candidate(event: object, session_name: str) -> object | None:
    """Extract a session date candidate from a FastF1 event-like object."""
    if event is None:
        return None

    get_session_date = getattr(event, "get_session_date", None)
    if not callable(get_session_date):
        return None

    try:
        return get_session_date(session_name)
    except Exception:
        return None


def _coerce_optional_int(value: object | None) -> int | None:
    """Convert an int-like value into an integer when possible."""
    if value is None:
        return None

    if not isinstance(value, int | float | str | bytes | bytearray):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
