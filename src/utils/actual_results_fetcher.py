"""Fetches actual results from competitive F1 sessions."""

import logging
from typing import Any, Literal

import fastf1

from src.types.prediction_types import QualifyingGridEntry
from src.utils.fastf1_resilience import call_with_resilience
from src.utils.grid_validation import validate_qualifying_grid
from src.utils.operational_observability import record_counter
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)
SessionCompletionState = Literal["completed", "incomplete", "unknown"]
_MIN_COMPETITIVE_ENTRIES_BY_SESSION = {
    "SQ": 18,
    "Sprint": 18,
    "Q": 18,
    "R": 18,
}


def _coerce_position(raw_position: Any) -> int:
    """Coerce FastF1 position payload to integer and fail closed for malformed values."""
    if raw_position is None:
        raise ValueError("missing position")
    if str(raw_position).strip().lower() in {"", "nan", "none"}:
        raise ValueError("missing position")
    try:
        return int(raw_position)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid result position value: {raw_position!r}") from exc


def fetch_actual_session_results(
    year: int, race_name: str, session_name: str
) -> list[QualifyingGridEntry] | None:
    """Fetch actual results from competitive session (SQ, Sprint, Q, R)."""
    try:
        labels = {"year": year, "race_name": race_name, "session_name": session_name}
        session = call_with_resilience(
            "fastf1_get_session",
            lambda: fastf1.get_session(year, race_name, session_name),
            labels=labels,
        )
        call_with_resilience(
            "fastf1_session_load_results",
            lambda: session.load(),
            labels=labels,
        )

        # Get results
        results = session.results

        if results is None or len(results) == 0:
            logger.warning(f"No results available for {race_name} {session_name}")
            record_counter("fastf1_results_empty_total", labels=labels)
            return None

        # Extract relevant data and fail closed on malformed rows.
        grid: list[QualifyingGridEntry] = []
        for row_index, (_, row) in enumerate(results.iterrows(), start=1):
            try:
                driver_raw = row.get("Abbreviation", row.get("DriverNumber", ""))
                driver = str(driver_raw).strip()
                if not driver:
                    raise ValueError("missing driver identifier")

                team_raw = row.get("TeamName")
                team_raw_str = str(team_raw).strip()
                if not team_raw_str:
                    raise ValueError("missing team name")
                team = map_team_to_characteristics(team_raw_str) or team_raw_str

                position_raw = row.get("Position")
                position = _coerce_position(position_raw)

                grid.append(
                    {
                        "position": position,
                        "driver": driver,
                        "team": str(team).strip(),
                    }
                )
            except Exception as e:
                logger.error(
                    "Malformed FastF1 results for %s %s at row %s: %s",
                    race_name,
                    session_name,
                    row_index,
                    e,
                )
                record_counter(
                    "fastf1_results_malformed_total",
                    labels={**labels, "row_index": row_index},
                )
                return None

        # Sort by position
        grid.sort(key=lambda item: int(item["position"]))

        min_entries = _MIN_COMPETITIVE_ENTRIES_BY_SESSION.get(str(session_name), 10)
        try:
            validated_grid = validate_qualifying_grid(
                grid,
                min_entries=min_entries,
                require_sequential_positions=True,
            )
        except ValueError as exc:
            logger.error(
                "Invalid FastF1 competitive results for %s %s: %s",
                race_name,
                session_name,
                exc,
            )
            record_counter("fastf1_results_invalid_total", labels=labels)
            return None

        logger.info(
            "Fetched %s results from %s %s",
            len(validated_grid),
            race_name,
            session_name,
        )
        return validated_grid

    except Exception as e:
        logger.error(f"Failed to fetch {session_name} results for {race_name}: {e}")
        record_counter(
            "fastf1_results_fetch_failure_total",
            labels={"year": year, "race_name": race_name, "session_name": session_name},
        )
        return None


def get_competitive_session_completion_state(
    year: int,
    race_name: str,
    session_name: str,
) -> SessionCompletionState:
    """
    Get competitive-session completion state.

    Only checks COMPETITIVE sessions (SQ, Sprint, Quali, Race).
    """
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    return detector.get_session_completion_state(year, race_name, session_name)


def is_competitive_session_completed(year: int, race_name: str, session_name: str) -> bool:
    """Backward-compatible boolean completion check for competitive sessions."""
    return get_competitive_session_completion_state(year, race_name, session_name) == "completed"
