"""Fetches actual results from competitive F1 sessions."""

import logging
from typing import Any

import fastf1

from src.utils.grid_validation import validate_qualifying_grid
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)
_MIN_COMPETITIVE_ENTRIES_BY_SESSION = {
    "SQ": 10,
    "Sprint": 10,
    "Q": 10,
    "R": 10,
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
) -> list[dict[str, Any]] | None:
    """Fetch actual results from competitive session (SQ, Sprint, Q, R)."""
    try:
        # Load session
        session = fastf1.get_session(year, race_name, session_name)
        session.load()

        # Get results
        results = session.results

        if results is None or len(results) == 0:
            logger.warning(f"No results available for {race_name} {session_name}")
            return None

        # Extract relevant data and fail closed on malformed rows.
        grid: list[dict[str, int | str]] = []
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
        return None


def is_competitive_session_completed(year: int, race_name: str, session_name: str) -> bool:
    """
    Check if a competitive session has completed and results are available.

    Only checks COMPETITIVE sessions (SQ, Sprint, Quali, Race).
    """
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    return detector.is_session_completed(year, race_name, session_name)
