"""Fetches actual results from competitive F1 sessions."""

import logging
from typing import Any, Literal, cast

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


def _coerce_optional_position(raw_position: Any) -> int | None:
    """Coerce a FastF1 position-like payload to an integer when present."""
    if raw_position is None:
        return None

    text = str(raw_position).strip()
    if text.lower() in {"", "nan", "none"}:
        return None

    try:
        return int(raw_position)
    except (TypeError, ValueError) as exc:
        try:
            numeric_position = float(raw_position)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid result position value: {raw_position!r}") from exc

        if not numeric_position.is_integer():
            raise ValueError(f"Invalid result position value: {raw_position!r}") from exc
        return int(numeric_position)


def _infer_trailing_qualifying_positions(
    grid_rows: list[dict[str, Any]],
    *,
    race_name: str,
    session_name: str,
    labels: dict[str, Any],
) -> None:
    """Fill trailing blank qualifying positions when FastF1 preserves row order."""
    if session_name not in {"Q", "SQ"}:
        return

    missing_indices = [
        index for index, entry in enumerate(grid_rows) if entry.get("position") is None
    ]
    if not missing_indices:
        return

    first_missing_index = missing_indices[0]
    if first_missing_index == 0:
        raise ValueError("no explicit qualifying positions available for inference")

    expected_missing = list(range(first_missing_index, len(grid_rows)))
    if missing_indices != expected_missing:
        raise ValueError("qualifying positions can only be inferred for a trailing missing block")

    known_positions = [cast(int, entry["position"]) for entry in grid_rows[:first_missing_index]]
    expected_known_positions = list(range(1, len(known_positions) + 1))
    if known_positions != expected_known_positions:
        raise ValueError(
            "qualifying positions can only be inferred when known positions are sequential"
        )

    next_position = len(known_positions) + 1
    for entry in grid_rows[first_missing_index:]:
        entry["position"] = next_position
        next_position += 1

    inferred_count = len(missing_indices)
    logger.warning(
        "FastF1 qualifying results for %s %s omitted %s trailing positions; "
        "inferring positions %s-%s from row order.",
        race_name,
        session_name,
        inferred_count,
        len(known_positions) + 1,
        len(grid_rows),
    )
    record_counter(
        "fastf1_results_position_inferred_total",
        labels={**labels, "count": inferred_count},
    )


def _refresh_partial_qualifying_results(
    session: Any,
    *,
    race_name: str,
    session_name: str,
    labels: dict[str, Any],
) -> Any:
    """Ask FastF1 to recompute missing qualifying positions from lap times when possible."""
    results = getattr(session, "results", None)
    if results is None or session_name not in {"Q", "SQ"}:
        return results

    columns = getattr(results, "columns", [])
    if "Position" not in columns:
        return results

    try:
        missing_positions = bool(results["Position"].isna().any())
    except (AttributeError, KeyError, TypeError, ValueError):
        return results

    if not missing_positions:
        return results

    recompute_results = getattr(session, "_calculate_quali_like_session_results", None)
    if not callable(recompute_results):
        return results

    try:
        recompute_results(force=True)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not recompute partial qualifying positions for %s %s: %s",
            race_name,
            session_name,
            exc,
        )
        record_counter("fastf1_results_recompute_failed_total", labels=labels)
        return results

    logger.info(
        "Recomputed partial qualifying positions from lap data for %s %s",
        race_name,
        session_name,
    )
    record_counter("fastf1_results_recomputed_total", labels=labels)
    return getattr(session, "results", results)


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
        results = _refresh_partial_qualifying_results(
            session,
            race_name=race_name,
            session_name=session_name,
            labels=labels,
        )

        if results is None or len(results) == 0:
            logger.warning("No results available for %s %s", race_name, session_name)
            record_counter("fastf1_results_empty_total", labels=labels)
            return None

        # Extract relevant data and fail closed on malformed rows.
        grid_rows: list[dict[str, Any]] = []
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

                position = _coerce_optional_position(row.get("Position"))
                if position is None:
                    position = _coerce_optional_position(row.get("ClassifiedPosition"))

                grid_rows.append(
                    {
                        "driver": driver,
                        "team": str(team).strip(),
                        "position": position,
                    }
                )
            except (AttributeError, KeyError, TypeError, ValueError) as e:
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

        try:
            _infer_trailing_qualifying_positions(
                grid_rows,
                race_name=race_name,
                session_name=session_name,
                labels=labels,
            )
            grid = validate_qualifying_grid(
                cast(list[QualifyingGridEntry], grid_rows),
                min_entries=_MIN_COMPETITIVE_ENTRIES_BY_SESSION.get(str(session_name), 10),
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

        # Sort by position
        grid.sort(key=lambda item: int(item["position"]))

        logger.info(
            "Fetched %s results from %s %s",
            len(grid),
            race_name,
            session_name,
        )
        return grid

    except (
        AttributeError,
        ConnectionError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
        logger.error("Failed to fetch %s results for %s: %s", session_name, race_name, e)
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
    """Return the completion state for a competitive session."""
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    return detector.get_session_completion_state(year, race_name, session_name)


def is_competitive_session_completed(year: int, race_name: str, session_name: str) -> bool:
    """Return `True` when a competitive session is complete."""
    return get_competitive_session_completion_state(year, race_name, session_name) == "completed"
