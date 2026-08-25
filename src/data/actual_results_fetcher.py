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
_QUALIFYING_LIKE_SESSIONS = {"Q", "SQ"}
_RACE_LIKE_SESSIONS = {"R", "SPRINT"}


def _clean_result_str(value: Any) -> str:
    """Coerce a FastF1 result cell to a trimmed string, treating NaN/none as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _result_row_is_dnf(row: Any) -> bool:
    """Return whether a race-result row is a did-not-finish / not-classified outcome.

    ``ClassifiedPosition`` is authoritative when present: a numeric value means the
    driver was classified (finished, possibly lapped), while a letter (``R``etired,
    ``D``isqualified, ``E``xcluded, ``W``ithdrawn, ``N``ot classified, ...) means a
    DNF. When it is absent the FastF1 ``Status`` string is used, where only
    "Finished" and "+N Lap(s)"/"Lapped" count as classified.
    """
    classified = _clean_result_str(row.get("ClassifiedPosition"))
    if classified:
        return not classified.isdigit()
    status = _clean_result_str(row.get("Status")).lower()
    if status:
        return not (status.startswith("finished") or "lap" in status)
    return False


def _session_load_options(session_name: str) -> dict[str, bool]:
    """Return the smallest FastF1 load options needed for final results."""
    is_qualifying_like = str(session_name).strip().upper() in _QUALIFYING_LIKE_SESSIONS
    return {
        "laps": is_qualifying_like,
        "telemetry": False,
        "weather": False,
        "messages": is_qualifying_like,
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
            lambda: session.load(**_session_load_options(session_name)),
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

        # Extract relevant data; skip malformed rows.
        is_race_like_session = str(session_name).strip().upper() in _RACE_LIKE_SESSIONS
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

                grid_row: dict[str, Any] = {
                    "driver": driver,
                    "team": str(team).strip(),
                    "position": position,
                }
                # Record DNF status for race-like sessions so finisher-only MAE and DNF
                # calibration can use it. Qualifying has no DNF concept.
                if is_race_like_session:
                    grid_row["dnf"] = _result_row_is_dnf(row)
                grid_rows.append(grid_row)
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


def fetch_actual_starting_grid(year: int, race_name: str) -> list[QualifyingGridEntry] | None:
    """Fetch the grid a completed race actually started from, penalties included.

    Qualifying classification is not the starting grid: a driver who qualifies P3 and
    takes a ten-place penalty is still classified P3 by the timing feed. FastF1 carries
    the post-penalty slot as ``GridPosition`` on the race session, which only exists once
    the race has run. Before that there is no automated source and callers must fall back
    to the qualifying classification.

    Returns ``None`` when the race has not run or the grid does not reconcile.
    """
    labels = {"year": year, "race_name": race_name, "session_name": "R"}
    try:
        session = call_with_resilience(
            "fastf1_get_session",
            lambda: fastf1.get_session(year, race_name, "R"),
            labels=labels,
        )
        call_with_resilience(
            "fastf1_session_load_results",
            lambda: session.load(**_session_load_options("R")),
            labels=labels,
        )
        results = getattr(session, "results", None)
        if results is None or len(results) == 0:
            logger.warning("No race results available for %s %s", race_name, year)
            record_counter("fastf1_starting_grid_empty_total", labels=labels)
            return None

        gridded: list[dict[str, Any]] = []
        pit_lane: list[dict[str, Any]] = []
        for _, row in results.iterrows():
            driver = _clean_result_str(row.get("Abbreviation")) or _clean_result_str(
                row.get("DriverNumber")
            )
            team_raw = _clean_result_str(row.get("TeamName"))
            if not driver or not team_raw:
                logger.error(
                    "Malformed FastF1 starting-grid row for %s %s: driver=%r team=%r",
                    race_name,
                    year,
                    driver,
                    team_raw,
                )
                record_counter("fastf1_starting_grid_malformed_total", labels=labels)
                return None
            entry: dict[str, Any] = {
                "driver": driver,
                "team": map_team_to_characteristics(team_raw) or team_raw,
                "classification": _coerce_optional_position(row.get("Position")),
            }
            slot = _coerce_optional_position(row.get("GridPosition"))
            if slot is None:
                logger.error("Missing GridPosition for %s at %s %s", driver, race_name, year)
                record_counter("fastf1_starting_grid_malformed_total", labels=labels)
                return None
            if slot == 0:
                # FastF1 reports a pit-lane start as grid slot zero. It is a real start,
                # not a missing value, and it holds no place on the grid itself.
                pit_lane.append(entry)
                continue
            entry["slot"] = slot
            gridded.append(entry)

        grid_rows: list[dict[str, Any]] = []
        for entry in sorted(gridded, key=lambda item: int(item["slot"])):
            grid_rows.append(
                {
                    "driver": entry["driver"],
                    "team": entry["team"],
                    "position": int(entry["slot"]),
                    "start_type": "grid",
                }
            )
        # Pit-lane starters hold no grid slot. They take the places behind the last
        # gridded car, in classification order, so the grid stays a 1-N permutation.
        next_position = len(grid_rows) + 1
        for entry in sorted(
            pit_lane,
            key=lambda item: (item["classification"] is None, item["classification"] or 0),
        ):
            grid_rows.append(
                {
                    "driver": entry["driver"],
                    "team": entry["team"],
                    "position": next_position,
                    "start_type": "pit_lane",
                }
            )
            next_position += 1

        try:
            grid = validate_qualifying_grid(
                cast(list[QualifyingGridEntry], grid_rows),
                min_entries=_MIN_COMPETITIVE_ENTRIES_BY_SESSION["R"],
                require_sequential_positions=True,
            )
        except ValueError as exc:
            logger.error("Invalid FastF1 starting grid for %s %s: %s", race_name, year, exc)
            record_counter("fastf1_starting_grid_invalid_total", labels=labels)
            return None

        logger.info("Fetched starting grid for %s %s: %s entries", race_name, year, len(grid))
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
        logger.error("Failed to fetch starting grid for %s %s: %s", race_name, year, e)
        record_counter("fastf1_starting_grid_fetch_failure_total", labels=labels)
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
