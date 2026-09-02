"""
Team Lineup Management - Race-by-Race

Extracts actual lineups from session data (handles all corner cases automatically).
Falls back to current_lineups.json for future predictions.
"""

import json
import logging
from pathlib import Path

import fastf1 as ff1

from src.utils.driver_substitutions import apply_substitutions

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def get_lineups_from_session(
    year: int,
    race_name: str,
    session_type: str = "Q",
) -> dict[str, list[str]] | None:
    """Extract actual race-weekend lineups from a loaded FastF1 session."""
    try:
        session = ff1.get_session(year, race_name, session_type)
        session.load(laps=False, telemetry=False, weather=False)

        if not hasattr(session, "results") or session.results is None:
            return None

        lineups = {}

        # Extract actual participants
        for team in session.results["TeamName"].unique():
            team_results = session.results[session.results["TeamName"] == team]
            drivers = team_results["Abbreviation"].tolist()

            # Take whoever actually participated
            if drivers:
                # Handle 1 or 2 drivers (edge case: DNF/DNS)
                lineups[team] = drivers[:2] if len(drivers) >= 2 else drivers

        return lineups

    except (
        AttributeError,
        ConnectionError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
        logger.warning(
            "Failed to extract lineups from %s (%s) %s session: %s. Lineups from this session will be unavailable.",
            race_name,
            year,
            session_type,
            e,
        )
        # Session data not available
        return None


def load_current_lineups(
    config_path: str = "data/current_lineups.json",
) -> dict[str, list[str]] | None:
    """Load the current lineup config used for future events and fallbacks."""
    config_file = Path(config_path)

    if not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not load lineup config %s: %s", config_file, exc)
        return None

    if not isinstance(data, dict):
        return None

    current_lineups = data.get("current_lineups", {})
    if not isinstance(current_lineups, dict):
        return None

    return {
        str(team_name): [str(driver_code) for driver_code in drivers]
        for team_name, drivers in current_lineups.items()
        if isinstance(drivers, list)
    }


def get_lineups(
    year: int,
    race_name: str | None = None,
    config_path: str = "data/current_lineups.json",
) -> dict[str, list[str]]:
    """Return the best available lineup mapping for one season or event.

    A race-scoped driver substitution (injury, a mid-season drop) is applied on top of
    whichever source answered, so every caller sees one consistent lineup for that race.
    """
    # For historical seasons with specific race, extract from data
    if year <= 2025 and race_name:
        session_lineups = get_lineups_from_session(year, race_name, "Q")

        if session_lineups:
            return apply_substitutions(session_lineups, race_name=race_name, year=year)

        # If session data failed, fall through to config

    # For future seasons or fallback, use current config
    current_lineups = load_current_lineups(config_path)

    if current_lineups:
        return apply_substitutions(current_lineups, race_name=race_name, year=year)

    raise ValueError(
        f"No lineup data available for {year}"
        + (f" - {race_name}" if race_name else "")
        + f"\nCreate config file at {config_path}"
    )
