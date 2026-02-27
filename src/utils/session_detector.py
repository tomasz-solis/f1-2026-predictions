"""Detects completed F1 sessions for prediction tracking."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fastf1

from src.utils.fastf1_resilience import call_with_resilience
from src.utils.operational_observability import record_alert, record_counter

logger = logging.getLogger(__name__)

SessionCompletionState = Literal["completed", "incomplete", "unknown"]


class SessionCompletionCheckError(RuntimeError):
    """Raised when FastF1 session completion cannot be determined reliably."""


class SessionDetector:
    """Detects completed F1 sessions for a race weekend."""

    # Session names mapping
    NORMAL_WEEKEND_SESSIONS = ["FP1", "FP2", "FP3"]
    SPRINT_WEEKEND_SESSIONS = [
        "FP1",
        "SQ",
        "Sprint",
    ]  # SQ = Sprint Qualifying, Sprint = Sprint Race

    # Fallback durations in hours when explicit session status is unavailable.
    SESSION_DURATIONS = {
        "FP1": 1.5,
        "FP2": 1.5,
        "FP3": 1.5,
        "SQ": 1.5,  # Sprint Qualifying
        "Sprint": 1.0,  # Sprint race is shorter
        "Q": 1.5,  # Qualifying
        "R": 2.5,  # Race
    }
    FINAL_STATUS_TOKENS = ("FINISHED", "FINALISED", "FINALIZED", "ENDED", "ABORTED")
    ACTIVE_STATUS_TOKENS = ("STARTED", "GREEN", "RUNNING", "RESTART", "SUSPENDED")

    def __init__(self):
        cache_dir = Path("data/raw/.fastf1_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            fastf1.Cache.enable_cache(str(cache_dir))
        except Exception as exc:
            logger.debug(f"Could not enable FastF1 cache in SessionDetector: {exc}")
        self._event_cache: dict[tuple[int, str], Any] = {}
        self._completion_cache: dict[tuple[int, str, str], SessionCompletionState] = {}

    def get_completed_sessions(self, year: int, race_name: str, is_sprint: bool) -> list[str]:
        """Get completed sessions for a race weekend."""
        try:
            # Determine which sessions to check based on weekend type
            sessions_to_check = (
                self.SPRINT_WEEKEND_SESSIONS if is_sprint else self.NORMAL_WEEKEND_SESSIONS
            )

            completed: list[str] = []
            for session_name in sessions_to_check:
                try:
                    completion_state = self.get_session_completion_state(
                        year, race_name, session_name
                    )
                    if completion_state == "completed":
                        completed.append(session_name)
                except Exception as exc:
                    logger.debug(f"Could not check session {session_name} for {race_name}: {exc}")
                    continue

            if not completed:
                logger.info(
                    f"No completed sessions found for {race_name} {year} "
                    f"(checked: {', '.join(sessions_to_check)})"
                )

            return completed

        except Exception as exc:
            error_msg = str(exc)
            if "not found" in error_msg.lower():
                logger.error(f"Race not found: {race_name} {year}. Check race name spelling.")
            elif "connect" in error_msg.lower() or "network" in error_msg.lower():
                logger.error(f"FastF1 API connection failed: {error_msg}")
            else:
                logger.error(f"Failed to detect sessions for {race_name} {year}: {error_msg}")
            return []

    def get_latest_completed_session(
        self, year: int, race_name: str, is_sprint: bool
    ) -> str | None:
        """Get the most recent completed session."""
        completed = self.get_completed_sessions(year, race_name, is_sprint)
        return completed[-1] if completed else None

    def is_session_completed(self, year: int, race_name: str, session_name: str) -> bool:
        """Backward-compatible completion check used by existing callers."""
        return self.get_session_completion_state(year, race_name, session_name) == "completed"

    def get_session_completion_state(
        self, year: int, race_name: str, session_name: str
    ) -> SessionCompletionState:
        """Check if a specific session is completed, incomplete, or unknown."""
        cache_key = (int(year), str(race_name), str(session_name))
        cached_state = self._completion_cache.get(cache_key)
        if cached_state is not None:
            return cached_state

        try:
            state = self._compute_session_completion_state(year, race_name, session_name)
        except SessionCompletionCheckError as exc:
            logger.warning(
                "Could not determine completion state for %s %s %s: %s",
                year,
                race_name,
                session_name,
                exc,
            )
            state = "unknown"
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Unexpected error while checking completion state for %s %s %s: %s",
                year,
                race_name,
                session_name,
                exc,
            )
            state = "unknown"

        if state == "unknown":
            labels = {"year": year, "race_name": race_name, "session_name": session_name}
            record_counter("fastf1_completion_unknown_total", labels=labels)
            record_alert(
                "fastf1_completion_unknown",
                (
                    "FastF1 completion status unknown; refusing lower-fidelity fallback "
                    f"for {race_name} {year} {session_name}."
                ),
                labels=labels,
            )

        self._completion_cache[cache_key] = state
        return state

    def _get_event_cached(self, year: int, race_name: str):
        cache_key = (int(year), str(race_name))
        cached_event = self._event_cache.get(cache_key)
        if cached_event is not None:
            return cached_event
        event = call_with_resilience(
            "fastf1_get_event",
            lambda: fastf1.get_event(year, race_name),
            labels={"year": year, "race_name": race_name},
        )
        self._event_cache[cache_key] = event
        return event

    def _compute_session_completion_state(
        self, year: int, race_name: str, session_name: str
    ) -> SessionCompletionState:
        """Internal completion check with strict error semantics."""
        event = self._get_event_cached(year, race_name)
        try:
            session_date = event.get_session_date(session_name)
        except Exception as exc:  # pragma: no cover - fastf1 edge behavior
            raise SessionCompletionCheckError(f"could not load session schedule: {exc}") from exc

        if session_date is None:
            return "incomplete"

        if session_date.tzinfo is None:
            session_date_utc = session_date.replace(tzinfo=UTC)
        else:
            session_date_utc = session_date.astimezone(UTC)

        now = datetime.now(UTC)
        if now < session_date_utc:
            return "incomplete"

        try:
            session = call_with_resilience(
                "fastf1_get_session",
                lambda: fastf1.get_session(year, race_name, session_name),
                labels={"year": year, "race_name": race_name, "session_name": session_name},
            )
        except Exception as exc:  # pragma: no cover - fastf1 edge behavior
            raise SessionCompletionCheckError(f"could not create FastF1 session: {exc}") from exc

        if session is None:
            raise SessionCompletionCheckError("FastF1 returned no session object")

        # Practice sessions require non-empty lap data; competitive sessions require results.
        is_practice = str(session_name).upper().startswith("FP")
        if is_practice:
            try:
                call_with_resilience(
                    "fastf1_session_load_practice",
                    lambda: session.load(laps=True, telemetry=False, weather=False, messages=False),
                    labels={"year": year, "race_name": race_name, "session_name": session_name},
                )
            except Exception as exc:
                raise SessionCompletionCheckError(f"practice session load failed: {exc}") from exc

            laps = getattr(session, "laps", None)
            has_laps = laps is not None and not laps.empty
            if not has_laps:
                return "incomplete"

            status_complete = self._session_status_completed(session)
            if status_complete is not None:
                return "completed" if status_complete else "incomplete"

            return (
                "completed"
                if self._fallback_elapsed_completion(session_date_utc, session_name, now)
                else "incomplete"
            )

        try:
            call_with_resilience(
                "fastf1_session_load_competitive",
                lambda: session.load(laps=False, telemetry=False, weather=False, messages=False),
                labels={"year": year, "race_name": race_name, "session_name": session_name},
            )
        except Exception as exc:
            raise SessionCompletionCheckError(f"competitive session load failed: {exc}") from exc

        results = getattr(session, "results", None)
        if results is None:
            return "incomplete"

        try:
            has_results = len(results) > 0
        except TypeError:
            has_results = True

        if not has_results:
            return "incomplete"

        status_complete = self._session_status_completed(session)
        if status_complete is False:
            return "incomplete"

        return "completed"

    def _session_status_completed(self, session) -> bool | None:
        """
        Check FastF1 session status feed for completion markers.

        Returns True/False when a clear status is available, otherwise None.
        """
        status_feed = getattr(session, "session_status", None)
        if status_feed is None:
            return None

        if getattr(status_feed, "empty", False):
            return None

        columns = getattr(status_feed, "columns", [])
        status_values = None
        for column in ("Status", "SessionStatus", "Message"):
            if column in columns:
                status_values = status_feed[column]
                break

        if status_values is None:
            return None

        try:
            cleaned = status_values.dropna().astype(str)
        except Exception:
            return None

        if cleaned.empty:
            return None

        latest = cleaned.iloc[-1].upper()
        if any(token in latest for token in self.FINAL_STATUS_TOKENS):
            return True
        if any(token in latest for token in self.ACTIVE_STATUS_TOKENS):
            return False
        return None

    def _fallback_elapsed_completion(
        self, session_date: datetime, session_name: str, now: datetime
    ) -> bool:
        """Fallback completion check when status stream is unavailable."""
        from datetime import timedelta

        duration = self.SESSION_DURATIONS.get(session_name, 2.0)
        return now >= session_date + timedelta(hours=duration)

    def get_sessions_for_weekend(self, is_sprint: bool) -> list[str]:
        """Get session names for a weekend type."""
        return self.SPRINT_WEEKEND_SESSIONS if is_sprint else self.NORMAL_WEEKEND_SESSIONS
