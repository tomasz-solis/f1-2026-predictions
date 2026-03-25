"""Extract and blend FP session performance with model predictions."""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import fastf1 as ff1
import pandas as pd

from src.utils import config_loader
from src.utils.fp_blending_flow import blend_available_sessions, build_session_priority
from src.utils.fp_blending_flow import (
    extract_team_performance_from_laps as _extract_team_performance_from_laps,
)
from src.utils.team_mapping import map_team_to_characteristics

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

_SESSION_DURATION_HOURS: dict[str, float] = {
    "FP1": 1.5,
    "FP2": 1.5,
    "FP3": 1.5,
    "SQ": 1.5,
    "Sprint": 1.0,
    "Sprint Qualifying": 1.5,
}


class CircuitBreaker:
    """Circuit breaker for FastF1 API rate limiting protection."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half_open

    def reset(self) -> None:
        """Reset circuit breaker to closed state (for testing)."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"

    def call(self, fn: Callable[[], Any]) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if (
                self.last_failure_time
                and (time.time() - self.last_failure_time) > self.recovery_timeout
            ):
                logger.info("Circuit breaker transitioning to half-open state")
                self.state = "half_open"
            else:
                raise RuntimeError(
                    "Circuit breaker is open; FastF1 requests are temporarily blocked"
                )

        try:
            result = fn()
            if self.state == "half_open":
                logger.info("Circuit breaker recovered, transitioning to closed state")
                self.failure_count = 0
                self.state = "closed"
            return result
        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                self.state = "open"
            raise


# Global circuit breaker instance (shared across all FastF1 calls)
_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)


def _fastf1_with_retry(
    fn: Callable[[], Any], max_retries: int = 3, initial_delay: float = 1.0
) -> Any:
    """Execute FastF1 API call with exponential backoff retry logic and circuit breaker protection."""
    for attempt in range(max_retries):
        try:
            return _circuit_breaker.call(fn)
        except RuntimeError as e:
            if "Circuit breaker is open" in str(e):
                raise
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2**attempt)
            logger.warning(
                f"FastF1 API error (attempt {attempt + 1}/{max_retries}): {e.__class__.__name__}: {e}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


class FPDataError(Enum):
    """Error codes for FP data extraction failures."""

    NOT_COMPLETED = "not_completed"
    API_FAILURE = "api_failure"
    STALE_DATA = "stale_data"
    INSUFFICIENT_LAPS = "insufficient_laps"


def _remove_outlier_laps(laps: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """Remove lap-time outliers using the IQR method."""
    if laps.empty:
        return laps

    lap_seconds = laps["LapTime"].dt.total_seconds().dropna()
    if lap_seconds.empty:
        return laps

    q1 = lap_seconds.quantile(0.25)
    q3 = lap_seconds.quantile(0.75)
    iqr = q3 - q1

    if iqr <= 0:
        return laps

    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)

    mask = laps["LapTime"].dt.total_seconds().between(lower_bound, upper_bound, inclusive="both")
    removed = int((~mask).sum())
    if removed > 0:
        logger.debug(f"Removed {removed} outlier laps (IQR threshold={threshold})")

    return laps[mask]


def _extract_short_run_lap_time(
    valid_laps: pd.DataFrame,
    target_compound: str | None = "SOFT",
) -> float | None:
    """Return a representative short-stint lap time (seconds) for a driver."""
    laps = valid_laps.copy()

    if "Compound" in laps.columns and target_compound:
        compound_laps = laps[laps["Compound"] == target_compound]
        if not compound_laps.empty:
            laps = compound_laps

    # Exclude in/out laps when available; short-run pace should reflect push laps.
    for pit_col in ("PitOutTime", "PitInTime"):
        if pit_col in laps.columns:
            laps = laps[laps[pit_col].isna()]

    if laps.empty:
        return None

    short_laps = pd.DataFrame()
    if "TyreLife" in laps.columns:
        tyre_life = pd.to_numeric(laps["TyreLife"], errors="coerce")
        short_laps = laps[(tyre_life >= 1) & (tyre_life <= 5)]

    candidates = short_laps if not short_laps.empty else laps
    candidates = _remove_outlier_laps(candidates)
    lap_seconds = candidates["LapTime"].dt.total_seconds().dropna()
    if lap_seconds.empty:
        return None

    top_n = max(1, min(3, len(lap_seconds)))
    return float(lap_seconds.nsmallest(top_n).median())


def _extract_long_run_lap_time(
    valid_laps: pd.DataFrame,
    min_long_run_laps: int,
    outlier_threshold: float,
    trim_ends: bool,
) -> float | None:
    """Return a representative long-stint lap time (seconds) for a driver."""
    laps = valid_laps.copy()
    if laps.empty:
        return None

    if "Stint" in laps.columns and laps["Stint"].notna().any():
        group_cols = ["Stint"]
        if "Compound" in laps.columns:
            group_cols.append("Compound")
        stints = [stint for _key, stint in laps.groupby(group_cols)]
    else:
        # Fallback when explicit stint numbering is unavailable.
        if "Compound" in laps.columns:
            laps = laps.sort_index()
            stint_ids = (laps["Compound"] != laps["Compound"].shift()).cumsum()
            stints = [stint for _key, stint in laps.groupby(stint_ids)]
        else:
            stints = [laps]

    best_time = None
    best_len = 0

    for stint in stints:
        if len(stint) < min_long_run_laps:
            continue

        stint = _remove_outlier_laps(stint, threshold=outlier_threshold)
        lap_seconds = stint["LapTime"].dt.total_seconds().dropna().sort_values()
        if len(lap_seconds) < min_long_run_laps:
            continue

        # Trim one lap from each end to reduce out/in-lap contamination.
        if trim_ends and len(lap_seconds) > 4:
            lap_seconds = lap_seconds.iloc[1:-1]

        rep_time = float(lap_seconds.mean())

        if len(stint) > best_len:
            best_len = len(stint)
            best_time = rep_time

    return best_time


def _extract_representative_lap_time(
    valid_laps: pd.DataFrame,
    run_focus: str,
    min_long_run_laps: int,
    preferred_short_run_compound: str | None,
    long_run_outlier_threshold: float,
    long_run_trim_ends: bool,
) -> float | None:
    """Extract representative lap time for short-run or long-run focus."""
    if run_focus == "long":
        return _extract_long_run_lap_time(
            valid_laps,
            min_long_run_laps=min_long_run_laps,
            outlier_threshold=long_run_outlier_threshold,
            trim_ends=long_run_trim_ends,
        )
    return _extract_short_run_lap_time(
        valid_laps,
        target_compound=preferred_short_run_compound,
    )


def get_fp_team_performance(
    year: int,
    race_name: str,
    session_type: str,
    max_data_age_hours: float | None = None,
    run_focus: str = "short",
) -> tuple[dict[str, float] | None, pd.DataFrame | None, FPDataError | None]:
    """Extract team performance from practice session with staleness and lap count validation."""
    if run_focus not in {"short", "long"}:
        raise ValueError("run_focus must be one of: 'short', 'long'")

    try:
        session = _fastf1_with_retry(lambda: ff1.get_session(year, race_name, session_type))
        if session is None:
            return None, None, FPDataError.API_FAILURE

        load_result = _fastf1_with_retry(
            lambda: session.load(laps=True, telemetry=False, weather=False)
        )
        if load_result is None:
            logger.debug(f"session.load() returned None for {session_type} at {race_name}")
            return None, None, FPDataError.API_FAILURE

        if not hasattr(session, "laps") or session.laps is None or session.laps.empty:
            return None, None, FPDataError.NOT_COMPLETED

        # Enforce data freshness via config (default one week).
        if max_data_age_hours is None:
            max_data_age_hours = float(
                config_loader.get("baseline_predictor.qualifying.max_session_age_hours", 168.0)
            )

        if hasattr(session, "date") and session.date:
            session_age = datetime.now(tz=session.date.tzinfo) - session.date
            if session_age > timedelta(hours=max_data_age_hours):
                logger.warning(
                    f"{session_type} for {race_name} is "
                    f"{session_age.total_seconds() / 3600:.1f}h old "
                    f"(max {max_data_age_hours:.1f}h) - rejecting stale data"
                )
                return None, None, FPDataError.STALE_DATA

        laps = session.laps
        min_long_run_laps = int(
            config_loader.get("baseline_predictor.race.weekend_long_run_min_laps", 12)
        )
        preferred_short_run_compound = config_loader.get(
            "baseline_predictor.qualifying.preferred_short_run_compound", "SOFT"
        )
        long_run_outlier_threshold = config_loader.get(
            "baseline_predictor.race.long_run_outlier_threshold", 1.5
        )
        long_run_trim_ends = bool(
            config_loader.get("baseline_predictor.race.long_run_trim_ends", True)
        )

        # Reject red-flagged/truncated sessions (<10 total laps)
        if len(laps) < 10:
            logger.warning(
                f"{session_type} for {race_name} has only {len(laps)} laps - likely red-flagged, rejecting"
            )
            return None, None, FPDataError.INSUFFICIENT_LAPS

        team_performance = _extract_team_performance_from_laps(
            laps=laps,
            run_focus=run_focus,
            min_long_run_laps=min_long_run_laps,
            preferred_short_run_compound=preferred_short_run_compound,
            long_run_outlier_threshold=float(long_run_outlier_threshold),
            long_run_trim_ends=long_run_trim_ends,
            extract_representative_lap_time_fn=_extract_representative_lap_time,
            map_team_to_characteristics_fn=map_team_to_characteristics,
        )
        if team_performance is None:
            return None, None, FPDataError.INSUFFICIENT_LAPS

        return team_performance, laps, None

    except Exception as e:
        logger.warning(
            f"FastF1 API failure for {session_type} at {race_name} ({year}): {e.__class__.__name__}: {e}"
        )
        return None, None, FPDataError.API_FAILURE


def get_fp_session_weather(year: int, race_name: str, session_type: str) -> str | None:
    """Infer FP session weather context from compound usage."""
    try:
        session = _fastf1_with_retry(lambda: ff1.get_session(year, race_name, session_type))
        if session is None:
            return None

        load_result = _fastf1_with_retry(
            lambda: session.load(laps=True, telemetry=False, weather=True)
        )
        if load_result is None or not hasattr(session, "laps"):
            return None

        laps = session.laps
        if laps is None or laps.empty or "Compound" not in laps.columns:
            return None

        wet_compounds = {"INTERMEDIATE", "WET"}
        wet_laps = laps[laps["Compound"].isin(wet_compounds)]
        total_laps = len(laps)
        if total_laps == 0:
            return None

        wet_ratio = len(wet_laps) / total_laps
        if wet_ratio == 0:
            return "dry"
        if wet_ratio > 0.5:
            return "wet"
        return "mixed"
    except Exception as e:
        logger.warning(f"Could not determine weather for {session_type}: {e}")
        return None


def get_best_fp_performance(
    year: int,
    race_name: str,
    is_sprint: bool = False,
    qualifying_stage: str = "auto",
    predicted_race_weather: str | None = None,
) -> tuple[str | None, dict[str, float] | None, pd.DataFrame | None]:
    """Get best available practice session with staleness checks and error reporting."""
    session_label, performance, laps, _ = get_best_fp_performance_with_session_laps(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        qualifying_stage=qualifying_stage,
        predicted_race_weather=predicted_race_weather,
    )
    return session_label, performance, laps


def get_best_fp_performance_with_session_laps(
    year: int,
    race_name: str,
    is_sprint: bool = False,
    qualifying_stage: str = "auto",
    predicted_race_weather: str | None = None,
) -> tuple[
    str | None,
    dict[str, float] | None,
    pd.DataFrame | None,
    dict[str, pd.DataFrame | None],
]:
    """
    Get the best FP blend and expose per-session laps used to build it.

    The additional laps map allows downstream callers to reuse already loaded
    session data (for example, driver-level FP adjustments) and avoid duplicate
    FastF1 calls in the same prediction run.
    """
    stage = (qualifying_stage or "auto").strip().lower()
    if stage not in {"auto", "sprint", "main"}:
        raise ValueError("qualifying_stage must be one of: 'auto', 'sprint', 'main'")

    session_priority = build_session_priority(
        is_sprint=is_sprint,
        qualifying_stage=stage,
    )

    event = None
    try:
        event = _fastf1_with_retry(lambda: ff1.get_event(year, race_name), max_retries=1)
    except Exception:
        event = None

    now_utc = datetime.now(UTC)
    errors_encountered = []
    available_sessions: list[dict[str, Any]] = []
    session_laps_by_code: dict[str, pd.DataFrame | None] = {}
    for session_code, session_label, session_weight in session_priority:
        if event is not None:
            try:
                raw_session_date = event.get_session_date(session_code)
            except Exception:
                raw_session_date = None

            if raw_session_date is not None:
                if raw_session_date.tzinfo is None:
                    session_date_utc = raw_session_date.replace(tzinfo=UTC)
                else:
                    session_date_utc = raw_session_date.astimezone(UTC)
                estimated_duration = _SESSION_DURATION_HOURS.get(session_code, 2.0)
                if now_utc < session_date_utc + timedelta(hours=float(estimated_duration)):
                    session_laps_by_code[session_code] = None
                    continue

        fp_data, session_laps, error = get_fp_team_performance(year, race_name, session_code)
        session_laps_by_code[session_code] = (
            session_laps if isinstance(session_laps, pd.DataFrame) else None
        )
        if fp_data is not None:
            available_sessions.append(
                {
                    "code": session_code,
                    "label": session_label,
                    "weight": float(session_weight),
                    "data": fp_data,
                    "laps": session_laps,
                }
            )
            continue
        if error:
            errors_encountered.append((session_code, error))

    blended_result = blend_available_sessions(available_sessions)
    if blended_result is not None:
        session_label, blended_data, blended_laps, primary_code = blended_result
        if predicted_race_weather is not None:
            fp_weather = get_fp_session_weather(year, race_name, primary_code)
            if fp_weather and fp_weather != predicted_race_weather:
                if len(available_sessions) == 1:
                    logger.warning(
                        f"{primary_code} weather was {fp_weather} but race prediction uses "
                        f"{predicted_race_weather}; FP blend confidence may be reduced"
                    )
                else:
                    logger.warning(
                        f"Primary FP weather ({primary_code}: {fp_weather}) differs from "
                        f"race prediction weather ({predicted_race_weather})"
                    )
        logger.info(f"Using {session_label} for blending")
        return session_label, blended_data, blended_laps, session_laps_by_code

    # Explain why practice data could not be blended.
    if errors_encountered:
        error_summary = ", ".join([f"{s}: {e.value}" for s, e in errors_encountered])
        logger.info(f"No valid practice data ({error_summary}) - using model-only predictions")
    else:
        logger.info("No practice data available - using model-only predictions")

    return None, None, None, session_laps_by_code


def blend_team_strength(
    model_strength: dict[str, float],
    fp_performance: dict[str, float] | None,
    blend_weight: float = 0.7,
) -> dict[str, float]:
    """Blend model predictions with FP data (70% practice + 30% model)."""
    if fp_performance is None:
        return model_strength

    # Check that both inputs cover the same teams.
    model_teams = set(model_strength.keys())
    fp_teams = set(fp_performance.keys())

    missing_from_fp = model_teams - fp_teams
    extra_in_fp = fp_teams - model_teams

    if missing_from_fp:
        logger.warning(
            f"Teams in model but missing from FP data (using model-only): {', '.join(sorted(missing_from_fp))}"
        )

    if extra_in_fp:
        logger.debug(
            f"Teams in FP data but not in model (ignoring): {', '.join(sorted(extra_in_fp))}"
        )

    blended = {}

    for team, model_score in model_strength.items():
        fp_score = fp_performance.get(team, model_score)

        if team in missing_from_fp:
            logger.debug(f"  {team}: Model-only (no FP data) = {model_score:.3f}")
            blended[team] = model_score
        else:
            blended_score = blend_weight * fp_score + (1 - blend_weight) * model_score
            logger.debug(
                f"  {team}: FP={fp_score:.3f}, Model={model_score:.3f} "
                f"→ Blended={blended_score:.3f}"
            )
            blended[team] = blended_score

    return blended
