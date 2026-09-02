"""Extract and blend FP session performance with model predictions."""

import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import fastf1 as ff1
import numpy as np
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from src.utils.fp_blending_flow import (
    blend_available_sessions,
    build_session_priority,
    robust_spread,
)
from src.utils.fp_blending_flow import (
    extract_team_performance_from_laps as _extract_team_performance_from_laps,
)
from src.utils.prediction_context import (
    get_config_value,
    get_prediction_reference_now,
    get_session_freshness_age,
)
from src.utils.team_mapping import map_team_to_characteristics

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
_ENABLED_FASTF1_CACHE_DIR: Path | None = None

_SESSION_DURATION_HOURS: dict[str, float] = {
    "FP1": 1.5,
    "FP2": 1.5,
    "FP3": 1.5,
    "SQ": 1.5,
    "Sprint": 1.0,
    "Sprint Qualifying": 1.5,
}
_FASTF1_ERRORS = (
    AttributeError,
    DataNotLoadedError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _iter_fastf1_cache_dirs() -> tuple[Path, ...]:
    """Return candidate FastF1 cache roots in priority order."""
    candidates: list[Path] = []
    env_cache_dir = os.getenv("F1_CACHE_DIR")
    if env_cache_dir:
        candidates.append(Path(env_cache_dir).expanduser())
    candidates.extend(
        [
            Path("data/raw/.fastf1_cache"),
            Path("data/raw/.fastf1_cache_testing"),
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)

    return tuple(unique_candidates)


def _enable_fastf1_cache(cache_dir: Path) -> None:
    """Enable one FastF1 cache directory if it is not already active."""
    global _ENABLED_FASTF1_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    if _ENABLED_FASTF1_CACHE_DIR == cache_dir:
        return

    ff1.Cache.enable_cache(str(cache_dir))
    _ENABLED_FASTF1_CACHE_DIR = cache_dir


def _load_session_with_cache_fallback(
    *,
    year: int,
    race_name: str,
    session_type: str,
    weather: bool,
) -> Any:
    """Load one FastF1 session while falling back across known local cache roots."""
    last_error: Exception | None = None

    def load_session_metadata() -> Any:
        """Fetch one FastF1 session object for the requested event."""
        return ff1.get_session(year, race_name, session_type)

    for cache_dir in _iter_fastf1_cache_dirs():
        try:
            _enable_fastf1_cache(cache_dir)
            session = _fastf1_with_retry(load_session_metadata)
            if session is None:
                continue

            def load_session_laps(current_session: Any = session) -> Any:
                """Load the requested session payload without telemetry/messages."""
                return current_session.load(
                    laps=True,
                    telemetry=False,
                    weather=weather,
                    messages=False,
                )

            _fastf1_with_retry(load_session_laps)
            return session
        except _FASTF1_ERRORS as exc:
            last_error = exc
            logger.debug(
                "FastF1 load failed for %s %s %s via cache %s: %s",
                year,
                race_name,
                session_type,
                cache_dir,
                exc,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("No FastF1 cache directories are configured")


def _get_event_with_cache_fallback(year: int, race_name: str) -> Any:
    """Load one FastF1 event while reusing known local cache roots."""
    last_error: Exception | None = None

    def load_event_metadata() -> Any:
        """Fetch one FastF1 event object for the requested race."""
        return ff1.get_event(year, race_name)

    for cache_dir in _iter_fastf1_cache_dirs():
        try:
            _enable_fastf1_cache(cache_dir)
            event = _fastf1_with_retry(load_event_metadata, max_retries=1)
            if event is not None:
                return event
        except _FASTF1_ERRORS as exc:
            last_error = exc
            logger.debug(
                "FastF1 event lookup failed for %s %s via cache %s: %s",
                year,
                race_name,
                cache_dir,
                exc,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("No FastF1 cache directories are configured")


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
        except _FASTF1_ERRORS:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                logger.error("Circuit breaker opened after %s failures", self.failure_count)
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
        except _FASTF1_ERRORS as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2**attempt)
            logger.warning(
                "FastF1 API error (attempt %s/%s): %s: %s. Retrying in %ss...",
                attempt + 1,
                max_retries,
                e.__class__.__name__,
                e,
                format(delay, ".1f"),
            )
            time.sleep(delay)


class FPDataError(Enum):
    """Error codes for FP data extraction failures."""

    NOT_COMPLETED = "not_completed"
    API_FAILURE = "api_failure"
    STALE_DATA = "stale_data"
    INSUFFICIENT_LAPS = "insufficient_laps"
    WET_SESSION = "wet_session"


def _session_rain_fraction(session: Any) -> float | None:
    """Return the fraction of weather samples that reported rainfall."""
    try:
        weather_data = getattr(session, "weather_data", None)
    except _FASTF1_ERRORS:
        return None
    if not isinstance(weather_data, pd.DataFrame) or weather_data.empty:
        return None
    if "Rainfall" not in weather_data.columns:
        return None

    rainfall = weather_data["Rainfall"]
    try:
        rainfall_fraction = rainfall.astype(bool).mean()
    except (TypeError, ValueError):
        rainfall_numeric = pd.to_numeric(rainfall, errors="coerce").dropna()
        if rainfall_numeric.empty:
            return None
        rainfall_fraction = rainfall_numeric.gt(0).mean()
    return float(rainfall_fraction)


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
        logger.debug("Removed %s outlier laps (IQR threshold=%s)", removed, threshold)

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
        session = _load_session_with_cache_fallback(
            year=year,
            race_name=race_name,
            session_type=session_type,
            weather=True,
        )
        if session is None:
            return None, None, FPDataError.API_FAILURE

        if not hasattr(session, "laps") or session.laps is None or session.laps.empty:
            return None, None, FPDataError.NOT_COMPLETED

        # Enforce data freshness via config (default one week).
        if max_data_age_hours is None:
            max_data_age_hours = float(
                get_config_value("baseline_predictor.qualifying.max_session_age_hours", 168.0)
            )

        if hasattr(session, "date") and session.date:
            session_age = get_session_freshness_age(session.date)
            if session_age > timedelta(hours=max_data_age_hours):
                logger.warning(
                    "%s for %s is %sh old (max %sh) - rejecting stale data",
                    session_type,
                    race_name,
                    format(session_age.total_seconds() / 3600, ".1f"),
                    format(max_data_age_hours, ".1f"),
                )
                return None, None, FPDataError.STALE_DATA

        laps = session.laps
        rain_fraction = _session_rain_fraction(session)
        if rain_fraction is not None and rain_fraction > 0.30:
            logger.warning(
                "%s for %s had %.0f%% rainfall - rejecting wet session data",
                session_type,
                race_name,
                rain_fraction * 100,
            )
            return None, None, FPDataError.WET_SESSION
        min_long_run_laps = int(
            get_config_value("baseline_predictor.race.weekend_long_run_min_laps", 12)
        )
        preferred_short_run_compound = get_config_value(
            "baseline_predictor.qualifying.preferred_short_run_compound", "SOFT"
        )
        long_run_outlier_threshold = get_config_value(
            "baseline_predictor.race.long_run_outlier_threshold", 1.5
        )
        long_run_trim_ends = bool(
            get_config_value("baseline_predictor.race.long_run_trim_ends", True)
        )
        fp_normalization = (
            str(get_config_value("baseline_predictor.qualifying.fp_normalization", "robust"))
            .strip()
            .lower()
        )
        fp_spread_k = float(
            get_config_value("baseline_predictor.qualifying.fp_robust_spread_k", 2.0)
        )
        fp_min_driver_laps = int(
            get_config_value("baseline_predictor.qualifying.fp_min_driver_laps", 4)
        )

        # Reject red-flagged/truncated sessions (<10 total laps)
        if len(laps) < 10:
            logger.warning(
                "%s for %s has only %s laps - likely red-flagged, rejecting",
                session_type,
                race_name,
                len(laps),
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
            normalization=fp_normalization,
            spread_k=fp_spread_k,
            min_driver_laps=fp_min_driver_laps,
        )
        if team_performance is None:
            return None, None, FPDataError.INSUFFICIENT_LAPS

        return team_performance, laps, None

    except _FASTF1_ERRORS as e:
        logger.warning(
            "FastF1 API failure for %s at %s (%s): %s: %s",
            session_type,
            race_name,
            year,
            e.__class__.__name__,
            e,
        )
        return None, None, FPDataError.API_FAILURE


def get_fp_session_weather(year: int, race_name: str, session_type: str) -> str | None:
    """Infer FP session weather context from compound usage."""
    try:
        session = _load_session_with_cache_fallback(
            year=year,
            race_name=race_name,
            session_type=session_type,
            weather=True,
        )
        if session is None or not hasattr(session, "laps"):
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
            return "rain"
        return "mixed"
    except _FASTF1_ERRORS as e:
        logger.warning("Could not determine weather for %s: %s", session_type, e)
        return None


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
        event = _get_event_with_cache_fallback(year, race_name)
    except _FASTF1_ERRORS:
        event = None

    now_utc = get_prediction_reference_now()
    errors_encountered = []
    available_sessions: list[dict[str, Any]] = []
    session_laps_by_code: dict[str, pd.DataFrame | None] = {}
    for session_code, session_label, session_weight in session_priority:
        if event is not None:
            try:
                raw_session_date = event.get_session_date(session_code)
            except _FASTF1_ERRORS:
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
                        "%s weather was %s but race prediction uses %s; FP blend confidence may be reduced",
                        primary_code,
                        fp_weather,
                        predicted_race_weather,
                    )
                else:
                    logger.warning(
                        "Primary FP weather (%s: %s) differs from race prediction weather (%s)",
                        primary_code,
                        fp_weather,
                        predicted_race_weather,
                    )
        logger.info("Using %s for blending", session_label)
        return session_label, blended_data, blended_laps, session_laps_by_code

    # Explain why practice data could not be blended.
    if errors_encountered:
        error_summary = ", ".join([f"{s}: {e.value}" for s, e in errors_encountered])
        logger.info("No valid practice data (%s) - using model-only predictions", error_summary)
    else:
        logger.info("No practice data available - using model-only predictions")

    return None, None, None, session_laps_by_code


def _scale_align_fp_to_model(
    model_strength: dict[str, float],
    fp_performance: dict[str, float],
    missing_from_fp: set[str],
) -> dict[str, float]:
    """Re-express the FP signal on the model strength's location and spread.

    The raw FP signal and the model strength are not on the same scale: FP pace
    is normalized within a single session (full 0-1 band, anchored to session
    extremes) while model strength lives in a calibrated, compressed band. Mixing
    them directly lets a noisy session push a team far outside its plausible
    range. Aligning the FP signal's median and (robust) spread onto the model's
    means a session can only *reorder teams within the model's band* rather than
    replace the prior wholesale. Returns aligned FP scores for the teams shared
    by both inputs; callers fall back to the raw value for any other team.
    """
    if not bool(get_config_value("baseline_predictor.qualifying.fp_scale_align", True)):
        return dict(fp_performance)

    common = [t for t in model_strength if t in fp_performance and t not in missing_from_fp]
    if len(common) < 3:
        # Too few shared teams to estimate a stable spread; leave FP untouched.
        return dict(fp_performance)

    fp_vals = np.asarray([float(fp_performance[t]) for t in common], dtype=float)
    model_vals = np.asarray([float(model_strength[t]) for t in common], dtype=float)
    fp_median = float(np.median(fp_vals))
    model_median = float(np.median(model_vals))
    fp_spread = robust_spread(fp_vals)
    model_spread = robust_spread(model_vals)

    if fp_spread <= 1e-9:
        # FP carries no usable spread; keep the model's own location for these teams.
        return {team: float(model_strength[team]) for team in common}

    ratio = float(get_config_value("baseline_predictor.qualifying.fp_align_spread_ratio", 1.0))
    target_spread = model_spread * max(0.0, ratio)
    return {
        team: float(
            np.clip(
                model_median
                + (float(fp_performance[team]) - fp_median) * (target_spread / fp_spread),
                0.0,
                1.0,
            )
        )
        for team in common
    }


def blend_team_strength(
    model_strength: dict[str, float],
    fp_performance: dict[str, float] | None,
    blend_weight: float = 0.7,
) -> dict[str, float]:
    """Blend model strength with scale-aligned FP pace (default 70% practice)."""
    if fp_performance is None:
        return model_strength

    # Check that both inputs cover the same teams.
    model_teams = set(model_strength.keys())
    fp_teams = set(fp_performance.keys())

    missing_from_fp = model_teams - fp_teams
    extra_in_fp = fp_teams - model_teams

    if missing_from_fp:
        logger.warning(
            "Teams in model but missing from FP data (using model-only): %s",
            ", ".join(sorted(missing_from_fp)),
        )

    if extra_in_fp:
        logger.debug(
            "Teams in FP data but not in model (ignoring): %s", ", ".join(sorted(extra_in_fp))
        )

    aligned_fp = _scale_align_fp_to_model(model_strength, fp_performance, missing_from_fp)

    blended = {}

    for team, model_score in model_strength.items():
        if team in missing_from_fp:
            logger.debug("  %s: Model-only (no FP data) = %s", team, format(model_score, ".3f"))
            blended[team] = model_score
            continue

        fp_score = aligned_fp.get(team, fp_performance.get(team, model_score))
        blended_score = blend_weight * fp_score + (1 - blend_weight) * model_score
        logger.debug(
            "  %s: FP=%s (raw %s), Model=%s → Blended=%s",
            team,
            format(fp_score, ".3f"),
            format(float(fp_performance.get(team, model_score)), ".3f"),
            format(model_score, ".3f"),
            format(blended_score, ".3f"),
        )
        blended[team] = blended_score

    return blended
