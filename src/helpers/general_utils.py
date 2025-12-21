"""

Originally from: https://github.com/tomasz-solis/formula1
Adapted for 2026 Bayesian prediction system.

==============================================================================

General utilities for the F1 analytics pipeline.

This module provides core infrastructure for F1 data processing including:
- Session loading with automatic FastF1→OpenF1 fallback
- Event schedule management and session filtering
- Weather data extraction
- Elevation data lookup via Open-Meteo API
- Caching and file management utilities
- Progress bar suppression for nested iterations

These utilities are used across all other helper modules to provide
consistent data access patterns and robust error handling.

Example:
    >>> from helpers.general_utils import load_session, get_elevation
    >>> info = load_session(2024, 'Monaco Grand Prix', 'Q')
    >>> if info['status'] == 'ok':
    ...     print(f"Loaded via {info['source']}")

Author: Tomasz Solis
Date: November 2025
"""

# Library imports
import os
import glob
import logging
import functools
import requests
import pandas as pd
import fastf1 as ff1
import warnings
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from contextlib import contextmanager
from tqdm import tqdm


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Suppress noisy FutureWarnings from fastf1
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1"
)
# Set logging level to ERROR to minimize output
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def _suppress_inner_tqdm():
    """
    Context manager to suppress nested tqdm progress bars.
    
    Monkey-patches tqdm.__init__ to force disable=True on all inner progress
    bars created within the context. Restores original behavior on exit.
    Prevents visual clutter when functions with progress bars call other
    functions that also use progress bars.
    
    Yields:
        None
        
    Example:
        >>> with _suppress_inner_tqdm():
        ...     for year in tqdm(years):  # This shows
        ...         for event in tqdm(events):  # This is suppressed
        ...             process(event)
                
    Note:
        Uses monkey-patching - not thread-safe. Intended for single-threaded
        scripts with nested iteration patterns.
    """
    try:
        from tqdm import tqdm as tqdm_module
        original = tqdm_module.__init__
        # Monkey-patch to disable nested bars
        tqdm_module.__init__ = lambda self, *a, **kw: original(self, *a, **{**kw, "disable": True})
        yield
    finally:
        # Restore original tqdm constructor
        tqdm_module.__init__ = original


# =============================================================================
# SESSION LOADING AND CACHING
# =============================================================================

def _session_date_col(event_format: str, event_row: pd.Series) -> dict[str, str]:
    """
    Map session symbolic names to FastF1 schedule date column names.
    
    FastF1 schedules use generic columns (Session1, Session2, etc.) that
    map to different session types depending on event format. Creates
    mapping from symbolic names (FP1, Q, R) to actual column names.
    
    Args:
        event_format: Event format from schedule ('conventional', 'sprint', etc.)
        event_row: Single row from FastF1 schedule DataFrame
    
    Returns:
        Dictionary mapping symbolic session names to schedule column names.
        Example: {'FP1': 'Session1DateUtc', 'Q': 'Session4DateUtc', ...}
    """
    symbolic_to_real: dict[str, str] = {
        "FP1": "Practice 1",
        "FP2": "Practice 2",
        "FP3": "Practice 3",
        "Q":   "Qualifying",
        "SQ":  "Sprint Qualifying",
        "SS":  "Sprint Shootout",
        "S":   "Sprint",
        "R":   "Race",
    }

    mapping: dict[str, str] = {}
    # Iterate through possible session columns
    for i in range(1, 6):  # Session1 to Session5
        label = event_row.get(f"Session{i}", "")
        for sym, real in symbolic_to_real.items():
            if label == real:
                mapping[sym] = f"Session{i}DateUtc"
    return mapping


def _official_schedule(year: int) -> pd.DataFrame:
    """
    Get official F1 schedule with multiple backend fallbacks.
    
    Attempts to load schedule using FastF1's backends in order:
    fastf1 → f1timing → ergast. Each backend may have different
    reliability and data freshness.

    Parameters:
        year: F1 championship year (e.g., 2024)

    Returns:
        DataFrame with F1 event schedule containing EventName, RoundNumber,
        EventFormat, Location, Session1DateUtc through Session5DateUtc, etc.
        
    Example:
        >>> sched = _official_schedule(2024)
        >>> print(sched[['EventName', 'EventFormat']].head())
    """
    try:
        sched = ff1.get_event_schedule(year, backend="fastf1")
        sched['Session1DateUtc'] = pd.to_datetime(sched['Session1DateUtc'], utc=True)
        return sched
    except Exception as e:
        print(f"⚠️ fastf1 backend failed: {e}")
        try:
            sched = ff1.get_event_schedule(year, backend="f1timing")
            sched['Session1DateUtc'] = pd.to_datetime(sched['Session1DateUtc'], utc=True)
            return sched
        except Exception as e:
            print(f"⚠️ f1timing backend failed: {e}")
            try:
                sched = ff1.get_event_schedule(year, backend="ergast")
                sched['Session1DateUtc'] = pd.to_datetime(sched['Session1DateUtc'], utc=True)
                return sched
            except Exception as e:
                print(f"❌ Failed to load event schedule for {year}: {e}")
                return None


def get_expected_sessions(year: int) -> Dict[str, List[str]]:
    """
    Determine which session names to expect for a given season.
    
    Maps each event to its expected session list based on event format.

    Parameters:
        year: F1 championship year

    Returns:
        Dictionary mapping 'YYYY_RR' key to list of session names
        
    Example:
        >>> sessions = get_expected_sessions(2024)
        >>> print(sessions['2024_01'])  # Bahrain
        ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race']
    """
    sched = _official_schedule(year)
    event_sessions: Dict[str, List[str]] = {}

    for _, row in sched.iterrows():
        rnd = int(row['RoundNumber'])
        if rnd == 0:
            # Skip non-championship events (testing)
            continue
        key = f"{year}_{rnd:02d}"
        fmt = row.get('EventFormat', '').lower()
        # Define valid sessions based on event format
        if fmt == 'sprint_shootout':
            valid = ["Practice 1", "Qualifying", "Sprint Shootout", "Sprint", "Race"]
        elif fmt == 'sprint_qualifying':
            valid = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]
        elif fmt == 'sprint':
            valid = ["Practice 1", "Qualifying", "Practice 2", "Sprint", "Race"]
        else:
            valid = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
        event_sessions[key] = valid
    return event_sessions


def _session_list(event_format: str) -> List[str]:
    """
    Map event format to list of session codes for data collection.
    
    Args:
        event_format: Event format string from schedule
    
    Returns:
        List of session codes in chronological order
        
    Example:
        >>> _session_list('conventional')
        ['FP1', 'FP2', 'FP3', 'Q', 'R']
    """
    fmt = (event_format or "").lower()
    if fmt == "testing":
        return []
    if fmt == "sprint_shootout":
        return ["FP1", "Q", "SS", "S", "R"]
    if fmt == "sprint_qualifying":
        return ["FP1", "SQ", "S", "Q", "R"]
    if fmt == "sprint":
        return ["FP1", "Q", "FP2", "S", "R"]
    # Default conventional format
    return ["FP1", "FP2", "FP3", "Q", "R"]


def _sessions_completed(format_type: str,
                        fp1_utc: datetime,
                        now: datetime) -> List[str]:
    """
    Determine which sessions have started based on FP1 time and current time.
    
    Uses typical session timing offsets relative to FP1 to determine
    which sessions have already begun.
    
    Args:
        format_type: Event format ('conventional', 'sprint', etc.)
        fp1_utc: FP1 start time in UTC
        now: Current time in UTC
    
    Returns:
        List of session codes that have started
        
    Example:
        >>> from datetime import datetime, timezone
        >>> fp1 = datetime(2024, 3, 22, 1, 30, tzinfo=timezone.utc)
        >>> now = datetime(2024, 3, 23, 6, 0, tzinfo=timezone.utc)
        >>> completed = _sessions_completed('conventional', fp1, now)
        >>> print(completed)
        ['FP1', 'FP2', 'FP3', 'Q']
    """
    # Normalize tz-aware to naive UTC
    if fp1_utc.tzinfo:
        fp1_utc = fp1_utc.astimezone(timezone.utc).replace(tzinfo=None)
    if now.tzinfo:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)

    # Offsets (hours) for each session relative to FP1
    mapping = {
        "conventional":      [('FP1',  0), ('FP2',  4), ('FP3', 24), ('Q', 28), ('R', 52)],
        "sprint_qualifying": [('FP1',  0), ('SQ',  4), ('S', 28),  ('Q', 28), ('R', 52)],
        "sprint_shootout":   [('FP1',  0), ('Q',   4), ('SS',28),  ('S', 28), ('R', 52)],
        "sprint":            [('FP1',  0), ('Q',   4), ('FP2',28),  ('S', 28), ('R', 52)],
    }
    key = format_type if format_type in mapping else "conventional"
    # Return labels whose scheduled time ≤ now
    return [label for label, offset in mapping[key] if (fp1_utc + timedelta(hours=offset)) <= now]


def _completed_sessions(schedule: pd.DataFrame,
                        now: datetime) -> List[tuple[int,str,str]]:
    """
    Generate list of all completed sessions across all events in schedule.
    
    Args:
        schedule: DataFrame from _official_schedule()
        now: Current datetime in UTC
    
    Returns:
        List of (year, event_name, session_code) tuples
        
    Example:
        >>> sched = _official_schedule(2024)
        >>> completed = _completed_sessions(sched, datetime.now(timezone.utc))
        >>> print(f"Total completed: {len(completed)}")
    """
    todo: List[tuple[int,str,str]] = []
    for _, ev in schedule.iterrows():
        fmt = str(ev.EventFormat).lower()
        name = ev.EventName or ""
        fp1_utc = ev.Session1DateUtc
        year_tag = fp1_utc.year
        
        # Skip testing events entirely
        if fmt == "testing" or "test" in name.lower():
            continue
            
        # Add each completed session
        for ses in _sessions_completed(fmt, fp1_utc, now):
            todo.append((year_tag, name, ses))
            
    return todo


# Generic loaders                   
def load_session(year: int, event_name: str, session_name: str) -> dict:
    """
    Load a session via FastF1 with fallback to OpenF1 API.
    
    Attempts FastF1 first (includes telemetry). If FastF1 fails, falls
    back to OpenF1 API (lap times only, no telemetry).

    Args:
        year: F1 season year
        event_name: Grand Prix name
        session_name: Session code ('FP1', 'Q', 'R', etc.)

    Returns:
        Dictionary with keys: source, session, laps, status, reason
        - source: 'fastf1', 'openf1', or None
        - session: FastF1.Session object or None
        - laps: pd.DataFrame or None
        - status: 'ok', 'fallback', or 'error'
        - reason: Error message if status='error'
        
    Example:
        >>> info = load_session(2024, 'Monaco Grand Prix', 'Q')
        >>> if info['status'] == 'ok':
        ...     session = info['session']
        ...     print(f"Loaded via {info['source']}")
    """
    # Map short codes to FastF1 full names
    session_map = {
        "FP1": "Practice 1",     
        "FP2": "Practice 2",
        "FP3": "Practice 3",     
        "Q":   "Qualifying",
        "R":   "Race",           
        "SQ":  "Sprint Qualifying",
        "S":   "Sprint",         
        "SS":  "Sprint Shootout"
    }
    ff_session_name = session_map.get(session_name, session_name)

    # Try FastF1 first
    try:
        session = ff1.get_session(year, event_name, ff_session_name)
        session.load(telemetry=True, laps=True)
        if session.laps.empty:
            raise ValueError("FastF1 session loaded but contains no lap data")
        return {
            "source": "fastf1",
            "session": session,
            "laps": session.laps,
            "status": "ok",
            "reason": None
        }
    except Exception as e:
        print(f"⚠️ FastF1 failed for {year} {event_name} {session_name}: {e}")

    # Fallback to OpenF1 API for lap times
    try:
        print("🔄 Falling back to OpenF1 API...")
        api_session_name = session_map.get(session_name, session_name)
        url = "https://api.openf1.org/v1/lap_times"
        params = {"year": year, "session": api_session_name}
        response = requests.get(url, params=params)
        response.raise_for_status()
        laps_df = pd.DataFrame(response.json())
        if laps_df.empty:
            raise ValueError("OpenF1 returned an empty dataset")
        return {
            "source": "openf1",
            "session": None,
            "laps": laps_df,
            "status": "fallback",
            "reason": None
        }
    except Exception as e:
        print(f"❌ OpenF1 fallback failed for {year} {event_name} {session_name}: {e}")
        return {
            "source": None,
            "session": None,
            "laps": None,
            "status": "error",
            "reason": str(e)
        }


def get_weather_info(session, year: int, event_name: str, session_name: str) -> dict:
    """
    Extract weather information from session with OpenF1 fallback.
    
    Attempts to get weather data from FastF1 session first. If unavailable,
    queries OpenF1 weather API endpoint.

    Args:
        session: FastF1.Session or None if loaded via OpenF1 fallback
        year: Season year
        event_name: Event name (e.g., 'Bahrain Grand Prix')
        session_name: Session label ('FP1', 'Race', etc.)

    Returns:
        Dictionary with average air temp, track temp, and rain boolean
        
    Example:
        >>> weather = get_weather_info(session, 2024, 'Singapore', 'Q')
        >>> print(f"Track temp: {weather['track_temp_avg']:.1f}°C")
    """
    # Use FastF1 session if available
    if session is not None and hasattr(session, "weather_data") and not session.weather_data.empty:
        weather = session.weather_data
        return {
            "air_temp_avg": weather["AirTemp"].mean(),
            "track_temp_avg": weather["TrackTemp"].mean(),
            "rain_detected": weather["Rainfall"].max() > 0
        }

    # Else fallback to OpenF1 weather endpoint
    try:
        url = "https://api.openf1.org/v1/weather"
        params = {"year": year, "session": session_name}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = pd.DataFrame(response.json())

        if data.empty:
            raise ValueError("No weather data from OpenF1")

        return {
            "air_temp_avg": data["air_temperature"].mean(),
            "track_temp_avg": data["track_temperature"].mean(),
            "rain_detected": (data["rainfall"] > 0).any()
        }

    except Exception as e:
        print(f"⚠️ Weather fallback failed for {year} {event_name} {session_name}: {e}")
        return {
            "air_temp_avg": np.nan,
            "track_temp_avg": np.nan,
            "rain_detected": np.nan
        }


@functools.lru_cache(maxsize=None)
def get_elevation(latitude: float, longitude: float, timeout: int = 10) -> float:
    """
    Query Open-Meteo API for ground elevation at given coordinates.
    
    Results cached via LRU cache since elevation doesn't change.

    Args:
        latitude: GPS latitude in decimal degrees
        longitude: GPS longitude in decimal degrees
        timeout: Request timeout in seconds (default: 10)

    Returns:
        Elevation in meters

    Raises:
        RuntimeError: If API returns no elevation
        
    Example:
        >>> elev = get_elevation(26.0325, 50.5106)  # Bahrain
        >>> print(f"Elevation: {elev}m")
    """
    url = f"https://api.open-meteo.com/v1/elevation?latitude={latitude}&longitude={longitude}"

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    payload = r.json()
    if "elevation" not in payload or payload["elevation"] is None:
        raise RuntimeError("Open-Meteo returned no elevation")

    return payload["elevation"][0]           # note: array, not dict


# =============================================================================
# PROFILE FILE MANAGEMENT
# =============================================================================

def is_update_needed(cache_path: str, season: int = datetime.now(timezone.utc).year) -> bool:
    """
    Decide whether the cache CSV needs to be refreshed.
    
    Checks if we're within a race weekend or near the next race start.

    Logic:
        1. If file doesn't exist → True
        2. If within ongoing race-weekend → True
        3. If within 6h before next FP1 → True
        4. Otherwise → False
        
    Args:
        cache_path: Path to cache CSV file
        season: F1 season year (default: current year)
    
    Returns:
        True if cache should be updated/rebuilt
        
    Example:
        >>> if is_update_needed('data/driver/2024_driver_profiles.csv', 2024):
        ...     print("Updating cache...")
    """
    # missing cache → definitely rebuild 
    if not os.path.exists(cache_path):
        return True

    try:
        sched = _official_schedule(season)
        now   = datetime.now(timezone.utc)

        # Check if within a race weekend (FP1 to Race + buffer)
        weekend_length = timedelta(days=4)
        ongoing = sched[
            (sched.Session1DateUtc <= now)
            & (sched.Session1DateUtc + weekend_length >= now)
        ]
        if not ongoing.empty:
            return True

        # Else, next race start -6h threshold
        upcoming = sched[sched.Session1DateUtc > now]
        if upcoming.empty:
            return False

        next_start = upcoming.iloc[0]["Session1DateUtc"]
        prebuffer  = timedelta(hours=6)
        
        return now >= next_start - prebuffer

    except Exception as e:
        print(f"⚠️  Could not check race schedule: {e}")
        return True  # safest fallback
        

def update_profiles_file(
    cache_path: str,
    start_year: int = None,
    end_year:   int = None,
    file_type:  str = "circuit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append sessions that (a) have already started and (b) are not yet cached.

    Supports 'circuit', 'driver', 'driver_timing'.

    For file_type=="circuit", builds each missing session in single-session mode.
    For file_type=="driver", fetches the FastF1 session then runs get_all_driver_features.
    For file_type=="driver_timing", fetches detailed timing for each driver.
    
    Args:
        cache_path: Path to existing cache CSV file
        start_year: First year to check for updates (default: current year)
        end_year: Last year to check (default: start_year)
        file_type: 'circuit', 'driver', or 'driver_timing'
    
    Returns:
        Tuple of (updated_df, skipped_df)
        
    Example:
        >>> updated, skipped = update_profiles_file('data/driver/2024_driver_profiles.csv',
        ...                                         2024, 2024, 'driver')
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")

    existing = pd.read_csv(path)
    existing_keys = {(r.year, r.event, r.session) for r in existing.itertuples()}

    now = datetime.now(timezone.utc)
    sy = start_year or now.year
    ey = end_year or sy

    new_chunks, skipped = [], []

    for year in range(sy, ey + 1):
        sched = _official_schedule(year)
        completed = sched[sched.Session1DateUtc < now]
        todo = _completed_sessions(completed, now)

        # COLLECT ALL MISSING SESSIONS FOR THIS YEAR
        missing_sessions = []
        for yr, ev_name, sess_label in todo:
            key = (yr, ev_name, sess_label)
            if key not in existing_keys:
                missing_sessions.append((ev_name, sess_label))
        
        if not missing_sessions:
            continue  # No new sessions for this year
        
        print(f"📥 Adding {len(missing_sessions)} missing session(s) for {year}...")
        for ev_name, sess_label in missing_sessions:
            print(f"   → {ev_name} {sess_label}")

        try:
            # BUILD ALL MISSING SESSIONS IN ONE CALL
            if file_type == "circuit":
                from .circuit_utils import _build_circuit_profile_df
                
                # Convert to only_specific format
                only_specific = {year: set(missing_sessions)}
                
                df_ok, df_fail = _build_circuit_profile_df(
                    start_year=year,
                    end_year=year,
                    only_specific=only_specific
                )

            elif file_type == "driver":
                from .driver_utils import _build_driver_profile_df
                
                # Convert to only_specific format
                only_specific = {year: set(missing_sessions)}
                
                df_ok, df_fail = _build_driver_profile_df(
                    start_year=year,
                    end_year=year,
                    only_specific=only_specific
                )
                
            elif file_type == "driver_timing":
                from .driver_utils import _build_detailed_telemetry

                out_dir = os.path.dirname(cache_path)
                os.makedirs(out_dir, exist_ok=True)

                existing_files = {
                    os.path.basename(p)
                    for p in glob.glob(os.path.join(out_dir, "*.parquet"))
                }

                chunks = []
                for ev_name, sess_label in missing_sessions:
                    fn = f"{year}_{ev_name.replace(' ', '_')}_{sess_label}.parquet"
                    if fn in existing_files:
                        continue

                    info = load_session(year, ev_name, sess_label)
                    if info.get("status") != "ok":
                        continue
                    sess_obj = info["session"]

                    df_tmp = _build_detailed_telemetry(sess_obj)
                    df_tmp["year"] = year
                    df_tmp["event"] = ev_name
                    df_tmp["session"] = sess_label

                    out_path = os.path.join(out_dir, fn)
                    df_tmp.to_parquet(out_path,
                                      engine="pyarrow",
                                      compression="snappy",
                                      index=False)
                    chunks.append(df_tmp)

                df_ok = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                df_fail = pd.DataFrame()

            else:
                raise ValueError(f"Unsupported file_type: {file_type!r}")

            # VALIDATION: Only add if data was extracted
            if not df_ok.empty:
                new_chunks.append(df_ok)
                print(f"   ✅ Extracted {len(df_ok)} row(s)")
            else:
                print(f"   ⚠️  No data available (likely DNS/no telemetry)")
                # Mark all as skipped
                for ev_name, sess_label in missing_sessions:
                    skipped.append({
                        "year": year,
                        "event": ev_name,
                        "session": sess_label,
                        "reason": "No telemetry data available"
                    })
            
            # Add failures from builder
            if df_fail is not None and not df_fail.empty:
                skipped.extend(df_fail.to_dict("records"))

        except Exception as e:
            print(f"⚠️ Failed to append sessions for {year}: {e}")
            for ev_name, sess_label in missing_sessions:
                skipped.append({
                    "year": year,
                    "event": ev_name,
                    "session": sess_label,
                    "reason": str(e)
                })

    # Save if we have new data
    if new_chunks:
        updated = pd.concat([existing, *new_chunks], ignore_index=True)
        if updated.empty or updated.shape[1] == 0:
            print(f"⚠️ Skipping save: empty or no columns [{path.name}]")
            return existing, pd.DataFrame(skipped)

        updated.to_csv(path, index=False)
        total = sum(len(df) for df in new_chunks)
        print(f"✅ Added {total} row(s).")
        return updated, pd.DataFrame(skipped)

    print("ℹ️  No new sessions to append.")
    return existing, pd.DataFrame(skipped)
    
    
def load_or_build_profiles(
    start_year: int,
    end_year: int,
    file_type: str = "circuit",
    gp_name: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load cached profiles if available, otherwise build from scratch or update.

    Args:
        start_year: First year to load/build
        end_year: Last year to load/build
        file_type: 'circuit', 'driver', or 'driver_timing'
        gp_name: If set and file_type=="circuit", only build for that GP

    Returns:
        Tuple of (df_profiles, df_skipped)
        
    Example:
        >>> profiles, skipped = load_or_build_profiles(2022, 2024, 'driver')
        >>> print(f"Total records: {len(profiles)}")
    """
    # driver_timing logic
    if file_type == "driver_timing":
        from .driver_utils import _build_driver_timing_profiles
        return _build_driver_timing_profiles(start_year, end_year)
       
    # circuit / driver logic
    end_year = end_year or start_year
    current_year = datetime.now(timezone.utc).year

    # Precompute only_specific mapping once, not inside the per-year loop
    only_specific: dict[int, set[tuple[str,str]]] | None = None
    if file_type == "circuit" and gp_name:
        from .circuit_utils import _build_circuit_profile_df
        only_specific = {}
        for yr in range(start_year, end_year + 1):
            sched = ff1.get_event_schedule(yr)
            row = sched[sched["EventName"] == gp_name]
            if row.empty:
                continue
            row = row.iloc[0]
            session_cols = [
                c for c in sched.columns
                if c.startswith("Session") and not c.endswith(("Date","DateUtc"))
            ]
            codes = [row[c] for c in session_cols if pd.notna(row[c])]
            only_specific[yr] = {(gp_name, code) for code in codes}

    all_data    = []
    all_skipped = []

    for year in range(start_year, end_year + 1):
        # driver_timing is already handled above, so we only get circuit/driver here
        cache_path = f"data/{file_type}/{year}_{file_type}_profiles.csv"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 1) If no cache → build from scratch
        if not os.path.exists(cache_path):
            print(f"📂 No cache for {year}. Rebuilding...")

            if file_type == "circuit" and only_specific:
                from .circuit_utils import _build_circuit_profile_df
                df, skipped = _build_circuit_profile_df(
                    year, year,
                    only_specific={year: only_specific.get(year, set())}
                )

            elif file_type == "circuit":
                from .circuit_utils import _build_circuit_profile_df
                df, skipped = _build_circuit_profile_df(year, year)

            elif file_type == "driver":
                from .driver_utils import _build_driver_profile_df
                df, skipped = _build_driver_profile_df(
                    start_year=year,
                    end_year=year
                )

            else:
                raise ValueError(f"Unsupported file_type: {file_type!r}")

            # only circuit & driver write CSV
            df.to_csv(cache_path, index=False)
            if not skipped.empty or skipped.shape[1] != 0:
                skip_dir = os.path.join("data", "skipped", file_type)
                os.makedirs(skip_dir, exist_ok=True)
                
                skip_path = os.path.join(skip_dir, f"{year}_{file_type}_skipped.csv")
                skipped.to_csv(skip_path, index=False)

        # 2) If it's the current year and needs updating
        elif year == current_year and is_update_needed(cache_path, season=year):
            print(f"🔁 Updating {file_type} profile for {year}...")
            df, skipped = update_profiles_file(cache_path, year, year, file_type)

        # 3) Otherwise just load the cached CSV
        else:
            print(f"✅ Using cached {file_type} profile for {year}")
            df = pd.read_csv(cache_path)
            from helpers.team_name_mapping import normalize_team_column
            df = normalize_team_column(df, col="team")
            skipped = pd.DataFrame()

        all_data.append(df)
        if not skipped.empty:
            all_skipped.append(skipped)

    df_all = pd.concat(all_data, ignore_index=True)    if all_data    else pd.DataFrame()
    skipped_all = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()

    return df_all, skipped_all


def ensure_year_dir(year: int, subdir: str = "data") -> str:
    """
    Create (if needed) and return a directory for a given year under subdir.

    Parameters:
        year: Year identifier
        subdir: Parent directory name (default: 'data')

    Returns:
        Full path to year-specific directory
        
    Example:
        >>> path = ensure_year_dir(2024)
        >>> print(path)
        data/2024
    """
    year_path = os.path.join(subdir, str(year))
    os.makedirs(year_path, exist_ok=True)
    return year_path


def load_classifications(
    start_year: int = 2022,
    end_year: int = 2025
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load qualifying and race classifications from SSOT files.
    
    Args:
        start_year: First year to load
        end_year: Last year to load
        
    Returns:
        Tuple of (qualifying_df, race_df)
    """
    quali_dfs = []
    race_dfs = []
    
    for year in range(start_year, end_year + 1):
        # Load qualifying
        quali_file = f'data/predictions/ssot/{year}_qualifying.csv'
        if os.path.exists(quali_file):
            df = pd.read_csv(quali_file)
            quali_dfs.append(df)
        
        # Load race
        race_file = f'data/predictions/ssot/{year}_race.csv'
        if os.path.exists(race_file):
            df = pd.read_csv(race_file)
            race_dfs.append(df)
    
    quali_df = pd.concat(quali_dfs, ignore_index=True) if quali_dfs else pd.DataFrame()
    race_df = pd.concat(race_dfs, ignore_index=True) if race_dfs else pd.DataFrame()
    
    from helpers.team_name_mapping import normalize_team_column
    quali_df = normalize_team_column(quali_df, col='team')
    race_df = normalize_team_column(race_df,col='team')

    return quali_df, race_df


def merge_driver_features_with_targets(
    driver_profiles: pd.DataFrame,
    start_year: int = 2022,
    end_year: int = 2025
) -> pd.DataFrame:
    """
    Merge driver features with classification targets and session dates.
    
    Args:
        driver_profiles: Raw driver session data
        start_year: First year to include
        end_year: Last year to include
        
    Returns:
        DataFrame with driver features + position targets + session dates + team
    """
    import os
    import pandas as pd
    
    print("🔗 Merging driver features with classification targets...")
    
    # Load classifications from SSOT files
    quali_dfs = []
    race_dfs = []
    
    for year in range(start_year, end_year + 1):
        quali_file = f'data/predictions/ssot/{year}_qualifying.csv'
        if os.path.exists(quali_file):
            quali_dfs.append(pd.read_csv(quali_file))
        race_file = f'data/predictions/ssot/{year}_race.csv'
        if os.path.exists(race_file):
            race_dfs.append(pd.read_csv(race_file))
    
    quali_df = pd.concat(quali_dfs, ignore_index=True) if quali_dfs else pd.DataFrame()
    race_df = pd.concat(race_dfs, ignore_index=True) if race_dfs else pd.DataFrame()
    
    if quali_df.empty and race_df.empty:
        print("   ⚠️  No classification data found!")
        return pd.DataFrame()
    
    from helpers.team_name_mapping import normalize_team_column
    quali_df = normalize_team_column(quali_df, col='team')
    race_df = normalize_team_column(race_df, col='team')

    print(f"   Loaded {len(quali_df) + len(race_df):,} classification records")
    
    # ========================================================================
    # CRITICAL: Get team from classification data
    # ========================================================================
    # Combine qualifying and race to get team information
    team_lookup = pd.DataFrame()
    
    if not quali_df.empty:
        team_lookup = quali_df[['year', 'event', 'driver', 'team']].drop_duplicates()
    
    if not race_df.empty:
        race_teams = race_df[['year', 'event', 'driver', 'team']].drop_duplicates()
        if team_lookup.empty:
            team_lookup = race_teams
        else:
            # Merge, preferring race data if conflict
            team_lookup = pd.concat([team_lookup, race_teams]).drop_duplicates(
                subset=['year', 'event', 'driver'],
                keep='last'
            )
    
    # Merge team info into driver profiles FIRST
    if not team_lookup.empty:
        driver_profiles = driver_profiles.merge(
            team_lookup,
            on=['year', 'event', 'driver'],
            how='left'
        )
        print(f"   Added team information")
    
    # Now merge positions
    if not quali_df.empty:
        driver_profiles = driver_profiles.merge(
            quali_df[['year', 'event', 'driver', 'qualifying_position']],
            on=['year', 'event', 'driver'],
            how='left',
            suffixes=('', '_quali')
        )
    
    if not race_df.empty:
        driver_profiles = driver_profiles.merge(
            race_df[['year', 'event', 'driver', 'race_position']],
            on=['year', 'event', 'driver'],
            how='left',
            suffixes=('', '_race')
        )
    
    # Add session dates
    if 'session_date' not in driver_profiles.columns:
        print("   📅 Adding session dates...")
        
        driver_profiles = driver_profiles.sort_values(['year', 'event'])
        
        driver_profiles['event_order'] = driver_profiles.groupby('year')['event'].transform(
            lambda x: pd.factorize(x, sort=True)[0]
        )
        
        driver_profiles['session_date'] = (
            pd.to_datetime(driver_profiles['year'].astype(str) + '-01-01') +
            pd.to_timedelta(driver_profiles['event_order'] * 14, unit='D')
        )
        
        # Prevents datetime64[ns] → object[Timestamp] corruption in merges/groupby
        driver_profiles['session_date'] = driver_profiles['session_date'].astype(str)
        
        driver_profiles = driver_profiles.drop(columns=['event_order'])

    # Summary
    print(f"   Merged dataset: {driver_profiles.shape}")
    
    with_positions = (
        driver_profiles['qualifying_position'].notna() | 
        driver_profiles['race_position'].notna()
    ).sum()
    
    print(f"   Sessions with positions: {with_positions:,}")
    
    if 'team' in driver_profiles.columns:
        teams_count = driver_profiles['team'].nunique()
        print(f"   Teams represented: {teams_count}")
    else:
        print(f"   Teams represented: N/A (column missing)")
    
    return driver_profiles