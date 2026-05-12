"""Read-only summarizers over FastF1 session DataFrames.

Each function takes one or more pandas DataFrames mirroring FastF1's
``session.laps``, ``session.results``, ``session.weather_data``, and
``session.track_status``, and returns a dataclass summary suitable for
the analyst to read directly or to serialise to JSON.

This module deliberately does not import ``fastf1``; the loader module
owns the dependency boundary so the summarizers are testable without
FastF1 installed.

Scope reminder (master execution plan, Phase 2):
- counting laps, retirements, weather samples, track status events,
  and qualifying-segment reach is in scope;
- pairing teammate laps, classifying laps as dry / wet / unreliable,
  emitting matched-pair or skipped-pair rows, or deciding what counts
  as a comparable lap is out of scope and belongs to Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass
class WeatherSummary:
    """Sample-level weather summary for one session.

    Reports raw rainfall sample counts only. Does not classify laps as
    dry, wet, or unreliable; that is Phase 3.
    """

    n_samples: int
    n_rainfall_true: int
    n_rainfall_false: int
    rainfall_fraction: float
    has_dry_samples: bool
    has_rain_samples: bool
    is_mixed: bool


@dataclass
class LapCountSummary:
    """Per-driver lap counts and team membership."""

    by_driver: dict[str, int]
    by_team: dict[str, list[str]]
    full_distance_drivers: list[str]
    partial_distance_drivers: list[str]
    max_observed_lap: int


@dataclass
class RetirementRecord:
    """One driver's retirement evidence."""

    driver_code: str
    team: str
    retirement_lap: int | None
    last_observed_lap: int | None
    classified_status: str
    is_early: bool


@dataclass
class RetirementSummary:
    """Drivers classified as not finishing."""

    retired_drivers: dict[str, RetirementRecord]


@dataclass
class TrackStatusEvent:
    """One track-status row."""

    status_code: str
    status_label: str
    start_time: pd.Timedelta | None


@dataclass
class TrackStatusSummary:
    """Summary of track-status events for one session."""

    events: list[TrackStatusEvent]
    n_safety_car_rows: int
    n_virtual_safety_car_rows: int
    n_red_flag_rows: int
    n_yellow_rows: int


@dataclass
class DriverQualiReach:
    """Which qualifying segments one driver reached."""

    driver_code: str
    team: str
    reached_q1: bool
    reached_q2: bool
    reached_q3: bool


@dataclass
class QualifyingSegmentSummary:
    """Per-driver qualifying-segment reach plus team-level coverage."""

    by_driver: dict[str, DriverQualiReach]
    teams_with_q3: list[str]
    teams_with_q1_eliminated: list[str]
    teams_with_split_segments: list[str]


@dataclass
class SessionInspection:
    """Combined inspection result for one session."""

    year: int
    event_name: str
    session_kind: str
    weather: WeatherSummary
    lap_counts: LapCountSummary
    retirements: RetirementSummary
    track_status: TrackStatusSummary
    qualifying: QualifyingSegmentSummary | None


# ---------------------------------------------------------------------------
# FastF1 numeric track-status codes
# ---------------------------------------------------------------------------

_TRACK_STATUS_LABELS: dict[str, str] = {
    "1": "green",
    "2": "yellow",
    "4": "sc",
    "5": "red",
    "6": "vsc_deployed",
    "7": "vsc_ending",
}


# ---------------------------------------------------------------------------
# Summarizers
# ---------------------------------------------------------------------------


def summarize_weather_samples(weather_df: pd.DataFrame) -> WeatherSummary:
    """Summarise rainfall samples in a session's weather data.

    Parameters
    ----------
    weather_df:
        DataFrame mirroring FastF1's ``session.weather_data``. Must
        contain a ``Rainfall`` column of boolean values.

    Returns
    -------
    WeatherSummary
        Counts and presence flags. Does not classify laps.

    Raises
    ------
    KeyError
        If the ``Rainfall`` column is missing.
    """
    if "Rainfall" not in weather_df.columns:
        raise KeyError("weather_df missing 'Rainfall' column")

    rainfall = weather_df["Rainfall"].astype(bool)
    n_total = int(len(rainfall))
    n_true = int(rainfall.sum())
    n_false = n_total - n_true
    fraction = float(n_true / n_total) if n_total else 0.0

    return WeatherSummary(
        n_samples=n_total,
        n_rainfall_true=n_true,
        n_rainfall_false=n_false,
        rainfall_fraction=fraction,
        has_dry_samples=n_false > 0,
        has_rain_samples=n_true > 0,
        is_mixed=n_true > 0 and n_false > 0,
    )


def summarize_lap_counts(
    laps_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> LapCountSummary:
    """Per-driver lap counts plus team-driver mapping.

    Parameters
    ----------
    laps_df:
        Mirrors ``session.laps``. Required columns: ``Driver``,
        ``LapNumber``.
    results_df:
        Mirrors ``session.results``. Required columns:
        ``Abbreviation``, ``TeamName``.

    Returns
    -------
    LapCountSummary
        Lap counts by driver, drivers grouped by team, and which
        drivers reached the session's max observed lap.

    Raises
    ------
    KeyError
        If a required column is missing from either DataFrame.
    """
    _require_columns(laps_df, {"Driver", "LapNumber"}, "laps_df")
    _require_columns(results_df, {"Abbreviation", "TeamName"}, "results_df")

    by_driver: dict[str, int] = laps_df.groupby("Driver")["LapNumber"].max().astype(int).to_dict()

    team_to_drivers: dict[str, list[str]] = {}
    for _, row in results_df.iterrows():
        team = str(row["TeamName"])
        driver = str(row["Abbreviation"])
        team_to_drivers.setdefault(team, []).append(driver)

    max_lap = max(by_driver.values(), default=0)
    full = [driver for driver, n in by_driver.items() if n == max_lap]
    partial = [driver for driver, n in by_driver.items() if n < max_lap]

    return LapCountSummary(
        by_driver=by_driver,
        by_team=team_to_drivers,
        full_distance_drivers=sorted(full),
        partial_distance_drivers=sorted(partial),
        max_observed_lap=int(max_lap),
    )


def summarize_retirements(
    results_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    *,
    early_threshold_lap: int = 10,
) -> RetirementSummary:
    """Drivers who did not finish, with last-observed-lap evidence.

    A driver is treated as retired when the FastF1 result ``Status`` is
    populated, is not ``"Finished"``, and does not contain ``"Lap"``
    (matching the project's existing DNF convention; classified lapped
    finishers are not retirements). Blank qualifying statuses are ignored.

    Parameters
    ----------
    results_df:
        Mirrors ``session.results``. Required columns:
        ``Abbreviation``, ``TeamName``, ``Status``.
    laps_df:
        Mirrors ``session.laps``. Required columns: ``Driver``,
        ``LapNumber``.
    early_threshold_lap:
        Drivers whose last observed lap is at or below this number are
        flagged ``is_early``. Default 10 matches the smoke-session
        early-DNF category.

    Returns
    -------
    RetirementSummary
        Map from driver code to RetirementRecord.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    _require_columns(results_df, {"Abbreviation", "TeamName", "Status"}, "results_df")
    _require_columns(laps_df, {"Driver", "LapNumber"}, "laps_df")

    last_lap_by_driver = laps_df.groupby("Driver")["LapNumber"].max().astype(int).to_dict()

    retirements: dict[str, RetirementRecord] = {}
    for _, row in results_df.iterrows():
        raw_status = row["Status"]
        if pd.isna(raw_status):
            continue
        status = str(raw_status).strip()
        if not status:
            continue
        if status == "Finished" or "Lap" in status:
            continue

        driver = str(row["Abbreviation"])
        last_lap = last_lap_by_driver.get(driver)
        is_early = last_lap is None or last_lap <= early_threshold_lap

        retirements[driver] = RetirementRecord(
            driver_code=driver,
            team=str(row["TeamName"]),
            retirement_lap=last_lap,
            last_observed_lap=last_lap,
            classified_status=status,
            is_early=bool(is_early),
        )

    return RetirementSummary(retired_drivers=retirements)


def summarize_track_status(track_status_df: pd.DataFrame) -> TrackStatusSummary:
    """Summarise track-status events for one session.

    Parameters
    ----------
    track_status_df:
        Mirrors ``session.track_status``. Required columns: ``Status``,
        ``Time``. ``Status`` is the raw FastF1 numeric code as string.

    Returns
    -------
    TrackStatusSummary
        One TrackStatusEvent per row plus counts of safety-car, VSC,
        red-flag, and yellow occurrences. Codes outside the known set
        are emitted with label ``"unknown_<code>"`` so the analyst can
        catch surprises.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    _require_columns(track_status_df, {"Status", "Time"}, "track_status_df")

    events: list[TrackStatusEvent] = []
    n_sc = n_vsc = n_red = n_yellow = 0

    for time_value, raw_status in zip(
        track_status_df["Time"], track_status_df["Status"], strict=True
    ):
        code = str(raw_status)
        label = _TRACK_STATUS_LABELS.get(code, f"unknown_{code}")
        events.append(
            TrackStatusEvent(
                status_code=code,
                status_label=label,
                start_time=time_value if pd.notna(time_value) else None,
            )
        )
        if label == "sc":
            n_sc += 1
        elif label.startswith("vsc"):
            n_vsc += 1
        elif label == "red":
            n_red += 1
        elif label == "yellow":
            n_yellow += 1

    return TrackStatusSummary(
        events=events,
        n_safety_car_rows=n_sc,
        n_virtual_safety_car_rows=n_vsc,
        n_red_flag_rows=n_red,
        n_yellow_rows=n_yellow,
    )


def summarize_qualifying_segments(
    laps_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> QualifyingSegmentSummary:
    """Per-driver Q1 / Q2 / Q3 reach and team-level coverage.

    Reads ``Q1``, ``Q2``, ``Q3`` time fields from ``results_df`` (FastF1's
    canonical encoding for qualifying sessions). A non-null time means
    the driver reached and set a time in that segment.

    The team-level metrics support Phase 2's "representative
    qualifying" smoke session, which needs at least one team in Q3 and
    at least one team eliminated in Q1.

    Parameters
    ----------
    laps_df:
        Mirrors ``session.laps``. Required columns: ``Driver``.
        Currently unused by the segment logic, kept for symmetry with
        the other summarizers.
    results_df:
        Mirrors ``session.results``. Required columns:
        ``Abbreviation``, ``TeamName``, ``Q1``, ``Q2``, ``Q3``.

    Returns
    -------
    QualifyingSegmentSummary
        Per-driver reach and three team-level lists: teams that
        reached Q3, teams eliminated in Q1, and teams whose two
        drivers split between Q3 and Q1.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    _require_columns(laps_df, {"Driver"}, "laps_df")
    _require_columns(
        results_df,
        {"Abbreviation", "TeamName", "Q1", "Q2", "Q3"},
        "results_df",
    )

    by_driver: dict[str, DriverQualiReach] = {}
    team_segments_reached: dict[str, set[str]] = {}

    for _, row in results_df.iterrows():
        driver = str(row["Abbreviation"])
        team = str(row["TeamName"])

        reached = {
            "Q1": bool(pd.notna(row["Q1"])),
            "Q2": bool(pd.notna(row["Q2"])),
            "Q3": bool(pd.notna(row["Q3"])),
        }
        by_driver[driver] = DriverQualiReach(
            driver_code=driver,
            team=team,
            reached_q1=reached["Q1"],
            reached_q2=reached["Q2"],
            reached_q3=reached["Q3"],
        )
        for segment, was_reached in reached.items():
            if was_reached:
                team_segments_reached.setdefault(team, set()).add(segment)

    teams_with_q3 = sorted(t for t, segs in team_segments_reached.items() if "Q3" in segs)
    teams_with_q1_eliminated = sorted(
        t for t, segs in team_segments_reached.items() if segs == {"Q1"}
    )

    teams_with_split: list[str] = []
    drivers_by_team: dict[str, list[str]] = {}
    for code, reach in by_driver.items():
        drivers_by_team.setdefault(reach.team, []).append(code)
    for team, codes in drivers_by_team.items():
        if len(codes) != 2:
            continue
        a, b = codes
        a_q3, b_q3 = by_driver[a].reached_q3, by_driver[b].reached_q3
        a_q2, b_q2 = by_driver[a].reached_q2, by_driver[b].reached_q2
        if (a_q3 and not b_q2) or (b_q3 and not a_q2):
            teams_with_split.append(team)

    return QualifyingSegmentSummary(
        by_driver=by_driver,
        teams_with_q3=teams_with_q3,
        teams_with_q1_eliminated=teams_with_q1_eliminated,
        teams_with_split_segments=sorted(teams_with_split),
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def inspect_session(
    *,
    year: int,
    event_name: str,
    session_kind: str,
    laps_df: pd.DataFrame,
    results_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    track_status_df: pd.DataFrame,
) -> SessionInspection:
    """Produce a full inspection summary from already-loaded FastF1 frames.

    Takes DataFrames rather than a FastF1 ``Session`` so the function is
    testable without FastF1 installed. The ``loader`` module produces a
    Session and the CLI extracts the four DataFrames before calling
    this.

    Parameters
    ----------
    year, event_name, session_kind:
        Identifiers recorded on the output for traceability.
    laps_df, results_df, weather_df, track_status_df:
        FastF1 session frames, with the column requirements documented
        on each summarizer.

    Returns
    -------
    SessionInspection
        Combined summary. ``qualifying`` is populated only when
        ``session_kind == "qualifying"``.
    """
    weather = summarize_weather_samples(weather_df)
    lap_counts = summarize_lap_counts(laps_df, results_df)
    retirements = summarize_retirements(results_df, laps_df)
    track_status = summarize_track_status(track_status_df)

    qualifying: QualifyingSegmentSummary | None = None
    if session_kind == "qualifying":
        qualifying = summarize_qualifying_segments(laps_df, results_df)

    return SessionInspection(
        year=year,
        event_name=event_name,
        session_kind=session_kind,
        weather=weather,
        lap_counts=lap_counts,
        retirements=retirements,
        track_status=track_status,
        qualifying=qualifying,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    """Raise KeyError if ``df`` is missing any of the required columns."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{df_name} missing columns: {missing}")
