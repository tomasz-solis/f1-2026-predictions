"""Canonical teammate matched-lap extraction.

The extractor returns one row per paired teammate lap. It does not emit
mirrored driver rows, because those would double-count the same evidence in
the teammate-network prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

SessionKind = Literal["race", "qualifying"]
WeatherMode = Literal["dry", "wet", "mixed", "unknown"]

ROW_TYPE_MATCHED_PAIR = "matched_pair"
ROW_TYPE_SKIPPED_PAIR = "skipped_pair"

SKIP_SINGLE_CAR_SESSION = "single_car_session"
SKIP_TEAM_DRIVER_SET_AMBIGUOUS = "team_driver_set_ambiguous"
SKIP_TEAMMATE_DNF_NO_MATCHED_LAPS = "teammate_dnf_no_matched_laps"
SKIP_WEATHER_ROUTING_EXCLUDES_SESSION = "weather_routing_excludes_session"
SKIP_LAP_LEVEL_WEATHER_UNRELIABLE = "lap_level_weather_unreliable"
SKIP_INSUFFICIENT_MATCHED_PAIRS = "insufficient_matched_pairs"
SKIP_NO_COMPOUND_OVERLAP = "no_compound_overlap"
SKIP_NO_COMMON_QUALI_SEGMENT = "no_common_quali_segment"
SKIP_ALL_LAPS_FILTERED_OUT = "all_laps_filtered_out"
SKIP_MISSING_LAP_TIME_DATA = "missing_lap_time_data"
SKIP_TRACK_STATUS_EXCLUDED_ALL_LAPS = "track_status_excluded_all_laps"

CANONICAL_SKIP_REASONS: tuple[str, ...] = (
    SKIP_SINGLE_CAR_SESSION,
    SKIP_TEAM_DRIVER_SET_AMBIGUOUS,
    SKIP_TEAMMATE_DNF_NO_MATCHED_LAPS,
    SKIP_WEATHER_ROUTING_EXCLUDES_SESSION,
    SKIP_LAP_LEVEL_WEATHER_UNRELIABLE,
    SKIP_INSUFFICIENT_MATCHED_PAIRS,
    SKIP_NO_COMPOUND_OVERLAP,
    SKIP_NO_COMMON_QUALI_SEGMENT,
    SKIP_ALL_LAPS_FILTERED_OUT,
    SKIP_MISSING_LAP_TIME_DATA,
    SKIP_TRACK_STATUS_EXCLUDED_ALL_LAPS,
)

WEATHER_DRY = "dry"
WEATHER_WET = "wet"
WEATHER_UNRELIABLE = "lap_level_mixed_unreliable"

TRACK_GREEN = "green"
TRACK_NON_GREEN = "non_green"
TRACK_UNKNOWN = "unknown"

OUTPUT_COLUMNS: tuple[str, ...] = (
    "row_type",
    "year",
    "race_name",
    "session_name",
    "session_kind",
    "team",
    "reference_driver_code",
    "comparison_driver_code",
    "reference_lap_number",
    "comparison_lap_number",
    "reference_lap_time_s",
    "comparison_lap_time_s",
    "matched_gap_s",
    "compound",
    "reference_stint",
    "comparison_stint",
    "stint_lap_index",
    "weather_bucket",
    "track_status_bucket",
    "reference_position_start",
    "reference_position_end",
    "comparison_position_start",
    "comparison_position_end",
    "skip_reason",
)

AGGREGATE_COLUMNS: tuple[str, ...] = (
    "reference_driver_code",
    "comparison_driver_code",
    "team",
    "year",
    "race_name",
    "session_name",
    "session_kind",
    "matched_gap_median_s",
    "matched_gap_se_s",
    "n_matched_pairs",
    "weather_bucket",
    "skip_reason",
)

FILTER_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "year",
    "race_name",
    "session_name",
    "session_kind",
    "team",
    "reference_driver_code",
    "comparison_driver_code",
    "raw_laps",
    "reference_raw_laps",
    "comparison_raw_laps",
    "missing_lap_time_laps",
    "pit_laps",
    "deleted_laps",
    "inaccurate_laps",
    "lap1_laps",
    "final_driver_laps",
    "non_green_laps",
    "sc_vsc_laps",
    "large_position_change_laps",
    "stint_outlier_laps",
    "lap_level_weather_unreliable_laps",
    "weather_mode_excluded_laps",
    "non_quick_qualifying_laps",
    "valid_laps",
    "candidate_matched_pairs",
    "matched_pair_rows",
    "skip_reason",
)


@dataclass(frozen=True)
class MatchedLapConfig:
    """Thresholds for canonical teammate matched-lap extraction."""

    min_matched_pairs_race: int = 8
    min_matched_pairs_quali: int = 3
    max_position_change_for_clean_lap: int = 2
    traffic_stint_sigma_threshold: float = 1.5
    tire_age_fallback_window_laps: int = 3
    early_teammate_dnf_lap_threshold: int = 10
    bootstrap_samples: int = 1000
    bootstrap_random_seed: int = 2026
    matched_gap_se_floor_s: float = 0.02
    qualifying_quicklap_threshold: float = 1.07


@dataclass(frozen=True)
class _SessionMeta:
    """Display metadata copied onto every output row."""

    year: int | None
    race_name: str | None
    session_name: str | None


def extract_matched_teammate_laps(
    session: Any,
    *,
    session_kind: SessionKind,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Return canonical teammate matched-lap observations for one session.

    Each valid row represents one teammate lap pair. The sign convention is
    ``matched_gap_s = comparison_lap_time_s - reference_lap_time_s``; positive
    values mean the alphabetically ordered reference driver was faster.
    """
    if session_kind not in {"race", "qualifying"}:
        raise ValueError(f"Unsupported session_kind: {session_kind!r}")
    if weather_mode not in {"dry", "wet", "mixed", "unknown"}:
        raise ValueError(f"Unsupported weather_mode: {weather_mode!r}")

    laps = _session_laps(session)
    if laps.empty:
        return _empty_output()

    _require_columns(laps, {"Driver", "Team"}, "session.laps")

    meta = _session_meta(session, session_kind)
    teams = _team_driver_map(laps, _session_results(session))
    rows: list[dict[str, Any]] = []

    for team, drivers in sorted(teams.items()):
        if len(drivers) == 1:
            rows.append(_skipped_row(meta, session_kind, team, drivers, SKIP_SINGLE_CAR_SESSION))
            continue
        if len(drivers) != 2:
            rows.append(
                _skipped_row(meta, session_kind, team, drivers, SKIP_TEAM_DRIVER_SET_AMBIGUOUS)
            )
            continue

        reference_driver, comparison_driver = sorted(drivers)
        pair_laps = laps[laps["Driver"].isin([reference_driver, comparison_driver])].copy()
        if session_kind == "race":
            pair_rows, skip_reason = _race_pair_rows(
                pair_laps,
                meta=meta,
                session_kind=session_kind,
                team=team,
                reference_driver=reference_driver,
                comparison_driver=comparison_driver,
                weather_data=_session_weather(session),
                weather_mode=weather_mode,
                config=config,
            )
        else:
            pair_rows, skip_reason = _qualifying_pair_rows(
                pair_laps,
                meta=meta,
                session_kind=session_kind,
                team=team,
                reference_driver=reference_driver,
                comparison_driver=comparison_driver,
                weather_data=_session_weather(session),
                session_status=_session_status(session),
                weather_mode=weather_mode,
                config=config,
            )

        if pair_rows:
            rows.extend(pair_rows)
        else:
            rows.append(
                _skipped_row(
                    meta,
                    session_kind,
                    team,
                    [reference_driver, comparison_driver],
                    skip_reason or SKIP_ALL_LAPS_FILTERED_OUT,
                )
            )

    return _ordered_output(rows)


def aggregate_matched_teammate_laps(
    matched_laps: pd.DataFrame,
    *,
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Aggregate raw matched pairs into one row per teammate pair.

    The median preserves the extractor's sign convention. The bootstrap
    standard error is estimated from matched-pair gaps and floored so tiny
    samples do not look more certain than they are.
    """
    if matched_laps.empty:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)

    _require_columns(matched_laps, set(OUTPUT_COLUMNS), "matched_laps")
    matched = matched_laps[matched_laps["row_type"] == ROW_TYPE_MATCHED_PAIR].copy()
    skipped = matched_laps[matched_laps["row_type"] == ROW_TYPE_SKIPPED_PAIR].copy()

    aggregate_rows: list[dict[str, Any]] = []
    group_columns = [
        "reference_driver_code",
        "comparison_driver_code",
        "team",
        "year",
        "race_name",
        "session_name",
        "session_kind",
        "weather_bucket",
    ]

    for group_key, group in matched.groupby(group_columns, dropna=False):
        key_data = dict(zip(group_columns, group_key, strict=True))
        gaps = group["matched_gap_s"].astype(float).to_numpy()
        min_pairs = (
            config.min_matched_pairs_quali
            if key_data["session_kind"] == "qualifying"
            else config.min_matched_pairs_race
        )
        if len(gaps) < min_pairs:
            aggregate_rows.append(
                {
                    **key_data,
                    "matched_gap_median_s": pd.NA,
                    "matched_gap_se_s": pd.NA,
                    "n_matched_pairs": int(len(gaps)),
                    "skip_reason": SKIP_INSUFFICIENT_MATCHED_PAIRS,
                }
            )
            continue
        aggregate_rows.append(
            {
                **key_data,
                "matched_gap_median_s": float(np.median(gaps)),
                "matched_gap_se_s": _bootstrap_median_se(gaps, config=config),
                "n_matched_pairs": int(len(gaps)),
                "skip_reason": pd.NA,
            }
        )

    for _, row in skipped.iterrows():
        aggregate_rows.append(
            {
                "reference_driver_code": row["reference_driver_code"],
                "comparison_driver_code": row["comparison_driver_code"],
                "team": row["team"],
                "year": row["year"],
                "race_name": row["race_name"],
                "session_name": row["session_name"],
                "session_kind": row["session_kind"],
                "matched_gap_median_s": pd.NA,
                "matched_gap_se_s": pd.NA,
                "n_matched_pairs": 0,
                "weather_bucket": row["weather_bucket"],
                "skip_reason": row["skip_reason"],
            }
        )

    return pd.DataFrame(aggregate_rows, columns=AGGREGATE_COLUMNS)


def diagnose_matched_lap_filters(
    session: Any,
    *,
    session_kind: SessionKind,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Return per-team filter diagnostics using the extractor's own rules.

    These rows are for smoke validation and review. They explain routine
    filtering, such as SC/VSC, pit laps, weather routing, and stint outliers,
    without turning those routine removals into pair-level skip reasons.
    """
    if session_kind not in {"race", "qualifying"}:
        raise ValueError(f"Unsupported session_kind: {session_kind!r}")
    if weather_mode not in {"dry", "wet", "mixed", "unknown"}:
        raise ValueError(f"Unsupported weather_mode: {weather_mode!r}")

    laps = _session_laps(session)
    if laps.empty:
        return pd.DataFrame(columns=FILTER_DIAGNOSTIC_COLUMNS)

    _require_columns(laps, {"Driver", "Team"}, "session.laps")
    meta = _session_meta(session, session_kind)
    teams = _team_driver_map(laps, _session_results(session))
    output = extract_matched_teammate_laps(
        session,
        session_kind=session_kind,
        weather_mode=weather_mode,
        config=config,
    )

    rows: list[dict[str, Any]] = []
    for team, drivers in sorted(teams.items()):
        pair_laps = laps[laps["Team"].eq(team)].copy()
        if len(drivers) != 2:
            rows.append(
                _filter_diagnostic_row(
                    meta,
                    session_kind,
                    team,
                    drivers,
                    pair_laps,
                    prepared=None,
                    valid=pd.DataFrame(),
                    candidate_matched_pairs=0,
                    output=output,
                    weather_mode=weather_mode,
                    config=config,
                )
            )
            continue

        reference_driver, comparison_driver = sorted(drivers)
        pair_laps = laps[laps["Driver"].isin([reference_driver, comparison_driver])].copy()
        prepared = _prepare_laps(
            pair_laps,
            weather_data=_session_weather(session),
            session_status=(
                _session_status(session) if session_kind == "qualifying" else pd.DataFrame()
            ),
        )
        if session_kind == "qualifying":
            prepared["quali_segment"] = _qualifying_segments(prepared, _session_status(session))
            valid = _valid_qualifying_laps(
                prepared,
                weather_mode=weather_mode,
                config=config,
            )
            candidate_matched_pairs = _qualifying_candidate_match_count(
                valid,
                reference_driver,
                comparison_driver,
                config,
            )
        else:
            valid = _valid_race_laps(prepared, weather_mode=weather_mode, config=config)
            candidate_matched_pairs = len(
                _pair_race_laps(valid, reference_driver, comparison_driver)
            )

        rows.append(
            _filter_diagnostic_row(
                meta,
                session_kind,
                team,
                [reference_driver, comparison_driver],
                pair_laps,
                prepared=prepared,
                valid=valid,
                candidate_matched_pairs=candidate_matched_pairs,
                output=output,
                weather_mode=weather_mode,
                config=config,
            )
        )

    return pd.DataFrame(rows, columns=FILTER_DIAGNOSTIC_COLUMNS)


def probe_qualifying_pair_constructs(
    session: Any,
    *,
    reference_driver: str,
    comparison_driver: str,
    team: str | None = None,
    weather_mode: WeatherMode,
    target_weather_bucket: Literal["dry", "wet"],
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Summarize qualifying construct variants for one ordered teammate pair.

    Returned deltas follow the extractor sign convention: positive means the
    requested ``reference_driver`` was faster than ``comparison_driver``.
    ``team`` narrows the probe to one stored teammate context when supplied.
    """
    if weather_mode not in {"dry", "wet", "mixed", "unknown"}:
        raise ValueError(f"Unsupported weather_mode: {weather_mode!r}")
    if target_weather_bucket not in {WEATHER_DRY, WEATHER_WET}:
        raise ValueError(f"Unsupported target weather bucket: {target_weather_bucket!r}")

    laps = _session_laps(session)
    pair_laps = laps[laps["Driver"].isin([reference_driver, comparison_driver])].copy()
    if team is not None and "Team" in pair_laps.columns:
        pair_laps = pair_laps[pair_laps["Team"].eq(team)].copy()
    if pair_laps.empty or set(pair_laps["Driver"]) != {reference_driver, comparison_driver}:
        return _empty_qualifying_construct_probe(
            pair_present=False,
            target_weather_bucket=target_weather_bucket,
        )

    prepared = _prepare_laps(
        pair_laps,
        weather_data=_session_weather(session),
        session_status=_session_status(session),
    )
    prepared["quali_segment"] = _qualifying_segments(prepared, _session_status(session))
    valid = _valid_qualifying_laps(prepared, weather_mode=weather_mode, config=config)
    valid_target = valid[valid["weather_bucket"].eq(target_weather_bucket)].copy()

    common_segments = _common_quali_segments(prepared, reference_driver, comparison_driver)
    valid_common_segments = _common_quali_segments(
        valid_target,
        reference_driver,
        comparison_driver,
    )
    matches, chosen_segments = _selected_qualifying_matches(
        valid,
        common_segments=common_segments,
        reference_driver=reference_driver,
        comparison_driver=comparison_driver,
        config=config,
    )
    target_matches = _matches_for_weather_bucket(matches, target_weather_bucket)
    current_delta = _median_match_delta(
        target_matches,
        min_pairs=config.min_matched_pairs_quali,
    )
    highest_common_segment = _highest_common_qualifying_segment(valid_common_segments)

    return {
        "pair_present": True,
        "target_weather_bucket": target_weather_bucket,
        "common_segments": sorted(common_segments),
        "valid_common_segments": sorted(valid_common_segments),
        "chosen_segments": chosen_segments,
        "current_construct_n_pairs": int(len(target_matches)),
        "current_construct_delta_s": current_delta,
        "highest_common_segment": highest_common_segment,
        "highest_common_best_delta_s": _best_lap_delta(
            valid_target[valid_target["quali_segment"].eq(highest_common_segment)],
            reference_driver=reference_driver,
            comparison_driver=comparison_driver,
        ),
        "any_valid_best_delta_s": _best_lap_delta(
            valid_target,
            reference_driver=reference_driver,
            comparison_driver=comparison_driver,
        ),
    }


def probe_race_pair_constructs(
    session: Any,
    *,
    reference_driver: str,
    comparison_driver: str,
    team: str | None = None,
    weather_mode: WeatherMode,
    target_weather_bucket: Literal["dry", "wet"],
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Summarize race construct variants for one ordered teammate pair.

    Returned deltas follow the extractor sign convention: positive means the
    requested ``reference_driver`` was faster than ``comparison_driver``.
    ``team`` narrows the probe to one stored teammate context when supplied.
    """
    if weather_mode not in {"dry", "wet", "mixed", "unknown"}:
        raise ValueError(f"Unsupported weather_mode: {weather_mode!r}")
    if target_weather_bucket not in {WEATHER_DRY, WEATHER_WET}:
        raise ValueError(f"Unsupported target weather bucket: {target_weather_bucket!r}")

    laps = _session_laps(session)
    pair_laps = laps[laps["Driver"].isin([reference_driver, comparison_driver])].copy()
    if team is not None and "Team" in pair_laps.columns:
        pair_laps = pair_laps[pair_laps["Team"].eq(team)].copy()
    if pair_laps.empty or set(pair_laps["Driver"]) != {reference_driver, comparison_driver}:
        return _empty_race_construct_probe(
            pair_present=False,
            target_weather_bucket=target_weather_bucket,
        )

    prepared = _prepare_laps(
        pair_laps,
        weather_data=_session_weather(session),
        session_status=pd.DataFrame(),
    )
    valid = _valid_race_laps(prepared, weather_mode=weather_mode, config=config)
    valid_target = valid[valid["weather_bucket"].eq(target_weather_bucket)].copy()
    matches = _pair_race_laps(valid, reference_driver, comparison_driver)
    target_matches = _matches_for_weather_bucket(matches, target_weather_bucket)

    return {
        "pair_present": True,
        "target_weather_bucket": target_weather_bucket,
        "current_construct_n_pairs": int(len(target_matches)),
        "current_construct_delta_s": _median_match_delta(
            target_matches,
            min_pairs=config.min_matched_pairs_race,
        ),
        "valid_reference_laps": int(valid_target["Driver"].eq(reference_driver).sum()),
        "valid_comparison_laps": int(valid_target["Driver"].eq(comparison_driver).sum()),
        "broad_valid_median_delta_s": _driver_lap_center_delta(
            valid_target,
            reference_driver=reference_driver,
            comparison_driver=comparison_driver,
            reducer="median",
        ),
        "broad_valid_mean_delta_s": _driver_lap_center_delta(
            valid_target,
            reference_driver=reference_driver,
            comparison_driver=comparison_driver,
            reducer="mean",
        ),
    }


def _session_laps(session: Any) -> pd.DataFrame:
    """Return a copy of session laps or an empty frame when absent."""
    laps = getattr(session, "laps", None)
    if laps is None:
        return pd.DataFrame()
    return pd.DataFrame(laps).copy()


def _session_weather(session: Any) -> pd.DataFrame:
    """Return a copy of FastF1 weather data or an empty frame."""
    weather = getattr(session, "weather_data", None)
    if weather is None:
        return pd.DataFrame()
    return pd.DataFrame(weather).copy()


def _session_results(session: Any) -> pd.DataFrame:
    """Return a copy of session results or an empty frame."""
    results = getattr(session, "results", None)
    if results is None:
        return pd.DataFrame()
    return pd.DataFrame(results).copy()


def _session_status(session: Any) -> pd.DataFrame:
    """Return a copy of qualifying session-status data or an empty frame."""
    status = getattr(session, "session_status", None)
    if status is None:
        return pd.DataFrame()
    return pd.DataFrame(status).copy()


def _session_meta(session: Any, session_kind: SessionKind) -> _SessionMeta:
    """Extract stable event metadata from a FastF1-like session."""
    event = getattr(session, "event", None)
    year = _first_present(
        getattr(session, "year", None),
        _event_value(event, "Year"),
        _year_from_date(_event_value(event, "EventDate")),
        _year_from_date(_event_value(event, "Session1Date")),
    )
    race_name = _first_present(
        getattr(session, "race_name", None),
        getattr(session, "event_name", None),
        _event_value(event, "EventName"),
        _event_value(event, "OfficialEventName"),
    )
    session_name = _first_present(
        getattr(session, "session_name", None),
        getattr(session, "name", None),
        "Race" if session_kind == "race" else "Qualifying",
    )
    return _SessionMeta(
        year=int(year) if pd.notna(year) else None,
        race_name=str(race_name) if pd.notna(race_name) else None,
        session_name=str(session_name) if pd.notna(session_name) else None,
    )


def _event_value(event: Any, key: str) -> Any:
    """Read one value from a dict-like or Series-like FastF1 event object."""
    if event is None:
        return None
    if isinstance(event, dict):
        return event.get(key)
    if hasattr(event, "get"):
        try:
            return event.get(key)
        except (KeyError, TypeError):
            return None
    return getattr(event, key, None)


def _year_from_date(value: Any) -> int | None:
    """Return a calendar year from a timestamp-like value."""
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return int(timestamp.year)


def _first_present(*values: Any) -> Any:
    """Return the first non-empty value in order."""
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None


def _team_driver_map(laps: pd.DataFrame, results: pd.DataFrame) -> dict[str, list[str]]:
    """Build a team-to-driver-code map from results first, then laps."""
    mapping: dict[str, set[str]] = {}
    if {"Abbreviation", "TeamName"}.issubset(results.columns):
        for _, row in results.iterrows():
            team = row["TeamName"]
            driver = row["Abbreviation"]
            if pd.notna(team) and pd.notna(driver):
                mapping.setdefault(str(team), set()).add(str(driver))

    for _, row in laps[["Team", "Driver"]].dropna().iterrows():
        mapping.setdefault(str(row["Team"]), set()).add(str(row["Driver"]))

    return {team: sorted(drivers) for team, drivers in mapping.items()}


def _race_pair_rows(
    pair_laps: pd.DataFrame,
    *,
    meta: _SessionMeta,
    session_kind: SessionKind,
    team: str,
    reference_driver: str,
    comparison_driver: str,
    weather_data: pd.DataFrame,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build race matched-pair rows for one two-driver team."""
    prepared = _prepare_laps(pair_laps, weather_data=weather_data, session_status=pd.DataFrame())
    raw_reason = _raw_data_skip_reason(prepared, [reference_driver, comparison_driver])
    if raw_reason:
        return [], raw_reason

    valid = _valid_race_laps(prepared, weather_mode=weather_mode, config=config)
    if valid.empty:
        return [], _empty_valid_lap_reason(
            prepared,
            reference_driver=reference_driver,
            comparison_driver=comparison_driver,
            weather_mode=weather_mode,
            config=config,
        )

    matches = _pair_race_laps(valid, reference_driver, comparison_driver)
    if matches.empty:
        if _has_early_no_comparison_dnf(
            prepared,
            [reference_driver, comparison_driver],
            config,
        ):
            return [], SKIP_TEAMMATE_DNF_NO_MATCHED_LAPS
        return [], _no_match_reason(valid, prepared, weather_mode)

    if len(matches) < config.min_matched_pairs_race:
        return [], SKIP_INSUFFICIENT_MATCHED_PAIRS

    return [
        _matched_output_row(
            meta,
            session_kind,
            team,
            reference_driver,
            comparison_driver,
            reference=row["reference"],
            comparison=row["comparison"],
        )
        for _, row in matches.iterrows()
    ], None


def _qualifying_pair_rows(
    pair_laps: pd.DataFrame,
    *,
    meta: _SessionMeta,
    session_kind: SessionKind,
    team: str,
    reference_driver: str,
    comparison_driver: str,
    weather_data: pd.DataFrame,
    session_status: pd.DataFrame,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> tuple[list[dict[str, Any]], str | None]:
    """Build qualifying matched-pair rows for one two-driver team."""
    prepared = _prepare_laps(
        pair_laps,
        weather_data=weather_data,
        session_status=session_status,
    )
    prepared["quali_segment"] = _qualifying_segments(prepared, session_status)
    raw_reason = _raw_data_skip_reason(prepared, [reference_driver, comparison_driver])
    if raw_reason:
        return [], raw_reason

    common_segments = _common_quali_segments(prepared, reference_driver, comparison_driver)
    if not common_segments:
        return [], SKIP_NO_COMMON_QUALI_SEGMENT

    valid = _valid_qualifying_laps(prepared, weather_mode=weather_mode, config=config)
    if valid.empty:
        return [], _empty_valid_lap_reason(
            prepared,
            reference_driver=reference_driver,
            comparison_driver=comparison_driver,
            weather_mode=weather_mode,
            config=config,
        )

    matches, _ = _selected_qualifying_matches(
        valid,
        common_segments=common_segments,
        reference_driver=reference_driver,
        comparison_driver=comparison_driver,
        config=config,
    )
    if matches.empty:
        return [], _no_match_reason(valid, prepared, weather_mode)

    if len(matches) < config.min_matched_pairs_quali:
        return [], SKIP_INSUFFICIENT_MATCHED_PAIRS

    return [
        _matched_output_row(
            meta,
            session_kind,
            team,
            reference_driver,
            comparison_driver,
            reference=row["reference"],
            comparison=row["comparison"],
        )
        for _, row in matches.iterrows()
    ], None


def _prepare_laps(
    laps: pd.DataFrame,
    *,
    weather_data: pd.DataFrame,
    session_status: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived fields needed by race and qualifying matching."""
    _require_columns(laps, {"Driver", "LapTime", "LapNumber", "Compound"}, "session.laps")
    prepared = laps.copy()
    prepared["lap_time_s"] = _lap_time_seconds(prepared["LapTime"])
    prepared["stint_lap_index"] = _stint_lap_index(prepared)
    prepared["position_end"] = _position_end(prepared)
    prepared["position_start"] = _position_start(prepared)
    prepared["track_status_bucket"] = prepared.apply(_track_status_bucket, axis=1)
    prepared["weather_bucket"] = prepared.apply(
        lambda row: _weather_bucket_for_lap(row, weather_data),
        axis=1,
    )
    prepared["is_final_driver_lap"] = _final_driver_lap_mask(prepared)
    prepared["is_lap_time_missing"] = prepared["lap_time_s"].isna()
    prepared["is_pit_lap"] = _pit_lap_mask(prepared)
    prepared["is_deleted_lap"] = _deleted_lap_mask(prepared)
    prepared["is_accurate_lap"] = _accurate_lap_mask(prepared)
    prepared["position_change"] = (
        prepared["position_end"].astype("float64") - prepared["position_start"].astype("float64")
    ).abs()
    return prepared


def _lap_time_seconds(values: pd.Series) -> pd.Series:
    """Convert FastF1 Timedelta lap times to seconds."""
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def _stint_lap_index(laps: pd.DataFrame) -> pd.Series:
    """Return a one-based lap index within each driver stint."""
    stint = laps["Stint"] if "Stint" in laps.columns else pd.Series(1, index=laps.index)
    temp = laps[["Driver", "LapNumber"]].copy()
    temp["Stint"] = stint
    temp["_original_index"] = temp.index
    temp = temp.sort_values(["Driver", "Stint", "LapNumber", "_original_index"])
    temp["stint_lap_index"] = temp.groupby(["Driver", "Stint"], dropna=False).cumcount() + 1
    return temp.set_index("_original_index").loc[laps.index, "stint_lap_index"]


def _position_end(laps: pd.DataFrame) -> pd.Series:
    """Read or derive the position at lap end."""
    for column in ("PositionEnd", "position_end", "Position"):
        if column in laps.columns:
            return pd.to_numeric(laps[column], errors="coerce")
    return pd.Series(pd.NA, index=laps.index, dtype="Float64")


def _position_start(laps: pd.DataFrame) -> pd.Series:
    """Read or derive the position at lap start."""
    for column in ("PositionStart", "position_start"):
        if column in laps.columns:
            return pd.to_numeric(laps[column], errors="coerce")
    if "Position" not in laps.columns:
        return pd.Series(pd.NA, index=laps.index, dtype="Float64")

    temp = laps[["Driver", "LapNumber", "Position"]].copy()
    temp["_original_index"] = temp.index
    temp = temp.sort_values(["Driver", "LapNumber", "_original_index"])
    temp["position_start"] = temp.groupby("Driver", dropna=False)["Position"].shift(1)
    return pd.to_numeric(
        temp.set_index("_original_index").loc[laps.index, "position_start"],
        errors="coerce",
    )


def _track_status_bucket(row: pd.Series) -> str:
    """Classify a lap's track status as green, non-green, or unknown."""
    status = row.get("TrackStatus", pd.NA)
    if pd.isna(status):
        return TRACK_UNKNOWN
    status_text = str(status)
    if status_text and set(status_text) <= {"1"}:
        return TRACK_GREEN
    return TRACK_NON_GREEN


def _weather_bucket_for_lap(row: pd.Series, weather_data: pd.DataFrame) -> str:
    """Classify one lap as dry, wet, or unreliable from interval samples."""
    if weather_data.empty or not {"Time", "Rainfall"}.issubset(weather_data.columns):
        return WEATHER_UNRELIABLE

    start = row.get("LapStartTime", pd.NaT)
    end = row.get("Time", pd.NaT)
    if pd.isna(start) and pd.notna(end) and pd.notna(row.get("LapTime", pd.NaT)):
        start = end - row["LapTime"]
    if pd.isna(start) or pd.isna(end) or end <= start:
        return WEATHER_UNRELIABLE

    samples = weather_data[(weather_data["Time"] >= start) & (weather_data["Time"] <= end)]
    if samples.empty:
        return WEATHER_UNRELIABLE

    rainfall = samples["Rainfall"].dropna().astype(bool)
    if rainfall.empty:
        return WEATHER_UNRELIABLE
    if bool((~rainfall).all()):
        return WEATHER_DRY
    if bool(rainfall.all()):
        return WEATHER_WET
    return WEATHER_UNRELIABLE


def _final_driver_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Return True for each driver's last observed lap."""
    max_laps = laps.groupby("Driver", dropna=False)["LapNumber"].transform("max")
    return laps["LapNumber"] == max_laps


def _pit_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Return True when a lap is a pit-in or pit-out lap."""
    pit_in = (
        laps["PitInTime"].notna()
        if "PitInTime" in laps.columns
        else pd.Series(False, index=laps.index)
    )
    pit_out = (
        laps["PitOutTime"].notna()
        if "PitOutTime" in laps.columns
        else pd.Series(False, index=laps.index)
    )
    return pit_in | pit_out


def _deleted_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Return True for deleted laps where FastF1 exposes deletion data."""
    if "Deleted" not in laps.columns:
        return pd.Series(False, index=laps.index)
    deleted = laps["Deleted"]
    return deleted.notna() & (deleted.astype(str).str.lower() != "false")


def _accurate_lap_mask(laps: pd.DataFrame) -> pd.Series:
    """Return True for accurate laps, defaulting to True if the field is absent."""
    if "IsAccurate" not in laps.columns:
        return pd.Series(True, index=laps.index)
    return laps["IsAccurate"].fillna(False).astype(bool)


def _raw_data_skip_reason(prepared: pd.DataFrame, drivers: list[str]) -> str | None:
    """Return an early skip reason for missing raw pair data."""
    if "LapTime" not in prepared.columns:
        return SKIP_MISSING_LAP_TIME_DATA
    for driver in drivers:
        driver_laps = prepared[prepared["Driver"] == driver]
        if driver_laps.empty or driver_laps["LapTime"].isna().all():
            return SKIP_MISSING_LAP_TIME_DATA
    return None


def _valid_race_laps(
    prepared: pd.DataFrame,
    *,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Filter race laps to those eligible for paired teammate evidence."""
    allowed_weather = _allowed_weather_buckets(weather_mode)
    position_change_ok = prepared["position_change"].isna() | (
        prepared["position_change"] <= config.max_position_change_for_clean_lap
    )
    mask = (
        prepared["lap_time_s"].notna()
        & prepared["Compound"].notna()
        & (prepared["LapNumber"] != 1)
        & ~prepared["is_final_driver_lap"]
        & ~prepared["is_pit_lap"]
        & ~prepared["is_deleted_lap"]
        & prepared["is_accurate_lap"]
        & prepared["track_status_bucket"].isin([TRACK_GREEN, TRACK_UNKNOWN])
        & position_change_ok
        & ~_stint_outlier_mask_with_config(prepared, config)
        & prepared["weather_bucket"].isin(allowed_weather)
    )
    return prepared[mask].copy()


def _valid_qualifying_laps(
    prepared: pd.DataFrame,
    *,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Filter qualifying laps to comparable push laps in reliable weather."""
    base_mask = _qualifying_base_lap_mask(prepared, weather_mode=weather_mode)
    quicklap_mask = _qualifying_quicklap_mask(prepared, base_mask=base_mask, config=config)
    return prepared[base_mask & quicklap_mask].copy()


def _qualifying_base_lap_mask(
    prepared: pd.DataFrame,
    *,
    weather_mode: WeatherMode,
) -> pd.Series:
    """Return qualifying laps that clear non-pace data-quality filters."""
    allowed_weather = _allowed_weather_buckets(weather_mode)
    return (
        prepared["lap_time_s"].notna()
        & prepared["Compound"].notna()
        & prepared["quali_segment"].notna()
        & ~prepared["is_pit_lap"]
        & ~prepared["is_deleted_lap"]
        & prepared["is_accurate_lap"]
        & prepared["track_status_bucket"].isin([TRACK_GREEN, TRACK_UNKNOWN])
        & prepared["weather_bucket"].isin(allowed_weather)
    )


def _qualifying_quicklap_mask(
    prepared: pd.DataFrame,
    *,
    base_mask: pd.Series,
    config: MatchedLapConfig,
) -> pd.Series:
    """Return qualifying laps inside the configured quick-lap threshold."""
    if config.qualifying_quicklap_threshold <= 1.0:
        raise ValueError("qualifying_quicklap_threshold must be greater than 1.0")

    mask = pd.Series(False, index=prepared.index)
    base = prepared[base_mask].copy()
    if base.empty:
        return mask

    for _, group in base.groupby(["quali_segment", "weather_bucket"], dropna=False):
        best_lap_s = pd.to_numeric(group["lap_time_s"], errors="coerce").min()
        if pd.isna(best_lap_s):
            continue
        threshold_s = float(best_lap_s) * config.qualifying_quicklap_threshold
        mask.loc[group.index] = pd.to_numeric(group["lap_time_s"], errors="coerce") < threshold_s
    return mask


def _non_quick_qualifying_lap_count(
    prepared: pd.DataFrame,
    *,
    session_kind: SessionKind,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> int:
    """Count qualifying laps filtered only by the quick-lap threshold."""
    if session_kind != "qualifying" or "quali_segment" not in prepared.columns:
        return 0
    base_mask = _qualifying_base_lap_mask(prepared, weather_mode=weather_mode)
    quicklap_mask = _qualifying_quicklap_mask(prepared, base_mask=base_mask, config=config)
    return int((base_mask & ~quicklap_mask).sum())


def _stint_outlier_mask_with_config(laps: pd.DataFrame, config: MatchedLapConfig) -> pd.Series:
    """Flag laps slower than their stint median by the configured sigma margin."""
    if "Stint" not in laps.columns:
        group_keys = ["Driver", "Compound"]
    else:
        group_keys = ["Driver", "Stint", "Compound"]

    mask = pd.Series(False, index=laps.index)
    base_filter = (
        laps["lap_time_s"].notna()
        & ~laps["is_pit_lap"]
        & ~laps["is_deleted_lap"]
        & laps["is_accurate_lap"]
    )
    for _, group in laps[base_filter].groupby(group_keys, dropna=False):
        clean_times = group["lap_time_s"].dropna()
        if len(clean_times) < 3:
            continue
        median = float(clean_times.median())
        sigma = float(clean_times.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        threshold = median + (config.traffic_stint_sigma_threshold * sigma)
        mask.loc[group.index] = group["lap_time_s"] > threshold
    return mask


def _allowed_weather_buckets(weather_mode: WeatherMode) -> set[str]:
    """Return lap-level weather buckets allowed for a session mode."""
    if weather_mode == "dry":
        return {WEATHER_DRY}
    if weather_mode == "wet":
        return {WEATHER_WET}
    return {WEATHER_DRY, WEATHER_WET}


def _empty_valid_lap_reason(
    prepared: pd.DataFrame,
    *,
    reference_driver: str,
    comparison_driver: str,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> str:
    """Explain why no valid laps remain after filtering."""
    if _has_early_no_comparison_dnf(prepared, [reference_driver, comparison_driver], config):
        return SKIP_TEAMMATE_DNF_NO_MATCHED_LAPS
    if prepared["track_status_bucket"].eq(TRACK_NON_GREEN).all():
        return SKIP_TRACK_STATUS_EXCLUDED_ALL_LAPS
    if prepared["weather_bucket"].eq(WEATHER_UNRELIABLE).all():
        return SKIP_LAP_LEVEL_WEATHER_UNRELIABLE
    reliable_weather = prepared["weather_bucket"].isin({WEATHER_DRY, WEATHER_WET})
    if (
        reliable_weather.any()
        and not prepared["weather_bucket"].isin(_allowed_weather_buckets(weather_mode)).any()
    ):
        return SKIP_WEATHER_ROUTING_EXCLUDES_SESSION
    return SKIP_ALL_LAPS_FILTERED_OUT


def _has_early_no_comparison_dnf(
    prepared: pd.DataFrame,
    drivers: list[str],
    config: MatchedLapConfig,
) -> bool:
    """Return True when a driver's early final lap leaves no usable comparison."""
    for driver in drivers:
        driver_laps = prepared[prepared["Driver"] == driver]
        if driver_laps.empty:
            return True
        max_lap = pd.to_numeric(driver_laps["LapNumber"], errors="coerce").max()
        if pd.notna(max_lap) and max_lap <= config.early_teammate_dnf_lap_threshold:
            remaining = driver_laps[
                (driver_laps["LapNumber"] != 1)
                & ~driver_laps["is_final_driver_lap"]
                & driver_laps["lap_time_s"].notna()
            ]
            if remaining.empty:
                return True
    return False


def _pair_race_laps(
    valid: pd.DataFrame,
    reference_driver: str,
    comparison_driver: str,
) -> pd.DataFrame:
    """Pair race laps by compound, weather bucket, stint-lap index, and order."""
    keyed = valid.copy()
    key_columns = ["Compound", "weather_bucket", "stint_lap_index"]
    keyed = keyed.sort_values(["Driver", *key_columns, "LapNumber"])
    keyed["match_order"] = keyed.groupby(["Driver", *key_columns], dropna=False).cumcount() + 1
    return _pair_by_keys(
        keyed,
        reference_driver,
        comparison_driver,
        [*key_columns, "match_order"],
    )


def _pair_qualifying_laps(
    valid: pd.DataFrame,
    reference_driver: str,
    comparison_driver: str,
) -> pd.DataFrame:
    """Pair qualifying push laps by run order within segment and compound."""
    keyed = valid.copy()
    key_columns = ["quali_segment", "Compound", "weather_bucket"]
    keyed = keyed.sort_values(["Driver", *key_columns, "Time", "LapNumber"])
    keyed["match_order"] = keyed.groupby(["Driver", *key_columns], dropna=False).cumcount() + 1
    return _pair_by_keys(
        keyed,
        reference_driver,
        comparison_driver,
        [*key_columns, "match_order"],
    )


def _selected_qualifying_matches(
    valid: pd.DataFrame,
    *,
    common_segments: set[str],
    reference_driver: str,
    comparison_driver: str,
    config: MatchedLapConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the exact qualifying matches selected by the extractor."""
    matched_frames: list[pd.DataFrame] = []
    chosen_segments: list[str] = []
    for segment in ("Q3", "Q2", "Q1"):
        if segment not in common_segments:
            continue
        segment_matches = _pair_qualifying_laps(
            valid[valid["quali_segment"].eq(segment)],
            reference_driver,
            comparison_driver,
        )
        if not segment_matches.empty:
            matched_frames.append(segment_matches)
            chosen_segments.append(segment)
        if sum(len(frame) for frame in matched_frames) >= config.min_matched_pairs_quali:
            break

    if not matched_frames:
        return pd.DataFrame(columns=["reference", "comparison"]), []
    return pd.concat(matched_frames, ignore_index=True), chosen_segments


def _matches_for_weather_bucket(matches: pd.DataFrame, weather_bucket: str) -> pd.DataFrame:
    """Return matched laps whose reference lap belongs to one weather bucket."""
    if matches.empty:
        return matches.copy()
    mask = matches["reference"].map(lambda row: row["weather_bucket"]).eq(weather_bucket)
    return matches[mask].copy()


def _median_match_delta(matches: pd.DataFrame, *, min_pairs: int) -> float | None:
    """Return the median comparison-minus-reference gap for enough matches."""
    if len(matches) < min_pairs:
        return None
    gaps = [
        float(row["comparison"]["lap_time_s"]) - float(row["reference"]["lap_time_s"])
        for _, row in matches.iterrows()
    ]
    return float(np.median(gaps))


def _highest_common_qualifying_segment(common_segments: set[str]) -> str | None:
    """Return the highest qualifying segment shared by both drivers."""
    return next((segment for segment in ("Q3", "Q2", "Q1") if segment in common_segments), None)


def _best_lap_delta(
    valid: pd.DataFrame,
    *,
    reference_driver: str,
    comparison_driver: str,
) -> float | None:
    """Return best-comparison minus best-reference lap time for valid rows."""
    if valid.empty:
        return None
    reference_best = valid.loc[valid["Driver"].eq(reference_driver), "lap_time_s"].min()
    comparison_best = valid.loc[valid["Driver"].eq(comparison_driver), "lap_time_s"].min()
    if pd.isna(reference_best) or pd.isna(comparison_best):
        return None
    return float(comparison_best - reference_best)


def _driver_lap_center_delta(
    valid: pd.DataFrame,
    *,
    reference_driver: str,
    comparison_driver: str,
    reducer: Literal["mean", "median"],
) -> float | None:
    """Return comparison-minus-reference lap-center delta for valid rows."""
    if valid.empty:
        return None
    grouped = valid.groupby("Driver", dropna=False)["lap_time_s"]
    if reducer == "mean":
        centers = grouped.mean()
    elif reducer == "median":
        centers = grouped.median()
    else:
        raise ValueError(f"Unsupported reducer: {reducer!r}")
    reference_center = centers.get(reference_driver)
    comparison_center = centers.get(comparison_driver)
    if pd.isna(reference_center) or pd.isna(comparison_center):
        return None
    return float(comparison_center - reference_center)


def _empty_qualifying_construct_probe(
    *,
    pair_present: bool,
    target_weather_bucket: str,
) -> dict[str, Any]:
    """Return an empty qualifying construct-probe payload."""
    return {
        "pair_present": pair_present,
        "target_weather_bucket": target_weather_bucket,
        "common_segments": [],
        "valid_common_segments": [],
        "chosen_segments": [],
        "current_construct_n_pairs": 0,
        "current_construct_delta_s": None,
        "highest_common_segment": None,
        "highest_common_best_delta_s": None,
        "any_valid_best_delta_s": None,
    }


def _empty_race_construct_probe(
    *,
    pair_present: bool,
    target_weather_bucket: str,
) -> dict[str, Any]:
    """Return an empty race construct-probe payload."""
    return {
        "pair_present": pair_present,
        "target_weather_bucket": target_weather_bucket,
        "current_construct_n_pairs": 0,
        "current_construct_delta_s": None,
        "valid_reference_laps": 0,
        "valid_comparison_laps": 0,
        "broad_valid_median_delta_s": None,
        "broad_valid_mean_delta_s": None,
    }


def _qualifying_candidate_match_count(
    valid: pd.DataFrame,
    reference_driver: str,
    comparison_driver: str,
    config: MatchedLapConfig,
) -> int:
    """Count qualifying matches the extractor would consider before gating."""
    common_segments = _common_quali_segments(valid, reference_driver, comparison_driver)
    matches, _ = _selected_qualifying_matches(
        valid,
        common_segments=common_segments,
        reference_driver=reference_driver,
        comparison_driver=comparison_driver,
        config=config,
    )
    return int(len(matches))


def _pair_by_keys(
    keyed: pd.DataFrame,
    reference_driver: str,
    comparison_driver: str,
    key_columns: list[str],
) -> pd.DataFrame:
    """Return rows containing paired reference and comparison lap records."""
    reference = keyed[keyed["Driver"] == reference_driver].copy()
    comparison = keyed[keyed["Driver"] == comparison_driver].copy()
    if reference.empty or comparison.empty:
        return pd.DataFrame(columns=["reference", "comparison"])

    merged = reference.merge(
        comparison,
        on=key_columns,
        suffixes=("_reference", "_comparison"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["reference", "comparison"])

    rows: list[dict[str, pd.Series]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "reference": _merged_side(row, "reference"),
                "comparison": _merged_side(row, "comparison"),
            }
        )
    return pd.DataFrame(rows)


def _merged_side(row: pd.Series, suffix: str) -> pd.Series:
    """Recover one side of a merged lap pair as an unsuffixed Series."""
    data: dict[str, Any] = {}
    suffix_text = f"_{suffix}"
    for column, value in row.items():
        if column.endswith(suffix_text):
            data[column[: -len(suffix_text)]] = value
        elif not column.endswith("_reference") and not column.endswith("_comparison"):
            data[column] = value
    return pd.Series(data)


def _no_match_reason(valid: pd.DataFrame, prepared: pd.DataFrame, weather_mode: WeatherMode) -> str:
    """Explain why valid laps could not be paired."""
    if valid.empty:
        return _empty_valid_lap_reason(
            prepared,
            reference_driver=str(prepared["Driver"].iloc[0]) if not prepared.empty else "",
            comparison_driver=str(prepared["Driver"].iloc[-1]) if not prepared.empty else "",
            weather_mode=weather_mode,
            config=MatchedLapConfig(),
        )

    compounds_by_driver = valid.groupby("Driver")["Compound"].apply(set)
    if len(compounds_by_driver) == 2:
        sets = list(compounds_by_driver)
        if sets[0].isdisjoint(sets[1]):
            return SKIP_NO_COMPOUND_OVERLAP
    return SKIP_INSUFFICIENT_MATCHED_PAIRS


def _qualifying_segments(laps: pd.DataFrame, session_status: pd.DataFrame) -> pd.Series:
    """Return Q1/Q2/Q3 labels from lap columns or session-status windows."""
    for column in ("Segment", "QualifyingSegment", "SessionPart", "quali_segment"):
        if column in laps.columns:
            return laps[column].map(_normalize_quali_segment)

    windows = _qualifying_windows(session_status)
    if not windows:
        return pd.Series(pd.NA, index=laps.index, dtype="object")

    labels = []
    for _, row in laps.iterrows():
        labels.append(
            _segment_for_lap_interval(
                row.get("LapStartTime", pd.NaT),
                row.get("Time", pd.NaT),
                windows,
            )
        )
    return pd.Series(labels, index=laps.index, dtype="object")


def _normalize_quali_segment(value: Any) -> object:
    """Normalize segment labels to Q1, Q2, or Q3."""
    if pd.isna(value):
        return pd.NA
    text = str(value).upper()
    if text in {"1", "Q1"}:
        return "Q1"
    if text in {"2", "Q2"}:
        return "Q2"
    if text in {"3", "Q3"}:
        return "Q3"
    return pd.NA


def _qualifying_windows(session_status: pd.DataFrame) -> list[tuple[str, Any, Any]]:
    """Infer Q1/Q2/Q3 windows from FastF1 session-status Start/Finish rows."""
    if session_status.empty or not {"Time", "Status"}.issubset(session_status.columns):
        return []

    windows: list[tuple[str, Any, Any]] = []
    starts: list[Any] = []
    for _, row in session_status.sort_values("Time").iterrows():
        status = str(row["Status"]).lower()
        if status == "started":
            starts.append(row["Time"])
        elif status == "finished" and starts:
            segment = f"Q{len(windows) + 1}"
            windows.append((segment, starts.pop(0), row["Time"]))
            if len(windows) == 3:
                break
    return windows


def _segment_for_lap_interval(
    lap_start: Any, lap_end: Any, windows: list[tuple[str, Any, Any]]
) -> object:
    """Return the qualifying segment whose window contains the lap start."""
    if pd.isna(lap_start) and pd.isna(lap_end):
        return pd.NA
    for segment, start, end in windows:
        if pd.notna(lap_start) and start <= lap_start <= end:
            return segment
        if pd.notna(lap_end) and start <= lap_end <= end:
            return segment
    return pd.NA


def _common_quali_segments(
    prepared: pd.DataFrame,
    reference_driver: str,
    comparison_driver: str,
) -> set[str]:
    """Return Q segments reached by both drivers."""
    segments_by_driver = {
        driver: set(prepared.loc[prepared["Driver"] == driver, "quali_segment"].dropna())
        for driver in (reference_driver, comparison_driver)
    }
    return segments_by_driver[reference_driver] & segments_by_driver[comparison_driver]


def _filter_diagnostic_row(
    meta: _SessionMeta,
    session_kind: SessionKind,
    team: str,
    drivers: list[str],
    raw_laps: pd.DataFrame,
    *,
    prepared: pd.DataFrame | None,
    valid: pd.DataFrame,
    candidate_matched_pairs: int,
    output: pd.DataFrame,
    weather_mode: WeatherMode,
    config: MatchedLapConfig,
) -> dict[str, Any]:
    """Build one per-team smoke diagnostic row."""
    sorted_drivers = sorted(drivers)
    reference_driver = sorted_drivers[0] if sorted_drivers else pd.NA
    comparison_driver = sorted_drivers[1] if len(sorted_drivers) > 1 else pd.NA
    team_output = output[output["team"].eq(team)] if "team" in output.columns else pd.DataFrame()
    skipped = team_output[team_output["row_type"].eq(ROW_TYPE_SKIPPED_PAIR)]
    skip_reason = skipped["skip_reason"].iloc[0] if not skipped.empty else pd.NA

    if prepared is None or prepared.empty:
        return {
            "year": meta.year,
            "race_name": meta.race_name,
            "session_name": meta.session_name,
            "session_kind": session_kind,
            "team": team,
            "reference_driver_code": reference_driver,
            "comparison_driver_code": comparison_driver,
            "raw_laps": int(len(raw_laps)),
            "reference_raw_laps": _driver_raw_lap_count(raw_laps, reference_driver),
            "comparison_raw_laps": _driver_raw_lap_count(raw_laps, comparison_driver),
            "missing_lap_time_laps": int(
                raw_laps.get("LapTime", pd.Series(dtype=object)).isna().sum()
            ),
            "pit_laps": 0,
            "deleted_laps": 0,
            "inaccurate_laps": 0,
            "lap1_laps": 0,
            "final_driver_laps": 0,
            "non_green_laps": 0,
            "sc_vsc_laps": 0,
            "large_position_change_laps": 0,
            "stint_outlier_laps": 0,
            "lap_level_weather_unreliable_laps": 0,
            "weather_mode_excluded_laps": 0,
            "non_quick_qualifying_laps": 0,
            "valid_laps": 0,
            "candidate_matched_pairs": 0,
            "matched_pair_rows": (
                int(team_output["row_type"].eq(ROW_TYPE_MATCHED_PAIR).sum())
                if not team_output.empty
                else 0
            ),
            "skip_reason": skip_reason,
        }

    allowed_weather = _allowed_weather_buckets(weather_mode)
    position_change_limit = config.max_position_change_for_clean_lap
    large_position_change = prepared["position_change"].notna() & (
        prepared["position_change"] > position_change_limit
    )
    weather_mode_excluded = prepared["weather_bucket"].isin({WEATHER_DRY, WEATHER_WET}) & ~prepared[
        "weather_bucket"
    ].isin(allowed_weather)

    return {
        "year": meta.year,
        "race_name": meta.race_name,
        "session_name": meta.session_name,
        "session_kind": session_kind,
        "team": team,
        "reference_driver_code": reference_driver,
        "comparison_driver_code": comparison_driver,
        "raw_laps": int(len(prepared)),
        "reference_raw_laps": _driver_raw_lap_count(prepared, reference_driver),
        "comparison_raw_laps": _driver_raw_lap_count(prepared, comparison_driver),
        "missing_lap_time_laps": int(prepared["lap_time_s"].isna().sum()),
        "pit_laps": int(prepared["is_pit_lap"].sum()),
        "deleted_laps": int(prepared["is_deleted_lap"].sum()),
        "inaccurate_laps": int((~prepared["is_accurate_lap"]).sum()),
        "lap1_laps": int((prepared["LapNumber"] == 1).sum()),
        "final_driver_laps": int(prepared["is_final_driver_lap"].sum()),
        "non_green_laps": int(prepared["track_status_bucket"].eq(TRACK_NON_GREEN).sum()),
        "sc_vsc_laps": int(prepared.apply(_is_sc_or_vsc_lap, axis=1).sum()),
        "large_position_change_laps": int(large_position_change.sum()),
        "stint_outlier_laps": int(_stint_outlier_mask_with_config(prepared, config).sum()),
        "lap_level_weather_unreliable_laps": int(
            prepared["weather_bucket"].eq(WEATHER_UNRELIABLE).sum()
        ),
        "weather_mode_excluded_laps": int(weather_mode_excluded.sum()),
        "non_quick_qualifying_laps": _non_quick_qualifying_lap_count(
            prepared,
            session_kind=session_kind,
            weather_mode=weather_mode,
            config=config,
        ),
        "valid_laps": int(len(valid)),
        "candidate_matched_pairs": int(candidate_matched_pairs),
        "matched_pair_rows": (
            int(team_output["row_type"].eq(ROW_TYPE_MATCHED_PAIR).sum())
            if not team_output.empty
            else 0
        ),
        "skip_reason": skip_reason,
    }


def _driver_raw_lap_count(laps: pd.DataFrame, driver: Any) -> int:
    """Count raw laps for a driver if a concrete driver code is available."""
    if pd.isna(driver) or "Driver" not in laps.columns:
        return 0
    return int(laps["Driver"].eq(driver).sum())


def _is_sc_or_vsc_lap(row: pd.Series) -> bool:
    """Return True when FastF1 lap status includes SC or VSC codes."""
    status = row.get("TrackStatus", pd.NA)
    if pd.isna(status):
        return False
    return any(code in str(status) for code in ("4", "6", "7"))


def _matched_output_row(
    meta: _SessionMeta,
    session_kind: SessionKind,
    team: str,
    reference_driver: str,
    comparison_driver: str,
    *,
    reference: pd.Series,
    comparison: pd.Series,
) -> dict[str, Any]:
    """Build one output row from a paired reference/comparison lap."""
    reference_lap_time_s = float(reference["lap_time_s"])
    comparison_lap_time_s = float(comparison["lap_time_s"])
    return {
        "row_type": ROW_TYPE_MATCHED_PAIR,
        "year": meta.year,
        "race_name": meta.race_name,
        "session_name": meta.session_name,
        "session_kind": session_kind,
        "team": team,
        "reference_driver_code": reference_driver,
        "comparison_driver_code": comparison_driver,
        "reference_lap_number": reference["LapNumber"],
        "comparison_lap_number": comparison["LapNumber"],
        "reference_lap_time_s": reference_lap_time_s,
        "comparison_lap_time_s": comparison_lap_time_s,
        "matched_gap_s": comparison_lap_time_s - reference_lap_time_s,
        "compound": reference["Compound"],
        "reference_stint": reference.get("Stint", pd.NA),
        "comparison_stint": comparison.get("Stint", pd.NA),
        "stint_lap_index": reference["stint_lap_index"],
        "weather_bucket": reference["weather_bucket"],
        "track_status_bucket": reference["track_status_bucket"],
        "reference_position_start": reference.get("position_start", pd.NA),
        "reference_position_end": reference.get("position_end", pd.NA),
        "comparison_position_start": comparison.get("position_start", pd.NA),
        "comparison_position_end": comparison.get("position_end", pd.NA),
        "skip_reason": pd.NA,
    }


def _skipped_row(
    meta: _SessionMeta,
    session_kind: SessionKind,
    team: str,
    drivers: list[str],
    skip_reason: str,
) -> dict[str, Any]:
    """Build one canonical skipped-pair diagnostic row."""
    if skip_reason not in CANONICAL_SKIP_REASONS:
        raise ValueError(f"Non-canonical skip reason: {skip_reason!r}")
    sorted_drivers = sorted(drivers)
    return {
        "row_type": ROW_TYPE_SKIPPED_PAIR,
        "year": meta.year,
        "race_name": meta.race_name,
        "session_name": meta.session_name,
        "session_kind": session_kind,
        "team": team,
        "reference_driver_code": sorted_drivers[0] if sorted_drivers else pd.NA,
        "comparison_driver_code": sorted_drivers[1] if len(sorted_drivers) > 1 else pd.NA,
        "reference_lap_number": pd.NA,
        "comparison_lap_number": pd.NA,
        "reference_lap_time_s": pd.NA,
        "comparison_lap_time_s": pd.NA,
        "matched_gap_s": pd.NA,
        "compound": pd.NA,
        "reference_stint": pd.NA,
        "comparison_stint": pd.NA,
        "stint_lap_index": pd.NA,
        "weather_bucket": pd.NA,
        "track_status_bucket": pd.NA,
        "reference_position_start": pd.NA,
        "reference_position_end": pd.NA,
        "comparison_position_start": pd.NA,
        "comparison_position_end": pd.NA,
        "skip_reason": skip_reason,
    }


def _ordered_output(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return rows in the public output column order."""
    if not rows:
        return _empty_output()
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _empty_output() -> pd.DataFrame:
    """Return an empty DataFrame with the extractor output schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Raise a labelled KeyError when required columns are absent."""
    missing = columns.difference(frame.columns)
    if missing:
        raise KeyError(f"{label} missing required columns: {sorted(missing)}")


def _bootstrap_median_se(gaps: np.ndarray, *, config: MatchedLapConfig) -> float:
    """Estimate median standard error by non-parametric bootstrap."""
    if len(gaps) < 2:
        return float(config.matched_gap_se_floor_s)

    rng = np.random.default_rng(config.bootstrap_random_seed)
    samples = rng.choice(gaps, size=(config.bootstrap_samples, len(gaps)), replace=True)
    bootstrap_medians = np.median(samples, axis=1)
    se = float(np.std(bootstrap_medians, ddof=1))
    if not np.isfinite(se):
        return float(config.matched_gap_se_floor_s)
    return max(se, float(config.matched_gap_se_floor_s))
