"""Tests for the Phase 2 smoke-session inspector.

These tests use synthetic DataFrames mirroring FastF1's column shape so
the inspector logic can be exercised without FastF1 installed.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
import pytest
from scripts.inspect_smoke_sessions import build_parser, format_text_summary

from src.diagnostics.smoke_inspector.inspector import (
    inspect_session,
    summarize_lap_counts,
    summarize_qualifying_segments,
    summarize_retirements,
    summarize_track_status,
    summarize_weather_samples,
)

# ---------------------------------------------------------------------------
# weather
# ---------------------------------------------------------------------------


def test_weather_summary_all_dry() -> None:
    """All-False rainfall samples produce a non-mixed dry summary."""
    weather = pd.DataFrame({"Rainfall": [False, False, False, False]})
    out = summarize_weather_samples(weather)
    assert out.n_samples == 4
    assert out.n_rainfall_true == 0
    assert out.n_rainfall_false == 4
    assert out.has_dry_samples is True
    assert out.has_rain_samples is False
    assert out.is_mixed is False
    assert out.rainfall_fraction == pytest.approx(0.0)


def test_weather_summary_all_wet() -> None:
    """All-True rainfall samples produce a non-mixed wet summary."""
    weather = pd.DataFrame({"Rainfall": [True, True, True]})
    out = summarize_weather_samples(weather)
    assert out.has_rain_samples is True
    assert out.has_dry_samples is False
    assert out.is_mixed is False
    assert out.rainfall_fraction == pytest.approx(1.0)


def test_weather_summary_mixed() -> None:
    """A mix of rainfall samples sets is_mixed True."""
    weather = pd.DataFrame({"Rainfall": [False, True, False, True]})
    out = summarize_weather_samples(weather)
    assert out.is_mixed is True
    assert out.rainfall_fraction == pytest.approx(0.5)


def test_weather_summary_empty_returns_zero_samples() -> None:
    """An empty weather frame yields zero samples without raising."""
    weather = pd.DataFrame({"Rainfall": pd.Series(dtype=bool)})
    out = summarize_weather_samples(weather)
    assert out.n_samples == 0
    assert out.is_mixed is False
    assert out.rainfall_fraction == pytest.approx(0.0)


def test_weather_summary_missing_column_raises() -> None:
    """Missing Rainfall column raises a clear KeyError."""
    weather = pd.DataFrame({"AirTemp": [20.0, 21.0]})
    with pytest.raises(KeyError, match="Rainfall"):
        summarize_weather_samples(weather)


# ---------------------------------------------------------------------------
# lap counts
# ---------------------------------------------------------------------------


def _make_results(driver_team_status: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a minimal results DataFrame for tests."""
    return pd.DataFrame(
        {
            "Abbreviation": [d for d, _, _ in driver_team_status],
            "TeamName": [t for _, t, _ in driver_team_status],
            "Status": [s for _, _, s in driver_team_status],
        }
    )


def _collect_mapping_keys(value: Any) -> set[str]:
    """Return every dictionary key found in a nested JSON-like object."""
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys.update(_collect_mapping_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_collect_mapping_keys(nested))
        return keys
    return set()


def test_lap_counts_full_distance_and_partial() -> None:
    """Drivers reaching max lap go in full; shorter drivers go in partial."""
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "PER", "PER"],
            "LapNumber": [1, 2, 3, 1, 2],
        }
    )
    results = _make_results([("VER", "Red Bull", "Finished"), ("PER", "Red Bull", "Engine")])
    out = summarize_lap_counts(laps, results)
    assert out.by_driver == {"VER": 3, "PER": 2}
    assert out.full_distance_drivers == ["VER"]
    assert out.partial_distance_drivers == ["PER"]
    assert out.by_team == {"Red Bull": ["VER", "PER"]}
    assert out.max_observed_lap == 3


def test_lap_counts_missing_required_column_raises() -> None:
    """Missing 'LapNumber' raises a labelled KeyError."""
    laps = pd.DataFrame({"Driver": ["VER"]})
    results = _make_results([("VER", "Red Bull", "Finished")])
    with pytest.raises(KeyError, match="LapNumber"):
        summarize_lap_counts(laps, results)


# ---------------------------------------------------------------------------
# retirements
# ---------------------------------------------------------------------------


def test_retirements_finished_status_excluded() -> None:
    """Drivers with Status='Finished' are not flagged as retired."""
    laps = pd.DataFrame({"Driver": ["VER"], "LapNumber": [50]})
    results = _make_results([("VER", "Red Bull", "Finished")])
    out = summarize_retirements(results, laps)
    assert out.retired_drivers == {}


def test_retirements_lap_status_excluded() -> None:
    """Drivers classified +1 Lap / +2 Laps are not flagged as retired."""
    laps = pd.DataFrame({"Driver": ["NOR"], "LapNumber": [49]})
    results = _make_results([("NOR", "McLaren", "+1 Lap")])
    out = summarize_retirements(results, laps)
    assert out.retired_drivers == {}


def test_retirements_early_dnf_flagged() -> None:
    """A retirement at or before the early threshold is flagged is_early=True."""
    laps = pd.DataFrame({"Driver": ["ALO"], "LapNumber": [3]})
    results = _make_results([("ALO", "Aston Martin", "Engine")])
    out = summarize_retirements(results, laps)
    assert "ALO" in out.retired_drivers
    rec = out.retired_drivers["ALO"]
    assert rec.is_early is True
    assert rec.retirement_lap == 3
    assert rec.classified_status == "Engine"


def test_retirements_late_dnf_not_early() -> None:
    """A retirement after the early threshold is is_early=False."""
    laps = pd.DataFrame({"Driver": ["RUS"], "LapNumber": [40]})
    results = _make_results([("RUS", "Mercedes", "Hydraulics")])
    out = summarize_retirements(results, laps, early_threshold_lap=10)
    rec = out.retired_drivers["RUS"]
    assert rec.is_early is False


def test_retirements_threshold_is_inclusive() -> None:
    """A retirement exactly at the threshold counts as early."""
    laps = pd.DataFrame({"Driver": ["TSU"], "LapNumber": [10]})
    results = _make_results([("TSU", "RB", "Collision")])
    out = summarize_retirements(results, laps, early_threshold_lap=10)
    assert out.retired_drivers["TSU"].is_early is True


def test_retirements_no_lap_data_keeps_record_with_none_lap() -> None:
    """A retired driver with no laps still appears with last_lap=None."""
    laps = pd.DataFrame({"Driver": [], "LapNumber": []}).astype({"Driver": str, "LapNumber": int})
    results = _make_results([("BOT", "Stake", "Accident")])
    out = summarize_retirements(results, laps)
    assert "BOT" in out.retired_drivers
    rec = out.retired_drivers["BOT"]
    assert rec.retirement_lap is None
    assert rec.is_early is False


# ---------------------------------------------------------------------------
# track status
# ---------------------------------------------------------------------------


def test_track_status_counts_safety_car_and_vsc() -> None:
    """Status codes 4 and 6/7 are counted as SC and VSC respectively."""
    track = pd.DataFrame(
        {
            "Status": ["1", "4", "1", "6", "7", "1"],
            "Time": pd.to_timedelta(
                ["0:00:00", "0:10:00", "0:15:00", "0:20:00", "0:22:00", "0:25:00"]
            ),
        }
    )
    out = summarize_track_status(track)
    assert out.n_safety_car == 1
    assert out.n_virtual_safety_car == 2
    assert out.n_red_flag == 0
    assert len(out.events) == 6


def test_text_summary_labels_track_status_as_rows() -> None:
    """The CLI summary should not imply raw status rows are incidents."""
    laps = pd.DataFrame({"Driver": ["VER"], "LapNumber": [50]})
    results = _make_results([("VER", "Red Bull", "Finished")])
    weather = pd.DataFrame({"Rainfall": [False, False]})
    track = pd.DataFrame(
        {
            "Status": ["1", "4", "6", "7"],
            "Time": pd.to_timedelta(["0:00:00", "0:10:00", "0:20:00", "0:22:00"]),
        }
    )

    inspection = inspect_session(
        year=2024,
        event_name="Bahrain Grand Prix",
        session_kind="race",
        laps_df=laps,
        results_df=results,
        weather_df=weather,
        track_status_df=track,
    )

    summary = format_text_summary(inspection)

    assert "track status rows:" in summary
    assert "SC_rows=1" in summary
    assert "VSC_rows=2" in summary
    assert "SC=1" not in summary
    assert "VSC=2" not in summary


def test_track_status_unknown_code_emits_unknown_label() -> None:
    """A code outside the known map is emitted with an unknown_<code> label."""
    track = pd.DataFrame({"Status": ["99"], "Time": pd.to_timedelta(["0:00:00"])})
    out = summarize_track_status(track)
    assert out.events[0].status_label == "unknown_99"


def test_track_status_empty_dataframe() -> None:
    """An empty DataFrame produces zero counts and no events."""
    track = pd.DataFrame({"Status": [], "Time": pd.to_timedelta([])})
    out = summarize_track_status(track)
    assert out.events == []
    assert out.n_safety_car == 0


# ---------------------------------------------------------------------------
# qualifying segments
# ---------------------------------------------------------------------------


def test_qualifying_team_with_q3_listed() -> None:
    """A team where both drivers reached Q3 is in teams_with_q3."""
    laps = pd.DataFrame({"Driver": ["VER", "PER"]})
    results = pd.DataFrame(
        {
            "Abbreviation": ["VER", "PER"],
            "TeamName": ["Red Bull", "Red Bull"],
            "Q1": pd.to_timedelta(["0:01:30", "0:01:31"]),
            "Q2": pd.to_timedelta(["0:01:29", "0:01:30"]),
            "Q3": pd.to_timedelta(["0:01:28", "0:01:29"]),
        }
    )
    out = summarize_qualifying_segments(laps, results)
    assert "Red Bull" in out.teams_with_q3
    assert "Red Bull" not in out.teams_with_q1_eliminated


def test_qualifying_team_q1_eliminated_listed() -> None:
    """A team where neither driver reached Q2 is listed in q1_eliminated."""
    laps = pd.DataFrame({"Driver": ["SAR", "ALB"]})
    results = pd.DataFrame(
        {
            "Abbreviation": ["SAR", "ALB"],
            "TeamName": ["Williams", "Williams"],
            "Q1": pd.to_timedelta(["0:01:32", "0:01:33"]),
            "Q2": [pd.NaT, pd.NaT],
            "Q3": [pd.NaT, pd.NaT],
        }
    )
    out = summarize_qualifying_segments(laps, results)
    assert "Williams" in out.teams_with_q1_eliminated
    assert "Williams" not in out.teams_with_q3


def test_qualifying_split_team_detected() -> None:
    """A team with one driver in Q3 and the other eliminated in Q1 is split."""
    laps = pd.DataFrame({"Driver": ["LEC", "SAI"]})
    results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "SAI"],
            "TeamName": ["Ferrari", "Ferrari"],
            "Q1": pd.to_timedelta(["0:01:30", "0:01:32"]),
            "Q2": [pd.to_timedelta("0:01:29"), pd.NaT],
            "Q3": [pd.to_timedelta("0:01:28"), pd.NaT],
        }
    )
    out = summarize_qualifying_segments(laps, results)
    assert "Ferrari" in out.teams_with_split_segments


def test_qualifying_missing_q_columns_raises() -> None:
    """Missing Q1/Q2/Q3 columns produce a labelled KeyError."""
    laps = pd.DataFrame({"Driver": ["VER"]})
    results = pd.DataFrame({"Abbreviation": ["VER"], "TeamName": ["Red Bull"]})
    with pytest.raises(KeyError, match="Q1"):
        summarize_qualifying_segments(laps, results)


# ---------------------------------------------------------------------------
# top-level inspect_session
# ---------------------------------------------------------------------------


def test_inspect_session_skips_qualifying_summary_for_race() -> None:
    """A race kind produces qualifying=None on the combined inspection."""
    laps = pd.DataFrame({"Driver": ["VER"], "LapNumber": [50]})
    results = _make_results([("VER", "Red Bull", "Finished")])
    weather = pd.DataFrame({"Rainfall": [False, False]})
    track = pd.DataFrame({"Status": ["1"], "Time": pd.to_timedelta(["0:00:00"])})

    out = inspect_session(
        year=2024,
        event_name="Bahrain Grand Prix",
        session_kind="race",
        laps_df=laps,
        results_df=results,
        weather_df=weather,
        track_status_df=track,
    )
    assert out.qualifying is None
    assert out.session_kind == "race"
    assert out.year == 2024
    assert out.event_name == "Bahrain Grand Prix"


def test_inspect_session_populates_qualifying_for_qualifying_kind() -> None:
    """A qualifying kind populates the qualifying summary."""
    laps = pd.DataFrame({"Driver": ["VER", "PER"]})
    results = pd.DataFrame(
        {
            "Abbreviation": ["VER", "PER"],
            "TeamName": ["Red Bull", "Red Bull"],
            "Status": ["Finished", "Finished"],
            "Q1": pd.to_timedelta(["0:01:30", "0:01:31"]),
            "Q2": pd.to_timedelta(["0:01:29", "0:01:30"]),
            "Q3": pd.to_timedelta(["0:01:28", "0:01:29"]),
        }
    )
    laps["LapNumber"] = [1, 1]
    weather = pd.DataFrame({"Rainfall": [False]})
    track = pd.DataFrame({"Status": ["1"], "Time": pd.to_timedelta(["0:00:00"])})

    out = inspect_session(
        year=2024,
        event_name="Bahrain Grand Prix",
        session_kind="qualifying",
        laps_df=laps,
        results_df=results,
        weather_df=weather,
        track_status_df=track,
    )
    assert out.qualifying is not None
    assert "Red Bull" in out.qualifying.teams_with_q3


def test_cli_default_cache_path_uses_repo_fastf1_convention() -> None:
    """The inspection CLI should default to the repo's FastF1 cache path."""
    parser = build_parser()
    args = parser.parse_args([])

    assert str(args.cache_dir) == "data/raw/.fastf1_cache"


def test_inspection_output_has_no_extractor_shaped_fields() -> None:
    """Phase 2 inspection output must not look like Phase 3 extractor rows."""
    laps = pd.DataFrame({"Driver": ["VER"], "LapNumber": [50]})
    results = _make_results([("VER", "Red Bull", "Finished")])
    weather = pd.DataFrame({"Rainfall": [False]})
    track = pd.DataFrame({"Status": ["1"], "Time": pd.to_timedelta(["0:00:00"])})

    out = inspect_session(
        year=2024,
        event_name="Bahrain Grand Prix",
        session_kind="race",
        laps_df=laps,
        results_df=results,
        weather_df=weather,
        track_status_df=track,
    )
    output_keys = _collect_mapping_keys(asdict(out))

    forbidden_extractor_keys = {
        "row_type",
        "matched_pair",
        "skipped_pair",
        "skip_reason",
        "matched_gap_s",
        "reference_driver_code",
        "comparison_driver_code",
        "stint_lap_index",
    }
    assert output_keys.isdisjoint(forbidden_extractor_keys)
