"""Tests for canonical teammate matched-lap extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from src.extractors.matched_laps import (
    SKIP_INSUFFICIENT_MATCHED_PAIRS,
    SKIP_LAP_LEVEL_WEATHER_UNRELIABLE,
    SKIP_NO_COMMON_QUALI_SEGMENT,
    SKIP_TEAMMATE_DNF_NO_MATCHED_LAPS,
    MatchedLapConfig,
    aggregate_matched_teammate_laps,
    diagnose_matched_lap_filters,
    extract_matched_teammate_laps,
)


@dataclass
class _Session:
    """Small FastF1-shaped session object for extractor tests."""

    laps: pd.DataFrame
    weather_data: pd.DataFrame
    results: pd.DataFrame | None = None
    session_status: pd.DataFrame | None = None
    year: int = 2024
    race_name: str = "Synthetic Grand Prix"
    session_name: str = "Race"


def _time(seconds: float) -> pd.Timedelta:
    """Return a session-relative timestamp."""
    return pd.to_timedelta(seconds, unit="s")


def _race_lap(
    *,
    driver: str,
    team: str,
    lap_number: int,
    lap_time_s: float,
    compound: str = "MEDIUM",
    stint: int = 1,
    track_status: str = "1",
    position: int = 5,
) -> dict[str, Any]:
    """Build one synthetic race lap with FastF1-like columns."""
    start = _time(lap_number * 100)
    return {
        "Driver": driver,
        "Team": team,
        "LapNumber": lap_number,
        "LapTime": _time(lap_time_s),
        "Compound": compound,
        "Stint": stint,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": track_status,
        "Position": position,
        "Deleted": None,
        "IsAccurate": True,
        "LapStartTime": start,
        "Time": start + _time(lap_time_s),
    }


def _race_laps(
    *,
    drivers: tuple[str, str] = ("AAA", "BBB"),
    team: str = "Example",
    lap_numbers: range = range(1, 6),
    reference_base_s: float = 90.0,
    comparison_base_s: float = 90.5,
) -> pd.DataFrame:
    """Build race laps for two teammates across the same stint."""
    rows: list[dict[str, Any]] = []
    for lap_number in lap_numbers:
        rows.append(
            _race_lap(
                driver=drivers[0],
                team=team,
                lap_number=lap_number,
                lap_time_s=reference_base_s + lap_number * 0.01,
            )
        )
        rows.append(
            _race_lap(
                driver=drivers[1],
                team=team,
                lap_number=lap_number,
                lap_time_s=comparison_base_s + lap_number * 0.01,
            )
        )
    return pd.DataFrame(rows)


def _weather_for_laps(
    lap_numbers: range,
    *,
    rainfall_by_lap: dict[int, list[bool]] | None = None,
) -> pd.DataFrame:
    """Build weather samples inside each synthetic lap interval."""
    rainfall_by_lap = rainfall_by_lap or {}
    rows: list[dict[str, Any]] = []
    for lap_number in lap_numbers:
        samples = rainfall_by_lap.get(lap_number, [False])
        for idx, rainfall in enumerate(samples):
            rows.append(
                {
                    "Time": _time(lap_number * 100 + 20 + idx * 20),
                    "Rainfall": rainfall,
                }
            )
    return pd.DataFrame(rows)


def _quali_lap(
    *,
    driver: str,
    team: str,
    lap_number: int,
    lap_time_s: float,
    segment: str,
    compound: str = "SOFT",
) -> dict[str, Any]:
    """Build one synthetic qualifying push lap."""
    start = _time(lap_number * 100)
    return {
        "Driver": driver,
        "Team": team,
        "LapNumber": lap_number,
        "LapTime": _time(lap_time_s),
        "Compound": compound,
        "Stint": lap_number,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": "1",
        "Position": pd.NA,
        "Deleted": None,
        "IsAccurate": True,
        "LapStartTime": start,
        "Time": start + _time(lap_time_s),
        "Segment": segment,
    }


def test_race_sign_convention_and_reference_order_are_deterministic() -> None:
    """Alphabetical reference ordering drives a positive faster-reference gap."""
    laps = _race_laps(drivers=("BBB", "AAA"), reference_base_s=90.6, comparison_base_s=90.0)
    session = _Session(laps=laps, weather_data=_weather_for_laps(range(1, 6)))
    config = MatchedLapConfig(min_matched_pairs_race=3)

    out = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="dry",
        config=config,
    )

    assert set(out["row_type"]) == {"matched_pair"}
    assert len(out) == 3
    assert set(out["reference_driver_code"]) == {"AAA"}
    assert set(out["comparison_driver_code"]) == {"BBB"}
    assert out["matched_gap_s"].tolist() == pytest.approx([0.6, 0.6, 0.6])


def test_race_rows_are_not_mirrored_as_independent_evidence() -> None:
    """The extractor emits one row per pair, never both driver directions."""
    laps = _race_laps(lap_numbers=range(1, 7))
    session = _Session(laps=laps, weather_data=_weather_for_laps(range(1, 7)))
    config = MatchedLapConfig(min_matched_pairs_race=4)

    out = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="dry",
        config=config,
    )

    assert len(out) == 4
    assert set(zip(out["reference_driver_code"], out["comparison_driver_code"], strict=True)) == {
        ("AAA", "BBB")
    }


def test_insufficient_race_pairs_emit_one_skip_row() -> None:
    """Below-threshold race pairs are gated as a skipped pair."""
    laps = _race_laps(lap_numbers=range(1, 5))
    session = _Session(laps=laps, weather_data=_weather_for_laps(range(1, 5)))
    config = MatchedLapConfig(min_matched_pairs_race=4)

    out = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="dry",
        config=config,
    )

    assert out["row_type"].tolist() == ["skipped_pair"]
    assert out["skip_reason"].tolist() == [SKIP_INSUFFICIENT_MATCHED_PAIRS]


def test_early_teammate_dnf_without_comparison_laps_is_skipped() -> None:
    """An early final lap can leave no valid teammate comparison."""
    laps = pd.concat(
        [
            _race_laps(drivers=("AAA", "AAA"), lap_numbers=range(1, 3)).iloc[::2],
            _race_laps(drivers=("BBB", "BBB"), lap_numbers=range(1, 8)).iloc[::2],
        ],
        ignore_index=True,
    )
    session = _Session(laps=laps, weather_data=_weather_for_laps(range(1, 8)))
    config = MatchedLapConfig(min_matched_pairs_race=3)

    out = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="dry",
        config=config,
    )

    assert out["row_type"].tolist() == ["skipped_pair"]
    assert out["skip_reason"].tolist() == [SKIP_TEAMMATE_DNF_NO_MATCHED_LAPS]


def test_mixed_race_routes_dry_and_wet_laps_but_excludes_unreliable_laps() -> None:
    """Mixed weather returns reliable dry/wet pairs and drops mixed intervals."""
    laps = _race_laps(lap_numbers=range(1, 6))
    weather = _weather_for_laps(
        range(1, 6),
        rainfall_by_lap={
            2: [False],
            3: [True],
            4: [False, True],
        },
    )
    session = _Session(laps=laps, weather_data=weather)
    config = MatchedLapConfig(min_matched_pairs_race=2)

    out = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="mixed",
        config=config,
    )

    assert set(out["row_type"]) == {"matched_pair"}
    assert set(out["weather_bucket"]) == {"dry", "wet"}
    assert len(out) == 2


def test_unreliable_weather_only_emits_canonical_skip_reason() -> None:
    """Pairs with only mixed or unmapped weather are skipped canonically."""
    laps = _race_laps(lap_numbers=range(1, 5))
    weather = _weather_for_laps(
        range(1, 5),
        rainfall_by_lap={lap_number: [False, True] for lap_number in range(1, 5)},
    )
    session = _Session(laps=laps, weather_data=weather)
    config = MatchedLapConfig(min_matched_pairs_race=1)

    out = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="mixed",
        config=config,
    )

    assert out["row_type"].tolist() == ["skipped_pair"]
    assert out["skip_reason"].tolist() == [SKIP_LAP_LEVEL_WEATHER_UNRELIABLE]


def test_qualifying_uses_highest_common_segment_until_minimum_is_met() -> None:
    """Q2 is used before Q1 when teammates do not both reach Q3."""
    rows = []
    for lap_number in (10, 11, 12):
        rows.append(
            _quali_lap(
                driver="AAA",
                team="Example",
                lap_number=lap_number,
                lap_time_s=89.0 + lap_number * 0.01,
                segment="Q1",
            )
        )
        rows.append(
            _quali_lap(
                driver="BBB",
                team="Example",
                lap_number=lap_number,
                lap_time_s=89.3 + lap_number * 0.01,
                segment="Q1",
            )
        )
    for lap_number in (20, 21, 22):
        rows.append(
            _quali_lap(
                driver="AAA",
                team="Example",
                lap_number=lap_number,
                lap_time_s=88.8 + lap_number * 0.01,
                segment="Q2",
            )
        )
        rows.append(
            _quali_lap(
                driver="BBB",
                team="Example",
                lap_number=lap_number,
                lap_time_s=89.2 + lap_number * 0.01,
                segment="Q2",
            )
        )
    rows.append(
        _quali_lap(
            driver="AAA",
            team="Example",
            lap_number=30,
            lap_time_s=88.5,
            segment="Q3",
        )
    )
    laps = pd.DataFrame(rows)
    session = _Session(
        laps=laps,
        weather_data=_weather_for_laps(range(10, 31)),
        session_name="Qualifying",
    )
    config = MatchedLapConfig(min_matched_pairs_quali=3)

    out = extract_matched_teammate_laps(
        session,
        session_kind="qualifying",
        weather_mode="dry",
        config=config,
    )

    assert set(out["row_type"]) == {"matched_pair"}
    assert out["reference_lap_number"].tolist() == [20, 21, 22]
    assert out["comparison_lap_number"].tolist() == [20, 21, 22]


def test_qualifying_without_common_segment_is_skipped() -> None:
    """Synthetic split qualifying coverage exercises no_common_quali_segment."""
    laps = pd.DataFrame(
        [
            _quali_lap(
                driver="AAA",
                team="Example",
                lap_number=10,
                lap_time_s=89.0,
                segment="Q2",
            ),
            _quali_lap(
                driver="BBB",
                team="Example",
                lap_number=20,
                lap_time_s=89.5,
                segment="Q1",
            ),
        ]
    )
    session = _Session(
        laps=laps,
        weather_data=_weather_for_laps(range(10, 21)),
        session_name="Qualifying",
    )
    config = MatchedLapConfig(min_matched_pairs_quali=1)

    out = extract_matched_teammate_laps(
        session,
        session_kind="qualifying",
        weather_mode="dry",
        config=config,
    )

    assert out["row_type"].tolist() == ["skipped_pair"]
    assert out["skip_reason"].tolist() == [SKIP_NO_COMMON_QUALI_SEGMENT]


def test_qualifying_excludes_non_quick_laps_before_pairing() -> None:
    """Qualifying matching ignores slow cooldown laps that still look accurate."""
    laps = pd.DataFrame(
        [
            _quali_lap(
                driver="AAA",
                team="Example",
                lap_number=10,
                lap_time_s=90.0,
                segment="Q1",
            ),
            _quali_lap(
                driver="BBB",
                team="Example",
                lap_number=10,
                lap_time_s=90.2,
                segment="Q1",
            ),
            _quali_lap(
                driver="AAA",
                team="Example",
                lap_number=11,
                lap_time_s=90.1,
                segment="Q1",
            ),
            _quali_lap(
                driver="BBB",
                team="Example",
                lap_number=11,
                lap_time_s=90.3,
                segment="Q1",
            ),
            _quali_lap(
                driver="AAA",
                team="Example",
                lap_number=12,
                lap_time_s=150.0,
                segment="Q1",
            ),
            _quali_lap(
                driver="BBB",
                team="Example",
                lap_number=12,
                lap_time_s=90.4,
                segment="Q1",
            ),
        ]
    )
    session = _Session(
        laps=laps,
        weather_data=_weather_for_laps(range(10, 13)),
        session_name="Qualifying",
    )
    config = MatchedLapConfig(min_matched_pairs_quali=2)

    out = extract_matched_teammate_laps(
        session,
        session_kind="qualifying",
        weather_mode="dry",
        config=config,
    )
    diagnostics = diagnose_matched_lap_filters(
        session,
        session_kind="qualifying",
        weather_mode="dry",
        config=config,
    )

    assert set(out["row_type"]) == {"matched_pair"}
    assert out["reference_lap_number"].tolist() == [10, 11]
    assert out["comparison_lap_number"].tolist() == [10, 11]
    assert diagnostics.loc[0, "non_quick_qualifying_laps"] == 1


def test_aggregate_matched_laps_keeps_one_pair_observation_with_bootstrap_se() -> None:
    """Aggregation keeps one teammate-pair row and carries the signed median."""
    laps = _race_laps(lap_numbers=range(1, 7))
    session = _Session(laps=laps, weather_data=_weather_for_laps(range(1, 7)))
    config = MatchedLapConfig(min_matched_pairs_race=4, bootstrap_samples=200)
    raw = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="dry",
        config=config,
    )

    aggregate = aggregate_matched_teammate_laps(raw, config=config)

    assert len(aggregate) == 1
    assert aggregate.loc[0, "n_matched_pairs"] == 4
    assert aggregate.loc[0, "matched_gap_median_s"] == pytest.approx(0.5)
    assert aggregate.loc[0, "matched_gap_se_s"] >= config.matched_gap_se_floor_s


def test_aggregate_matched_laps_gates_weather_buckets_independently() -> None:
    """Sparse dry/wet buckets are skipped even when the raw pair clears the total."""
    laps = _race_laps(lap_numbers=range(1, 6))
    weather = _weather_for_laps(
        range(1, 6),
        rainfall_by_lap={
            2: [False],
            3: [True],
            4: [False, True],
        },
    )
    session = _Session(laps=laps, weather_data=weather)
    config = MatchedLapConfig(min_matched_pairs_race=2)
    raw = extract_matched_teammate_laps(
        session,
        session_kind="race",
        weather_mode="mixed",
        config=config,
    )

    aggregate = aggregate_matched_teammate_laps(raw, config=config)

    assert sorted(aggregate["weather_bucket"].tolist()) == ["dry", "wet"]
    assert aggregate["n_matched_pairs"].tolist() == [1, 1]
    assert aggregate["skip_reason"].tolist() == [
        SKIP_INSUFFICIENT_MATCHED_PAIRS,
        SKIP_INSUFFICIENT_MATCHED_PAIRS,
    ]
    assert aggregate["matched_gap_median_s"].isna().all()
    assert aggregate["matched_gap_se_s"].isna().all()


def test_filter_diagnostics_expose_routine_filter_counts() -> None:
    """Smoke diagnostics report filtering without changing skip reasons."""
    laps = _race_laps(lap_numbers=range(1, 9))
    laps["PitInTime"] = laps["PitInTime"].astype("object")
    laps.loc[(laps["Driver"] == "AAA") & (laps["LapNumber"] == 2), "TrackStatus"] = "4"
    laps.loc[(laps["Driver"] == "BBB") & (laps["LapNumber"] == 2), "TrackStatus"] = "67"
    laps.loc[(laps["Driver"] == "AAA") & (laps["LapNumber"] == 3), "PitInTime"] = _time(350)
    laps.loc[(laps["Driver"] == "BBB") & (laps["LapNumber"] == 5), "LapTime"] = _time(180)
    weather = _weather_for_laps(
        range(1, 9),
        rainfall_by_lap={4: [False, True]},
    )
    session = _Session(laps=laps, weather_data=weather)
    config = MatchedLapConfig(
        min_matched_pairs_race=1,
        traffic_stint_sigma_threshold=0.5,
    )

    diagnostics = diagnose_matched_lap_filters(
        session,
        session_kind="race",
        weather_mode="mixed",
        config=config,
    )

    row = diagnostics.iloc[0]
    assert row["sc_vsc_laps"] == 2
    assert row["pit_laps"] == 1
    assert row["lap_level_weather_unreliable_laps"] == 2
    assert row["stint_outlier_laps"] >= 1
    assert row["matched_pair_rows"] == row["candidate_matched_pairs"]
