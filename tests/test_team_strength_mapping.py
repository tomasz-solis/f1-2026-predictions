"""Tests for Phase 7 team-strength calibration helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.team_strength_mapping import (
    build_construct_aligned_driver_observations,
    evaluate_policy_folds,
)


def _matched_row(
    *,
    year: int,
    race_name: str,
    session_kind: str,
    team: str,
    reference: str,
    comparison: str,
    reference_lap_time_s: float,
    comparison_lap_time_s: float,
) -> dict[str, object]:
    """Build one raw matched-lap row for calibration-helper tests."""
    return {
        "row_type": "matched_pair",
        "year": year,
        "race_name": race_name,
        "session_name": "Race" if session_kind == "race" else "Qualifying",
        "session_kind": session_kind,
        "team": team,
        "reference_driver_code": reference,
        "comparison_driver_code": comparison,
        "reference_lap_time_s": reference_lap_time_s,
        "comparison_lap_time_s": comparison_lap_time_s,
        "weather_bucket": "dry",
    }


def test_construct_aligned_observations_keep_driver_and_field_medians() -> None:
    """Matched rows expand into one driver observation with field-relative seconds."""
    raw = pd.DataFrame(
        [
            _matched_row(
                year=2024,
                race_name="Example GP",
                session_kind="race",
                team="Fast",
                reference="AAA",
                comparison="BBB",
                reference_lap_time_s=90.0,
                comparison_lap_time_s=90.4,
            ),
            _matched_row(
                year=2024,
                race_name="Example GP",
                session_kind="race",
                team="Fast",
                reference="AAA",
                comparison="BBB",
                reference_lap_time_s=90.2,
                comparison_lap_time_s=90.6,
            ),
            _matched_row(
                year=2024,
                race_name="Example GP",
                session_kind="race",
                team="Slow",
                reference="CCC",
                comparison="DDD",
                reference_lap_time_s=91.0,
                comparison_lap_time_s=91.4,
            ),
            _matched_row(
                year=2024,
                race_name="Example GP",
                session_kind="race",
                team="Slow",
                reference="CCC",
                comparison="DDD",
                reference_lap_time_s=91.2,
                comparison_lap_time_s=91.6,
            ),
        ]
    )

    observations = build_construct_aligned_driver_observations(raw)

    aaa = observations.loc[observations["driver_code"].eq("AAA")].iloc[0]
    ddd = observations.loc[observations["driver_code"].eq("DDD")].iloc[0]
    assert aaa["driver_median_s"] == pytest.approx(90.1)
    assert ddd["driver_median_s"] == pytest.approx(91.5)
    assert aaa["field_median_s"] == pytest.approx(90.8)
    assert aaa["observed_driver_to_field_s"] == pytest.approx(0.7)
    assert ddd["observed_driver_to_field_s"] == pytest.approx(-0.7)
    assert aaa["team_strength_same_session"] == pytest.approx(1.0)
    assert ddd["team_strength_same_session"] == pytest.approx(0.0)


def test_shared_scalar_proxies_are_explicit_and_time_aware() -> None:
    """Race-derived proxies remain visible instead of masquerading as qualifying state."""
    rows: list[dict[str, object]] = []
    for race_name, fast_base, slow_base in (
        ("Round 1", 90.0, 91.0),
        ("Round 2", 90.4, 90.8),
    ):
        rows.extend(
            [
                _matched_row(
                    year=2024,
                    race_name=race_name,
                    session_kind="race",
                    team="Fast",
                    reference="AAA",
                    comparison="BBB",
                    reference_lap_time_s=fast_base,
                    comparison_lap_time_s=fast_base + 0.2,
                ),
                _matched_row(
                    year=2024,
                    race_name=race_name,
                    session_kind="race",
                    team="Slow",
                    reference="CCC",
                    comparison="DDD",
                    reference_lap_time_s=slow_base,
                    comparison_lap_time_s=slow_base + 0.2,
                ),
                _matched_row(
                    year=2024,
                    race_name=race_name,
                    session_kind="qualifying",
                    team="Fast",
                    reference="AAA",
                    comparison="BBB",
                    reference_lap_time_s=80.0,
                    comparison_lap_time_s=80.2,
                ),
                _matched_row(
                    year=2024,
                    race_name=race_name,
                    session_kind="qualifying",
                    team="Slow",
                    reference="CCC",
                    comparison="DDD",
                    reference_lap_time_s=80.6,
                    comparison_lap_time_s=80.8,
                ),
            ]
        )

    observations = build_construct_aligned_driver_observations(pd.DataFrame(rows))
    fast_round_2_quali = observations[
        observations["race_name"].eq("Round 2")
        & observations["session_kind"].eq("qualifying")
        & observations["team"].eq("Fast")
    ].iloc[0]
    fast_round_1_quali = observations[
        observations["race_name"].eq("Round 1")
        & observations["session_kind"].eq("qualifying")
        & observations["team"].eq("Fast")
    ].iloc[0]

    assert fast_round_2_quali["team_strength_race_event"] == pytest.approx(1.0)
    assert fast_round_2_quali["team_strength_race_season_mean"] == pytest.approx(1.0)
    assert (
        fast_round_1_quali["team_strength_race_trailing_mean"]
        != fast_round_1_quali["team_strength_race_trailing_mean"]
    )
    assert fast_round_2_quali["team_strength_race_trailing_mean"] == pytest.approx(1.0)


def test_policy_fold_evaluation_reports_combined_prediction_metrics() -> None:
    """Candidate evaluation returns folds and driver residual means for review."""
    rows: list[dict[str, object]] = []
    for year, faster_base in zip((2022, 2023, 2024, 2025), (90.0, 90.1, 90.2, 90.3), strict=True):
        rows.extend(
            [
                _matched_row(
                    year=year,
                    race_name=f"Round {year}",
                    session_kind="race",
                    team="Fast",
                    reference="AAA",
                    comparison="BBB",
                    reference_lap_time_s=faster_base,
                    comparison_lap_time_s=faster_base + 0.2,
                ),
                _matched_row(
                    year=year,
                    race_name=f"Round {year}",
                    session_kind="race",
                    team="Slow",
                    reference="CCC",
                    comparison="DDD",
                    reference_lap_time_s=faster_base + 1.0,
                    comparison_lap_time_s=faster_base + 1.2,
                ),
                _matched_row(
                    year=year,
                    race_name=f"Round {year}",
                    session_kind="qualifying",
                    team="Fast",
                    reference="AAA",
                    comparison="BBB",
                    reference_lap_time_s=faster_base - 10.0,
                    comparison_lap_time_s=faster_base - 9.8,
                ),
                _matched_row(
                    year=year,
                    race_name=f"Round {year}",
                    session_kind="qualifying",
                    team="Slow",
                    reference="CCC",
                    comparison="DDD",
                    reference_lap_time_s=faster_base - 9.0,
                    comparison_lap_time_s=faster_base - 8.8,
                ),
            ]
        )
    observations = build_construct_aligned_driver_observations(
        pd.DataFrame(rows),
        driver_mu_by_kind={
            "race": {"AAA": 0.1, "BBB": -0.1, "CCC": 0.1, "DDD": -0.1},
            "qualifying": {"AAA": 0.1, "BBB": -0.1, "CCC": 0.1, "DDD": -0.1},
        },
    )

    evaluation = evaluate_policy_folds(observations, policy="same_session_construct")

    assert evaluation["policy"] == "same_session_construct"
    assert len(evaluation["folds"]) == 8
    assert {row["session_kind"] for row in evaluation["folds"]} == {"race", "qualifying"}
    assert evaluation["per_driver_residual_means"]
