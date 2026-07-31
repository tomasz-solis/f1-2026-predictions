from __future__ import annotations

import pandas as pd
import pytest

from src.features.race_practice_evidence import (
    RacePracticeEvidenceConfig,
    apply_race_practice_evidence,
    build_race_practice_evidence,
)


def _long_run_laps() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for driver, team, base, degradation in (
        ("AAA", "Team A", 90.0, 0.10),
        ("BBB", "Team B", 91.0, 0.15),
    ):
        for index in range(10):
            tyre_age = index + 1
            raw_slope = degradation - 0.045
            rows.append(
                {
                    "Driver": driver,
                    "Team": team,
                    "LapNumber": index + 1,
                    "LapTime": pd.Timedelta(seconds=base + (raw_slope * index)),
                    "Compound": "SOFT",
                    "TyreLife": tyre_age,
                    "Stint": 1,
                    "Time": pd.Timedelta(seconds=120 + (index * 95)),
                    "IsAccurate": True,
                    "Deleted": False,
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TrackStatus": "1",
                }
            )
    return pd.DataFrame(rows)


def _team_laps(team: str) -> pd.DataFrame:
    laps = _long_run_laps()
    return laps[laps["Team"].eq(team)].copy().reset_index(drop=True)


def test_build_race_practice_evidence_extracts_comparable_degradation() -> None:
    payload = build_race_practice_evidence(
        {"FP2": _long_run_laps()},
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )

    assert payload["artifact_type"] == "race_practice_evidence"
    assert payload["diagnostics"]["long_run_stints"] == 2
    team_a = payload["teams"]["Team A"]["compounds"]["SOFT"]
    team_b = payload["teams"]["Team B"]["compounds"]["SOFT"]
    assert team_a["tire_deg_slope_s_per_lap"] == pytest.approx(0.10, abs=0.002)
    assert team_b["tire_deg_slope_s_per_lap"] == pytest.approx(0.15, abs=0.002)
    assert team_a["pace_comparison_status"] == "matched"
    assert team_b["pace_comparison_status"] == "matched"
    assert team_a["matched_pace_buckets"] == 1
    assert team_a["matched_pace_laps"] == 10
    assert team_a["matched_pace_stints"] == 1
    assert team_a["pace_performance"] == 1.0
    assert team_b["pace_performance"] == 0.0


def test_unmatched_fp_sessions_do_not_create_cross_team_pace_ranking() -> None:
    payload = build_race_practice_evidence(
        {
            "FP1": _team_laps("Team A"),
            "FP2": _team_laps("Team B"),
        },
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )

    team_a = payload["teams"]["Team A"]["compounds"]["SOFT"]
    team_b = payload["teams"]["Team B"]["compounds"]["SOFT"]
    assert team_a["reference_pace_s"] < team_b["reference_pace_s"]
    assert team_a["pace_comparison_status"] == "no_matched_bucket"
    assert team_b["pace_comparison_status"] == "no_matched_bucket"
    assert "pace_performance" not in team_a
    assert "pace_performance" not in team_b
    assert team_a["tire_deg_slope_s_per_lap"] == pytest.approx(0.10, abs=0.002)
    assert team_b["tire_deg_slope_s_per_lap"] == pytest.approx(0.15, abs=0.002)


def test_unmatched_tyre_age_windows_do_not_create_pace_ranking() -> None:
    laps = _long_run_laps()
    laps.loc[laps["Team"].eq("Team B"), "TyreLife"] += 10

    payload = build_race_practice_evidence(
        {"FP2": laps},
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )

    assert all(
        metrics["pace_comparison_status"] == "no_matched_bucket"
        and "pace_performance" not in metrics
        for team in payload["teams"].values()
        for metrics in team["compounds"].values()
    )


def test_unmatched_evolution_windows_do_not_create_pace_ranking() -> None:
    laps = _long_run_laps()
    later = laps["Team"].eq("Team B")
    laps.loc[later, "Time"] += pd.Timedelta(minutes=25)

    payload = build_race_practice_evidence(
        {"FP2": laps},
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )

    assert all(
        metrics["pace_comparison_status"] == "no_matched_bucket"
        and "pace_performance" not in metrics
        for team in payload["teams"].values()
        for metrics in team["compounds"].values()
    )


def test_race_evidence_rejects_short_interrupted_and_non_green_runs() -> None:
    laps = _long_run_laps().iloc[:10].copy()
    laps.loc[laps["LapNumber"] == 5, "LapNumber"] = 20
    laps.loc[laps["LapNumber"] == 8, "TrackStatus"] = "4"

    payload = build_race_practice_evidence(
        laps,
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
        config=RacePracticeEvidenceConfig(min_long_run_laps=8),
    )

    assert payload["teams"] == {}
    assert "no_comparable_long_runs" in payload["diagnostics"]["fallback_reasons"]
    assert payload["diagnostics"]["excluded_laps_by_reason"]["non_green"] == 1


def test_race_evidence_is_dry_only() -> None:
    payload = build_race_practice_evidence(
        _long_run_laps(),
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
        weather="mixed",
    )

    assert payload["teams"] == {}
    assert payload["diagnostics"]["fallback_reasons"] == ["dry_only_candidate"]


@pytest.mark.parametrize(
    ("column", "dry_value", "wet_value"),
    [
        ("Rainfall", False, True),
        ("weather_bucket", "dry", "wet"),
    ],
)
def test_wet_or_rainfall_laps_cannot_enter_dry_pace_evidence(
    column: str,
    dry_value: object,
    wet_value: object,
) -> None:
    laps = _long_run_laps()
    laps[column] = dry_value
    laps.loc[laps["Team"].eq("Team B"), column] = wet_value

    payload = build_race_practice_evidence(
        {"FP2": laps},
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
        weather="dry",
    )

    assert set(payload["teams"]) == {"Team A"}
    metrics = payload["teams"]["Team A"]["compounds"]["SOFT"]
    assert metrics["pace_comparison_status"] == "no_matched_bucket"
    assert "pace_performance" not in metrics
    assert payload["diagnostics"]["excluded_laps_by_reason"]["wet_or_rainfall"] == 10


def test_apply_race_practice_evidence_is_explicit_and_compound_specific() -> None:
    evidence = build_race_practice_evidence(
        _long_run_laps(),
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )
    soft = evidence["teams"]["Team A"]["compounds"]["SOFT"]
    soft["n_laps"] = 16
    soft["n_stints"] = 2
    soft["matched_pace_laps"] = 16
    soft["matched_pace_stints"] = 2
    driver_info = {
        "AAA": {
            "team": "Team A",
            "team_strength_by_compound": {"SOFT": 0.50, "MEDIUM": 0.60},
            "tire_deg_by_compound": {"SOFT": 0.20, "MEDIUM": 0.10},
        }
    }

    applied = apply_race_practice_evidence(driver_info, evidence)

    assert applied["AAA"]["team_strength_by_compound"]["SOFT"] > 0.50
    assert applied["AAA"]["team_strength_by_compound"]["MEDIUM"] == 0.60
    assert 0.10 < applied["AAA"]["tire_deg_by_compound"]["SOFT"] < 0.20
    assert "race_practice_evidence_applied" in applied["AAA"]


def test_unmatched_stints_may_update_degradation_but_not_team_pace() -> None:
    evidence = build_race_practice_evidence(
        {
            "FP1": _team_laps("Team A"),
            "FP2": _team_laps("Team B"),
        },
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )
    soft = evidence["teams"]["Team A"]["compounds"]["SOFT"]
    soft["n_laps"] = 16
    soft["n_stints"] = 2
    driver_info = {
        "AAA": {
            "team": "Team A",
            "team_strength_by_compound": {"SOFT": 0.50},
            "tire_deg_by_compound": {"SOFT": 0.20},
        }
    }

    applied = apply_race_practice_evidence(driver_info, evidence)

    assert applied["AAA"]["team_strength_by_compound"]["SOFT"] == 0.50
    assert 0.10 < applied["AAA"]["tire_deg_by_compound"]["SOFT"] < 0.20


def test_apply_race_practice_evidence_ignores_one_sparse_stint() -> None:
    evidence = build_race_practice_evidence(
        _long_run_laps(),
        year=2026,
        event_name="Example Grand Prix",
        checkpoint="FP2",
    )
    driver_info = {
        "AAA": {
            "team": "Team A",
            "team_strength_by_compound": {"SOFT": 0.50},
            "tire_deg_by_compound": {"SOFT": 0.20},
        }
    }

    applied = apply_race_practice_evidence(driver_info, evidence)

    assert applied["AAA"]["team_strength_by_compound"]["SOFT"] == 0.50
    assert applied["AAA"]["tire_deg_by_compound"]["SOFT"] == 0.20
    assert "race_practice_evidence_applied" not in applied["AAA"]
