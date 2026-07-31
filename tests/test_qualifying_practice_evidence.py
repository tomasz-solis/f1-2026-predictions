from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.qualifying_practice_evidence import (
    QUALIFYING_PRACTICE_EVIDENCE_SCHEMA,
    QUALIFYING_PRACTICE_EVIDENCE_VERSION,
    PracticeNormalizationPrior,
    build_qualifying_practice_evidence,
    fit_practice_normalization,
)


def _base_laps(rows: list[dict]) -> pd.DataFrame:
    laps = pd.DataFrame(rows)
    defaults = {
        "IsAccurate": True,
        "Deleted": False,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "TrackStatus": "1",
        "Rainfall": False,
    }
    for column, value in defaults.items():
        if column not in laps:
            laps[column] = value
    return laps


def _row(
    *,
    driver: str,
    team: str,
    lap_number: int,
    stint: int,
    compound: str,
    lap_s: float,
    tyre_age: float,
    sector_s: tuple[float, float, float] | None = None,
) -> dict:
    sectors = sector_s or (lap_s / 3.0, lap_s / 3.0, lap_s / 3.0)
    return {
        "Driver": driver,
        "Team": team,
        "LapNumber": lap_number,
        "Stint": stint,
        "Compound": compound,
        "LapTime": pd.Timedelta(seconds=lap_s),
        "TyreLife": tyre_age,
        "Time": pd.Timedelta(seconds=lap_number * 100),
        "Sector1Time": pd.Timedelta(seconds=sectors[0]),
        "Sector2Time": pd.Timedelta(seconds=sectors[1]),
        "Sector3Time": pd.Timedelta(seconds=sectors[2]),
    }


def test_dry_clean_filter_records_exclusive_rejection_reasons():
    rows = [
        _row(
            driver="AAA",
            team="Team A",
            lap_number=index + 1,
            stint=index + 1,
            compound="SOFT",
            lap_s=90.0 if index < 8 else 120.0,
            tyre_age=1,
        )
        for index in range(9)
    ]
    laps = _base_laps(rows)
    laps.loc[1, "IsAccurate"] = False
    laps.loc[2, "Deleted"] = True
    laps.loc[3, "PitInTime"] = pd.Timestamp("2026-01-01T00:00:01")
    laps.loc[4, "TrackStatus"] = "2"
    laps.loc[5, "Compound"] = "INTERMEDIATE"
    laps.loc[6, "Rainfall"] = True
    laps["Aborted"] = False
    laps.loc[7, "Aborted"] = True

    payload = build_qualifying_practice_evidence(
        laps,
        session_code="FP2",
        session_is_dry=True,
    )

    assert payload["exclusions"]["accepted_clean_laps"] == 1
    assert payload["exclusions"]["by_reason"] == {
        "aborted": 1,
        "deleted": 1,
        "implausible_or_aborted": 1,
        "inaccurate": 1,
        "non_green": 1,
        "pit_lap": 1,
        "wet_compound": 1,
        "wet_lap": 1,
    }
    assert payload["drivers"]["AAA"]["features"]["source_run_class"] == "quali_sim"


def test_unknown_or_wet_session_fails_closed_before_extracting_laps():
    laps = _base_laps(
        [
            _row(
                driver="AAA",
                team="Team A",
                lap_number=1,
                stint=1,
                compound="SOFT",
                lap_s=90.0,
                tyre_age=1,
            )
        ]
    )

    payload = build_qualifying_practice_evidence(
        laps,
        session_code="FP1",
        session_is_dry=None,
    )

    assert payload["eligibility"] == {
        "eligible": False,
        "fallback_reasons": ["dry_session_not_confirmed"],
    }
    assert payload["drivers"] == {}

    confirmed = build_qualifying_practice_evidence(
        laps,
        session_code="FP1",
        session_is_dry=np.bool_(True),
    )
    assert confirmed["eligibility"]["eligible"] is True


def test_run_classification_uses_five_and_eight_lap_boundaries():
    rows: list[dict] = []
    lap_number = 1
    for stint, compound, count in ((1, "SOFT", 4), (2, "MEDIUM", 8), (3, "HARD", 6)):
        for stint_lap in range(count):
            rows.append(
                _row(
                    driver="AAA",
                    team="Team A",
                    lap_number=lap_number,
                    stint=stint,
                    compound=compound,
                    lap_s=90.0 + (stint * 0.5) + (stint_lap * 0.05),
                    tyre_age=stint_lap + 1,
                )
            )
            lap_number += 1

    payload = build_qualifying_practice_evidence(
        _base_laps(rows),
        session_code="FP2",
        session_is_dry=True,
    )

    classifications = {
        (run["compound"], run["clean_consecutive_laps"]): run["classification"]
        for run in payload["runs"]
    }
    assert classifications == {
        ("SOFT", 4): "quali_sim",
        ("MEDIUM", 8): "race_sim",
        ("HARD", 6): "other",
    }
    assert payload["drivers"]["AAA"]["counts"]["runs"] == {
        "quali_sim": 1,
        "race_sim": 1,
        "other": 1,
    }
    assert payload["drivers"]["AAA"]["features"]["source_run_class"] == "quali_sim"


def test_quali_sim_requires_every_lap_in_run_to_be_on_fresh_tyres():
    rows = [
        _row(
            driver="AAA",
            team="Team A",
            lap_number=index + 1,
            stint=1,
            compound="SOFT",
            lap_s=90.0 + index * 0.05,
            tyre_age=tyre_age,
        )
        for index, tyre_age in enumerate((1, 2, 6))
    ]

    payload = build_qualifying_practice_evidence(
        _base_laps(rows),
        session_code="FP3",
        session_is_dry=True,
    )

    assert payload["runs"][0]["classification"] == "other"
    assert payload["runs"][0]["full_run_tyre_age_eligible"] is False


def test_non_green_gap_breaks_consecutive_race_run():
    rows = [
        _row(
            driver="AAA",
            team="Team A",
            lap_number=index + 1,
            stint=1,
            compound="MEDIUM",
            lap_s=91.0 + index * 0.05,
            tyre_age=index + 1,
        )
        for index in range(9)
    ]
    laps = _base_laps(rows)
    laps.loc[4, "TrackStatus"] = "2"

    payload = build_qualifying_practice_evidence(
        laps,
        session_code="FP1",
        session_is_dry=True,
    )

    assert sorted(run["clean_consecutive_laps"] for run in payload["runs"]) == [4, 4]
    assert all(run["classification"] != "race_sim" for run in payload["runs"])


def test_theoretical_sectors_never_cross_compatible_run_buckets():
    rows = [
        _row(
            driver="AAA",
            team="Team A",
            lap_number=1,
            stint=1,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=1,
            sector_s=(20.0, 35.0, 35.0),
        ),
        _row(
            driver="AAA",
            team="Team A",
            lap_number=2,
            stint=1,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=1,
            sector_s=(35.0, 20.0, 35.0),
        ),
        _row(
            driver="AAA",
            team="Team A",
            lap_number=3,
            stint=2,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=1,
            sector_s=(35.0, 35.0, 20.0),
        ),
        _row(
            driver="AAA",
            team="Team A",
            lap_number=4,
            stint=2,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=1,
            sector_s=(35.0, 35.0, 20.0),
        ),
    ]
    # Extend the session so the first two laps share an evolution bucket.  A global
    # Frankenstein lap would be 60s; the compatible first-run bucket is 75s.
    rows.extend(
        _row(
            driver="BBB",
            team="Team B",
            lap_number=lap_number,
            stint=1,
            compound="MEDIUM",
            lap_s=95.0,
            tyre_age=lap_number,
        )
        for lap_number in range(5, 13)
    )

    payload = build_qualifying_practice_evidence(
        _base_laps(rows),
        session_code="FP3",
        session_is_dry=True,
    )

    features = payload["drivers"]["AAA"]["features"]
    assert features["theoretical_adjusted_lap_s"] == pytest.approx(75.0)
    assert features["theoretical_adjusted_lap_s"] > 60.0
    assert features["execution_loss_s"] == pytest.approx(15.0)


def test_run_feature_candidates_are_deterministic_and_never_cross_runs():
    rows = [
        _row(
            driver="AAA",
            team="Team A",
            lap_number=1,
            stint=1,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=1,
            sector_s=(20.0, 35.0, 35.0),
        ),
        _row(
            driver="AAA",
            team="Team A",
            lap_number=2,
            stint=1,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=2,
            sector_s=(35.0, 20.0, 35.0),
        ),
        _row(
            driver="AAA",
            team="Team A",
            lap_number=3,
            stint=2,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=1,
            sector_s=(35.0, 35.0, 20.0),
        ),
        _row(
            driver="AAA",
            team="Team A",
            lap_number=4,
            stint=2,
            compound="SOFT",
            lap_s=90.0,
            tyre_age=2,
            sector_s=(35.0, 35.0, 20.0),
        ),
    ]
    rows.extend(
        _row(
            driver="BBB",
            team="Team B",
            lap_number=lap_number,
            stint=1,
            compound="MEDIUM",
            lap_s=95.0,
            tyre_age=lap_number - 4,
        )
        for lap_number in range(5, 13)
    )
    laps = _base_laps(rows)

    first = build_qualifying_practice_evidence(
        laps,
        session_code="FP3",
        session_is_dry=True,
    )
    second = build_qualifying_practice_evidence(
        laps,
        session_code="FP3",
        session_is_dry=True,
    )

    candidates = first["drivers"]["AAA"]["run_feature_candidates"]
    assert candidates == second["drivers"]["AAA"]["run_feature_candidates"]
    assert [row["run_id"] for row in candidates] == sorted(row["run_id"] for row in candidates)
    assert len(candidates) == 2
    assert candidates[0]["theoretical_adjusted_lap_s"] == pytest.approx(75.0)
    assert candidates[1]["theoretical_adjusted_lap_s"] == pytest.approx(90.0)
    assert all(row["theoretical_adjusted_lap_s"] > 60.0 for row in candidates)
    assert first["parameters"]["run_feature_candidate_scope"] == "single_compatible_run"


def _normalization_comparisons() -> pd.DataFrame:
    compound_effects = {"SOFT": 0.0, "MEDIUM": 0.6, "HARD": 1.1}
    rows: list[dict] = []
    pair_types = [("MEDIUM", "SOFT"), ("HARD", "SOFT"), ("HARD", "MEDIUM")]
    age_deltas = (-2.0, -1.0, 1.0, 2.0)
    evolution_deltas = (-0.5, 0.5)
    index = 0
    for compound_a, compound_b in pair_types:
        for age_delta in age_deltas:
            for evolution_delta in evolution_deltas:
                delta_s = (
                    compound_effects[compound_a]
                    - compound_effects[compound_b]
                    + (0.10 * age_delta)
                    + (-0.40 * evolution_delta)
                )
                rows.append(
                    {
                        "driver": f"D{index % 4}",
                        "team": f"T{index % 3}",
                        "lap_time_a_s": 90.0 + delta_s,
                        "lap_time_b_s": 90.0,
                        "compound_a": compound_a,
                        "compound_b": compound_b,
                        "tyre_age_a": 3.0 + age_delta / 2.0,
                        "tyre_age_b": 3.0 - age_delta / 2.0,
                        "evolution_a": 0.5 + evolution_delta / 2.0,
                        "evolution_b": 0.5 - evolution_delta / 2.0,
                    }
                )
                index += 1
    return pd.DataFrame(rows)


def test_sparse_normalization_uses_explicit_track_class_prior():
    prior = PracticeNormalizationPrior(
        compound_effect_s={"SOFT": 0.0, "MEDIUM": 0.7, "HARD": 1.2},
        tyre_age_effect_s_per_lap=0.08,
        evolution_effect_s_per_unit=-0.3,
        uncertainty_s=0.25,
        source="high_speed_track_prior_v2",
    )

    fitted = fit_practice_normalization(_normalization_comparisons().iloc[:7], prior=prior)

    assert fitted.provenance == "track_class_prior"
    assert fitted.compound_effect_s["MEDIUM"] == pytest.approx(0.7)
    assert fitted.tyre_age_effect_s_per_lap == pytest.approx(0.08)
    assert fitted.prior_source == "high_speed_track_prior_v2"
    assert "insufficient_comparisons:7<8" in fitted.fallback_reasons


def test_sufficient_normalization_is_empirical_and_shrunk_without_fixed_delta():
    prior = PracticeNormalizationPrior(
        compound_effect_s={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        uncertainty_s=0.5,
        source="neutral_track_class_prior",
    )

    fitted = fit_practice_normalization(_normalization_comparisons(), prior=prior)

    assert fitted.provenance == "empirical_shrunk"
    assert fitted.comparison_count == 24
    assert fitted.driver_count == 4
    assert fitted.team_count == 3
    assert 0.0 < fitted.compound_effect_s["MEDIUM"] < 0.6
    assert fitted.compound_effect_s["HARD"] > fitted.compound_effect_s["MEDIUM"]
    assert 0.0 < fitted.tyre_age_effect_s_per_lap < 0.10
    assert -0.40 < fitted.evolution_effect_s_per_unit < 0.0
    assert fitted.coefficient_provenance["compound:MEDIUM"] == "empirical_shrunk"


def test_sidecar_is_versioned_json_safe_and_quantiles_favor_faster_driver():
    rows: list[dict] = []
    for driver, team, base in (("AAA", "Team A", 90.0), ("BBB", "Team B", 91.0)):
        for index in range(3):
            rows.append(
                _row(
                    driver=driver,
                    team=team,
                    lap_number=index + 1,
                    stint=1,
                    compound="SOFT",
                    lap_s=base + (index * 0.1),
                    tyre_age=index + 1,
                )
            )

    payload = build_qualifying_practice_evidence(
        _base_laps(rows),
        session_code="FP2",
        session_is_dry=True,
        track_name="Example GP",
        track_class="high_speed",
    )

    assert payload["artifact_type"] == QUALIFYING_PRACTICE_EVIDENCE_SCHEMA
    assert payload["schema_version"] == QUALIFYING_PRACTICE_EVIDENCE_VERSION
    assert payload["drivers"]["AAA"]["feature_quantiles"]["best_adjusted_lap_s"] == 1.0
    assert payload["drivers"]["BBB"]["feature_quantiles"]["best_adjusted_lap_s"] == 0.0
    assert np.isfinite(payload["drivers"]["AAA"]["features"]["measurement_uncertainty_s"])
    json.dumps(payload, allow_nan=False, sort_keys=True)
