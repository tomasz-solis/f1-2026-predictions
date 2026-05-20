"""Tests for held-out team-strength refit candidate diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.team_strength_refit_candidate_test import (
    CURRENT_CANDIDATE,
    SCALE_ONLY_CANDIDATE,
    UNCERTAINTY_ONLY_CANDIDATE,
    build_team_strength_refit_candidate_test,
    format_team_strength_refit_candidate_test_markdown,
    team_strength_refit_test_artifact_key,
)


def _write_json(path: Path, payload: dict) -> Path:
    """Write JSON test data and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mapping_payload() -> dict:
    """Return a frozen mapping that under-scales the synthetic 2026 rows."""
    return {
        "policy": "same_session_construct",
        "mappings": {
            session_kind: {
                "session_kind": session_kind,
                "policy": "same_session_construct",
                "intercept_s": 0.0,
                "slope_s_per_unit": 1.0,
                "training_years": [2022, 2023, 2024, 2025],
            }
            for session_kind in ("race", "qualifying")
        },
    }


def _observations() -> pd.DataFrame:
    """Return rows where a 2026 slope refit should beat the frozen mapping."""
    rows = []
    for session_kind in ("race", "qualifying"):
        for index, race_name in enumerate(("Race 1", "Race 2", "Race 3", "Race 4")):
            bump = index * 0.02
            rows.extend(
                [
                    {
                        "year": 2026,
                        "race_name": race_name,
                        "session_name": "Race" if session_kind == "race" else "Qualifying",
                        "session_kind": session_kind,
                        "team": "Team A",
                        "driver_code": "D1",
                        "observed_driver_to_field_s": 1.0 + bump,
                        "driver_rating_mu_s": 0.0,
                        "team_strength_same_session": 1.0,
                    },
                    {
                        "year": 2026,
                        "race_name": race_name,
                        "session_name": "Race" if session_kind == "race" else "Qualifying",
                        "session_kind": session_kind,
                        "team": "Team B",
                        "driver_code": "D2",
                        "observed_driver_to_field_s": -1.0 - bump,
                        "driver_rating_mu_s": 0.0,
                        "team_strength_same_session": 0.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_refit_candidate_test_compares_current_and_scale_candidate(tmp_path: Path) -> None:
    """Scale candidates should improve MSE on synthetic under-scaled data."""
    mapping_path = _write_json(tmp_path / "mapping.json", _mapping_payload())
    prior_path = _write_json(tmp_path / "prior.json", {"race_network": {}, "quali_network": {}})
    observations_path = tmp_path / "observations.csv"
    _observations().to_csv(observations_path, index=False)

    artifact = build_team_strength_refit_candidate_test(
        year=2026,
        mapping_artifact_path=mapping_path,
        prior_artifact_path=prior_path,
        observations_path=observations_path,
        raw_matched_laps_path=None,
    )

    aggregate = {row["candidate"]: row for row in artifact["aggregate"]}
    assert artifact["status"] == "measured"
    assert (
        aggregate[SCALE_ONLY_CANDIDATE]["weighted_mse_s2"]
        < aggregate[CURRENT_CANDIDATE]["weighted_mse_s2"]
    )
    assert (
        aggregate[UNCERTAINTY_ONLY_CANDIDATE]["weighted_mse_s2"]
        == aggregate[CURRENT_CANDIDATE]["weighted_mse_s2"]
    )
    assert artifact["decision_assessment"]["state"] == (
        "refit_candidate_worth_full_prediction_replay"
    )

    markdown = format_team_strength_refit_candidate_test_markdown(artifact)
    assert "# Team-Strength Refit Candidate Test" in markdown
    assert "uncertainty-only keeps identical medians" in markdown


def test_refit_candidate_test_key_is_stable() -> None:
    """The artifact key should remain stable across file and DB storage."""
    assert team_strength_refit_test_artifact_key(2026) == (
        "2026::team_strength_refit_candidate_test"
    )
