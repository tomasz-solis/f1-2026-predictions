"""Tests for team-strength construct-row audit diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.team_strength_construct_audit import (
    build_construct_prediction_rows,
    build_team_strength_construct_audit,
    format_team_strength_construct_audit_markdown,
    team_strength_construct_audit_artifact_key,
)
from src.models.team_strength_mapping import LinearTeamStrengthMapping


def _write_json(path: Path, payload: dict) -> Path:
    """Write JSON test data and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mapping_payload() -> dict:
    """Return a compact mapping artifact with historical reference folds."""
    folds = []
    for session_kind in ("race", "qualifying"):
        for holdout_year, slope in ((2024, 0.9), (2025, 1.1)):
            folds.append(
                {
                    "session_kind": session_kind,
                    "holdout_year": holdout_year,
                    "n_rows": 4,
                    "prediction_slope": slope,
                    "r_squared": 0.8,
                    "rmse_s": 0.2,
                }
            )
    return {
        "policy": "same_session_construct",
        "validation": {"folds": folds},
        "mappings": {
            session_kind: {
                "session_kind": session_kind,
                "policy": "same_session_construct",
                "intercept_s": 0.0,
                "slope_s_per_unit": 2.0,
                "training_years": [2022, 2023, 2024, 2025],
            }
            for session_kind in ("race", "qualifying")
        },
    }


def _observations() -> pd.DataFrame:
    """Return synthetic construct rows with a clear 2026 scale mismatch."""
    rows = []
    for session_kind, scale in (("race", 2.0), ("qualifying", 1.4)):
        for race_name, bump in (("Race 1", 0.0), ("Race 2", 0.2)):
            rows.extend(
                [
                    {
                        "year": 2026,
                        "race_name": race_name,
                        "session_name": "Race" if session_kind == "race" else "Qualifying",
                        "session_kind": session_kind,
                        "team": "Team A",
                        "driver_code": "D1",
                        "n_construct_laps": 8,
                        "observed_driver_to_field_s": scale + bump,
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
                        "n_construct_laps": 8,
                        "observed_driver_to_field_s": -scale - bump,
                        "driver_rating_mu_s": 0.0,
                        "team_strength_same_session": 0.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_construct_prediction_rows_attach_mapping_residuals() -> None:
    """Prediction rows should expose team, driver, and residual components."""
    observations = _observations()
    rows = build_construct_prediction_rows(
        observations=observations,
        mappings={
            "race": LinearTeamStrengthMapping(
                session_kind="race",
                policy="same_session_construct",
                intercept_s=0.0,
                slope_s_per_unit=2.0,
                training_years=(2022, 2023),
            ),
            "qualifying": LinearTeamStrengthMapping(
                session_kind="qualifying",
                policy="same_session_construct",
                intercept_s=0.0,
                slope_s_per_unit=2.0,
                training_years=(2022, 2023),
            ),
        },
        mapping_policy="same_session_construct",
        year=2026,
    )

    first = rows[rows["driver_code"].eq("D1") & rows["session_kind"].eq("race")].iloc[0]
    assert first["predicted_team_s"] == 1.0
    assert first["predicted_driver_to_field_s"] == 1.0
    assert first["residual_s"] == 1.0


def test_team_strength_construct_audit_reports_scale_and_influence(tmp_path: Path) -> None:
    """The audit should persist metrics, top residuals, and leave-one influence."""
    mapping_path = _write_json(tmp_path / "mapping.json", _mapping_payload())
    candidate_path = _write_json(tmp_path / "candidate.json", {"policy_evaluations": {}})
    prior_path = _write_json(tmp_path / "prior.json", {"race_network": {}, "quali_network": {}})
    observations_path = tmp_path / "observations.csv"
    _observations().to_csv(observations_path, index=False)

    artifact = build_team_strength_construct_audit(
        year=2026,
        mapping_artifact_path=mapping_path,
        candidate_diagnostics_path=candidate_path,
        prior_artifact_path=prior_path,
        observations_path=observations_path,
        raw_matched_laps_path=None,
        max_detail_rows=5,
    )

    race_metrics = artifact["metrics_by_session_kind"]["race"]
    assert artifact["status"] == "measured"
    assert artifact["row_count"] == 8
    assert race_metrics["prediction_slope"] > 2.0
    assert race_metrics["outside_historical_one_se_band"] is True
    assert artifact["leave_one_race"]["race"]
    assert artifact["largest_abs_residual_rows"][0]["abs_residual_s"] >= 1.0

    markdown = format_team_strength_construct_audit_markdown(artifact)
    assert "# Team-Strength Construct-Row Audit" in markdown
    assert "Team-target slope" in markdown


def test_team_strength_construct_audit_key_is_stable() -> None:
    """The artifact key should remain stable across file and DB storage."""
    assert (
        team_strength_construct_audit_artifact_key(2026)
        == "2026::team_strength_construct_row_audit"
    )
