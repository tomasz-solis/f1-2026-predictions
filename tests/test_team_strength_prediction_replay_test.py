"""Tests for full prediction replay comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.team_strength_prediction_replay_test import (
    build_race_scale_only_mapping_payload,
    compare_prediction_replay_summaries,
    format_team_strength_prediction_replay_test_markdown,
    team_strength_prediction_replay_artifact_key,
)
from src.models.team_strength_mapping import LinearTeamStrengthMapping


def _write_json(path: Path, payload: dict) -> Path:
    """Write JSON test data and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_race_scale_only_mapping_payload_updates_race_only() -> None:
    """The scale candidate should change race slope and leave qualifying intact."""
    frozen_payload = {
        "policy": "same_session_construct",
        "mappings": {
            "race": {"intercept_s": 0.0, "slope_s_per_unit": 1.0},
            "qualifying": {"intercept_s": 0.0, "slope_s_per_unit": 2.0},
        },
    }
    observations = pd.DataFrame(
        [
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Race",
                "session_kind": "race",
                "team": "Team A",
                "driver_code": "D1",
                "observed_driver_to_field_s": 1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 1.0,
            },
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Race",
                "session_kind": "race",
                "team": "Team B",
                "driver_code": "D2",
                "observed_driver_to_field_s": -1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 0.0,
            },
            {
                "year": 2026,
                "race_name": "Race 2",
                "session_name": "Race",
                "session_kind": "race",
                "team": "Team A",
                "driver_code": "D1",
                "observed_driver_to_field_s": 1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 1.0,
            },
        ]
    )

    payload = build_race_scale_only_mapping_payload(
        frozen_mapping_payload=frozen_payload,
        frozen_race_mapping=LinearTeamStrengthMapping(
            session_kind="race",
            policy="same_session_construct",
            intercept_s=0.0,
            slope_s_per_unit=1.0,
            training_years=(2022,),
        ),
        observations=observations,
        mapping_policy="same_session_construct",
        year=2026,
        holdout_race="Race 2",
    )

    assert payload["mappings"]["race"]["slope_s_per_unit"] == pytest.approx(2.0)
    assert payload["mappings"]["qualifying"]["slope_s_per_unit"] == 2.0


def test_compare_prediction_replay_summaries_computes_position_mse(tmp_path: Path) -> None:
    """Candidate replay comparison should use paired target position MSE."""
    current_prediction = _write_json(
        tmp_path / "current_prediction.json",
        {
            "targets": {
                "grand_prix_race": {
                    "eligible_at_save": True,
                    "predicted_order": [
                        {"driver": "A", "position": 2},
                        {"driver": "B", "position": 1},
                    ],
                }
            },
            "actuals": {
                "targets": {
                    "grand_prix_race": [
                        {"driver": "A", "position": 1},
                        {"driver": "B", "position": 2},
                    ]
                }
            },
        },
    )
    candidate_prediction = _write_json(
        tmp_path / "candidate_prediction.json",
        {
            "targets": {
                "grand_prix_race": {
                    "eligible_at_save": True,
                    "predicted_order": [
                        {"driver": "A", "position": 1},
                        {"driver": "B", "position": 2},
                    ],
                }
            },
            "actuals": {
                "targets": {
                    "grand_prix_race": [
                        {"driver": "A", "position": 1},
                        {"driver": "B", "position": 2},
                    ]
                }
            },
        },
    )
    current_summary = _write_json(
        tmp_path / "current_summary.json",
        {
            "checkpoints": [
                {
                    "race_name": "Race 1",
                    "checkpoint_session": "PRE",
                    "prediction_path": str(current_prediction),
                }
            ]
        },
    )
    candidate_summary = _write_json(
        tmp_path / "candidate_summary.json",
        {
            "checkpoints": [
                {
                    "race_name": "Race 1",
                    "checkpoint_session": "PRE",
                    "prediction_path": str(candidate_prediction),
                }
            ]
        },
    )

    artifact = compare_prediction_replay_summaries(
        year=2026,
        current_summary_path=current_summary,
        candidate_summaries={"Race 1": candidate_summary},
        candidate_name="candidate",
    )

    assert artifact["race_target_aggregate"][0]["candidate_mse"] == 0.0
    assert artifact["race_target_aggregate"][0]["current_mse"] == 1.0
    assert artifact["decision_assessment"]["state"] == (
        "supports_race_only_prediction_replay_candidate"
    )
    assert "# Team-Strength Prediction Replay Test" in (
        format_team_strength_prediction_replay_test_markdown(artifact)
    )


def test_team_strength_prediction_replay_artifact_key_is_stable() -> None:
    """The replay-test artifact key should remain stable."""
    assert team_strength_prediction_replay_artifact_key(2026) == (
        "2026::team_strength_prediction_replay_test"
    )
