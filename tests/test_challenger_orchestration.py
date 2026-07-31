"""Tests for the research-only challenger orchestration boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.analysis.challenger_orchestration import (
    fetch_official_race_grid_for_research,
    run_challenger_pipeline,
)
from src.analysis.challenger_release import classify_shadow_registration
from src.persistence.research_sidecar import ResearchSidecarStore
from tests.challenger_test_helpers import strict_manifest


class _Config:
    def __init__(self, variant: str) -> None:
        self.variant = variant

    def get(self, key: str, default: Any = None) -> Any:
        if key == "baseline_predictor.model_variant":
            return self.variant
        return default


class _Predictor:
    def __init__(self, variant: str, *, return_scenarios: bool = True) -> None:
        self.config = _Config(variant)
        self.variant = variant
        self.return_scenarios = return_scenarios
        self.q1_manifest = strict_manifest(
            variant if variant != "does_not_exist" else "q1_qualifying_practice"
        )
        self.qualifying_calls: list[dict[str, Any]] = []
        self.race_calls: list[dict[str, Any]] = []

    def predict_qualifying(self, **kwargs: Any) -> Mapping[str, Any]:
        self.qualifying_calls.append(kwargs)
        result: dict[str, Any] = {
            "grid": [
                {"driver": "A", "team": "One", "position": 1},
                {"driver": "B", "team": "Two", "position": 2},
            ],
            "grid_source_detail": "predicted_marginal_fallback",
            "grid_uncertainty_mode": "marginal_fallback",
        }
        if kwargs["include_grid_scenarios"] and self.return_scenarios:
            result["grid_scenarios"] = [["A", "B"], ["B", "A"]]
        if kwargs["include_challenger_evidence"]:
            result["qualifying_practice_evidence"] = {
                "FP1": {"drivers": {"A": {"clean_lap_count": 4}}}
            }
            result["qualifying_practice_challenger"] = {
                "used": True,
                "variant": self.variant,
                "artifact_launch": {
                    "candidate_id": self.q1_manifest["candidate_id"],
                    "variant_id": self.variant,
                    "manifest_digest": f"sha256:{self.q1_manifest['manifest_sha256']}",
                    "bundle_digest": f"sha256:{'b' * 64}",
                    "launch_digest": f"sha256:{'c' * 64}",
                },
            }
        return result

    def predict_race(self, **kwargs: Any) -> Mapping[str, Any]:
        self.race_calls.append(kwargs)
        return {
            "finish_order": [{"driver": "A", "position": 1}],
            "grid_scenario_count": len(kwargs.get("grid_scenarios") or []),
            "nested": {
                "grid_scenarios": [["must", "not", "escape"]],
                "race_practice_evidence": {"must": "not escape"},
            },
        }


def _manifest(variant: str) -> dict[str, Any]:
    return strict_manifest(variant)


def _contains_private_key(value: Any) -> bool:
    private = {
        "grid_scenarios",
        "qualifying_practice_evidence",
        "race_practice_evidence",
    }
    if isinstance(value, Mapping):
        return bool(private.intersection(value)) or any(
            _contains_private_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_private_key(item) for item in value)
    return False


def test_full_pipeline_handoffs_private_inputs_scrubs_output_and_persists(tmp_path: Path) -> None:
    predictor = _Predictor("full_challenger")
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    race_evidence = {"drivers": {"A": {"long_run_lap_count": 12}}}

    result = run_challenger_pipeline(
        predictor,
        variant_id="full_challenger",
        qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
        race_kwargs={"weather": "dry", "n_simulations": 20},
        race_practice_evidence=race_evidence,
        manifest=_manifest("full_challenger"),
        sidecar_store=store,
    )

    assert predictor.qualifying_calls == [
        {
            "year": 2026,
            "race_name": "Test Grand Prix",
            "include_grid_scenarios": True,
            "include_challenger_evidence": True,
        }
    ]
    race_call = predictor.race_calls[0]
    assert race_call["grid_scenarios"] == [["A", "B"], ["B", "A"]]
    assert race_call["grid_source_detail"] == "predicted_joint"
    assert race_call["race_practice_evidence"] == race_evidence
    assert not _contains_private_key(result["qualifying"])
    assert not _contains_private_key(result["race"])
    assert result["production_activation"] is False
    assert result["qualifying"]["grid_scenario_count"] == 2
    assert result["qualifying"]["grid_uncertainty_mode"] == "joint_scenarios"
    assert result["race"]["race_practice_evidence_count"] == 1

    sidecars = result["research_sidecars"]
    assert set(sidecars) == {
        "manifest",
        "qualifying_practice_evidence",
        "grid_scenarios",
        "race_practice_evidence",
    }
    assert all(reference["digest"].startswith("sha256:") for reference in sidecars.values())
    scenario_sidecar = json.loads(Path(sidecars["grid_scenarios"]["path"]).read_text())
    assert scenario_sidecar["payload"]["scenarios"] == [["A", "B"], ["B", "A"]]


def test_q1_only_requests_evidence_not_grid_scenarios_or_race() -> None:
    predictor = _Predictor("q1_qualifying_practice")

    result = run_challenger_pipeline(
        predictor,
        variant_id="q1_qualifying_practice",
        qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
    )

    assert predictor.qualifying_calls[0]["include_challenger_evidence"] is True
    assert predictor.qualifying_calls[0]["include_grid_scenarios"] is False
    assert predictor.race_calls == []
    assert result["race"] is None
    assert result["qualifying"]["qualifying_practice_evidence_session_count"] == 1
    assert not _contains_private_key(result["qualifying"])


def test_preregistered_pipeline_can_freeze_both_scrubbed_forecasts(tmp_path: Path) -> None:
    predictor = _Predictor("q1_qualifying_practice")
    manifest = _manifest("q1_qualifying_practice")
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)

    result = run_challenger_pipeline(
        predictor,
        variant_id="q1_qualifying_practice",
        qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
        manifest=manifest,
        sidecar_store=store,
        champion_prediction={
            "model_variant": "champion",
            "qualifying": {"grid": [{"driver": "A", "position": 1}]},
            "race": None,
        },
        forecast_year=2026,
        forecast_event_name="Test Grand Prix",
        forecast_frozen_at="2026-07-18T11:30:00Z",
    )

    reference = result["research_sidecars"]["frozen_forecasts"]
    registration = classify_shadow_registration(
        manifest,
        qualifying_start_at="2026-07-18T14:00:00Z",
        frozen_forecast_reference=reference,
    )
    assert registration["classification"] == "preregistered_shadow"


def test_q1_freeze_rejects_launch_bound_to_another_manifest(tmp_path: Path) -> None:
    predictor = _Predictor("q1_qualifying_practice")
    predictor.q1_manifest["manifest_sha256"] = "d" * 64
    manifest = _manifest("q1_qualifying_practice")

    with pytest.raises(ValueError, match="manifest digest does not match"):
        run_challenger_pipeline(
            predictor,
            variant_id="q1_qualifying_practice",
            qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
            manifest=manifest,
            sidecar_store=ResearchSidecarStore(
                tmp_path / "research",
                repo_root=tmp_path,
            ),
        )


def test_r1_fails_closed_when_qualifying_does_not_return_joint_scenarios() -> None:
    predictor = _Predictor("r1_joint_grid", return_scenarios=False)

    with pytest.raises(ValueError, match="R1 requires complete joint grid_scenarios"):
        run_challenger_pipeline(
            predictor,
            variant_id="r1_joint_grid",
            qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
            race_kwargs={"weather": "dry"},
        )

    assert predictor.race_calls == []


@pytest.mark.parametrize("variant", ["champion", "does_not_exist"])
def test_invalid_or_champion_variant_is_rejected(variant: str) -> None:
    predictor = _Predictor(variant)

    with pytest.raises(ValueError, match="variant|non-champion"):
        run_challenger_pipeline(
            predictor,
            variant_id=variant,
            qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
        )


def test_variant_config_mismatch_and_private_argument_overrides_are_rejected() -> None:
    mismatch = _Predictor("q1_qualifying_practice")
    with pytest.raises(ValueError, match="does not match"):
        run_challenger_pipeline(
            mismatch,
            variant_id="q0_driver_state",
            qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
        )

    predictor = _Predictor("q1_qualifying_practice")
    with pytest.raises(ValueError, match="reserved arguments"):
        run_challenger_pipeline(
            predictor,
            variant_id="q1_qualifying_practice",
            qualifying_kwargs={"include_challenger_evidence": False},
        )


def test_race_evidence_and_sidecar_arguments_are_variant_scoped() -> None:
    predictor = _Predictor("q1_qualifying_practice")
    with pytest.raises(ValueError, match="variant containing R0"):
        run_challenger_pipeline(
            predictor,
            variant_id="q1_qualifying_practice",
            qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
            race_practice_evidence={"drivers": {}},
        )

    with pytest.raises(ValueError, match="supplied together"):
        run_challenger_pipeline(
            predictor,
            variant_id="q1_qualifying_practice",
            qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
            manifest=_manifest("q1_qualifying_practice"),
        )


def test_official_starting_grid_takes_precedence_and_disables_joint_noise() -> None:
    predictor = _Predictor("r1_joint_grid")
    official_grid = [
        {
            "driver": "B",
            "team": "Two",
            "position": 1,
            "start_type": "grid",
        },
        {
            "driver": "A",
            "team": "One",
            "position": 2,
            "start_type": "pit_lane",
        },
    ]

    run_challenger_pipeline(
        predictor,
        variant_id="r1_joint_grid",
        qualifying_kwargs={"year": 2026, "race_name": "Test Grand Prix"},
        race_kwargs={"weather": "dry"},
        race_grid=official_grid,
    )

    race_call = predictor.race_calls[0]
    assert race_call["qualifying_grid"] == official_grid
    assert race_call["grid_source_detail"] == "actual_starting_grid"
    assert race_call["grid_scenarios"] is None


def test_research_official_grid_fetcher_returns_explicit_provenance() -> None:
    official_grid = [
        {"driver": "A", "team": "One", "position": 1, "start_type": "grid"},
        {"driver": "B", "team": "Two", "position": 2, "start_type": "pit_lane"},
    ]
    classification = [
        {"driver": "A", "team": "One", "position": 1},
        {"driver": "B", "team": "Two", "position": 2},
    ]
    with patch(
        "src.analysis.challenger_orchestration.fetch_official_starting_grid",
        return_value=official_grid,
    ) as fetcher:
        resolved = fetch_official_race_grid_for_research(
            2026,
            "Test Grand Prix",
            qualifying_classification=classification,
        )

    assert resolved == (official_grid, "actual_starting_grid")
    fetcher.assert_called_once_with(
        2026,
        "Test Grand Prix",
        session_name="R",
        qualifying_classification=classification,
    )
