from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.challenger_governance import DEFAULT_CONFIG_PATHS, build_challenger_manifest
from src.models.qualifying_practice_bundle import (
    build_qualifying_practice_bundle,
    build_qualifying_practice_launch_envelope,
    file_sha256,
    stable_json_sha256,
)
from src.models.qualifying_practice_challenger import (
    DEFAULT_FEATURE_COLUMNS,
    FittedQualifyingPracticeModel,
)
from src.models.qualifying_practice_evidence import PracticeNormalizationPrior
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin


class _Config:
    def __init__(self, values: dict[str, object]):
        self.values = values

    def get(self, key: str, default=None):
        return self.values.get(key, default)


class _Predictor(BaselineQualifyingMixin):
    def __init__(self, values: dict[str, object]):
        self.config = _Config(values)


_LIVE_INFERENCE_CUTOFF = datetime(2099, 1, 1, tzinfo=UTC)


def _session_laps() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for driver, base in (("AAA", 90.0), ("BBB", 90.3)):
        for index in range(3):
            lap_time = base + (index * 0.05)
            rows.append(
                {
                    "Driver": driver,
                    "Team": "Team A",
                    "LapNumber": index + 1,
                    "Stint": index + 1,
                    "Compound": "SOFT",
                    "LapTime": pd.Timedelta(seconds=lap_time),
                    "TyreLife": 1,
                    "Time": pd.Timedelta(seconds=100 + index * 100),
                    "Sector1Time": pd.Timedelta(seconds=lap_time / 3),
                    "Sector2Time": pd.Timedelta(seconds=lap_time / 3),
                    "Sector3Time": pd.Timedelta(seconds=lap_time / 3),
                    "IsAccurate": True,
                    "Deleted": False,
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TrackStatus": "1",
                    "Rainfall": False,
                }
            )
    return pd.DataFrame(rows)


def _drivers() -> list[dict]:
    return [
        {
            "driver": "AAA",
            "team": "Team A",
            "team_strength": 0.7,
            "quali_pace": 0.6,
            "skill": 0.6,
        },
        {
            "driver": "BBB",
            "team": "Team A",
            "team_strength": 0.7,
            "quali_pace": 0.6,
            "skill": 0.6,
        },
    ]


def _manifest(semantic_config_path: Path) -> dict:
    return build_challenger_manifest(
        repo_root=Path.cwd(),
        candidate_id="q1-test",
        variant_id="q1_qualifying_practice",
        feature_schema="qualifying-practice-v2",
        input_snapshot_ids=["q1-test-snapshot"],
        cutoff_at="2026-06-15T00:00:00Z",
        created_at="2026-06-16T00:00:00Z",
        simulation_counts={"qualifying": 200, "race": 200},
        config_paths=[*DEFAULT_CONFIG_PATHS, semantic_config_path],
    )


def _write_artifacts(
    tmp_path: Path,
    *,
    checkpoint: str = "FP2",
    include_normalization: bool = True,
    pace_feature: str = "best_adjusted_lap_s",
    track_class_by_event: dict[str, str] | None = None,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate_root = tmp_path / "q1-test"
    artifact_cutoff = "2026-06-01T12:00:00Z"
    artifact_generated = "2026-06-02T12:00:00Z"
    semantic_config_path = tmp_path / "q1-semantic.yaml"
    semantic_settings = {
        "model_variant": "q1_qualifying_practice",
        "candidate_id": "q1-test",
        "launch_envelope_path": str(candidate_root / "launch.json"),
        "bundle_path": str(candidate_root / "bundle.json"),
        "track_class_by_event": track_class_by_event
        if track_class_by_event is not None
        else {"2026:Example Grand Prix": "high_downforce"},
        "uncertainty_scale": 1.0,
    }
    semantic_config_path.write_text(
        json.dumps(
            {
                "artifact_type": "qualifying_practice_candidate_definition",
                "schema_version": 1,
                **semantic_settings,
            }
        ),
        encoding="utf-8",
    )
    common_metadata = {
        "candidate_id": "q1-test",
        "checkpoint": checkpoint,
        "session_kind": "main",
        "dry_only": True,
        "cutoff_timestamp": artifact_cutoff,
        "input_snapshot_id": "q1-test-snapshot",
        "input_sha256": "a" * 64,
        "replay_seeds": [17, 42, 91],
        "event_timestamp_column": "event_start_at",
        "max_input_timestamp": "2026-05-31T12:00:00Z",
    }
    normalization_paths: list[Path] = []
    normalization = PracticeNormalizationPrior(uncertainty_s=0.2).normalized()
    if include_normalization:
        normalization_path = (
            candidate_root / "normalizations" / "main" / checkpoint.lower() / "high_downforce.json"
        )
        normalization_path.parent.mkdir(parents=True)
        normalization_payload = {
            "artifact_type": "qualifying_practice_normalization",
            "schema_version": 1,
            "generated_at": artifact_generated,
            "normalization": {
                "reference_compound": normalization.reference_compound,
                "compound_effect_s": normalization.compound_effect_s,
                "tyre_age_effect_s_per_lap": normalization.tyre_age_effect_s_per_lap,
                "evolution_effect_s_per_unit": normalization.evolution_effect_s_per_unit,
                "measurement_uncertainty_s": normalization.uncertainty_s,
                "provenance": "track_class_prior",
                "prior_source": normalization.source,
                "comparison_coverage": {"comparisons": 0, "drivers": 0, "teams": 0},
                "empirical_weight": 0.0,
                "coefficient_provenance": {},
                "fallback_reasons": ["test_prior"],
            },
            "training_metadata": {**common_metadata, "track_class": "high_downforce"},
        }
        normalization_path.write_text(json.dumps(normalization_payload), encoding="utf-8")
        normalization_paths.append(normalization_path)

    coefficients = [0.0 for _ in DEFAULT_FEATURE_COLUMNS]
    coefficients[DEFAULT_FEATURE_COLUMNS.index(pace_feature)] = (
        1.0 if pace_feature == "prior_utility" else -1.0
    )
    model = FittedQualifyingPracticeModel(
        checkpoint=checkpoint,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        coefficients=tuple(coefficients),
        feature_medians=tuple(0.0 for _ in DEFAULT_FEATURE_COLUMNS),
        feature_scales=tuple(1.0 for _ in DEFAULT_FEATURE_COLUMNS),
        temperature=0.5,
        training_events=30,
        generated_at=artifact_generated,
    )
    model_path = candidate_root / "models" / "main" / f"{checkpoint.lower()}.json"
    model_path.parent.mkdir(parents=True)
    model_payload = model.to_dict()
    model_payload["training_metadata"] = common_metadata
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    manifest = _manifest(semantic_config_path)
    bundle = build_qualifying_practice_bundle(
        candidate_id="q1-test",
        variant_id="q1_qualifying_practice",
        manifest=manifest,
        bundle_directory=candidate_root,
        model_paths=[model_path],
        normalization_paths=normalization_paths,
    )
    bundle_path = candidate_root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    launch = build_qualifying_practice_launch_envelope(
        candidate_id="q1-test",
        variant_id="q1_qualifying_practice",
        manifest=manifest,
        bundle_path=bundle_path,
        launch_directory=candidate_root,
        semantic_config_path=semantic_config_path,
        repo_root=Path.cwd(),
    )
    launch_path = candidate_root / "launch.json"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    return {
        "baseline_predictor.model_variant": "q1_qualifying_practice",
        "baseline_predictor.qualifying.practice_challenger.launch_envelope_path": str(launch_path),
    }


def _run_q1(
    predictor: _Predictor,
    *,
    session_kind: str = "main",
    track_laps: bool = True,
) -> tuple[dict[str, list[int]] | None, dict, dict]:
    return predictor._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()} if track_laps else {},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind=session_kind,
        inference_cutoff=_LIVE_INFERENCE_CUTOFF,
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
    )


def _launch_and_bundle_paths(config: dict[str, object]) -> tuple[Path, Path]:
    launch_path = Path(
        str(config["baseline_predictor.qualifying.practice_challenger.launch_envelope_path"])
    )
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    return launch_path, launch_path.parent / launch["bundle"]["path"]


def _refresh_launch_bundle_binding(launch_path: Path, bundle_path: Path) -> None:
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    launch["bundle"]["sha256"] = file_sha256(bundle_path)
    launch["bundle"]["bundle_sha256"] = bundle["bundle_sha256"]
    launch["launch_sha256"] = stable_json_sha256(
        {key: value for key, value in launch.items() if key != "launch_sha256"}
    )
    launch_path.write_text(json.dumps(launch), encoding="utf-8")


def test_q1_runtime_is_opt_in_and_uses_fitted_artifacts(tmp_path) -> None:
    predictor = _Predictor(_write_artifacts(tmp_path))

    records, diagnostics, evidence = predictor._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=_LIVE_INFERENCE_CUTOFF,
        weather="dry",
        n_simulations=200,
        rng=np.random.default_rng(17),
    )

    assert records is not None
    assert diagnostics["used"] is True
    assert diagnostics["checkpoint"] == "FP2"
    assert diagnostics["artifact_launch"]["variant_id"] == "q1_qualifying_practice"
    assert diagnostics["artifact_launch"]["launch_digest"].startswith("sha256:")
    assert diagnostics["artifact_launch"]["bundle_digest"].startswith("sha256:")
    assert set(records) == {"AAA", "BBB"}
    assert evidence["FP2"]["artifact_type"] == "qualifying_practice_evidence"


def test_q1_runtime_fails_closed_when_artifact_is_missing(tmp_path) -> None:
    predictor = _Predictor(
        {
            "baseline_predictor.model_variant": "q1_qualifying_practice",
            "baseline_predictor.qualifying.practice_challenger.launch_envelope_path": str(
                tmp_path / "missing-launch.json"
            ),
        }
    )

    records, diagnostics, evidence = predictor._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=_LIVE_INFERENCE_CUTOFF,
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
    )

    assert records is None
    assert diagnostics["fallback_reason"] == "missing_launch_artifact"
    assert evidence == {}


def test_q1_runtime_fails_closed_when_referenced_artifact_is_tampered(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    _launch_path, bundle_path = _launch_and_bundle_paths(config)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model_path = bundle_path.parent / bundle["models"]["main"]["FP2"]["path"]
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    model_payload["temperature"] = 0.75
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    records, diagnostics, evidence = _run_q1(_Predictor(config))

    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"
    assert evidence == {}


def test_q1_runtime_fails_closed_when_pinned_bundle_digest_changes(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    _launch_path, bundle_path = _launch_and_bundle_paths(config)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["metadata"] = {"note": "post-freeze mutation"}
    bundle["bundle_sha256"] = stable_json_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    records, diagnostics, evidence = _run_q1(_Predictor(config))

    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"
    assert evidence == {}


def test_live_q1_runtime_rejects_launch_created_after_prediction_cutoff(tmp_path) -> None:
    predictor = _Predictor(_write_artifacts(tmp_path))

    records, diagnostics, evidence = predictor._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=datetime(2026, 6, 20, tzinfo=UTC),
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
    )

    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"
    assert evidence == {}


def test_q1_runtime_rejects_manifest_tamper_inside_rehashed_launch(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    launch_path, _bundle_path = _launch_and_bundle_paths(config)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["manifest"]["metadata"]["tampered"] = True
    launch["launch_sha256"] = stable_json_sha256(
        {key: value for key, value in launch.items() if key != "launch_sha256"}
    )
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    records, diagnostics, evidence = _run_q1(_Predictor(config))

    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"
    assert evidence == {}


def test_q1_runtime_rejects_semantic_settings_detached_from_source(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    launch_path, _bundle_path = _launch_and_bundle_paths(config)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    settings = launch["semantic_config"]["settings"]
    settings["uncertainty_scale"] = 2.0
    launch["semantic_config"]["settings_sha256"] = stable_json_sha256(settings)
    launch["launch_sha256"] = stable_json_sha256(
        {key: value for key, value in launch.items() if key != "launch_sha256"}
    )
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    records, diagnostics, evidence = _run_q1(_Predictor(config))

    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"
    assert evidence == {}


def test_bundle_assembly_rejects_artifact_generated_after_manifest(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    launch_path, bundle_path = _launch_and_bundle_paths(config)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model_path = bundle_path.parent / bundle["models"]["main"]["FP2"]["path"]
    normalization_path = (
        bundle_path.parent / bundle["normalizations"]["main"]["FP2"]["high_downforce"]["path"]
    )
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    model_payload["generated_at"] = "2026-06-17T00:00:00Z"
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="after its allowed creation boundary"):
        build_qualifying_practice_bundle(
            candidate_id="q1-test",
            variant_id="q1_qualifying_practice",
            manifest=launch["manifest"],
            bundle_directory=bundle_path.parent,
            model_paths=[model_path],
            normalization_paths=[normalization_path],
        )


def test_q1_runtime_fails_closed_on_wrong_session_kind_or_track_class(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    records, diagnostics, _evidence = _run_q1(_Predictor(config), session_kind="sprint")
    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"

    config = _write_artifacts(
        tmp_path / "missing-track",
        track_class_by_event={},
    )
    records, diagnostics, _evidence = _run_q1(_Predictor(config))
    assert records is None
    assert diagnostics["fallback_reason"] == "missing_track_class_binding"


def test_q1_runtime_rejects_artifact_frozen_after_manifest_cutoff(tmp_path) -> None:
    config = _write_artifacts(tmp_path)
    launch_path, bundle_path = _launch_and_bundle_paths(config)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model_reference = bundle["models"]["main"]["FP2"]
    model_path = bundle_path.parent / model_reference["path"]
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    model_payload["generated_at"] = "2026-06-16T12:00:00Z"
    model_payload["training_metadata"]["cutoff_timestamp"] = "2026-06-16T00:00:00Z"
    model_payload["training_metadata"]["max_input_timestamp"] = "2026-06-15T12:00:00Z"
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    model_reference["sha256"] = file_sha256(model_path)
    model_reference["cutoff_timestamp"] = "2026-06-16T00:00:00Z"
    bundle["bundle_sha256"] = stable_json_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    )
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    _refresh_launch_bundle_binding(launch_path, bundle_path)

    records, diagnostics, evidence = _run_q1(_Predictor(config))

    assert records is None
    assert diagnostics["fallback_reason"] == "invalid_launch_artifact"
    assert evidence == {}


def test_q1_pre_checkpoint_uses_prior_model_without_practice_normalization(tmp_path) -> None:
    predictor = _Predictor(
        _write_artifacts(
            tmp_path,
            checkpoint="PRE",
            include_normalization=False,
            pace_feature="prior_utility",
        )
    )

    records, diagnostics, evidence = predictor._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        # Even if a caller accidentally supplies later laps, PRE must not consume them.
        session_laps_by_type={"FP1": _session_laps()},
        checkpoint_label="PRE",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=_LIVE_INFERENCE_CUTOFF,
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
    )

    assert records is not None
    assert diagnostics["used"] is True
    assert diagnostics["checkpoint"] == "PRE"
    assert diagnostics["sessions_used"] == []
    assert evidence == {}


def test_champion_does_not_read_q1_artifacts() -> None:
    predictor = _Predictor({"baseline_predictor.model_variant": "champion"})

    records, diagnostics, evidence = predictor._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=datetime(2026, 6, 20, tzinfo=UTC),
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
    )

    assert records is None
    assert diagnostics == {"used": False, "variant": "champion", "fallback_reason": None}
    assert evidence == {}


def test_champion_path_is_byte_identical_regardless_of_retrospective_diagnostic_flag() -> None:
    """retrospective_diagnostic must never change champion's own path -- it isn't even
    reachable (component_enabled(cfg, "q1") is False), but prove it explicitly."""
    predictor = _Predictor({"baseline_predictor.model_variant": "champion"})
    kwargs = dict(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=datetime(2026, 6, 20, tzinfo=UTC),
        weather="dry",
        n_simulations=20,
    )
    default_result = predictor._run_qualifying_practice_challenger(
        rng=np.random.default_rng(17), **kwargs
    )
    flagged_result = predictor._run_qualifying_practice_challenger(
        rng=np.random.default_rng(17), retrospective_diagnostic=True, **kwargs
    )
    assert (
        default_result
        == flagged_result
        == (
            None,
            {"used": False, "variant": "champion", "fallback_reason": None},
            {},
        )
    )


def test_live_q1_path_is_byte_identical_whether_the_flag_is_omitted_or_explicitly_false(
    tmp_path,
) -> None:
    """Default OFF must mean OFF: omitting the parameter and passing False explicitly
    must produce the exact same live-shadow Q1 result (the case this parameter was
    added for -- a launch envelope created well before its live inference cutoff)."""
    config = _write_artifacts(tmp_path)
    omitted_records, omitted_diagnostics, omitted_evidence = _run_q1(_Predictor(config))
    explicit_records, explicit_diagnostics, explicit_evidence = _Predictor(
        config
    )._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=_LIVE_INFERENCE_CUTOFF,
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
        retrospective_diagnostic=False,
    )
    assert omitted_records == explicit_records
    assert omitted_diagnostics == explicit_diagnostics
    assert omitted_evidence.keys() == explicit_evidence.keys()
    assert omitted_diagnostics["used"] is True
    assert omitted_diagnostics["artifact_launch"]["retrospective_diagnostic"] is False


def test_retrospective_diagnostic_true_resolves_a_historically_dated_launch(tmp_path) -> None:
    """The one behavior this flag exists to enable: a bundle built "today" (real
    wall-clock created_at) can resolve against a fold whose own historical cutoff has
    already passed, ONLY when explicitly opted in, and the result is permanently
    labeled so it can never be mistaken for a live/preregistered forecast."""
    config = _write_artifacts(tmp_path)
    past_cutoff = datetime(2026, 6, 20, tzinfo=UTC)

    without_flag = _Predictor(config)._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=past_cutoff,
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
    )
    assert without_flag[0] is None
    assert without_flag[1]["fallback_reason"] == "invalid_launch_artifact"

    with_flag_records, with_flag_diagnostics, _evidence = _Predictor(
        config
    )._run_qualifying_practice_challenger(
        all_drivers=_drivers(),
        session_laps_by_type={"FP2": _session_laps()},
        checkpoint_label="FP2",
        year=2026,
        race_name="Example Grand Prix",
        session_kind="main",
        inference_cutoff=past_cutoff,
        weather="dry",
        n_simulations=20,
        rng=np.random.default_rng(17),
        retrospective_diagnostic=True,
    )
    assert with_flag_records is not None
    assert with_flag_diagnostics["used"] is True
    assert with_flag_diagnostics["artifact_launch"]["retrospective_diagnostic"] is True
