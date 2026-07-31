"""Focused tests for the research-only Q1 fitting command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from scripts import fit_qualifying_practice_challenger as cli

from src.analysis.challenger_governance import (
    build_challenger_manifest,
    file_sha256,
    stable_json_sha256,
)
from src.models.qualifying_practice_challenger import (
    DEFAULT_FEATURE_COLUMNS,
    FittedQualifyingPracticeModel,
)
from src.models.qualifying_practice_runtime import load_practice_normalization


def _normalization_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    drivers_and_teams = (("A", "T1"), ("B", "T2"), ("C", "T3"), ("D", "T1"))
    for repeat in range(2):
        for driver, team in drivers_and_teams:
            rows.append(
                {
                    "event_id": f"event-{repeat}",
                    "event_start_at": "2026-06-01T10:00:00+00:00",
                    "checkpoint": "FP2",
                    "session_kind": "main",
                    "track_class": "high_downforce",
                    "is_dry": True,
                    "driver": driver,
                    "team": team,
                    "lap_time_a_s": 90.0 + repeat * 0.01,
                    "lap_time_b_s": 90.5 + repeat * 0.01,
                    "compound_a": "SOFT",
                    "compound_b": "MEDIUM",
                    "tyre_age_a": 1,
                    "tyre_age_b": 1,
                    "evolution_a": 0.5,
                    "evolution_b": 0.5,
                }
            )
    rows.extend(
        [
            {**rows[0], "checkpoint": "FP1"},
            {**rows[1], "is_dry": False},
        ]
    )
    return pd.DataFrame(rows)


def _model_rows(
    *,
    start: int,
    events: int,
    checkpoint: str = "FP2",
    session_kind: str = "main",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_number in range(start, start + events):
        for driver, position, direction in (("AAA", 1, 1.0), ("BBB", 2, -1.0)):
            row: dict[str, object] = {
                "event_id": f"event-{event_number}",
                "event_start_at": "2026-06-01T10:00:00+00:00",
                "checkpoint": checkpoint,
                "session_kind": session_kind,
                "is_dry": True,
                "driver": driver,
                "actual_position": position,
            }
            row.update({column: 0.0 for column in DEFAULT_FEATURE_COLUMNS})
            row["prior_utility"] = direction
            row["best_adjusted_lap_s"] = 90.0 - (0.25 * direction)
            rows.append(row)
    return pd.DataFrame(rows)


def _base_arguments(
    *,
    command: str,
    repo: Path,
    source: Path,
    output: Path | None,
    session_kind: str = "main",
) -> list[str]:
    arguments = [
        command,
        "--input",
        str(source),
        "--candidate-id",
        "q1-test",
        "--input-snapshot-id",
        "snapshot-001",
        "--checkpoint",
        "FP2",
        "--session-kind",
        session_kind,
        "--cutoff",
        "2026-07-17T12:00:00+00:00",
        "--repo-root",
        str(repo),
    ]
    if output is not None:
        arguments[3:3] = ["--output", str(output)]
    return arguments


def _bundle_manifest(semantic_config_path: Path | None = None) -> dict:
    manifest = build_challenger_manifest(
        repo_root=Path.cwd(),
        candidate_id="q1-test",
        variant_id="q1_qualifying_practice",
        feature_schema="qualifying-practice-v2",
        input_snapshot_ids=["snapshot-001"],
        cutoff_at="2026-07-18T12:00:00Z",
        simulation_counts={"qualifying": 300, "race": 300},
    )
    if semantic_config_path is not None:
        config_files = manifest["provenance"]["configuration"]["files"]
        config_files.append(
            {
                "path": semantic_config_path.relative_to(
                    semantic_config_path.parents[2]
                ).as_posix(),
                "sha256": file_sha256(semantic_config_path),
            }
        )
        manifest["provenance"]["configuration"]["effective_bundle_sha256"] = stable_json_sha256(
            config_files
        )
        manifest["manifest_sha256"] = stable_json_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
    return manifest


def test_normalization_cli_filters_checkpoint_and_writes_loadable_artifact(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "comparisons.csv"
    _normalization_rows().rename(columns={"event_start_at": "session_started_at"}).to_csv(
        source, index=False
    )
    output = (
        repo
        / "data"
        / "processed"
        / "model_artifacts"
        / "qualifying_practice"
        / "challengers"
        / "q1-test"
        / "fp2-normalization.json"
    )

    arguments = _base_arguments(
        command="normalization",
        repo=repo,
        source=source,
        output=output,
    )
    arguments.extend(
        [
            "--track-class",
            "high_downforce",
            "--event-timestamp-column",
            "session_started_at",
        ]
    )
    assert cli.main(arguments) == 0

    payload = json.loads(output.read_text())
    assert datetime.fromisoformat(payload["generated_at"]) >= datetime.fromisoformat(
        "2026-07-17T12:00:00+00:00"
    )
    assert payload["generated_at"] != payload["training_metadata"]["cutoff_timestamp"]
    assert payload["normalization"]["provenance"] == "empirical_shrunk"
    assert payload["training_metadata"]["replay_seeds"] == [17, 42, 91]
    assert payload["training_metadata"]["event_timestamp_column"] == "session_started_at"
    assert payload["training_metadata"]["max_input_timestamp"] == ("2026-06-01T10:00:00+00:00")
    assert payload["training_metadata"]["row_counts"] == {
        "input_rows": 10,
        "selected_rows": 8,
        "excluded_other_checkpoint": 1,
        "excluded_other_session_kind": 0,
        "excluded_non_dry": 1,
        "excluded_other_track_class": 0,
    }
    loaded = load_practice_normalization(output)
    assert loaded is not None
    assert loaded.comparison_count == 8

    duplicate_output = output.with_name("fp2-normalization-copy.json")
    duplicate_arguments = list(arguments)
    duplicate_arguments[duplicate_arguments.index("--output") + 1] = str(duplicate_output)
    assert cli.main(duplicate_arguments) == 0
    duplicate_payload = json.loads(duplicate_output.read_text())
    assert duplicate_payload["generated_at"] >= payload["generated_at"]
    duplicate_payload.pop("generated_at")
    payload.pop("generated_at")
    assert duplicate_payload == payload


def test_model_cli_enforces_main_event_gate_and_calibrates_disjoint_holdout(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    training_path = repo / "training.csv"
    calibration_path = repo / "calibration.csv"
    _model_rows(start=1, events=30).to_csv(training_path, index=False)
    _model_rows(start=101, events=3).to_csv(calibration_path, index=False)
    output = (
        repo / "data" / "model_diagnostics" / "challenger_research" / "q1-test" / "fp2-model.json"
    )

    arguments = _base_arguments(
        command="model",
        repo=repo,
        source=training_path,
        output=output,
    )
    arguments.extend(["--calibration-input", str(calibration_path)])
    assert cli.main(arguments) == 0

    payload = json.loads(output.read_text())
    model = FittedQualifyingPracticeModel.from_dict(payload)
    assert model.checkpoint == "FP2"
    assert model.training_events == 30
    assert datetime.fromisoformat(model.generated_at) >= datetime.fromisoformat(
        "2026-07-17T12:00:00+00:00"
    )
    assert model.generated_at != payload["training_metadata"]["cutoff_timestamp"]
    assert model.temperature in {0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}
    assert payload["training_metadata"]["temperature_calibration"] == ("disjoint_holdout_log_loss")
    assert payload["training_metadata"]["minimum_training_events"] == 30
    assert payload["training_metadata"]["max_input_timestamp"] == ("2026-06-01T10:00:00+00:00")
    assert payload["training_metadata"]["calibration_max_input_timestamp"] == (
        "2026-06-01T10:00:00+00:00"
    )


def test_canonical_layout_builds_manifest_bound_runtime_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    candidate_root = (
        repo
        / "data"
        / "processed"
        / "model_artifacts"
        / "qualifying_practice"
        / "challengers"
        / "q1-test"
    )
    normalization_input = repo / "normalization.csv"
    training_input = repo / "training.csv"
    calibration_input = repo / "calibration.csv"
    _normalization_rows().to_csv(normalization_input, index=False)
    _model_rows(start=1, events=30).to_csv(training_input, index=False)
    _model_rows(start=101, events=3).to_csv(calibration_input, index=False)

    normalization_args = _base_arguments(
        command="normalization",
        repo=repo,
        source=normalization_input,
        output=None,
    )
    normalization_args.extend(
        ["--candidate-root", str(candidate_root), "--track-class", "high_downforce"]
    )
    assert cli.main(normalization_args) == 0

    model_args = _base_arguments(
        command="model",
        repo=repo,
        source=training_input,
        output=None,
    )
    model_args.extend(
        [
            "--candidate-root",
            str(candidate_root),
            "--calibration-input",
            str(calibration_input),
        ]
    )
    assert cli.main(model_args) == 0

    semantic_config_path = repo / "config" / "research" / "q1-test.yaml"
    semantic_config_path.parent.mkdir(parents=True)
    semantic_config_path.write_text(
        json.dumps(
            {
                "artifact_type": "qualifying_practice_candidate_definition",
                "schema_version": 1,
                "model_variant": "q1_qualifying_practice",
                "launch_envelope_path": (
                    "data/processed/model_artifacts/qualifying_practice/"
                    "challengers/q1-test/launch.json"
                ),
                "bundle_path": (
                    "data/processed/model_artifacts/qualifying_practice/"
                    "challengers/q1-test/bundle.json"
                ),
                "candidate_id": "q1-test",
                "track_class_by_event": {"2026:Example Grand Prix": "high_downforce"},
                "uncertainty_scale": 1.0,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = repo / "manifest.json"
    manifest = _bundle_manifest(semantic_config_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        cli.main(
            [
                "bundle",
                "--candidate-id",
                "q1-test",
                "--variant-id",
                "q1_qualifying_practice",
                "--manifest",
                str(manifest_path),
                "--candidate-root",
                str(candidate_root),
                "--repo-root",
                str(repo),
            ]
        )
        == 0
    )

    bundle = json.loads((candidate_root / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["manifest_sha256"] == manifest["manifest_sha256"]
    assert bundle["models"]["main"]["FP2"]["path"] == "models/main/fp2.json"
    assert bundle["normalizations"]["main"]["FP2"]["high_downforce"]["path"] == (
        "normalizations/main/fp2/high_downforce.json"
    )
    assert len(bundle["bundle_sha256"]) == 64
    assert (
        cli.main(
            [
                "launch",
                "--candidate-id",
                "q1-test",
                "--variant-id",
                "q1_qualifying_practice",
                "--manifest",
                str(manifest_path),
                "--semantic-config",
                str(semantic_config_path),
                "--candidate-root",
                str(candidate_root),
                "--repo-root",
                str(repo),
            ]
        )
        == 0
    )
    launch = json.loads((candidate_root / "launch.json").read_text(encoding="utf-8"))
    assert launch["manifest_sha256"] == manifest["manifest_sha256"]
    assert launch["bundle"]["bundle_sha256"] == bundle["bundle_sha256"]
    assert launch["semantic_config"]["settings"]["track_class_by_event"] == {
        "2026:Example Grand Prix": "high_downforce"
    }
    assert len(launch["launch_sha256"]) == 64


def test_model_cli_allows_sprint_training_at_eight_event_gate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    training_path = repo / "training.csv"
    calibration_path = repo / "calibration.csv"
    _model_rows(start=1, events=8, session_kind="sprint").to_csv(training_path, index=False)
    _model_rows(start=101, events=2, session_kind="sprint").to_csv(
        calibration_path,
        index=False,
    )
    output = (
        repo
        / "data"
        / "processed"
        / "model_artifacts"
        / "qualifying_practice"
        / "challengers"
        / "sprint-fp2.json"
    )
    arguments = _base_arguments(
        command="model",
        repo=repo,
        source=training_path,
        output=output,
        session_kind="sprint",
    )
    arguments.extend(["--calibration-input", str(calibration_path)])

    assert cli.main(arguments) == 0
    payload = json.loads(output.read_text())
    assert payload["training_metadata"]["training_event_count"] == 8
    assert payload["training_metadata"]["minimum_training_events"] == 8


@pytest.mark.parametrize(("session_kind", "event_count"), [("main", 29), ("sprint", 7)])
def test_model_cli_rejects_training_below_session_gate(
    tmp_path: Path,
    session_kind: str,
    event_count: int,
) -> None:
    repo = tmp_path / session_kind
    repo.mkdir()
    training_path = repo / "training.csv"
    calibration_path = repo / "calibration.csv"
    _model_rows(start=1, events=event_count, session_kind=session_kind).to_csv(
        training_path,
        index=False,
    )
    _model_rows(start=101, events=2, session_kind=session_kind).to_csv(
        calibration_path,
        index=False,
    )
    output = (
        repo
        / "data"
        / "processed"
        / "model_artifacts"
        / "qualifying_practice"
        / "challengers"
        / "too-small.json"
    )
    arguments = _base_arguments(
        command="model",
        repo=repo,
        source=training_path,
        output=output,
        session_kind=session_kind,
    )
    arguments.extend(["--calibration-input", str(calibration_path)])

    with pytest.raises(ValueError, match="requires at least"):
        cli.main(arguments)
    assert not output.exists()


def test_model_cli_rejects_overlapping_calibration_events(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    training_path = repo / "training.csv"
    calibration_path = repo / "calibration.csv"
    _model_rows(start=1, events=30).to_csv(training_path, index=False)
    _model_rows(start=30, events=2).to_csv(calibration_path, index=False)
    output = repo / "data" / "model_diagnostics" / "challenger_research" / "overlap.json"
    arguments = _base_arguments(
        command="model",
        repo=repo,
        source=training_path,
        output=output,
    )
    arguments.extend(["--calibration-input", str(calibration_path)])

    with pytest.raises(ValueError, match="must be disjoint"):
        cli.main(arguments)


def test_output_must_not_target_active_model_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    active = (
        repo
        / "data"
        / "processed"
        / "model_artifacts"
        / "qualifying_practice"
        / "models"
        / "fp2.json"
    )
    with pytest.raises(ValueError, match="challenger-only root"):
        cli.validate_output_path(active, repo_root=repo)


@pytest.mark.parametrize(
    "event_timestamp",
    ["2026-07-17T12:00:00+00:00", "2026-07-17T12:00:01+00:00"],
)
def test_normalization_rejects_equal_or_future_selected_rows(
    tmp_path: Path,
    event_timestamp: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "comparisons.csv"
    rows = _normalization_rows()
    rows.loc[0, "event_start_at"] = event_timestamp
    rows.to_csv(source, index=False)
    output = (
        repo / "data" / "model_diagnostics" / "challenger_research" / "leaking-normalization.json"
    )
    arguments = _base_arguments(
        command="normalization",
        repo=repo,
        source=source,
        output=output,
    )
    arguments.extend(["--track-class", "high_downforce"])

    with pytest.raises(ValueError, match="strictly before cutoff"):
        cli.main(arguments)
    assert not output.exists()


def test_model_rejects_naive_calibration_timestamps(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    training_path = repo / "training.csv"
    calibration_path = repo / "calibration.csv"
    _model_rows(start=1, events=30).to_csv(training_path, index=False)
    calibration = _model_rows(start=101, events=2)
    calibration["event_start_at"] = "2026-06-01T10:00:00"
    calibration.to_csv(calibration_path, index=False)
    output = repo / "data" / "model_diagnostics" / "challenger_research" / "naive-calibration.json"
    arguments = _base_arguments(
        command="model",
        repo=repo,
        source=training_path,
        output=output,
    )
    arguments.extend(["--calibration-input", str(calibration_path)])

    with pytest.raises(ValueError, match="must include a timezone"):
        cli.main(arguments)
    assert not output.exists()


def test_output_is_immutable_once_written(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = repo / "data" / "model_diagnostics" / "challenger_research" / "frozen.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.validate_output_path(output, repo_root=repo)
