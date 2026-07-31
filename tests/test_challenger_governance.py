"""Tests for research-only challenger manifests and promotion gates."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.analysis.challenger_governance import (
    PairedMetricSummary,
    QualifyingGateMetrics,
    RaceGateMetrics,
    RaceMetricViews,
    build_challenger_manifest,
    collect_git_provenance,
    evaluate_qualifying_gate,
    evaluate_race_gate,
    paired_weekend_bootstrap,
    race_candidate_kind_for_components,
    stable_json_sha256,
    validate_challenger_manifest,
)
from tests.challenger_test_helpers import strict_manifest, strict_replay_provenance


def test_race_candidate_kind_separates_race_inputs_from_anchor_changes() -> None:
    assert race_candidate_kind_for_components({"r0"}) == "race_input_or_grid_propagation"
    assert race_candidate_kind_for_components({"r1"}) == "race_input_or_grid_propagation"
    assert race_candidate_kind_for_components({"r0", "r2_no_anchor"}) == "anchor_or_physics"


def _git(repo: Path, *arguments: str) -> None:
    """Run a git setup command in a temporary repository."""
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def _initialise_repo(tmp_path: Path) -> Path:
    """Create a minimal committed repository with both effective configs."""
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "default.yaml").write_text("model:\n  version: test\n", encoding="utf-8")
    (repo / "config" / "production_config.json").write_text(
        '{"season": 2026}\n',
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Challenger Test",
        "-c",
        "user.email=challenger@example.test",
        "commit",
        "-m",
        "initial",
    )
    return repo


def _summary(*, events: int = 30, improvement: float = 0.2, ci_low: float = 0.1):
    """Build a compact paired metric summary for gate tests."""
    return PairedMetricSummary(
        events=events,
        champion_mean=3.0,
        challenger_mean=3.0 - improvement,
        improvement=improvement,
        ci90_low=ci_low,
        ci90_high=0.3,
        event_ids=tuple(f"event-{index:02d}" for index in range(events)),
    )


def _race_views(
    *,
    conditional_improvement: float = 0.0,
    end_to_end_improvement: float = 0.1,
) -> RaceMetricViews:
    """Build both named race-evaluation views."""
    return RaceMetricViews(
        conditional_actual_grid=_summary(
            events=30,
            improvement=conditional_improvement,
            ci_low=conditional_improvement,
        ),
        end_to_end_predicted_grid=_summary(
            events=30,
            improvement=end_to_end_improvement,
            ci_low=end_to_end_improvement,
        ),
    )


def test_manifest_keeps_champion_default_and_freezes_provenance(tmp_path: Path) -> None:
    """A candidate manifest must record all replay inputs without activating it."""
    repo = _initialise_repo(tmp_path)
    (repo / "research_notes.txt").write_text("uncommitted", encoding="utf-8")

    manifest = build_challenger_manifest(
        repo_root=repo,
        candidate_id="q0_belgium_replay",
        variant_id="q0_driver_state",
        feature_schema={"version": "qualifying-practice-v1", "columns": ["driver"]},
        input_snapshot_ids=["2026::belgian::fp2"],
        cutoff_at="2026-07-17T15:00:00+02:00",
        created_at=datetime(2026, 7, 17, 13, 1, tzinfo=UTC),
        simulation_counts={"qualifying": 5000, "race": 3000},
    )

    assert manifest["default_variant"] == "champion"
    assert manifest["runtime_activation_allowed"] is False
    assert manifest["variants"]["champion"]["default"] is True
    assert manifest["variant_id"] == "q0_driver_state"
    assert manifest["variants"]["q0_driver_state"]["default"] is False
    assert manifest["variants"]["q0_driver_state"]["components"] == ["q0"]
    assert manifest["cutoff_at"] == "2026-07-17T13:00:00Z"
    assert manifest["provenance"]["seeds"] == [17, 42, 91]
    assert manifest["provenance"]["checkpoints"] == ["PRE", "FP1", "FP2", "FP3"]
    assert manifest["provenance"]["dry_only"] is True
    assert manifest["provenance"]["simulation_counts"] == {
        "qualifying": 5000,
        "race": 3000,
    }
    assert [row["path"] for row in manifest["provenance"]["configuration"]["files"]] == [
        "config/default.yaml",
        "config/production_config.json",
    ]
    assert len(manifest["provenance"]["git"]["source_sha"]) == 40
    assert manifest["provenance"]["git"]["is_dirty"] is True
    assert len(manifest["provenance"]["git"]["dirty_diff_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert validate_challenger_manifest(manifest).variant_id == "q0_driver_state"


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("input_snapshot_ids", [], "input_snapshot_ids"),
        ("seeds", [17, 42], "seeds must be exactly"),
        ("checkpoints", ["PRE", "FP1", "FP2"], "checkpoints must be exactly"),
        ("dry_only", False, "dry_only"),
        ("simulation_counts", {}, "simulation_counts"),
    ],
)
def test_shared_manifest_validator_rejects_digest_valid_incomplete_provenance(
    field: str,
    replacement: object,
    match: str,
) -> None:
    manifest = strict_manifest()
    manifest["provenance"][field] = replacement
    manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )

    with pytest.raises(ValueError, match=match):
        validate_challenger_manifest(manifest)


def test_shared_manifest_validator_requires_information_cutoff_before_creation() -> None:
    manifest = strict_manifest()
    manifest["cutoff_at"] = "2026-07-18T12:00:00Z"
    manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )

    with pytest.raises(ValueError, match="cutoff_at <= created_at"):
        validate_challenger_manifest(manifest)


def test_dirty_digest_tracks_untracked_file_content(tmp_path: Path) -> None:
    """Changing an untracked file's bytes must change replay provenance."""
    repo = _initialise_repo(tmp_path)
    note = repo / "note.txt"
    note.write_text("first", encoding="utf-8")
    first = collect_git_provenance(repo)["dirty_diff_sha256"]
    note.write_text("second", encoding="utf-8")
    second = collect_git_provenance(repo)["dirty_diff_sha256"]

    assert first != second


def test_manifest_rejects_unknown_or_mismatched_registered_variant(tmp_path: Path) -> None:
    """Manifest components must come from the shared runtime variant registry."""
    repo = _initialise_repo(tmp_path)
    common = {
        "repo_root": repo,
        "candidate_id": "candidate-run",
        "feature_schema": "v1",
        "input_snapshot_ids": [],
        "cutoff_at": "2026-07-17T13:00:00Z",
        "simulation_counts": {"qualifying": 100},
    }

    with pytest.raises(ValueError, match="unknown challenger variant_id"):
        build_challenger_manifest(variant_id="made_up", **common)
    with pytest.raises(ValueError, match="must match the model variant registry"):
        build_challenger_manifest(
            variant_id="q0_driver_state",
            components=["q1"],
            **common,
        )


def test_paired_weekend_bootstrap_is_deterministic_and_paired() -> None:
    """The bootstrap must use only common finite weekends and preserve direction."""
    champion = {f"race-{index}": 3.0 + index / 10 for index in range(10)}
    challenger = {f"race-{index}": 2.8 + index / 10 for index in range(10)}
    challenger["challenger-only"] = 0.0

    first = paired_weekend_bootstrap(champion, challenger, n_resamples=1000, seed=17)
    second = paired_weekend_bootstrap(champion, challenger, n_resamples=1000, seed=17)

    assert first == second
    assert first.events == 10
    assert first.improvement == pytest.approx(0.2)
    assert first.ci90_low == pytest.approx(0.2)


def test_qualifying_gate_passes_only_with_full_grid_and_race_guardrails() -> None:
    """A well-calibrated main-Q candidate can pass every target-specific check."""
    result = evaluate_qualifying_gate(
        QualifyingGateMetrics(
            target="main_qualifying",
            grid_mae=_summary(),
            h2h_brier_relative_improvement=0.06,
            h2h_log_loss_delta=-0.01,
            ece_delta=0.01,
            interval_coverage=0.90,
            interval_width_relative_change=0.05,
            checkpoint_mae_regressions={
                "PRE": 0.02,
                "FP1": 0.08,
                "FP2": -0.1,
                "FP3": 0.03,
            },
            race_views=_race_views(conditional_improvement=0.0, end_to_end_improvement=0.01),
            movements_requiring_review=12,
            movements_reviewed=12,
            manifest=strict_manifest(),
            replay_provenance=strict_replay_provenance(),
        )
    )

    assert result.passed is True
    assert result.reasons == ()


def test_qualifying_gate_blocks_weak_ci_slice_regression_and_missing_review() -> None:
    """A mean gain cannot hide uncertainty, checkpoint harm, or unaudited movement."""
    result = evaluate_qualifying_gate(
        QualifyingGateMetrics(
            target="main_qualifying",
            grid_mae=_summary(events=29, ci_low=-0.01),
            h2h_brier_relative_improvement=0.06,
            h2h_log_loss_delta=-0.01,
            ece_delta=0.01,
            interval_coverage=0.90,
            interval_width_relative_change=0.05,
            checkpoint_mae_regressions={
                "PRE": 0.02,
                "FP1": 0.08,
                "FP2": 0.11,
                "FP3": 0.01,
            },
            race_views=_race_views(),
            movements_requiring_review=3,
            movements_reviewed=2,
            manifest=strict_manifest(),
            replay_provenance=strict_replay_provenance(),
        )
    )

    assert result.passed is False
    assert result.checks["minimum_scored_events"] is False
    assert result.checks["grid_mae_ci90_above_zero"] is False
    assert result.checks["checkpoint_slices_within_tolerance"] is False
    assert result.checks["movement_review_complete"] is False


@pytest.mark.parametrize(
    ("candidate_kind", "conditional_improvement", "end_to_end_improvement", "expected"),
    [
        ("qualifying_only", 0.0, -0.02, True),
        ("qualifying_only", 0.001, 0.2, False),
        ("race_input_or_grid_propagation", -0.05, 0.10, True),
        ("race_input_or_grid_propagation", -0.051, 0.20, False),
        ("anchor_or_physics", 0.10, 0.10, True),
        ("anchor_or_physics", 0.09, 0.20, False),
    ],
)
def test_race_gate_applies_component_specific_thresholds(
    candidate_kind: str,
    conditional_improvement: float,
    end_to_end_improvement: float,
    expected: bool,
) -> None:
    """Race promotion requirements depend on what the candidate changes."""
    variant = {
        "qualifying_only": "q1_qualifying_practice",
        "race_input_or_grid_propagation": "r0_long_run",
        "anchor_or_physics": "r2_no_anchor",
    }[candidate_kind]
    result = evaluate_race_gate(
        RaceGateMetrics(
            target="grand_prix_race",
            race_views=_race_views(
                conditional_improvement=conditional_improvement,
                end_to_end_improvement=end_to_end_improvement,
            ),
            winner_accuracy_delta_pp=0.0,
            top3_accuracy_delta_pp=-2.0,
            dnf_brier_delta=0.005,
            manifest=strict_manifest(variant),
            replay_provenance=strict_replay_provenance(),
        )
    )

    assert result.passed is expected


def test_race_gate_blocks_headline_and_dnf_regressions() -> None:
    """MAE gains do not excuse winner, top-three, or DNF calibration failures."""
    result = evaluate_race_gate(
        RaceGateMetrics(
            target="grand_prix_race",
            race_views=_race_views(conditional_improvement=0.0, end_to_end_improvement=0.2),
            winner_accuracy_delta_pp=-0.1,
            top3_accuracy_delta_pp=-2.1,
            dnf_brier_delta=0.006,
            manifest=strict_manifest("r1_joint_grid"),
            replay_provenance=strict_replay_provenance(),
        )
    )

    assert result.passed is False
    assert result.checks["winner_accuracy_not_regressed"] is False
    assert result.checks["top3_accuracy_within_tolerance"] is False
    assert result.checks["dnf_brier_within_tolerance"] is False


def test_race_gate_rejects_nonfinite_or_unmatched_replay_views() -> None:
    """Malformed or differently scoped replay summaries cannot pass promotion."""
    views = RaceMetricViews(
        conditional_actual_grid=_summary(events=30, improvement=float("inf")),
        end_to_end_predicted_grid=_summary(events=29, improvement=0.2),
    )
    result = evaluate_race_gate(
        RaceGateMetrics(
            target="grand_prix_race",
            race_views=views,
            winner_accuracy_delta_pp=0.0,
            top3_accuracy_delta_pp=0.0,
            dnf_brier_delta=0.0,
            manifest=strict_manifest("r0_long_run"),
            replay_provenance=strict_replay_provenance(),
        )
    )

    assert result.passed is False
    assert result.checks["finite_metrics"] is False
    assert result.checks["race_view_event_counts_match"] is False
