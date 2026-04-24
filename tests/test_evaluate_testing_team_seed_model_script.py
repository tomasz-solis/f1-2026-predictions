"""Tests for the testing-team-seed evaluation script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_eval_module():
    """Load the evaluation script as a module for direct testing."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_testing_team_seed_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_testing_team_seed_model_script",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_markdown_summary_emits_compact_year_sections(tmp_path):
    """Markdown summary should keep one short section per evaluated year."""
    module = _load_eval_module()
    output_path = tmp_path / "summary.md"

    module._write_markdown_summary(
        comparisons=[
            {
                "year": 2024,
                "wins_promotion_gate": True,
                "deltas": {
                    "race_mae_improvement": 0.2,
                    "qualifying_mae_improvement": 0.1,
                    "top3_accuracy_delta": 4.0,
                    "winner_accuracy_delta": 2.0,
                    "overlap_race_mae_delta": 0.05,
                    "overlap_qualifying_mae_delta": 0.02,
                },
            }
        ],
        output_path=output_path,
    )

    written = output_path.read_text()
    assert "## 2024" in written
    assert "Promotion gate" in written


def test_copy_processed_tree_copies_processed_and_learning_state(tmp_path):
    """Evaluation harness should isolate processed data under a temp root."""
    module = _load_eval_module()
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    (source_root / "processed" / "car_characteristics").mkdir(parents=True)
    (
        source_root / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    ).write_text(json.dumps({"teams": {}}))
    (source_root / "learning_state.json").write_text(json.dumps({"season": 2026}))

    module._copy_processed_tree(
        source_data_root=source_root,
        target_data_root=target_root,
    )

    assert (
        target_root / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    ).exists()
    assert (target_root / "learning_state.json").exists()


def test_build_conformal_rows_matches_actuals_by_driver_not_list_order():
    """Conformal extraction should compare each driver against its own actual result."""
    module = _load_eval_module()

    rows = module._build_conformal_rows_from_report(
        {
            "race_results": [
                {
                    "status": "ok",
                    "qualifying_regime": "practice_backed",
                    "race_regime": "checkpoint_backed",
                    "qualifying_prediction_rows": [
                        {"driver": "NOR", "median_position": 1, "p5": 1, "p95": 2},
                        {"driver": "VER", "median_position": 2, "p5": 1, "p95": 3},
                    ],
                    "qualifying_actual_rows": [
                        {"driver": "VER", "position": 1},
                        {"driver": "NOR", "position": 2},
                    ],
                    "race_prediction_rows": [
                        {"driver": "VER", "median_position": 1, "p5": 1, "p95": 2},
                        {"driver": "NOR", "median_position": 2, "p5": 1, "p95": 3},
                    ],
                    "race_actual_rows": [
                        {"driver": "NOR", "position": 1},
                        {"driver": "VER", "position": 2},
                    ],
                }
            ]
        }
    )

    assert rows == [
        {
            "session": "qualifying",
            "regime": "practice_backed",
            "residual": 1,
            "covered": True,
        },
        {
            "session": "qualifying",
            "regime": "practice_backed",
            "residual": 1,
            "covered": True,
        },
        {
            "session": "race",
            "regime": "checkpoint_backed",
            "residual": 1,
            "covered": True,
        },
        {
            "session": "race",
            "regime": "checkpoint_backed",
            "residual": 1,
            "covered": True,
        },
    ]


def test_build_model_artifacts_skips_empty_race_dataset(tmp_path):
    """Artifact builds should continue when one residual dataset is unavailable."""
    module = _load_eval_module()

    qualifying_dataset = pd.DataFrame(
        [
            {
                "season_year": 2024,
                "race_name": "Australian Grand Prix",
                "target_residual_positions": 0.5,
            }
        ]
    )
    empty_race_dataset = pd.DataFrame()

    class _DummyModel:
        """Simple fitted-model stand-in for evaluation-script tests."""

        def __init__(self, label: str):
            self.label = label

        def summary(self) -> dict[str, str]:
            """Return a compact fake summary payload."""
            return {"label": self.label}

    def _save_dummy_model(*, model, artifact_path, summary_path):
        """Write placeholder artifact files for the enabled path."""
        del model
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("artifact")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}")

    module.build_qualifying_residual_dataset = lambda **_kwargs: qualifying_dataset
    module.summarize_qualifying_residual_dataset = lambda dataset: {"rows": int(len(dataset))}
    module.fit_qualifying_residual_model = lambda dataset, **_kwargs: _DummyModel(
        f"qualifying:{len(dataset)}"
    )
    module.save_qualifying_residual_model = _save_dummy_model

    module.build_race_residual_dataset = lambda **_kwargs: empty_race_dataset
    module.summarize_race_residual_dataset = lambda dataset: {"rows": int(len(dataset))}
    module.fit_race_residual_model = lambda dataset, **_kwargs: _DummyModel(f"race:{len(dataset)}")
    module.save_race_residual_model = _save_dummy_model

    artifacts = module._build_model_artifacts(
        years=(2022, 2023, 2024),
        output_dir=tmp_path,
        config_path="config/default.yaml",
        seed=42,
        max_races=6,
    )

    assert artifacts["qualifying"]["enabled"] is True
    assert artifacts["qualifying"]["model_summary"] == {"label": "qualifying:1"}
    assert artifacts["qualifying"]["dataset_summary"] == {"rows": 1}
    json.dumps(artifacts)
    assert Path(artifacts["qualifying"]["artifact_path"]).exists()

    assert artifacts["race"]["enabled"] is False
    assert artifacts["race"]["model_summary"] is None
    assert artifacts["race"]["dataset_summary"] == {"rows": 0}
    assert "No training rows were available" in str(artifacts["race"]["error"])
    assert not Path(artifacts["race"]["artifact_path"]).exists()


def test_race_delta_summary_counts_variant_wins_and_losses():
    """Race delta summary should count per-race movement against a baseline."""
    module = _load_eval_module()

    summary = module._race_delta_summary(
        experimental_report={
            "race_results": [
                {
                    "status": "ok",
                    "race_name": "Race 1",
                    "qualifying_mae": 4.0,
                    "race_mae": 5.0,
                },
                {
                    "status": "ok",
                    "race_name": "Race 2",
                    "qualifying_mae": 2.0,
                    "race_mae": 3.0,
                },
            ]
        },
        baseline_report={
            "race_results": [
                {
                    "status": "ok",
                    "race_name": "Race 1",
                    "qualifying_mae": 3.0,
                    "race_mae": 4.0,
                },
                {
                    "status": "ok",
                    "race_name": "Race 2",
                    "qualifying_mae": 4.0,
                    "race_mae": 4.0,
                },
            ]
        },
    )

    assert summary["races_compared"] == 2
    assert summary["qualifying_worse_count"] == 1
    assert summary["qualifying_better_count"] == 1
    assert summary["race_worse_count"] == 1
    assert summary["race_better_count"] == 1
    assert summary["mean_qualifying_delta"] == -0.5
    assert summary["mean_race_delta"] == 0.0


def test_run_component_ablation_writes_variant_reports(tmp_path):
    """Ablation runner should isolate variants and compare each one to champion."""
    module = _load_eval_module()
    source_root = tmp_path / "data"
    (source_root / "processed" / "car_characteristics").mkdir(parents=True)
    (
        source_root / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    ).write_text(json.dumps({"teams": {}}))
    qualifying_artifact = tmp_path / "qualifying.pkl"
    race_artifact = tmp_path / "race.pkl"
    conformal_artifact = tmp_path / "conformal.json"
    qualifying_artifact.write_text("q")
    race_artifact.write_text("r")
    conformal_artifact.write_text("{}")

    module.build_testing_model_team_payload = lambda **_kwargs: {"teams": {"A": {}}}
    module._calibration_summary = lambda report: {"race": report["summary"].get("race_mae_mean")}
    module._select_worst_weekends = lambda report, top_n=2: []

    calls: list[dict] = []

    def _fake_backtest(**kwargs):
        """Return predictable metrics for champion and each ablation variant."""
        calls.append(kwargs)
        data_root_name = kwargs["data_root"].name
        race_mae = 5.0
        qualifying_mae = 4.0
        if data_root_name == "testing_seed_only_data":
            race_mae = 6.0
            qualifying_mae = 5.0
        elif data_root_name == "race_residual_only_data":
            race_mae = 4.5
        return {
            "summary": {
                "race_mae_mean": race_mae,
                "qualifying_mae_mean": qualifying_mae,
                "top3_accuracy_mean": 10.0,
                "winner_accuracy_percent": 0.0,
            },
            "overlap": {},
            "race_results": [
                {
                    "status": "ok",
                    "race_name": "Race 1",
                    "race_mae": race_mae,
                    "qualifying_mae": qualifying_mae,
                }
            ],
        }

    module._run_compact_backtest = _fake_backtest

    ablation = module._run_component_ablation(
        season_year=2026,
        training_years=(2022, 2023, 2024),
        repo_data_root=source_root,
        output_dir=tmp_path / "out",
        weather="dry",
        qualifying_simulations=1,
        race_simulations=1,
        seed=42,
        config_path="config/default.yaml",
        qualifying_artifact_path=qualifying_artifact,
        race_artifact_path=race_artifact,
        conformal_artifact_path=conformal_artifact,
    )

    labels = [variant["label"] for variant in ablation["variants"]]
    assert labels == [
        "testing_seed_only",
        "qualifying_residual_only",
        "race_residual_only",
        "conformal_only",
        "testing_seed_plus_residuals",
        "full_challenger",
    ]
    assert len(calls) == 7
    testing_seed_call = next(
        call for call in calls if call["data_root"].name == "testing_seed_only_data"
    )
    assert (
        testing_seed_call["config_overrides"][
            "baseline_predictor.qualifying.qualifying_residual_model.enabled"
        ]
        is False
    )
    plus_residuals_call = next(
        call for call in calls if call["data_root"].name == "testing_seed_plus_residuals_data"
    )
    assert (
        plus_residuals_call["config_overrides"][
            "baseline_predictor.qualifying.qualifying_residual_model.allow_with_testing_seed"
        ]
        is True
    )
    full_challenger_call = next(
        call for call in calls if call["data_root"].name == "full_challenger_data"
    )
    assert (
        full_challenger_call["config_overrides"][
            "baseline_predictor.qualifying.qualifying_residual_model.allow_with_testing_seed"
        ]
        is False
    )
    race_residual = next(
        variant for variant in ablation["variants"] if variant["label"] == "race_residual_only"
    )
    assert race_residual["comparison"]["deltas"]["race_mae_improvement"] == 0.5
    assert (tmp_path / "out" / "ablation.json").exists()
    assert (tmp_path / "out" / "ablation.md").exists()


def test_run_compact_backtest_stops_after_repeated_missing_actuals(tmp_path):
    """Evaluation should stop asking FastF1 once actuals disappear repeatedly."""
    module = _load_eval_module()
    attempted_races: list[str] = []
    naive_races: list[str] = []

    module._build_evaluation_predictor = lambda **_kwargs: object()
    module.get_races_for_year = lambda **_kwargs: [
        "Race 1",
        "Race 2",
        "Race 3",
        "Race 4",
    ]

    def _run_single_race_backtest(**kwargs):
        """Record attempted races and return the rate-limit skip shape."""
        attempted_races.append(kwargs["race_name"])
        return {
            "race_name": kwargs["race_name"],
            "status": "skipped",
            "reason": "missing_actual_results",
        }

    def _run_previous_race_naive_backtest(*, year, race_names):
        """Record the naive baseline race list without doing more fetches."""
        del year
        naive_races.extend(race_names)
        return {"summary": {}, "race_results": []}

    module.run_single_race_backtest = _run_single_race_backtest
    module.run_previous_race_naive_backtest = _run_previous_race_naive_backtest
    module.build_overlap_comparison = lambda **_kwargs: {}

    report = module._run_compact_backtest(
        year=2022,
        data_root=tmp_path,
        max_races=None,
        weather="dry",
        qualifying_simulations=1,
        race_simulations=1,
        seed=42,
        config_path="config/default.yaml",
        max_consecutive_missing_actuals=3,
    )

    assert attempted_races == ["Race 1", "Race 2", "Race 3"]
    assert naive_races == attempted_races
    assert report["races"] == attempted_races
    assert report["summary"]["races_skipped"] == 3


def test_stitch_existing_outputs_writes_json_safe_summary(tmp_path):
    """Interrupted evaluation runs should be recoverable without rerunning backtests."""
    module = _load_eval_module()
    output_dir = tmp_path / "research"
    output_dir.mkdir()
    (output_dir / "holdout_summary.json").write_text(
        json.dumps(
            {
                "years": [2022],
                "comparisons": [
                    {
                        "year": 2022,
                        "comparisons": {
                            "testing_vs_ranking": {
                                "label": "testing_model_vs_ranking",
                                "wins_promotion_gate": False,
                                "deltas": {"race_mae_improvement": -0.1},
                            }
                        },
                    }
                ],
            }
        )
    )
    for folder_name in ("season_2022", "live_2026"):
        folder = output_dir / folder_name
        folder.mkdir()
        (folder / "comparison.json").write_text(
            json.dumps(
                {
                    "comparison": {
                        "label": folder_name,
                        "wins_promotion_gate": False,
                        "deltas": {"race_mae_improvement": 0.0},
                    }
                }
            )
        )

    artifact_dir = output_dir / "model_artifacts" / "qualifying_residual"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "qualifying_residual_model.pkl").write_text("model")
    (artifact_dir / "qualifying_residual_model.summary.json").write_text(
        json.dumps({"model": "qualifying"})
    )

    summary = module._stitch_existing_outputs(
        output_dir=output_dir,
        season_year=2022,
        live_year=2026,
    )

    json.dumps(summary)
    written_summary = json.loads((output_dir / "summary.json").read_text())
    assert written_summary["artifacts"]["qualifying"]["enabled"] is True
    assert written_summary["artifacts"]["race"]["enabled"] is False
    assert (output_dir / "summary.md").exists()
