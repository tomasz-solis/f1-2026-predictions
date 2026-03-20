"""Tests for the Supabase dashboard-artifact cleanup script."""

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "normalize_dashboard_artifacts_in_db.py"
    )
    spec = importlib.util.spec_from_file_location(
        "normalize_dashboard_artifacts_in_db_script",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plan_cleanup_prefers_richer_prediction_row():
    """Prediction cleanup should collapse duplicate versions into one canonical row."""
    module = _load_script_module()
    rows = [
        module.ArtifactRow(
            id="older",
            artifact_type="prediction",
            artifact_key="2026::Chinese Grand Prix::FP1",
            version=1,
            run_id=None,
            data={
                "metadata": {
                    "year": 2026,
                    "race_name": "Chinese Grand Prix",
                    "session_name": "FP1",
                },
                "actuals": {"qualifying": None, "race": None, "targets": {}},
            },
            created_at="2026-03-20T10:00:00+00:00",
            updated_at="2026-03-20T10:00:00+00:00",
        ),
        module.ArtifactRow(
            id="winner",
            artifact_type="prediction",
            artifact_key="2026::Chinese Grand Prix::FP1",
            version=2,
            run_id="run-1",
            data={
                "metadata": {
                    "year": 2026,
                    "race_name": "Chinese Grand Prix",
                    "session_name": "FP1",
                },
                "actuals": {
                    "qualifying": [{"position": 1, "driver": "VER"}],
                    "race": None,
                    "targets": {"grand_prix_race": [{"position": 1, "driver": "VER"}]},
                },
            },
            created_at="2026-03-20T11:00:00+00:00",
            updated_at="2026-03-20T11:00:00+00:00",
        ),
    ]

    plan = module.plan_cleanup(rows)

    assert plan.scanned_rows == 2
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.artifact_type == "prediction"
    assert action.normalized_key == "2026::Chinese Grand Prix::FP1"
    assert action.winner_id == "winner"
    assert action.canonical_version == 1
    assert action.update_winner is True
    assert action.delete_ids == ("older",)


def test_plan_cleanup_normalizes_accuracy_snapshot_metadata():
    """Accuracy-snapshot cleanup should normalize key casing even without duplicates."""
    module = _load_script_module()
    row = module.ArtifactRow(
        id="snapshot-row",
        artifact_type="accuracy_snapshot",
        artifact_key="2026:: Chinese   Grand Prix :: fp1 :: Grand_Prix_Race ",
        version=1,
        run_id=None,
        data={
            "metadata": {
                "year": 2026,
                "race_name": " Chinese   Grand Prix ",
                "checkpoint_session": " fp1 ",
                "target_key": " Grand_Prix_Race ",
            }
        },
        created_at="2026-03-20T11:00:00+00:00",
        updated_at="2026-03-20T11:05:00+00:00",
    )

    normalized = module.normalize_artifact_row(row)
    plan = module.plan_cleanup([row])

    assert normalized.normalized_key == "2026::Chinese Grand Prix::FP1::grand_prix_race"
    assert normalized.normalized_data["metadata"]["checkpoint_session"] == "FP1"
    assert normalized.normalized_data["metadata"]["target_key"] == "grand_prix_race"
    assert len(plan.actions) == 1
    assert plan.actions[0].winner_id == "snapshot-row"
    assert plan.actions[0].update_winner is True
    assert plan.actions[0].delete_ids == ()
