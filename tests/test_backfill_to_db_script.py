from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module():
    """Load the backfill script as an importable test module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_to_db.py"
    spec = importlib.util.spec_from_file_location("backfill_to_db_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discover_artifacts_includes_snapshots_and_skips_runtime_state_artifacts(tmp_path):
    module = _load_script_module()

    car_file = tmp_path / "processed" / "car_characteristics" / "2026_car_characteristics.json"
    car_file.parent.mkdir(parents=True, exist_ok=True)
    car_file.write_text(json.dumps({"version": 15, "teams": {}}))

    snapshot_file = (
        tmp_path / "car_characteristics_snapshot" / "2026" / "Australian Grand Prix" / "FP1.json"
    )
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file.write_text(
        json.dumps(
            {
                "version": 1,
                "event_name": "Australian Grand Prix",
                "session_name": "FP1",
                "teams": {},
            }
        )
    )

    driver_file = (
        tmp_path / "processed" / "driver_characteristics" / "2026_driver_characteristics.json"
    )
    driver_file.parent.mkdir(parents=True, exist_ok=True)
    driver_file.write_text(json.dumps({"version": 3, "year": 2026, "drivers": {}}))

    practice_state = tmp_path / "systems" / "practice_characteristics_state.json"
    practice_state.parent.mkdir(parents=True, exist_ok=True)
    practice_state.write_text(json.dumps({"races": {"2026::Australian Grand Prix": {}}}))

    artifacts = module.discover_artifacts(tmp_path)
    artifact_lookup = {(item["artifact_type"], item["artifact_key"]) for item in artifacts}

    assert ("car_characteristics", "2026::car_characteristics") in artifact_lookup
    assert (
        "car_characteristics_snapshot",
        "2026::Australian Grand Prix::FP1",
    ) in artifact_lookup
    assert ("driver_characteristics", "2026::driver_characteristics") in artifact_lookup
    assert all(artifact_type != "practice_state" for artifact_type, _key in artifact_lookup)


def test_discover_runtime_state_records_maps_files_to_namespaces(tmp_path):
    module = _load_script_module()

    learning_file = tmp_path / "learning_state.json"
    learning_file.write_text(json.dumps({"season": 2026, "races_completed": 3}))

    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "practice_characteristics_state.json").write_text(
        json.dumps({"races": {"2026::Australian Grand Prix": {"sessions": ["FP1", "Q"]}}})
    )
    (systems_dir / "event_boundary_refresh_state.json").write_text(
        json.dumps({"races": {"2026::Australian Grand Prix": {"latest_elapsed_session": "R"}}})
    )
    (systems_dir / "precomputed_predictions.json").write_text(
        json.dumps({"entries": {"pred-key": {"updated_at": "2026-03-10T15:13:51+00:00"}}})
    )
    (systems_dir / "precomputed_base_features.json").write_text(
        json.dumps({"entries": {"base-key": {"updated_at": "2026-03-10T15:13:51+00:00"}}})
    )
    (systems_dir / "precompute_horizon_index.json").write_text(
        json.dumps({"entries": {"2026::hash": {"ready_races": ["Australian Grand Prix"]}}})
    )
    (systems_dir / "session_automation_schedule.json").write_text(
        json.dumps({"2026": {"enabled": True, "weather": "dry"}})
    )

    payloads = module.discover_runtime_state_records(tmp_path)
    payload_by_namespace = {item["namespace"]: item for item in payloads}

    assert payload_by_namespace["race_learning"]["records"]["2026"]["races_completed"] == 3
    assert payload_by_namespace["practice_characteristics"]["records"][
        "2026::Australian Grand Prix"
    ]["sessions"] == ["FP1", "Q"]
    assert (
        payload_by_namespace["event_boundary_refresh"]["records"]["2026::Australian Grand Prix"][
            "latest_elapsed_session"
        ]
        == "R"
    )
    assert "pred-key" in payload_by_namespace["precomputed_predictions"]["records"]
    assert "base-key" in payload_by_namespace["precomputed_prediction_base_features"]["records"]
    assert "2026::hash" in payload_by_namespace["prediction_precompute_horizon_index"]["records"]
    assert payload_by_namespace["session_automation_schedule"]["records"]["2026"]["enabled"] is True


def test_backfill_runtime_state_batches_upserts(monkeypatch):
    module = _load_script_module()
    calls: list[tuple[str, dict[str, dict[str, object]]]] = []

    class _FakeStore:
        """Capture runtime-state upserts for assertions."""

        def upsert_many(self, namespace: str, records: dict[str, dict[str, object]]) -> None:
            calls.append((namespace, records))

    monkeypatch.setattr(module, "RuntimeStateStore", lambda: _FakeStore())

    success, failure = module.backfill_runtime_state(
        [
            {
                "file_path": Path("data/systems/practice_characteristics_state.json"),
                "namespace": "practice_characteristics",
                "records": {
                    "2026::Australian Grand Prix": {"sessions": ["FP1"]},
                    "2026::Chinese Grand Prix": {"sessions": ["FP1", "SQ"]},
                },
                "record_count": 2,
                "checksum": "abc123",
            }
        ],
        dry_run=False,
        batch_size=1,
    )

    assert success == 2
    assert failure == 0
    assert calls == [
        ("practice_characteristics", {"2026::Australian Grand Prix": {"sessions": ["FP1"]}}),
        (
            "practice_characteristics",
            {"2026::Chinese Grand Prix": {"sessions": ["FP1", "SQ"]}},
        ),
    ]
