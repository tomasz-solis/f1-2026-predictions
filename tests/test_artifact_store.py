from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.persistence.artifact_store as artifact_store_module
from src.persistence.artifact_store import ArtifactStore


def test_get_file_path_mappings(tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    car_path = store._get_file_path("car_characteristics", "2026::car_characteristics")
    driver_path = store._get_file_path("driver_characteristics", "2026::driver_characteristics")
    legacy_driver_path = store._get_file_path("driver_characteristics", "driver_characteristics")
    debuts_path = store._get_file_path("driver_debuts", "driver_debuts")
    track_path = store._get_file_path("track_characteristics", "2026::track_characteristics")
    prediction_path = store._get_file_path("prediction", "2026::Bahrain Grand Prix::qualifying")
    default_path = store._get_file_path("custom", "a::b")
    learning_state_path = store._get_file_path("learning_state", "2026::learning_state")
    legacy_learning_state_path = store._get_file_path("learning_state", "learning_state")

    assert str(car_path).endswith("processed/car_characteristics/2026_car_characteristics.json")
    assert str(driver_path).endswith(
        "processed/driver_characteristics/2026_driver_characteristics.json"
    )
    assert str(legacy_driver_path).endswith("processed/driver_characteristics.json")
    assert str(debuts_path).endswith("driver_debuts.json")
    assert str(track_path).endswith(
        "processed/track_characteristics/2026_track_characteristics.json"
    )
    assert str(learning_state_path).endswith("learning_state/2026_learning_state.json")
    assert str(legacy_learning_state_path).endswith("learning_state.json")
    assert str(prediction_path).endswith(
        "predictions/2026/bahrain_grand_prix/bahrain_grand_prix_qualifying.json"
    )
    assert str(default_path).endswith("custom/a/b.json")


def test_write_and_read_file_roundtrip(tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    payload = {"value": 7}
    store._write_file("custom", "alpha::beta", payload)

    loaded = store._read_file("custom", "alpha::beta")
    missing = store._read_file("custom", "missing")

    assert loaded == payload
    assert missing is None


def test_save_artifact_file_only_auto_version(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    patcher.setattr(artifact_store_module, "should_write_to_file", lambda: True)
    patcher.setattr(artifact_store_module, "should_write_to_db", lambda: False)
    patcher.setattr(store, "get_latest_version", lambda artifact_type, artifact_key: 2)

    result = store.save_artifact(
        artifact_type="custom",
        artifact_key="alpha",
        data={"ok": True},
        run_id="not-a-uuid",
    )

    assert result["version"] == 3
    assert result["storage"] == "file_only"
    assert result["run_id"] is None


def test_save_artifact_prefers_db_result(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    patcher.setattr(artifact_store_module, "should_write_to_file", lambda: False)
    patcher.setattr(artifact_store_module, "should_write_to_db", lambda: True)
    mocked_write_db = MagicMock(return_value={"id": "db-row", "version": 5})
    patcher.setattr(store, "_write_db", mocked_write_db)

    result = store.save_artifact(
        artifact_type="custom",
        artifact_key="alpha",
        data={"ok": True},
        version=5,
    )

    assert result["id"] == "db-row"
    mocked_write_db.assert_called_once()


def test_save_artifact_raises_when_all_writes_fail(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    patcher.setattr(artifact_store_module, "should_write_to_file", lambda: True)
    patcher.setattr(artifact_store_module, "should_write_to_db", lambda: True)
    patcher.setattr(store, "_write_file", MagicMock(side_effect=RuntimeError("file down")))
    patcher.setattr(store, "_write_db", MagicMock(side_effect=RuntimeError("db down")))

    with pytest.raises(RuntimeError, match="All writes failed"):
        store.save_artifact("custom", "alpha", {"ok": True}, version=1)


def test_load_artifact_db_first_then_file_fallback(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    patcher.setattr(artifact_store_module, "should_read_db_first", lambda: True)
    patcher.setattr(store, "_read_db", MagicMock(return_value={"from": "db"}))
    patcher.setattr(store, "_read_file", MagicMock(return_value={"from": "file"}))

    assert store.load_artifact("custom", "alpha") == {"from": "db"}

    patcher.setattr(store, "_read_db", MagicMock(side_effect=RuntimeError("db fail")))
    assert store.load_artifact("custom", "alpha") == {"from": "file"}


def test_list_artifacts_db_failure_falls_back_to_files(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    patcher.setattr(artifact_store_module, "should_read_db_first", lambda: True)
    patcher.setattr(store, "_list_db", MagicMock(side_effect=RuntimeError("db fail")))
    patcher.setattr(store, "_list_files", MagicMock(return_value=[{"artifact_key": "a"}]))

    result = store.list_artifacts("custom")
    assert result == [{"artifact_key": "a"}]


def test_get_latest_version_db_file_and_default(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)
    patcher.setattr(artifact_store_module, "should_read_db_first", lambda: True)

    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.execute.return_value = SimpleNamespace(data=[{"version": 7}])
    supabase = MagicMock()
    supabase.table.return_value = query
    patcher.setattr(artifact_store_module, "get_supabase_client", lambda: supabase)

    assert store.get_latest_version("custom", "alpha") == 7

    patcher.setattr(
        artifact_store_module,
        "get_supabase_client",
        lambda: (_ for _ in ()).throw(RuntimeError("no db")),
    )
    patcher.setattr(store, "_read_file", lambda artifact_type, artifact_key: {"version": 3})
    assert store.get_latest_version("custom", "alpha") == 3

    patcher.setattr(store, "_read_file", lambda artifact_type, artifact_key: None)
    assert store.get_latest_version("custom", "alpha") == 0


def test_write_db_read_db_and_list_db(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    query = MagicMock()
    query.upsert.return_value = query
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.like.return_value = query
    query.execute.side_effect = [
        SimpleNamespace(data=[{"id": "row-1", "version": 2}]),
        SimpleNamespace(data=[{"data": {"hello": "world"}}]),
        SimpleNamespace(data=[{"artifact_key": "alpha", "version": 2}]),
    ]

    supabase = MagicMock()
    supabase.table.return_value = query
    patcher.setattr(artifact_store_module, "get_supabase_client", lambda: supabase)

    row = store._write_db(
        artifact_type="custom",
        artifact_key="alpha",
        data={"hello": "world"},
        version=2,
        run_uuid=None,
    )
    assert row["id"] == "row-1"

    read = store._read_db("custom", "alpha", version="latest", run_id="run-1")
    assert read == {"hello": "world"}

    listed = store._list_db("custom", key_prefix="a", limit=10)
    assert listed == [{"artifact_key": "alpha", "version": 2}]


def test_write_db_raises_when_no_rows_returned(patcher, tmp_path):
    store = ArtifactStore(data_root=tmp_path)

    query = MagicMock()
    query.upsert.return_value = query
    query.execute.return_value = SimpleNamespace(data=[])

    supabase = MagicMock()
    supabase.table.return_value = query
    patcher.setattr(artifact_store_module, "get_supabase_client", lambda: supabase)

    with pytest.raises(RuntimeError, match="DB insert returned no data"):
        store._write_db("custom", "alpha", {"ok": True}, version=1, run_uuid=None)


def test_list_files_scans_fallback_storage_and_sorts_by_mtime(tmp_path):
    store = ArtifactStore(data_root=tmp_path)
    artifacts_dir = tmp_path / "custom"
    artifacts_dir.mkdir(parents=True)

    older = artifacts_dir / "older.json"
    older.write_text('{"version": 1, "value": "old"}')
    newer = artifacts_dir / "newer.json"
    newer.write_text('{"version": 2, "value": "new"}')
    broken = artifacts_dir / "broken.json"
    broken.write_text("{not-json")

    base_ts = 1_700_000_000
    os.utime(older, (base_ts, base_ts))
    os.utime(newer, (base_ts + 120, base_ts + 120))

    rows = store._list_files("custom", key_prefix=None, limit=10)

    assert [row["artifact_key"] for row in rows] == ["newer", "older"]
    assert rows[0]["version"] == 2
    assert rows[1]["version"] == 1


def test_list_files_prediction_supports_key_prefix_filter(tmp_path):
    store = ArtifactStore(data_root=tmp_path)
    prediction_file = (
        tmp_path
        / "predictions"
        / "2026"
        / "bahrain_grand_prix"
        / "bahrain_grand_prix_qualifying.json"
    )
    prediction_file.parent.mkdir(parents=True)
    prediction_file.write_text('{"version": 3, "grid": []}')

    rows = store._list_files(
        "prediction",
        key_prefix="2026::bahrain_grand_prix::qualifying",
        limit=5,
    )

    assert len(rows) == 1
    assert rows[0]["artifact_key"] == "2026::bahrain_grand_prix::qualifying"
    assert rows[0]["version"] == 3


def test_list_files_driver_debuts_supports_single_json(tmp_path):
    store = ArtifactStore(data_root=tmp_path)
    debuts_file = tmp_path / "driver_debuts.json"
    debuts_file.write_text('{"driver_debuts": {"HAM": 2007}, "version": 1}')

    rows = store._list_files("driver_debuts", key_prefix="driver", limit=5)

    assert len(rows) == 1
    assert rows[0]["artifact_key"] == "driver_debuts"
    assert rows[0]["version"] == 1
