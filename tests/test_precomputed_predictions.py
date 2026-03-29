"""Tests for precomputed prediction persistence helpers."""

from src.dashboard import precomputed_predictions as store


def test_save_and_load_precompute_horizon_index_file_roundtrip(patcher, tmp_path):
    """Horizon index should persist and load from file backend when DB is disabled."""
    horizon_path = tmp_path / "precompute_horizon_index.json"

    patcher.setattr(store, "_PRECOMPUTE_HORIZON_INDEX_FILE", horizon_path)
    patcher.setattr(store, "should_read_db_first", lambda: False)
    patcher.setattr(store, "should_write_to_db", lambda: False)
    patcher.setattr(store, "should_write_to_file", lambda: True)

    store.save_precompute_horizon_index(
        year=2026,
        artifact_hash="artifact_hash",
        boundary_signature="boundary_sig",
        anchor_race_name="Australian Grand Prix",
        anchor_session_name="FP1",
        expected_targets=[
            "Australian Grand Prix",
            "Chinese Grand Prix",
            "Japanese Grand Prix",
        ],
        ready_races=["Australian Grand Prix", "Chinese Grand Prix"],
        weather_scenarios=["dry", "mixed", "rain"],
        race_boundaries={
            "Australian Grand Prix": "sig_anchor",
            "Chinese Grand Prix": "sig_future",
        },
    )

    loaded = store.load_precompute_horizon_index(year=2026, artifact_hash="artifact_hash")

    assert loaded is not None
    assert loaded["boundary_signature"] == "boundary_sig"
    assert loaded["anchor_race_name"] == "Australian Grand Prix"
    assert loaded["anchor_session_name"] == "FP1"
    assert loaded["ready_races"] == ["Australian Grand Prix", "Chinese Grand Prix"]
    assert loaded["race_boundaries"] == {
        "Australian Grand Prix": "sig_anchor",
        "Chinese Grand Prix": "sig_future",
    }


def test_load_precompute_horizon_index_db_only_does_not_merge_file_fallback(patcher):
    """DB-first horizon reads should not touch stale local file state in db_only mode."""
    patcher.setattr(store, "should_read_db_first", lambda: True)
    patcher.setattr(store, "should_write_to_db", lambda: True)
    patcher.setattr(store, "should_write_to_file", lambda: False)
    patcher.setattr(
        store,
        "RuntimeStateStore",
        lambda: type(
            "_DbStore",
            (),
            {
                "get_record": staticmethod(
                    lambda namespace, state_key: {
                        "year": 2026,
                        "artifact_hash": "artifact_hash",
                        "boundary_signature": "sig_db",
                        "anchor_race_name": "Australian Grand Prix",
                        "anchor_session_name": "FP2",
                        "expected_targets": ["Australian Grand Prix"],
                        "ready_races": ["Australian Grand Prix"],
                    }
                )
            },
        )(),
    )
    patcher.setattr(
        store,
        "_load_horizon_index_state",
        lambda: (_ for _ in ()).throw(AssertionError("file fallback should not be read")),
    )

    loaded = store.load_precompute_horizon_index(year=2026, artifact_hash="artifact_hash")

    assert loaded is not None
    assert loaded["boundary_signature"] == "sig_db"
    assert loaded["anchor_session_name"] == "FP2"


def test_load_precomputed_prediction_preserves_boundary_context(patcher, tmp_path):
    """Loaded cached predictions should expose the boundary session that produced them."""
    precompute_path = tmp_path / "precomputed_predictions.json"

    patcher.setattr(store, "_PRECOMPUTED_PREDICTIONS_FILE", precompute_path)
    patcher.setattr(store, "should_read_db_first", lambda: False)
    patcher.setattr(store, "should_write_to_db", lambda: False)
    patcher.setattr(store, "should_write_to_file", lambda: True)

    store.save_precomputed_prediction(
        year=2026,
        race_name="Chinese Grand Prix",
        weather="dry",
        artifact_hash="artifact_hash",
        boundary_signature="sig_q",
        is_sprint=True,
        prediction_results={"main_race": {"finish_order": []}},
        metadata={
            "source_race_name": "Chinese Grand Prix",
            "boundary_session_name": "Q",
        },
    )

    loaded = store.load_precomputed_prediction(
        year=2026,
        race_name="Chinese Grand Prix",
        weather="dry",
        artifact_hash="artifact_hash",
        boundary_signature="sig_q",
    )

    assert loaded is not None
    assert loaded["_prediction_context"]["boundary_session_name"] == "Q"
    assert isinstance(loaded["_prediction_context"].get("persisted_updated_at"), str)
    assert loaded["_prediction_context"]["persisted_updated_at"]


def test_list_precomputed_race_names_filters_by_year_hash_and_boundary(patcher, tmp_path):
    """Race listing should return only entries matching the requested key dimensions."""
    precompute_path = tmp_path / "precomputed_predictions.json"

    patcher.setattr(store, "_PRECOMPUTED_PREDICTIONS_FILE", precompute_path)
    patcher.setattr(store, "should_read_db_first", lambda: False)
    patcher.setattr(store, "should_write_to_db", lambda: False)
    patcher.setattr(store, "should_write_to_file", lambda: True)

    store.save_precomputed_prediction(
        year=2026,
        race_name="Australian Grand Prix",
        weather="dry",
        artifact_hash="artifact_hash",
        boundary_signature="sig_a",
        is_sprint=False,
        prediction_results={"qualifying": {"grid": []}, "race": {"finish_order": []}},
    )
    store.save_precomputed_prediction(
        year=2026,
        race_name="Chinese Grand Prix",
        weather="rain",
        artifact_hash="artifact_hash",
        boundary_signature="sig_b",
        is_sprint=True,
        prediction_results={"sprint_quali": {"grid": []}, "sprint_race": {"finish_order": []}},
    )
    store.save_precomputed_prediction(
        year=2026,
        race_name="Miami Grand Prix",
        weather="dry",
        artifact_hash="different_hash",
        boundary_signature="sig_a",
        is_sprint=True,
        prediction_results={"sprint_quali": {"grid": []}, "sprint_race": {"finish_order": []}},
    )

    listed_all = store.list_precomputed_race_names(
        year=2026,
        artifact_hash="artifact_hash",
    )
    listed_boundary = store.list_precomputed_race_names(
        year=2026,
        artifact_hash="artifact_hash",
        boundary_signature="sig_a",
    )

    assert listed_all == ["Australian Grand Prix", "Chinese Grand Prix"]
    assert listed_boundary == ["Australian Grand Prix"]


def test_save_precomputed_prediction_raises_in_db_only_when_db_write_fails(patcher):
    """DB-only mode should raise so scheduler/app can detect persistence failures."""
    patcher.setattr(store, "should_read_db_first", lambda: True)
    patcher.setattr(store, "should_write_to_db", lambda: True)
    patcher.setattr(store, "should_write_to_file", lambda: False)
    patcher.setattr(
        store,
        "RuntimeStateStore",
        lambda: type(
            "_BrokenStore",
            (),
            {
                "upsert_record": staticmethod(
                    lambda namespace, state_key, payload: (_ for _ in ()).throw(
                        RuntimeError("db unavailable")
                    )
                )
            },
        )(),
    )

    try:
        store.save_precomputed_prediction(
            year=2026,
            race_name="Australian Grand Prix",
            weather="dry",
            artifact_hash="artifact_hash",
            boundary_signature="sig_a",
            is_sprint=False,
            prediction_results={"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )
        raise AssertionError("Expected RuntimeError in db_only mode.")
    except RuntimeError as exc:
        assert "db_only mode" in str(exc)


def test_save_precompute_horizon_index_raises_in_db_only_when_db_write_fails(patcher):
    """DB-only horizon writes should raise so operators can detect silent warmup failure."""
    patcher.setattr(store, "should_read_db_first", lambda: True)
    patcher.setattr(store, "should_write_to_db", lambda: True)
    patcher.setattr(store, "should_write_to_file", lambda: False)
    patcher.setattr(
        store,
        "RuntimeStateStore",
        lambda: type(
            "_BrokenStore",
            (),
            {
                "upsert_record": staticmethod(
                    lambda namespace, state_key, payload: (_ for _ in ()).throw(
                        RuntimeError("db unavailable")
                    )
                )
            },
        )(),
    )

    try:
        store.save_precompute_horizon_index(
            year=2026,
            artifact_hash="artifact_hash",
            boundary_signature="sig_a",
            anchor_race_name="Australian Grand Prix",
            anchor_session_name="FP1",
            expected_targets=["Australian Grand Prix"],
            ready_races=["Australian Grand Prix"],
            weather_scenarios=["dry"],
            race_boundaries={"Australian Grand Prix": "sig_a"},
        )
        raise AssertionError("Expected RuntimeError in db_only mode.")
    except RuntimeError as exc:
        assert "db_only mode" in str(exc)


def test_list_precomputed_race_names_db_only_does_not_merge_file_fallback(patcher):
    """DB-first mode without file writes should not pull stale race names from local files."""
    patcher.setattr(store, "should_read_db_first", lambda: True)
    patcher.setattr(store, "should_write_to_db", lambda: True)
    patcher.setattr(store, "should_write_to_file", lambda: False)
    patcher.setattr(
        store,
        "RuntimeStateStore",
        lambda: type(
            "_DbStore",
            (),
            {
                "load_namespace": staticmethod(
                    lambda namespace: {
                        "k1": {
                            "year": 2026,
                            "artifact_hash": "artifact_hash",
                            "boundary_signature": "sig_a",
                            "race_name": "Australian Grand Prix",
                        }
                    }
                )
            },
        )(),
    )
    patcher.setattr(
        store,
        "_load_file_state",
        lambda: (_ for _ in ()).throw(AssertionError("file fallback should not be read")),
    )

    listed = store.list_precomputed_race_names(
        year=2026,
        artifact_hash="artifact_hash",
        boundary_signature="sig_a",
    )

    assert listed == ["Australian Grand Prix"]


def test_prune_db_namespace_entries_uses_key_only_strategy_when_available():
    """DB pruning should avoid loading full namespace payloads when key APIs are available."""

    class _Store:
        def __init__(self):
            self.deleted: list[str] = []

        def count_records(self, namespace: str) -> int:
            assert namespace == "precomputed_predictions"
            return 5

        def list_oldest_state_keys(self, namespace: str, *, limit: int) -> list[str]:
            assert namespace == "precomputed_predictions"
            assert limit == 2
            return ["k1", "k2"]

        def load_namespace(self, namespace: str):
            raise AssertionError("Full namespace load should not be used in optimized prune.")

        def delete_records(self, namespace: str, keys: list[str]) -> None:
            assert namespace == "precomputed_predictions"
            self.deleted = list(keys)

    fake_store = _Store()
    store._prune_db_namespace_entries(
        "precomputed_predictions",
        max_entries=3,
        store=fake_store,
    )
    assert fake_store.deleted == ["k1", "k2"]
