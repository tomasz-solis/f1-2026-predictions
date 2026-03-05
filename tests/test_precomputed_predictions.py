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
        anchor_race_name="Bahrain Grand Prix",
        anchor_session_name="FP1",
        expected_targets=[
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
        ],
        ready_races=["Bahrain Grand Prix", "Saudi Arabian Grand Prix"],
        weather_scenarios=["dry", "mixed", "rain"],
        race_boundaries={
            "Bahrain Grand Prix": "sig_anchor",
            "Saudi Arabian Grand Prix": "sig_future",
        },
    )

    loaded = store.load_precompute_horizon_index(year=2026, artifact_hash="artifact_hash")

    assert loaded is not None
    assert loaded["boundary_signature"] == "boundary_sig"
    assert loaded["anchor_race_name"] == "Bahrain Grand Prix"
    assert loaded["anchor_session_name"] == "FP1"
    assert loaded["ready_races"] == ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]
    assert loaded["race_boundaries"] == {
        "Bahrain Grand Prix": "sig_anchor",
        "Saudi Arabian Grand Prix": "sig_future",
    }


def test_list_precomputed_race_names_filters_by_year_hash_and_boundary(patcher, tmp_path):
    """Race listing should return only entries matching the requested key dimensions."""
    precompute_path = tmp_path / "precomputed_predictions.json"

    patcher.setattr(store, "_PRECOMPUTED_PREDICTIONS_FILE", precompute_path)
    patcher.setattr(store, "should_read_db_first", lambda: False)
    patcher.setattr(store, "should_write_to_db", lambda: False)
    patcher.setattr(store, "should_write_to_file", lambda: True)

    store.save_precomputed_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
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

    assert listed_all == ["Bahrain Grand Prix", "Chinese Grand Prix"]
    assert listed_boundary == ["Bahrain Grand Prix"]


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
            race_name="Bahrain Grand Prix",
            weather="dry",
            artifact_hash="artifact_hash",
            boundary_signature="sig_a",
            is_sprint=False,
            prediction_results={"qualifying": {"grid": []}, "race": {"finish_order": []}},
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
                            "race_name": "Bahrain Grand Prix",
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

    assert listed == ["Bahrain Grand Prix"]
