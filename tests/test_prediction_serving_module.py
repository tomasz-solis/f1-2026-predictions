"""Tests for extracted persisted-serving request-path helpers."""

import logging

import pytest

from src.dashboard import prediction_serving
from src.dashboard.live_prediction_flow import PrecomputedPredictionUnavailableError


def test_load_served_prediction_bundle_prefers_current_persisted_payload():
    notifications: list[str] = []

    bundle = prediction_serving.load_served_prediction_bundle(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        is_sprint=False,
        boundary_refresh={
            "boundary_signature": "stable_sig",
            "latest_elapsed_session": "FP2",
        },
        session_detector=object(),
        precompute_settings={"horizon_races": 3},
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        compute_artifact_hash_fn=lambda versions: "artifact_hash",
        resolve_race_boundary_context_fn=lambda **kwargs: ("unused", "PRE"),
        resolve_precompute_targets_fn=lambda **kwargs: ["Bahrain Grand Prix"],
        prediction_cache_key_fn=lambda **kwargs: "cache_key",
        get_cached_prediction_fn=lambda key: None,
        resolve_persisted_boundary_fallback_fn=lambda **kwargs: None,
        load_precomputed_prediction_fn=lambda **kwargs: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
        cached_prediction_matches_persisted_fn=lambda **kwargs: False,
        store_cached_prediction_fn=lambda key, prediction: None,
        load_warmed_boundary_fallback_prediction_fn=lambda **kwargs: None,
        load_precompute_horizon_index_fn=lambda **kwargs: {
            "boundary_signature": "stable_sig",
            "anchor_race_name": "Bahrain Grand Prix",
            "expected_targets": ["Bahrain Grand Prix"],
            "ready_races": ["Bahrain Grand Prix"],
        },
        served_prediction_boundary_session_name_fn=lambda **kwargs: "FP2",
        prediction_unavailable_error_type=PrecomputedPredictionUnavailableError,
        notify_fn=notifications.append,
        logger=logging.getLogger("test"),
    )

    assert notifications == ["Loaded persisted prediction..."]
    assert bundle["prediction_cache_hit"] is False
    assert bundle["boundary_session_name"] == "FP2"
    assert bundle["precompute_summary"]["ready_races"] == ["Bahrain Grand Prix"]


def test_load_served_prediction_bundle_uses_warmed_fallback_when_current_boundary_missing():
    notifications: list[str] = []

    bundle = prediction_serving.load_served_prediction_bundle(
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
        is_sprint=True,
        boundary_refresh={
            "boundary_signature": "sig_sq",
            "latest_elapsed_session": "SQ",
        },
        session_detector=object(),
        precompute_settings={"horizon_races": 3},
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        compute_artifact_hash_fn=lambda versions: "artifact_hash",
        resolve_race_boundary_context_fn=lambda **kwargs: ("unused", "PRE"),
        resolve_precompute_targets_fn=lambda **kwargs: ["Chinese Grand Prix"],
        prediction_cache_key_fn=lambda **kwargs: "cache_key",
        get_cached_prediction_fn=lambda key: None,
        resolve_persisted_boundary_fallback_fn=lambda **kwargs: {
            "served_boundary_signature": "sig_fp1",
            "served_boundary_session_name": "FP1",
        },
        load_precomputed_prediction_fn=lambda **kwargs: None,
        cached_prediction_matches_persisted_fn=lambda **kwargs: False,
        store_cached_prediction_fn=lambda key, prediction: None,
        load_warmed_boundary_fallback_prediction_fn=lambda **kwargs: (
            {"sprint_quali": {"grid": []}, "sprint_race": {"finish_order": []}},
            {
                "served_boundary_signature": "sig_fp1",
                "served_boundary_session_name": "FP1",
                "mode": "served_warmed_boundary",
            },
        ),
        load_precompute_horizon_index_fn=lambda **kwargs: None,
        served_prediction_boundary_session_name_fn=lambda **kwargs: "FP1",
        prediction_unavailable_error_type=PrecomputedPredictionUnavailableError,
        notify_fn=notifications.append,
        logger=logging.getLogger("test"),
    )

    assert bundle["prediction_results"]["sprint_quali"]["grid"] == []
    assert bundle["boundary_session_name"] == "FP1"
    assert bundle["boundary_fallback"]["mode"] == "served_warmed_boundary"
    assert notifications == []


def test_load_served_prediction_bundle_fails_closed_when_nothing_is_available():
    with pytest.raises(
        PrecomputedPredictionUnavailableError,
        match=r"Bahrain Grand Prix 2026 \[dry\] at checkpoint FP2",
    ):
        prediction_serving.load_served_prediction_bundle(
            race_name="Bahrain Grand Prix",
            weather="dry",
            year=2026,
            is_sprint=False,
            boundary_refresh={
                "boundary_signature": "stable_sig",
                "latest_elapsed_session": "FP2",
            },
            session_detector=object(),
            precompute_settings={"horizon_races": 3},
            get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
            compute_artifact_hash_fn=lambda versions: "artifact_hash",
            resolve_race_boundary_context_fn=lambda **kwargs: ("unused", "PRE"),
            resolve_precompute_targets_fn=lambda **kwargs: ["Bahrain Grand Prix"],
            prediction_cache_key_fn=lambda **kwargs: "cache_key",
            get_cached_prediction_fn=lambda key: None,
            resolve_persisted_boundary_fallback_fn=lambda **kwargs: None,
            load_precomputed_prediction_fn=lambda **kwargs: None,
            cached_prediction_matches_persisted_fn=lambda **kwargs: False,
            store_cached_prediction_fn=lambda key, prediction: None,
            load_warmed_boundary_fallback_prediction_fn=lambda **kwargs: None,
            load_precompute_horizon_index_fn=lambda **kwargs: None,
            served_prediction_boundary_session_name_fn=lambda **kwargs: "FP2",
            prediction_unavailable_error_type=PrecomputedPredictionUnavailableError,
            notify_fn=lambda message: None,
            logger=logging.getLogger("test"),
        )


def test_load_served_prediction_bundle_serves_latest_when_artifact_hash_differs():
    notifications: list[str] = []
    seen_calls: list[dict] = []

    def _latest_for_boundary(**kwargs):
        seen_calls.append(kwargs)
        return {"qualifying": {"grid": []}, "race": {"finish_order": []}}

    bundle = prediction_serving.load_served_prediction_bundle(
        race_name="Belgian Grand Prix",
        weather="dry",
        year=2026,
        is_sprint=False,
        boundary_refresh={"boundary_signature": "stable_sig", "latest_elapsed_session": "FP2"},
        session_detector=object(),
        precompute_settings={"horizon_races": 3},
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        compute_artifact_hash_fn=lambda versions: "local_hash_that_differs",
        resolve_race_boundary_context_fn=lambda **kwargs: ("unused", "PRE"),
        resolve_precompute_targets_fn=lambda **kwargs: ["Belgian Grand Prix"],
        prediction_cache_key_fn=lambda **kwargs: "cache_key",
        get_cached_prediction_fn=lambda key: None,
        resolve_persisted_boundary_fallback_fn=lambda **kwargs: None,
        load_precomputed_prediction_fn=lambda **kwargs: None,  # exact key misses
        cached_prediction_matches_persisted_fn=lambda **kwargs: False,
        store_cached_prediction_fn=lambda key, prediction: None,
        load_warmed_boundary_fallback_prediction_fn=lambda **kwargs: None,
        load_precompute_horizon_index_fn=lambda **kwargs: None,
        load_latest_prediction_for_boundary_fn=_latest_for_boundary,
        served_prediction_boundary_session_name_fn=lambda **kwargs: "FP2",
        prediction_unavailable_error_type=PrecomputedPredictionUnavailableError,
        notify_fn=notifications.append,
        logger=logging.getLogger("test"),
    )

    assert bundle["prediction_results"]["qualifying"]["grid"] == []
    # Fallback keeps the checkpoint exact, only relaxes the model version.
    assert seen_calls and seen_calls[0]["boundary_signature"] == "stable_sig"
    assert any("latest available forecast" in note for note in notifications)
