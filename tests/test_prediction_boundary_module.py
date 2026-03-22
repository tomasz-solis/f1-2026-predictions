"""Tests for extracted dashboard boundary and warmed-fallback helpers."""

import logging

from src.dashboard import prediction_boundary


def test_get_prediction_precompute_settings_normalizes_weather_and_numeric_fields():
    settings = prediction_boundary.get_prediction_precompute_settings(
        get_prediction_precompute_config_fn=lambda: {
            "weather_scenarios": ["DRY", "rain", "bogus", "dry"],
            "horizon_races": "4",
            "max_file_entries": "64",
        },
        logger=logging.getLogger("test"),
    )

    assert settings["weather_scenarios"] == ["dry", "rain"]
    assert settings["horizon_races"] == 4
    assert settings["max_file_entries"] == 64


def test_resolve_precompute_targets_skips_testing_rows():
    targets = prediction_boundary.resolve_precompute_targets(
        year=2026,
        race_name="Bahrain Grand Prix",
        horizon_races=3,
        get_schedule_rows_fn=lambda year: (
            ("Pre-Season Testing", "testing"),
            ("Bahrain Grand Prix", "conventional"),
            ("Saudi Arabian Grand Prix", "conventional"),
            ("In-Season Track Test", "testing"),
            ("Australian Grand Prix", "conventional"),
        ),
        logger=logging.getLogger("test"),
    )

    assert targets == ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix"]


def test_resolve_persisted_boundary_fallback_prefers_race_specific_boundary():
    fallback = prediction_boundary.resolve_persisted_boundary_fallback(
        year=2026,
        race_name="Chinese Grand Prix",
        artifact_hash="artifact_hash",
        current_boundary_signature="sig_sq",
        current_boundary_session_name="SQ",
        load_precompute_horizon_index_fn=lambda **kwargs: {
            "ready_races": ["Chinese Grand Prix"],
            "anchor_race_name": "Chinese Grand Prix",
            "anchor_session_name": "FP1",
            "boundary_signature": "sig_anchor",
            "race_boundaries": {"Chinese Grand Prix": "sig_fp1"},
        },
    )

    assert fallback == {
        "current_boundary_signature": "sig_sq",
        "current_boundary_session_name": "SQ",
        "served_boundary_signature": "sig_fp1",
        "served_boundary_session_name": "FP1",
    }


def test_load_warmed_boundary_fallback_prediction_notifies_and_marks_mode():
    notifications: list[str] = []

    loaded = prediction_boundary.load_warmed_boundary_fallback_prediction(
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
        artifact_hash="artifact_hash",
        fallback_metadata={
            "served_boundary_signature": "sig_fp1",
            "served_boundary_session_name": "FP1",
        },
        notify_fn=notifications.append,
        load_precomputed_prediction_fn=lambda **kwargs: {
            "sprint_quali": {"grid": []},
            "sprint_race": {"finish_order": []},
        },
    )

    assert loaded is not None
    prediction_payload, fallback_meta = loaded
    assert "sprint_quali" in prediction_payload
    assert notifications == [
        "Current checkpoint is ahead of the warmed horizon; serving the latest "
        "persisted checkpoint until warmup catches up..."
    ]
    assert fallback_meta["mode"] == "served_warmed_boundary"
    assert fallback_meta["warmed_boundary_signature"] == "sig_fp1"
    assert fallback_meta["warmed_boundary_session_name"] == "FP1"
