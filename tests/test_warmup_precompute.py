"""Tests for background warmup precompute orchestration."""

from datetime import UTC, datetime

import pytest

from src.dashboard import warmup


def test_run_warmup_precompute_cycle_is_idempotent(patcher):
    """Second warmup run should reuse existing base/prediction payloads without recompute."""
    fixed_now = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(warmup, "_refresh_anchor_practice_characteristics", lambda **kwargs: {})
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="PRE",
            expected_checkpoint="PRE",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="boundary_sig",
        ),
    )
    patcher.setattr(warmup, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(warmup, "compute_artifact_hash", lambda artifact_versions: "artifact_hash")
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: object())
    patcher.setattr(warmup, "is_sprint_weekend", lambda year, race_name: False)

    base_cache: dict[tuple[str, str, str, str, str], dict] = {}
    prediction_cache: dict[tuple[str, str, str, str, str], dict] = {}
    base_calls = {"compute": 0}
    weather_calls = {"compute": 0}
    horizon_calls: list[dict] = []

    def _base_key(kwargs: dict) -> tuple[str, str, str, str, str]:
        return (
            str(kwargs["year"]),
            str(kwargs["race_name"]),
            str(kwargs["checkpoint"]),
            str(kwargs["artifact_hash"]),
            str(kwargs["boundary_signature"]),
        )

    def _prediction_key(kwargs: dict) -> tuple[str, str, str, str, str]:
        return (
            str(kwargs["year"]),
            str(kwargs["race_name"]),
            str(kwargs["weather"]),
            str(kwargs["artifact_hash"]),
            str(kwargs["boundary_signature"]),
        )

    patcher.setattr(
        warmup,
        "load_precomputed_base_features",
        lambda **kwargs: base_cache.get(_base_key(kwargs)),
    )
    patcher.setattr(
        warmup,
        "save_precomputed_base_features",
        lambda **kwargs: base_cache.__setitem__(_base_key(kwargs), kwargs["base_features"]),
    )
    patcher.setattr(
        warmup,
        "load_precomputed_prediction",
        lambda **kwargs: prediction_cache.get(_prediction_key(kwargs)),
    )
    patcher.setattr(
        warmup,
        "save_precomputed_prediction",
        lambda **kwargs: prediction_cache.__setitem__(
            _prediction_key(kwargs), kwargs["prediction_results"]
        ),
    )
    patcher.setattr(
        warmup,
        "compute_base_features",
        lambda *args, **kwargs: (
            base_calls.__setitem__("compute", base_calls["compute"] + 1),
            {
                "is_sprint": False,
                "qualifying": {"grid": []},
                "qualifying_grid_for_race": [],
                "race_input_confidence": 0.7,
                "timing": {"qualifying": 0.1},
            },
        )[1],
    )
    patcher.setattr(
        warmup,
        "compute_weather_predictions",
        lambda base_features, weather, predictor, year, target_race: (
            weather_calls.__setitem__("compute", weather_calls["compute"] + 1),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )
    patcher.setattr(
        warmup, "save_precompute_horizon_index", lambda **kwargs: horizon_calls.append(kwargs)
    )

    first = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)
    second = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert first.status == "success"
    assert first.base_generated == 1
    assert first.predictions_generated == 3
    assert first.ready_races == ["Bahrain Grand Prix"]
    assert second.status == "success"
    assert second.base_reused == 1
    assert second.predictions_reused == 3
    assert second.base_generated == 0
    assert second.predictions_generated == 0
    assert base_calls == {"compute": 1}
    assert weather_calls == {"compute": 3}
    assert len(horizon_calls) == 2
    assert horizon_calls[0]["race_boundaries"] == {"Bahrain Grand Prix": "boundary_sig"}


def test_run_warmup_precompute_cycle_refreshes_practice_before_hashing(patcher):
    """Warmup should refresh practice artifacts before deriving the precompute hash."""
    fixed_now = datetime(2026, 3, 13, 8, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Chinese Grand Prix",
            anchor_is_sprint=True,
            target_races=("Chinese Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="FP1",
            expected_checkpoint="FP1",
            latest_ready_checkpoint="FP1",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="sig_fp1",
        ),
    )
    patcher.setattr(warmup, "is_sprint_weekend", lambda year, race_name: True)

    call_order: list[str] = []
    artifact_versions = {"car_characteristics::2026::car_characteristics": (37, "before")}

    def _refresh_anchor_practice_characteristics(**kwargs):
        del kwargs
        call_order.append("practice_refresh")
        artifact_versions["car_characteristics::2026::car_characteristics"] = (38, "after")
        return {
            "updated": True,
            "completed_fp_sessions": ["FP1"],
            "teams_updated": 10,
        }

    patcher.setattr(
        warmup,
        "_refresh_anchor_practice_characteristics",
        _refresh_anchor_practice_characteristics,
    )
    patcher.setattr(
        warmup,
        "get_artifact_versions",
        lambda year=2026: (call_order.append("artifact_versions"), dict(artifact_versions))[1],
    )
    patcher.setattr(
        warmup,
        "compute_artifact_hash",
        lambda versions: f"artifact_hash_v{versions['car_characteristics::2026::car_characteristics'][0]}",
    )
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: object())
    patcher.setattr(warmup, "load_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(
        warmup,
        "compute_base_features",
        lambda *args, **kwargs: {
            "is_sprint": True,
            "sprint_quali": {"grid": [], "grid_source": "PREDICTED"},
            "sprint_grid_for_race": [],
            "sprint_race_input_confidence": 0.7,
            "main_quali": {"grid": [], "grid_source": "PREDICTED"},
            "main_grid_for_race": [],
            "main_race_input_confidence": 0.7,
            "timing": {"sprint_quali": 0.1, "main_quali": 0.1},
        },
    )
    saved_prediction_hashes: list[str] = []
    patcher.setattr(
        warmup,
        "compute_weather_predictions",
        lambda base_features, weather, predictor, year, target_race: {
            "sprint_quali": {"grid": []},
            "sprint_race": {"finish_order": []},
            "main_quali": {"grid": []},
            "main_race": {"finish_order": []},
        },
    )
    patcher.setattr(warmup, "save_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(
        warmup,
        "save_precomputed_prediction",
        lambda **kwargs: saved_prediction_hashes.append(str(kwargs["artifact_hash"])),
    )
    patcher.setattr(warmup, "save_precompute_horizon_index", lambda **kwargs: None)

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert result.status == "success"
    assert call_order[:2] == ["practice_refresh", "artifact_versions"]
    assert saved_prediction_hashes == ["artifact_hash_v38"]
    assert result.practice_updated is True
    assert result.practice_completed_sessions == ["FP1"]
    assert result.practice_teams_updated == 10


def test_run_warmup_precompute_cycle_skips_target_with_unknown_weekend_type(patcher):
    """Warmup should skip a target race when its weekend format cannot be resolved."""
    fixed_now = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(warmup, "_refresh_anchor_practice_characteristics", lambda **kwargs: {})
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix", "Mystery Grand Prix"),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="PRE",
            expected_checkpoint="PRE",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature=f"{race_name}-sig",
        ),
    )
    patcher.setattr(warmup, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(warmup, "compute_artifact_hash", lambda artifact_versions: "artifact_hash")
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: object())
    patcher.setattr(warmup, "load_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "load_precomputed_prediction", lambda **kwargs: None)

    def _is_sprint_weekend(year, race_name):
        if race_name == "Mystery Grand Prix":
            raise ValueError("missing schedule row")
        return False

    patcher.setattr(warmup, "is_sprint_weekend", _is_sprint_weekend)

    result = warmup.run_warmup_precompute_cycle(
        2026,
        now_utc=fixed_now,
        dry_run=True,
        verify_db_writes=False,
    )

    assert result.status == "partial_success"
    assert result.ready_races == ["Bahrain Grand Prix"]
    assert [context["race_name"] for context in result.target_contexts] == ["Bahrain Grand Prix"]
    assert any("Mystery Grand Prix [weekend_format]" in error for error in result.errors)


def test_run_warmup_precompute_cycle_returns_quickly_when_checkpoint_not_ready(patcher):
    """Warmup should skip compute work when the target checkpoint is not ready."""
    fixed_now = datetime(2026, 3, 6, 8, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="FP1",
            expected_checkpoint="FP1",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=False,
            reason="FP1_not_ready",
            boundary_signature="boundary_sig",
        ),
    )

    status_heartbeats: list[dict] = []
    patcher.setattr(
        warmup,
        "_record_not_ready_status",
        lambda year, anchor_race_name, context, now_utc: status_heartbeats.append(
            {
                "year": year,
                "anchor_race_name": anchor_race_name,
                "reason": context.reason,
            }
        ),
    )
    patcher.setattr(
        warmup,
        "get_artifact_versions",
        lambda year=2026: (_ for _ in ()).throw(AssertionError("Should not compute artifacts")),
    )

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert result.status == "not_ready"
    assert result.reason == "FP1_not_ready"
    assert result.base_generated == 0
    assert result.predictions_generated == 0
    assert status_heartbeats == [
        {
            "year": 2026,
            "anchor_race_name": "Bahrain Grand Prix",
            "reason": "FP1_not_ready",
        }
    ]


def test_run_warmup_precompute_cycle_uses_target_race_boundary_signatures(patcher):
    """Future race predictions should be keyed by each target race boundary state."""
    fixed_now = datetime(2026, 3, 6, 9, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix", "Saudi Arabian Grand Prix"),
        ),
    )

    def _checkpoint_context(year, race_name, is_sprint, now_utc, session_detector):
        del year, is_sprint, now_utc, session_detector
        if race_name == "Bahrain Grand Prix":
            return warmup.CheckpointContext(
                checkpoint="FP1",
                expected_checkpoint="FP1",
                latest_ready_checkpoint="FP1",
                checkpoint_ready=True,
                reason="ready",
                boundary_signature="sig_anchor",
            )
        return warmup.CheckpointContext(
            checkpoint="PRE",
            expected_checkpoint="PRE",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="sig_future",
        )

    patcher.setattr(warmup, "_resolve_checkpoint_context", _checkpoint_context)
    patcher.setattr(warmup, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(warmup, "compute_artifact_hash", lambda artifact_versions: "artifact_hash")
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: object())
    patcher.setattr(warmup, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(warmup, "load_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(
        warmup,
        "compute_base_features",
        lambda *args, **kwargs: {
            "is_sprint": False,
            "qualifying": {"grid": []},
            "qualifying_grid_for_race": [],
            "race_input_confidence": 0.7,
            "timing": {"qualifying": 0.1},
        },
    )
    patcher.setattr(
        warmup,
        "compute_weather_predictions",
        lambda base_features, weather, predictor, year, target_race: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )
    patcher.setattr(warmup, "save_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "save_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(warmup, "_refresh_anchor_practice_characteristics", lambda **kwargs: {})

    saved_prediction_keys: list[tuple[str, str]] = []
    patcher.setattr(
        warmup,
        "save_precomputed_prediction",
        lambda **kwargs: saved_prediction_keys.append(
            (kwargs["race_name"], kwargs["boundary_signature"])
        ),
    )

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert result.status == "success"
    assert ("Bahrain Grand Prix", "sig_anchor") in saved_prediction_keys
    assert ("Saudi Arabian Grand Prix", "sig_future") in saved_prediction_keys


def test_run_warmup_precompute_cycle_uses_checkpoint_overlay_predictor_for_target_compute(
    patcher,
):
    """Warmup should reuse one checkpoint-aware predictor across base and weather compute."""
    fixed_now = datetime(2026, 3, 6, 9, 0, tzinfo=UTC)
    base_predictor = object()
    overlay_predictor = object()
    base_predictor_calls: list[object] = []
    weather_predictor_calls: list[object] = []
    overlay_calls: list[dict[str, object]] = []

    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(warmup, "_refresh_anchor_practice_characteristics", lambda **kwargs: {})
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="FP1",
            expected_checkpoint="FP1",
            latest_ready_checkpoint="FP1",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="sig_anchor",
        ),
    )
    patcher.setattr(warmup, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(warmup, "compute_artifact_hash", lambda artifact_versions: "artifact_hash")
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: base_predictor)
    patcher.setattr(warmup, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(warmup, "load_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(
        warmup,
        "build_checkpoint_overlay_predictor",
        lambda **kwargs: overlay_calls.append(dict(kwargs)) or overlay_predictor,
    )
    patcher.setattr(
        warmup,
        "compute_base_features",
        lambda *args, **kwargs: (
            base_predictor_calls.append(kwargs["predictor"]),
            {
                "is_sprint": False,
                "qualifying": {"grid": []},
                "qualifying_grid_for_race": [],
                "race_input_confidence": 0.7,
                "timing": {"qualifying": 0.1},
            },
        )[1],
    )
    patcher.setattr(
        warmup,
        "compute_weather_predictions",
        lambda base_features, weather, predictor, year, target_race: (
            weather_predictor_calls.append(predictor),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )
    patcher.setattr(warmup, "save_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "save_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(warmup, "save_precompute_horizon_index", lambda **kwargs: None)

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert result.status == "success"
    assert overlay_calls == [
        {
            "base_predictor": base_predictor,
            "year": 2026,
            "race_name": "Bahrain Grand Prix",
            "checkpoint_session": "FP1",
            "is_sprint": False,
        }
    ]
    assert base_predictor_calls == [overlay_predictor]
    assert weather_predictor_calls == [overlay_predictor]


def test_run_warmup_precompute_cycle_dry_run_plans_without_writes(patcher):
    """Dry-run should report planned work without compute/persistence side effects."""
    fixed_now = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="PRE",
            expected_checkpoint="PRE",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="boundary_sig",
        ),
    )
    patcher.setattr(warmup, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(warmup, "compute_artifact_hash", lambda artifact_versions: "artifact_hash")
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: object())
    patcher.setattr(warmup, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        warmup,
        "_refresh_anchor_practice_characteristics",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("practice refresh should not run in dry-run")
        ),
    )
    patcher.setattr(warmup, "load_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "load_precomputed_prediction", lambda **kwargs: None)

    patcher.setattr(
        warmup,
        "compute_base_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compute_base_features should not run in dry-run")
        ),
    )
    patcher.setattr(
        warmup,
        "compute_weather_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compute_weather_predictions should not run in dry-run")
        ),
    )
    patcher.setattr(
        warmup,
        "save_precomputed_base_features",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("save_precomputed_base_features should not run in dry-run")
        ),
    )
    patcher.setattr(
        warmup,
        "save_precomputed_prediction",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("save_precomputed_prediction should not run in dry-run")
        ),
    )
    patcher.setattr(
        warmup,
        "save_precompute_horizon_index",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("save_precompute_horizon_index should not run in dry-run")
        ),
    )

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now, dry_run=True)

    assert result.status == "dry_run"
    assert result.dry_run is True
    assert result.base_generated == 1
    assert result.predictions_generated == 3
    assert result.ready_races == ["Bahrain Grand Prix"]


def test_run_warmup_precompute_cycle_reports_db_verification_warning_on_missing_readback(patcher):
    """Warmup should report DB verification warning when read-back cannot find saved record."""
    fixed_now = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: False)
    patcher.setattr(warmup, "_refresh_anchor_practice_characteristics", lambda **kwargs: {})
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="PRE",
            expected_checkpoint="PRE",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="boundary_sig",
        ),
    )
    patcher.setattr(warmup, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(warmup, "compute_artifact_hash", lambda artifact_versions: "artifact_hash")
    patcher.setattr(warmup, "_load_predictor", lambda artifact_versions, year: object())
    patcher.setattr(warmup, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(warmup, "load_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(warmup, "_can_verify_db_writes", lambda: True)
    patcher.setattr(warmup, "_verify_runtime_state_record", lambda namespace, state_key: False)
    patcher.setattr(
        warmup,
        "compute_base_features",
        lambda *args, **kwargs: {
            "is_sprint": False,
            "qualifying": {"grid": []},
            "qualifying_grid_for_race": [],
            "race_input_confidence": 0.7,
            "timing": {"qualifying": 0.1},
        },
    )
    patcher.setattr(
        warmup,
        "compute_weather_predictions",
        lambda base_features, weather, predictor, year, target_race: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )
    patcher.setattr(warmup, "save_precomputed_base_features", lambda **kwargs: None)
    patcher.setattr(warmup, "save_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(warmup, "save_precompute_horizon_index", lambda **kwargs: None)

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert result.status == "success"
    assert result.db_verification_warnings


def test_run_warmup_precompute_cycle_returns_locked_when_another_worker_holds_lock(patcher):
    """Warmup should skip compute work when a DB-backed lock is held by another worker."""
    fixed_now = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)
    patcher.setattr(warmup, "should_write_to_db", lambda: True)
    patcher.setattr(
        warmup,
        "RuntimeStateStore",
        lambda: type(
            "_LockingStore",
            (),
            {
                "acquire_lock": staticmethod(
                    lambda lock_key, owner_id, ttl_seconds=900: False  # noqa: ARG005
                ),
                "release_lock": staticmethod(
                    lambda lock_key, owner_id: (_ for _ in ()).throw(
                        AssertionError("release_lock should not run when lock is not acquired")
                    )
                ),
            },
        )(),
    )
    patcher.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name="Bahrain Grand Prix",
            anchor_is_sprint=False,
            target_races=("Bahrain Grand Prix",),
        ),
    )
    patcher.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="PRE",
            expected_checkpoint="PRE",
            latest_ready_checkpoint="PRE",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature="boundary_sig",
        ),
    )
    patcher.setattr(
        warmup,
        "get_artifact_versions",
        lambda year=2026: (_ for _ in ()).throw(
            AssertionError("get_artifact_versions should not run when lock is held")
        ),
    )

    result = warmup.run_warmup_precompute_cycle(2026, now_utc=fixed_now)

    assert result.status == "locked"
    assert result.reason == "another_worker_holds_lock"


def test_compute_base_features_uses_actual_qualifying_section_after_completed_q(patcher):
    """Warmup base features should stop predicting qualifying once Q is complete."""

    class _Predictor:
        def predict_qualifying(self, **kwargs):
            raise AssertionError("predict_qualifying should not run for completed qualifying")

    patcher.setattr(
        warmup,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (
            ([{"position": 1, "driver": "RUS", "team": "Mercedes"}], "ACTUAL")
            if session_name == "Q"
            else (None, "INCOMPLETE")
        ),
    )

    result = warmup.compute_base_features(
        2026,
        "Australian Grand Prix",
        "Q",
        "artifact_hash",
        "boundary_signature",
        predictor=_Predictor(),
        is_sprint=False,
    )

    assert result["qualifying"]["result_mode"] == "ACTUAL"
    assert result["qualifying"]["grid"][0]["driver"] == "RUS"
    assert result["race_input_confidence"] == 1.0


def test_compute_base_features_uses_stored_checkpoint_profiles_for_qualifying(patcher):
    """Warmup should pin qualifying inference to the stored checkpoint profile state."""

    class _Predictor:
        def __init__(self):
            self.qualifying_kwargs = None

        def predict_qualifying(self, **kwargs):
            self.qualifying_kwargs = kwargs
            return {
                "grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
                "data_confidence_score": 0.9,
                "data_source": "FP3 short-stint",
            }

    predictor = _Predictor()

    patcher.setattr(
        warmup,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )
    patcher.setattr(
        warmup,
        "fetch_grid_if_available",
        lambda year, race_name, session_name, predicted_grid: (predicted_grid, "PREDICTED"),
    )

    warmup.compute_base_features(
        2026,
        "Australian Grand Prix",
        "FP3",
        "artifact_hash",
        "boundary_signature",
        predictor=predictor,
        is_sprint=False,
    )

    assert predictor.qualifying_kwargs is not None
    assert predictor.qualifying_kwargs["practice_signal_mode"] == "stored_profiles"
    assert predictor.qualifying_kwargs["checkpoint_session_name"] == "FP3"


def test_compute_weather_predictions_uses_actual_sprint_race_after_completion(patcher):
    """Warmup weather overlays should not regenerate completed sprint races."""

    class _Predictor:
        def predict_sprint_race(self, **kwargs):
            raise AssertionError("predict_sprint_race should not run for completed sprint race")

        def predict_race(self, **kwargs):
            return {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

    patcher.setattr(
        warmup,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (
            ([{"position": 1, "driver": "RUS", "team": "Mercedes"}], "ACTUAL")
            if session_name == "Sprint"
            else (None, "INCOMPLETE")
        ),
    )

    result = warmup.compute_weather_predictions(
        {
            "is_sprint": True,
            "sprint_quali": {"grid_source": "ACTUAL"},
            "sprint_grid_for_race": [{"position": 1, "driver": "RUS", "team": "Mercedes"}],
            "sprint_race_input_confidence": 1.0,
            "main_quali": {"grid_source": "PREDICTED"},
            "main_grid_for_race": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "main_race_input_confidence": 0.7,
            "timing": {"sprint_quali": 0.1, "main_quali": 0.1},
        },
        "dry",
        predictor=_Predictor(),
        year=2026,
        target_race="Chinese Grand Prix",
    )

    assert result["sprint_race"]["result_mode"] == "ACTUAL"
    assert "no penalties applied" in result["sprint_race"]["starting_grid_note"].lower()


def test_compute_weather_predictions_passes_sprint_race_input_confidence(patcher):
    """Warmup sprint simulations should receive the stored sprint input confidence."""

    class _Predictor:
        def __init__(self):
            self.sprint_kwargs = None

        def predict_sprint_race(self, **kwargs):
            self.sprint_kwargs = kwargs
            return {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

        def predict_race(self, **kwargs):
            return {"finish_order": [{"position": 1, "driver": "NOR", "team": "McLaren"}]}

    predictor = _Predictor()

    patcher.setattr(
        warmup,
        "fetch_actual_competitive_results_if_completed",
        lambda year, race_name, session_name: (None, "INCOMPLETE"),
    )

    warmup.compute_weather_predictions(
        {
            "is_sprint": True,
            "sprint_quali": {"grid_source": "PREDICTED"},
            "sprint_grid_for_race": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "sprint_race_input_confidence": 0.58,
            "main_quali": {"grid_source": "PREDICTED"},
            "main_grid_for_race": [{"position": 1, "driver": "NOR", "team": "McLaren"}],
            "main_race_input_confidence": 0.72,
            "timing": {"sprint_quali": 0.1, "main_quali": 0.1},
        },
        "dry",
        predictor=predictor,
        year=2026,
        target_race="Chinese Grand Prix",
    )

    assert predictor.sprint_kwargs is not None
    assert predictor.sprint_kwargs["input_confidence"] == pytest.approx(0.58)
