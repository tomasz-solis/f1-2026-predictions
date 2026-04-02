"""Tests for persisted dashboard prediction loading."""

import pytest

from src.dashboard import live_prediction_flow


def _default_boundary_refresh(*, year, race_name, is_sprint, session_detector=None):
    """Return a stable checkpoint payload for default pipeline tests."""
    del year, race_name, is_sprint, session_detector
    return {
        "refresh_needed": False,
        "reason": "no_change",
        "new_sessions": [],
        "boundary_signature": "stable_sig",
        "latest_elapsed_session": "FP2",
    }


def _stub_single_target(patcher):
    """Keep pipeline tests independent from the real schedule when target scope is irrelevant."""
    patcher.setattr(
        live_prediction_flow,
        "_resolve_precompute_targets",
        lambda year, race_name, horizon_races: [race_name],
    )


@pytest.fixture(autouse=True)
def _reset_prediction_cache(patcher):
    """Keep dashboard prediction-cache tests isolated from each other."""
    live_prediction_flow.clear_prediction_result_cache()
    patcher.setattr(
        live_prediction_flow,
        "_get_prediction_precompute_settings",
        lambda: {
            "enabled": True,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(live_prediction_flow, "compute_artifact_hash", lambda versions: "artifact_hash")
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "load_precompute_horizon_index", lambda **kwargs: None)
    yield
    live_prediction_flow.clear_prediction_result_cache()


def _base_core_kwargs(**overrides):
    """Build default request-path callbacks for persisted-only pipeline tests."""
    kwargs = {
        "race_name": "Australian Grand Prix",
        "weather": "dry",
        "year": 2026,
        "force_refresh": False,
        "progress_callback": None,
        "clear_fastf1_race_cache_fn": lambda year, race_name: None,
        "auto_update_if_needed_fn": lambda force_recheck=False, year=2026: None,
        "is_sprint_weekend_fn": lambda year, race_name: False,
        "detect_event_boundary_refresh_if_needed_fn": _default_boundary_refresh,
        "auto_update_practice_characteristics_if_needed_fn": (
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        "clear_resource_cache_fn": lambda: None,
        "clear_data_cache_fn": lambda: None,
        "get_artifact_versions_fn": lambda year=2026: {"k": (1, "ts")},
        "run_prediction_fn": lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run_prediction should not execute in persisted-only dashboard mode")
        ),
    }
    kwargs.update(overrides)
    return kwargs


def test_execute_live_prediction_pipeline_loads_persisted_prediction_and_emits_read_only_progress(
    patcher,
):
    """A normal click should load warmed artifacts without mutating state."""
    _stub_single_target(patcher)
    progress_messages: list[str] = []
    persisted_prediction = {
        "qualifying": {"grid": []},
        "race": {"finish_order": []},
    }

    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: persisted_prediction,
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        **_base_core_kwargs(progress_callback=progress_messages.append)
    )

    assert progress_messages == [
        "Loading persisted prediction artifacts...",
        "Resolving weekend format...",
        "Warmup owns practice refresh; dashboard request path stays read-only...",
        "Loaded persisted prediction...",
    ]
    assert output["prediction_results"] == persisted_prediction
    assert output["prediction_cache_hit"] is False
    assert output["practice_update"] == {"updated": False, "completed_fp_sessions": []}
    assert output["precompute_summary"]["skipped_reason"] == "request_path_read_only"

    timing = output["pipeline_timing"]
    assert set(timing) == {
        "cache_clear",
        "weekend_lookup",
        "boundary_check",
        "practice_update_check",
        "prediction_load",
        "total",
    }
    assert timing["total"] >= 0.0


def test_execute_live_prediction_pipeline_rejects_force_refresh(patcher):
    """Manual refresh should fail closed so the UI stays read-only."""
    _stub_single_target(patcher)
    with pytest.raises(
        live_prediction_flow.PrecomputedPredictionUnavailableError,
        match="Manual dashboard refresh is disabled",
    ):
        live_prediction_flow.execute_live_prediction_pipeline_core(
            **_base_core_kwargs(force_refresh=True)
        )


def test_execute_live_prediction_pipeline_raises_when_persisted_prediction_is_missing(patcher):
    """The dashboard should tell operators to warm artifacts instead of simulating inline."""
    _stub_single_target(patcher)
    with pytest.raises(
        live_prediction_flow.PrecomputedPredictionUnavailableError,
        match="Run warmup or trigger the scheduled job",
    ):
        live_prediction_flow.execute_live_prediction_pipeline_core(**_base_core_kwargs())


def test_execute_live_prediction_pipeline_fails_closed_when_weekend_lookup_breaks(patcher):
    """Request-path weekend lookup failures should surface as dashboard-specific errors."""
    _stub_single_target(patcher)

    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("persisted lookup should not run when weekend lookup fails")
        ),
    )

    with pytest.raises(
        live_prediction_flow.PrecomputedPredictionUnavailableError,
        match="Could not resolve weekend format for Australian Grand Prix 2026",
    ):
        live_prediction_flow.execute_live_prediction_pipeline_core(
            **_base_core_kwargs(
                is_sprint_weekend_fn=lambda year, race_name: (_ for _ in ()).throw(
                    ValueError("missing schedule row")
                )
            )
        )


def test_execute_live_prediction_pipeline_uses_warmed_boundary_fallback_when_current_boundary_ahead(
    patcher,
):
    """When live boundary is ahead, serve the last warmed checkpoint instead of regenerating."""
    _stub_single_target(patcher)
    progress_messages: list[str] = []
    load_calls: list[str] = []

    def _load_precomputed_prediction(**kwargs):
        boundary_signature = str(kwargs.get("boundary_signature", ""))
        load_calls.append(boundary_signature)
        if boundary_signature == "sig_fp1":
            return {"sprint_quali": {"grid": []}, "sprint_race": {"finish_order": []}}
        return None

    patcher.setattr(
        live_prediction_flow, "load_precomputed_prediction", _load_precomputed_prediction
    )
    patcher.setattr(
        live_prediction_flow,
        "load_precompute_horizon_index",
        lambda **kwargs: {
            "ready_races": ["Chinese Grand Prix"],
            "expected_targets": ["Chinese Grand Prix"],
            "anchor_race_name": "Chinese Grand Prix",
            "anchor_session_name": "FP1",
            "boundary_signature": "sig_fp1",
            "race_boundaries": {"Chinese Grand Prix": "sig_fp1"},
        },
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        **_base_core_kwargs(
            race_name="Chinese Grand Prix",
            progress_callback=progress_messages.append,
            is_sprint_weekend_fn=lambda year, race_name: True,
            detect_event_boundary_refresh_if_needed_fn=(
                lambda year, race_name, is_sprint, session_detector=None: {
                    "refresh_needed": True,
                    "reason": "session_boundary_delta",
                    "new_sessions": ["SQ"],
                    "boundary_signature": "sig_sq",
                    "latest_elapsed_session": "SQ",
                }
            ),
        )
    )

    assert load_calls == ["sig_sq", "sig_fp1"]
    assert output["prediction_cache_hit"] is False
    assert output["boundary_session_name"] == "FP1"
    assert output["boundary_fallback"] == {
        "current_boundary_signature": "sig_sq",
        "current_boundary_session_name": "SQ",
        "served_boundary_signature": "sig_fp1",
        "served_boundary_session_name": "FP1",
        "mode": "served_warmed_boundary",
        "warmed_boundary_signature": "sig_fp1",
        "warmed_boundary_session_name": "FP1",
    }
    assert "A newer checkpoint exists; waiting for warmup to persist it..." in progress_messages
    assert (
        "Current checkpoint is ahead of the warmed horizon; serving the latest persisted checkpoint until warmup catches up..."
        in progress_messages
    )


def test_execute_live_prediction_pipeline_raises_when_boundary_ahead_but_no_warmed_fallback_exists(
    patcher,
):
    """A newer checkpoint without a warmed fallback should stay fail-closed and report the gap."""
    _stub_single_target(patcher)
    load_calls: list[str] = []

    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: (load_calls.append(str(kwargs.get("boundary_signature", ""))), None)[1],
    )
    patcher.setattr(
        live_prediction_flow,
        "load_precompute_horizon_index",
        lambda **kwargs: {
            "ready_races": ["Chinese Grand Prix"],
            "expected_targets": ["Chinese Grand Prix"],
            "anchor_race_name": "Chinese Grand Prix",
            "anchor_session_name": "FP1",
            "boundary_signature": "sig_fp1",
            "race_boundaries": {"Chinese Grand Prix": "sig_fp1"},
        },
    )

    with pytest.raises(
        live_prediction_flow.PrecomputedPredictionUnavailableError,
        match=r"Chinese Grand Prix 2026 \[dry\] at checkpoint SQ",
    ):
        live_prediction_flow.execute_live_prediction_pipeline_core(
            **_base_core_kwargs(
                race_name="Chinese Grand Prix",
                is_sprint_weekend_fn=lambda year, race_name: True,
                detect_event_boundary_refresh_if_needed_fn=(
                    lambda year, race_name, is_sprint, session_detector=None: {
                        "refresh_needed": True,
                        "reason": "session_boundary_delta",
                        "new_sessions": ["SQ"],
                        "boundary_signature": "sig_sq",
                        "latest_elapsed_session": "SQ",
                    }
                ),
            )
        )

    assert load_calls == ["sig_sq", "sig_fp1"]


def test_execute_live_prediction_pipeline_resolves_boundary_when_refresh_payload_omits_it(
    patcher,
):
    """Persisted lookup should fall back to resolved boundary metadata when needed."""
    _stub_single_target(patcher)
    load_calls: list[str] = []

    def _load_precomputed_prediction(**kwargs):
        boundary_signature = str(kwargs.get("boundary_signature", ""))
        load_calls.append(boundary_signature)
        if boundary_signature == "sig_resolved":
            return {"qualifying": {"grid": []}, "race": {"finish_order": []}}
        return None

    patcher.setattr(
        live_prediction_flow, "load_precomputed_prediction", _load_precomputed_prediction
    )
    patcher.setattr(
        live_prediction_flow,
        "_resolve_race_boundary_context",
        lambda year, race_name, is_sprint, session_detector=None: ("sig_resolved", "FP1"),
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        **_base_core_kwargs(
            detect_event_boundary_refresh_if_needed_fn=(
                lambda year, race_name, is_sprint, session_detector=None: {
                    "refresh_needed": False,
                    "reason": "no_change",
                    "new_sessions": [],
                    "boundary_signature": "",
                    "latest_elapsed_session": None,
                }
            )
        )
    )

    assert load_calls == ["sig_resolved"]
    assert output["prediction_cache_hit"] is False
    assert output["boundary_session_name"] == "FP1"


def test_execute_live_prediction_pipeline_invalidates_memory_cache_when_boundary_changes(
    patcher,
):
    """Different checkpoint signatures should produce different memory-cache keys."""
    _stub_single_target(patcher)
    load_calls: list[str] = []
    signatures = ["sig_a", "sig_a", "sig_b"]

    def _load_precomputed_prediction(**kwargs):
        boundary_signature = str(kwargs.get("boundary_signature", ""))
        load_calls.append(boundary_signature)
        return {
            "qualifying": {"grid": [], "boundary_signature": boundary_signature},
            "race": {"finish_order": []},
            "_prediction_context": {
                "persisted_updated_at": f"2026-03-18T10:00:0{0 if boundary_signature == 'sig_a' else 1}+00:00"
            },
        }

    patcher.setattr(
        live_prediction_flow, "load_precomputed_prediction", _load_precomputed_prediction
    )

    common_kwargs = _base_core_kwargs(
        detect_event_boundary_refresh_if_needed_fn=(
            lambda year, race_name, is_sprint, session_detector=None: {
                "refresh_needed": False,
                "reason": "no_change",
                "new_sessions": [],
                "boundary_signature": signatures.pop(0),
                "latest_elapsed_session": "FP2",
            }
        )
    )

    first = live_prediction_flow.execute_live_prediction_pipeline_core(**common_kwargs)
    second = live_prediction_flow.execute_live_prediction_pipeline_core(**common_kwargs)
    third = live_prediction_flow.execute_live_prediction_pipeline_core(**common_kwargs)

    assert first["prediction_results"]["qualifying"]["boundary_signature"] == "sig_a"
    assert second["prediction_results"]["qualifying"]["boundary_signature"] == "sig_a"
    assert third["prediction_results"]["qualifying"]["boundary_signature"] == "sig_b"
    assert first["prediction_cache_hit"] is False
    assert second["prediction_cache_hit"] is True
    assert third["prediction_cache_hit"] is False
    assert load_calls == ["sig_a", "sig_a", "sig_b"]


def test_execute_live_prediction_pipeline_refreshes_same_boundary_when_persisted_payload_changes(
    patcher,
):
    """Same-boundary rewrites should replace stale RAM entries instead of hiding them."""
    _stub_single_target(patcher)
    load_calls: list[str] = []
    payloads = iter(
        [
            {
                "qualifying": {"grid_source": "PREDICTED", "grid": []},
                "race": {"finish_order": []},
                "_prediction_context": {"persisted_updated_at": "2026-03-18T10:00:00+00:00"},
            },
            {
                "qualifying": {"grid_source": "ACTUAL", "grid": []},
                "race": {"finish_order": []},
                "_prediction_context": {"persisted_updated_at": "2026-03-18T10:05:00+00:00"},
            },
        ]
    )

    def _load_precomputed_prediction(**kwargs):
        load_calls.append(str(kwargs.get("boundary_signature", "")))
        return next(payloads)

    patcher.setattr(
        live_prediction_flow, "load_precomputed_prediction", _load_precomputed_prediction
    )

    first = live_prediction_flow.execute_live_prediction_pipeline_core(**_base_core_kwargs())
    second = live_prediction_flow.execute_live_prediction_pipeline_core(**_base_core_kwargs())

    assert first["prediction_results"]["qualifying"]["grid_source"] == "PREDICTED"
    assert second["prediction_results"]["qualifying"]["grid_source"] == "ACTUAL"
    assert first["prediction_cache_hit"] is False
    assert second["prediction_cache_hit"] is False
    assert load_calls == ["stable_sig", "stable_sig"]


def test_execute_live_prediction_pipeline_ignores_stale_horizon_summary_metadata(patcher):
    """Ready-race summaries should be ignored when saved horizon metadata is stale."""
    _stub_single_target(patcher)
    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: {"qualifying": {"grid": []}, "race": {"finish_order": []}},
    )
    patcher.setattr(
        live_prediction_flow,
        "load_precompute_horizon_index",
        lambda **kwargs: {
            "boundary_signature": "stale_sig",
            "anchor_race_name": "Australian Grand Prix",
            "expected_targets": ["Australian Grand Prix"],
            "ready_races": ["Australian Grand Prix"],
        },
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(**_base_core_kwargs())

    assert output["precompute_summary"]["ready_races"] == []


def test_execute_live_prediction_pipeline_skips_request_path_mutations(patcher):
    """Request-path callbacks are still wired in, but persisted mode should not call them."""
    _stub_single_target(patcher)
    persisted_prediction = {
        "qualifying": {"grid": []},
        "race": {"finish_order": []},
    }
    mutation_calls = {
        "race_update": 0,
        "practice_update": 0,
        "clear_fastf1": 0,
        "clear_resource": 0,
        "clear_data": 0,
        "run_prediction": 0,
    }

    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: persisted_prediction,
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        **_base_core_kwargs(
            clear_fastf1_race_cache_fn=lambda year, race_name: mutation_calls.__setitem__(
                "clear_fastf1", mutation_calls["clear_fastf1"] + 1
            ),
            auto_update_if_needed_fn=lambda force_recheck=False,
            year=2026: mutation_calls.__setitem__("race_update", mutation_calls["race_update"] + 1),
            auto_update_practice_characteristics_if_needed_fn=(
                lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
                    mutation_calls.__setitem__(
                        "practice_update", mutation_calls["practice_update"] + 1
                    ),
                    {"updated": True, "completed_fp_sessions": ["FP1"]},
                )[1]
            ),
            clear_resource_cache_fn=lambda: mutation_calls.__setitem__(
                "clear_resource", mutation_calls["clear_resource"] + 1
            ),
            clear_data_cache_fn=lambda: mutation_calls.__setitem__(
                "clear_data", mutation_calls["clear_data"] + 1
            ),
            run_prediction_fn=lambda *args, **kwargs: (
                mutation_calls.__setitem__("run_prediction", mutation_calls["run_prediction"] + 1),
                {"qualifying": {"grid": []}, "race": {"finish_order": []}},
            )[1],
        )
    )

    assert mutation_calls == {
        "race_update": 0,
        "practice_update": 0,
        "clear_fastf1": 0,
        "clear_resource": 0,
        "clear_data": 0,
        "run_prediction": 0,
    }
    assert output["prediction_results"] == persisted_prediction
    assert output["practice_update"] == {"updated": False, "completed_fp_sessions": []}


def test_resolve_precompute_targets_skips_testing_by_event_format(patcher):
    """Precompute targeting should ignore testing rows when building the warmed horizon."""
    patcher.setattr(
        "src.utils.weekend.get_schedule_rows",
        lambda year: (
            ("Pre-Season Track Day", "testing"),
            ("Australian Grand Prix", "conventional"),
            ("Chinese Grand Prix", "sprint"),
            ("In-Season Testing", "conventional"),
            ("Japanese Grand Prix", "conventional"),
        ),
    )

    targets = live_prediction_flow._resolve_precompute_targets(
        year=2026,
        race_name="Australian Grand Prix",
        horizon_races=3,
    )

    assert targets == ["Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix"]
