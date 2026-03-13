"""Tests for live prediction pipeline orchestration."""

import pytest

from src.dashboard import live_prediction_flow, pages


@pytest.fixture(autouse=True)
def _default_boundary_refresh_stub(patcher):
    live_prediction_flow.clear_prediction_result_cache()
    patcher.setattr(
        live_prediction_flow,
        "_get_prediction_precompute_settings",
        lambda: {
            "enabled": False,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "save_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "load_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "save_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "",
        },
    )


def test_execute_live_prediction_pipeline_refresh_call_order(patcher):
    call_order: list[str] = []

    patcher.setattr(
        pages,
        "auto_update_if_needed",
        lambda force_recheck=False, year=2026: call_order.append("race_update"),
    )
    patcher.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (call_order.append("sprint_check"), True)[1],
    )

    def _practice_update(
        year: int,
        race_name: str,
        is_sprint: bool,
        force_recheck: bool = False,
        session_detector=None,
    ):
        _ = session_detector
        call_order.append("practice_update")
        assert year == 2026
        assert race_name == "Chinese Grand Prix"
        assert is_sprint is True
        return {"updated": False, "completed_fp_sessions": []}

    patcher.setattr(pages, "auto_update_practice_characteristics_if_needed", _practice_update)
    patcher.setattr(
        pages, "_clear_fastf1_race_cache", lambda year, race_name: call_order.append("cache_clear")
    )
    patcher.setattr(
        pages,
        "get_artifact_versions",
        lambda year=2026: (call_order.append("artifact_versions"), {"k": (1, "ts")})[1],
    )

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        call_order.append("run_prediction")
        assert race_name == "Chinese Grand Prix"
        assert weather == "dry"
        assert versions == {"k": (1, "ts")}
        assert is_sprint is True
        assert year == 2026
        return {"sprint_quali": {"grid": []}}

    patcher.setattr(pages, "run_prediction", _run_prediction)

    output = pages.execute_live_prediction_pipeline(
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,  # Don't clear cache for this test
    )

    assert call_order == [
        "race_update",
        "sprint_check",
        "practice_update",
        "artifact_versions",
        "run_prediction",
    ]
    assert output["is_sprint"] is True


def test_execute_live_prediction_pipeline_clears_cache_before_prediction_when_practice_updated(
    patcher,
):
    call_order: list[str] = []

    patcher.setattr(
        pages,
        "auto_update_if_needed",
        lambda force_recheck=False, year=2026: call_order.append("race_update"),
    )
    patcher.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (call_order.append("sprint_check"), False)[1],
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
            call_order.append("practice_update"),
            {"updated": True, "completed_fp_sessions": ["FP1"], "teams_updated": 2},
        )[1],
    )
    patcher.setattr(
        pages, "_clear_fastf1_race_cache", lambda year, race_name: call_order.append("cache_clear")
    )
    patcher.setattr(
        pages,
        "get_artifact_versions",
        lambda year=2026: (call_order.append("artifact_versions"), {"k": (4, "ts4")})[1],
    )

    patcher.setattr(
        pages.st,
        "cache_resource",
        type(
            "_CacheResource",
            (),
            {"clear": staticmethod(lambda: call_order.append("clear_resource"))},
        ),
    )
    patcher.setattr(
        pages.st,
        "cache_data",
        type("_CacheData", (), {"clear": staticmethod(lambda: call_order.append("clear_data"))}),
    )

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        call_order.append("run_prediction")
        assert versions == {"k": (4, "ts4")}
        assert is_sprint is False
        return {"qualifying": {"grid": []}, "race": {"finish_order": []}}

    patcher.setattr(pages, "run_prediction", _run_prediction)

    pages.execute_live_prediction_pipeline(
        "Bahrain Grand Prix", "dry", year=2026, force_refresh=False
    )

    assert call_order == [
        "race_update",
        "sprint_check",
        "practice_update",
        "clear_resource",
        "clear_data",
        "artifact_versions",
        "run_prediction",
    ]


def test_execute_live_prediction_pipeline_raises_when_practice_update_fails(patcher):
    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
            _ for _ in ()
        ).throw(RuntimeError("refresh failed")),
    )
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)

    run_called = {"value": False}

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        run_called["value"] = True
        raise AssertionError("run_prediction should not be called when refresh fails")

    patcher.setattr(pages, "run_prediction", _run_prediction)

    with pytest.raises(RuntimeError, match="refresh failed"):
        pages.execute_live_prediction_pipeline(
            "Bahrain Grand Prix", "dry", year=2026, force_refresh=False
        )

    assert run_called["value"] is False


def test_execute_live_prediction_pipeline_raises_when_sprint_lookup_fails(patcher):
    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (_ for _ in ()).throw(ValueError("bad race")),
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (3, "ts3")})

    patcher.setattr(
        pages,
        "run_prediction",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_prediction should not execute")),
    )

    with pytest.raises(ValueError, match="bad race"):
        pages.execute_live_prediction_pipeline("Unknown GP", "dry", year=2026, force_refresh=False)


def test_execute_live_prediction_pipeline_emits_progress_and_timing(patcher):
    progress_messages: list[str] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )

    output = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=progress_messages.append,
    )

    assert progress_messages == [
        "Checking completed races and model updates...",
        "Resolving weekend format...",
        "Checking completed practice sessions...",
        "Running qualifying and race simulations...",
    ]

    timing = output["pipeline_timing"]
    assert set(timing) == {
        "race_update_check",
        "weekend_lookup",
        "practice_update_check",
        "prediction_run",
        "total",
    }
    assert timing["total"] >= 0.0


def test_execute_live_prediction_pipeline_passes_year_to_auto_update_when_supported(patcher):
    seen_calls: list[tuple[int, bool]] = []

    def _auto_update(year: int, force_recheck: bool = False):
        seen_calls.append((year, force_recheck))

    patcher.setattr(pages, "auto_update_if_needed", _auto_update)
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2027,
        force_refresh=True,
    )

    assert seen_calls == [(2027, True)]


def test_execute_live_prediction_pipeline_passes_year_to_artifact_versions_when_supported(patcher):
    seen_years: list[int] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )

    def _artifact_versions(year: int = 2026):
        seen_years.append(year)
        return {"k": (1, "ts")}

    patcher.setattr(pages, "get_artifact_versions", _artifact_versions)
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2027,
        force_refresh=False,
    )

    assert seen_years == [2027]


def test_execute_live_prediction_pipeline_with_force_refresh_clears_cache_and_rechecks(patcher):
    """Test that force_refresh=True clears FastF1 cache and forces session recheck."""
    call_order: list[str] = []
    force_recheck_calls = {"race_update": False, "practice_update": False}

    def mock_race_update(force_recheck=False, year=2026):
        call_order.append("race_update")
        force_recheck_calls["race_update"] = force_recheck

    def mock_practice_update(
        year,
        race_name,
        is_sprint,
        force_recheck=False,
        session_detector=None,
    ):
        _ = session_detector
        call_order.append("practice_update")
        force_recheck_calls["practice_update"] = force_recheck
        return {"updated": False, "completed_fp_sessions": []}

    patcher.setattr(pages, "auto_update_if_needed", mock_race_update)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(pages, "auto_update_practice_characteristics_if_needed", mock_practice_update)
    patcher.setattr(
        pages, "_clear_fastf1_race_cache", lambda year, race_name: call_order.append("cache_clear")
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )
    patcher.setattr(
        pages.st,
        "cache_resource",
        type(
            "_CacheResource",
            (),
            {"clear": staticmethod(lambda: call_order.append("clear_resource"))},
        ),
    )
    patcher.setattr(
        pages.st,
        "cache_data",
        type("_CacheData", (), {"clear": staticmethod(lambda: call_order.append("clear_data"))}),
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=True,
    )

    # Verify cache was cleared first
    assert call_order[0] == "cache_clear"
    # Verify force_recheck was passed to update functions
    assert force_recheck_calls["race_update"] is True
    assert force_recheck_calls["practice_update"] is True
    # Verify caches were cleared (since force_refresh=True)
    assert "clear_resource" in call_order
    assert "clear_data" in call_order


def test_execute_live_prediction_pipeline_auto_refreshes_on_event_boundary_delta(patcher):
    call_order: list[str] = []
    force_recheck_calls: list[bool] = []

    def _race_update(force_recheck=False, year=2026):
        call_order.append("race_update")
        force_recheck_calls.append(bool(force_recheck))

    patcher.setattr(pages, "auto_update_if_needed", _race_update)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: (
            call_order.append("boundary_check"),
            {
                "refresh_needed": True,
                "reason": "session_boundary_delta",
                "new_sessions": ["FP2"],
            },
        )[1],
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
            call_order.append(f"practice_update:{force_recheck}"),
            {"updated": False, "completed_fp_sessions": ["FP1", "FP2"]},
        )[1],
    )
    patcher.setattr(
        pages, "_clear_fastf1_race_cache", lambda year, race_name: call_order.append("cache_clear")
    )
    patcher.setattr(
        pages.st,
        "cache_resource",
        type(
            "_CacheResource",
            (),
            {"clear": staticmethod(lambda: call_order.append("clear_resource"))},
        ),
    )
    patcher.setattr(
        pages.st,
        "cache_data",
        type("_CacheData", (), {"clear": staticmethod(lambda: call_order.append("clear_data"))}),
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            call_order.append("run_prediction"),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    output = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert force_recheck_calls == [False, False]
    assert output["boundary_refresh"]["refresh_needed"] is True
    assert output["boundary_refresh"]["new_sessions"] == ["FP2"]
    assert call_order == [
        "race_update",
        "boundary_check",
        "cache_clear",
        "race_update",
        "practice_update:False",
        "clear_resource",
        "clear_data",
        "run_prediction",
    ]


def test_execute_live_prediction_pipeline_auto_refreshes_on_sprint_boundary_delta(patcher):
    call_order: list[str] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: True)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: (
            call_order.append("boundary_check"),
            {
                "refresh_needed": True,
                "reason": "session_data_changed",
                "new_sessions": ["SQ"],
                "boundary_signature": "sq_ready",
            },
        )[1],
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
            call_order.append("practice_update"),
            {"updated": False, "completed_fp_sessions": ["FP1"]},
        )[1],
    )
    patcher.setattr(
        pages, "_clear_fastf1_race_cache", lambda year, race_name: call_order.append("cache_clear")
    )
    patcher.setattr(
        pages.st,
        "cache_resource",
        type("_CacheResource", (), {"clear": staticmethod(lambda: call_order.append("clear_r"))}),
    )
    patcher.setattr(
        pages.st,
        "cache_data",
        type("_CacheData", (), {"clear": staticmethod(lambda: call_order.append("clear_d"))}),
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            call_order.append(f"run_prediction:{is_sprint}"),
            {"sprint_quali": {"grid": []}, "sprint_race": {"finish_order": []}},
        )[1],
    )

    output = pages.execute_live_prediction_pipeline(
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert output["is_sprint"] is True
    assert output["boundary_refresh"]["new_sessions"] == ["SQ"]
    assert call_order == [
        "boundary_check",
        "cache_clear",
        "practice_update",
        "clear_r",
        "clear_d",
        "run_prediction:True",
    ]


def test_execute_live_prediction_pipeline_invalidates_prediction_cache_on_boundary_signature_change(
    patcher,
):
    run_calls = {"direct": 0}
    signatures = ["sig_a", "sig_a", "sig_b"]

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": signatures.pop(0),
        },
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            run_calls.__setitem__("direct", run_calls["direct"] + 1),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    first = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )
    second = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )
    third = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert first["prediction_cache_hit"] is False
    assert second["prediction_cache_hit"] is True
    assert third["prediction_cache_hit"] is False
    assert run_calls == {"direct": 2}


def test_execute_live_prediction_pipeline_cache_hit_recomputes_when_competitive_results_change(
    patcher,
):
    run_calls = {"direct": 0}
    fastf1_refresh_calls = {"completion": 0, "results": 0}

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "stable_sig",
        },
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            run_calls.__setitem__("direct", run_calls["direct"] + 1),
            {
                "qualifying": {
                    "grid_source": "PREDICTED",
                    "grid": [
                        {"position": 1, "driver": "VER", "team": "Red Bull"},
                        {"position": 2, "driver": "LEC", "team": "Ferrari"},
                    ],
                },
                "race": {"finish_order": []},
            },
        )[1],
    )

    patcher.setattr(
        "src.utils.actual_results_fetcher.get_competitive_session_completion_state",
        lambda year, race_name, session_name: (
            fastf1_refresh_calls.__setitem__("completion", fastf1_refresh_calls["completion"] + 1),
            "completed",
        )[1],
    )
    patcher.setattr(
        "src.utils.actual_results_fetcher.fetch_actual_session_results",
        lambda year, race_name, session_name: (
            fastf1_refresh_calls.__setitem__("results", fastf1_refresh_calls["results"] + 1),
            [
                {"position": 1, "driver": "NOR", "team": "McLaren"},
                {"position": 2, "driver": "VER", "team": "Red Bull"},
            ],
        )[1],
    )

    first = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )
    second = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert first["prediction_cache_hit"] is False
    assert second["prediction_cache_hit"] is False
    assert run_calls == {"direct": 2}
    assert fastf1_refresh_calls == {"completion": 1, "results": 1}


def test_execute_live_prediction_pipeline_preserves_actual_cache_when_fastf1_status_unknown(
    patcher,
):
    run_calls = {"direct": 0}

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "stable_sig",
        },
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            run_calls.__setitem__("direct", run_calls["direct"] + 1),
            {
                "qualifying": {
                    "grid_source": "ACTUAL",
                    "grid": [
                        {"position": 1, "driver": "VER", "team": "Red Bull"},
                        {"position": 2, "driver": "LEC", "team": "Ferrari"},
                    ],
                },
                "race": {"finish_order": []},
            },
        )[1],
    )

    patcher.setattr(
        "src.utils.actual_results_fetcher.get_competitive_session_completion_state",
        lambda year, race_name, session_name: "unknown",
    )

    first = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )
    second = pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert first["prediction_cache_hit"] is False
    assert second["prediction_cache_hit"] is True
    assert run_calls == {"direct": 1}


def test_execute_live_prediction_pipeline_reuses_persisted_prediction_when_memory_empty(
    patcher,
):
    run_calls = {"direct": 0}

    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: {"qualifying": {"grid": []}, "race": {"finish_order": []}},
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "stable",
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda race_name, weather, versions, is_sprint=False, year=2026: (
            run_calls.__setitem__("direct", run_calls["direct"] + 1),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    assert output["prediction_cache_hit"] is True
    assert run_calls == {"direct": 0}


def test_execute_live_prediction_pipeline_raises_when_db_mode_requires_persisted_prediction(
    patcher,
):
    patcher.setattr(live_prediction_flow, "should_read_db_first", lambda: True)
    patcher.setattr(
        live_prediction_flow,
        "_get_prediction_precompute_settings",
        lambda: {
            "enabled": True,
            "inline_enabled": False,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)

    with pytest.raises(
        live_prediction_flow.PrecomputedPredictionUnavailableError,
        match="Persisted prediction is not available",
    ):
        live_prediction_flow.execute_live_prediction_pipeline_core(
            race_name="Chinese Grand Prix",
            weather="dry",
            year=2026,
            force_refresh=False,
            progress_callback=None,
            clear_fastf1_race_cache_fn=lambda year, race_name: None,
            auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
            is_sprint_weekend_fn=lambda year, race_name: True,
            detect_event_boundary_refresh_if_needed_fn=lambda year,
            race_name,
            is_sprint,
            session_detector=None: {
                "refresh_needed": False,
                "reason": "no_change",
                "new_sessions": [],
                "boundary_signature": "stable_sig",
                "latest_elapsed_session": None,
            },
            auto_update_practice_characteristics_if_needed_fn=(
                lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                    "updated": False,
                    "completed_fp_sessions": [],
                }
            ),
            clear_resource_cache_fn=lambda: None,
            clear_data_cache_fn=lambda: None,
            get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
            run_prediction_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("run_prediction should not execute in persisted-only mode")
            ),
        )


def test_execute_live_prediction_pipeline_resolves_current_race_boundary_when_refresh_empty(
    patcher,
):
    """Current-race persisted lookup should use resolved boundary when refresh payload omits it."""
    load_calls: list[str] = []
    run_calls = {"direct": 0}

    def _load_precomputed(**kwargs):
        load_calls.append(str(kwargs.get("boundary_signature", "")))
        if str(kwargs.get("boundary_signature", "")) == "sig_resolved":
            return {"qualifying": {"grid": []}, "race": {"finish_order": []}}
        return None

    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", _load_precomputed)
    patcher.setattr(
        live_prediction_flow,
        "_resolve_race_boundary_context",
        lambda year, race_name, is_sprint, session_detector=None: ("sig_resolved", "FP1"),
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "",
            "latest_elapsed_session": None,
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda race_name, weather, versions, is_sprint=False, year=2026: (
            run_calls.__setitem__("direct", run_calls["direct"] + 1),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    assert load_calls == ["sig_resolved"]
    assert output["prediction_cache_hit"] is True
    assert run_calls == {"direct": 0}


def test_execute_live_prediction_pipeline_precomputes_weather_scenarios_after_boundary(
    patcher,
):
    run_calls: list[tuple[str, str]] = []
    save_calls: list[tuple[str, str]] = []

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
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "load_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "save_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(
        live_prediction_flow,
        "save_precomputed_prediction",
        lambda **kwargs: save_calls.append((kwargs["race_name"], kwargs["weather"])),
    )
    patcher.setattr(
        live_prediction_flow,
        "_resolve_precompute_targets",
        lambda year, race_name, horizon_races: [race_name],
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": True,
            "reason": "session_boundary_delta",
            "new_sessions": ["FP1"],
            "boundary_signature": "sig",
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda race_name, weather, versions, is_sprint=False, year=2026: (
            run_calls.append((race_name, weather)),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    assert run_calls == [
        ("Bahrain Grand Prix", "dry"),
        ("Bahrain Grand Prix", "mixed"),
        ("Bahrain Grand Prix", "rain"),
    ]
    assert save_calls == run_calls
    assert output["precompute_summary"]["triggered"] is True
    assert output["precompute_summary"]["generated"] == 2
    assert output["precompute_summary"]["reused"] == 0
    assert output["precompute_summary"]["targets"] == ["Bahrain Grand Prix"]


def test_execute_live_prediction_pipeline_precompute_uses_target_boundary_signatures(patcher):
    save_calls: list[tuple[str, str, str, str]] = []
    horizon_calls: list[dict[str, object]] = []

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
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "load_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(
        live_prediction_flow,
        "save_precompute_horizon_index",
        lambda **kwargs: horizon_calls.append(kwargs),
    )
    patcher.setattr(
        live_prediction_flow,
        "_resolve_precompute_targets",
        lambda year, race_name, horizon_races: [race_name, "Saudi Arabian Grand Prix"],
    )
    patcher.setattr(
        live_prediction_flow,
        "_resolve_race_boundary_context",
        lambda year, race_name, is_sprint, session_detector=None: (
            ("sig_future", "PRE")
            if race_name == "Saudi Arabian Grand Prix"
            else ("sig_anchor", "FP1")
        ),
    )
    patcher.setattr(
        live_prediction_flow,
        "save_precomputed_prediction",
        lambda **kwargs: save_calls.append(
            (
                kwargs["race_name"],
                kwargs["weather"],
                kwargs["boundary_signature"],
                str(kwargs.get("metadata", {}).get("boundary_session_name", "")),
            )
        ),
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": True,
            "reason": "session_boundary_delta",
            "new_sessions": ["FP1"],
            "boundary_signature": "sig_anchor",
            "latest_elapsed_session": "FP1",
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda race_name, weather, versions, is_sprint=False, year=2026: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )

    assert output["precompute_summary"]["triggered"] is True
    assert ("Saudi Arabian Grand Prix", "mixed", "sig_future", "PRE") in save_calls
    assert ("Saudi Arabian Grand Prix", "rain", "sig_future", "PRE") in save_calls
    assert ("Bahrain Grand Prix", "mixed", "sig_anchor", "FP1") in save_calls
    assert horizon_calls
    assert horizon_calls[0]["race_boundaries"] == {
        "Bahrain Grand Prix": "sig_anchor",
        "Saudi Arabian Grand Prix": "sig_future",
    }


def test_execute_live_prediction_pipeline_precompute_horizon_uses_configured_count(patcher):
    horizon_values: list[int] = []

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
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "load_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "save_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "save_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(
        live_prediction_flow,
        "_resolve_precompute_targets",
        lambda year, race_name, horizon_races: (
            horizon_values.append(int(horizon_races)),
            [race_name],
        )[1],
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": True,
            "reason": "session_boundary_delta",
            "new_sessions": ["FP1"],
            "boundary_signature": "sig_scope",
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda race_name, weather, versions, is_sprint=False, year=2026: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )

    assert horizon_values == [3]
    assert output["precompute_summary"]["targets"] == ["Bahrain Grand Prix"]


def test_execute_live_prediction_pipeline_skips_inline_precompute_when_disabled(patcher):
    """Inline horizon precompute should be skipped when inline mode is disabled."""
    run_calls: list[tuple[str, str]] = []

    patcher.setattr(
        live_prediction_flow,
        "_get_prediction_precompute_settings",
        lambda: {
            "enabled": True,
            "inline_enabled": False,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(live_prediction_flow, "load_precomputed_prediction", lambda **kwargs: None)
    patcher.setattr(live_prediction_flow, "load_precompute_horizon_index", lambda **kwargs: None)
    patcher.setattr(
        live_prediction_flow,
        "save_precompute_horizon_index",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("save_precompute_horizon_index should not run when inline is disabled")
        ),
    )
    patcher.setattr(
        live_prediction_flow,
        "save_precomputed_prediction",
        lambda **kwargs: None,
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=True,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": True,
            "reason": "session_boundary_delta",
            "new_sessions": ["FP1"],
            "boundary_signature": "sig_scope",
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda race_name, weather, versions, is_sprint=False, year=2026: (
            run_calls.append((race_name, weather)),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    assert run_calls == [("Bahrain Grand Prix", "dry")]
    assert output["precompute_summary"]["triggered"] is False
    assert output["precompute_summary"]["skipped_reason"] == "inline_precompute_disabled"


def test_execute_live_prediction_pipeline_skips_artifact_updates_in_persisted_mode(patcher):
    """Persisted-only mode should not mutate artifacts in the request path."""
    persisted_prediction = {
        "qualifying": {"grid": []},
        "race": {"finish_order": []},
    }
    race_update_calls: list[bool] = []
    practice_update_calls: list[bool] = []

    patcher.setattr(
        live_prediction_flow,
        "_get_prediction_precompute_settings",
        lambda: {
            "enabled": True,
            "inline_enabled": False,
            "horizon_races": 3,
            "weather_scenarios": ["dry", "mixed", "rain"],
            "max_file_entries": 2048,
        },
    )
    patcher.setattr(live_prediction_flow, "should_read_db_first", lambda: True)
    patcher.setattr(
        live_prediction_flow,
        "load_precomputed_prediction",
        lambda **kwargs: persisted_prediction,
    )

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
        progress_callback=None,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: race_update_calls.append(
            bool(force_recheck)
        ),
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "stable_sig",
            "latest_elapsed_session": None,
        },
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
                practice_update_calls.append(bool(force_recheck)),
                {"updated": True, "completed_fp_sessions": ["FP1"], "teams_updated": 3},
            )[1]
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: {"k": (1, "ts")},
        run_prediction_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("run_prediction should not execute in persisted-only mode")
        ),
    )

    assert race_update_calls == []
    assert practice_update_calls == []
    assert output["practice_update"] == {"updated": False, "completed_fp_sessions": []}
    assert output["prediction_results"] == persisted_prediction


def test_resolve_precompute_targets_skips_testing_by_event_format(patcher):
    """Precompute target resolution should ignore testing rows by EventFormat and name."""
    patcher.setattr(
        "src.utils.weekend.get_schedule_rows",
        lambda year: (
            ("Pre-Season Track Day", "testing"),
            ("Bahrain Grand Prix", "conventional"),
            ("Saudi Arabian Grand Prix", "conventional"),
            ("In-Season Testing", "conventional"),
            ("Australian Grand Prix", "conventional"),
        ),
    )

    targets = live_prediction_flow._resolve_precompute_targets(
        year=2026,
        race_name="Bahrain Grand Prix",
        horizon_races=3,
    )

    assert targets == ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix"]


def test_execute_live_prediction_pipeline_cache_is_reused_across_scopes():
    run_calls = {"direct": 0}

    def _run_prediction(
        race_name: str,
        weather: str,
        versions: dict,
        is_sprint: bool,
        year: int,
    ):
        del race_name, weather, versions, is_sprint, year
        run_calls["direct"] += 1
        return {"qualifying": {"grid": []}, "race": {"finish_order": []}}

    common_kwargs = {
        "race_name": "Bahrain Grand Prix",
        "weather": "dry",
        "year": 2026,
        "force_refresh": False,
        "progress_callback": None,
        "clear_fastf1_race_cache_fn": lambda year, race_name: None,
        "auto_update_if_needed_fn": lambda force_recheck=False, year=2026: None,
        "is_sprint_weekend_fn": lambda year, race_name: False,
        "detect_event_boundary_refresh_if_needed_fn": lambda year,
        race_name,
        is_sprint,
        session_detector=None: {
            "refresh_needed": False,
            "reason": "no_change",
            "new_sessions": [],
            "boundary_signature": "stable",
        },
        "auto_update_practice_characteristics_if_needed_fn": (
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        "clear_resource_cache_fn": lambda: None,
        "clear_data_cache_fn": lambda: None,
        "get_artifact_versions_fn": lambda year=2026: {"k": (1, "ts")},
        "run_prediction_fn": _run_prediction,
    }

    first = live_prediction_flow.execute_live_prediction_pipeline_core(
        **common_kwargs,
    )
    second = live_prediction_flow.execute_live_prediction_pipeline_core(
        **common_kwargs,
    )

    assert first["prediction_cache_hit"] is False
    assert second["prediction_cache_hit"] is True
    assert run_calls == {"direct": 1}


def test_execute_live_prediction_pipeline_reuses_detector_between_boundary_and_practice_checks(
    patcher,
):
    detector_ids: list[int] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "detect_event_boundary_refresh_if_needed",
        lambda year, race_name, is_sprint, session_detector=None: (
            detector_ids.append(id(session_detector)),
            {
                "refresh_needed": False,
                "reason": "no_change",
                "new_sessions": [],
                "boundary_signature": "stable_sig",
            },
        )[1],
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: (
            detector_ids.append(id(session_detector)),
            {"updated": False, "completed_fp_sessions": []},
        )[1],
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: {
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert len(detector_ids) == 2
    assert detector_ids[0] == detector_ids[1]


def test_execute_live_prediction_pipeline_executes_direct_prediction_path(patcher):
    call_order: list[str] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            call_order.append("direct"),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert call_order == ["direct"]


def test_execute_live_prediction_pipeline_force_refresh_uses_direct_prediction_path(patcher):
    call_order: list[str] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages.st,
        "cache_resource",
        type("_CacheResource", (), {"clear": staticmethod(lambda: None)}),
    )
    patcher.setattr(
        pages.st,
        "cache_data",
        type("_CacheData", (), {"clear": staticmethod(lambda: None)}),
    )
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            call_order.append("direct"),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=True,
    )

    assert call_order == ["direct"]


def test_execute_live_prediction_pipeline_rechecks_live_sessions_on_repeated_runs(patcher):
    run_calls = {"direct": 0}

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False, year=2026: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        pages,
        "run_prediction",
        lambda race_name, weather, _versions, is_sprint, year: (
            run_calls.__setitem__("direct", run_calls["direct"] + 1),
            {"qualifying": {"grid": []}, "race": {"finish_order": []}},
        )[1],
    )

    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )
    pages.execute_live_prediction_pipeline(
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
        force_refresh=False,
    )

    assert run_calls == {"direct": 1}
