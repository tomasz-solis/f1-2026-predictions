"""Tests for live prediction pipeline orchestration."""

import pytest

from src.dashboard import pages


def test_execute_live_prediction_pipeline_refresh_call_order(patcher):
    call_order: list[str] = []

    patcher.setattr(
        pages, "auto_update_if_needed", lambda force_recheck=False: call_order.append("race_update")
    )
    patcher.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (call_order.append("sprint_check"), True)[1],
    )

    def _practice_update(year: int, race_name: str, is_sprint: bool, force_recheck: bool = False):
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
        lambda: (call_order.append("artifact_versions"), {"k": (1, "ts")})[1],
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
        pages, "auto_update_if_needed", lambda force_recheck=False: call_order.append("race_update")
    )
    patcher.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (call_order.append("sprint_check"), False)[1],
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False: (
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
        lambda: (call_order.append("artifact_versions"), {"k": (4, "ts4")})[1],
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
    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False: (_ for _ in ()).throw(
            RuntimeError("refresh failed")
        ),
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
    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False: None)
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(
        pages,
        "is_sprint_weekend",
        lambda year, race_name: (_ for _ in ()).throw(ValueError("bad race")),
    )
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda: {"k": (3, "ts3")})

    patcher.setattr(
        pages,
        "run_prediction",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_prediction should not execute")),
    )

    with pytest.raises(ValueError, match="bad race"):
        pages.execute_live_prediction_pipeline("Unknown GP", "dry", year=2026, force_refresh=False)


def test_execute_live_prediction_pipeline_emits_progress_and_timing(patcher):
    progress_messages: list[str] = []

    patcher.setattr(pages, "auto_update_if_needed", lambda force_recheck=False: None)
    patcher.setattr(pages, "_clear_fastf1_race_cache", lambda year, race_name: None)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(
        pages,
        "auto_update_practice_characteristics_if_needed",
        lambda year, race_name, is_sprint, force_recheck=False: {
            "updated": False,
            "completed_fp_sessions": [],
        },
    )
    patcher.setattr(pages, "get_artifact_versions", lambda: {"k": (1, "ts")})
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


def test_execute_live_prediction_pipeline_with_force_refresh_clears_cache_and_rechecks(patcher):
    """Test that force_refresh=True clears FastF1 cache and forces session recheck."""
    call_order: list[str] = []
    force_recheck_calls = {"race_update": False, "practice_update": False}

    def mock_race_update(force_recheck=False):
        call_order.append("race_update")
        force_recheck_calls["race_update"] = force_recheck

    def mock_practice_update(year, race_name, is_sprint, force_recheck=False):
        call_order.append("practice_update")
        force_recheck_calls["practice_update"] = force_recheck
        return {"updated": False, "completed_fp_sessions": []}

    patcher.setattr(pages, "auto_update_if_needed", mock_race_update)
    patcher.setattr(pages, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(pages, "auto_update_practice_characteristics_if_needed", mock_practice_update)
    patcher.setattr(
        pages, "_clear_fastf1_race_cache", lambda year, race_name: call_order.append("cache_clear")
    )
    patcher.setattr(pages, "get_artifact_versions", lambda: {"k": (1, "ts")})
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
