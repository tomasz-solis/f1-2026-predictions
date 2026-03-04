"""Tests for dashboard page helpers and routing."""

import pandas as pd

from src.dashboard import pages


def test_load_race_options_filters_testing_and_tags_sprint(patcher):
    pages._load_race_options_cached.clear()

    schedule = pd.DataFrame(
        {
            "EventName": [
                "Bahrain Grand Prix",
                "Chinese Grand Prix",
                "Pre-Season Testing",
            ],
            "EventFormat": ["conventional", "sprint", None],
        }
    )

    patcher.setattr(pages.fastf1, "get_event_schedule", lambda year: schedule)
    patcher.setattr(pages.st, "error", lambda _msg: (_ for _ in ()).throw(AssertionError))

    options = pages._load_race_options()

    assert options == ["Bahrain Grand Prix", "Chinese Grand Prix (Sprint)"]


def test_load_race_options_uses_fallback_when_schedule_fails(patcher):
    pages._load_race_options_cached.clear()

    warnings: list[str] = []
    patcher.setattr(
        pages.fastf1,
        "get_event_schedule",
        lambda _year: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    patcher.setattr(
        pages,
        "_get_schedule_rows",
        lambda _year: (("Bahrain Grand Prix", "conventional"), ("Chinese Grand Prix", "sprint")),
    )
    patcher.setattr(pages.st, "warning", lambda message: warnings.append(str(message)))
    patcher.setattr(pages.st, "error", lambda _message: (_ for _ in ()).throw(AssertionError))

    options = pages._load_race_options()

    assert warnings == []
    assert options == ["Bahrain Grand Prix", "Chinese Grand Prix (Sprint)"]


def test_load_race_options_warns_when_fastf1_and_fallback_unavailable(patcher):
    pages._load_race_options_cached.clear()

    warnings: list[str] = []
    patcher.setattr(
        pages.fastf1,
        "get_event_schedule",
        lambda _year: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    patcher.setattr(pages, "_get_schedule_rows", lambda _year: tuple())
    patcher.setattr(pages.st, "warning", lambda message: warnings.append(str(message)))

    options = pages._load_race_options()

    assert warnings
    assert "Failed to load 2026 calendar" in warnings[0]
    assert "Bahrain Grand Prix" in options


def test_load_race_options_uses_requested_year(patcher):
    pages._load_race_options_cached.clear()

    years_seen: list[int] = []

    def _get_schedule(year: int):
        years_seen.append(year)
        return pd.DataFrame(
            {
                "EventName": ["Bahrain Grand Prix"],
                "EventFormat": ["conventional"],
            }
        )

    patcher.setattr(pages.fastf1, "get_event_schedule", _get_schedule)
    patcher.setattr(pages.st, "error", lambda _msg: (_ for _ in ()).throw(AssertionError))

    options = pages._load_race_options(2027)

    assert years_seen == [2027]
    assert options == ["Bahrain Grand Prix"]


def test_cache_dir_race_matching_handles_date_prefixed_event_dirs():
    assert pages._cache_dir_matches_race(
        "2025-04-13_Bahrain_Grand_Prix",
        "Bahrain Grand Prix",
    )
    assert pages._cache_dir_matches_race(
        "BahrainGrandPrix",
        "Bahrain Grand Prix",
    )
    assert not pages._cache_dir_matches_race(
        "2025-04-20_Saudi_Arabian_Grand_Prix",
        "Bahrain Grand Prix",
    )
    assert pages._cache_dir_matches_race(
        "2025-11-09_Sao_Paulo_Grand_Prix",
        "São Paulo Grand Prix",
    )


def test_latest_data_status_message_prefers_latest_elapsed_session():
    message = pages._latest_data_status_message(
        race_name="Australian Grand Prix",
        year=2026,
        boundary_refresh={"latest_elapsed_session": "FP2"},
        practice_update={"completed_fp_sessions": ["FP1", "FP2"]},
    )

    assert "Latest datapoint in use: Australian Grand Prix 2026 - Free Practice 2 (FP2)" in message


def test_latest_data_status_message_uses_practice_sessions_when_no_elapsed():
    message = pages._latest_data_status_message(
        race_name="Australian Grand Prix",
        year=2026,
        boundary_refresh={"latest_elapsed_session": None},
        practice_update={"completed_fp_sessions": ["FP1"]},
    )

    assert "Latest datapoint in use: Australian Grand Prix 2026 - Free Practice 1 (FP1)" in message


def test_latest_data_status_message_handles_schedule_unavailable():
    message = pages._latest_data_status_message(
        race_name="Australian Grand Prix",
        year=2026,
        boundary_refresh={"reason": "schedule_unavailable"},
        practice_update={},
    )

    assert "schedule is currently unavailable" in message


def test_clear_fastf1_race_cache_removes_date_prefixed_race_dirs_only(patcher, tmp_path):
    primary_cache = tmp_path / "fastf1_cache"
    testing_cache = tmp_path / "fastf1_cache_testing"

    target_primary = primary_cache / "2025" / "2025-04-13_Bahrain_Grand_Prix"
    target_testing = testing_cache / "2025" / "2025-04-13_Bahrain_Grand_Prix"
    untouched_other_race = primary_cache / "2025" / "2025-04-20_Saudi_Arabian_Grand_Prix"

    target_primary.mkdir(parents=True, exist_ok=True)
    target_testing.mkdir(parents=True, exist_ok=True)
    untouched_other_race.mkdir(parents=True, exist_ok=True)

    (target_primary / "marker.txt").write_text("stale")
    (target_testing / "marker.txt").write_text("stale")
    (untouched_other_race / "marker.txt").write_text("keep")

    patcher.setattr(pages, "_FASTF1_CACHE_DIRS", (primary_cache, testing_cache))

    pages._clear_fastf1_race_cache(2025, "Bahrain Grand Prix")

    assert not target_primary.exists()
    assert not target_testing.exists()
    assert untouched_other_race.exists()


def test_save_prediction_if_enabled_saves_new_session(patcher):
    saved_payload: dict = {}
    info_messages: list[str] = []

    class _Detector:
        def get_latest_completed_session(self, year: int, race_name: str, is_sprint: bool):
            assert year == 2026
            assert race_name == "Bahrain Grand Prix"
            assert is_sprint is False
            return "FP3"

    class _Logger:
        def has_prediction_for_session(self, year: int, race_name: str, session_name: str):
            assert (year, race_name, session_name) == (2026, "Bahrain Grand Prix", "FP3")
            return False

        def save_prediction(self, **kwargs):
            saved_payload.update(kwargs)

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr(pages.st, "info", lambda message: info_messages.append(str(message)))
    patcher.setattr(pages.st, "warning", lambda _message: None)

    pages._save_prediction_if_enabled(
        enable_logging=True,
        prediction_results={
            "qualifying": {"grid": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]},
            "race": {"finish_order": [{"driver": "VER", "team": "Red Bull Racing", "position": 1}]},
        },
        is_sprint=False,
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
    )

    assert saved_payload["year"] == 2026
    assert saved_payload["race_name"] == "Bahrain Grand Prix"
    assert saved_payload["session_name"] == "FP3"
    assert saved_payload["weather"] == "dry"
    assert "Prediction saved for accuracy tracking (after FP3)" in info_messages


def test_save_prediction_if_enabled_persists_checkpoint_summary_when_store_available(patcher):
    info_messages: list[str] = []
    checkpoint_saves: list[dict] = []

    class _Detector:
        def get_latest_completed_session(self, year: int, race_name: str, is_sprint: bool):
            assert (year, race_name, is_sprint) == (2026, "Bahrain Grand Prix", False)
            return "FP2"

    class _ArtifactStore:
        def save_artifact(self, **kwargs):
            checkpoint_saves.append(kwargs)

    class _Logger:
        def __init__(self):
            self.artifact_store = _ArtifactStore()

        def has_prediction_for_session(self, year: int, race_name: str, session_name: str):
            assert (year, race_name, session_name) == (2026, "Bahrain Grand Prix", "FP2")
            return False

        def save_prediction(self, **kwargs):
            assert kwargs["session_name"] == "FP2"

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr(pages.st, "info", lambda message: info_messages.append(str(message)))
    patcher.setattr(pages.st, "warning", lambda _message: None)

    pages._save_prediction_if_enabled(
        enable_logging=True,
        prediction_results={
            "qualifying": {
                "grid_source": "PREDICTED",
                "data_source": "Short-stint blend",
                "grid": [
                    {"driver": "VER", "team": "Red Bull Racing", "position": 1, "confidence": 61.0}
                ],
            },
            "race": {
                "grid_source": "PREDICTED",
                "finish_order": [
                    {"driver": "VER", "team": "Red Bull Racing", "position": 1, "confidence": 58.0}
                ],
            },
        },
        is_sprint=False,
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
    )

    assert len(checkpoint_saves) == 1
    assert checkpoint_saves[0]["artifact_type"] == "prediction_checkpoint"
    assert checkpoint_saves[0]["artifact_key"] == "2026::Bahrain Grand Prix::FP2"
    payload = checkpoint_saves[0]["data"]
    assert payload["metadata"]["session_name"] == "FP2"
    assert payload["qualifying"]["mean_confidence"] == 61.0
    assert payload["race"]["mean_confidence"] == 58.0
    assert "Prediction saved for accuracy tracking (after FP2)" in info_messages


def test_save_prediction_if_enabled_reports_existing_prediction(patcher):
    info_messages: list[str] = []

    class _Detector:
        def get_latest_completed_session(self, year: int, race_name: str, is_sprint: bool):
            return "SQ"

    class _Logger:
        def has_prediction_for_session(self, year: int, race_name: str, session_name: str):
            return True

        def save_prediction(self, **_kwargs):
            raise AssertionError("save should not be called")

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr(pages.st, "info", lambda message: info_messages.append(str(message)))

    pages._save_prediction_if_enabled(
        enable_logging=True,
        prediction_results={
            "main_quali": {"grid": []},
            "main_race": {"finish_order": []},
        },
        is_sprint=True,
        race_name="Chinese Grand Prix",
        weather="dry",
        year=2026,
    )

    assert "Prediction for SQ already saved (max 1 per session)" in info_messages


def test_save_prediction_if_enabled_handles_no_completed_sessions(patcher):
    info_messages: list[str] = []

    class _Detector:
        def get_latest_completed_session(self, year: int, race_name: str, is_sprint: bool):
            return None

    class _Logger:
        def has_prediction_for_session(self, year: int, race_name: str, session_name: str):
            return False

    patcher.setattr("src.utils.session_detector.SessionDetector", _Detector)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr(pages.st, "info", lambda message: info_messages.append(str(message)))

    pages._save_prediction_if_enabled(
        enable_logging=True,
        prediction_results={
            "qualifying": {"grid": []},
            "race": {"finish_order": []},
        },
        is_sprint=False,
        race_name="Bahrain Grand Prix",
        weather="dry",
        year=2026,
    )

    assert "No completed sessions yet; prediction not saved" in info_messages[0]


def test_render_prediction_results_routes_normal_weekend(patcher):
    rendered_sections: list[str] = []

    patcher.setattr(pages.st, "success", lambda _msg: None)
    patcher.setattr(pages.st, "markdown", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "header", lambda _msg: None)
    patcher.setattr(pages.st, "info", lambda _msg: None)
    patcher.setattr(
        pages,
        "display_prediction_result",
        lambda _result, title, is_race=False: rendered_sections.append(
            f"{title}:{'race' if is_race else 'quali'}"
        ),
    )

    pages._render_prediction_results(
        prediction_results={
            "qualifying": {"timing": {"total": 1.1}, "grid": []},
            "race": {"finish_order": []},
        },
        is_sprint=False,
    )

    assert rendered_sections == [
        "Qualifying Prediction:quali",
        "Race Prediction:race",
    ]


def test_render_prediction_results_reports_cache_hit_runtime_from_pipeline(patcher):
    rendered_sections: list[str] = []
    success_messages: list[str] = []

    patcher.setattr(pages.st, "success", lambda message: success_messages.append(str(message)))
    patcher.setattr(pages.st, "markdown", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "header", lambda _msg: None)
    patcher.setattr(pages.st, "info", lambda _msg: None)
    patcher.setattr(
        pages,
        "display_prediction_result",
        lambda _result, title, is_race=False: rendered_sections.append(
            f"{title}:{'race' if is_race else 'quali'}"
        ),
    )

    pages._render_prediction_results(
        prediction_results={
            "qualifying": {"timing": {"total": 12.65}, "grid": []},
            "race": {"finish_order": []},
        },
        is_sprint=False,
        prediction_cache_hit=True,
        pipeline_timing={"total": 0.1},
    )

    assert success_messages == ["Prediction loaded from cache in 0.10s"]
    assert rendered_sections == [
        "Qualifying Prediction:quali",
        "Race Prediction:race",
    ]


def test_render_prediction_results_routes_sprint_weekend(patcher):
    rendered_sections: list[str] = []

    patcher.setattr(pages.st, "success", lambda _msg: None)
    patcher.setattr(pages.st, "markdown", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "header", lambda _msg: None)
    patcher.setattr(pages.st, "info", lambda _msg: None)
    patcher.setattr(
        pages,
        "display_prediction_result",
        lambda _result, title, is_race=False: rendered_sections.append(
            f"{title}:{'race' if is_race else 'quali'}"
        ),
    )

    pages._render_prediction_results(
        prediction_results={
            "sprint_quali": {"timing": {"total": 1.2}, "grid": []},
            "sprint_race": {"finish_order": []},
            "main_quali": {"grid": []},
            "main_race": {"finish_order": []},
        },
        is_sprint=True,
    )

    assert rendered_sections == [
        "Sprint Qualifying Prediction:quali",
        "Sprint Race Prediction:race",
        "Main Qualifying Prediction:quali",
        "Main Race Prediction:race",
    ]


def test_render_page_routes_by_selected_tab(patcher):
    called: list[str] = []

    patcher.setattr(pages, "render_live_prediction_page", lambda _enabled: called.append("live"))
    patcher.setattr(pages, "render_model_insights_page", lambda: called.append("insights"))
    patcher.setattr(pages, "render_team_comparison_page", lambda: called.append("comparison"))
    patcher.setattr(pages, "render_prediction_accuracy_page", lambda: called.append("accuracy"))
    patcher.setattr(pages, "render_contact_page", lambda: called.append("contact"))

    pages.render_page("Prediction", enable_logging=True)
    pages.render_page("Live Prediction", enable_logging=True)
    pages.render_page("Model & Learning", enable_logging=False)
    pages.render_page("Model Insights", enable_logging=False)
    pages.render_page("Team Comparison", enable_logging=False)
    pages.render_page("Prediction Accuracy", enable_logging=False)
    pages.render_page("Contact", enable_logging=False)
    pages.render_page("About", enable_logging=False)
    pages.render_page("Other", enable_logging=False)

    assert called == [
        "live",
        "live",
        "insights",
        "insights",
        "comparison",
        "accuracy",
        "contact",
        "contact",
        "live",
    ]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_page_streamlit(patcher):
    patcher.setattr(pages.st, "header", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "subheader", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "markdown", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "info", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "caption", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "success", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "metric", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "write", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "dataframe", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "plotly_chart", lambda *_args, **_kwargs: None)
    patcher.setattr(
        pages.st,
        "selectbox",
        lambda _label, options, index=0, **_kwargs: options[index] if options else None,
    )
    patcher.setattr(
        pages.st,
        "multiselect",
        lambda _label, options, default=None, **_kwargs: default if default is not None else [],
    )
    patcher.setattr(pages.st, "warning", lambda *_args, **_kwargs: None)
    patcher.setattr(
        pages.st,
        "columns",
        lambda n, **_kwargs: [_Ctx() for _ in range(n if isinstance(n, int) else len(n))],
    )
    patcher.setattr(pages.st, "container", lambda *_args, **_kwargs: _Ctx())
    patcher.setattr(pages.st, "expander", lambda _label: _Ctx())


def test_render_model_insights_page_executes(patcher):
    _stub_page_streamlit(patcher)
    pages.render_model_insights_page()


def test_render_team_comparison_page_executes(patcher):
    _stub_page_streamlit(patcher)
    calls: list[int] = []
    patcher.setattr(pages, "_render_team_comparison_section", lambda year: calls.append(year))
    patcher.setattr(pages, "_get_selected_season", lambda default=pages.DEFAULT_SEASON: 2026)

    pages.render_team_comparison_page()

    assert calls == [pages.DEFAULT_SEASON]


def test_render_team_comparison_page_uses_selected_season(patcher):
    _stub_page_streamlit(patcher)
    calls: list[int] = []
    patcher.setattr(pages, "_render_team_comparison_section", lambda year: calls.append(year))
    patcher.setattr(pages, "_get_selected_season", lambda default=pages.DEFAULT_SEASON: 2027)

    pages.render_team_comparison_page()

    assert calls == [2027]


def test_render_contact_page_executes(patcher):
    _stub_page_streamlit(patcher)
    pages.render_contact_page()


def test_render_prediction_accuracy_page_handles_no_predictions(patcher):
    _stub_page_streamlit(patcher)
    messages: list[str] = []
    patcher.setattr(pages.st, "info", lambda message: messages.append(str(message)))

    class _Logger:
        def get_all_predictions(self, year: int):
            assert year == pages.DEFAULT_SEASON
            return []

    class _Metrics:
        pass

    patcher.setattr(pages, "_get_selected_season", lambda default=pages.DEFAULT_SEASON: 2026)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)

    pages.render_prediction_accuracy_page()

    assert any("No predictions saved yet" in message for message in messages)


def test_render_prediction_accuracy_page_uses_selected_season(patcher):
    _stub_page_streamlit(patcher)

    class _Logger:
        def get_all_predictions(self, year: int):
            assert year == 2027
            return []

    class _Metrics:
        pass

    patcher.setattr(pages, "_get_selected_season", lambda default=pages.DEFAULT_SEASON: 2027)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)
    patcher.setattr(pages.st, "info", lambda _message: None)

    pages.render_prediction_accuracy_page()


def test_render_prediction_accuracy_page_with_actuals(patcher):
    _stub_page_streamlit(patcher)
    writes: list[str] = []
    patcher.setattr(pages.st, "write", lambda message: writes.append(str(message)))

    prediction_record = {
        "metadata": {"race_name": "Bahrain Grand Prix", "session_name": "FP3"},
        "actuals": {"qualifying": [{"driver": "VER"}], "race": [{"driver": "VER"}]},
    }

    class _Logger:
        def get_all_predictions(self, year: int):
            assert year == pages.DEFAULT_SEASON
            return [prediction_record]

    class _Metrics:
        def aggregate_metrics(self, _predictions):
            return {
                "qualifying": {
                    "exact_accuracy": {"mean": 45.0},
                    "mae": {"mean": 2.1},
                    "within_3": {"mean": 70.0},
                    "correlation": {"mean": 0.81},
                },
                "race": {
                    "exact_accuracy": {"mean": 35.0},
                    "mae": {"mean": 2.8},
                    "within_3": {"mean": 62.0},
                    "winner_accuracy": {"percentage": 25.0},
                },
            }

        def calculate_all_metrics(self, _prediction):
            return {
                "metadata": {"race_name": "Bahrain Grand Prix", "session_name": "FP3"},
                "qualifying": {
                    "exact_accuracy": 45.0,
                    "mae": 2.1,
                    "within_1": 30.0,
                    "correlation": 0.81,
                },
                "race": {
                    "exact_accuracy": 35.0,
                    "mae": 2.8,
                    "within_3": 62.0,
                    "winner_correct": True,
                    "podium": {"correct_drivers": 2},
                },
            }

    patcher.setattr(pages, "_get_selected_season", lambda default=pages.DEFAULT_SEASON: 2026)
    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)

    pages.render_prediction_accuracy_page()

    assert any("Bahrain Grand Prix" in message for message in writes)


def test_render_live_prediction_page_passes_selected_season_to_pipeline_and_save(patcher):
    _stub_page_streamlit(patcher)

    selected_years: dict[str, int | bool] = {}
    error_messages: list[str] = []

    def _selectbox(label, options, index=0, **_kwargs):
        if label == "Season":
            return 2027
        if label == "Select Grand Prix":
            return "Bahrain Grand Prix"
        if label == "Weather Forecast":
            return "dry"
        return options[index] if options else None

    patcher.setattr(pages.st, "selectbox", _selectbox)
    patcher.setattr(pages.st, "toggle", lambda *_args, **_kwargs: False)
    patcher.setattr(pages.st, "button", lambda *_args, **_kwargs: True)
    patcher.setattr(pages.st, "error", lambda message: error_messages.append(str(message)))
    patcher.setattr(pages.st, "spinner", lambda *_args, **_kwargs: _Ctx())
    patcher.setattr(
        pages.st,
        "empty",
        lambda: type(
            "_Status",
            (),
            {"info": lambda self, _msg: None, "empty": lambda self: None},
        )(),
    )
    patcher.setattr(
        pages, "_load_race_options", lambda year=pages.DEFAULT_SEASON: ["Bahrain Grand Prix"]
    )
    patcher.setattr(
        pages,
        "execute_live_prediction_pipeline",
        lambda race_name,
        weather,
        year,
        force_refresh,
        progress_callback=None,
        precompute_include_next_weekend=None: (
            (
                selected_years.__setitem__("pipeline", year),
                selected_years.__setitem__(
                    "precompute_scope", bool(precompute_include_next_weekend)
                ),
                {
                    "prediction_results": {
                        "qualifying": {"grid": []},
                        "race": {"finish_order": []},
                    },
                    "is_sprint": False,
                    "practice_update": {"updated": False, "completed_fp_sessions": []},
                    "pipeline_timing": {},
                },
            )[2]
        ),
    )
    patcher.setattr(
        pages,
        "_save_prediction_if_enabled",
        lambda **kwargs: selected_years.__setitem__("save", kwargs["year"]),
    )
    patcher.setattr(pages, "_render_prediction_results", lambda *_args, **_kwargs: None)

    pages.render_live_prediction_page(enable_logging=False)

    assert error_messages == []
    assert selected_years["pipeline"] == 2027
    assert selected_years["precompute_scope"] is False


def test_render_live_prediction_page_keeps_precompute_scope_locked_to_selected_gp_only(patcher):
    _stub_page_streamlit(patcher)
    captured_scope: dict[str, bool] = {}

    def _selectbox(label, options, index=0, **_kwargs):
        if label == "Season":
            return 2027
        if label == "Select Grand Prix":
            return "Bahrain Grand Prix"
        if label == "Weather Forecast":
            return "dry"
        return options[index] if options else None

    patcher.setattr(pages.st, "selectbox", _selectbox)
    patcher.setattr(pages.st, "toggle", lambda *_args, **_kwargs: False)
    patcher.setattr(pages.st, "button", lambda *_args, **_kwargs: True)
    patcher.setattr(pages.st, "spinner", lambda *_args, **_kwargs: _Ctx())
    patcher.setattr(
        pages.st,
        "empty",
        lambda: type(
            "_Status",
            (),
            {"info": lambda self, _msg: None, "empty": lambda self: None},
        )(),
    )
    patcher.setattr(
        pages, "_load_race_options", lambda year=pages.DEFAULT_SEASON: ["Bahrain Grand Prix"]
    )
    patcher.setattr(
        pages,
        "execute_live_prediction_pipeline",
        lambda race_name,
        weather,
        year,
        force_refresh,
        progress_callback=None,
        precompute_include_next_weekend=None: (
            (
                captured_scope.__setitem__("value", bool(precompute_include_next_weekend)),
                {
                    "prediction_results": {
                        "qualifying": {"grid": []},
                        "race": {"finish_order": []},
                    },
                    "is_sprint": False,
                    "practice_update": {"updated": False, "completed_fp_sessions": []},
                    "pipeline_timing": {},
                },
            )[1]
        ),
    )
    patcher.setattr(pages, "_save_prediction_if_enabled", lambda **kwargs: None)
    patcher.setattr(pages, "_render_prediction_results", lambda *_args, **_kwargs: None)

    pages.render_live_prediction_page(enable_logging=False)

    assert captured_scope == {"value": False}


def test_build_team_comparison_dataframe_uses_profile_metrics():
    teams_payload = {
        "Team A": {
            "overall_performance": 0.8,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.7,
                    "slow_corner_performance": 0.6,
                    "medium_corner_performance": 0.5,
                    "fast_corner_performance": 0.4,
                    "braking_performance": 0.65,
                    "top_speed": 0.55,
                    "tire_deg_performance": 0.75,
                }
            },
        },
        "Team B": {
            "overall_performance": 0.7,
            "testing_characteristics": {
                "run_profile": "balanced",
                "overall_pace": 0.2,
            },
        },
    }

    frame, neutral_fallbacks = pages._build_team_comparison_dataframe(
        teams_payload=teams_payload,
        selected_teams=["Team A", "Team B"],
        profile="balanced",
    )

    assert list(frame["Team"]) == ["Team A", "Team B"]
    assert frame.loc[frame["Team"] == "Team A", "Slow Corners"].iloc[0] == 0.6
    assert frame.loc[frame["Team"] == "Team B", "Slow Corners"].iloc[0] == 0.5
    assert neutral_fallbacks > 0


def test_team_brand_color_uses_flagship_palette():
    assert pages._team_brand_color("Ferrari") == "#DC0000"
    assert pages._team_brand_color("Scuderia Ferrari") == "#DC0000"
    assert pages._team_brand_color("McLaren") == "#FF8700"
    assert pages._team_brand_color("Unknown Team") == pages._DEFAULT_TEAM_COLOR


def test_default_team_selection_prefers_big4_order():
    teams = ["Williams", "Ferrari", "McLaren", "Red Bull Racing", "Mercedes", "Aston Martin"]

    selected = pages._default_team_selection(teams, max_teams=4)

    assert selected == ["McLaren", "Mercedes", "Ferrari", "Red Bull Racing"]
