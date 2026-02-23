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

    errors: list[str] = []
    patcher.setattr(
        pages.fastf1,
        "get_event_schedule",
        lambda _year: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    patcher.setattr(pages.st, "error", lambda message: errors.append(str(message)))

    options = pages._load_race_options()

    assert errors
    assert "Failed to load 2026 calendar" in errors[0]
    assert "Bahrain Grand Prix" in options


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
    patcher.setattr(pages, "render_prediction_accuracy_page", lambda: called.append("accuracy"))
    patcher.setattr(pages, "render_about_page", lambda: called.append("about"))

    pages.render_page("Live Prediction", enable_logging=True)
    pages.render_page("Model & Learning", enable_logging=False)
    pages.render_page("Model Insights", enable_logging=False)
    pages.render_page("Prediction Accuracy", enable_logging=False)
    pages.render_page("About", enable_logging=False)
    pages.render_page("Other", enable_logging=False)

    assert called == ["live", "insights", "insights", "accuracy", "about", "about"]


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
    patcher.setattr(pages.st, "success", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "metric", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "write", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "warning", lambda *_args, **_kwargs: None)
    patcher.setattr(pages.st, "columns", lambda n: [_Ctx() for _ in range(n)])
    patcher.setattr(pages.st, "expander", lambda _label: _Ctx())


def test_render_model_insights_page_executes(patcher):
    _stub_page_streamlit(patcher)
    pages.render_model_insights_page()


def test_render_about_page_executes(patcher):
    _stub_page_streamlit(patcher)
    pages.render_about_page()


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

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)

    pages.render_prediction_accuracy_page()

    assert any("No predictions saved yet" in message for message in messages)


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

    patcher.setattr("src.utils.prediction_logger.PredictionLogger", _Logger)
    patcher.setattr("src.utils.prediction_metrics.PredictionMetrics", _Metrics)

    pages.render_prediction_accuracy_page()

    assert any("Bahrain Grand Prix" in message for message in writes)
