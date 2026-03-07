"""Tests for dashboard rendering helpers."""

import pandas as pd

from src.dashboard import rendering


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_streamlit(patcher):
    calls: list[tuple[str, str]] = []

    patcher.setattr(rendering.st, "subheader", lambda msg: calls.append(("subheader", str(msg))))
    patcher.setattr(rendering.st, "caption", lambda msg: calls.append(("caption", str(msg))))
    patcher.setattr(rendering.st, "info", lambda msg: calls.append(("info", str(msg))))
    patcher.setattr(rendering.st, "warning", lambda msg: calls.append(("warning", str(msg))))
    patcher.setattr(rendering.st, "success", lambda msg: calls.append(("success", str(msg))))
    patcher.setattr(rendering.st, "header", lambda msg: calls.append(("header", str(msg))))
    patcher.setattr(
        rendering.st, "markdown", lambda msg, **_kwargs: calls.append(("markdown", str(msg)))
    )
    patcher.setattr(
        rendering.st,
        "metric",
        lambda *args, **kwargs: calls.append(
            ("metric", str(kwargs.get("label", args[0] if args else "")))
        ),
    )
    patcher.setattr(rendering.st, "progress", lambda *_args, **_kwargs: None)
    patcher.setattr(rendering.st, "write", lambda msg: calls.append(("write", str(msg))))
    patcher.setattr(rendering.st, "dataframe", lambda *_args, **_kwargs: None)
    patcher.setattr(rendering.st, "columns", lambda n: [_Ctx() for _ in range(n)])

    def _expander(label, *_, **__):
        calls.append(("expander", str(label)))
        return _Ctx()

    patcher.setattr(rendering.st, "expander", _expander)

    return calls


def test_render_compound_strategies_shows_top_entries(patcher):
    calls = _stub_streamlit(patcher)

    rendering._render_compound_strategies(
        {
            "SOFT->MEDIUM": 0.42,
            "MEDIUM->HARD": 0.35,
            "SOFT->HARD": 0.15,
            "HARD->MEDIUM": 0.08,
        }
    )

    assert ("subheader", "Tire Compound Strategies") in calls
    metric_labels = [value for kind, value in calls if kind == "metric"]
    assert metric_labels[:3] == ["SOFT->MEDIUM", "MEDIUM->HARD", "SOFT->HARD"]


def test_render_pit_lap_distribution_builds_summary(patcher):
    calls = _stub_streamlit(patcher)

    rendering._render_pit_lap_distribution({"lap_10-15": 10, "lap_20-25": 30, "lap_15-20": 20})

    assert ("subheader", "Pit Stop Windows") in calls
    info_messages = [value for kind, value in calls if kind == "info"]
    assert any("Most likely pit window" in msg for msg in info_messages)


def test_render_track_temperature_context_shows_blend_details(patcher):
    calls = _stub_streamlit(patcher)

    rendering._render_track_temperature_context(
        {
            "track_temperature_context": {
                "track_temperature_c": 31.4,
                "source": "session_weather_blend",
                "session_name": "Q",
                "session_temperature_source": "track_temp",
                "session_weight": 0.70,
                "forecast_weight": 0.30,
            }
        }
    )

    info_messages = [value for kind, value in calls if kind == "info"]
    assert any(
        "Track temperature input: 31.4C (70% Q weather + 30% race-weather baseline)" == message
        for message in info_messages
    )


def test_render_weather_feature_context_shows_practice_source(patcher):
    calls = _stub_streamlit(patcher)

    rendering._render_weather_feature_context(
        {
            "weather_feature_context": {
                "available": True,
                "source_session": "FP3",
                "selected_weather": "dry",
                "practice_weather_bucket": "dry",
                "chaos_multiplier": 1.04,
            }
        }
    )

    info_messages = [value for kind, value in calls if kind == "info"]
    assert any(
        "Weather feature input: FP3 practice weather (dry). Scenario selected: dry. Uncertainty adjustment active (chaos x1.04)."
        == message
        for message in info_messages
    )


def test_render_race_result_warns_on_high_dnf(patcher):
    calls = _stub_streamlit(patcher)

    df = pd.DataFrame(
        [
            {
                "position": 1,
                "driver": "VER",
                "team": "Red Bull Racing",
                "confidence": 65.2,
                "podium_probability": 70.1,
                "dnf_probability": 0.05,
            },
            {
                "position": 2,
                "driver": "NOR",
                "team": "McLaren",
                "confidence": 61.4,
                "podium_probability": 58.2,
                "dnf_probability": 0.30,
            },
            {
                "position": 3,
                "driver": "LEC",
                "team": "Ferrari",
                "confidence": 59.8,
                "podium_probability": 54.4,
                "dnf_probability": 0.15,
            },
        ]
    )

    rendering._render_race_result(df)

    warning_messages = [value for kind, value in calls if kind == "warning"]
    assert any("High DNF risk" in msg for msg in warning_messages)


def test_render_race_result_explains_sorting_and_interval(patcher):
    calls = _stub_streamlit(patcher)

    df = pd.DataFrame(
        [
            {
                "position": 1,
                "driver": "VER",
                "team": "Red Bull Racing",
                "position_blend_score": 1.82,
                "confidence": 58.0,
                "podium_probability": 64.2,
                "dnf_probability": 0.04,
                "p5": 1,
                "p95": 4,
            }
        ]
    )

    rendering._render_race_result(df)

    captions = [value for kind, value in calls if kind == "caption"]
    table_html_blocks = [value for kind, value in calls if kind == "markdown" and "<table" in value]
    assert any("Rows are ranked by expected finishing position" in text for text in captions)
    assert any("90% Pos Range" in text for text in captions)
    assert table_html_blocks
    assert all(">Status<" not in html for html in table_html_blocks)


def test_render_race_result_warns_on_low_confidence_signals(patcher):
    calls = _stub_streamlit(patcher)

    df = pd.DataFrame(
        [
            {
                "position": 1,
                "driver": "VER",
                "team": "Red Bull Racing",
                "position_blend_score": 2.10,
                "confidence": 49.0,
                "podium_probability": 41.0,
                "dnf_probability": 0.08,
                "p5": 1,
                "p95": 11,
            },
            {
                "position": 2,
                "driver": "NOR",
                "team": "McLaren",
                "position_blend_score": 2.44,
                "confidence": 50.0,
                "podium_probability": 40.0,
                "dnf_probability": 0.09,
                "p5": 1,
                "p95": 10,
            },
        ]
    )
    df.attrs["input_confidence"] = 0.42

    rendering._render_race_result(df)

    warnings = [value for kind, value in calls if kind == "warning"]
    details = [value for kind, value in calls if kind == "markdown"]
    assert any("Low confidence run" in text for text in warnings)
    assert any("(+1 more)" in text for text in warnings)
    assert any("Low confidence run" in text for text in details)
    assert any("Low input-data confidence" in text for text in details)


def test_render_qualifying_result_splits_grid_columns(patcher):
    calls = _stub_streamlit(patcher)
    df = pd.DataFrame(
        [{"position": idx, "driver": f"D{idx:02d}", "team": "Team"} for idx in range(1, 23)]
    )

    rendering._render_qualifying_result(df)

    markdown_blocks = [value for kind, value in calls if kind == "markdown"]
    assert any("Q1 Eliminated (Final Grid P17-P22)" in block for block in markdown_blocks)
    assert any("Q2 Eliminated (Final Grid P11-P16)" in block for block in markdown_blocks)
    assert any("Q3 Shootout (Final Grid P1-P10)" in block for block in markdown_blocks)


def test_display_prediction_result_routes_race_sections(patcher):
    calls = _stub_streamlit(patcher)
    routed: list[str] = []

    patcher.setattr(
        rendering,
        "_render_compound_strategies",
        lambda _strategies: routed.append("compound"),
    )
    patcher.setattr(
        rendering,
        "_render_pit_lap_distribution",
        lambda _distribution: routed.append("pit"),
    )
    patcher.setattr(rendering, "_render_race_result", lambda _df: routed.append("race"))

    rendering.display_prediction_result(
        result={
            "grid_source": "ACTUAL",
            "finish_order": [
                {
                    "position": 1,
                    "driver": "VER",
                    "team": "Red Bull Racing",
                    "confidence": 62.0,
                    "podium_probability": 68.0,
                    "dnf_probability": 0.07,
                }
            ],
            "compound_strategies": {"SOFT->MEDIUM": 1.0},
            "pit_lap_distribution": {"lap_15-20": 20},
            "characteristics_profile_used": "long_run",
            "teams_with_characteristics_profile": 11,
            "track_temperature_context": {
                "track_temperature_c": 31.4,
                "source": "session_weather_blend",
                "session_name": "Q",
                "session_temperature_source": "track_temp",
                "session_weight": 0.70,
                "forecast_weight": 0.30,
            },
            "weather_feature_context": {
                "available": True,
                "source_session": "FP3",
                "selected_weather": "dry",
                "practice_weather_bucket": "dry",
                "wind_speed_kph": 18.0,
                "chaos_multiplier": 1.04,
            },
        },
        prediction_name="Race Prediction",
        is_race=True,
    )

    assert routed == ["compound", "pit", "race"]
    assert ("success", "Using ACTUAL grid from completed session") in calls


def test_display_prediction_result_routes_qualifying_sections(patcher):
    calls = _stub_streamlit(patcher)
    routed: list[str] = []

    patcher.setattr(rendering, "_render_qualifying_result", lambda _df: routed.append("quali"))

    rendering.display_prediction_result(
        result={
            "grid_source": "PREDICTED",
            "data_source": "Short-stint blend (FP3 + FP2 + FP1)",
            "blend_used": True,
            "grid": [{"position": 1, "driver": "VER", "team": "Red Bull Racing"}],
        },
        prediction_name="Qualifying Prediction",
        is_race=False,
    )

    assert routed == ["quali"]
    warnings = [value for kind, value in calls if kind == "warning"]
    details = [value for kind, value in calls if kind == "markdown"]
    assert any("Grid source: PREDICTED qualifying grid." in text for text in warnings)
    assert any("(+1 more)" in text for text in warnings)
    assert any("Grid source: PREDICTED qualifying grid." in text for text in details)
    assert any(
        "Data source: Short-stint blend (FP3 + FP2 + FP1) (70% practice data + 30% model)." in text
        for text in details
    )


def test_display_prediction_result_routes_actual_qualifying_classification(patcher):
    calls = _stub_streamlit(patcher)
    routed: list[str] = []

    patcher.setattr(
        rendering,
        "_render_actual_classification",
        lambda _df, caption: routed.append(str(caption)),
    )

    rendering.display_prediction_result(
        result={
            "result_mode": "ACTUAL",
            "classification_note": "Showing ACTUAL qualifying classification from the completed session.",
            "classification_caption": "No grid penalties are applied here.",
            "grid": [{"position": 1, "driver": "RUS", "team": "Mercedes"}],
        },
        prediction_name="Qualifying Result",
        is_race=False,
    )

    assert routed == ["No grid penalties are applied here."]
    assert (
        "success",
        "Showing ACTUAL qualifying classification from the completed session.",
    ) in calls


def test_display_prediction_result_renders_teammate_head_to_head_probabilities(patcher):
    calls = _stub_streamlit(patcher)
    routed: list[str] = []

    patcher.setattr(rendering, "_render_qualifying_result", lambda _df: routed.append("quali"))

    rendering.display_prediction_result(
        result={
            "grid_source": "PREDICTED",
            "data_source": "Testing short-run profile blend (no weekend practice data)",
            "blend_used": False,
            "grid": [{"position": 1, "driver": "VER", "team": "Red Bull Racing"}],
            "teammate_head_to_head": [
                {
                    "team": "Red Bull Racing",
                    "driver_a": "VER",
                    "driver_b": "HAD",
                    "p_driver_a_ahead": 0.803,
                    "n_samples": 3000,
                }
            ],
        },
        prediction_name="Qualifying Prediction",
        is_race=False,
    )

    assert routed == ["quali"]
    expander_labels = [value for kind, value in calls if kind == "expander"]
    assert any("Teammate Matchups" in text for text in expander_labels)
    markdown_blocks = [value for kind, value in calls if kind == "markdown"]
    assert any("How to read:" in text for text in markdown_blocks)
    assert any("VER over HAD" in text for text in markdown_blocks)
    assert any("80.3%" in text for text in markdown_blocks)
