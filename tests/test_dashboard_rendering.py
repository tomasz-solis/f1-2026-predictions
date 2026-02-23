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
    patcher.setattr(rendering.st, "expander", lambda _label: _Ctx())

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


def test_render_qualifying_result_splits_grid_columns(patcher):
    calls = _stub_streamlit(patcher)
    df = pd.DataFrame(
        [{"position": idx, "driver": f"D{idx:02d}", "team": "Team"} for idx in range(1, 23)]
    )

    rendering._render_qualifying_result(df)

    markdown_blocks = [value for kind, value in calls if kind == "markdown"]
    assert any("P1-10" in block for block in markdown_blocks)
    assert any("P11-15" in block for block in markdown_blocks)
    assert any("P16-22" in block for block in markdown_blocks)


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
    assert ("info", "Using PREDICTED grid") in calls
    assert (
        "success",
        "Using Short-stint blend (FP3 + FP2 + FP1) (70% practice data + 30% model)",
    ) in calls
