"""Tests for the extracted dashboard prediction cascade renderer."""

from src.dashboard import prediction_cascade


class _Ctx:
    """Small context manager used to stand in for a Streamlit container."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_render_prediction_results_core_routes_sprint_sections_and_result_titles():
    """Sprint rendering should use session tabs and enrich race sections with paired grids."""

    class _Streamlit:
        """Collect rendered dashboard markup without using Streamlit itself."""

        def __init__(self) -> None:
            self.markdown_messages: list[str] = []
            self.tab_labels: list[list[str]] = []

        def markdown(self, message: str, **_kwargs) -> None:
            self.markdown_messages.append(str(message))

        def tabs(self, labels: list[str]) -> list[_Ctx]:
            self.tab_labels.append(list(labels))
            return [_Ctx() for _label in labels]

    streamlit = _Streamlit()
    rendered_sections: list[tuple[dict, str, bool]] = []

    prediction_cascade.render_prediction_results_core(
        prediction_results={
            "sprint_quali": {
                "timing": {"total": 1.2},
                "grid": [{"position": 1, "driver": "RUS", "team": "Mercedes"}],
                "result_mode": "ACTUAL",
            },
            "sprint_race": {"finish_order": []},
            "main_quali": {"grid": [{"position": 1, "driver": "LEC", "team": "Ferrari"}]},
            "main_race": {"finish_order": []},
        },
        is_sprint=True,
        display_prediction_result_fn=lambda result, title, is_race: rendered_sections.append(
            (result, title, is_race)
        ),
        st_module=streamlit,
    )

    assert any(
        "Predictions complete in 1.20s" in message for message in streamlit.markdown_messages
    )
    assert any("Sprint Weekend" in message for message in streamlit.markdown_messages)
    assert streamlit.tab_labels == [
        ["1. Sprint Quali", "2. Sprint Race", "3. Main Quali", "4. Main Race"]
    ]
    assert [
        f"{title}:{'race' if is_race else 'quali'}" for _result, title, is_race in rendered_sections
    ] == [
        "Sprint Qualifying Result:quali",
        "Sprint Race Prediction:race",
        "Main Qualifying Prediction:quali",
        "Main Race Prediction:race",
    ]
    assert rendered_sections[1][0]["starting_grid"][0]["driver"] == "RUS"
    assert rendered_sections[1][0]["starting_session_name"] == "SQ"
    assert rendered_sections[3][0]["starting_grid"][0]["driver"] == "LEC"
    assert rendered_sections[3][0]["starting_session_name"] == "Q"


def test_render_prediction_results_core_prefers_pipeline_timing_for_cache_hits():
    """Cache-hit banner should report the request runtime instead of simulated timing."""

    class _Streamlit:
        """Collect only the rendered markup for cache-hit assertions."""

        def __init__(self) -> None:
            self.markdown_messages: list[str] = []

        def markdown(self, message: str, **_kwargs) -> None:
            self.markdown_messages.append(str(message))

        def tabs(self, labels: list[str]) -> list[_Ctx]:
            return [_Ctx() for _label in labels]

    streamlit = _Streamlit()

    prediction_cascade.render_prediction_results_core(
        prediction_results={
            "qualifying": {"timing": {"total": 12.65}, "grid": []},
            "race": {"finish_order": []},
        },
        is_sprint=False,
        display_prediction_result_fn=lambda *_args, **_kwargs: None,
        st_module=streamlit,
        prediction_cache_hit=True,
        pipeline_timing={"total": 0.1},
    )

    assert any(
        "Prediction loaded from cache in 0.10s" in message
        for message in streamlit.markdown_messages
    )
