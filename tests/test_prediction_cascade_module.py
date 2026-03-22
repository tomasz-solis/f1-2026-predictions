"""Tests for the extracted dashboard prediction cascade renderer."""

from src.dashboard import prediction_cascade


def test_render_prediction_results_core_routes_sprint_sections_and_result_titles():
    """Sprint cascade rendering should preserve order and rename completed sections."""

    class _Streamlit:
        """Collect rendered dashboard notices without using Streamlit itself."""

        def __init__(self) -> None:
            self.success_messages: list[str] = []
            self.headers: list[str] = []
            self.info_messages: list[str] = []
            self.markdown_calls = 0

        def success(self, message: str) -> None:
            self.success_messages.append(str(message))

        def header(self, message: str) -> None:
            self.headers.append(str(message))

        def info(self, message: str) -> None:
            self.info_messages.append(str(message))

        def markdown(self, _message: str) -> None:
            self.markdown_calls += 1

    streamlit = _Streamlit()
    rendered_sections: list[str] = []

    prediction_cascade.render_prediction_results_core(
        prediction_results={
            "sprint_quali": {"timing": {"total": 1.2}, "grid": [], "result_mode": "ACTUAL"},
            "sprint_race": {"finish_order": []},
            "main_quali": {"grid": []},
            "main_race": {"finish_order": []},
        },
        is_sprint=True,
        display_prediction_result_fn=lambda _result, title, is_race: rendered_sections.append(
            f"{title}:{'race' if is_race else 'quali'}"
        ),
        st_module=streamlit,
    )

    assert streamlit.success_messages == ["Predictions complete in 1.20s"]
    assert streamlit.headers == ["Sprint Weekend Cascade"]
    assert streamlit.info_messages == [
        "Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race"
    ]
    assert streamlit.markdown_calls == 1
    assert rendered_sections == [
        "Sprint Qualifying Result:quali",
        "Sprint Race Prediction:race",
        "Main Qualifying Prediction:quali",
        "Main Race Prediction:race",
    ]


def test_render_prediction_results_core_prefers_pipeline_timing_for_cache_hits():
    """Cache-hit banner should report the request runtime instead of simulated timing."""

    class _Streamlit:
        """Collect only the success banner for cache-hit assertions."""

        def __init__(self) -> None:
            self.success_messages: list[str] = []

        def success(self, message: str) -> None:
            self.success_messages.append(str(message))

        def header(self, _message: str) -> None:
            return None

        def info(self, _message: str) -> None:
            return None

        def markdown(self, _message: str) -> None:
            return None

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

    assert streamlit.success_messages == ["Prediction loaded from cache in 0.10s"]
