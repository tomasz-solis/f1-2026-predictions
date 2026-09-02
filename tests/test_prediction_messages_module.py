"""Tests for extracted live-prediction message helpers."""

from src.dashboard import prediction_messages


def test_build_runtime_messages_collects_key_runtime_notices():
    messages = prediction_messages.build_runtime_messages(
        selected_season=2026,
        race_name="Chinese Grand Prix",
        is_sprint=True,
        boundary_refresh={
            "latest_elapsed_session": "SQ",
            "refresh_needed": True,
            "reason": "session_boundary_delta",
            "new_sessions": ["SPRINT"],
        },
        practice_update={
            "updated": False,
            "completed_fp_sessions": ["FP1"],
            "retried_events": ["Japanese Grand Prix"],
        },
        prediction_cache_hit=True,
        boundary_fallback={
            "current_boundary_session_name": "SPRINT",
            "warmed_boundary_session_name": "SQ",
        },
        precompute_summary={
            "triggered": True,
            "generated": 2,
            "reused": 1,
            "targets": ["Chinese Grand Prix", "Japanese Grand Prix"],
            "ready_races": ["Chinese Grand Prix"],
            "errors": ["rain run failed"],
        },
        completed_races_count=2,
    )

    texts = [message for _level, message in messages]

    assert any(
        "only 2/3 completed Grand Prix race results are in the model" in text for text in texts
    )
    assert any("Sprint weekend mode active" in text for text in texts)
    assert any("Prediction reused from cache" in text for text in texts)
    assert any(
        "Serving the latest available persisted checkpoint SQ instead" in text for text in texts
    )
    assert any("Practice backlog updates deferred" in text for text in texts)
    assert any(
        "Boundary precompute completed: 2 scenario(s) generated, 1 reused" in text for text in texts
    )
    assert any("Some precompute scenarios failed: rain run failed" in text for text in texts)


def test_build_runtime_messages_suppresses_reset_warning_after_three_races():
    messages = prediction_messages.build_runtime_messages(
        selected_season=2026,
        race_name="Miami Grand Prix",
        is_sprint=False,
        boundary_refresh={"latest_elapsed_session": "FP3"},
        practice_update={"updated": False, "completed_fp_sessions": []},
        prediction_cache_hit=False,
        boundary_fallback={},
        precompute_summary={},
        completed_races_count=3,
    )

    texts = [message for _level, message in messages]

    assert not any("2026 regulation reset" in text for text in texts)
    assert any("Latest datapoint in use" in text for text in texts)


def test_build_runtime_messages_keeps_reset_warning_when_count_is_missing():
    messages = prediction_messages.build_runtime_messages(
        selected_season=2026,
        race_name="Miami Grand Prix",
        is_sprint=False,
        boundary_refresh={},
        practice_update={"updated": False, "completed_fp_sessions": []},
        prediction_cache_hit=False,
        boundary_fallback={},
        precompute_summary={},
    )

    texts = [message for _level, message in messages]

    assert any("completed Grand Prix race evidence is unavailable" in text for text in texts)


def test_render_collapsible_runtime_messages_deduplicates_details():
    banners: list[tuple[str, str, str]] = []
    markdown_calls: list[str] = []

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Streamlit:
        def expander(self, _label: str, expanded: bool = False):
            assert expanded is False
            return _Ctx()

        def markdown(self, value: str):
            markdown_calls.append(value)

    prediction_messages.render_collapsible_runtime_messages(
        [
            ("warning", "Latest checkpoint missing"),
            ("warning", "Latest checkpoint missing"),
            ("info", "Cache reused"),
        ],
        render_notice_banner_fn=lambda message, tone, label, st_module: banners.append(
            (message, tone, label)
        ),
        st_module=_Streamlit(),
    )

    assert banners == [("Latest checkpoint missing (+1 more)", "warning", "Forecast details")]
    assert markdown_calls == [
        "- **Warning:** Latest checkpoint missing",
        "- **Info:** Cache reused",
    ]


def test_pipeline_timing_and_runtime_counter_captions_format_compactly():
    timing_caption = prediction_messages.pipeline_timing_caption(
        {
            "boundary_check": 0.4,
            "weekend_lookup": 1.2,
            "practice_update_check": 2.3,
            "prediction_load": 4.5,
            "total": 8.4,
        }
    )
    counters_caption = prediction_messages.runtime_health_counters_caption(
        {
            "counters": {
                "fastf1_completion_unknown_total": 2,
                "fastf1_downgrade_prevented_total": 0,
                "practice_backlog_retry_total": "3",
                "ignored_metric": 99,
            }
        }
    )

    assert timing_caption == (
        "Pipeline timing: boundary check 0.4s | weekend lookup 1.2s | "
        "practice check 2.3s | prediction load 4.5s | total 8.4s"
    )
    assert counters_caption == (
        "Runtime health counters: fastf1_completion_unknown_total=2 | "
        "practice_backlog_retry_total=3"
    )
