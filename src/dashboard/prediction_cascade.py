"""Helpers for rendering the top-level prediction cascade in the dashboard."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

PredictionResults = dict[str, Any]

_SPRINT_CASCADE = (
    ("sprint_quali", "Sprint Qualifying Prediction", False),
    ("sprint_race", "Sprint Race Prediction", True),
    ("main_quali", "Main Qualifying Prediction", False),
    ("main_race", "Main Race Prediction", True),
)
_NORMAL_CASCADE = (
    ("qualifying", "Qualifying Prediction", False),
    ("race", "Race Prediction", True),
)


def _section_title(section: Mapping[str, Any], default_title: str) -> str:
    """Swap Prediction for Result once a section becomes actual."""
    result_mode = str(section.get("result_mode", "")).strip().upper()
    if result_mode == "ACTUAL":
        return default_title.replace("Prediction", "Result")
    return default_title


def _runtime_message(
    *,
    prediction_results: PredictionResults,
    prediction_cache_hit: bool,
    pipeline_timing: Mapping[str, Any] | None,
) -> str:
    """Build the summary message shown above rendered prediction sections."""
    first_result: object = next(iter(prediction_results.values()), {})
    timing = first_result.get("timing", {}) if isinstance(first_result, Mapping) else {}
    total_runtime = (
        float(pipeline_timing["total"])
        if isinstance(pipeline_timing, Mapping)
        and isinstance(pipeline_timing.get("total"), int | float)
        else None
    )
    simulated_runtime = (
        float(timing["total"])
        if isinstance(timing, Mapping) and isinstance(timing.get("total"), int | float)
        else None
    )

    if prediction_cache_hit:
        if total_runtime is not None:
            return f"Prediction loaded from cache in {total_runtime:.2f}s"
        return "Prediction loaded from cache."
    if simulated_runtime is not None:
        return f"Predictions complete in {simulated_runtime:.2f}s"
    if total_runtime is not None:
        return f"Predictions complete in {total_runtime:.2f}s"
    return "Predictions complete."


def render_prediction_results_core(
    *,
    prediction_results: PredictionResults,
    is_sprint: bool,
    display_prediction_result_fn: Callable[[PredictionResults, str, bool], None],
    st_module: Any,
    prediction_cache_hit: bool = False,
    pipeline_timing: Mapping[str, Any] | None = None,
) -> None:
    """Render the saved prediction sections for the active weekend format."""
    st_module.success(
        _runtime_message(
            prediction_results=prediction_results,
            prediction_cache_hit=prediction_cache_hit,
            pipeline_timing=pipeline_timing,
        )
    )

    st_module.markdown("---")
    cascade: tuple[tuple[str, str, bool], ...]
    if is_sprint:
        st_module.header("Sprint Weekend Cascade")
        st_module.info(
            "Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race"
        )
        cascade = _SPRINT_CASCADE
    else:
        st_module.header("Normal Weekend Cascade")
        st_module.info("Weekend flow: Qualifying → Race")
        cascade = _NORMAL_CASCADE

    for section_name, default_title, is_race in cascade:
        section = prediction_results[section_name]
        display_prediction_result_fn(
            section,
            _section_title(section, default_title),
            is_race,
        )
