"""Helpers for rendering the top-level prediction cascade in the dashboard."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from html import escape
from typing import Any

from src.dashboard.race_context import attach_starting_grid_context

PredictionResults = dict[str, Any]
CascadeEntry = tuple[str, str, bool, str]

_SPRINT_CASCADE = (
    ("sprint_quali", "Sprint Qualifying Prediction", False, "Sprint Quali"),
    ("sprint_race", "Sprint Race Prediction", True, "Sprint Race"),
    ("main_quali", "Main Qualifying Prediction", False, "Main Quali"),
    ("main_race", "Main Race Prediction", True, "Main Race"),
)
_NORMAL_CASCADE = (
    ("qualifying", "Qualifying Prediction", False, "Qualifying"),
    ("race", "Race Prediction", True, "Race"),
)
_PAIRED_GRID_SECTIONS = {
    "sprint_race": ("sprint_quali", "SQ"),
    "main_race": ("main_quali", "Q"),
    "race": ("qualifying", "Q"),
}


def _section_title(section: Mapping[str, Any], default_title: str) -> str:
    """Swap Prediction for Result once a section becomes actual."""
    result_mode = str(section.get("result_mode", "")).strip().upper()
    if result_mode == "ACTUAL":
        return default_title.replace("Prediction", "Result")
    return default_title


def _section_state(section: Mapping[str, Any]) -> str:
    """Return the compact state label used in the session overview."""
    if str(section.get("result_mode", "")).strip().upper() == "ACTUAL":
        return "Result"
    return "Forecast"


def _section_meta(section: Mapping[str, Any], *, section_name: str, is_race: bool) -> str:
    """Build one short supporting line for the session overview tile."""
    if is_race:
        grid_source = str(section.get("grid_source", "")).strip().upper()
        paired_session = str(section.get("starting_session_name", "")).strip().upper()
        if not paired_session and section_name in _PAIRED_GRID_SECTIONS:
            paired_session = _PAIRED_GRID_SECTIONS[section_name][1]
        source_label = "actual grid" if grid_source == "ACTUAL" else "predicted grid"
        if paired_session:
            return f"Starts from {paired_session} {source_label}."
        return f"Starts from {source_label}."

    data_source = str(section.get("data_source", "")).strip()
    if str(section.get("result_mode", "")).strip().upper() == "ACTUAL":
        return "Completed classification."
    if "checkpoint profile blend" in data_source.lower():
        return "Checkpoint blend input."
    if data_source:
        return data_source if len(data_source) <= 44 else f"{data_source[:41].rstrip()}..."
    return "Grid projection."


def _run_summary_tone(prediction_cache_hit: bool) -> str:
    """Choose a visual tone for the run summary banner."""
    return "info" if prediction_cache_hit else "success"


def _render_run_summary(
    *,
    message: str,
    prediction_cache_hit: bool,
    st_module: Any,
) -> None:
    """Render prediction runtime as a compact dashboard status line."""
    tone = _run_summary_tone(prediction_cache_hit)
    st_module.markdown(
        (
            f'<div class="ts-run-summary ts-run-summary--{escape(tone)}">'
            '<div class="ts-run-summary__label">Run status</div>'
            f'<div class="ts-run-summary__value">{escape(message)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_session_overview_html(
    *,
    cascade: tuple[CascadeEntry, ...],
    prediction_results: PredictionResults,
    is_sprint: bool,
) -> str:
    """Build the weekend session overview shown above the tabs."""
    weekend_label = "Sprint Weekend" if is_sprint else "Race Weekend"
    flow_label = (
        "Sprint Quali -> Sprint Race -> Main Quali -> Main Race"
        if is_sprint
        else "Qualifying -> Race"
    )
    session_tiles: list[str] = []
    for index, (section_name, default_title, is_race, tab_label) in enumerate(cascade, start=1):
        raw_section = prediction_results.get(section_name, {})
        section = raw_section if isinstance(raw_section, Mapping) else {}
        state = _section_state(section)
        state_class = "result" if state == "Result" else "forecast"
        meta = _section_meta(section, section_name=section_name, is_race=is_race)
        title = _section_title(section, default_title).replace(" Prediction", "")
        session_tiles.append(
            f'<article class="ts-session-tile ts-session-tile--{escape(state_class)}">'
            f'<div class="ts-session-tile__index">{index}</div>'
            "<div>"
            f'<div class="ts-session-tile__title">{escape(title or tab_label)}</div>'
            f'<div class="ts-session-tile__state">{escape(state)}</div>'
            f'<div class="ts-session-tile__meta">{escape(meta)}</div>'
            "</div>"
            "</article>"
        )

    return (
        '<section class="ts-session-overview">'
        '<div class="ts-session-overview__head">'
        f'<div class="ts-session-overview__label">{escape(weekend_label)}</div>'
        f'<div class="ts-session-overview__flow">{escape(flow_label)}</div>'
        "</div>"
        f'<div class="ts-session-track">{"".join(session_tiles)}</div>'
        "</section>"
    )


def _section_for_display(
    *,
    prediction_results: PredictionResults,
    section_name: str,
    is_race: bool,
) -> PredictionResults:
    """Return a display payload with paired-grid context added when missing."""
    raw_section = prediction_results[section_name]
    section = deepcopy(raw_section) if isinstance(raw_section, dict) else raw_section
    if not is_race or not isinstance(section, dict):
        return section

    existing_grid = section.get("starting_grid")
    paired = _PAIRED_GRID_SECTIONS.get(section_name)
    if isinstance(existing_grid, list) and existing_grid:
        if paired and not str(section.get("starting_session_name", "")).strip():
            section["starting_session_name"] = paired[1]
        return section

    if not paired:
        return section

    qualifying_section = prediction_results.get(paired[0], {})
    if not isinstance(qualifying_section, Mapping):
        return section
    starting_grid = qualifying_section.get("grid")
    attach_starting_grid_context(section, starting_grid, paired[1])
    return section


def _render_sections(
    *,
    cascade: tuple[CascadeEntry, ...],
    prediction_results: PredictionResults,
    display_prediction_result_fn: Callable[[PredictionResults, str, bool], None],
    st_module: Any,
) -> None:
    """Render each session inside tabs, with a sequential fallback for tests and old Streamlit."""
    tab_labels = [f"{index}. {tab_label}" for index, (*_, tab_label) in enumerate(cascade, start=1)]
    tabs_fn = getattr(st_module, "tabs", None)
    if callable(tabs_fn):
        tabs = tabs_fn(tab_labels)
        if len(tabs) == len(cascade):
            for tab, (section_name, default_title, is_race, _tab_label) in zip(
                tabs,
                cascade,
                strict=False,
            ):
                section = _section_for_display(
                    prediction_results=prediction_results,
                    section_name=section_name,
                    is_race=is_race,
                )
                title = _section_title(section, default_title)
                with tab:
                    display_prediction_result_fn(section, title, is_race)
            return

    for section_name, default_title, is_race, _tab_label in cascade:
        section = _section_for_display(
            prediction_results=prediction_results,
            section_name=section_name,
            is_race=is_race,
        )
        display_prediction_result_fn(section, _section_title(section, default_title), is_race)


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
    cascade: tuple[CascadeEntry, ...]
    runtime_message = _runtime_message(
        prediction_results=prediction_results,
        prediction_cache_hit=prediction_cache_hit,
        pipeline_timing=pipeline_timing,
    )
    _render_run_summary(
        message=runtime_message,
        prediction_cache_hit=prediction_cache_hit,
        st_module=st_module,
    )

    if is_sprint:
        cascade = _SPRINT_CASCADE
    else:
        cascade = _NORMAL_CASCADE

    st_module.markdown(
        _build_session_overview_html(
            cascade=cascade,
            prediction_results=prediction_results,
            is_sprint=is_sprint,
        ),
        unsafe_allow_html=True,
    )
    _render_sections(
        cascade=cascade,
        prediction_results=prediction_results,
        display_prediction_result_fn=display_prediction_result_fn,
        st_module=st_module,
    )
