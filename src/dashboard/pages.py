"""Dashboard pages and page-level orchestration."""

import logging
from collections.abc import Callable

import fastf1
import streamlit as st

from src.utils.weekend import is_sprint_weekend

from . import team_comparison as _team_comparison
from .accuracy_view import (
    render_overall_accuracy_metrics,
    render_per_race_breakdown,
    render_saved_predictions_summary,
)
from .cache import get_artifact_versions
from .live_prediction_flow import (
    execute_live_prediction_pipeline_core as _execute_live_prediction_pipeline_core,
)
from .live_prediction_flow import (
    render_prediction_results_core as _render_prediction_results_core,
)
from .live_prediction_flow import (
    save_prediction_if_enabled_core as _save_prediction_if_enabled_core,
)
from .page_content import (
    CONTACT_PAGE_HTML,
    MODEL_INSIGHTS_MARKDOWN,
    QUALIFYING_HYPERPARAMETERS_MARKDOWN,
    RACE_HYPERPARAMETERS_MARKDOWN,
)
from .prediction_flow import run_prediction
from .rendering import display_prediction_result
from .update_flow import auto_update_if_needed, auto_update_practice_characteristics_if_needed

logger = logging.getLogger(__name__)
DEFAULT_SEASON = 2026

# Backwards-compatible exports for tests and existing imports.
_DEFAULT_TEAM_COLOR = _team_comparison._DEFAULT_TEAM_COLOR
_build_team_comparison_dataframe = _team_comparison._build_team_comparison_dataframe
_coerce_unit_metric = _team_comparison._coerce_unit_metric
_collect_profile_names = _team_comparison._collect_profile_names
_default_team_selection = _team_comparison._default_team_selection
_hex_to_rgba = _team_comparison._hex_to_rgba
_load_team_characteristics_payload = _team_comparison._load_team_characteristics_payload
_render_team_comparison_section = _team_comparison._render_team_comparison_section
_resolve_profile_metrics = _team_comparison._resolve_profile_metrics
_team_brand_color = _team_comparison._team_brand_color


def render_team_comparison_page() -> None:
    """Render standalone team comparison tab."""
    st.header("Team Comparison")
    st.markdown(
        "Compare team characteristic fingerprints from testing/practice inputs. "
        "Profile metrics and season-prior baseline are separate signals and can diverge."
    )
    _render_team_comparison_section(year=DEFAULT_SEASON)


def _clear_fastf1_race_cache(year: int, race_name: str) -> None:
    """
    Clear FastF1 cache for a specific race to force fresh data fetch.

    This invalidates all cached session data for the race, including practice sessions,
    qualifying, and race results. The next FastF1 call will fetch fresh data from the API.
    """
    import shutil
    from pathlib import Path

    cache_dirs = [
        Path("data/raw/.fastf1_cache"),
        Path("data/raw/.fastf1_cache_testing"),
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        try:
            # FastF1 cache structure: {cache_dir}/{year}/{race_name}/...
            race_cache_path = cache_dir / str(year) / race_name.replace(" ", "_")
            if race_cache_path.exists():
                shutil.rmtree(race_cache_path)
                logger.info(f"Cleared FastF1 cache for {race_name} {year} at {race_cache_path}")

            # Also try with spaces removed completely
            race_cache_path_alt = cache_dir / str(year) / race_name.replace(" ", "")
            if race_cache_path_alt.exists():
                shutil.rmtree(race_cache_path_alt)
                logger.info(f"Cleared alternate FastF1 cache at {race_cache_path_alt}")

        except Exception as e:
            logger.warning(f"Could not clear FastF1 cache at {cache_dir}: {e}")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_race_options_cached(year: int) -> tuple[list[str], str | None]:
    """Load race options and cache schedule fetches for responsiveness."""
    try:
        schedule = fastf1.get_event_schedule(year)
        race_events = schedule[
            (schedule["EventFormat"].notna())
            & (~schedule["EventName"].str.contains("Testing", case=False, na=False))
        ].copy()

        race_options = []
        for _, event in race_events.iterrows():
            race_name = event["EventName"]
            event_format = str(event["EventFormat"]).lower()
            if "sprint" in event_format:
                race_options.append(f"{race_name} (Sprint)")
            else:
                race_options.append(race_name)

        return race_options, None
    except Exception as exc:
        return (
            [
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
                "Australian Grand Prix",
                "Japanese Grand Prix",
                "Chinese Grand Prix",
                "Miami Grand Prix",
            ],
            str(exc),
        )


def _load_race_options() -> list[str]:
    """Load race options from FastF1 schedule with sprint labels."""
    race_options, error = _load_race_options_cached(DEFAULT_SEASON)
    if error:
        st.error(f"Failed to load {DEFAULT_SEASON} calendar: {error}")
    return race_options


def _save_prediction_if_enabled(
    enable_logging: bool,
    prediction_results: dict,
    is_sprint: bool,
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
) -> None:
    """Persist prediction artifacts for later accuracy tracking."""
    from src.utils.prediction_logger import PredictionLogger
    from src.utils.session_detector import SessionDetector

    _save_prediction_if_enabled_core(
        enable_logging=enable_logging,
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        race_name=race_name,
        weather=weather,
        year=year,
        detector_factory=SessionDetector,
        prediction_logger_factory=PredictionLogger,
        st_module=st,
    )


def _render_prediction_results(prediction_results: dict, is_sprint: bool) -> None:
    """Render prediction sections for sprint and normal weekends."""
    _render_prediction_results_core(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        display_prediction_result_fn=display_prediction_result,
        st_module=st,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _run_prediction_cached(
    race_name: str,
    weather: str,
    artifact_versions_key: tuple[tuple[str, tuple[int, str]], ...],
    is_sprint: bool,
    year: int,
) -> dict:
    """Run prediction with caching for unchanged inputs and artifact versions."""
    artifact_versions = dict(artifact_versions_key)
    return run_prediction(
        race_name,
        weather,
        artifact_versions,
        is_sprint=is_sprint,
        year=year,
    )


def execute_live_prediction_pipeline(
    race_name: str,
    weather: str,
    year: int = DEFAULT_SEASON,
    force_refresh: bool = True,
    use_cached_prediction: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Refresh input data and execute a prediction run.

    Kept separate from Streamlit rendering so tests can assert refresh call order.

    Args:
        race_name: The name of the race
        weather: Weather forecast for the race
        year: Season year
        force_refresh: If True, clears FastF1 cache and forces re-check of session completion
        use_cached_prediction: If True, reuses cached prediction results when inputs are unchanged
        progress_callback: Optional callback for progress updates
    """
    return _execute_live_prediction_pipeline_core(
        race_name=race_name,
        weather=weather,
        year=year,
        force_refresh=force_refresh,
        use_cached_prediction=use_cached_prediction,
        progress_callback=progress_callback,
        clear_fastf1_race_cache_fn=_clear_fastf1_race_cache,
        auto_update_if_needed_fn=auto_update_if_needed,
        is_sprint_weekend_fn=is_sprint_weekend,
        auto_update_practice_characteristics_if_needed_fn=auto_update_practice_characteristics_if_needed,
        clear_resource_cache_fn=st.cache_resource.clear,
        clear_data_cache_fn=st.cache_data.clear,
        get_artifact_versions_fn=get_artifact_versions,
        run_prediction_fn=run_prediction,
        run_prediction_cached_fn=_run_prediction_cached,
    )


def render_live_prediction_page(enable_logging: bool) -> None:
    st.header("Race Weekend Prediction")
    st.markdown(
        "Generate a practice-aware weekend forecast for qualifying and race. "
        "Use forced refresh when new sessions have just completed."
    )

    race_options = _load_race_options()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        race_selection = st.selectbox("Select Grand Prix", race_options)
        race_name = race_selection.replace(" (Sprint)", "")

    with col2:
        weather = st.selectbox("Weather Forecast", ["dry", "rain", "mixed"])

    st.subheader("Run Options")
    options_col, action_col = st.columns((2, 1), gap="large")
    with options_col:
        force_refresh = st.toggle(
            "Force Data Refresh",
            value=False,
            help=(
                "When enabled, clears source caches and recomputes from fresh session data. "
                "When disabled, unchanged inputs can reuse cached predictions."
            ),
        )
    with action_col:
        st.caption(" ")
        generate_prediction = st.button(
            "Generate Prediction",
            type="primary",
            width="stretch",
        )

    use_cached_prediction = not force_refresh
    mode_text = "Mode: Force refresh" if force_refresh else "Mode: Use cached prediction"
    st.markdown(f'<div class="run-options-note">{mode_text}</div>', unsafe_allow_html=True)

    if generate_prediction:
        status_placeholder = st.empty()

        with st.spinner("Running simulation pipeline..."):
            try:

                def update_status(message: str) -> None:
                    status_placeholder.info(f"Loading: {message}")

                pipeline_output = execute_live_prediction_pipeline(
                    race_name=race_name,
                    weather=weather,
                    year=DEFAULT_SEASON,
                    force_refresh=force_refresh,
                    use_cached_prediction=use_cached_prediction,
                    progress_callback=update_status,
                )
                prediction_results = pipeline_output["prediction_results"]
                is_sprint = bool(pipeline_output["is_sprint"])
                practice_update = pipeline_output["practice_update"]
                pipeline_timing = pipeline_output.get("pipeline_timing", {})
                status_placeholder.empty()

                st.warning("2026 regulation reset: predictions are uncertain until races complete.")

                if is_sprint:
                    st.info(
                        "**Sprint Weekend** - System predicts Sprint Qualifying (Friday) → "
                        "Sprint Race (Saturday) → Main Qualifying (Saturday) → Main Race (Sunday). "
                        "Sprint predictions use adjusted chaos modeling "
                        "(30% less variance, grid position +10% importance)."
                    )

                if practice_update.get("updated"):
                    st.success(
                        "Updated car characteristics from completed practice sessions: "
                        f"{', '.join(practice_update['completed_fp_sessions'])} "
                        f"({practice_update['teams_updated']} teams)"
                    )
                elif practice_update.get("completed_fp_sessions"):
                    st.info(
                        "Practice characteristics already up to date for sessions: "
                        f"{', '.join(practice_update['completed_fp_sessions'])}"
                    )

                if pipeline_timing:
                    timing_parts = [
                        f"updates {pipeline_timing.get('race_update_check', 0.0):.1f}s",
                        f"weekend lookup {pipeline_timing.get('weekend_lookup', 0.0):.1f}s",
                        f"practice check {pipeline_timing.get('practice_update_check', 0.0):.1f}s",
                        f"prediction {pipeline_timing.get('prediction_run', 0.0):.1f}s",
                        f"total {pipeline_timing.get('total', 0.0):.1f}s",
                    ]
                    st.caption("Pipeline timing: " + " | ".join(timing_parts))

                _save_prediction_if_enabled(
                    enable_logging=enable_logging,
                    prediction_results=prediction_results,
                    is_sprint=is_sprint,
                    race_name=race_name,
                    weather=weather,
                    year=DEFAULT_SEASON,
                )

                _render_prediction_results(prediction_results, is_sprint)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure data files are generated. Run: "
                    "`python scripts/extract_driver_characteristics.py --years 2023,2024,2025`"
                )


def render_model_insights_page() -> None:
    st.header("Model and Learning Runtime")
    st.markdown(MODEL_INSIGHTS_MARKDOWN)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(QUALIFYING_HYPERPARAMETERS_MARKDOWN)

    with col2:
        st.markdown(RACE_HYPERPARAMETERS_MARKDOWN)


def render_prediction_accuracy_page() -> None:
    st.header("Prediction Accuracy Tracker")

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.prediction_metrics import PredictionMetrics

    logger_inst = PredictionLogger()
    metrics_calc = PredictionMetrics()

    all_predictions = logger_inst.get_all_predictions(DEFAULT_SEASON)

    if not all_predictions:
        st.info(
            "No predictions saved yet. Enable 'Save Predictions for Accuracy Tracking' "
            "in the sidebar and generate predictions after practice sessions."
        )
        return

    st.success(f"Found {len(all_predictions)} saved prediction(s)")

    predictions_with_actuals = [
        prediction
        for prediction in all_predictions
        if prediction.get("actuals")
        and (prediction["actuals"].get("qualifying") or prediction["actuals"].get("race"))
    ]

    if predictions_with_actuals:
        agg_metrics = metrics_calc.aggregate_metrics(predictions_with_actuals)
        render_overall_accuracy_metrics(agg_metrics)
        render_per_race_breakdown(predictions_with_actuals, metrics_calc)
    else:
        st.info(
            "Predictions saved, but no actual results added yet. After each race, "
            "you can update predictions with actual results to calculate accuracy."
        )

    render_saved_predictions_summary(all_predictions)


def render_contact_page() -> None:
    st.header("Contact")
    st.markdown(CONTACT_PAGE_HTML, unsafe_allow_html=True)


def render_about_page() -> None:
    """Backwards-compatible alias for older routes."""
    render_contact_page()


def render_page(page: str, enable_logging: bool) -> None:
    if page in {"Prediction", "Live Prediction"}:
        render_live_prediction_page(enable_logging)
    elif page in {"Model & Learning", "Model Insights"}:
        render_model_insights_page()
    elif page == "Team Comparison":
        render_team_comparison_page()
    elif page == "Prediction Accuracy":
        render_prediction_accuracy_page()
    elif page in {"Contact", "About"}:
        render_contact_page()
    else:
        render_live_prediction_page(enable_logging)
