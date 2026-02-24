"""Dashboard pages and page-level orchestration."""

import logging
import time
from collections.abc import Callable

import fastf1
import streamlit as st

from src.utils.weekend import is_sprint_weekend

from .cache import get_artifact_versions
from .prediction_flow import run_prediction
from .rendering import display_prediction_result
from .update_flow import auto_update_if_needed, auto_update_practice_characteristics_if_needed

logger = logging.getLogger(__name__)
DEFAULT_SEASON = 2026


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
    if not enable_logging:
        return

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    logger_inst = PredictionLogger()

    latest_session = detector.get_latest_completed_session(year, race_name, is_sprint)

    if latest_session:
        if not logger_inst.has_prediction_for_session(year, race_name, latest_session):
            try:
                if is_sprint:
                    quali_grid = prediction_results["main_quali"]["grid"]
                    race_finish = prediction_results["main_race"]["finish_order"]
                    fp_blend_info = prediction_results.get("main_quali", {}).get(
                        "fp_blend_info", {}
                    )
                else:
                    quali_grid = prediction_results["qualifying"]["grid"]
                    race_finish = prediction_results["race"]["finish_order"]
                    fp_blend_info = prediction_results.get("qualifying", {}).get(
                        "fp_blend_info", {}
                    )

                logger_inst.save_prediction(
                    year=year,
                    race_name=race_name,
                    session_name=latest_session,
                    qualifying_prediction=quali_grid,
                    race_prediction=race_finish,
                    weather=weather,
                    fp_blend_info=fp_blend_info,
                )
                st.info(f"Prediction saved for accuracy tracking (after {latest_session})")
            except Exception as e:
                st.warning(f"Could not save prediction: {e}")
        else:
            st.info(f"Prediction for {latest_session} already saved (max 1 per session)")
    else:
        st.info("No completed sessions yet; prediction not saved (will save after FP1/FP2/FP3/SQ)")


def _render_prediction_results(prediction_results: dict, is_sprint: bool) -> None:
    first_result = list(prediction_results.values())[0]
    timing = first_result.get("timing", {})
    if timing:
        st.success(f"Predictions complete in {timing['total']:.2f}s")
    else:
        st.success("Predictions complete.")

    if is_sprint:
        st.markdown("---")
        st.header("Sprint Weekend Cascade")
        st.info("Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race")

        display_prediction_result(
            prediction_results["sprint_quali"],
            "Sprint Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["sprint_race"],
            "Sprint Race Prediction",
            is_race=True,
        )
        display_prediction_result(
            prediction_results["main_quali"],
            "Main Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["main_race"],
            "Main Race Prediction",
            is_race=True,
        )
    else:
        st.markdown("---")
        st.header("Normal Weekend Cascade")
        st.info("Weekend flow: Qualifying → Race")

        display_prediction_result(
            prediction_results["qualifying"],
            "Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["race"],
            "Race Prediction",
            is_race=True,
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
    pipeline_timing: dict[str, float] = {}
    pipeline_start = time.time()

    def _notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    # Clear FastF1 cache to force fresh data fetch if force_refresh enabled
    if force_refresh:
        _notify("Clearing FastF1 cache for fresh data...")
        clear_start = time.time()
        _clear_fastf1_race_cache(year, race_name)
        pipeline_timing["cache_clear"] = time.time() - clear_start

    update_start = time.time()
    _notify("Checking completed races and model updates...")
    auto_update_if_needed(force_recheck=force_refresh)
    pipeline_timing["race_update_check"] = time.time() - update_start

    weekend_start = time.time()
    _notify("Resolving weekend format...")
    is_sprint = is_sprint_weekend(year, race_name)
    pipeline_timing["weekend_lookup"] = time.time() - weekend_start

    practice_start = time.time()
    _notify("Checking completed practice sessions...")
    practice_update = auto_update_practice_characteristics_if_needed(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        force_recheck=force_refresh,
    )
    pipeline_timing["practice_update_check"] = time.time() - practice_start

    # Refresh cache on the same click after practice updates write new characteristics.
    if practice_update.get("updated") or force_refresh:
        _notify("Refreshing local caches after updates...")
        st.cache_resource.clear()
        st.cache_data.clear()

    # Capture versions after updates so cache invalidation keys include latest writes.
    prediction_start = time.time()
    artifact_versions = get_artifact_versions()
    if use_cached_prediction and not force_refresh:
        _notify("Running qualifying and race simulations (cache enabled)...")
        artifact_versions_key = tuple(sorted(artifact_versions.items()))
        prediction_results = _run_prediction_cached(
            race_name=race_name,
            weather=weather,
            artifact_versions_key=artifact_versions_key,
            is_sprint=is_sprint,
            year=year,
        )
    else:
        _notify("Running qualifying and race simulations...")
        prediction_results = run_prediction(
            race_name,
            weather,
            artifact_versions,
            is_sprint=is_sprint,
            year=year,
        )
    pipeline_timing["prediction_run"] = time.time() - prediction_start
    pipeline_timing["total"] = time.time() - pipeline_start

    return {
        "prediction_results": prediction_results,
        "is_sprint": is_sprint,
        "practice_update": practice_update,
        "pipeline_timing": pipeline_timing,
        "practice_update_error": None,
    }


def render_live_prediction_page(enable_logging: bool) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Race Weekend Prediction")

    race_options = _load_race_options()

    col1, col2 = st.columns(2)

    with col1:
        race_selection = st.selectbox("Select Grand Prix", race_options)
        race_name = race_selection.replace(" (Sprint)", "")

    with col2:
        weather = st.selectbox("Weather Forecast", ["dry", "rain", "mixed"])

    st.markdown("### Run Options")
    run_mode = st.radio(
        "Prediction Mode",
        ["Use Cached Prediction", "Force Data Refresh"],
        index=0,
        help=(
            "Choose exactly one mode. Cached mode reuses unchanged results. "
            "Force refresh clears source caches and recomputes."
        ),
    )
    use_cached_prediction = run_mode == "Use Cached Prediction"
    force_refresh = run_mode == "Force Data Refresh"

    if st.button("Generate Prediction", type="primary"):
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

    st.markdown("</div>", unsafe_allow_html=True)


def render_model_insights_page() -> None:
    st.header("Model and Learning Runtime")

    st.markdown("""
    ### Runtime path

    The dashboard currently runs `Baseline2026Predictor` for both qualifying and race.

    **1. Team strength**
    - Uses baseline (pre-season), testing directionality, and current-season performance.
    - Applies a race-by-race weight schedule that shifts toward current-season data.
    - Uses compound-aware modifiers when validated compound samples are available.

    **2. Qualifying**
    - Uses best available weekend practice data for team pace blending.
    - Falls back to testing short-run profiles when no weekend practice data is available.
    - Applies model-only stabilization for teammate gaps and experience tiers.
    - Runs Monte Carlo simulations and reports median/interval grid outputs.

    **3. Race**
    - Uses predicted or actual qualifying grid depending on session availability.
    - Uses lap-by-lap simulation with team pace, racecraft, strategy, and reliability.
    - Applies track-aware overtaking and pit timing bias (undercut/overcut tendency).
    - Derives podium probability from ranked simulation outcomes for consistency.

    **4. Learning**
    - Saved predictions with actuals update a persistent calibration state.
    - Driver and teammate residual errors are tracked per session type.
    - Learned adjustments are applied in qualifying and race scoring.

    **5. Supporting systems**
    - Auto-updater ingests completed races into characteristics.
    - Testing updater refreshes run-profile and compound characteristics.
    - Bayesian ranking tools remain available for offline analysis workflows.
    """)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Qualifying (active path):**
        - Team/driver score: 70% team + 30% driver
        - Practice blend when available; testing fallback otherwise
        - Model-only teammate gap controls + learned adjustment offsets
        - Output: Monte Carlo median grid + confidence intervals
        """)

    with col2:
        st.markdown("""
        **Race (active path):**
        - Base pace weight: 40% (track-adjusted)
        - Grid influence: dynamic by overtaking difficulty
        - Driver skill term: 20%
        - DNF probability + chaos + strategy + safety car modifiers
        - Podium probability from ranked outcomes with monotonic smoothing
        """)


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
        p
        for p in all_predictions
        if p.get("actuals") and (p["actuals"].get("qualifying") or p["actuals"].get("race"))
    ]

    if predictions_with_actuals:
        st.markdown("---")
        st.subheader("Overall Accuracy")

        agg_metrics = metrics_calc.aggregate_metrics(predictions_with_actuals)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Qualifying Metrics**")
            if "qualifying" in agg_metrics:
                q_metrics = agg_metrics["qualifying"]
                st.metric(
                    "Exact Position Accuracy",
                    f"{q_metrics['exact_accuracy']['mean']:.1f}%",
                    help="% of drivers predicted in exact correct position",
                )
                st.metric(
                    "Mean Position Error (MAE)",
                    f"{q_metrics['mae']['mean']:.2f} positions",
                    help="Average position error",
                )
                st.metric(
                    "Within ±3 Positions",
                    f"{q_metrics['within_3']['mean']:.1f}%",
                    help="% of predictions within 3 positions",
                )
                st.metric(
                    "Correlation",
                    f"{q_metrics['correlation']['mean']:.3f}",
                    help="Spearman correlation (-1 to 1, higher is better)",
                )

        with col2:
            st.markdown("**Race Metrics**")
            if "race" in agg_metrics:
                r_metrics = agg_metrics["race"]
                st.metric(
                    "Exact Position Accuracy",
                    f"{r_metrics['exact_accuracy']['mean']:.1f}%",
                    help="% of drivers predicted in exact correct position",
                )
                st.metric(
                    "Mean Position Error (MAE)",
                    f"{r_metrics['mae']['mean']:.2f} positions",
                    help="Average position error",
                )
                st.metric(
                    "Within ±3 Positions",
                    f"{r_metrics['within_3']['mean']:.1f}%",
                    help="% of predictions within 3 positions",
                )
                st.metric(
                    "Winner Prediction Accuracy",
                    f"{r_metrics['winner_accuracy']['percentage']:.1f}%",
                    help="% of races where winner was correctly predicted",
                )

        st.markdown("---")
        st.subheader("Per-Race Breakdown")

        for pred in predictions_with_actuals:
            metrics = metrics_calc.calculate_all_metrics(pred)
            if metrics:
                race_name = metrics["metadata"]["race_name"]
                session_name = metrics["metadata"]["session_name"]

                with st.expander(f"{race_name} (Predicted after {session_name})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        if "qualifying" in metrics:
                            st.markdown("**Qualifying**")
                            q = metrics["qualifying"]
                            st.write(f"- Exact: {q['exact_accuracy']:.1f}%")
                            st.write(f"- MAE: {q['mae']:.2f} positions")
                            st.write(f"- Within ±1: {q['within_1']:.1f}%")
                            st.write(f"- Correlation: {q['correlation']:.3f}")

                    with col2:
                        if "race" in metrics:
                            st.markdown("**Race**")
                            r = metrics["race"]
                            st.write(f"- Exact: {r['exact_accuracy']:.1f}%")
                            st.write(f"- MAE: {r['mae']:.2f} positions")
                            st.write(f"- Within ±3: {r['within_3']:.1f}%")
                            st.write(
                                f"- Winner: {'Correct' if r['winner_correct'] else 'Incorrect'}"
                            )
                            st.write(
                                f"- Podium: {r['podium']['correct_drivers']}/3 drivers correct"
                            )
    else:
        st.info(
            "Predictions saved, but no actual results added yet. After each race, "
            "you can update predictions with actual results to calculate accuracy."
        )

    st.markdown("---")
    st.subheader("All Saved Predictions")

    for pred in all_predictions:
        metadata = pred["metadata"]
        race_name = metadata["race_name"]
        session_name = metadata["session_name"]
        has_actuals = bool(
            pred.get("actuals")
            and (pred["actuals"].get("qualifying") or pred["actuals"].get("race"))
        )

        status_text = "Results added" if has_actuals else "Awaiting results"
        st.write(f"**{race_name}** (after {session_name}) - {status_text}")


def render_about_page() -> None:
    st.header("About This Project")

    st.markdown("""
    #### Trackside Labs Prediction Engine

    This project focuses on weekend predictions for the 2026 season.

    The core runtime in the app is:
    - Weight-scheduled team strength blending (baseline/testing/current season)
    - Practice-session blending for qualifying when data is available
    - Monte Carlo race simulation using grid, pace, skill, and reliability signals

    **Technology stack**
    - Python
    - FastF1 for session data
    - NumPy and pandas for modeling/data handling
    - Streamlit for the UI

    **Validation and testing**
    - Pytest coverage for key modules
    - Notebooks for schedule/blending validation
    - Session-based prediction tracking in the dashboard

    ---

    **Author:** Tomasz Solis

    **[GitHub](https://github.com/tomasz-solis)**

    **[LinkedIn](https://linkedin.com/in/tomaszsolis)**

    Private repository
    """)

    st.info(
        "Independent project for analysis and experimentation. Not affiliated with any racing series, teams, or governing bodies."
    )


def render_page(page: str, enable_logging: bool) -> None:
    if page == "Live Prediction":
        render_live_prediction_page(enable_logging)
    elif page in {"Model & Learning", "Model Insights"}:
        render_model_insights_page()
    elif page == "Prediction Accuracy":
        render_prediction_accuracy_page()
    elif page == "About":
        render_about_page()
    else:
        render_about_page()
