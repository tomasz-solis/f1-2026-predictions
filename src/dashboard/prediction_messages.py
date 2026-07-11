"""Helpers for live-prediction notices, status text, and observability copy."""

from __future__ import annotations

from typing import Any

from .live_prediction_flow import PrecomputedPredictionUnavailableError

_SESSION_LABELS = {
    "FP1": "Free Practice 1",
    "FP2": "Free Practice 2",
    "FP3": "Free Practice 3",
    "SQ": "Sprint Qualifying",
    "SPRINT": "Sprint Race",
    "Q": "Qualifying",
    "R": "Grand Prix Race",
}
_SESSION_ORDER = {
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "SQ": 4,
    "SPRINT": 5,
    "Q": 6,
    "R": 7,
}
_WATCHED_COUNTERS = [
    "fastf1_completion_unknown_total",
    "fastf1_downgrade_prevented_total",
    "practice_backlog_retry_total",
    "fastf1_circuit_trip_total",
]
_REGULATION_RESET_EVIDENCE_RACES = 3


def _is_competitive_session_status_unavailable(error: Exception) -> bool:
    """Avoid importing the full prediction stack only to recognize this exception."""
    return error.__class__.__name__ == "CompetitiveSessionStatusUnavailableError"


def coerce_completed_races_count(value: Any) -> int | None:
    """Normalize a persisted completed-race count into a non-negative integer."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, count)


def _build_2026_regulation_reset_message(
    completed_races_count: int | None,
) -> tuple[str, str] | None:
    """Return the 2026 reset warning only while race evidence is still thin."""
    completed_count = coerce_completed_races_count(completed_races_count)
    if completed_count is None:
        return (
            "warning",
            "2026 regulation reset: completed Grand Prix race evidence is unavailable here, "
            "so predictions still carry early-season uncertainty.",
        )

    if completed_count >= _REGULATION_RESET_EVIDENCE_RACES:
        return None

    result_label = "result is" if completed_count == 1 else "results are"
    return (
        "warning",
        "2026 regulation reset: only "
        f"{completed_count}/{_REGULATION_RESET_EVIDENCE_RACES} completed Grand Prix race "
        f"{result_label} in the model, so predictions still carry early-season uncertainty.",
    )


def latest_data_status_message(
    race_name: str,
    year: int,
    boundary_refresh: dict[str, object],
    practice_update: dict[str, object],
    *,
    session_labels: dict[str, str] | None = None,
    session_order: dict[str, int] | None = None,
) -> str:
    """Build a short user-facing summary of the freshest session data in use."""
    labels = session_labels or _SESSION_LABELS
    ordering = session_order or _SESSION_ORDER

    def _session_label(session_name: str) -> str:
        normalized = str(session_name).strip().upper()
        return labels.get(normalized, normalized or "Unknown session")

    latest_elapsed = str(boundary_refresh.get("latest_elapsed_session") or "").strip().upper()
    if latest_elapsed:
        latest_label = _session_label(latest_elapsed)
        return (
            "Latest datapoint in use: "
            f"{race_name} {year} - {latest_label} ({latest_elapsed}). "
            "Predictions include all data available through this session."
        )

    completed_fp_raw = practice_update.get("completed_fp_sessions")
    completed_fp = (
        [str(session).strip().upper() for session in completed_fp_raw]
        if isinstance(completed_fp_raw, list)
        else []
    )
    if completed_fp:
        latest_practice = max(
            completed_fp,
            key=lambda session_name: ordering.get(str(session_name).upper(), 0),
        )
        return (
            "Latest datapoint in use: "
            f"{race_name} {year} - {_session_label(latest_practice)} ({latest_practice}). "
            "No completed qualifying/race sessions yet."
        )

    reason = str(boundary_refresh.get("reason", "")).strip().lower()
    if reason == "schedule_unavailable":
        return (
            "Live session schedule is currently unavailable. "
            "Using the latest persisted artifacts and cached race-weekend state."
        )

    return (
        "Latest datapoint in use: pre-weekend baseline/testing only. "
        f"No completed sessions yet for {race_name} {year}."
    )


def build_precompute_horizon_message(
    precompute_filter_meta: dict[str, Any],
    *,
    race_options: list[str],
    selected_race_prediction_available: bool,
) -> tuple[str, str]:
    """Describe which races are currently visible in the warmed prediction horizon."""
    if bool(precompute_filter_meta.get("applied")):
        ready_count = len(precompute_filter_meta.get("ready_races", []))
        expected_targets = precompute_filter_meta.get("expected_targets", [])
        horizon_count = len(expected_targets) if isinstance(expected_targets, list) else ready_count
        anchor_race = str(precompute_filter_meta.get("anchor_race_name", "")).strip()
        anchor_session = str(precompute_filter_meta.get("anchor_session_name", "")).strip()
        if (
            bool(precompute_filter_meta.get("fallback_boundary_active"))
            and anchor_race
            and anchor_session
        ):
            return (
                (
                    f"Forecasts are ready for {ready_count} of the next {horizon_count} race "
                    "weekends. The newest session data is still being prepared, so this shows the "
                    "most recent ready forecast."
                ),
                "success",
            )
        if anchor_race and anchor_session:
            return (
                (
                    f"Forecasts are ready for {ready_count} of the next {horizon_count} race "
                    "weekends. More appear as upcoming weekends are prepared."
                ),
                "success",
            )
        return (
            f"Forecasts are ready for the next {ready_count} race weekends. "
            "More appear as upcoming weekends are prepared.",
            "success",
        )

    if str(precompute_filter_meta.get("stale_reason", "")).strip() == "artifact_hash_mismatch":
        planned_races = precompute_filter_meta.get("planned_races", [])
        visible_count = len(race_options)
        planned_count = len(planned_races) if isinstance(planned_races, list) else visible_count
        return (
            (
                f"Showing forecasts for the next {visible_count} of {planned_count} scheduled "
                "races. The rest are being refreshed for the latest model and will appear shortly."
            ),
            "info",
        )

    if bool(precompute_filter_meta.get("scope_applied")):
        planned_races = precompute_filter_meta.get("planned_races", [])
        visible_count = len(race_options)
        planned_count = len(planned_races) if isinstance(planned_races, list) else visible_count
        if selected_race_prediction_available:
            return (
                (
                    f"Showing forecasts for the next {visible_count} of {planned_count} scheduled "
                    "races. This weekend's forecast is ready; later weekends are still being "
                    "prepared."
                ),
                "info",
            )
        return (
            (
                f"Showing forecasts for the next {visible_count} of {planned_count} scheduled "
                "races. Later weekends are still being prepared."
            ),
            "info",
        )

    return (
        (
            "Forecasts are being prepared for the current race weekend and the ones just after it. "
            "Check back shortly."
        ),
        "info",
    )


def build_runtime_messages(
    *,
    selected_season: int,
    race_name: str,
    is_sprint: bool,
    boundary_refresh: dict[str, Any],
    practice_update: dict[str, Any],
    prediction_cache_hit: bool,
    boundary_fallback: dict[str, Any],
    precompute_summary: dict[str, Any],
    completed_races_count: int | None = None,
    latest_data_status_message_fn: Any = latest_data_status_message,
) -> list[tuple[str, str]]:
    """Build the runtime notice stack shown after a prediction load."""
    runtime_messages: list[tuple[str, str]] = []
    if selected_season == 2026:
        reset_message = _build_2026_regulation_reset_message(completed_races_count)
        if reset_message:
            runtime_messages.append(reset_message)
    else:
        runtime_messages.append(
            (
                "info",
                f"{selected_season} season selected: predictions use currently available "
                "session data and learned artifacts for this season.",
            )
        )

    if is_sprint:
        runtime_messages.append(
            (
                "info",
                "Sprint weekend mode active: Sprint Qualifying -> Sprint Race -> "
                "Main Qualifying -> Main Race cascade.",
            )
        )
    runtime_messages.append(
        (
            "info",
            latest_data_status_message_fn(
                race_name=race_name,
                year=selected_season,
                boundary_refresh=boundary_refresh,
                practice_update=practice_update,
            ),
        )
    )
    if prediction_cache_hit:
        runtime_messages.append(
            (
                "info",
                "Prediction reused from cache (inputs unchanged, no new boundary data).",
            )
        )
    if isinstance(boundary_fallback, dict) and boundary_fallback:
        current_checkpoint = str(boundary_fallback.get("current_boundary_session_name", "")).strip()
        warmed_checkpoint = str(boundary_fallback.get("warmed_boundary_session_name", "")).strip()
        runtime_messages.append(
            (
                "warning",
                "Latest completed checkpoint "
                f"{current_checkpoint or 'current'} is not warmed yet. "
                "Serving the latest available persisted checkpoint "
                f"{warmed_checkpoint or 'PRE'} instead.",
            )
        )
    if practice_update.get("updated"):
        runtime_messages.append(
            (
                "success",
                "Updated car characteristics from completed practice sessions: "
                f"{', '.join(practice_update['completed_fp_sessions'])} "
                f"({practice_update['teams_updated']} teams)",
            )
        )
    elif practice_update.get("completed_fp_sessions"):
        runtime_messages.append(
            (
                "info",
                "Practice characteristics already up to date for sessions: "
                f"{', '.join(practice_update['completed_fp_sessions'])}",
            )
        )

    retried_events = practice_update.get("retried_events", [])
    if retried_events:
        runtime_messages.append(
            (
                "warning",
                "Practice backlog updates deferred due to active processing lock: "
                f"{', '.join(str(event) for event in retried_events)}",
            )
        )

    if boundary_refresh.get("refresh_needed"):
        new_sessions = boundary_refresh.get("new_sessions", [])
        reason = boundary_refresh.get("reason", "session_boundary_delta")
        if new_sessions:
            runtime_messages.append(
                (
                    "info",
                    "A newer checkpoint was detected but is still waiting on warmup "
                    f"({reason}): {', '.join(new_sessions)}",
                )
            )
        else:
            runtime_messages.append(
                (
                    "info",
                    f"A newer checkpoint was detected but is still waiting on warmup ({reason}).",
                )
            )

    if isinstance(precompute_summary, dict) and precompute_summary.get("triggered"):
        generated = int(precompute_summary.get("generated", 0))
        reused = int(precompute_summary.get("reused", 0))
        targets = precompute_summary.get("targets", [])
        ready_races = precompute_summary.get("ready_races", [])
        target_label = ", ".join(str(target) for target in targets) or race_name
        ready_count = len(ready_races) if isinstance(ready_races, list) else 0
        runtime_messages.append(
            (
                "info",
                "Boundary precompute completed: "
                f"{generated} scenario(s) generated, {reused} reused "
                f"for {target_label}. Ready races: {ready_count}.",
            )
        )
        errors = precompute_summary.get("errors", [])
        if isinstance(errors, list) and errors:
            runtime_messages.append(
                (
                    "warning",
                    "Some precompute scenarios failed: "
                    + "; ".join(str(error) for error in errors[:3]),
                )
            )

    return runtime_messages


def render_collapsible_runtime_messages(
    messages: list[tuple[str, str]],
    *,
    render_notice_banner_fn: Any,
    st_module: Any,
) -> None:
    """Render runtime notices compactly to avoid stacked info and warning banners."""
    unique_messages: list[tuple[str, str]] = []
    for level, message in messages:
        normalized_level = str(level).strip().lower()
        normalized_message = str(message).strip()
        if not normalized_message:
            continue
        item = (normalized_level, normalized_message)
        if item not in unique_messages:
            unique_messages.append(item)

    if not unique_messages:
        return

    primary_level, primary_message = unique_messages[0]
    remaining_count = len(unique_messages) - 1
    summary_text = (
        primary_message if remaining_count == 0 else f"{primary_message} (+{remaining_count} more)"
    )
    render_notice_banner_fn(
        summary_text,
        tone=primary_level,
        label="Forecast details",
        st_module=st_module,
    )

    if remaining_count <= 0:
        return

    try:
        expander = st_module.expander("What's behind this forecast", expanded=False)
    except TypeError:
        expander = st_module.expander("What's behind this forecast")

    with expander:
        for level, message in unique_messages:
            prefix = "Info"
            if level == "warning":
                prefix = "Warning"
            elif level == "success":
                prefix = "Success"
            st_module.markdown(f"- **{prefix}:** {message}")


def pipeline_timing_caption(pipeline_timing: dict[str, Any] | None) -> str | None:
    """Format the dashboard pipeline timing summary when timing data is present."""
    if not isinstance(pipeline_timing, dict) or not pipeline_timing:
        return None
    timing_parts = [
        f"boundary check {pipeline_timing.get('boundary_check', 0.0):.1f}s",
        f"weekend lookup {pipeline_timing.get('weekend_lookup', 0.0):.1f}s",
        f"practice check {pipeline_timing.get('practice_update_check', 0.0):.1f}s",
        f"prediction load {pipeline_timing.get('prediction_load', 0.0):.1f}s",
        f"total {pipeline_timing.get('total', 0.0):.1f}s",
    ]
    return "Pipeline timing: " + " | ".join(timing_parts)


def iter_observability_alerts(observability: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize observability alerts into display-ready severity and message pairs."""
    alerts = observability.get("alerts", []) if isinstance(observability, dict) else []
    formatted_alerts: list[tuple[str, str]] = []
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        severity = str(alert.get("severity", "warning")).lower()
        name = str(alert.get("name", "runtime_alert")).strip()
        message = str(alert.get("message", "")).strip()
        if not message:
            continue
        formatted_alerts.append((severity, f"[{name}] {message}"))
    return formatted_alerts


def runtime_health_counters_caption(observability: dict[str, Any]) -> str | None:
    """Summarize the non-zero runtime counters the dashboard watches closely."""
    counters = observability.get("counters", {}) if isinstance(observability, dict) else {}
    if not isinstance(counters, dict):
        return None

    active: list[str] = []
    for key in _WATCHED_COUNTERS:
        if key not in counters:
            continue
        try:
            value = int(counters[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            active.append(f"{key}={value}")
    if not active:
        return None
    return "Runtime health counters: " + " | ".join(active)


def prediction_failure_hint(error: Exception) -> str | None:
    """Return the most relevant user-facing hint for a prediction failure."""
    message = str(error).strip()
    normalized_message = message.lower()

    if _is_competitive_session_status_unavailable(error) or (
        "could not verify completion state" in normalized_message
        and "predicted grid" in normalized_message
    ):
        return (
            "FastF1 has not exposed a reliable completion state for that session yet. "
            "This is a live-data sync problem, not a missing artifact problem. "
            "Retry shortly; if the session is clearly finished, clear that race's FastF1 cache "
            "and rerun."
        )

    artifact_error_markers = (
        "driver characteristics",
        "track characteristics",
        "extract_driver_characteristics.py",
        "could not locate driver characteristics fallback",
    )
    if any(marker in normalized_message for marker in artifact_error_markers):
        return (
            "Make sure data files are generated. Run: "
            "`python scripts/extract_driver_characteristics.py --years 2023,2024,2025,2026`"
            " (prefer a background job or local shell on Render; web-shell runs can hit memory limits)."
        )

    if (
        isinstance(error, PrecomputedPredictionUnavailableError)
        and "could not resolve weekend format" in normalized_message
    ):
        return (
            "The schedule lookup for that race failed, so the dashboard refused to guess "
            "whether it is a sprint or conventional weekend. Verify the race name/year "
            "and refresh the schedule data before retrying."
        )

    if isinstance(error, PrecomputedPredictionUnavailableError):
        return (
            "The dashboard is currently in persisted-prediction mode, so it will not simulate on demand. "
            "Warm the 3-race horizon first with "
            "`python scripts/warmup_precompute.py --year 2026` "
            "(add `--require-db` only when you want DB-backed warmup to be mandatory)."
        )

    return None
