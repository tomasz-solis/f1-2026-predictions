"""Extract and normalize team performance relative to the field."""

import numpy as np


def _normalize_metric_range(
    metric_values: dict[str, float],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    """Normalize one metric so the session leader lands at 1.0."""
    if not metric_values:
        return {}

    best_value = max(metric_values.values()) if higher_is_better else min(metric_values.values())
    worst_value = min(metric_values.values()) if higher_is_better else max(metric_values.values())

    if np.isclose(best_value, worst_value):
        return {team_name: 0.5 for team_name in metric_values}

    normalized_scores: dict[str, float] = {}
    value_range = worst_value - best_value
    for team_name, value in metric_values.items():
        if higher_is_better:
            score = (value - worst_value) / (best_value - worst_value)
        else:
            score = 1.0 - ((value - best_value) / value_range)
        normalized_scores[team_name] = float(np.clip(score, 0.0, 1.0))

    return normalized_scores


def extract_all_teams_performance(
    all_team_data: dict[str, dict], session_name: str = "fp1"
) -> dict[str, dict]:
    """Extract and normalize team performance relative to each other from specified session."""
    raw_metrics = {}

    for team, sessions in all_team_data.items():
        # Find matching session
        session_data = _find_session(sessions, session_name)
        if not session_data:
            continue

        metrics = _extract_raw_metrics(session_data)
        if metrics:
            raw_metrics[team] = metrics

    if not raw_metrics:
        return {}

    return _normalize_relative(raw_metrics)


def _find_session(sessions: dict, session_name: str) -> dict | None:
    """
    Find session that matches name with flexible matching.

    Handles variations like:
    - 'fp1' matches 'bahrain_grand_prix_fp1'
    - 'sprint_quali' matches 'miami_grand_prix_sprint_qualifying'
    - 'fp2' matches 'bahrain_fp2'
    """
    # Direct match
    if session_name in sessions:
        return sessions[session_name]

    # Normalize session name for flexible matching
    session_normalized = session_name.lower().replace("_", "").replace(" ", "")

    # Try exact suffix match first
    suffix = f"_{session_name}"
    for key, data in sessions.items():
        if key.endswith(suffix):
            return data

    # Flexible match: check if normalized session name is in key
    # This handles 'sprint_quali' matching 'sprint_qualifying'
    for key, data in sessions.items():
        key_normalized = key.lower().replace("_", "").replace(" ", "")
        if session_normalized in key_normalized:
            # Make sure it's actually the session type, not just coincidence
            # Check if it ends with the session pattern
            if key_normalized.endswith(session_normalized):
                return data

    return None


def _extract_raw_metrics(session_data: dict) -> dict:
    """Extract raw metrics (no normalization)."""
    metrics = {}

    if "sector_times" in session_data:
        st = session_data["sector_times"]
        if st.get("s1"):
            metrics["s1"] = st["s1"]
        if st.get("s2"):
            metrics["s2"] = st["s2"]
        if st.get("s3"):
            metrics["s3"] = st["s3"]

    if "speed_profile" in session_data:
        sp = session_data["speed_profile"]
        if sp.get("top_speed"):
            metrics["top_speed"] = sp["top_speed"]

    if "braking_profile" in session_data:
        braking = session_data["braking_profile"]
        if braking.get("braking_pct") is not None:
            metrics["braking"] = braking["braking_pct"]

    if "consistency" in session_data:
        cons = session_data["consistency"]
        if cons.get("std_lap_time"):
            metrics["std"] = cons["std_lap_time"]

    return metrics


def _normalize_relative(raw_metrics: dict[str, dict]) -> dict[str, dict]:
    """Normalize team metrics so each session metric spans the full 0-1 field range."""
    metric_aliases = {
        "s1": ("slow_corner_performance", False),
        "s2": ("medium_corner_performance", False),
        "s3": ("fast_corner_performance", False),
        "braking": ("braking_performance", False),
        "top_speed": ("top_speed", True),
        "std": ("consistency", False),
    }

    metric_scores: dict[str, dict[str, float]] = {}
    for raw_metric_name, (_output_metric, higher_is_better) in metric_aliases.items():
        metric_values = {
            team_name: float(metrics[raw_metric_name])
            for team_name, metrics in raw_metrics.items()
            if raw_metric_name in metrics
        }
        metric_scores[raw_metric_name] = _normalize_metric_range(
            metric_values,
            higher_is_better=higher_is_better,
        )

    normalized: dict[str, dict[str, float]] = {}
    for team_name in raw_metrics:
        perf: dict[str, float] = {}

        for raw_metric_name, (output_metric, _higher_is_better) in metric_aliases.items():
            score = metric_scores.get(raw_metric_name, {}).get(team_name)
            if score is None:
                continue
            perf[output_metric] = score

        normalized[team_name] = perf

    return normalized
