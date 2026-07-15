"""Metric normalization and comparison-table helpers for team radar views."""

from copy import deepcopy
from math import isfinite
from typing import Any

import pandas as pd

from src.utils.team_mapping import canonicalize_team, map_team_to_characteristics

_TEAM_RADAR_METRICS: tuple[tuple[str, str], ...] = (
    ("slow_corner_performance", "Slow Corners"),
    ("medium_corner_performance", "Medium Corners"),
    ("fast_corner_performance", "Fast Corners"),
    ("braking_performance", "Braking"),
    ("top_speed", "Top Speed"),
    ("tire_deg_performance", "Tire Deg"),
)
_TEAM_BRAND_COLORS: dict[str, str] = {
    "RED BULL": "#3671C6",
    "MCLAREN": "#FF8700",
    "FERRARI": "#DC0000",
    "MERCEDES": "#00D2BE",
    "ASTON MARTIN": "#006F62",
    "ALPINE": "#2293D1",
    "HAAS": "#7C8798",
    "RB": "#6692FF",
    "WILLIAMS": "#005AFF",
    "AUDI": "#C4122E",
    "CADILLAC": "#2A4AA0",
}
_DEFAULT_TEAM_COLOR = "#B6BABD"
_DEFAULT_BIG4_CANONICAL: tuple[str, ...] = ("MCLAREN", "MERCEDES", "FERRARI", "RED BULL")
_UNIT_CHART_RANGE_PADDING = 0.02
_TIRE_DEG_SLOPE_DISPLAY_RANGE: tuple[float, float] = (-0.20, 0.40)
_DISPLAY_SCORE_FLOOR = 0.10
_DISPLAY_SCORE_CEILING = 1.00
_DISPLAY_SCORE_RANGE: tuple[float, float] = (_DISPLAY_SCORE_FLOOR, _DISPLAY_SCORE_CEILING)
_TIRE_DEG_SCORE_DISPLAY_RANGE: tuple[float, float] = _DISPLAY_SCORE_RANGE
_TOP_SPEED_KPH_BUFFER = 5.0
_TOP_SPEED_SCORE_DISPLAY_RANGE: tuple[float, float] = _DISPLAY_SCORE_RANGE
_RADAR_AXIS_DISPLAY_MAX = 1.05
_RAW_SECTOR_MIN_PADDING_SECONDS = 0.02
_RAW_PACE_MIN_PADDING_SECONDS = 0.08
_RAW_BRAKING_MIN_PADDING_PERCENT = 1.0
_RAW_PACE_FIELD = "overall_pace_seconds"
_RAW_METRIC_FIELDS: dict[str, tuple[str, bool, float]] = {
    "overall_pace": (_RAW_PACE_FIELD, False, _RAW_PACE_MIN_PADDING_SECONDS),
    "slow_corner_performance": ("slow_corner_seconds", False, _RAW_SECTOR_MIN_PADDING_SECONDS),
    "medium_corner_performance": ("medium_corner_seconds", False, _RAW_SECTOR_MIN_PADDING_SECONDS),
    "fast_corner_performance": ("fast_corner_seconds", False, _RAW_SECTOR_MIN_PADDING_SECONDS),
    "braking_performance": ("braking_pct", False, _RAW_BRAKING_MIN_PADDING_PERCENT),
}


def _coerce_unit_metric(value: Any) -> float | None:
    """Normalize metric values into [0.0, 1.0] when possible."""
    if not isinstance(value, int | float):
        return None
    value_float = float(value)
    if not isfinite(value_float):
        return None
    return max(0.0, min(1.0, value_float))


def _team_brand_color(team_name: str) -> str:
    """Resolve canonical team color; use neutral fallback for unknown names."""
    canonical_id = canonicalize_team(team_name)
    if isinstance(canonical_id, str):
        color = _TEAM_BRAND_COLORS.get(canonical_id)
        if color:
            return color
    return _DEFAULT_TEAM_COLOR


def _unit_chart_axis_range(padding: float = _UNIT_CHART_RANGE_PADDING) -> list[float]:
    """Return a slightly padded 0-1 range so endpoint markers are not clipped."""
    bounded_padding = max(0.0, float(padding))
    return [0.0 - bounded_padding, 1.0 + bounded_padding]


def _normalize_tire_deg_slope_for_display(
    slope: float,
    *,
    slope_range: tuple[float, float] = _TIRE_DEG_SLOPE_DISPLAY_RANGE,
    score_range: tuple[float, float] = _TIRE_DEG_SCORE_DISPLAY_RANGE,
) -> float:
    """
    Convert tire-deg slope into a stable fallback display score.

    The comparison views prefer session-relative tire-deg scaling when a
    snapshot has multiple raw slopes. This helper is the fallback for cases
    where we only have one usable sample and still want a meaningful
    non-session-relative score instead of a neutral midpoint.
    """
    slope_lower, slope_upper = sorted((float(slope_range[0]), float(slope_range[1])))
    bounded_slope = min(max(float(slope), slope_lower), slope_upper)
    if slope_upper <= slope_lower:
        return 0.5

    normalized = 1.0 - ((bounded_slope - slope_lower) / (slope_upper - slope_lower))
    return _project_normalized_value_to_display_score(normalized, score_range=score_range)


def _project_normalized_value_to_display_score(
    normalized_value: float,
    *,
    score_range: tuple[float, float] = _DISPLAY_SCORE_RANGE,
) -> float:
    """Map a normalized 0-1 value into the display score range."""
    score_lower, score_upper = sorted((float(score_range[0]), float(score_range[1])))
    bounded_normalized = max(0.0, min(1.0, float(normalized_value)))
    return float(score_lower + (bounded_normalized * (score_upper - score_lower)))


def _normalize_top_speed_kph_for_display(
    top_speed_kph: float,
    *,
    domain_range: tuple[float, float],
    score_range: tuple[float, float] = _TOP_SPEED_SCORE_DISPLAY_RANGE,
) -> float:
    """
    Convert raw top speed into a session-relative display score.

    Absolute km/h is strongly track-dependent, so the dashboard compares each
    team's trap speed against the current snapshot's observed spread. The score
    range starts above zero so the slowest sampled team reads as "worst in this
    snapshot" rather than "broken".
    """
    return _normalize_metric_for_display(
        top_speed_kph,
        domain_range=domain_range,
        higher_is_better=True,
        score_range=score_range,
    )


def _normalize_metric_for_display(
    metric_value: float,
    *,
    domain_range: tuple[float, float],
    higher_is_better: bool,
    score_range: tuple[float, float] = _DISPLAY_SCORE_RANGE,
) -> float:
    """Convert a raw metric into a display score within the configured range."""
    metric_lower, metric_upper = sorted((float(domain_range[0]), float(domain_range[1])))
    bounded_value = min(max(float(metric_value), metric_lower), metric_upper)
    if metric_upper <= metric_lower:
        return 0.5

    normalized = (bounded_value - metric_lower) / (metric_upper - metric_lower)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return _project_normalized_value_to_display_score(normalized, score_range=score_range)


def _build_raw_metric_display_scale(
    teams_payload: dict[str, Any],
    profile: str,
    *,
    raw_metric_key: str,
    min_padding: float = _RAW_SECTOR_MIN_PADDING_SECONDS,
) -> tuple[float, float] | None:
    """
    Build a session-relative raw-metric range from the current snapshot/profile.

    When multiple raw samples exist, the best and worst sampled teams should
    reach the display endpoints. If every sample is identical, keep a tiny
    centered range so the score stays neutral instead of collapsing.
    """
    raw_values: list[float] = []
    for team_data in teams_payload.values():
        if not isinstance(team_data, dict):
            continue
        metrics_payload = _resolve_profile_metrics(team_data, profile)
        raw_value = metrics_payload.get(raw_metric_key)
        if not isinstance(raw_value, int | float):
            continue
        raw_value_float = float(raw_value)
        if not isfinite(raw_value_float):
            continue
        raw_values.append(raw_value_float)

    if not raw_values:
        return None

    observed_min = min(raw_values)
    observed_max = max(raw_values)
    if observed_max > observed_min:
        return (observed_min, observed_max)

    padding = max(float(min_padding), 1e-9)
    return (observed_min - padding, observed_max + padding)


def _build_top_speed_display_scale(
    teams_payload: dict[str, Any],
    profile: str,
    *,
    buffer_kph: float = _TOP_SPEED_KPH_BUFFER,
) -> tuple[float, float] | None:
    """
    Build a session-relative top-speed range from the current snapshot/profile.

    When multiple samples exist, the slowest and fastest teams map to the score
    endpoints. If every sample is identical, keep a tiny centered range so the
    result stays neutral.
    """
    raw_values: list[float] = []
    for team_data in teams_payload.values():
        if not isinstance(team_data, dict):
            continue
        metrics_payload = _resolve_profile_metrics(team_data, profile)
        raw_value = metrics_payload.get("top_speed_kph")
        if not isinstance(raw_value, int | float):
            continue
        raw_value_float = float(raw_value)
        if not isfinite(raw_value_float):
            continue
        raw_values.append(raw_value_float)

    if not raw_values:
        return None

    observed_min = min(raw_values)
    observed_max = max(raw_values)
    if observed_max > observed_min:
        return (observed_min, observed_max)

    padding = max(float(buffer_kph), 1e-9)
    return (observed_min - padding, observed_max + padding)


def _build_tire_deg_display_scale(
    teams_payload: dict[str, Any],
    profile: str,
) -> tuple[float, float] | None:
    """
    Build a session-relative tire-deg slope range when multiple raw samples exist.

    A single slope sample is not enough to rank teams meaningfully. Historical
    fallbacks also cannot be ranked against current-session slopes as though
    they came from the same conditions. In both cases, the caller uses the
    stable absolute-slope mapping instead.
    """
    raw_slopes: list[float] = []
    for team_data in teams_payload.values():
        if not isinstance(team_data, dict):
            continue
        metrics_payload = _resolve_profile_metrics(team_data, profile)
        if metrics_payload.get("_tire_deg_history_fallback") is True:
            return None
        raw_slope = metrics_payload.get("tire_deg_slope")
        if not isinstance(raw_slope, int | float):
            continue
        raw_slope_float = float(raw_slope)
        if not isfinite(raw_slope_float):
            continue
        raw_slopes.append(raw_slope_float)

    if len(raw_slopes) < 2:
        return None

    observed_min = min(raw_slopes)
    observed_max = max(raw_slopes)
    if observed_max <= observed_min:
        return None
    return (observed_min, observed_max)


def _resolve_top_speed_metric_value(
    metrics_payload: dict[str, Any],
    top_speed_display_scale: tuple[float, float] | None,
) -> float | None:
    """Resolve top speed from raw km/h when available, else stored normalized value."""
    raw_top_speed = metrics_payload.get("top_speed_kph")
    if (
        top_speed_display_scale is not None
        and isinstance(raw_top_speed, int | float)
        and isfinite(float(raw_top_speed))
    ):
        return _normalize_top_speed_kph_for_display(
            float(raw_top_speed),
            domain_range=top_speed_display_scale,
        )
    return _coerce_unit_metric(metrics_payload.get("top_speed"))


def _resolve_raw_metric_value(
    metrics_payload: dict[str, Any],
    *,
    raw_metric_key: str,
    display_scale: tuple[float, float] | None,
    higher_is_better: bool,
    fallback_key: str,
) -> float | None:
    """Resolve a raw snapshot metric or fall back to the stored normalized score."""
    raw_metric_value = metrics_payload.get(raw_metric_key)
    if (
        display_scale is not None
        and isinstance(raw_metric_value, int | float)
        and isfinite(float(raw_metric_value))
    ):
        return _normalize_metric_for_display(
            float(raw_metric_value),
            domain_range=display_scale,
            higher_is_better=higher_is_better,
        )
    return _coerce_unit_metric(metrics_payload.get(fallback_key))


def _resolve_tire_deg_metric_value(
    metrics_payload: dict[str, Any],
    *,
    tire_deg_display_scale: tuple[float, float] | None,
) -> float | None:
    """Resolve the tire-deg value used in the comparison radar/table/history."""
    raw_slope = metrics_payload.get("tire_deg_slope")
    if isinstance(raw_slope, int | float):
        raw_slope_float = float(raw_slope)
        if isfinite(raw_slope_float) and tire_deg_display_scale is not None:
            return _normalize_metric_for_display(
                raw_slope_float,
                domain_range=tire_deg_display_scale,
                higher_is_better=False,
                score_range=_TIRE_DEG_SCORE_DISPLAY_RANGE,
            )
        return _normalize_tire_deg_slope_for_display(raw_slope_float)

    return _coerce_unit_metric(metrics_payload.get("tire_deg_performance"))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB colors to rgba strings for Plotly fills."""
    normalized = str(hex_color).strip().lstrip("#")
    if len(normalized) != 6:
        return f"rgba(124, 135, 152, {alpha})"
    try:
        red = int(normalized[0:2], 16)
        green = int(normalized[2:4], 16)
        blue = int(normalized[4:6], 16)
    except ValueError:
        return f"rgba(124, 135, 152, {alpha})"
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _default_team_selection(team_names: list[str], max_teams: int = 4) -> list[str]:
    """Pick a stable, readable default comparison set from the strongest teams."""
    if not team_names:
        return []

    by_canonical: dict[str, str] = {}
    for team_name in team_names:
        canonical_name = canonicalize_team(team_name)
        canonical_id = canonical_name if isinstance(canonical_name, str) else ""
        by_canonical.setdefault(canonical_id, team_name)

    defaults: list[str] = []
    for canonical_name in _DEFAULT_BIG4_CANONICAL:
        default_team_name = by_canonical.get(canonical_name)
        if default_team_name and default_team_name not in defaults:
            defaults.append(default_team_name)
        if len(defaults) >= max_teams:
            return defaults[:max_teams]

    for team_name in team_names:
        if team_name not in defaults:
            defaults.append(team_name)
        if len(defaults) >= max_teams:
            break

    return defaults[:max_teams]


def _collect_profile_names(teams_payload: dict[str, Any]) -> list[str]:
    """Collect testing profile names while keeping the common order intuitive."""
    names: list[str] = []
    for team_data in teams_payload.values():
        if not isinstance(team_data, dict):
            continue
        profiles = team_data.get("testing_characteristics_profiles")
        if isinstance(profiles, dict):
            for name in profiles:
                normalized = str(name).strip()
                if normalized and normalized not in names:
                    names.append(normalized)

        single_profile = team_data.get("testing_characteristics")
        if isinstance(single_profile, dict):
            run_profile = str(single_profile.get("run_profile", "")).strip()
            if run_profile and run_profile not in names:
                names.append(run_profile)

    preferred_order = ["balanced", "short_run", "long_run"]
    ordered = [name for name in preferred_order if name in names]
    remaining = sorted(name for name in names if name not in preferred_order)
    return ordered + remaining


def _resolve_profile_metrics(team_data: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return the most relevant metrics payload for the selected profile."""
    profiles = team_data.get("testing_characteristics_profiles")
    if isinstance(profiles, dict):
        metrics_payload = profiles.get(profile)
        if isinstance(metrics_payload, dict):
            return metrics_payload

    fallback_payload = team_data.get("testing_characteristics")
    if isinstance(fallback_payload, dict):
        fallback_profile = str(fallback_payload.get("run_profile", "")).strip()
        if fallback_profile == profile:
            return fallback_payload
    return {}


def _is_missing_payload_value(value: Any) -> bool:
    """Return True when the incoming payload slot carries no usable value."""
    if value is None:
        return True
    if isinstance(value, dict):
        return not value
    return False


def _merge_team_payload_values(existing: Any, incoming: Any) -> Any:
    """Merge dict payloads recursively while preferring real values over blanks."""
    if _is_missing_payload_value(existing):
        return deepcopy(incoming)
    if _is_missing_payload_value(incoming):
        return existing
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = deepcopy(existing)
        for key, value in incoming.items():
            merged[key] = _merge_team_payload_values(merged.get(key), value)
        return merged
    return deepcopy(existing)


def _canonicalize_teams_payload_for_comparison(
    teams_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map payload keys to comparison display names and merge alias teams."""
    canonical_payload: dict[str, dict[str, Any]] = {}
    for raw_team_name, raw_team_data in teams_payload.items():
        if not isinstance(raw_team_data, dict):
            continue
        mapped_name = map_team_to_characteristics(str(raw_team_name))
        display_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )
        existing_payload = canonical_payload.get(display_name)
        if existing_payload is None:
            canonical_payload[display_name] = deepcopy(raw_team_data)
            continue
        canonical_payload[display_name] = _merge_team_payload_values(
            existing_payload,
            raw_team_data,
        )
    return canonical_payload


def _has_profile_metrics(team_data: dict[str, Any], profile: str) -> bool:
    """Return True when the selected profile exposes at least one usable metric."""
    metrics_payload = _resolve_profile_metrics(team_data, profile)
    if not isinstance(metrics_payload, dict):
        return False

    metric_keys = (
        "overall_pace",
        "slow_corner_performance",
        "medium_corner_performance",
        "fast_corner_performance",
        "braking_performance",
        "top_speed",
        "tire_deg_performance",
        _RAW_PACE_FIELD,
        "slow_corner_seconds",
        "medium_corner_seconds",
        "fast_corner_seconds",
        "braking_pct",
        "top_speed_kph",
        "tire_deg_slope",
    )
    for metric_key in metric_keys:
        metric_value = metrics_payload.get(metric_key)
        if metric_value is not None:
            return True
    return False


def _strip_raw_display_inputs(metrics_payload: dict[str, Any]) -> None:
    """Remove raw metric fields so a fallback team cannot distort latest-session scales."""
    raw_keys = {
        _RAW_PACE_FIELD,
        "top_speed_kph",
        "tire_deg_slope",
        *(
            raw_metric_key
            for raw_metric_key, _higher_is_better, _min_padding in _RAW_METRIC_FIELDS.values()
        ),
    }
    for key in raw_keys:
        metrics_payload.pop(key, None)


def _uses_same_event_average_fallback(team_payload: dict[str, Any] | None) -> bool:
    """Return True when the latest comparison view uses a weekend-average approximation."""
    if not isinstance(team_payload, dict):
        return False
    return str(team_payload.get("comparison_fallback_source", "")).strip() == "same_event_average"


def _prepare_team_payload_for_comparison_scales(
    team_data: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Sanitize approximated team payloads before building comparison display scales."""
    sanitized_team_data = deepcopy(team_data)
    if not _uses_same_event_average_fallback(sanitized_team_data):
        return sanitized_team_data

    metrics_payload = _resolve_profile_metrics(sanitized_team_data, profile)
    if isinstance(metrics_payload, dict):
        _strip_raw_display_inputs(metrics_payload)
    testing_payload = sanitized_team_data.get("testing_characteristics")
    if isinstance(testing_payload, dict):
        _strip_raw_display_inputs(testing_payload)
    return sanitized_team_data


def _comparison_session_display_columns() -> tuple[str, ...]:
    """Return the session-derived comparison columns shown in the radar/table."""
    return ("Overall Pace", *[label for _, label in _TEAM_RADAR_METRICS])


def _resolve_profile_overall_pace_display_value(
    metrics_payload: dict[str, Any],
    *,
    raw_metric_display_scales: dict[str, tuple[float, float] | None],
) -> float | None:
    """Resolve the comparison-table pace score for one profile payload."""
    return _resolve_raw_metric_value(
        metrics_payload,
        raw_metric_key=_RAW_PACE_FIELD,
        display_scale=raw_metric_display_scales.get("overall_pace"),
        higher_is_better=False,
        fallback_key="overall_pace",
    )


def _resolve_profile_display_metric_value(
    payload_key: str,
    metrics_payload: dict[str, Any],
    *,
    tire_deg_display_scale: tuple[float, float] | None,
    top_speed_display_scale: tuple[float, float] | None,
    raw_metric_display_scales: dict[str, tuple[float, float] | None],
    skip_placeholder_braking: bool = False,
) -> float | None:
    """Resolve one radar metric using the same rules as the comparison table."""
    if payload_key == "tire_deg_performance":
        return _resolve_tire_deg_metric_value(
            metrics_payload,
            tire_deg_display_scale=tire_deg_display_scale,
        )
    if payload_key == "top_speed":
        return _resolve_top_speed_metric_value(metrics_payload, top_speed_display_scale)
    if payload_key == "braking_performance" and skip_placeholder_braking:
        if _uses_placeholder_braking(metrics_payload):
            return None
    if payload_key in _RAW_METRIC_FIELDS:
        raw_metric_key, higher_is_better, _min_padding = _RAW_METRIC_FIELDS[payload_key]
        return _resolve_raw_metric_value(
            metrics_payload,
            raw_metric_key=raw_metric_key,
            display_scale=raw_metric_display_scales.get(payload_key),
            higher_is_better=higher_is_better,
            fallback_key=payload_key,
        )
    return _coerce_unit_metric(metrics_payload.get(payload_key))


def _uses_placeholder_braking(metrics_payload: dict[str, Any]) -> bool:
    """
    Return True when braking looks like a legacy stand-in rather than a real metric.

    Older snapshots often copied `slow_corner_performance` straight into braking.
    Once a raw braking proxy is stored, we trust the snapshot even if the
    normalized values happen to match by coincidence.
    """
    raw_braking = metrics_payload.get("braking_pct")
    if isinstance(raw_braking, int | float) and isfinite(float(raw_braking)):
        return False

    braking_value = _coerce_unit_metric(metrics_payload.get("braking_performance"))
    if braking_value is None:
        return True

    slow_corner_value = _coerce_unit_metric(metrics_payload.get("slow_corner_performance"))
    if slow_corner_value is None:
        return False

    return abs(braking_value - slow_corner_value) < 1e-9


def _resolve_team_comparison_row(
    *,
    team_name: str,
    team_data: dict[str, Any],
    profile: str,
    tire_deg_display_scale: tuple[float, float] | None,
    top_speed_display_scale: tuple[float, float] | None,
    raw_metric_display_scales: dict[str, tuple[float, float] | None],
    skip_placeholder_braking: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    """Build one comparison row and record which columns still lack real data."""
    metrics_payload = _resolve_profile_metrics(team_data, profile)
    row: dict[str, Any] = {"Team": team_name}
    missing_columns: set[str] = set()

    overall_perf = _coerce_unit_metric(team_data.get("overall_performance"))
    row["Overall Performance"] = overall_perf if overall_perf is not None else 0.5
    if overall_perf is None:
        missing_columns.add("Overall Performance")

    overall_pace = _resolve_profile_overall_pace_display_value(
        metrics_payload,
        raw_metric_display_scales=raw_metric_display_scales,
    )
    row["Overall Pace"] = overall_pace if overall_pace is not None else 0.5
    if overall_pace is None:
        missing_columns.add("Overall Pace")

    for payload_key, label in _TEAM_RADAR_METRICS:
        metric_value = _resolve_profile_display_metric_value(
            payload_key,
            metrics_payload,
            tire_deg_display_scale=tire_deg_display_scale,
            top_speed_display_scale=top_speed_display_scale,
            raw_metric_display_scales=raw_metric_display_scales,
            skip_placeholder_braking=skip_placeholder_braking,
        )
        row[label] = metric_value if metric_value is not None else 0.5
        if metric_value is None:
            missing_columns.add(label)

    radar_values = [float(row[label]) for _, label in _TEAM_RADAR_METRICS]
    radar_composite = float(sum(radar_values) / len(radar_values))
    row["Radar Composite"] = radar_composite
    row["Radar Minus Prior"] = radar_composite - float(row["Overall Performance"])
    return row, missing_columns


def _build_team_comparison_missing_column_map(
    *,
    teams_payload: dict[str, Any],
    selected_teams: list[str],
    profile: str,
) -> dict[str, set[str]]:
    """Return unresolved comparison columns after trusted-metric checks."""
    scale_payload = {
        team_name: _prepare_team_payload_for_comparison_scales(team_data, profile)
        for team_name, team_data in teams_payload.items()
        if isinstance(team_data, dict)
    }
    tire_deg_display_scale = _build_tire_deg_display_scale(scale_payload, profile)
    top_speed_display_scale = _build_top_speed_display_scale(scale_payload, profile)
    raw_metric_display_scales = {
        metric_key: _build_raw_metric_display_scale(
            scale_payload,
            profile,
            raw_metric_key=raw_metric_key,
            min_padding=min_padding,
        )
        for metric_key, (
            raw_metric_key,
            _higher_is_better,
            min_padding,
        ) in _RAW_METRIC_FIELDS.items()
    }

    missing_column_map: dict[str, set[str]] = {}
    for team_name in selected_teams:
        team_data = teams_payload.get(team_name)
        if not isinstance(team_data, dict):
            continue
        _row, missing_columns = _resolve_team_comparison_row(
            team_name=team_name,
            team_data=team_data,
            profile=profile,
            tire_deg_display_scale=tire_deg_display_scale,
            top_speed_display_scale=top_speed_display_scale,
            raw_metric_display_scales=raw_metric_display_scales,
            skip_placeholder_braking=True,
        )
        if missing_columns:
            missing_column_map[team_name] = missing_columns

    return missing_column_map


def _build_team_comparison_dataframe(
    teams_payload: dict[str, Any],
    selected_teams: list[str],
    profile: str,
) -> tuple[pd.DataFrame, int]:
    """Build team comparison frame and return count of neutral fallbacks used."""
    rows: list[dict[str, Any]] = []
    neutral_fallback_count = 0
    scale_payload = {
        team_name: _prepare_team_payload_for_comparison_scales(team_data, profile)
        for team_name, team_data in teams_payload.items()
        if isinstance(team_data, dict)
    }
    tire_deg_display_scale = _build_tire_deg_display_scale(scale_payload, profile)
    top_speed_display_scale = _build_top_speed_display_scale(scale_payload, profile)
    raw_metric_display_scales = {
        metric_key: _build_raw_metric_display_scale(
            scale_payload,
            profile,
            raw_metric_key=raw_metric_key,
            min_padding=min_padding,
        )
        for metric_key, (
            raw_metric_key,
            _higher_is_better,
            min_padding,
        ) in _RAW_METRIC_FIELDS.items()
    }

    for team_name in selected_teams:
        team_data = teams_payload.get(team_name)
        if not isinstance(team_data, dict):
            continue

        row, missing_columns = _resolve_team_comparison_row(
            team_name=team_name,
            team_data=team_data,
            profile=profile,
            tire_deg_display_scale=tire_deg_display_scale,
            top_speed_display_scale=top_speed_display_scale,
            raw_metric_display_scales=raw_metric_display_scales,
        )
        neutral_fallback_count += len(missing_columns)
        rows.append(row)

    if not rows:
        return pd.DataFrame(), neutral_fallback_count

    rows.sort(key=lambda row: float(row.get("Overall Pace", 0.0)), reverse=True)
    frame = pd.DataFrame(rows)
    return frame, neutral_fallback_count
