"""Sidecar-only long-run practice evidence for race challengers.

The champion predictor stores blended compound characteristics on the active car
payload.  This module intentionally does not mutate that payload.  It extracts a
versioned research artifact that a race challenger may opt into explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "race_practice_evidence_v2"
DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})
MIN_R0_COMPOUND_LAPS = 16
MIN_R0_COMPOUND_STINTS = 2
MIN_R0_TEAM_COVERAGE = 0.50


@dataclass(frozen=True)
class RacePracticeEvidenceConfig:
    """Deterministic extraction controls for comparable long runs."""

    min_long_run_laps: int = 8
    track_evolution_window_minutes: float = 20.0
    outlier_mad_scale: float = 4.0
    fuel_correction_s_per_lap: float = 0.045
    shrinkage_laps: float = 16.0
    reference_tyre_age: float = 5.0


def build_race_practice_evidence(
    sessions: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    year: int,
    event_name: str,
    checkpoint: str,
    weather: str = "dry",
    config: RacePracticeEvidenceConfig | None = None,
    prior_deg_by_team: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Build a serializable, non-mutating long-run evidence sidecar.

    ``prior_deg_by_team`` is optional and is expected to come from a versioned
    historical/track-class artifact.  No universal compound offset is invented
    when it is absent.
    """

    cfg = config or RacePracticeEvidenceConfig()
    normalized_weather = str(weather).strip().lower()
    session_map = _coerce_sessions(sessions)
    exclusions: dict[str, int] = {}
    fallback_reasons: list[str] = []

    if normalized_weather != "dry":
        return _empty_payload(
            year=year,
            event_name=event_name,
            checkpoint=checkpoint,
            sessions_used=list(session_map),
            weather=normalized_weather,
            exclusions=exclusions,
            fallback_reasons=["dry_only_candidate"],
        )

    cleaned_frames: list[pd.DataFrame] = []
    for session_name, laps in session_map.items():
        cleaned = _clean_laps(laps, session_name=session_name, exclusions=exclusions)
        if not cleaned.empty:
            cleaned_frames.append(cleaned)

    if not cleaned_frames:
        return _empty_payload(
            year=year,
            event_name=event_name,
            checkpoint=checkpoint,
            sessions_used=list(session_map),
            weather=normalized_weather,
            exclusions=exclusions,
            fallback_reasons=["no_reliable_dry_laps"],
        )

    clean = pd.concat(cleaned_frames, ignore_index=True)
    clean = _attach_track_evolution(clean, cfg=cfg)
    stint_rows = _extract_long_run_stints(clean, cfg=cfg, exclusions=exclusions)
    if not stint_rows:
        fallback_reasons.append("no_comparable_long_runs")

    drivers = _aggregate_entities(
        stint_rows,
        entity_key="driver",
        cfg=cfg,
        prior_deg_by_team=prior_deg_by_team,
    )
    teams = _aggregate_entities(
        stint_rows,
        entity_key="team",
        cfg=cfg,
        prior_deg_by_team=prior_deg_by_team,
    )
    _attach_compound_pace_performance(teams, stint_rows)

    usable_laps = int(sum(int(row["n_laps"]) for row in stint_rows))
    return {
        "artifact_type": "race_practice_evidence",
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "year": int(year),
        "event_name": str(event_name),
        "checkpoint": str(checkpoint).strip().upper(),
        "sessions_used": list(session_map),
        "weather": normalized_weather,
        "teams": teams,
        "drivers": drivers,
        "diagnostics": {
            "input_laps": int(sum(len(frame) for frame in session_map.values())),
            "clean_laps": int(len(clean)),
            "usable_long_run_laps": usable_laps,
            "long_run_stints": int(len(stint_rows)),
            "excluded_laps_by_reason": dict(sorted(exclusions.items())),
            "fallback_reasons": fallback_reasons,
        },
    }


def apply_race_practice_evidence(
    driver_info_map: dict[str, dict[str, Any]],
    evidence: Mapping[str, Any] | None,
    *,
    strength_blend_cap: float = 0.30,
) -> dict[str, dict[str, Any]]:
    """Apply an R0 sidecar to prepared race inputs without mutating team artifacts."""

    if not isinstance(evidence, Mapping) or evidence.get("weather") != "dry":
        return driver_info_map
    teams = evidence.get("teams")
    if not isinstance(teams, Mapping):
        return driver_info_map

    for info in driver_info_map.values():
        team_payload = teams.get(str(info.get("team", "")))
        if not isinstance(team_payload, Mapping):
            continue
        compounds = team_payload.get("compounds")
        if not isinstance(compounds, Mapping):
            continue

        strengths = dict(info.get("team_strength_by_compound") or {})
        degradation = dict(info.get("tire_deg_by_compound") or {})
        applied: dict[str, dict[str, float]] = {}
        for compound, raw_metrics in compounds.items():
            if not isinstance(raw_metrics, Mapping):
                continue
            if not _robust_compound_evidence(raw_metrics):
                continue
            n_laps = _finite_float(raw_metrics.get("n_laps")) or 0.0
            evidence_weight = float(np.clip(n_laps / (n_laps + 16.0), 0.0, 1.0))
            pace_performance = _finite_float(raw_metrics.get("pace_performance"))
            if (
                pace_performance is not None
                and compound in strengths
                and _robust_matched_pace_evidence(raw_metrics)
            ):
                blend = min(float(strength_blend_cap), strength_blend_cap * evidence_weight)
                strengths[str(compound)] = float(
                    np.clip(
                        ((1.0 - blend) * float(strengths[str(compound)]))
                        + (blend * pace_performance),
                        0.0,
                        1.0,
                    )
                )
            deg = _finite_float(raw_metrics.get("tire_deg_slope_s_per_lap"))
            if deg is not None:
                current = _finite_float(degradation.get(str(compound)))
                degradation[str(compound)] = (
                    deg
                    if current is None
                    else float(((1.0 - evidence_weight) * current) + (evidence_weight * deg))
                )
            applied[str(compound)] = {
                "evidence_weight": round(evidence_weight, 6),
                "n_laps": float(n_laps),
            }

        info["team_strength_by_compound"] = strengths
        info["tire_deg_by_compound"] = degradation
        if applied:
            info["race_practice_evidence_applied"] = applied
    return driver_info_map


def summarize_race_practice_coverage(
    driver_info_map: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure whether R0 has robust, matching evidence across the field.

    One isolated long run must not reshape a 22-car race.  Coverage is counted
    only when a team has at least one matching compound backed by two qualifying
    long-run stints and sixteen clean laps in total.
    """

    expected_teams = {
        str(info.get("team", "")).strip()
        for info in driver_info_map.values()
        if str(info.get("team", "")).strip()
    }
    raw_teams = evidence.get("teams")
    teams = raw_teams if isinstance(raw_teams, Mapping) else {}
    eligible_teams: set[str] = set()
    for team in expected_teams:
        payload = teams.get(team)
        compounds = payload.get("compounds") if isinstance(payload, Mapping) else None
        if not isinstance(compounds, Mapping):
            continue
        supported_compounds = {
            str(compound)
            for info in driver_info_map.values()
            if str(info.get("team", "")).strip() == team
            for compound in (info.get("team_strength_by_compound") or {})
        }
        if any(
            str(compound) in supported_compounds
            and isinstance(metrics, Mapping)
            and _robust_compound_evidence(metrics)
            for compound, metrics in compounds.items()
        ):
            eligible_teams.add(team)

    required_teams = max(1, int(np.ceil(len(expected_teams) * MIN_R0_TEAM_COVERAGE)))
    return {
        "field_teams": len(expected_teams),
        "eligible_teams": len(eligible_teams),
        "required_eligible_teams": required_teams,
        "team_coverage": (len(eligible_teams) / len(expected_teams) if expected_teams else 0.0),
        "eligible": bool(expected_teams) and len(eligible_teams) >= required_teams,
    }


def _robust_compound_evidence(metrics: Mapping[str, Any]) -> bool:
    n_laps = _finite_float(metrics.get("n_laps")) or 0.0
    n_stints = _finite_float(metrics.get("n_stints")) or 0.0
    return n_laps >= MIN_R0_COMPOUND_LAPS and n_stints >= MIN_R0_COMPOUND_STINTS


def _robust_matched_pace_evidence(metrics: Mapping[str, Any]) -> bool:
    matched_laps = _finite_float(metrics.get("matched_pace_laps")) or 0.0
    matched_stints = _finite_float(metrics.get("matched_pace_stints")) or 0.0
    matched_buckets = _finite_float(metrics.get("matched_pace_buckets")) or 0.0
    return (
        metrics.get("pace_comparison_status") == "matched"
        and matched_laps >= MIN_R0_COMPOUND_LAPS
        and matched_stints >= MIN_R0_COMPOUND_STINTS
        and matched_buckets >= 1.0
    )


def _coerce_sessions(
    sessions: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if isinstance(sessions, pd.DataFrame):
        return {"UNKNOWN": sessions.copy()}
    return {
        str(name).strip().upper(): frame.copy()
        for name, frame in sessions.items()
        if isinstance(frame, pd.DataFrame)
    }


def _empty_payload(
    *,
    year: int,
    event_name: str,
    checkpoint: str,
    sessions_used: list[str],
    weather: str,
    exclusions: Mapping[str, int],
    fallback_reasons: list[str],
) -> dict[str, Any]:
    return {
        "artifact_type": "race_practice_evidence",
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "year": int(year),
        "event_name": str(event_name),
        "checkpoint": str(checkpoint).strip().upper(),
        "sessions_used": sessions_used,
        "weather": weather,
        "teams": {},
        "drivers": {},
        "diagnostics": {
            "input_laps": 0,
            "clean_laps": 0,
            "usable_long_run_laps": 0,
            "long_run_stints": 0,
            "excluded_laps_by_reason": dict(sorted(exclusions.items())),
            "fallback_reasons": fallback_reasons,
        },
    }


def _clean_laps(
    laps: pd.DataFrame,
    *,
    session_name: str,
    exclusions: dict[str, int],
) -> pd.DataFrame:
    required = {"Driver", "Team", "LapTime", "Compound"}
    if laps.empty or not required.issubset(laps.columns):
        _count(exclusions, "missing_required_columns", len(laps))
        return pd.DataFrame()

    frame = laps.copy()
    frame["lap_time_s"] = _seconds(frame["LapTime"])
    frame["Compound"] = frame["Compound"].astype(str).str.upper().str.strip()
    valid = frame["lap_time_s"].notna() & frame["Compound"].isin(DRY_COMPOUNDS)
    _record_mask_exclusions(valid, exclusions, "missing_time_or_dry_compound")

    dry_status = pd.Series(True, index=frame.index)
    has_lap_weather = False
    if "Rainfall" in frame.columns:
        has_lap_weather = True
        rainfall = frame["Rainfall"]
        dry_status &= rainfall.notna() & ~rainfall.map(_truthy)
    for column in ("weather_bucket", "WeatherBucket"):
        if column not in frame.columns:
            continue
        has_lap_weather = True
        weather_bucket = frame[column]
        dry_status &= weather_bucket.notna() & weather_bucket.astype(
            str
        ).str.strip().str.lower().eq("dry")
    if has_lap_weather:
        _record_mask_exclusions(dry_status, exclusions, "wet_or_rainfall")
        valid &= dry_status

    if "IsAccurate" in frame.columns:
        accurate = frame["IsAccurate"].fillna(False).astype(bool)
        _record_mask_exclusions(accurate, exclusions, "inaccurate")
        valid &= accurate
    if "Deleted" in frame.columns:
        deleted = frame["Deleted"].notna() & ~frame["Deleted"].astype(str).str.lower().isin(
            {"false", "0", "none", "nan"}
        )
        _record_mask_exclusions(~deleted, exclusions, "deleted")
        valid &= ~deleted
    for column in ("PitInTime", "PitOutTime"):
        if column in frame.columns:
            clean_pit = frame[column].isna()
            _record_mask_exclusions(clean_pit, exclusions, column.lower())
            valid &= clean_pit
    if "TrackStatus" in frame.columns:
        status = frame["TrackStatus"]
        green = status.isna() | status.astype(str).map(
            lambda value: bool(value) and set(value) <= {"1"}
        )
        _record_mask_exclusions(green, exclusions, "non_green")
        valid &= green

    frame = frame[valid].copy()
    if frame.empty:
        return frame
    frame["session_name"] = session_name
    frame["LapNumber"] = pd.to_numeric(
        frame.get("LapNumber", pd.Series(np.arange(len(frame)), index=frame.index)),
        errors="coerce",
    )
    frame["TyreLife"] = pd.to_numeric(
        frame.get("TyreLife", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    if "Stint" not in frame.columns:
        frame["Stint"] = frame.groupby("Driver", dropna=False)["Compound"].transform(
            lambda values: values.ne(values.shift()).cumsum()
        )
    frame["Stint"] = frame["Stint"].fillna(-1)
    frame["elapsed_s"] = _elapsed_seconds(frame)
    return frame


def _attach_track_evolution(
    clean: pd.DataFrame,
    *,
    cfg: RacePracticeEvidenceConfig,
) -> pd.DataFrame:
    frame = clean.copy()
    group_median = frame.groupby(["session_name", "Driver", "Compound"], dropna=False)[
        "lap_time_s"
    ].transform("median")
    frame["pace_residual_s"] = frame["lap_time_s"] - group_median
    width_s = max(60.0, float(cfg.track_evolution_window_minutes) * 60.0)
    frame["evolution_bin"] = np.floor(frame["elapsed_s"] / width_s).astype(int)
    evolution = frame.groupby(["session_name", "evolution_bin"], dropna=False)[
        "pace_residual_s"
    ].transform("median")
    frame["track_evolution_s"] = evolution.fillna(0.0)
    frame["adjusted_lap_s"] = frame["lap_time_s"] - frame["track_evolution_s"]
    return frame


def _extract_long_run_stints(
    clean: pd.DataFrame,
    *,
    cfg: RacePracticeEvidenceConfig,
    exclusions: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["session_name", "Driver", "Team", "Stint", "Compound"]
    for key, stint in clean.groupby(keys, dropna=False):
        stint = stint.sort_values(["LapNumber", "elapsed_s"])
        for segment in _consecutive_segments(stint):
            segment = _remove_robust_outliers(segment, cfg=cfg)
            if len(segment) < cfg.min_long_run_laps:
                _count(exclusions, "short_or_interrupted_run", len(segment))
                continue
            tyre_age = segment["TyreLife"].copy()
            if tyre_age.notna().sum() < 3:
                tyre_age = pd.Series(np.arange(1, len(segment) + 1), index=segment.index)
                tyre_age_source = "stint_index"
            else:
                tyre_age = tyre_age.interpolate(limit_direction="both")
                tyre_age_source = "reported"
            x = tyre_age.to_numpy(dtype=float)
            y = segment["adjusted_lap_s"].to_numpy(dtype=float)
            raw_slope, intercept = np.polyfit(x, y, 1)
            corrected_slope = float(raw_slope + cfg.fuel_correction_s_per_lap)
            if not -0.10 <= corrected_slope <= 0.50:
                _count(exclusions, "implausible_degradation", len(segment))
                continue
            reference_pace = float(intercept + (corrected_slope * cfg.reference_tyre_age))
            fitted = intercept + (raw_slope * x)
            consistency = _mad(y - fitted)
            tyre_age_window = (round(float(np.min(x)), 3), round(float(np.max(x)), 3))
            evolution_window = (
                int(segment["evolution_bin"].min()),
                int(segment["evolution_bin"].max()),
            )
            rows.append(
                {
                    "session": str(key[0]),
                    "driver": str(key[1]),
                    "team": str(key[2]),
                    "stint": str(key[3]),
                    "compound": str(key[4]),
                    "n_laps": int(len(segment)),
                    "reference_pace_s": reference_pace,
                    "tire_deg_slope_s_per_lap": corrected_slope,
                    "consistency_mad_s": consistency,
                    "tyre_age_source": tyre_age_source,
                    "tyre_age_window": tyre_age_window,
                    "evolution_window": evolution_window,
                    "dry_status": "dry",
                }
            )
    return rows


def _aggregate_entities(
    stint_rows: list[dict[str, Any]],
    *,
    entity_key: str,
    cfg: RacePracticeEvidenceConfig,
    prior_deg_by_team: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, Any]:
    if not stint_rows:
        return {}
    frame = pd.DataFrame(stint_rows)
    output: dict[str, Any] = {}
    for entity, entity_rows in frame.groupby(entity_key, dropna=False):
        compounds: dict[str, Any] = {}
        for compound, rows in entity_rows.groupby("compound", dropna=False):
            weights = rows["n_laps"].to_numpy(dtype=float)
            pace = _weighted_median(rows["reference_pace_s"].to_numpy(dtype=float), weights)
            raw_deg = _weighted_median(
                rows["tire_deg_slope_s_per_lap"].to_numpy(dtype=float), weights
            )
            n_laps = int(rows["n_laps"].sum())
            prior = None
            if prior_deg_by_team is not None:
                team = str(rows["team"].iloc[0])
                prior = _finite_float(prior_deg_by_team.get(team, {}).get(str(compound)))
            if prior is None:
                deg = raw_deg
                deg_source = "observed_only"
            else:
                observed_weight = float(n_laps / (n_laps + cfg.shrinkage_laps))
                deg = float((observed_weight * raw_deg) + ((1.0 - observed_weight) * prior))
                deg_source = "observed_shrunk_to_track_prior"
            compounds[str(compound)] = {
                "reference_pace_s": round(float(pace), 6),
                "pace_se_s": round(_robust_se(rows["reference_pace_s"]), 6),
                "tire_deg_slope_s_per_lap": round(float(deg), 6),
                "raw_tire_deg_slope_s_per_lap": round(float(raw_deg), 6),
                "tire_deg_se_s_per_lap": round(_robust_se(rows["tire_deg_slope_s_per_lap"]), 6),
                "consistency_mad_s": round(
                    _weighted_median(rows["consistency_mad_s"].to_numpy(), weights), 6
                ),
                "n_laps": n_laps,
                "n_stints": int(len(rows)),
                "deg_source": deg_source,
            }
        output[str(entity)] = {
            "compounds": compounds,
            "n_laps": int(entity_rows["n_laps"].sum()),
            "n_stints": int(len(entity_rows)),
            "evidence_quality": _quality_label(int(entity_rows["n_laps"].sum())),
        }
    return output


def _attach_compound_pace_performance(
    teams: dict[str, Any],
    stint_rows: list[dict[str, Any]],
) -> None:
    """Attach pace only when teams share the complete comparison context."""

    for team_payload in teams.values():
        for metrics in team_payload.get("compounds", {}).values():
            metrics.update(
                {
                    "pace_comparison_status": "no_matched_bucket",
                    "matched_pace_buckets": 0,
                    "matched_pace_laps": 0,
                    "matched_pace_stints": 0,
                }
            )

    buckets: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in stint_rows:
        tyre_age_window = tuple(row["tyre_age_window"])
        evolution_window = tuple(row["evolution_window"])
        if evolution_window[0] != evolution_window[1] or row["dry_status"] != "dry":
            continue
        bucket = (
            str(row["session"]),
            str(row["compound"]),
            str(row["tyre_age_source"]),
            *tyre_age_window,
            *evolution_window,
            str(row["dry_status"]),
        )
        buckets[bucket][str(row["team"])].append(row)

    matched: dict[tuple[str, str], list[tuple[float, int, int]]] = defaultdict(list)
    for bucket in sorted(buckets, key=lambda value: tuple(str(item) for item in value)):
        team_rows = buckets[bucket]
        if len(team_rows) < 2:
            continue
        compound = str(bucket[1])
        entries: dict[str, tuple[float, int, int]] = {}
        for team, rows in team_rows.items():
            weights = np.asarray([int(row["n_laps"]) for row in rows], dtype=float)
            paces = np.asarray([float(row["reference_pace_s"]) for row in rows], dtype=float)
            entries[team] = (
                _weighted_median(paces, weights),
                int(weights.sum()),
                len(rows),
            )
        unique_paces = sorted({entry[0] for entry in entries.values()})
        pace_denominator = max(1, len(unique_paces) - 1)
        for team, (pace, n_laps, n_stints) in entries.items():
            performance = (
                0.5
                if len(unique_paces) == 1
                else 1.0 - (unique_paces.index(pace) / pace_denominator)
            )
            matched[(team, compound)].append((performance, n_laps, n_stints))

    for (team, compound), comparisons in matched.items():
        metrics = teams[team]["compounds"][compound]
        weights = np.asarray([comparison[1] for comparison in comparisons], dtype=float)
        performances = np.asarray([comparison[0] for comparison in comparisons], dtype=float)
        metrics.update(
            {
                "pace_comparison_status": "matched",
                "pace_performance": round(float(np.average(performances, weights=weights)), 6),
                "matched_pace_buckets": len(comparisons),
                "matched_pace_laps": int(weights.sum()),
                "matched_pace_stints": int(sum(comparison[2] for comparison in comparisons)),
            }
        )


def _consecutive_segments(stint: pd.DataFrame) -> list[pd.DataFrame]:
    if "LapNumber" not in stint.columns or stint["LapNumber"].isna().all():
        return [stint]
    lap_numbers = stint["LapNumber"].to_numpy(dtype=float)
    boundaries = np.concatenate(([True], np.diff(lap_numbers) > 1.0))
    groups = np.cumsum(boundaries)
    return [segment for _, segment in stint.groupby(groups)]


def _remove_robust_outliers(
    stint: pd.DataFrame,
    *,
    cfg: RacePracticeEvidenceConfig,
) -> pd.DataFrame:
    values = stint["adjusted_lap_s"].to_numpy(dtype=float)
    median = float(np.median(values))
    mad = _mad(values)
    if mad <= 0:
        return stint
    limit = max(0.5, cfg.outlier_mad_scale * 1.4826 * mad)
    return stint[np.abs(stint["adjusted_lap_s"] - median) <= limit]


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def _elapsed_seconds(frame: pd.DataFrame) -> pd.Series:
    for column in ("Time", "LapStartTime"):
        if column in frame.columns:
            seconds = _seconds(frame[column])
            if seconds.notna().any():
                return seconds.interpolate(limit_direction="both").fillna(0.0)
    lap_number = pd.to_numeric(frame["LapNumber"], errors="coerce")
    return lap_number.fillna(lap_number.median()).fillna(0.0) * 90.0


def _mad(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _robust_se(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size <= 1:
        return 0.0
    return float((1.4826 * _mad(finite)) / np.sqrt(finite.size))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cutoff = float(sorted_weights.sum()) / 2.0
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _quality_label(n_laps: int) -> str:
    if n_laps >= 32:
        return "high"
    if n_laps >= 16:
        return "medium"
    return "low"


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes", "y", "rain", "wet"}


def _record_mask_exclusions(
    keep_mask: pd.Series,
    exclusions: dict[str, int],
    reason: str,
) -> None:
    _count(exclusions, reason, int((~keep_mask.fillna(False)).sum()))


def _count(exclusions: dict[str, int], reason: str, count: int) -> None:
    if count > 0:
        exclusions[reason] = exclusions.get(reason, 0) + int(count)
