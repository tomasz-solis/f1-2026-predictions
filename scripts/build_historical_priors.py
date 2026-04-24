# ruff: noqa: E402
"""Build season-scoped preseason priors for historical backtests.

The replay loader already prefers ``<year>_*.json`` artifacts when they exist.
What was missing was a clean way to produce those files without accidentally
using the target season's own race results as priors.

This helper keeps the split explicit:

- team priors come from one source season before the target year
- driver priors come from earlier seasons plus the target-season lineup
- track priors come from earlier seasons only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calculate_team_performance import (
    calculate_team_performance_from_races,
    rank_teams_by_performance,
)
from scripts.generate_2026_baseline import calculate_track_characteristics

from src.utils.lineups import get_lineups_from_session
from src.utils.normalization import rank_normalize
from src.utils.schema_validation import (
    strip_legacy_bayesian_fields,
    validate_driver_characteristics,
    validate_team_characteristics,
    validate_track_characteristics,
)
from src.utils.team_mapping import map_team_to_characteristics
from src.utils.weekend import get_schedule_rows

logger = logging.getLogger(__name__)

_TEAM_PRESEASON_FLOOR = 0.35
_TEAM_PRESEASON_RANGE = 0.50
_TEAM_PRESEASON_EXPONENT = 1.35
_TEAM_UNCERTAINTY_MIN = 0.12
_TEAM_UNCERTAINTY_MAX = 0.30
_DRIVER_PRIOR_FLOOR = 0.30
_DRIVER_PRIOR_CEILING = 0.90
_TEAM_BASED_ROOKIE_CAP = 0.62
_DEFAULT_DRIVER_SIGMA = 2.5


def _coerce_float(value: Any) -> float | None:
    """Return one finite float when possible."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _build_directionality_from_metrics(
    metrics: dict[str, float],
    *,
    directionality_scale: float = 0.10,
) -> dict[str, float]:
    """Convert normalized pace traits into small centered directionality deltas."""
    metric_map = {
        "max_speed": "top_speed",
        "slow_corner_speed": "slow_corner_performance",
        "medium_corner_speed": "medium_corner_performance",
        "high_corner_speed": "fast_corner_performance",
    }

    fallback_pace = _coerce_float(metrics.get("overall_pace"))
    directionality: dict[str, float] = {}
    for output_key, metric_name in metric_map.items():
        metric_value = _coerce_float(metrics.get(metric_name))
        if metric_value is None and fallback_pace is not None and metric_name != "top_speed":
            metric_value = fallback_pace
        if metric_value is None:
            metric_value = 0.5

        centered = (metric_value - 0.5) * float(directionality_scale)
        directionality[output_key] = round(float(np.clip(centered, -0.2, 0.2)), 4)

    return directionality


def _canonicalize_team_source_payload(
    raw_team_payload: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collapse raw source teams onto canonical characteristics keys."""
    canonical_payload: dict[str, dict[str, Any]] = {}
    for raw_team_name, raw_values in raw_team_payload.items():
        if not isinstance(raw_values, dict):
            continue
        canonical_team = map_team_to_characteristics(str(raw_team_name)) or str(raw_team_name)
        existing = canonical_payload.get(canonical_team)
        if existing is None:
            canonical_payload[canonical_team] = dict(raw_values)
            continue

        existing_races = int(existing.get("races_analyzed", 0) or 0)
        candidate_races = int(raw_values.get("races_analyzed", 0) or 0)
        if candidate_races >= existing_races:
            canonical_payload[canonical_team] = dict(raw_values)
    return canonical_payload


def _resolve_team_metric_scores(
    canonical_payload: dict[str, dict[str, Any]],
    *,
    metric_key: str,
    fallback_scores: dict[str, float],
    higher_is_better: bool,
) -> dict[str, float]:
    """Resolve one normalized team metric map with rank-based fallback."""
    explicit_scores: dict[str, float] = {}
    fallback_values: dict[str, float] = {}

    for team_name, team_values in canonical_payload.items():
        explicit_value = _coerce_float(team_values.get(metric_key))
        if explicit_value is not None:
            explicit_scores[team_name] = float(np.clip(explicit_value, 0.0, 1.0))
            continue

        fallback_value = _coerce_float(team_values.get("overall_performance"))
        if fallback_value is not None:
            fallback_values[team_name] = fallback_value

    if explicit_scores:
        resolved = dict(fallback_scores)
        resolved.update(explicit_scores)
        return resolved

    if fallback_values:
        return rank_normalize(fallback_values, higher_is_better=higher_is_better)

    return dict(fallback_scores)


def _scale_preseason_team_strength(overall_score: float) -> float:
    """Map a 0-1 field-relative team score onto the predictor's preseason scale."""
    bounded_score = float(np.clip(overall_score, 0.0, 1.0))
    scaled = _TEAM_PRESEASON_FLOOR + (
        _TEAM_PRESEASON_RANGE * (bounded_score**_TEAM_PRESEASON_EXPONENT)
    )
    return round(float(np.clip(scaled, 0.0, 1.0)), 3)


def _scale_preseason_team_uncertainty(overall_score: float, raw_uncertainty: float | None) -> float:
    """Blend raw season variability into a conservative preseason uncertainty band."""
    baseline = _TEAM_UNCERTAINTY_MIN + ((1.0 - float(np.clip(overall_score, 0.0, 1.0))) * 0.14)
    source_uncertainty = float(np.clip(_coerce_float(raw_uncertainty) or 0.20, 0.10, 0.40))
    blended = (0.60 * baseline) + (0.40 * source_uncertainty)
    return round(float(np.clip(blended, _TEAM_UNCERTAINTY_MIN, _TEAM_UNCERTAINTY_MAX)), 2)


def _build_team_profile(
    *,
    run_profile: str,
    overall_pace: float,
    top_speed: float | None = None,
    slow_corner_performance: float | None = None,
    medium_corner_performance: float | None = None,
    fast_corner_performance: float | None = None,
    consistency: float | None = None,
    tire_deg_performance: float | None = None,
) -> dict[str, float | str]:
    """Build one stored testing-style team profile from normalized source metrics."""
    profile: dict[str, float | str] = {
        "run_profile": run_profile,
        "overall_pace": round(float(np.clip(overall_pace, 0.0, 1.0)), 3),
    }

    optional_metrics = {
        "top_speed": top_speed,
        "slow_corner_performance": slow_corner_performance,
        "medium_corner_performance": medium_corner_performance,
        "fast_corner_performance": fast_corner_performance,
        "consistency": consistency,
        "tire_deg_performance": tire_deg_performance,
    }
    for metric_name, metric_value in optional_metrics.items():
        if metric_value is None:
            continue
        profile[metric_name] = round(float(np.clip(metric_value, 0.0, 1.0)), 3)

    return profile


def _build_team_profiles(
    *,
    overall_score: float,
    top_speed_score: float,
    slow_corner_score: float,
    medium_corner_score: float,
    fast_corner_score: float,
    consistency_score: float,
    tire_deg_score: float,
) -> dict[str, dict[str, float | str]]:
    """Build balanced, short-run, and long-run team profiles for replay fallback."""
    balanced = _build_team_profile(
        run_profile="balanced",
        overall_pace=overall_score,
        top_speed=top_speed_score,
        slow_corner_performance=slow_corner_score,
        medium_corner_performance=medium_corner_score,
        fast_corner_performance=fast_corner_score,
        consistency=consistency_score,
        tire_deg_performance=tire_deg_score,
    )
    short_run = _build_team_profile(
        run_profile="short_run",
        overall_pace=(
            (0.70 * overall_score) + (0.20 * top_speed_score) + (0.10 * fast_corner_score)
        ),
        top_speed=top_speed_score,
        medium_corner_performance=medium_corner_score,
        fast_corner_performance=fast_corner_score,
        consistency=consistency_score,
    )
    long_run = _build_team_profile(
        run_profile="long_run",
        overall_pace=(
            (0.60 * overall_score) + (0.25 * tire_deg_score) + (0.15 * consistency_score)
        ),
        slow_corner_performance=slow_corner_score,
        medium_corner_performance=medium_corner_score,
        consistency=consistency_score,
        tire_deg_performance=tire_deg_score,
    )
    return {
        "balanced": balanced,
        "short_run": short_run,
        "long_run": long_run,
    }


def _repair_driver_prior_metadata(
    drivers_payload: dict[str, Any],
    *,
    target_year: int,
) -> None:
    """Fix season markers for lineup-seeded rookies in historical prior builds."""
    for driver_entry in drivers_payload.values():
        if not isinstance(driver_entry, dict):
            continue

        if driver_entry.get("prior_source") != "team_based_prior":
            continue

        experience = driver_entry.get("experience")
        if not isinstance(experience, dict):
            experience = {}
            driver_entry["experience"] = experience

        experience["years_of_experience"] = 0
        experience["tier"] = "rookie"
        experience["total_races"] = int(experience.get("total_races", 0) or 0)

        debut_year = _coerce_float(experience.get("debut_year"))
        if debut_year is None or int(debut_year) > int(target_year):
            experience["debut_year"] = int(target_year)


def _resolve_driver_prior_shrink(driver_entry: dict[str, Any]) -> float:
    """Return a conservative preseason shrink factor for driver priors."""
    experience = driver_entry.get("experience", {}) if isinstance(driver_entry, dict) else {}
    prior_source = str(driver_entry.get("prior_source", "")).strip().lower()
    tier = str(experience.get("tier", "")).strip().lower()
    total_races = int(experience.get("total_races", 0) or 0)

    if prior_source == "team_based_prior" or total_races == 0:
        return 0.40
    if tier == "rookie":
        return 0.50
    if tier == "developing":
        return 0.58
    if tier == "established":
        return 0.62
    return 0.65


def _shrink_driver_metric(
    value: float,
    *,
    field_mean: float,
    shrink_factor: float,
) -> float:
    """Pull one driver metric toward the field mean for preseason use."""
    adjusted = field_mean + ((float(value) - field_mean) * float(shrink_factor))
    return float(np.clip(adjusted, _DRIVER_PRIOR_FLOOR, _DRIVER_PRIOR_CEILING))


def _reseed_driver_bayesian_state(
    drivers_payload: dict[str, Any],
    *,
    grid_size: int,
    target_year: int,
) -> None:
    """Rebuild Bayesian preseason state from the rewritten driver skill scores."""
    bounded_grid_size = max(2, int(grid_size))
    for driver_entry in drivers_payload.values():
        if not isinstance(driver_entry, dict):
            continue

        skill = _coerce_float(driver_entry.get("racecraft", {}).get("skill_score"))
        if skill is None:
            continue

        prior_source = str(driver_entry.get("prior_source", "")).strip().lower()
        experience = driver_entry.get("experience", {}) if isinstance(driver_entry, dict) else {}
        tier = str(experience.get("tier", "")).strip().lower()
        if prior_source == "team_based_prior":
            rating_sigma = 3.2
        elif tier == "rookie":
            rating_sigma = 2.9
        elif tier == "developing":
            rating_sigma = 2.7
        else:
            rating_sigma = _DEFAULT_DRIVER_SIGMA

        driver_entry["bayesian"] = {
            "rating_mu": round(1.0 + (skill * max(bounded_grid_size - 1, 1)), 3),
            "rating_sigma": round(float(rating_sigma), 3),
            "sessions_observed": 0,
            "seeded_from": "extraction_prior",
            "last_updated": None,
            "season_year": int(target_year),
        }


def _conservatively_rebalance_driver_priors(
    drivers_payload: dict[str, Any],
    *,
    grid_size: int,
    target_year: int,
) -> None:
    """Shrink historical driver priors toward the field mean for preseason replay."""
    metric_paths = (
        ("pace", "quali_pace"),
        ("pace", "race_pace"),
        ("racecraft", "skill_score"),
        ("racecraft", "overtaking_skill"),
    )
    field_means: dict[tuple[str, str], float] = {}
    for section_name, metric_name in metric_paths:
        values = [
            value
            for driver_entry in drivers_payload.values()
            if isinstance(driver_entry, dict)
            and isinstance(driver_entry.get(section_name), dict)
            and (value := _coerce_float(driver_entry[section_name].get(metric_name))) is not None
        ]
        if values:
            field_means[(section_name, metric_name)] = float(
                np.mean(np.asarray(values, dtype=float))
            )

    for driver_entry in drivers_payload.values():
        if not isinstance(driver_entry, dict):
            continue

        shrink_factor = _resolve_driver_prior_shrink(driver_entry)
        for section_name, metric_name in metric_paths:
            section = driver_entry.get(section_name)
            if not isinstance(section, dict):
                continue
            current_value = _coerce_float(section.get(metric_name))
            field_mean = field_means.get((section_name, metric_name))
            if current_value is None or field_mean is None:
                continue
            section[metric_name] = round(
                _shrink_driver_metric(
                    current_value,
                    field_mean=field_mean,
                    shrink_factor=shrink_factor,
                ),
                3,
            )

        if driver_entry.get("prior_source") == "team_based_prior":
            racecraft = driver_entry.get("racecraft", {})
            pace = driver_entry.get("pace", {})
            if isinstance(racecraft, dict):
                skill_score = _coerce_float(racecraft.get("skill_score"))
                overtaking_skill = _coerce_float(racecraft.get("overtaking_skill"))
                if skill_score is not None:
                    racecraft["skill_score"] = round(
                        float(min(skill_score, _TEAM_BASED_ROOKIE_CAP)),
                        3,
                    )
                if overtaking_skill is not None:
                    racecraft["overtaking_skill"] = round(
                        float(min(overtaking_skill, _TEAM_BASED_ROOKIE_CAP)),
                        3,
                    )
            if isinstance(pace, dict):
                quali_pace = _coerce_float(pace.get("quali_pace"))
                race_pace = _coerce_float(pace.get("race_pace"))
                if quali_pace is not None:
                    pace["quali_pace"] = round(float(min(quali_pace, _TEAM_BASED_ROOKIE_CAP)), 3)
                if race_pace is not None:
                    pace["race_pace"] = round(float(min(race_pace, _TEAM_BASED_ROOKIE_CAP)), 3)

    _reseed_driver_bayesian_state(
        drivers_payload,
        grid_size=grid_size,
        target_year=target_year,
    )


def parse_years_arg(raw_years: str | None) -> list[int]:
    """Parse a comma-separated year list into a sorted unique sequence."""
    if raw_years is None or not str(raw_years).strip():
        return []

    parsed: list[int] = []
    for chunk in str(raw_years).split(","):
        year_text = chunk.strip()
        if not year_text:
            continue
        parsed.append(int(year_text))
    return sorted(set(parsed))


def default_history_years(target_year: int, window: int = 3) -> list[int]:
    """Return the previous ``window`` completed seasons before ``target_year``."""
    if window <= 0:
        raise ValueError("history window must be positive")
    start_year = int(target_year) - int(window)
    if start_year < 1950:
        raise ValueError("history window reaches before modern Formula 1 seasons")
    return list(range(start_year, int(target_year)))


def build_prior_note(target_year: int, source_description: str) -> str:
    """Render one short note describing how a preseason prior was built."""
    return (
        f"Preseason {target_year} prior built from {source_description}. "
        "Used for historical replay without target-season race leakage."
    )


def utcnow_iso() -> str:
    """Return a stable UTC timestamp for artifact metadata."""
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_team_prior_payload(
    *,
    target_year: int,
    source_year: int,
    raw_team_payload: dict[str, dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Rewrite race-derived team metrics into a preseason-schema payload."""
    season_note = build_prior_note(target_year, f"{source_year} race lap-time performance")
    canonical_payload = _canonicalize_team_source_payload(raw_team_payload)
    overall_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_overall_pace",
        fallback_scores={},
        higher_is_better=True,
    )
    top_speed_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_top_speed",
        fallback_scores=overall_scores,
        higher_is_better=True,
    )
    slow_corner_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_slow_corner_performance",
        fallback_scores=overall_scores,
        higher_is_better=True,
    )
    medium_corner_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_medium_corner_performance",
        fallback_scores=overall_scores,
        higher_is_better=True,
    )
    fast_corner_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_fast_corner_performance",
        fallback_scores=overall_scores,
        higher_is_better=True,
    )
    consistency_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_consistency",
        fallback_scores=overall_scores,
        higher_is_better=True,
    )
    tire_deg_scores = _resolve_team_metric_scores(
        canonical_payload,
        metric_key="normalized_tire_deg_performance",
        fallback_scores=consistency_scores,
        higher_is_better=True,
    )
    payload: dict[str, Any] = {
        "year": int(target_year),
        "generated_at": generated_at or utcnow_iso(),
        "data_freshness": "BASELINE_PRESEASON",
        "note": season_note,
        "races_completed": 0,
        "teams": {},
    }

    for canonical_team, raw_values in canonical_payload.items():
        if not isinstance(raw_values, dict):
            continue

        races_analyzed = int(raw_values.get("races_analyzed", 0) or 0)
        championship_position = raw_values.get("championship_position")
        team_note = (
            f"{source_year} source rank P{championship_position}, {races_analyzed} race(s) analyzed"
            if championship_position is not None
            else f"{source_year} lap-time model, {races_analyzed} race(s) analyzed"
        )
        overall_score = overall_scores.get(canonical_team, 0.5)
        top_speed_score = top_speed_scores.get(canonical_team, overall_score)
        slow_corner_score = slow_corner_scores.get(canonical_team, overall_score)
        medium_corner_score = medium_corner_scores.get(canonical_team, overall_score)
        fast_corner_score = fast_corner_scores.get(canonical_team, overall_score)
        consistency_score = consistency_scores.get(canonical_team, 1.0 - overall_score)
        tire_deg_score = tire_deg_scores.get(canonical_team, consistency_score)
        team_profiles = _build_team_profiles(
            overall_score=overall_score,
            top_speed_score=top_speed_score,
            slow_corner_score=slow_corner_score,
            medium_corner_score=medium_corner_score,
            fast_corner_score=fast_corner_score,
            consistency_score=consistency_score,
            tire_deg_score=tire_deg_score,
        )
        balanced_profile = dict(team_profiles["balanced"])
        has_directionality_metrics = any(
            _coerce_float(raw_values.get(metric_name)) is not None
            for metric_name in (
                "normalized_top_speed",
                "normalized_slow_corner_performance",
                "normalized_medium_corner_performance",
                "normalized_fast_corner_performance",
            )
        )

        preseason_performance = _scale_preseason_team_strength(overall_score)
        team_entry: dict[str, Any] = {
            "overall_performance": preseason_performance,
            "preseason_overall_performance": preseason_performance,
            "uncertainty": _scale_preseason_team_uncertainty(
                overall_score,
                _coerce_float(raw_values.get("uncertainty")),
            ),
            "note": (
                f"{team_note}; preseason strength scaled from normalized {source_year} race pace."
            ),
            "last_updated": None,
            "races_completed": 0,
            "current_season_performance": [],
            "testing_characteristics": balanced_profile,
            "testing_characteristics_profiles": team_profiles,
        }
        if has_directionality_metrics:
            team_entry["directionality"] = _build_directionality_from_metrics(
                {
                    "overall_pace": overall_score,
                    "top_speed": top_speed_score,
                    "slow_corner_performance": slow_corner_score,
                    "medium_corner_performance": medium_corner_score,
                    "fast_corner_performance": fast_corner_score,
                }
            )
        payload["teams"][canonical_team] = team_entry

    if not payload["teams"]:
        raise ValueError(f"Could not build team priors for season {target_year}")
    return payload


def rewrite_track_prior_payload(
    raw_payload: dict[str, Any],
    *,
    target_year: int,
    source_years: list[int],
) -> dict[str, Any]:
    """Rewrite generated track data so the saved season matches the replay year."""
    if not source_years:
        raise ValueError("track prior generation requires at least one source year")

    payload = dict(raw_payload)
    payload["year"] = int(target_year)
    payload["data_freshness"] = "BASELINE_PRESEASON"
    payload["generated_at"] = payload.get("generated_at") or utcnow_iso()
    payload["generated_from"] = (
        f"Historical averages from {min(source_years)}-{max(source_years)}"
        if len(source_years) > 1
        else f"Historical averages from {source_years[0]}"
    )
    payload["note"] = build_prior_note(
        target_year,
        f"track-history window {payload['generated_from'].split(' from ', 1)[1]}",
    )
    return payload


def rewrite_driver_prior_payload(
    raw_payload: dict[str, Any],
    *,
    target_year: int,
    source_years: list[int],
    lineup_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach season metadata to extracted driver priors."""
    if not source_years:
        raise ValueError("driver prior generation requires at least one source year")

    payload = dict(raw_payload)
    drivers = payload.get("drivers")
    if isinstance(drivers, dict):
        stripped = strip_legacy_bayesian_fields(drivers)
        if stripped > 0:
            logger.info(
                "Stripped %s legacy Bayesian field(s) from extracted driver priors",
                stripped,
            )
        _repair_driver_prior_metadata(drivers, target_year=target_year)
        lineup_grid_size = 20
        if isinstance(lineup_payload, dict):
            lineups = lineup_payload.get("current_lineups", {})
            if isinstance(lineups, dict):
                active_driver_count = len(
                    {
                        str(driver_code).strip().upper()
                        for team_drivers in lineups.values()
                        if isinstance(team_drivers, list)
                        for driver_code in team_drivers
                        if driver_code
                    }
                )
                if active_driver_count >= 2:
                    lineup_grid_size = active_driver_count
        _conservatively_rebalance_driver_priors(
            drivers,
            grid_size=lineup_grid_size,
            target_year=target_year,
        )
    payload["year"] = int(target_year)
    payload["last_updated"] = payload.get("last_updated") or utcnow_iso()
    payload["note"] = build_prior_note(
        target_year,
        f"driver form from seasons {', '.join(str(year) for year in source_years)}",
    )
    return payload


def resolve_first_race_name(target_year: int) -> str:
    """Return the first non-testing race listed in the season schedule."""
    for race_name, event_format in get_schedule_rows(int(target_year)):
        normalized_name = str(race_name).strip()
        normalized_format = str(event_format).strip().lower()
        if not normalized_name:
            continue
        if "testing" in normalized_name.lower() or "testing" in normalized_format:
            continue
        return normalized_name
    raise ValueError(f"Could not resolve the first scheduled race for season {target_year}")


def build_lineup_seed_payload(
    *,
    target_year: int,
    race_name: str,
    lineups: dict[str, list[str]],
) -> dict[str, Any]:
    """Package one season lineup mapping in the extractor's expected JSON layout."""
    normalized = {
        str(team_name): [str(driver_code).strip().upper() for driver_code in drivers if driver_code]
        for team_name, drivers in lineups.items()
        if isinstance(drivers, list) and drivers
    }
    if not normalized:
        raise ValueError(
            f"Could not build a lineup seed for season {target_year}: no team mappings found"
        )
    return {
        "season": int(target_year),
        "source_race": str(race_name),
        "note": (
            "Lineup seed extracted from the target season's first qualifying session so "
            "rookie and substitution priors stay season-correct."
        ),
        "current_lineups": normalized,
    }


def resolve_lineup_seed_payload(
    *,
    target_year: int,
    lineup_file: Path | None = None,
) -> dict[str, Any]:
    """Return an explicit lineup payload for historical driver prior extraction."""
    if lineup_file is not None:
        with open(lineup_file) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Lineup seed file {lineup_file} did not contain a JSON object")
        if not isinstance(payload.get("current_lineups"), dict):
            raise ValueError(
                f"Lineup seed file {lineup_file} is missing a valid current_lineups mapping"
            )
        return payload

    first_race_name = resolve_first_race_name(target_year)
    lineups = get_lineups_from_session(target_year, first_race_name, "Q")
    if not isinstance(lineups, dict) or not lineups:
        raise ValueError(
            "Could not derive target-season lineups from FastF1. "
            "Pass --lineup-file with a season-specific lineup JSON."
        )
    return build_lineup_seed_payload(
        target_year=target_year,
        race_name=first_race_name,
        lineups=lineups,
    )


def build_driver_extraction_command(
    *,
    source_years: list[int],
    output_path: Path,
    lineup_file: Path,
    request_delay: float,
    max_attempts: int,
    timeout_budget_seconds: float,
) -> list[str]:
    """Build the exact driver-extraction command used by this helper."""
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "extract_driver_characteristics.py"),
        "--years",
        ",".join(str(year) for year in source_years),
        "--output",
        str(output_path),
        "--lineup-file",
        str(lineup_file),
        "--request-delay",
        str(request_delay),
        "--max-attempts",
        str(max_attempts),
        "--timeout-budget-seconds",
        str(timeout_budget_seconds),
    ]


def build_team_priors(*, target_year: int, source_year: int, output_path: Path) -> dict[str, Any]:
    """Compute, validate, and persist one season-scoped team prior payload."""
    raw_team_payload = rank_teams_by_performance(calculate_team_performance_from_races(source_year))
    team_payload = build_team_prior_payload(
        target_year=target_year,
        source_year=source_year,
        raw_team_payload=raw_team_payload,
    )
    validate_team_characteristics(team_payload, expected_year=target_year)
    write_json(output_path, team_payload)
    return team_payload


def build_track_priors(
    *,
    target_year: int,
    source_years: list[int],
    output_path: Path,
) -> dict[str, Any]:
    """Compute, validate, and persist one season-scoped track prior payload."""
    with tempfile.TemporaryDirectory(prefix="historical-track-priors-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        calculate_track_characteristics(source_years, temp_dir)
        raw_output_path = temp_dir / "track_characteristics" / "2026_track_characteristics.json"
        with open(raw_output_path) as handle:
            raw_payload = json.load(handle)

    track_payload = rewrite_track_prior_payload(
        raw_payload,
        target_year=target_year,
        source_years=source_years,
    )
    validate_track_characteristics(track_payload, expected_year=target_year)
    write_json(output_path, track_payload)
    return track_payload


def build_driver_priors(
    *,
    target_year: int,
    source_years: list[int],
    output_path: Path,
    lineup_payload: dict[str, Any],
    request_delay: float,
    max_attempts: int,
    timeout_budget_seconds: float,
) -> dict[str, Any]:
    """Run the driver extractor with explicit target-season lineups."""
    with tempfile.TemporaryDirectory(prefix="historical-driver-priors-") as temp_dir_name:
        lineup_path = Path(temp_dir_name) / f"{target_year}_lineups.json"
        write_json(lineup_path, lineup_payload)
        command = build_driver_extraction_command(
            source_years=source_years,
            output_path=output_path,
            lineup_file=lineup_path,
            request_delay=request_delay,
            max_attempts=max_attempts,
            timeout_budget_seconds=timeout_budget_seconds,
        )
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    with open(output_path) as handle:
        raw_payload = json.load(handle)

    driver_payload = rewrite_driver_prior_payload(
        raw_payload,
        target_year=target_year,
        source_years=source_years,
        lineup_payload=lineup_payload,
    )
    validate_driver_characteristics(driver_payload, expected_year=target_year)
    write_json(output_path, driver_payload)
    return driver_payload


def parse_args() -> argparse.Namespace:
    """Parse the small CLI surface for historical prior generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Season to build priors for.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Processed artifact root to write season-scoped files into.",
    )
    parser.add_argument(
        "--team-source-year",
        type=int,
        default=None,
        help="Season used for team priors. Defaults to target year minus one.",
    )
    parser.add_argument(
        "--driver-years",
        type=str,
        default=None,
        help="Comma-separated seasons used for driver priors. Defaults to the previous 3 years.",
    )
    parser.add_argument(
        "--track-years",
        type=str,
        default=None,
        help="Comma-separated seasons used for track priors. Defaults to the previous 3 years.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=3,
        help="Fallback lookback window when --driver-years or --track-years is omitted.",
    )
    parser.add_argument(
        "--lineup-file",
        type=str,
        default=None,
        help=(
            "Optional season-specific lineup JSON in current_lineups format. "
            "When omitted, the script extracts lineups from the target season's first Q session."
        ),
    )
    parser.add_argument("--skip-teams", action="store_true", help="Skip team prior generation.")
    parser.add_argument("--skip-drivers", action="store_true", help="Skip driver prior generation.")
    parser.add_argument("--skip-tracks", action="store_true", help="Skip track prior generation.")
    parser.add_argument(
        "--driver-request-delay",
        type=float,
        default=0.80,
        help="Seconds to sleep after each FastF1 network call in the driver extractor.",
    )
    parser.add_argument(
        "--driver-max-attempts",
        type=int,
        default=8,
        help="Max retries per FastF1 request in the driver extractor.",
    )
    parser.add_argument(
        "--driver-timeout-budget-seconds",
        type=float,
        default=180.0,
        help="Retry budget per FastF1 request in the driver extractor.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the requested season-scoped preseason prior files."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    target_year = int(args.year)
    data_dir = Path(args.data_dir)
    team_source_year = int(args.team_source_year or (target_year - 1))
    driver_years = parse_years_arg(args.driver_years) or default_history_years(
        target_year,
        args.history_window,
    )
    track_years = parse_years_arg(args.track_years) or default_history_years(
        target_year,
        args.history_window,
    )

    if team_source_year >= target_year:
        raise ValueError("--team-source-year must be earlier than --year")
    if any(year >= target_year for year in driver_years):
        raise ValueError("--driver-years must exclude the target season")
    if any(year >= target_year for year in track_years):
        raise ValueError("--track-years must exclude the target season")

    logger.info("Building preseason priors for %s under %s", target_year, data_dir)

    outputs: dict[str, str] = {}
    if not args.skip_teams:
        team_output = data_dir / "car_characteristics" / f"{target_year}_car_characteristics.json"
        build_team_priors(
            target_year=target_year,
            source_year=team_source_year,
            output_path=team_output,
        )
        outputs["teams"] = str(team_output)
        logger.info("Wrote team priors to %s", team_output)

    if not args.skip_tracks:
        track_output = (
            data_dir / "track_characteristics" / f"{target_year}_track_characteristics.json"
        )
        build_track_priors(
            target_year=target_year,
            source_years=track_years,
            output_path=track_output,
        )
        outputs["tracks"] = str(track_output)
        logger.info("Wrote track priors to %s", track_output)

    if not args.skip_drivers:
        lineup_payload = resolve_lineup_seed_payload(
            target_year=target_year,
            lineup_file=Path(args.lineup_file) if args.lineup_file else None,
        )
        driver_output = (
            data_dir / "driver_characteristics" / f"{target_year}_driver_characteristics.json"
        )
        build_driver_priors(
            target_year=target_year,
            source_years=driver_years,
            output_path=driver_output,
            lineup_payload=lineup_payload,
            request_delay=float(args.driver_request_delay),
            max_attempts=int(args.driver_max_attempts),
            timeout_budget_seconds=float(args.driver_timeout_budget_seconds),
        )
        outputs["drivers"] = str(driver_output)
        logger.info("Wrote driver priors to %s", driver_output)

    if not outputs:
        logger.warning("Nothing to do: all prior categories were skipped.")
        return 0

    logger.info("Historical prior build complete for %s", target_year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
