"""Team-strength and compound-selection helpers for baseline predictors."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.systems.weight_schedule import ScheduleType
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger("src.predictors.baseline_2026")

_LEGACY_2026_STANDINGS_SEED_ANCHORS = {
    "McLaren": 0.85,
    "Mercedes": 0.75,
    "Red Bull Racing": 0.74,
    "Ferrari": 0.70,
    "Williams": 0.55,
    "RB": 0.48,
    "Aston Martin": 0.47,
    "Haas F1 Team": 0.43,
    "Alpine": 0.40,
    "Audi": 0.38,
    "Sauber": 0.38,
    "Cadillac F1": 0.35,
}


def _coerce_unit_interval(value: object) -> float | None:
    """Return a bounded float when a payload value is a valid unit-interval number."""
    if not isinstance(value, int | float | str):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return float(np.clip(numeric_value, 0.0, 1.0))


def _looks_like_legacy_2026_standings_seed(team_data: Mapping[str, object]) -> bool:
    """Return whether a team payload looks like the old 2025-seeded 2026 file."""
    note = str(team_data.get("note") or "").lower()
    return "2025 p" in note and "2026" in note and "seed" in note


def resolve_preseason_baseline(
    team_data: Mapping[str, object],
    *,
    team_name: str | None = None,
    season_year: int | None = None,
) -> float:
    """Return the original season anchor before live race updates are applied.

    Runtime team strength blends a slow preseason anchor with current-season
    observations. If the updater has already moved ``overall_performance``
    toward early race results, using that same value as the anchor would count
    a tiny live sample twice.
    """
    preseason_baseline = _coerce_unit_interval(team_data.get("preseason_overall_performance"))
    if preseason_baseline is not None:
        return preseason_baseline

    if season_year == 2026 and team_name and _looks_like_legacy_2026_standings_seed(team_data):
        mapped_team = map_team_to_characteristics(
            team_name,
            known_teams=set(_LEGACY_2026_STANDINGS_SEED_ANCHORS),
        )
        if mapped_team in _LEGACY_2026_STANDINGS_SEED_ANCHORS:
            return _LEGACY_2026_STANDINGS_SEED_ANCHORS[mapped_team]

    overall_performance = _coerce_unit_interval(team_data.get("overall_performance"))
    return 0.5 if overall_performance is None else overall_performance


def resolve_team_data(*, teams: Mapping[str, object], team: str) -> dict[str, Any]:
    """Resolve team payload using alias-aware mapping before falling back to empty."""
    team_data = teams.get(team)
    if isinstance(team_data, dict):
        return team_data

    known_teams = set(teams.keys())
    mapped_team = map_team_to_characteristics(team, known_teams=known_teams)
    if not isinstance(mapped_team, str) or not mapped_team:
        return {}

    mapped_team_data = teams.get(mapped_team)
    return mapped_team_data if isinstance(mapped_team_data, dict) else {}


def calculate_track_suitability(
    *,
    team_data: Mapping[str, Any],
    track_profile: Mapping[str, Any],
) -> float:
    """Return a small track-suitability modifier from car directionality and layout mix."""
    directionality = team_data.get("directionality", {})
    if not isinstance(directionality, dict) or not directionality:
        return 0.0

    if "straights_pct" not in track_profile:
        return 0.0

    total_pct = (
        track_profile.get("straights_pct", 0)
        + track_profile.get("slow_corners_pct", 0)
        + track_profile.get("medium_corners_pct", 0)
        + track_profile.get("high_corners_pct", 0)
    )
    if total_pct == 0:
        return 0.0

    return (
        directionality.get("max_speed", 0) * (track_profile.get("straights_pct", 0) / total_pct)
        + directionality.get("slow_corner_speed", 0)
        * (track_profile.get("slow_corners_pct", 0) / total_pct)
        + directionality.get("medium_corner_speed", 0)
        * (track_profile.get("medium_corners_pct", 0) / total_pct)
        + directionality.get("high_corner_speed", 0)
        * (track_profile.get("high_corners_pct", 0) / total_pct)
    )


def get_blended_team_strength(
    *,
    context: Any,
    team: str,
    race_name: str,
    cfg: Any,
    schedules: Mapping[str, object],
    get_recommended_schedule_fn: Any,
    calculate_blended_performance_fn: Any,
) -> float:
    """Blend baseline, track suitability, and current form into one team score."""
    team_data = context._resolve_team_data(team)

    baseline = resolve_preseason_baseline(
        team_data,
        team_name=team,
        season_year=int(getattr(context, "season_year", getattr(context, "year", 2026))),
    )
    testing_modifier = context.calculate_track_suitability(team, race_name)
    testing_score = float(np.clip(baseline + testing_modifier, 0.0, 1.0))
    current = context._get_current_season_score(
        team,
        team_data,
        fallback=baseline,
        race_name=race_name,
    )

    race_number = context._get_contextual_races_completed(race_name) + 1
    configured_schedule = cfg.get("baseline_predictor.team_strength_schedule", None)
    schedule: ScheduleType
    if isinstance(configured_schedule, str) and configured_schedule in schedules:
        schedule = cast(ScheduleType, configured_schedule)
    else:
        schedule = get_recommended_schedule_fn(is_regulation_change=True)

    return float(
        calculate_blended_performance_fn(
            baseline_score=baseline,
            testing_modifier=testing_score,
            current_score=current,
            race_number=race_number,
            schedule=schedule,
        )
    )


def select_race_compound(*, race_name: str, season_year: int, cfg: Any) -> str:
    """Select the likely primary race compound from season Pirelli metadata."""
    try:
        candidate_years = [season_year]
        if season_year > 2020:
            candidate_years.append(season_year - 1)
        if 2025 not in candidate_years:
            candidate_years.append(2025)

        pirelli_file = Path("data") / f"{season_year}_pirelli_info.json"
        if not pirelli_file.exists():
            fallback_file = next(
                (
                    Path("data") / f"{candidate_year}_pirelli_info.json"
                    for candidate_year in candidate_years[1:]
                    if (Path("data") / f"{candidate_year}_pirelli_info.json").exists()
                ),
                None,
            )
            if fallback_file is None:
                return "MEDIUM"
            pirelli_file = fallback_file

        with open(pirelli_file) as handle:
            pirelli_data = json.load(handle)

        race_key = race_name.lower().replace(" ", "_").replace("-", "_")
        track_info = pirelli_data.get(race_key, {})
        if not track_info or "tyre_stress" not in track_info:
            return "MEDIUM"

        tyre_stress = track_info["tyre_stress"]
        high_threshold = cfg.get("baseline_predictor.compound_selection.high_stress_threshold", 3.5)
        low_threshold = cfg.get("baseline_predictor.compound_selection.low_stress_threshold", 2.5)
        default_stress = cfg.get(
            "baseline_predictor.compound_selection.default_stress_fallback",
            3.0,
        )
        stress_score = (
            tyre_stress.get("traction", default_stress)
            + tyre_stress.get("braking", default_stress)
            + tyre_stress.get("lateral", default_stress)
            + tyre_stress.get("asphalt_abrasion", default_stress)
        ) / 4.0

        if stress_score > high_threshold:
            return "HARD"
        if stress_score < low_threshold:
            return "SOFT"
        return "MEDIUM"
    except Exception as exc:
        logger.debug("Could not determine race compound for %s: %s", race_name, exc)
        return "MEDIUM"


def get_compound_adjusted_team_strength(
    *,
    context: Any,
    team: str,
    race_name: str,
    compound: str,
    cfg: Any,
    should_use_compound_adjustments_fn: Any,
    get_compound_performance_modifier_fn: Any,
) -> float:
    """Apply a compound-specific performance tweak to the blended team score."""
    base_strength = context.get_blended_team_strength(team, race_name)
    team_data = context._resolve_team_data(team)
    compound_chars = team_data.get("compound_characteristics", {})

    min_laps_threshold = cfg.get("baseline_predictor.race.min_laps_for_compound_data", 10)
    if not should_use_compound_adjustments_fn(
        compound_chars,
        min_laps_threshold=min_laps_threshold,
    ):
        return base_strength

    compound_modifier = get_compound_performance_modifier_fn(compound_chars, compound)
    adjusted_strength = float(np.clip(base_strength + compound_modifier, 0.0, 1.0))
    logger.debug(
        "  %s on %s: base=%s + compound=%s = %s",
        team,
        compound,
        format(base_strength, ".3f"),
        format(compound_modifier, "+.3f"),
        format(adjusted_strength, ".3f"),
    )
    return adjusted_strength
