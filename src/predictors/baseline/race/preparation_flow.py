"""Core flow helpers for race preparation mixin."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from src.types.prediction_types import DriverRaceInfo, QualifyingGridEntry
from src.utils import config_loader
from src.utils.team_mapping import map_team_to_characteristics

_EXPERIENCE_DNF_MODIFIERS = {
    "rookie": 0.05,
    "second_year": 0.03,
    "developing": 0.02,
    "established": 0.00,
    "veteran": -0.01,
    "sunset": -0.005,
}


def _coerce_tire_deg_slope(value: Any, *, default: float) -> float:
    """Return a finite tire degradation slope, falling back to default when missing."""
    try:
        fallback = float(default)
    except (TypeError, ValueError):
        fallback = 0.15

    try:
        slope = float(value)
    except (TypeError, ValueError):
        return fallback

    if not np.isfinite(slope):
        return fallback
    return slope


def _resolve_experience_tier_from_years(years_experience: int) -> str:
    """Map years-of-experience to canonical experience tier."""
    if years_experience <= 0:
        return "rookie"
    if years_experience == 1:
        return "second_year"
    if years_experience <= 3:
        return "developing"
    if years_experience <= 6:
        return "established"
    if years_experience <= 14:
        return "veteran"
    return "sunset"


def resolve_effective_experience_tier_for_race(driver_data: dict, current_year: int) -> str:
    """Resolve experience tier for the current prediction year.

    Reuse the same tier logic as qualifying so both paths stay in sync.
    """
    from src.predictors.baseline.qualifying_preparation import resolve_effective_experience_tier

    return resolve_effective_experience_tier(driver_data, current_year)


def infer_missing_driver_experience_tier(
    driver_code: str,
    *,
    current_year: int,
    load_driver_debut_years_fn: Callable[[], dict[str, int]],
) -> str:
    """Infer tier for missing driver profiles from debut years and current year."""
    debut_years = load_driver_debut_years_fn()
    debut_year = debut_years.get(driver_code)
    if debut_year is None:
        return "rookie"

    years_experience = max(0, current_year - int(debut_year))
    return _resolve_experience_tier_from_years(years_experience)


def _apply_missing_driver_tier_penalties(
    inferred_tier: str,
    *,
    rookie_dnf_penalty: float,
    rookie_quali_penalty: float,
    rookie_race_penalty: float,
    rookie_skill_penalty: float,
    rookie_overtaking_penalty: float,
    second_year_penalty_scale: float,
    quali_pace: float,
    race_pace: float,
    skill_score: float,
    overtaking_skill: float,
    dnf_rate: float,
) -> tuple[float, float, float, float, float]:
    """Apply rookie/second-year penalty profile on top of baseline values."""
    if inferred_tier == "rookie":
        return (
            quali_pace - rookie_quali_penalty,
            race_pace - rookie_race_penalty,
            skill_score - rookie_skill_penalty,
            overtaking_skill - rookie_overtaking_penalty,
            dnf_rate + rookie_dnf_penalty,
        )

    if inferred_tier == "second_year":
        scale = second_year_penalty_scale
        return (
            quali_pace - (rookie_quali_penalty * scale),
            race_pace - (rookie_race_penalty * scale),
            skill_score - (rookie_skill_penalty * scale),
            overtaking_skill - (rookie_overtaking_penalty * scale),
            dnf_rate + (rookie_dnf_penalty * scale),
        )

    return quali_pace, race_pace, skill_score, overtaking_skill, dnf_rate


def build_missing_driver_fallback(
    driver_code: str,
    team: str,
    *,
    config: Any | None,
    infer_missing_driver_experience_tier_fn: Callable[[str], str],
    get_teammate_driver_data_fn: Callable[[str, str], tuple[str, dict] | None],
    logger: logging.Logger,
) -> dict:
    """Build synthetic profile for active-lineup drivers missing characteristics."""
    cfg = config or config_loader
    default_skill = cfg.get("baseline_predictor.qualifying.default_skill", 0.5)
    inferred_tier = infer_missing_driver_experience_tier_fn(driver_code)
    teammate_weight = cfg.get("baseline_predictor.race.missing_driver_teammate_weight", 0.75)
    teammate_weight = float(np.clip(teammate_weight, 0.0, 1.0))
    default_dnf = cfg.get("baseline_predictor.race.missing_driver_default_dnf_rate", 0.10)
    rookie_dnf_penalty = cfg.get("baseline_predictor.race.missing_driver_rookie_dnf_penalty", 0.02)
    rookie_quali_penalty = cfg.get(
        "baseline_predictor.race.missing_driver_rookie_quali_penalty", 0.08
    )
    rookie_race_penalty = cfg.get(
        "baseline_predictor.race.missing_driver_rookie_race_penalty", 0.07
    )
    rookie_skill_penalty = cfg.get(
        "baseline_predictor.race.missing_driver_rookie_skill_penalty", 0.08
    )
    rookie_overtaking_penalty = cfg.get(
        "baseline_predictor.race.missing_driver_rookie_overtaking_penalty", 0.06
    )
    second_year_penalty_scale = float(
        cfg.get(
            "baseline_predictor.race.missing_driver_second_year_penalty_scale",
            cfg.get("baseline_predictor.race.missing_driver_sophomore_penalty_scale", 0.55),
        )
    )
    second_year_penalty_scale = float(np.clip(second_year_penalty_scale, 0.0, 1.0))

    if inferred_tier == "sophomore":
        inferred_tier = "second_year"

    teammate_entry = get_teammate_driver_data_fn(driver_code, team)
    if teammate_entry:
        teammate_code, teammate_data = teammate_entry
        teammate_pace = teammate_data.get("pace", {})
        teammate_racecraft = teammate_data.get("racecraft", {})
        teammate_dnf = teammate_data.get("dnf_risk", {}).get("dnf_rate", default_dnf)

        quali_pace = (teammate_weight * teammate_pace.get("quali_pace", 0.5)) + (
            (1.0 - teammate_weight) * 0.5
        )
        race_pace = (teammate_weight * teammate_pace.get("race_pace", 0.5)) + (
            (1.0 - teammate_weight) * 0.5
        )
        skill_score = (teammate_weight * teammate_racecraft.get("skill_score", default_skill)) + (
            (1.0 - teammate_weight) * default_skill
        )
        overtaking_skill = (
            teammate_weight * teammate_racecraft.get("overtaking_skill", default_skill)
        ) + ((1.0 - teammate_weight) * default_skill)
        dnf_rate = teammate_dnf

        quali_pace, race_pace, skill_score, overtaking_skill, dnf_rate = (
            _apply_missing_driver_tier_penalties(
                inferred_tier,
                rookie_dnf_penalty=rookie_dnf_penalty,
                rookie_quali_penalty=rookie_quali_penalty,
                rookie_race_penalty=rookie_race_penalty,
                rookie_skill_penalty=rookie_skill_penalty,
                rookie_overtaking_penalty=rookie_overtaking_penalty,
                second_year_penalty_scale=second_year_penalty_scale,
                quali_pace=quali_pace,
                race_pace=race_pace,
                skill_score=skill_score,
                overtaking_skill=overtaking_skill,
                dnf_rate=dnf_rate,
            )
        )

        logger.info(
            f"Driver {driver_code} missing characteristics; using teammate-informed fallback from "
            f"{teammate_code} for {team} (tier={inferred_tier})"
        )
        return {
            "pace": {
                "quali_pace": float(np.clip(quali_pace, 0.0, 1.0)),
                "race_pace": float(np.clip(race_pace, 0.0, 1.0)),
            },
            "racecraft": {
                "skill_score": float(np.clip(skill_score, 0.0, 1.0)),
                "overtaking_skill": float(np.clip(overtaking_skill, 0.0, 1.0)),
            },
            "dnf_risk": {
                "dnf_rate": float(np.clip(max(teammate_dnf, dnf_rate), 0.0, 0.35)),
            },
            "experience": {"tier": inferred_tier},
        }

    logger.warning(
        f"Driver {driver_code} missing characteristics; using neutral fallback for {team}"
    )
    neutral_pace = 0.5
    neutral_race = 0.5
    neutral_skill = default_skill
    neutral_overtaking = default_skill
    neutral_dnf = default_dnf
    neutral_pace, neutral_race, neutral_skill, neutral_overtaking, neutral_dnf = (
        _apply_missing_driver_tier_penalties(
            inferred_tier,
            rookie_dnf_penalty=rookie_dnf_penalty,
            rookie_quali_penalty=rookie_quali_penalty,
            rookie_race_penalty=rookie_race_penalty,
            rookie_skill_penalty=rookie_skill_penalty,
            rookie_overtaking_penalty=rookie_overtaking_penalty,
            second_year_penalty_scale=second_year_penalty_scale,
            quali_pace=neutral_pace,
            race_pace=neutral_race,
            skill_score=neutral_skill,
            overtaking_skill=neutral_overtaking,
            dnf_rate=neutral_dnf,
        )
    )
    return {
        "pace": {
            "quali_pace": float(np.clip(neutral_pace, 0.0, 1.0)),
            "race_pace": float(np.clip(neutral_race, 0.0, 1.0)),
        },
        "racecraft": {
            "skill_score": float(np.clip(neutral_skill, 0.0, 1.0)),
            "overtaking_skill": float(np.clip(neutral_overtaking, 0.0, 1.0)),
        },
        "dnf_risk": {"dnf_rate": float(np.clip(neutral_dnf, 0.0, 0.35))},
        "experience": {"tier": inferred_tier},
    }


def _resolve_racecraft_metrics(
    driver_data: dict,
    defensive_skill_weights: dict[str, float],
) -> tuple[float, float, float, float]:
    """Extract skill/race pace metrics from driver characteristics payload."""
    pace_data = driver_data.get("pace", {})
    quali_pace = pace_data.get("quali_pace", 0.5)
    race_pace = pace_data.get("race_pace", 0.5)
    race_advantage = race_pace - quali_pace

    racecraft = driver_data.get("racecraft", {})
    skill = racecraft.get("skill_score", 0.5)
    overtaking_skill = racecraft.get("overtaking_skill", 0.5)
    defensive_skill = racecraft.get("defensive_skill")
    if defensive_skill is None:
        defensive_skill = (
            defensive_skill_weights.get("overtaking_component", 0.65) * overtaking_skill
            + defensive_skill_weights.get("skill_component", 0.35) * skill
        )

    return race_advantage, skill, overtaking_skill, float(np.clip(defensive_skill, 0.0, 1.0))


def _compute_driver_dnf_probability(
    driver_data: dict,
    *,
    team_uncertainty: float,
    dnf_rate_historical_cap: float,
    dnf_rate_final_cap: float,
    dnf_rate_floor: float,
    team_uncertainty_dnf_multiplier: float,
    resolve_effective_experience_tier_for_race_fn: Callable[[dict], str],
) -> float:
    """Compute capped DNF probability with experience, uncertainty, and a safety floor."""
    raw_dnf_rate = driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10)
    try:
        parsed_dnf_rate = float(raw_dnf_rate)
    except (TypeError, ValueError):
        parsed_dnf_rate = 0.10
    if not np.isfinite(parsed_dnf_rate):
        parsed_dnf_rate = 0.10

    dnf_rate = min(max(0.0, parsed_dnf_rate), dnf_rate_historical_cap)
    experience_tier = resolve_effective_experience_tier_for_race_fn(driver_data)
    experience_dnf_modifier = _EXPERIENCE_DNF_MODIFIERS.get(experience_tier, 0.0)

    if team_uncertainty >= 0.40:
        adjusted_dnf = (
            dnf_rate
            + experience_dnf_modifier
            + (team_uncertainty * team_uncertainty_dnf_multiplier)
        )
    else:
        adjusted_dnf = dnf_rate + experience_dnf_modifier

    floor = float(np.clip(dnf_rate_floor, 0.0, dnf_rate_final_cap))
    return max(floor, min(adjusted_dnf, dnf_rate_final_cap))


def _load_preparation_config(
    config: Any | None,
) -> tuple[float, float, float, float, dict, dict, float]:
    """Load shared prep configuration for race info builders."""
    cfg = config or config_loader
    dnf_rate_historical_cap = cfg.get("baseline_predictor.race.dnf_rate_historical_cap", 0.20)
    dnf_rate_final_cap = cfg.get("baseline_predictor.race.dnf_rate_final_cap", 0.35)
    dnf_rate_floor = cfg.get("baseline_predictor.race.dnf_rate_floor", 0.02)
    long_profile_scale = cfg.get("baseline_predictor.race.testing_long_run_modifier_scale", 0.05)
    long_profile_weights = cfg.get(
        "baseline_predictor.race.testing_profile_weights.long_run",
        {
            "overall_pace": 0.50,
            "tire_deg_performance": 0.35,
            "consistency": 0.15,
        },
    )
    defensive_skill_weights = cfg.get(
        "baseline_predictor.race.defensive_skill_weights",
        {
            "overtaking_component": 0.65,
            "skill_component": 0.35,
        },
    )
    team_uncertainty_dnf_multiplier = cfg.get(
        "baseline_predictor.race.team_uncertainty_dnf_multiplier", 0.20
    )
    return (
        dnf_rate_historical_cap,
        dnf_rate_final_cap,
        dnf_rate_floor,
        long_profile_scale,
        long_profile_weights,
        defensive_skill_weights,
        team_uncertainty_dnf_multiplier,
    )


def _resolve_team_payload(teams: dict[str, dict], team: str) -> dict:
    """Resolve a team payload with alias-aware lookup before returning defaults."""
    team_payload = teams.get(team)
    if isinstance(team_payload, dict):
        return team_payload

    known_teams = set(teams.keys())
    mapped_team = map_team_to_characteristics(team, known_teams=known_teams)
    if not isinstance(mapped_team, str) or not mapped_team:
        return {}

    mapped_payload = teams.get(mapped_team)
    return mapped_payload if isinstance(mapped_payload, dict) else {}


def prepare_driver_info_core(
    qualifying_grid: list[QualifyingGridEntry],
    race_name: str | None,
    race_compound: str,
    *,
    teams: dict[str, dict],
    config: Any | None,
    get_compound_adjusted_team_strength_fn: Callable[[str, str, str], float],
    compute_testing_profile_modifier_fn: Callable[[str, str, dict, float], tuple[float, bool]],
    get_driver_data_or_fallback_fn: Callable[[str, str], dict],
    resolve_effective_experience_tier_for_race_fn: Callable[[dict], str],
) -> tuple[dict[str, DriverRaceInfo], int]:
    """Build driver info map with team strength, profile modifiers, and DNF probabilities."""
    (
        dnf_rate_historical_cap,
        dnf_rate_final_cap,
        dnf_rate_floor,
        long_profile_scale,
        long_profile_weights,
        defensive_skill_weights,
        team_uncertainty_dnf_multiplier,
    ) = _load_preparation_config(config)

    driver_info_map: dict[str, DriverRaceInfo] = {}
    teams_with_long_profile: set[str] = set()

    for entry in qualifying_grid:
        driver_code = entry["driver"]
        team = entry["team"]
        grid_pos = entry["position"]
        team_payload = _resolve_team_payload(teams, team)

        if race_name:
            team_strength = get_compound_adjusted_team_strength_fn(team, race_name, race_compound)
        else:
            team_strength = team_payload.get("overall_performance", 0.50)

        long_modifier, has_long_profile = compute_testing_profile_modifier_fn(
            team,
            "long_run",
            long_profile_weights,
            long_profile_scale,
        )
        team_strength = np.clip(team_strength + long_modifier, 0.0, 1.0)
        if has_long_profile:
            teams_with_long_profile.add(team)

        driver_data = get_driver_data_or_fallback_fn(driver_code, team)
        race_advantage, skill, overtaking_skill, defensive_skill = _resolve_racecraft_metrics(
            driver_data=driver_data,
            defensive_skill_weights=defensive_skill_weights,
        )

        team_uncertainty = team_payload.get("uncertainty", 0.30)
        dnf_probability = _compute_driver_dnf_probability(
            driver_data,
            team_uncertainty=team_uncertainty,
            dnf_rate_historical_cap=dnf_rate_historical_cap,
            dnf_rate_final_cap=dnf_rate_final_cap,
            dnf_rate_floor=dnf_rate_floor,
            team_uncertainty_dnf_multiplier=team_uncertainty_dnf_multiplier,
            resolve_effective_experience_tier_for_race_fn=resolve_effective_experience_tier_for_race_fn,
        )

        driver_info_map[driver_code] = {
            "driver": driver_code,
            "team": team,
            "grid_pos": grid_pos,
            "team_strength": team_strength,
            "skill": skill,
            "race_advantage": race_advantage,
            "overtaking_skill": overtaking_skill,
            "defensive_skill": defensive_skill,
            "dnf_probability": dnf_probability,
        }

    return driver_info_map, len(teams_with_long_profile)


def prepare_driver_info_with_compounds_core(
    qualifying_grid: list[QualifyingGridEntry],
    race_name: str | None,
    *,
    teams: dict[str, dict],
    config: Any | None,
    get_blended_team_strength_fn: Callable[[str, str], float],
    compute_testing_profile_modifier_fn: Callable[[str, str, dict, float], tuple[float, bool]],
    get_driver_data_or_fallback_fn: Callable[[str, str], dict],
    resolve_effective_experience_tier_for_race_fn: Callable[[dict], str],
    get_compound_performance_modifier_fn: Callable[[dict[str, dict], str], float],
) -> tuple[dict[str, DriverRaceInfo], int]:
    """Build driver info map with per-compound team strengths for lap-by-lap simulation."""
    (
        dnf_rate_historical_cap,
        dnf_rate_final_cap,
        dnf_rate_floor,
        long_profile_scale,
        long_profile_weights,
        defensive_skill_weights,
        team_uncertainty_dnf_multiplier,
    ) = _load_preparation_config(config)
    cfg = config or config_loader
    default_tire_deg_slope = _coerce_tire_deg_slope(
        cfg.get("baseline_predictor.race.tire_physics.default_deg_slope", 0.15),
        default=0.15,
    )

    driver_info_map: dict[str, DriverRaceInfo] = {}
    teams_with_long_profile: set[str] = set()

    for entry in qualifying_grid:
        driver_code = entry["driver"]
        team = entry["team"]
        grid_pos = entry["position"]
        team_payload = _resolve_team_payload(teams, team)

        if race_name:
            base_team_strength = get_blended_team_strength_fn(team, race_name)
        else:
            base_team_strength = team_payload.get("overall_performance", 0.50)

        long_modifier, has_long_profile = compute_testing_profile_modifier_fn(
            team,
            "long_run",
            long_profile_weights,
            long_profile_scale,
        )
        base_team_strength = np.clip(base_team_strength + long_modifier, 0.0, 1.0)
        if has_long_profile:
            teams_with_long_profile.add(team)

        team_compound_chars = team_payload.get("compound_characteristics", {})
        team_strength_by_compound: dict[str, float] = {}
        tire_deg_by_compound: dict[str, float] = {}
        for compound in ("SOFT", "MEDIUM", "HARD"):
            if compound in team_compound_chars:
                modifier = get_compound_performance_modifier_fn(team_compound_chars, compound)
                adjusted_strength = base_team_strength + modifier
                tire_deg_slope = _coerce_tire_deg_slope(
                    team_compound_chars[compound].get("tire_deg_slope"),
                    default=default_tire_deg_slope,
                )
            else:
                adjusted_strength = base_team_strength
                tire_deg_slope = default_tire_deg_slope

            team_strength_by_compound[compound] = np.clip(adjusted_strength, 0.0, 1.0)
            tire_deg_by_compound[compound] = float(tire_deg_slope)

        driver_data = get_driver_data_or_fallback_fn(driver_code, team)
        race_advantage, skill, overtaking_skill, defensive_skill = _resolve_racecraft_metrics(
            driver_data=driver_data,
            defensive_skill_weights=defensive_skill_weights,
        )
        team_uncertainty = team_payload.get("uncertainty", 0.30)
        dnf_probability = _compute_driver_dnf_probability(
            driver_data,
            team_uncertainty=team_uncertainty,
            dnf_rate_historical_cap=dnf_rate_historical_cap,
            dnf_rate_final_cap=dnf_rate_final_cap,
            dnf_rate_floor=dnf_rate_floor,
            team_uncertainty_dnf_multiplier=team_uncertainty_dnf_multiplier,
            resolve_effective_experience_tier_for_race_fn=resolve_effective_experience_tier_for_race_fn,
        )

        driver_info_map[driver_code] = {
            "driver": driver_code,
            "team": team,
            "grid_pos": grid_pos,
            "team_strength": base_team_strength,
            "team_strength_by_compound": team_strength_by_compound,
            "tire_deg_by_compound": tire_deg_by_compound,
            "skill": skill,
            "race_advantage": race_advantage,
            "overtaking_skill": overtaking_skill,
            "defensive_skill": defensive_skill,
            "dnf_probability": dnf_probability,
        }

    return driver_info_map, len(teams_with_long_profile)
