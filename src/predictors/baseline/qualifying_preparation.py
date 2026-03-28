"""Preparation helpers for qualifying driver strength assembly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def resolve_effective_experience_tier(
    driver_data: dict[str, Any],
    prediction_year: int | None,
) -> str:
    """Resolve experience tier at prediction time to avoid stale preseason labels."""
    experience = driver_data.get("experience", {}) if isinstance(driver_data, dict) else {}
    stored_tier = str(experience.get("tier", "unknown"))
    if stored_tier == "sophomore":
        stored_tier = "second_year"
    if prediction_year is None:
        return stored_tier

    stored_years = experience.get("years_of_experience", 0)
    debut_year = experience.get("debut_year")
    try:
        effective_years = int(stored_years)
    except (TypeError, ValueError):
        effective_years = None

    try:
        debut_year_int = int(debut_year) if debut_year is not None else None
    except (TypeError, ValueError):
        debut_year_int = None

    if debut_year_int is not None and prediction_year >= debut_year_int:
        computed_years = prediction_year - debut_year_int
        if effective_years is None:
            effective_years = computed_years
        else:
            effective_years = max(effective_years, computed_years)

    if effective_years is None:
        return stored_tier

    if effective_years <= 0:
        return "rookie"
    if effective_years == 1:
        return "second_year"
    if effective_years <= 3:
        return "developing"
    if effective_years <= 6:
        return "established"
    if effective_years <= 14:
        return "veteran"
    return "sunset"


def extract_experience_total_races(driver_data: dict[str, Any]) -> int | None:
    """Extract total races from driver profile when available."""
    experience = driver_data.get("experience", {}) if isinstance(driver_data, dict) else {}
    total_races = experience.get("total_races")
    try:
        parsed = int(total_races)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _score_profile(
    profile_metrics: dict[str, float] | None,
    metric_weights: dict[str, float],
) -> float | None:
    if not profile_metrics:
        return None

    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in metric_weights.items():
        try:
            metric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if metric_weight <= 0:
            continue
        value = profile_metrics.get(metric_name)
        if value is None:
            continue
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(metric_value):
            continue
        metric_value = float(np.clip(metric_value, 0.0, 1.0))
        weighted_sum += metric_value * metric_weight
        total_weight += metric_weight
    if total_weight <= 0:
        return None
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))


def build_testing_short_run_fallback(
    *,
    lineups: dict[str, list[str]],
    metric_weights: dict[str, float],
    cfg: Any,
    get_testing_characteristics_for_profile: Callable[[str, str], dict[str, float] | None],
    checkpoint_session_name: str | None = None,
    qualifying_stage: str = "auto",
) -> dict[str, float] | None:
    """Build a qualifying fallback from stored short-run testing profiles.

    Qualifying is mostly about single-lap bite: tire warm-up, rotation on low
    fuel, and how much peak grip the car can unlock in a short window. That is
    why the fallback leans on the stored short-run profile first and only uses a
    balanced profile as a stabilizer when the short-run signal looks noisy.
    Race-style long-run behavior still matters later on Sunday, but it is the
    wrong thing to anchor a one-lap prediction to.
    """
    min_teams = int(cfg.get("baseline_predictor.qualifying.testing_fallback_min_teams", 8))
    if min_teams < 2:
        min_teams = 2
    short_weight_min = float(
        cfg.get("baseline_predictor.qualifying.testing_fallback_short_weight_min", 0.35)
    )
    short_weight_max = float(
        cfg.get("baseline_predictor.qualifying.testing_fallback_short_weight_max", 0.85)
    )
    checkpoint_name = str(checkpoint_session_name or "").strip().upper()
    stage_name = str(qualifying_stage or "auto").strip().lower()
    if checkpoint_name == "SPRINT" and stage_name == "main":
        # After the sprint race, the balanced checkpoint profile has absorbed race-style signal.
        # Main qualifying still targets one-lap pace, so lean fully on the short-run profile.
        after_sprint_short_weight = cfg.get(
            "baseline_predictor.qualifying.testing_fallback_after_sprint_main_short_weight",
        )
        if after_sprint_short_weight is not None:
            pure_short_weight = float(np.clip(after_sprint_short_weight, 0.0, 1.0))
            short_weight_min = pure_short_weight
            short_weight_max = pure_short_weight
    divergence_scale = float(
        cfg.get("baseline_predictor.qualifying.testing_fallback_divergence_scale", 1.4)
    )

    team_scores: dict[str, float] = {}
    for team in lineups:
        short_profile = get_testing_characteristics_for_profile(team, "short_run")
        balanced_profile = get_testing_characteristics_for_profile(team, "balanced")
        short_score = _score_profile(short_profile, metric_weights=metric_weights)
        balanced_score = _score_profile(balanced_profile, metric_weights=metric_weights)

        if short_score is None and balanced_score is None:
            continue
        if short_score is None:
            if balanced_score is None:
                continue
            team_scores[team] = float(balanced_score)
            continue
        if balanced_score is None:
            team_scores[team] = float(short_score)
            continue

        divergence = abs(float(short_score) - float(balanced_score))
        short_weight = float(
            np.clip(
                1.0 - (divergence * divergence_scale),
                min(short_weight_min, short_weight_max),
                max(short_weight_min, short_weight_max),
            )
        )
        blended_score = (short_weight * float(short_score)) + (
            (1.0 - short_weight) * float(balanced_score)
        )
        team_scores[team] = float(np.clip(blended_score, 0.0, 1.0))

    if len(team_scores) < min_teams:
        return None

    return team_scores


def apply_testing_fallback_adjustment(
    *,
    model_strengths: dict[str, float],
    testing_fallback_performance: dict[str, float] | None,
    cfg: Any,
    practice_like_profile_label: str | None = None,
    reference_blend_weight: float | None = None,
) -> dict[str, float]:
    """
    Apply a conservative testing-derived adjustment on top of model strengths.

    Testing programs are noisy (fuel/load/run-plan effects), so we use them as a
    bounded relative nudge instead of treating them as direct pace replacement.
    """
    if not testing_fallback_performance:
        return model_strengths

    absolute_blend_weight = float(
        cfg.get("baseline_predictor.qualifying.testing_fallback_absolute_blend_weight", 0.22)
    )
    absolute_blend_weight = float(np.clip(absolute_blend_weight, 0.0, 1.0))
    scale = float(cfg.get("baseline_predictor.qualifying.testing_fallback_modifier_scale", 0.06))
    clip_range = cfg.get(
        "baseline_predictor.qualifying.testing_fallback_modifier_clip_range", [-0.03, 0.03]
    )
    if (
        isinstance(clip_range, list)
        and len(clip_range) == 2
        and float(clip_range[0]) < float(clip_range[1])
    ):
        min_clip, max_clip = float(clip_range[0]), float(clip_range[1])
    else:
        min_clip, max_clip = -0.03, 0.03

    normalized_profile_label = str(practice_like_profile_label or "").strip().lower()
    uses_weekend_checkpoint_profiles = normalized_profile_label in {
        "fp1",
        "fp2",
        "fp3",
        "sprint qualifying",
        "sprint pace signal",
    }
    if uses_weekend_checkpoint_profiles and reference_blend_weight is not None:
        reference_weight = float(np.clip(reference_blend_weight, 0.0, 1.0))
        absolute_blend_weight = max(
            absolute_blend_weight,
            reference_weight
            * float(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_checkpoint_blend_weight_scale",
                    0.65,
                )
            ),
        )
        absolute_blend_weight = float(
            np.clip(
                absolute_blend_weight,
                0.0,
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_checkpoint_blend_weight_cap",
                    0.55,
                ),
            )
        )

        scale = max(
            float(scale),
            reference_weight
            * float(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_checkpoint_modifier_scale",
                    0.10,
                )
            ),
        )
        scale = float(
            np.clip(
                scale,
                0.0,
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_checkpoint_modifier_cap",
                    0.08,
                ),
            )
        )

        clip_scale = float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_checkpoint_clip_scale",
                0.055,
            )
        )
        clip_cap = float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_checkpoint_clip_cap",
                0.045,
            )
        )
        dynamic_clip = float(np.clip(reference_weight * clip_scale, 0.0, clip_cap))
        if dynamic_clip > 0.0:
            min_clip = min(min_clip, -dynamic_clip)
            max_clip = max(max_clip, dynamic_clip)

    values = [float(v) for v in testing_fallback_performance.values() if np.isfinite(float(v))]
    if not values:
        return model_strengths
    field_median = float(np.median(values))

    adjusted: dict[str, float] = {}
    for team, model_score in model_strengths.items():
        fallback_score = testing_fallback_performance.get(team)
        if fallback_score is None:
            adjusted[team] = model_score
            continue

        blended_base = ((1.0 - absolute_blend_weight) * float(model_score)) + (
            absolute_blend_weight * float(fallback_score)
        )
        centered = float(fallback_score) - field_median
        modifier = float(np.clip(centered * scale, min_clip, max_clip))
        adjusted[team] = float(np.clip(blended_base + modifier, 0.0, 1.0))

    return adjusted


def build_driver_list_with_strengths_core(
    *,
    lineups: dict[str, list[str]],
    fp_performance: dict[str, float] | None,
    testing_fallback_performance: dict[str, float] | None,
    practice_like_profile_label: str | None,
    practice_like_blend_weight: float | None,
    race_name: str,
    prediction_year: int | None,
    drivers: dict[str, dict[str, Any]],
    cfg: Any,
    short_profile_weights: dict[str, float],
    fp_blend_weight: float,
    get_blended_team_strength_fn: Callable[[str, str], float],
    compute_testing_profile_modifier_fn: Callable[..., tuple[float, bool]],
    blend_team_strength_fn: Callable[..., dict[str, float]],
    apply_testing_fallback_adjustment_fn: Callable[..., dict[str, float]],
    resolve_effective_experience_tier_fn: Callable[[dict[str, Any], int | None], str],
    extract_experience_total_races_fn: Callable[[dict[str, Any]], int | None],
    get_learned_position_adjustment_fn: Callable[..., float],
    get_checkpoint_driver_delta_seconds_fn: Callable[[str, str], float | None] | None = None,
    get_driver_data_or_fallback_fn: Callable[[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Build driver list with blended team/driver strengths and testing modifiers."""
    all_drivers: list[dict[str, Any]] = []
    model_strengths: dict[str, float] = {}
    teams_with_short_profile = 0

    uses_checkpoint_practice_profiles = (
        practice_like_profile_label is not None
        and practice_like_blend_weight is not None
        and testing_fallback_performance is not None
    )
    checkpoint_practice_blend_weight: float | None = None
    checkpoint_testing_fallback_performance: dict[str, float] | None = None
    if uses_checkpoint_practice_profiles:
        assert practice_like_blend_weight is not None
        assert testing_fallback_performance is not None
        checkpoint_practice_blend_weight = float(np.clip(practice_like_blend_weight, 0.0, 1.0))
        checkpoint_testing_fallback_performance = testing_fallback_performance
    short_profile_scale = cfg.get(
        "baseline_predictor.qualifying.testing_short_run_modifier_scale", 0.04
    )
    default_skill = cfg.get("baseline_predictor.qualifying.default_skill", 0.5)
    default_team_strength = cfg.get("baseline_predictor.qualifying.default_team_strength", 0.5)

    for team in lineups:
        model_strength = get_blended_team_strength_fn(team, race_name)
        short_modifier, has_short_profile = compute_testing_profile_modifier_fn(
            team=team,
            profile="short_run",
            metric_weights=short_profile_weights,
            scale=short_profile_scale,
        )
        if not uses_checkpoint_practice_profiles:
            model_strength = np.clip(model_strength + short_modifier, 0.0, 1.0)
        if has_short_profile:
            teams_with_short_profile += 1
        model_strengths[team] = model_strength

    if fp_performance is not None:
        blended_strengths = blend_team_strength_fn(
            model_strengths,
            fp_performance,
            blend_weight=fp_blend_weight,
        )
    elif uses_checkpoint_practice_profiles:
        assert checkpoint_practice_blend_weight is not None
        assert checkpoint_testing_fallback_performance is not None
        blended_strengths = blend_team_strength_fn(
            model_strengths,
            checkpoint_testing_fallback_performance,
            blend_weight=checkpoint_practice_blend_weight,
        )
    else:
        blended_strengths = apply_testing_fallback_adjustment_fn(
            model_strengths=model_strengths,
            testing_fallback_performance=testing_fallback_performance,
            practice_like_profile_label=practice_like_profile_label,
            reference_blend_weight=practice_like_blend_weight,
        )

    for team, team_drivers in lineups.items():
        team_strength = blended_strengths.get(team, default_team_strength)
        for driver_code in team_drivers:
            driver_data = drivers.get(driver_code)
            if not driver_data:
                if callable(get_driver_data_or_fallback_fn):
                    try:
                        driver_data = get_driver_data_or_fallback_fn(driver_code, team)
                    except ValueError:
                        driver_data = {}
                else:
                    driver_data = {}
            skill = driver_data.get("racecraft", {}).get("skill_score", default_skill)
            quali_pace = driver_data.get("pace", {}).get("quali_pace", 0.5)
            if uses_checkpoint_practice_profiles and callable(
                get_checkpoint_driver_delta_seconds_fn
            ):
                checkpoint_driver_delta_seconds = get_checkpoint_driver_delta_seconds_fn(
                    team,
                    driver_code,
                )
                if checkpoint_driver_delta_seconds is not None:
                    assert checkpoint_practice_blend_weight is not None
                    smoothing_seconds = float(
                        cfg.get(
                            "baseline_predictor.qualifying.checkpoint_driver_profile_smoothing_seconds",
                            0.35,
                        )
                    )
                    smoothing_seconds = max(1e-6, smoothing_seconds)
                    normalized_checkpoint_delta = float(
                        np.clip(
                            -float(checkpoint_driver_delta_seconds) / smoothing_seconds,
                            -1.0,
                            1.0,
                        )
                    )
                    checkpoint_weight = checkpoint_practice_blend_weight
                    quali_scale = float(
                        cfg.get(
                            "baseline_predictor.qualifying.checkpoint_driver_profile_quali_scale",
                            0.10,
                        )
                    )
                    skill_scale = float(
                        cfg.get(
                            "baseline_predictor.qualifying.checkpoint_driver_profile_skill_scale",
                            0.02,
                        )
                    )
                    quali_pace = float(
                        np.clip(
                            quali_pace
                            + (normalized_checkpoint_delta * quali_scale * checkpoint_weight),
                            0.01,
                            0.99,
                        )
                    )
                    skill = float(
                        np.clip(
                            skill + (normalized_checkpoint_delta * skill_scale * checkpoint_weight),
                            0.01,
                            0.99,
                        )
                    )
            experience_tier = resolve_effective_experience_tier_fn(driver_data, prediction_year)
            experience_total_races = extract_experience_total_races_fn(driver_data)
            learned_position_adjustment = get_learned_position_adjustment_fn(
                team=team,
                driver=driver_code,
                teammates=team_drivers,
                session="qualifying",
            )

            all_drivers.append(
                {
                    "driver": driver_code,
                    "team": team,
                    "team_strength": team_strength,
                    "skill": skill,
                    "quali_pace": quali_pace,
                    "experience_tier": experience_tier,
                    "experience_total_races": experience_total_races,
                    "learned_position_adjustment": learned_position_adjustment,
                }
            )

    return all_drivers, teams_with_short_profile
