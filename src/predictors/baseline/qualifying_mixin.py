"""Qualifying and sprint-race mixin for Baseline2026Predictor."""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

import numpy as np

from src.types.prediction_types import QualifyingGridEntry
from src.utils import config_loader
from src.utils.fp_blending import blend_team_strength, get_best_fp_performance
from src.utils.lineups import get_lineups
from src.utils.validation_helpers import (
    validate_enum,
    validate_positive_int,
    validate_year,
)
from src.utils.weekend import is_sprint_weekend

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineQualifyingMixin:
    """Shared qualifying and sprint-race methods for Baseline2026Predictor."""

    def _get_testing_profile_weights(
        self, profile: str, defaults: dict[str, float]
    ) -> dict[str, float]:
        """Get configured testing profile weights with safe fallback."""
        cfg = getattr(self, "config", config_loader)
        weights = cfg.get(f"baseline_predictor.race.testing_profile_weights.{profile}", defaults)
        return weights if isinstance(weights, dict) and weights else defaults

    def _resolve_effective_experience_tier(
        self, driver_data: dict[str, Any], prediction_year: int | None
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

    def _extract_experience_total_races(self, driver_data: dict[str, Any]) -> int | None:
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

    def _build_testing_short_run_fallback(
        self,
        lineups: dict[str, list[str]],
        metric_weights: dict[str, float],
    ) -> dict[str, float] | None:
        """Build a team-pace fallback from stored short-run testing profiles."""
        cfg = getattr(self, "config", config_loader)
        min_teams = int(cfg.get("baseline_predictor.qualifying.testing_fallback_min_teams", 8))
        if min_teams < 2:
            min_teams = 2
        short_weight_min = float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_short_weight_min", 0.35)
        )
        short_weight_max = float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_short_weight_max", 0.85)
        )
        divergence_scale = float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_divergence_scale", 1.4)
        )

        team_scores: dict[str, float] = {}

        def _score_profile(profile_metrics: dict[str, float] | None) -> float | None:
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

        for team in lineups:
            short_profile = self._get_testing_characteristics_for_profile(team, "short_run")
            balanced_profile = self._get_testing_characteristics_for_profile(team, "balanced")
            short_score = _score_profile(short_profile)
            balanced_score = _score_profile(balanced_profile)

            if short_score is None and balanced_score is None:
                continue
            if short_score is None:
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

    def _apply_testing_fallback_adjustment(
        self,
        model_strengths: dict[str, float],
        testing_fallback_performance: dict[str, float] | None,
    ) -> dict[str, float]:
        """
        Apply a conservative testing-derived adjustment on top of model strengths.

        Testing programs are noisy (fuel/load/run-plan effects), so we use them as a
        bounded relative nudge instead of treating them as direct pace replacement.
        """
        if not testing_fallback_performance:
            return model_strengths

        cfg = getattr(self, "config", config_loader)
        absolute_blend_weight = float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_absolute_blend_weight", 0.22)
        )
        absolute_blend_weight = float(np.clip(absolute_blend_weight, 0.0, 1.0))
        scale = float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_modifier_scale", 0.06)
        )
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

        values = [float(v) for v in testing_fallback_performance.values() if np.isfinite(float(v))]
        if not values:
            return model_strengths
        field_median = float(np.median(values))

        adjusted = {}
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

    def _get_learned_position_adjustment(
        self,
        *,
        team: str,
        driver: str,
        teammates: list[str],
        session: str = "qualifying",
    ) -> float:
        """Return learned position adjustment from systematic calibration state."""
        calibration_system = getattr(self, "calibration_system", None)
        if calibration_system is None:
            return 0.0

        getter = getattr(calibration_system, "get_combined_position_adjustment", None)
        if not callable(getter):
            return 0.0

        cfg = getattr(self, "config", config_loader)
        min_samples = int(cfg.get("baseline_predictor.learning.min_samples", 1))
        driver_error_scale = float(cfg.get("baseline_predictor.learning.driver_error_scale", 0.18))
        teammate_gap_scale = float(cfg.get("baseline_predictor.learning.teammate_gap_scale", 0.10))
        max_adjustment = float(cfg.get("baseline_predictor.learning.max_adjustment", 2.5))

        try:
            return float(
                getter(
                    team=team,
                    driver=driver,
                    teammates=teammates,
                    session=session,
                    min_samples=max(1, min_samples),
                    driver_error_scale=driver_error_scale,
                    teammate_gap_scale=teammate_gap_scale,
                    max_adjustment=max_adjustment,
                )
            )
        except Exception as exc:
            logger.debug(f"Could not load learned qualifying adjustment for {driver}: {exc}")
            return 0.0

    def _build_driver_list_with_strengths(
        self,
        lineups: dict[str, list[str]],
        fp_performance: dict[str, float] | None,
        testing_fallback_performance: dict[str, float] | None,
        race_name: str,
        is_sprint: bool,
        prediction_year: int | None = None,
    ) -> tuple[list[dict], int]:
        """Build driver list with blended team/driver strengths and testing modifiers."""
        cfg = getattr(self, "config", config_loader)
        all_drivers = []
        model_strengths = {}
        teams_with_short_profile = 0

        short_profile_scale = cfg.get(
            "baseline_predictor.qualifying.testing_short_run_modifier_scale", 0.04
        )
        short_profile_weights = self._get_testing_profile_weights(
            "short_run",
            {
                "overall_pace": 0.55,
                "top_speed": 0.20,
                "medium_corner_performance": 0.15,
                "fast_corner_performance": 0.10,
            },
        )
        fp_blend_weight = cfg.get("baseline_predictor.qualifying.fp_blend_weight", 0.7)
        default_skill = cfg.get("baseline_predictor.qualifying.default_skill", 0.5)
        default_team_strength = cfg.get("baseline_predictor.qualifying.default_team_strength", 0.5)

        for team in lineups:
            model_strength = self.get_blended_team_strength(team, race_name)
            short_modifier, has_short_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="short_run",
                metric_weights=short_profile_weights,
                scale=short_profile_scale,
            )
            model_strength = np.clip(model_strength + short_modifier, 0.0, 1.0)
            if has_short_profile:
                teams_with_short_profile += 1
            model_strengths[team] = model_strength

        if fp_performance is not None:
            blended_strengths = blend_team_strength(
                model_strengths,
                fp_performance,
                blend_weight=fp_blend_weight,
            )
        else:
            blended_strengths = self._apply_testing_fallback_adjustment(
                model_strengths=model_strengths,
                testing_fallback_performance=testing_fallback_performance,
            )

        for team, drivers in lineups.items():
            team_strength = blended_strengths.get(team, default_team_strength)
            for driver_code in drivers:
                driver_data = self.drivers.get(driver_code)
                if not driver_data:
                    fallback_loader = getattr(self, "_get_driver_data_or_fallback", None)
                    if callable(fallback_loader):
                        try:
                            driver_data = fallback_loader(driver_code, team)
                        except ValueError:
                            driver_data = {}
                    else:
                        driver_data = {}
                skill = driver_data.get("racecraft", {}).get("skill_score", default_skill)
                quali_pace = driver_data.get("pace", {}).get("quali_pace", 0.5)
                experience_tier = self._resolve_effective_experience_tier(
                    driver_data=driver_data,
                    prediction_year=prediction_year,
                )
                experience_total_races = self._extract_experience_total_races(driver_data)
                learned_position_adjustment = self._get_learned_position_adjustment(
                    team=team,
                    driver=driver_code,
                    teammates=drivers,
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

    def _run_qualifying_simulations(
        self,
        all_drivers: list[dict],
        n_simulations: int,
        is_sprint: bool,
        has_practice_data: bool,
        rng: np.random.Generator,
    ) -> dict[str, list[int]]:
        """Run Monte Carlo qualifying simulations and return position records."""
        cfg = getattr(self, "config", config_loader)
        position_records = {d["driver"]: [] for d in all_drivers}

        noise_std_sprint = cfg.get("baseline_predictor.qualifying.noise_std_sprint", 0.025)
        noise_std_normal = cfg.get("baseline_predictor.qualifying.noise_std_normal", 0.02)
        noise_std = noise_std_sprint if is_sprint else noise_std_normal

        team_weight = cfg.get("baseline_predictor.qualifying.team_weight", 0.7)
        skill_weight = cfg.get("baseline_predictor.qualifying.skill_weight", 0.3)
        learning_position_to_score_scale = float(
            cfg.get("baseline_predictor.qualifying.learning.position_to_score_scale", 0.03)
        )
        learning_with_practice_multiplier = float(
            cfg.get("baseline_predictor.qualifying.learning.with_practice_multiplier", 0.65)
        )
        effective_learning_scale = (
            learning_position_to_score_scale
            if not has_practice_data
            else (learning_position_to_score_scale * learning_with_practice_multiplier)
        )
        team_strength_compression = cfg.get(
            "baseline_predictor.qualifying.team_strength_compression", 0.50
        )
        driver_quali_pace_weight = cfg.get(
            "baseline_predictor.qualifying.driver_quali_pace_weight", 0.70
        )
        driver_skill_weight = cfg.get("baseline_predictor.qualifying.driver_skill_weight", 0.30)
        driver_weight_sum = driver_quali_pace_weight + driver_skill_weight
        if driver_weight_sum <= 0:
            driver_quali_pace_weight, driver_skill_weight = 0.70, 0.30
            driver_weight_sum = 1.0
        model_only_driver_signal_shrink = cfg.get(
            "baseline_predictor.qualifying.model_only_driver_signal_shrink", 0.35
        )
        model_only_driver_signal_shrink = float(np.clip(model_only_driver_signal_shrink, 0.0, 1.0))
        model_only_experience_shrink = cfg.get(
            "baseline_predictor.qualifying.model_only_experience_shrink",
            {
                "rookie": 0.45,
                "second_year": 0.30,
                "developing": 0.20,
                "sunset": 0.05,
                "unknown": 0.30,
            },
        )
        if not isinstance(model_only_experience_shrink, dict):
            model_only_experience_shrink = {}
        if (
            "second_year" not in model_only_experience_shrink
            and "sophomore" in model_only_experience_shrink
        ):
            model_only_experience_shrink["second_year"] = model_only_experience_shrink["sophomore"]
        team_driver_signal_means: dict[str, float] = {}
        for driver_info in all_drivers:
            driver_signal = (
                (driver_info["quali_pace"] * driver_quali_pace_weight)
                + (driver_info["skill"] * driver_skill_weight)
            ) / driver_weight_sum
            team_driver_signal_means.setdefault(driver_info["team"], 0.0)
            team_driver_signal_means[driver_info["team"]] += driver_signal
        team_counts: dict[str, int] = {}
        for driver_info in all_drivers:
            team_counts[driver_info["team"]] = team_counts.get(driver_info["team"], 0) + 1
        for team_name, total_signal in team_driver_signal_means.items():
            count = team_counts.get(team_name, 1)
            team_driver_signal_means[team_name] = total_signal / count

        driver_offset_cap = cfg.get("baseline_predictor.qualifying.driver_offset_cap", 0.18)
        driver_signal_softness = cfg.get(
            "baseline_predictor.qualifying.driver_signal_softness", 0.20
        )
        if driver_signal_softness <= 0:
            driver_signal_softness = 0.20
        teammate_setup_std = cfg.get("baseline_predictor.qualifying.teammate_setup_std", 0.015)
        model_only_teammate_anchor_scale = cfg.get(
            "baseline_predictor.qualifying.model_only_teammate_anchor_scale", 0.12
        )
        model_only_teammate_anchor_cap = cfg.get(
            "baseline_predictor.qualifying.model_only_teammate_anchor_cap", 0.04
        )
        model_only_anchor_experience_multiplier = cfg.get(
            "baseline_predictor.qualifying.model_only_teammate_anchor_experience_multiplier",
            {
                "rookie": 0.30,
                "second_year": 0.45,
                "developing": 0.55,
                "sunset": 1.00,
                "unknown": 0.45,
            },
        )
        if not isinstance(model_only_anchor_experience_multiplier, dict):
            model_only_anchor_experience_multiplier = {}
        if (
            "second_year" not in model_only_anchor_experience_multiplier
            and "sophomore" in model_only_anchor_experience_multiplier
        ):
            model_only_anchor_experience_multiplier["second_year"] = (
                model_only_anchor_experience_multiplier["sophomore"]
            )
        model_only_teammate_gap_cap_by_tier = cfg.get(
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience",
            {
                "rookie": 0.16,
                "second_year": 0.12,
                "developing": 0.10,
                "unknown": 0.12,
            },
        )
        if not isinstance(model_only_teammate_gap_cap_by_tier, dict):
            model_only_teammate_gap_cap_by_tier = {}
        if (
            "second_year" not in model_only_teammate_gap_cap_by_tier
            and "sophomore" in model_only_teammate_gap_cap_by_tier
        ):
            model_only_teammate_gap_cap_by_tier["second_year"] = (
                model_only_teammate_gap_cap_by_tier["sophomore"]
            )
        model_only_teammate_gap_cap_max_races_by_tier = cfg.get(
            "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience",
            {
                "rookie": 40,
                "second_year": 55,
                "developing": 55,
                "unknown": 45,
            },
        )
        if not isinstance(model_only_teammate_gap_cap_max_races_by_tier, dict):
            model_only_teammate_gap_cap_max_races_by_tier = {}
        if (
            "second_year" not in model_only_teammate_gap_cap_max_races_by_tier
            and "sophomore" in model_only_teammate_gap_cap_max_races_by_tier
        ):
            model_only_teammate_gap_cap_max_races_by_tier["second_year"] = (
                model_only_teammate_gap_cap_max_races_by_tier["sophomore"]
            )
        model_only_teammate_gap_cap_min_scale = float(
            cfg.get("baseline_predictor.qualifying.model_only_teammate_gap_cap_min_scale", 0.35)
        )
        model_only_teammate_gap_cap_min_scale = float(
            np.clip(model_only_teammate_gap_cap_min_scale, 0.0, 1.0)
        )

        def _resolve_model_only_teammate_gap_cap(driver_info: dict[str, Any]) -> float | None:
            experience_tier = str(driver_info.get("experience_tier", "unknown"))
            if experience_tier == "sophomore":
                experience_tier = "second_year"
            cap_value = model_only_teammate_gap_cap_by_tier.get(experience_tier)
            if cap_value is None:
                return None
            try:
                cap = float(cap_value)
            except (TypeError, ValueError):
                return None
            if cap <= 0:
                return None

            max_races_value = model_only_teammate_gap_cap_max_races_by_tier.get(experience_tier)
            if max_races_value is None:
                return cap

            total_races = driver_info.get("experience_total_races")
            if total_races is None:
                return cap

            try:
                total_races_int = int(total_races)
                max_races_int = int(max_races_value)
                if total_races_int > max_races_int:
                    return None
            except (TypeError, ValueError):
                return cap
            if max_races_int <= 0:
                return cap
            sample_ratio = float(np.clip(total_races_int / max_races_int, 0.0, 1.0))
            reliability_scale = model_only_teammate_gap_cap_min_scale + (
                (1.0 - model_only_teammate_gap_cap_min_scale) * sample_ratio
            )
            return cap * reliability_scale

        if not has_practice_data:
            team_weight *= cfg.get(
                "baseline_predictor.qualifying.model_only_team_weight_multiplier", 0.82
            )
            skill_weight *= cfg.get(
                "baseline_predictor.qualifying.model_only_skill_weight_multiplier", 1.35
            )
            total_weight = team_weight + skill_weight
            if total_weight <= 0:
                team_weight, skill_weight = 0.66, 0.34
            else:
                team_weight /= total_weight
                skill_weight /= total_weight

            team_strength_compression *= cfg.get(
                "baseline_predictor.qualifying.model_only_team_compression_multiplier", 0.87
            )
            team_strength_compression = float(np.clip(team_strength_compression, 0.20, 1.0))

            driver_offset_cap *= cfg.get(
                "baseline_predictor.qualifying.model_only_driver_offset_cap_multiplier", 1.33
            )
            driver_offset_cap = float(np.clip(driver_offset_cap, 0.05, 0.30))

            noise_std *= cfg.get("baseline_predictor.qualifying.model_only_noise_multiplier", 1.12)
            teammate_setup_std *= cfg.get(
                "baseline_predictor.qualifying.model_only_teammate_setup_multiplier", 1.10
            )

        weekend_form_std = cfg.get("baseline_predictor.qualifying.weekend_form_std", 0.0)
        if not has_practice_data:
            weekend_form_std *= cfg.get(
                "baseline_predictor.qualifying.model_only_weekend_form_multiplier", 1.0
            )
        weekend_form = {
            d["driver"]: rng.normal(0, weekend_form_std) if weekend_form_std > 0 else 0.0
            for d in all_drivers
        }

        for _ in range(n_simulations):
            driver_scores = []
            for driver_info in all_drivers:
                compressed_team_strength = 0.5 + (
                    (driver_info["team_strength"] - 0.5) * team_strength_compression
                )
                compressed_team_strength = np.clip(compressed_team_strength, 0.0, 1.0)

                raw_driver_signal = (
                    (driver_info["quali_pace"] * driver_quali_pace_weight)
                    + (driver_info["skill"] * driver_skill_weight)
                ) / driver_weight_sum
                driver_signal = raw_driver_signal
                model_only_gap_cap = None
                if not has_practice_data:
                    model_only_gap_cap = _resolve_model_only_teammate_gap_cap(driver_info)
                    if model_only_gap_cap is not None:
                        team_mean_for_cap = team_driver_signal_means.get(
                            driver_info["team"], driver_signal
                        )
                        driver_signal = float(
                            np.clip(
                                driver_signal,
                                team_mean_for_cap - model_only_gap_cap,
                                team_mean_for_cap + model_only_gap_cap,
                            )
                        )
                if not has_practice_data and model_only_driver_signal_shrink > 0:
                    team_mean = team_driver_signal_means.get(driver_info["team"], driver_signal)
                    experience_tier = str(driver_info.get("experience_tier", "unknown"))
                    if experience_tier == "sophomore":
                        experience_tier = "second_year"
                    extra_shrink = model_only_experience_shrink.get(
                        experience_tier, model_only_experience_shrink.get("unknown", 0.0)
                    )
                    negative_delta_threshold = float(
                        cfg.get(
                            "baseline_predictor.qualifying.model_only_negative_delta_threshold",
                            0.08,
                        )
                    )
                    negative_delta_shrink_scale = float(
                        cfg.get(
                            "baseline_predictor.qualifying.model_only_negative_delta_shrink_scale",
                            1.0,
                        )
                    )
                    negative_delta_shrink_cap = float(
                        cfg.get(
                            "baseline_predictor.qualifying.model_only_negative_delta_shrink_cap",
                            0.25,
                        )
                    )
                    delta_from_team = driver_signal - team_mean
                    extra_negative_delta_shrink = 0.0
                    if delta_from_team < -negative_delta_threshold:
                        extra_negative_delta_shrink = min(
                            max(
                                0.0,
                                ((-delta_from_team) - negative_delta_threshold)
                                * negative_delta_shrink_scale,
                            ),
                            max(0.0, negative_delta_shrink_cap),
                        )
                    total_shrink = float(
                        np.clip(
                            model_only_driver_signal_shrink
                            + float(extra_shrink)
                            + float(extra_negative_delta_shrink),
                            0.0,
                            0.95,
                        )
                    )
                    driver_signal = team_mean + ((driver_signal - team_mean) * (1.0 - total_shrink))
                if not has_practice_data:
                    if model_only_gap_cap is not None:
                        team_mean = team_driver_signal_means.get(driver_info["team"], driver_signal)
                        driver_signal = max(driver_signal, team_mean - model_only_gap_cap)
                bounded_driver_signal = 0.5 + (
                    np.tanh((driver_signal - 0.5) / driver_signal_softness) * driver_offset_cap
                )

                score = (compressed_team_strength * team_weight) + (
                    bounded_driver_signal * skill_weight
                )
                learned_position_adjustment = float(
                    driver_info.get("learned_position_adjustment", 0.0)
                )
                score += learned_position_adjustment * effective_learning_scale
                score += weekend_form.get(driver_info["driver"], 0.0)
                if not has_practice_data and model_only_teammate_anchor_scale > 0:
                    team_mean_raw = team_driver_signal_means.get(
                        driver_info["team"], raw_driver_signal
                    )
                    teammate_delta = raw_driver_signal - team_mean_raw
                    if model_only_gap_cap is not None:
                        teammate_delta = float(
                            np.clip(teammate_delta, -model_only_gap_cap, model_only_gap_cap)
                        )
                    anchor_adjustment = np.clip(
                        teammate_delta * model_only_teammate_anchor_scale,
                        -model_only_teammate_anchor_cap,
                        model_only_teammate_anchor_cap,
                    )
                    experience_tier = str(driver_info.get("experience_tier", "unknown"))
                    if experience_tier == "sophomore":
                        experience_tier = "second_year"
                    anchor_tier_multiplier = float(
                        model_only_anchor_experience_multiplier.get(experience_tier, 1.0)
                    )
                    anchor_adjustment *= max(0.0, anchor_tier_multiplier)
                    score += anchor_adjustment
                score += rng.normal(0, teammate_setup_std)
                score += rng.normal(0, noise_std)

                driver_scores.append(
                    {
                        "driver": driver_info["driver"],
                        "team": driver_info["team"],
                        "score": score,
                    }
                )

            driver_scores.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(driver_scores):
                position_records[item["driver"]].append(i + 1)

        return position_records

    def _aggregate_grid_results(
        self, position_records: dict[str, list[int]], all_drivers: list[dict]
    ) -> list[QualifyingGridEntry]:
        """Aggregate simulation results into final grid with confidence intervals."""
        cfg = getattr(self, "config", config_loader)
        grid: list[QualifyingGridEntry] = []
        confidence_std_multiplier = cfg.get(
            "baseline_predictor.qualifying.confidence_std_multiplier", 5.0
        )
        confidence_cap = cfg.get("baseline_predictor.qualifying.confidence_cap", 60)
        confidence_min = cfg.get("baseline_predictor.qualifying.confidence_min", 40)

        for driver_info in all_drivers:
            positions = position_records[driver_info["driver"]]
            median_pos = int(np.median(positions))
            mean_pos = float(np.mean(positions))
            p5 = int(np.percentile(positions, 5))
            p95 = int(np.percentile(positions, 95))

            position_std = np.std(positions)
            confidence = max(
                confidence_min,
                min(confidence_cap, confidence_cap - (position_std * confidence_std_multiplier)),
            )

            grid.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "position": median_pos,
                    "median_position": median_pos,
                    "_mean_position": mean_pos,
                    "p5": p5,
                    "p95": p95,
                    "confidence": float(round(confidence, 1)),
                }
            )

        # Resolve median ties with the underlying simulation mean so teammate order
        # does not collapse into insertion-order blocks when medians are equal.
        grid.sort(key=lambda x: (x["median_position"], x["_mean_position"], x["driver"]))

        for i, item in enumerate(grid):
            item["position"] = i + 1
            item.pop("_mean_position", None)

        return grid

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
    ) -> dict[str, Any]:
        """Predict qualifying with Monte Carlo simulation (sprint/normal weekends)."""
        cfg = getattr(self, "config", config_loader)

        validate_year(year, "year", min_year=2020, max_year=2030)
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        validate_enum(qualifying_stage, "qualifying_stage", ["auto", "sprint", "main"])

        try:
            is_sprint = is_sprint_weekend(year, race_name)
        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Could not determine sprint weekend for {race_name}: {e}")
            is_sprint = False

        seed_material = f"{self.seed}:{year}:{race_name}:{qualifying_stage}:{int(is_sprint)}"
        seed = int(sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)

        lineups = get_lineups(year, race_name)

        session_name, fp_performance, session_laps = get_best_fp_performance(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            qualifying_stage=qualifying_stage,
        )

        if session_laps is not None:
            self._update_compound_characteristics_from_session(
                session_laps, race_name, year, is_sprint
            )

        short_profile_weights = self._get_testing_profile_weights(
            "short_run",
            {
                "overall_pace": 0.55,
                "top_speed": 0.20,
                "medium_corner_performance": 0.15,
                "fast_corner_performance": 0.10,
            },
        )
        testing_fallback_performance = None
        if session_name is None and fp_performance is None:
            testing_fallback_performance = self._build_testing_short_run_fallback(
                lineups=lineups,
                metric_weights=short_profile_weights,
            )
        testing_fallback_used = testing_fallback_performance is not None

        all_drivers, teams_with_short_profile = self._build_driver_list_with_strengths(
            lineups,
            fp_performance,
            testing_fallback_performance,
            race_name,
            is_sprint,
            prediction_year=year,
        )
        if cfg.get("baseline_predictor.qualifying.enable_driver_fp_adjustment", True):
            from src.utils.driver_fp_adjustment import calculate_driver_fp_modifiers

            fp_session_types = ["FP1"] if is_sprint else ["FP1", "FP2", "FP3"]
            modifier_scale = cfg.get(
                "baseline_predictor.qualifying.driver_fp_adjustment_scale", 0.10
            )
            smoothing_seconds = cfg.get(
                "baseline_predictor.qualifying.driver_fp_adjustment_smoothing", 0.50
            )
            driver_fp_modifiers = calculate_driver_fp_modifiers(
                year=year,
                race_name=race_name,
                session_types=fp_session_types,
                scale=modifier_scale,
                smoothing_seconds=smoothing_seconds,
            )
            for driver_info in all_drivers:
                fp_modifier = driver_fp_modifiers.get(driver_info["driver"], 0.0)
                if fp_modifier == 0.0:
                    continue
                driver_info["skill"] = np.clip(driver_info["skill"] + fp_modifier, 0.01, 0.99)

        position_records = self._run_qualifying_simulations(
            all_drivers, n_simulations, is_sprint, session_name is not None, rng
        )

        grid = self._aggregate_grid_results(position_records, all_drivers)

        if session_name is not None:
            data_source = session_name
        elif testing_fallback_used:
            data_source = "Testing short-run profile blend (no weekend practice data)"
        else:
            data_source = "Model-only (no practice/testing data)"

        return {
            "grid": grid,
            "data_source": data_source,
            "blend_used": session_name is not None,
            "testing_fallback_used": testing_fallback_used,
            "qualifying_stage": qualifying_stage,
            "characteristics_profile_used": "short_run",
            "teams_with_characteristics_profile": teams_with_short_profile,
        }

    def predict_sprint_race(
        self,
        sprint_quali_grid: list[dict],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
    ) -> dict[str, Any]:
        """Predict Sprint Race with reduced chaos and increased grid influence."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

        result = self.predict_race(
            qualifying_grid=sprint_quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=True,
        )

        return result
