"""Monte Carlo simulation helpers for qualifying predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def run_qualifying_simulations(
    *,
    all_drivers: list[dict],
    n_simulations: int,
    is_sprint: bool,
    has_practice_data: bool,
    rng: np.random.Generator,
    cfg: Any,
    logger: Any,
    has_testing_fallback_data: bool = False,
) -> dict[str, list[int]]:
    """Run Monte Carlo qualifying simulations and return position records.

    Testing fallback data provides team-level signal even when weekend practice
    laps are unavailable. In that case, avoid the strict model-only teammate
    regularization stack and inject a small driver-specific weekend-form spread
    so early-season fallback runs do not collapse into rigid team blocks.
    """
    position_records = {d["driver"]: [] for d in all_drivers}
    # Testing fallback provides team-level pace guidance but no direct driver-level
    # weekend telemetry. Keep low-data weighting/noise behavior, but avoid full
    # teammate regularization when fallback is available.
    use_model_only_profile = not has_practice_data
    apply_model_only_teammate_regularization = use_model_only_profile and (
        not has_testing_fallback_data
    )

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
    driver_signal_softness = cfg.get("baseline_predictor.qualifying.driver_signal_softness", 0.20)
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
        model_only_teammate_gap_cap_by_tier["second_year"] = model_only_teammate_gap_cap_by_tier[
            "sophomore"
        ]
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

    if use_model_only_profile:
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

    if has_testing_fallback_data and not has_practice_data:
        # Testing fallback has team-level signal but weaker direct driver calibration.
        # Shift a bit of weight from team to driver signal and add controlled dispersion.
        team_weight *= cfg.get(
            "baseline_predictor.qualifying.testing_fallback_team_weight_multiplier",
            0.78,
        )
        skill_weight *= cfg.get(
            "baseline_predictor.qualifying.testing_fallback_skill_weight_multiplier",
            1.35,
        )
        total_weight = team_weight + skill_weight
        if total_weight <= 0:
            team_weight, skill_weight = 0.52, 0.48
        else:
            team_weight /= total_weight
            skill_weight /= total_weight

        noise_std *= cfg.get(
            "baseline_predictor.qualifying.testing_fallback_noise_multiplier", 1.35
        )
        teammate_setup_std *= cfg.get(
            "baseline_predictor.qualifying.testing_fallback_teammate_setup_multiplier",
            1.25,
        )

    weekend_form_std = cfg.get("baseline_predictor.qualifying.weekend_form_std", 0.0)
    if use_model_only_profile:
        weekend_form_std *= cfg.get(
            "baseline_predictor.qualifying.model_only_weekend_form_multiplier", 1.0
        )
    if has_testing_fallback_data and not has_practice_data:
        weekend_form_floor = float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_weekend_form_std_floor",
                0.028,
            )
        )
        weekend_form_std = max(float(weekend_form_std), weekend_form_floor)
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
            if apply_model_only_teammate_regularization:
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
            if apply_model_only_teammate_regularization and model_only_driver_signal_shrink > 0:
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
            if apply_model_only_teammate_regularization:
                if model_only_gap_cap is not None:
                    team_mean = team_driver_signal_means.get(driver_info["team"], driver_signal)
                    driver_signal = max(driver_signal, team_mean - model_only_gap_cap)
            bounded_driver_signal = 0.5 + (
                np.tanh((driver_signal - 0.5) / driver_signal_softness) * driver_offset_cap
            )

            score = (compressed_team_strength * team_weight) + (
                bounded_driver_signal * skill_weight
            )
            learned_position_adjustment = float(driver_info.get("learned_position_adjustment", 0.0))
            score += learned_position_adjustment * effective_learning_scale
            score += weekend_form.get(driver_info["driver"], 0.0)
            if apply_model_only_teammate_regularization and model_only_teammate_anchor_scale > 0:
                team_mean_raw = team_driver_signal_means.get(driver_info["team"], raw_driver_signal)
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
