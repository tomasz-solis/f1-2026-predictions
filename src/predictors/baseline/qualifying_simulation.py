"""Monte Carlo simulation helpers for qualifying predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegularizationConfig:
    """Extra teammate controls for a qualifying run."""

    driver_signal_shrink: float
    experience_shrink: dict[str, float]
    teammate_gap_cap_by_tier: dict[str, float]
    teammate_gap_cap_max_races_by_tier: dict[str, float]
    teammate_gap_cap_min_scale: float
    teammate_anchor_scale: float
    teammate_anchor_cap: float
    anchor_experience_multiplier: dict[str, float]
    negative_delta_threshold: float
    negative_delta_shrink_scale: float
    negative_delta_shrink_cap: float
    max_total_shrink: float


@dataclass(frozen=True)
class QualiSimConfig:
    """Resolved settings for one qualifying run."""

    noise_std: float
    team_weight: float
    skill_weight: float
    team_strength_compression: float
    driver_offset_cap: float
    driver_signal_softness: float
    driver_quali_pace_weight: float
    driver_skill_weight: float
    effective_learning_scale: float
    weekend_form_std: float
    teammate_setup_std: float
    recent_form_scale: float
    recent_form_cap: float

    apply_regularization: bool
    apply_recent_form_adjustment: bool
    regularization: RegularizationConfig | None


def _normalize_experience_tier(raw_tier: object) -> str:
    """Normalize experience-tier aliases used across config and driver profiles."""
    normalized = str(raw_tier or "unknown").strip().lower()
    return "second_year" if normalized == "sophomore" else normalized


def _normalized_tier_mapping(raw_mapping: object) -> dict[str, float]:
    """Return an experience-tier mapping with numeric values and alias normalization."""
    if not isinstance(raw_mapping, dict):
        return {}

    normalized_mapping: dict[str, float] = {}
    for raw_key, raw_value in raw_mapping.items():
        tier = _normalize_experience_tier(raw_key)
        try:
            normalized_mapping[tier] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return normalized_mapping


def _resolve_teammate_gap_cap(
    *,
    driver_info: dict[str, Any],
    cap_by_tier: dict[str, float],
    max_races_by_tier: dict[str, float],
    min_scale: float,
) -> float | None:
    """Resolve dynamic teammate gap cap for a driver based on experience and sample size."""
    experience_tier = _normalize_experience_tier(driver_info.get("experience_tier", "unknown"))
    cap = cap_by_tier.get(experience_tier)
    if cap is None or cap <= 0:
        return None

    max_races = max_races_by_tier.get(experience_tier)
    if max_races is None:
        return cap

    total_races = driver_info.get("experience_total_races")
    if total_races is None:
        return cap

    try:
        total_races_int = int(total_races)
        max_races_int = int(max_races)
        if total_races_int > max_races_int:
            return None
    except (TypeError, ValueError):
        return cap

    if max_races_int <= 0:
        return cap

    sample_ratio = float(np.clip(total_races_int / max_races_int, 0.0, 1.0))
    reliability_scale = min_scale + ((1.0 - min_scale) * sample_ratio)
    return cap * reliability_scale


def _clip_driver_signal_by_team_gap(
    *,
    driver_signal: float,
    team_mean: float,
    gap_cap: float | None,
) -> float:
    """Clip driver signal around team mean when a teammate-gap cap is configured."""
    if gap_cap is None:
        return driver_signal
    return float(np.clip(driver_signal, team_mean - gap_cap, team_mean + gap_cap))


def _fallback_experience_multiplier(
    mapping: dict[str, float],
    experience_tier: str,
    *,
    default: float,
) -> float:
    """Read an experience-tier multiplier with normalized tier alias handling."""
    tier = _normalize_experience_tier(experience_tier)
    return float(mapping.get(tier, default))


def _rebalance_component_weights(
    *,
    team_weight: float,
    skill_weight: float,
    fallback_team_weight: float,
    fallback_skill_weight: float,
) -> tuple[float, float]:
    """Renormalize team and driver weights after multipliers."""
    total_weight = team_weight + skill_weight
    if total_weight <= 0:
        return fallback_team_weight, fallback_skill_weight
    return team_weight / total_weight, skill_weight / total_weight


def _recent_form_adjustment(
    *,
    driver_info: dict[str, Any],
    driver_signal: float,
    recent_form_scale: float,
    recent_form_cap: float,
) -> float:
    """Convert Bayesian form drift into a small bounded qualifying adjustment."""
    bayesian_skill_score = driver_info.get("bayesian_skill_score")
    if bayesian_skill_score is None:
        return 0.0
    try:
        bayesian_skill_value = float(bayesian_skill_score)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(bayesian_skill_value):
        return 0.0

    if recent_form_scale <= 0 or recent_form_cap <= 0:
        return 0.0

    form_gap = bayesian_skill_value - float(driver_signal)
    return float(np.clip(form_gap * recent_form_scale, -recent_form_cap, recent_form_cap))


def _compute_regularized_driver_signal(
    *,
    driver_info: dict[str, Any],
    raw_driver_signal: float,
    team_driver_signal_means: dict[str, float],
    sim_cfg: QualiSimConfig,
) -> tuple[float, float | None]:
    """Apply the active teammate-regularization regime to a driver signal."""
    regularization = sim_cfg.regularization
    driver_signal = raw_driver_signal
    gap_cap: float | None = None
    team_name = str(driver_info["team"])
    team_mean = team_driver_signal_means.get(team_name, driver_signal)

    if sim_cfg.apply_regularization and regularization is not None:
        gap_cap = _resolve_teammate_gap_cap(
            driver_info=driver_info,
            cap_by_tier=regularization.teammate_gap_cap_by_tier,
            max_races_by_tier=regularization.teammate_gap_cap_max_races_by_tier,
            min_scale=regularization.teammate_gap_cap_min_scale,
        )
        driver_signal = _clip_driver_signal_by_team_gap(
            driver_signal=driver_signal,
            team_mean=team_mean,
            gap_cap=gap_cap,
        )

        if regularization.driver_signal_shrink > 0:
            experience_tier = _normalize_experience_tier(
                driver_info.get("experience_tier", "unknown")
            )
            extra_shrink = regularization.experience_shrink.get(
                experience_tier,
                regularization.experience_shrink.get("unknown", 0.0),
            )
            delta_from_team = driver_signal - team_mean
            extra_negative_delta_shrink = 0.0
            if delta_from_team < -regularization.negative_delta_threshold:
                extra_negative_delta_shrink = min(
                    max(
                        0.0,
                        ((-delta_from_team) - regularization.negative_delta_threshold)
                        * regularization.negative_delta_shrink_scale,
                    ),
                    max(0.0, regularization.negative_delta_shrink_cap),
                )
            total_shrink = float(
                np.clip(
                    regularization.driver_signal_shrink
                    + float(extra_shrink)
                    + float(extra_negative_delta_shrink),
                    0.0,
                    regularization.max_total_shrink,
                )
            )
            driver_signal = team_mean + ((driver_signal - team_mean) * (1.0 - total_shrink))

        driver_signal = _clip_driver_signal_by_team_gap(
            driver_signal=driver_signal,
            team_mean=team_mean,
            gap_cap=gap_cap,
        )

    if sim_cfg.apply_recent_form_adjustment:
        adjustment = _recent_form_adjustment(
            driver_info=driver_info,
            driver_signal=driver_signal,
            recent_form_scale=sim_cfg.recent_form_scale,
            recent_form_cap=sim_cfg.recent_form_cap,
        )
        if adjustment:
            driver_signal = float(np.clip(driver_signal + adjustment, 0.0, 1.0))

    return driver_signal, gap_cap


def _compute_teammate_anchor_adjustment(
    *,
    driver_info: dict[str, Any],
    raw_driver_signal: float,
    team_driver_signal_means: dict[str, float],
    gap_cap: float | None,
    sim_cfg: QualiSimConfig,
) -> float:
    """Compute the active teammate-aware anchor adjustment for one driver."""
    regularization = sim_cfg.regularization
    if not sim_cfg.apply_regularization or regularization is None:
        return 0.0
    if regularization.teammate_anchor_scale <= 0:
        return 0.0

    team_name = str(driver_info["team"])
    team_mean_raw = team_driver_signal_means.get(team_name, raw_driver_signal)
    teammate_delta = raw_driver_signal - team_mean_raw
    if gap_cap is not None:
        teammate_delta = float(np.clip(teammate_delta, -gap_cap, gap_cap))

    anchor = np.clip(
        teammate_delta * regularization.teammate_anchor_scale,
        -regularization.teammate_anchor_cap,
        regularization.teammate_anchor_cap,
    )
    tier_mult = _fallback_experience_multiplier(
        regularization.anchor_experience_multiplier,
        str(driver_info.get("experience_tier", "unknown")),
        default=1.0,
    )
    return float(anchor * max(0.0, tier_mult))


def _score_single_driver_in_simulation(
    *,
    driver_info: dict[str, Any],
    raw_driver_signal: float,
    regularized_signal: float,
    gap_cap: float | None,
    team_driver_signal_means: dict[str, float],
    sim_cfg: QualiSimConfig,
    weekend_form_offset: float,
    rng: np.random.Generator,
) -> float:
    """Combine team, driver, form, and noise into one simulation score."""
    compressed_team = 0.5 + (
        (float(driver_info["team_strength"]) - 0.5) * sim_cfg.team_strength_compression
    )
    compressed_team = float(np.clip(compressed_team, 0.0, 1.0))

    bounded_driver = 0.5 + (
        np.tanh((regularized_signal - 0.5) / sim_cfg.driver_signal_softness)
        * sim_cfg.driver_offset_cap
    )

    score = (compressed_team * sim_cfg.team_weight) + (bounded_driver * sim_cfg.skill_weight)
    score += float(driver_info.get("learned_position_adjustment", 0.0)) * (
        sim_cfg.effective_learning_scale
    )
    score += weekend_form_offset
    score += _compute_teammate_anchor_adjustment(
        driver_info=driver_info,
        raw_driver_signal=raw_driver_signal,
        team_driver_signal_means=team_driver_signal_means,
        gap_cap=gap_cap,
        sim_cfg=sim_cfg,
    )
    score += float(rng.normal(0, sim_cfg.teammate_setup_std))
    score += float(rng.normal(0, sim_cfg.noise_std))
    return float(score)


def _build_quali_sim_config(
    *,
    cfg: Any,
    is_sprint: bool,
    has_practice_data: bool,
    has_testing_fallback_data: bool,
) -> QualiSimConfig:
    """Build the qualifying simulation config."""
    use_model_only_profile = not has_practice_data
    apply_model_only_teammate_regularization = use_model_only_profile and (
        not has_testing_fallback_data
    )
    apply_testing_fallback_teammate_guard = (
        use_model_only_profile
        and has_testing_fallback_data
        and bool(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_teammate_guard_enabled",
                True,
            )
        )
    )
    apply_recent_form_adjustment = not (has_testing_fallback_data and not has_practice_data)

    noise_std_sprint = float(cfg.get("baseline_predictor.qualifying.noise_std_sprint", 0.025))
    noise_std_normal = float(cfg.get("baseline_predictor.qualifying.noise_std_normal", 0.02))
    noise_std = noise_std_sprint if is_sprint else noise_std_normal

    team_weight = float(cfg.get("baseline_predictor.qualifying.team_weight", 0.7))
    skill_weight = float(cfg.get("baseline_predictor.qualifying.skill_weight", 0.3))
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
    team_strength_compression = float(
        cfg.get("baseline_predictor.qualifying.team_strength_compression", 0.50)
    )
    driver_quali_pace_weight = float(
        cfg.get("baseline_predictor.qualifying.driver_quali_pace_weight", 0.70)
    )
    driver_skill_weight = float(cfg.get("baseline_predictor.qualifying.driver_skill_weight", 0.30))
    driver_weight_sum = driver_quali_pace_weight + driver_skill_weight
    if driver_weight_sum <= 0:
        driver_quali_pace_weight, driver_skill_weight = 0.70, 0.30

    driver_offset_cap = float(cfg.get("baseline_predictor.qualifying.driver_offset_cap", 0.18))
    driver_signal_softness = float(
        cfg.get("baseline_predictor.qualifying.driver_signal_softness", 0.20)
    )
    if driver_signal_softness <= 0:
        driver_signal_softness = 0.20
    teammate_setup_std = float(cfg.get("baseline_predictor.qualifying.teammate_setup_std", 0.015))
    recent_form_scale = float(cfg.get("baseline_predictor.qualifying.recent_form_scale", 0.12))
    recent_form_cap = float(cfg.get("baseline_predictor.qualifying.recent_form_cap", 0.03))

    if has_practice_data:
        team_weight *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_team_weight_multiplier",
                0.94,
            )
        )
        skill_weight *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_skill_weight_multiplier",
                1.12,
            )
        )
        team_weight, skill_weight = _rebalance_component_weights(
            team_weight=team_weight,
            skill_weight=skill_weight,
            fallback_team_weight=0.62,
            fallback_skill_weight=0.38,
        )

        team_strength_compression *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_team_compression_multiplier",
                0.88,
            )
        )
        team_strength_compression = float(np.clip(team_strength_compression, 0.20, 1.0))

        driver_offset_cap *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_driver_offset_cap_multiplier",
                1.33,
            )
        )
        driver_offset_cap = float(np.clip(driver_offset_cap, 0.05, 0.24))

        teammate_setup_std *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_teammate_setup_multiplier",
                1.05,
            )
        )

    if use_model_only_profile:
        team_weight *= float(
            cfg.get("baseline_predictor.qualifying.model_only_team_weight_multiplier", 0.82)
        )
        skill_weight *= float(
            cfg.get("baseline_predictor.qualifying.model_only_skill_weight_multiplier", 1.35)
        )
        team_weight, skill_weight = _rebalance_component_weights(
            team_weight=team_weight,
            skill_weight=skill_weight,
            fallback_team_weight=0.66,
            fallback_skill_weight=0.34,
        )

        team_strength_compression *= float(
            cfg.get("baseline_predictor.qualifying.model_only_team_compression_multiplier", 0.87)
        )
        team_strength_compression = float(np.clip(team_strength_compression, 0.20, 1.0))

        driver_offset_cap *= float(
            cfg.get("baseline_predictor.qualifying.model_only_driver_offset_cap_multiplier", 1.33)
        )
        driver_offset_cap = float(np.clip(driver_offset_cap, 0.05, 0.30))

        noise_std *= float(
            cfg.get("baseline_predictor.qualifying.model_only_noise_multiplier", 1.12)
        )
        teammate_setup_std *= float(
            cfg.get("baseline_predictor.qualifying.model_only_teammate_setup_multiplier", 1.10)
        )

    if has_testing_fallback_data and not has_practice_data:
        team_weight *= float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_team_weight_multiplier", 0.92)
        )
        skill_weight *= float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_skill_weight_multiplier", 1.08)
        )
        team_weight, skill_weight = _rebalance_component_weights(
            team_weight=team_weight,
            skill_weight=skill_weight,
            fallback_team_weight=0.52,
            fallback_skill_weight=0.48,
        )

        driver_offset_cap *= float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_driver_offset_cap_multiplier",
                1.33,
            )
        )
        driver_offset_cap = float(np.clip(driver_offset_cap, 0.05, 0.30))

        noise_std *= float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_noise_multiplier", 1.30)
        )
        teammate_setup_std *= float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_teammate_setup_multiplier",
                1.20,
            )
        )

    weekend_form_std = float(cfg.get("baseline_predictor.qualifying.weekend_form_std", 0.0))
    if use_model_only_profile:
        weekend_form_std *= float(
            cfg.get("baseline_predictor.qualifying.model_only_weekend_form_multiplier", 1.0)
        )
    if has_testing_fallback_data and not has_practice_data:
        weekend_form_floor = float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_weekend_form_std_floor",
                0.008,
            )
        )
        weekend_form_std = max(float(weekend_form_std), weekend_form_floor)

    regularization: RegularizationConfig | None = None
    if apply_model_only_teammate_regularization:
        regularization = RegularizationConfig(
            driver_signal_shrink=float(
                np.clip(
                    float(
                        cfg.get(
                            "baseline_predictor.qualifying.model_only_driver_signal_shrink",
                            0.35,
                        )
                    ),
                    0.0,
                    1.0,
                )
            ),
            experience_shrink=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.model_only_experience_shrink",
                    {
                        "rookie": 0.45,
                        "second_year": 0.30,
                        "developing": 0.20,
                        "sunset": 0.05,
                        "unknown": 0.30,
                    },
                )
            ),
            teammate_gap_cap_by_tier=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.model_only_teammate_gap_cap_by_experience",
                    {
                        "rookie": 0.16,
                        "second_year": 0.12,
                        "developing": 0.10,
                        "unknown": 0.12,
                    },
                )
            ),
            teammate_gap_cap_max_races_by_tier=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.model_only_teammate_gap_cap_max_races_by_experience",
                    {
                        "rookie": 40,
                        "second_year": 55,
                        "developing": 55,
                        "unknown": 45,
                    },
                )
            ),
            teammate_gap_cap_min_scale=float(
                np.clip(
                    float(
                        cfg.get(
                            "baseline_predictor.qualifying.model_only_teammate_gap_cap_min_scale",
                            0.35,
                        )
                    ),
                    0.0,
                    1.0,
                )
            ),
            teammate_anchor_scale=float(
                cfg.get("baseline_predictor.qualifying.model_only_teammate_anchor_scale", 0.12)
            ),
            teammate_anchor_cap=float(
                cfg.get("baseline_predictor.qualifying.model_only_teammate_anchor_cap", 0.04)
            ),
            anchor_experience_multiplier=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.model_only_teammate_anchor_experience_multiplier",
                    {
                        "rookie": 0.30,
                        "second_year": 0.45,
                        "developing": 0.55,
                        "sunset": 1.00,
                        "unknown": 0.45,
                    },
                )
            ),
            negative_delta_threshold=float(
                cfg.get("baseline_predictor.qualifying.model_only_negative_delta_threshold", 0.08)
            ),
            negative_delta_shrink_scale=float(
                cfg.get(
                    "baseline_predictor.qualifying.model_only_negative_delta_shrink_scale",
                    1.0,
                )
            ),
            negative_delta_shrink_cap=float(
                cfg.get("baseline_predictor.qualifying.model_only_negative_delta_shrink_cap", 0.25)
            ),
            max_total_shrink=0.95,
        )
    elif apply_testing_fallback_teammate_guard:
        regularization = RegularizationConfig(
            driver_signal_shrink=float(
                np.clip(
                    float(
                        cfg.get(
                            "baseline_predictor.qualifying.testing_fallback_driver_signal_shrink",
                            0.14,
                        )
                    ),
                    0.0,
                    1.0,
                )
            ),
            experience_shrink=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_experience_shrink",
                    {
                        "rookie": 0.22,
                        "second_year": 0.14,
                        "developing": 0.09,
                        "sunset": 0.03,
                        "unknown": 0.14,
                    },
                )
            ),
            teammate_gap_cap_by_tier=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_by_experience",
                    {
                        "rookie": 0.22,
                        "second_year": 0.18,
                        "developing": 0.14,
                        "unknown": 0.18,
                    },
                )
            ),
            teammate_gap_cap_max_races_by_tier=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_max_races_by_experience",
                    {
                        "rookie": 40,
                        "second_year": 55,
                        "developing": 55,
                        "unknown": 45,
                    },
                )
            ),
            teammate_gap_cap_min_scale=float(
                np.clip(
                    float(
                        cfg.get(
                            "baseline_predictor.qualifying.testing_fallback_teammate_gap_cap_min_scale",
                            0.30,
                        )
                    ),
                    0.0,
                    1.0,
                )
            ),
            teammate_anchor_scale=float(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_teammate_anchor_scale", 0.07
                )
            ),
            teammate_anchor_cap=float(
                cfg.get("baseline_predictor.qualifying.testing_fallback_teammate_anchor_cap", 0.025)
            ),
            anchor_experience_multiplier=_normalized_tier_mapping(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_teammate_anchor_experience_multiplier",
                    {
                        "rookie": 0.55,
                        "second_year": 0.70,
                        "developing": 0.80,
                        "sunset": 1.00,
                        "unknown": 0.70,
                    },
                )
            ),
            negative_delta_threshold=float(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_negative_delta_threshold",
                    0.12,
                )
            ),
            negative_delta_shrink_scale=float(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_negative_delta_shrink_scale",
                    0.7,
                )
            ),
            negative_delta_shrink_cap=float(
                cfg.get(
                    "baseline_predictor.qualifying.testing_fallback_negative_delta_shrink_cap",
                    0.12,
                )
            ),
            max_total_shrink=0.90,
        )

    return QualiSimConfig(
        noise_std=float(noise_std),
        team_weight=float(team_weight),
        skill_weight=float(skill_weight),
        team_strength_compression=float(team_strength_compression),
        driver_offset_cap=float(driver_offset_cap),
        driver_signal_softness=float(driver_signal_softness),
        driver_quali_pace_weight=float(driver_quali_pace_weight),
        driver_skill_weight=float(driver_skill_weight),
        effective_learning_scale=float(effective_learning_scale),
        weekend_form_std=float(weekend_form_std),
        teammate_setup_std=float(teammate_setup_std),
        recent_form_scale=float(recent_form_scale),
        recent_form_cap=float(recent_form_cap),
        apply_regularization=regularization is not None,
        apply_recent_form_adjustment=apply_recent_form_adjustment,
        regularization=regularization,
    )


def run_qualifying_simulations(
    *,
    all_drivers: list[dict[str, Any]],
    n_simulations: int,
    is_sprint: bool,
    has_practice_data: bool,
    rng: np.random.Generator,
    cfg: Any,
    logger: Any,
    has_testing_fallback_data: bool = False,
) -> dict[str, list[int]]:
    """Run Monte Carlo qualifying simulations and return position histories.

    When we only have testing data, keep enough driver variance that the grid
    does not turn into pure team order.
    """
    _ = logger
    sim_cfg = _build_quali_sim_config(
        cfg=cfg,
        is_sprint=is_sprint,
        has_practice_data=has_practice_data,
        has_testing_fallback_data=has_testing_fallback_data,
    )

    position_records: dict[str, list[int]] = {str(d["driver"]): [] for d in all_drivers}
    driver_weight_sum = sim_cfg.driver_quali_pace_weight + sim_cfg.driver_skill_weight

    team_driver_signal_means: dict[str, float] = {}
    for driver_info in all_drivers:
        driver_signal = (
            (float(driver_info["quali_pace"]) * sim_cfg.driver_quali_pace_weight)
            + (float(driver_info["skill"]) * sim_cfg.driver_skill_weight)
        ) / driver_weight_sum
        team_name = str(driver_info["team"])
        team_driver_signal_means.setdefault(team_name, 0.0)
        team_driver_signal_means[team_name] += driver_signal

    team_counts: dict[str, int] = {}
    for driver_info in all_drivers:
        team_name = str(driver_info["team"])
        team_counts[team_name] = team_counts.get(team_name, 0) + 1

    for team_name, total_signal in team_driver_signal_means.items():
        count = team_counts.get(team_name, 1)
        team_driver_signal_means[team_name] = total_signal / count

    weekend_form = {
        str(driver_info["driver"]): (
            float(rng.normal(0, sim_cfg.weekend_form_std)) if sim_cfg.weekend_form_std > 0 else 0.0
        )
        for driver_info in all_drivers
    }

    for _ in range(n_simulations):
        driver_scores: list[dict[str, Any]] = []
        for driver_info in all_drivers:
            raw_driver_signal = (
                (float(driver_info["quali_pace"]) * sim_cfg.driver_quali_pace_weight)
                + (float(driver_info["skill"]) * sim_cfg.driver_skill_weight)
            ) / driver_weight_sum
            regularized_signal, gap_cap = _compute_regularized_driver_signal(
                driver_info=driver_info,
                raw_driver_signal=raw_driver_signal,
                team_driver_signal_means=team_driver_signal_means,
                sim_cfg=sim_cfg,
            )
            score = _score_single_driver_in_simulation(
                driver_info=driver_info,
                raw_driver_signal=raw_driver_signal,
                regularized_signal=regularized_signal,
                gap_cap=gap_cap,
                team_driver_signal_means=team_driver_signal_means,
                sim_cfg=sim_cfg,
                weekend_form_offset=weekend_form.get(str(driver_info["driver"]), 0.0),
                rng=rng,
            )

            driver_scores.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "score": score,
                }
            )

        driver_scores.sort(key=lambda item: float(item["score"]), reverse=True)
        for index, item in enumerate(driver_scores, start=1):
            position_records[str(item["driver"])].append(index)

    return position_records
