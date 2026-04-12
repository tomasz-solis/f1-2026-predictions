"""Monte Carlo simulation helpers for qualifying predictions."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.validation_helpers import normalize_weather_key


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
    wet_skill_adjustment: float,
    rng: np.random.Generator,
) -> float:
    """Combine team, driver, form, wet skill, and noise into one simulation score."""
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
    score += wet_skill_adjustment
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


@dataclass(frozen=True)
class _DataAvailabilityContext:
    """Which data sources are available for this qualifying prediction."""

    has_practice_data: bool
    has_testing_fallback_data: bool

    @property
    def use_model_only_profile(self) -> bool:
        """True when no practice session data is available."""
        return not self.has_practice_data

    @property
    def apply_model_only_teammate_regularization(self) -> bool:
        """True when no practice AND no testing fallback data is available."""
        return self.use_model_only_profile and not self.has_testing_fallback_data

    @property
    def apply_recent_form_adjustment(self) -> bool:
        """True unless we're in testing-fallback-only mode."""
        return not (self.has_testing_fallback_data and not self.has_practice_data)


def _load_base_quali_weights(cfg: Any, is_sprint: bool) -> dict[str, float]:
    """Load raw qualifying weights from config, with defensive guards."""
    noise_std_sprint = float(cfg.get("baseline_predictor.qualifying.noise_std_sprint", 0.025))
    noise_std_normal = float(cfg.get("baseline_predictor.qualifying.noise_std_normal", 0.02))

    driver_quali_pace_weight = float(
        cfg.get("baseline_predictor.qualifying.driver_quali_pace_weight", 0.70)
    )
    driver_skill_weight = float(cfg.get("baseline_predictor.qualifying.driver_skill_weight", 0.30))
    # Guard: reset to defaults if config produces non-positive sum
    if driver_quali_pace_weight + driver_skill_weight <= 0:
        driver_quali_pace_weight, driver_skill_weight = 0.70, 0.30

    driver_signal_softness = float(
        cfg.get("baseline_predictor.qualifying.driver_signal_softness", 0.20)
    )
    # Guard: prevent division by zero downstream
    if driver_signal_softness <= 0:
        driver_signal_softness = 0.20

    return {
        "noise_std": noise_std_sprint if is_sprint else noise_std_normal,
        "team_weight": float(cfg.get("baseline_predictor.qualifying.team_weight", 0.60)),
        "skill_weight": float(cfg.get("baseline_predictor.qualifying.skill_weight", 0.40)),
        "team_strength_compression": float(
            cfg.get("baseline_predictor.qualifying.team_strength_compression", 0.60)
        ),
        "driver_offset_cap": float(
            cfg.get("baseline_predictor.qualifying.driver_offset_cap", 0.18)
        ),
        "driver_signal_softness": driver_signal_softness,
        "driver_quali_pace_weight": driver_quali_pace_weight,
        "driver_skill_weight": driver_skill_weight,
        "teammate_setup_std": float(
            cfg.get("baseline_predictor.qualifying.teammate_setup_std", 0.015)
        ),
        "recent_form_scale": float(
            cfg.get("baseline_predictor.qualifying.recent_form_scale", 0.12)
        ),
        "recent_form_cap": float(cfg.get("baseline_predictor.qualifying.recent_form_cap", 0.03)),
        "learning_position_to_score_scale": float(
            cfg.get("baseline_predictor.qualifying.learning.position_to_score_scale", 0.03)
        ),
        "learning_with_practice_multiplier": float(
            cfg.get("baseline_predictor.qualifying.learning.with_practice_multiplier", 0.65)
        ),
    }


def _apply_data_source_multipliers(
    weights: dict[str, float],
    ctx: _DataAvailabilityContext,
    cfg: Any,
) -> dict[str, float]:
    """Apply practice/model-only/testing-fallback multipliers to base weights.

    IMPORTANT: These are THREE SEQUENTIAL if-blocks, NOT if/elif/else.
    When has_testing_fallback_data=True and has_practice_data=False, BOTH
    the model-only AND testing-fallback blocks fire. Multipliers stack.
    """
    # --- Block 1: Practice data multipliers ---
    if ctx.has_practice_data:
        weights["team_weight"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_team_weight_multiplier",
                0.94,
            )
        )
        weights["skill_weight"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_skill_weight_multiplier",
                1.12,
            )
        )
        weights["team_weight"], weights["skill_weight"] = _rebalance_component_weights(
            team_weight=weights["team_weight"],
            skill_weight=weights["skill_weight"],
            fallback_team_weight=0.62,
            fallback_skill_weight=0.38,
        )

        weights["team_strength_compression"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_team_compression_multiplier",
                0.88,
            )
        )
        weights["team_strength_compression"] = float(
            np.clip(weights["team_strength_compression"], 0.20, 1.0)
        )

        weights["driver_offset_cap"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_driver_offset_cap_multiplier",
                1.33,
            )
        )
        weights["driver_offset_cap"] = float(np.clip(weights["driver_offset_cap"], 0.05, 0.24))

        weights["teammate_setup_std"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.practice_data_teammate_setup_multiplier",
                1.05,
            )
        )

    # --- Block 2: Model-only multipliers ---
    # This is a SEPARATE if, NOT elif. It fires whenever has_practice_data=False.
    if ctx.use_model_only_profile:
        weights["team_weight"] *= float(
            cfg.get("baseline_predictor.qualifying.model_only_team_weight_multiplier", 0.82)
        )
        weights["skill_weight"] *= float(
            cfg.get("baseline_predictor.qualifying.model_only_skill_weight_multiplier", 1.35)
        )
        weights["team_weight"], weights["skill_weight"] = _rebalance_component_weights(
            team_weight=weights["team_weight"],
            skill_weight=weights["skill_weight"],
            fallback_team_weight=0.66,
            fallback_skill_weight=0.34,
        )

        weights["team_strength_compression"] *= float(
            cfg.get("baseline_predictor.qualifying.model_only_team_compression_multiplier", 0.87)
        )
        weights["team_strength_compression"] = float(
            np.clip(weights["team_strength_compression"], 0.20, 1.0)
        )

        weights["driver_offset_cap"] *= float(
            cfg.get("baseline_predictor.qualifying.model_only_driver_offset_cap_multiplier", 1.33)
        )
        weights["driver_offset_cap"] = float(np.clip(weights["driver_offset_cap"], 0.05, 0.30))

        weights["noise_std"] *= float(
            cfg.get("baseline_predictor.qualifying.model_only_noise_multiplier", 1.12)
        )
        weights["teammate_setup_std"] *= float(
            cfg.get("baseline_predictor.qualifying.model_only_teammate_setup_multiplier", 1.10)
        )

    # --- Block 3: Testing fallback multipliers ---
    # This is a SEPARATE if, NOT elif. It stacks ON TOP of Block 2.
    if ctx.has_testing_fallback_data and not ctx.has_practice_data:
        weights["team_weight"] *= float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_team_weight_multiplier", 0.92)
        )
        weights["skill_weight"] *= float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_skill_weight_multiplier", 1.08)
        )
        weights["team_weight"], weights["skill_weight"] = _rebalance_component_weights(
            team_weight=weights["team_weight"],
            skill_weight=weights["skill_weight"],
            fallback_team_weight=0.52,
            fallback_skill_weight=0.48,
        )

        weights["driver_offset_cap"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_driver_offset_cap_multiplier",
                1.33,
            )
        )
        weights["driver_offset_cap"] = float(np.clip(weights["driver_offset_cap"], 0.05, 0.30))

        weights["noise_std"] *= float(
            cfg.get("baseline_predictor.qualifying.testing_fallback_noise_multiplier", 1.30)
        )
        weights["teammate_setup_std"] *= float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_teammate_setup_multiplier",
                1.20,
            )
        )

    # --- Weekend form std ---
    weekend_form_std = float(cfg.get("baseline_predictor.qualifying.weekend_form_std", 0.0))
    if ctx.use_model_only_profile:
        weekend_form_std *= float(
            cfg.get("baseline_predictor.qualifying.model_only_weekend_form_multiplier", 1.0)
        )
    if ctx.has_testing_fallback_data and not ctx.has_practice_data:
        weekend_form_floor = float(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_weekend_form_std_floor",
                0.008,
            )
        )
        weekend_form_std = max(float(weekend_form_std), weekend_form_floor)
    weights["weekend_form_std"] = weekend_form_std

    return weights


def _build_model_only_regularization(cfg: Any) -> RegularizationConfig:
    """Build RegularizationConfig for model-only mode (no practice, no testing)."""
    return RegularizationConfig(
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


def _build_testing_fallback_regularization(cfg: Any) -> RegularizationConfig:
    """Build RegularizationConfig for testing-fallback mode."""
    return RegularizationConfig(
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
            cfg.get("baseline_predictor.qualifying.testing_fallback_teammate_anchor_scale", 0.07)
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


def _build_quali_sim_config(
    *,
    cfg: Any,
    is_sprint: bool,
    has_practice_data: bool,
    has_testing_fallback_data: bool,
) -> QualiSimConfig:
    """Build qualifying simulation config from data context and settings."""
    ctx = _DataAvailabilityContext(
        has_practice_data=has_practice_data,
        has_testing_fallback_data=has_testing_fallback_data,
    )
    weights = _load_base_quali_weights(cfg, is_sprint)
    weights = _apply_data_source_multipliers(weights, ctx, cfg)

    effective_learning_scale = (
        weights["learning_position_to_score_scale"]
        if not ctx.has_practice_data
        else (
            weights["learning_position_to_score_scale"]
            * weights["learning_with_practice_multiplier"]
        )
    )

    # Regularization — resolved here because it requires cfg lookup
    apply_testing_fallback_teammate_guard = (
        ctx.use_model_only_profile
        and ctx.has_testing_fallback_data
        and bool(
            cfg.get(
                "baseline_predictor.qualifying.testing_fallback_teammate_guard_enabled",
                True,
            )
        )
    )

    regularization: RegularizationConfig | None = None
    if ctx.apply_model_only_teammate_regularization:
        regularization = _build_model_only_regularization(cfg)
    elif apply_testing_fallback_teammate_guard:
        regularization = _build_testing_fallback_regularization(cfg)

    return QualiSimConfig(
        noise_std=weights["noise_std"],
        team_weight=weights["team_weight"],
        skill_weight=weights["skill_weight"],
        team_strength_compression=weights["team_strength_compression"],
        driver_offset_cap=weights["driver_offset_cap"],
        driver_signal_softness=weights["driver_signal_softness"],
        driver_quali_pace_weight=weights["driver_quali_pace_weight"],
        driver_skill_weight=weights["driver_skill_weight"],
        effective_learning_scale=effective_learning_scale,
        weekend_form_std=weights["weekend_form_std"],
        teammate_setup_std=weights["teammate_setup_std"],
        recent_form_scale=weights["recent_form_scale"],
        recent_form_cap=weights["recent_form_cap"],
        apply_regularization=regularization is not None,
        apply_recent_form_adjustment=ctx.apply_recent_form_adjustment,
        regularization=regularization,
    )


def _compute_wet_skill_adjustment(
    driver_info: dict[str, Any],
    *,
    weather: str,
    wet_skill_weight: float,
    wet_skill_neutral: float,
    mixed_wet_blend: float,
) -> float:
    """Return the score delta from a driver's wet skill relative to the neutral baseline.

    Positive values mean the driver gains in wet conditions; negative means they lose.
    Mixed conditions apply a fractional blend of the full wet effect.
    """
    weather_key = normalize_weather_key(weather)
    if weather_key not in {"rain", "mixed"}:
        return 0.0
    raw_wet_skill = driver_info.get("wet_skill")
    wet_skill = float(raw_wet_skill if raw_wet_skill is not None else wet_skill_neutral)
    full_adjustment = (wet_skill - wet_skill_neutral) * wet_skill_weight
    if weather_key == "mixed":
        return full_adjustment * float(mixed_wet_blend)
    return full_adjustment


def run_qualifying_simulations(
    *,
    all_drivers: list[dict[str, Any]],
    n_simulations: int,
    is_sprint: bool,
    has_practice_data: bool,
    rng: np.random.Generator,
    cfg: Any,
    logger: Any,
    weather: str = "dry",
    has_testing_fallback_data: bool = False,
) -> dict[str, list[int]]:
    """Run Monte Carlo qualifying simulations and return position histories.

    When we only have testing data, keep enough driver variance that the grid
    does not turn into pure team order. Wet and mixed weather increase noise and
    apply per-driver wet_skill adjustments so the hierarchy can shuffle.
    """
    _ = logger
    sim_cfg = _build_quali_sim_config(
        cfg=cfg,
        is_sprint=is_sprint,
        has_practice_data=has_practice_data,
        has_testing_fallback_data=has_testing_fallback_data,
    )

    # Scale noise for wet/mixed conditions.
    weather_key = normalize_weather_key(weather)
    wet_noise_multiplier = float(
        cfg.get("baseline_predictor.qualifying.wet_noise_multiplier", 1.60)
    )
    mixed_noise_multiplier = float(
        cfg.get("baseline_predictor.qualifying.mixed_noise_multiplier", 1.28)
    )
    if weather_key == "rain":
        noise_scale = wet_noise_multiplier
    elif weather_key == "mixed":
        noise_scale = mixed_noise_multiplier
    else:
        noise_scale = 1.0

    # Rebuild sim_cfg with scaled noise rather than mutating the frozen dataclass.
    sim_cfg = dataclasses.replace(
        sim_cfg,
        noise_std=sim_cfg.noise_std * noise_scale,
        teammate_setup_std=sim_cfg.teammate_setup_std * noise_scale,
    )

    wet_skill_weight = float(cfg.get("baseline_predictor.qualifying.wet_skill_weight", 0.18))
    if is_sprint:
        sprint_wet_scale = float(
            cfg.get("baseline_predictor.qualifying.sprint_wet_skill_scale", 0.75)
        )
        wet_skill_weight *= sprint_wet_scale
    wet_skill_neutral = float(cfg.get("baseline_predictor.qualifying.wet_skill_neutral", 0.70))
    mixed_wet_blend = float(cfg.get("baseline_predictor.mixed_wet_blend", 0.50))

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

    # Weekend form is drawn once per driver for the entire simulation batch,
    # not per individual sim iteration. This models persistent weekend effects
    # (setup quality, driver comfort with the track, weather sensitivity)
    # rather than lap-to-lap variance. The per-iteration randomness is
    # captured separately by the noise_std and teammate_setup_std draws
    # inside _score_single_driver_in_simulation.
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
            wet_adj = _compute_wet_skill_adjustment(
                driver_info,
                weather=weather,
                wet_skill_weight=wet_skill_weight,
                wet_skill_neutral=wet_skill_neutral,
                mixed_wet_blend=mixed_wet_blend,
            )
            score = _score_single_driver_in_simulation(
                driver_info=driver_info,
                raw_driver_signal=raw_driver_signal,
                regularized_signal=regularized_signal,
                gap_cap=gap_cap,
                team_driver_signal_means=team_driver_signal_means,
                sim_cfg=sim_cfg,
                weekend_form_offset=weekend_form.get(str(driver_info["driver"]), 0.0),
                wet_skill_adjustment=wet_adj,
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
