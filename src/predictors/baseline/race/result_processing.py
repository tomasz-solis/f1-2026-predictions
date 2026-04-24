"""Finish-order and confidence helpers for race prediction flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.predictors.baseline.early_season_uncertainty import (
    resolve_early_season_confidence_penalty,
    resolve_early_season_interval_extension,
)


def estimate_predicted_grid_uncertainty_share(
    *,
    grid_position_samples_by_driver: dict[str, list[float]],
    field_size: int,
    cfg: Any,
) -> float:
    """Estimate how uncertain the sampled starting grid is on a 0-1 scale."""
    if field_size <= 1 or not grid_position_samples_by_driver:
        return 0.0

    interval_widths: list[float] = []
    for samples in grid_position_samples_by_driver.values():
        if len(samples) < 2:
            continue
        p5 = float(np.percentile(samples, 5))
        p95 = float(np.percentile(samples, 95))
        interval_widths.append(max(0.0, p95 - p5))

    if not interval_widths:
        return 0.0

    activation_width = float(
        cfg.get("baseline_predictor.race.predicted_grid_uncertainty.activation_width", 2.0)
    )
    width_scale = float(
        max(
            1e-6,
            cfg.get(
                "baseline_predictor.race.predicted_grid_uncertainty.width_scale",
                max(3.0, field_size / 3.0),
            ),
        )
    )
    mean_width = float(np.mean(interval_widths))
    return float(np.clip((mean_width - activation_width) / width_scale, 0.0, 1.0))


def apply_low_confidence_interval_floor(
    *,
    finish_order: list[dict[str, Any]],
    input_confidence: float | None,
    cfg: Any,
    field_size: int,
) -> None:
    """Widen top-driver position ranges when run confidence is explicitly low."""
    if input_confidence is None or field_size <= 1:
        return

    confidence = float(np.clip(input_confidence, 0.0, 1.0))
    threshold = float(
        np.clip(
            cfg.get(
                "baseline_predictor.race.position_interval_floor.apply_below_input_confidence",
                0.65,
            ),
            0.0,
            1.0,
        )
    )
    if threshold <= 0.0 or confidence >= threshold:
        return

    top_n = int(cfg.get("baseline_predictor.race.position_interval_floor.top_n", 3))
    top_n = max(1, min(top_n, field_size))
    min_width = int(cfg.get("baseline_predictor.race.position_interval_floor.min_width", 1))
    min_width = max(0, min(min_width, field_size - 1))
    max_extra_width = int(
        cfg.get("baseline_predictor.race.position_interval_floor.max_extra_width", 1)
    )
    max_extra_width = max(0, min(max_extra_width, field_size - 1))

    low_confidence_share = float((threshold - confidence) / max(threshold, 1e-6))
    target_width = min(
        field_size - 1,
        min_width + int(round(low_confidence_share * max_extra_width)),
    )
    if target_width <= 0:
        return

    for row in finish_order:
        try:
            position = int(row.get("position", field_size))
            p5 = int(row.get("p5", position))
            p95 = int(row.get("p95", position))
        except (TypeError, ValueError):
            continue

        if position > top_n:
            continue

        p5 = max(1, min(p5, field_size))
        p95 = max(p5, min(p95, field_size))
        current_width = p95 - p5
        if current_width >= target_width:
            row["p5"] = p5
            row["p95"] = p95
            continue

        deficit = target_width - current_width
        upper_room = field_size - p95
        add_upper = min(upper_room, deficit)
        deficit -= add_upper

        lower_room = p5 - 1
        add_lower = min(lower_room, deficit)
        deficit -= add_lower

        if deficit > 0 and upper_room > add_upper:
            extra_upper = min(upper_room - add_upper, deficit)
            add_upper += extra_upper
            deficit -= extra_upper
        if deficit > 0 and lower_room > add_lower:
            extra_lower = min(lower_room - add_lower, deficit)
            add_lower += extra_lower

        row["p5"] = p5 - add_lower
        row["p95"] = p95 + add_upper


def apply_learned_interval_radius(
    *,
    finish_order: list[dict[str, Any]],
    learned_interval_radius: float,
    field_size: int,
) -> None:
    """Apply a learned residual-radius floor to published finish intervals."""
    target_half_width = int(np.ceil(max(0.0, learned_interval_radius)))
    if target_half_width <= 0 or field_size <= 1:
        return

    for row in finish_order:
        try:
            center = int(row.get("median_position", row.get("position", field_size)))
            p5 = int(row.get("p5", center))
            p95 = int(row.get("p95", center))
        except (TypeError, ValueError):
            continue

        center = max(1, min(center, field_size))
        lower = max(1, min(p5, p95, field_size))
        upper = max(lower, min(max(p5, p95), field_size))
        current_half_width = max(center - lower, upper - center)
        required_half_width = max(current_half_width, target_half_width)

        row["p5"] = max(1, center - required_half_width)
        row["p95"] = min(field_size, center + required_half_width)


def apply_early_season_team_uncertainty_adjustments(
    *,
    finish_order: list[dict[str, Any]],
    driver_info_map: dict[str, dict[str, Any]],
    cfg: Any,
    field_size: int,
) -> None:
    """Widen finish intervals and lower confidence when preseason uncertainty is still live."""
    if field_size <= 1:
        return

    confidence_floor = float(cfg.get("baseline_predictor.race.confidence.min", 40.0))
    confidence_cap = float(cfg.get("baseline_predictor.race.confidence.max", 60.0))

    for row in finish_order:
        driver_code = str(row.get("driver", "")).strip()
        if not driver_code:
            continue

        driver_info = driver_info_map.get(driver_code, {})
        interval_extension = resolve_early_season_interval_extension(
            team_uncertainty=driver_info.get("team_uncertainty"),
            races_completed=driver_info.get("season_races_completed"),
            cfg=cfg,
            prefix="baseline_predictor.race",
        )
        if interval_extension > 0:
            try:
                center = int(row.get("median_position", row.get("position", field_size)))
                p5 = int(row.get("p5", center))
                p95 = int(row.get("p95", center))
            except (TypeError, ValueError):
                center = int(row.get("position", field_size))
                p5 = center
                p95 = center

            center = max(1, min(center, field_size))
            row["p5"] = max(1, min(p5, center - interval_extension))
            row["p95"] = min(field_size, max(p95, center + interval_extension))

        confidence_penalty = resolve_early_season_confidence_penalty(
            team_uncertainty=driver_info.get("team_uncertainty"),
            races_completed=driver_info.get("season_races_completed"),
            cfg=cfg,
            prefix="baseline_predictor.race",
        )
        if confidence_penalty <= 0.0:
            continue

        try:
            confidence = float(row.get("confidence", confidence_floor))
        except (TypeError, ValueError):
            confidence = confidence_floor
        row["confidence"] = round(
            float(np.clip(confidence - confidence_penalty, confidence_floor, confidence_cap)),
            1,
        )


def apply_hypothetical_points_floor(
    *,
    info: dict[str, Any],
    position_blend_score: float,
    blended_position_samples: list[float],
    reference_grid_pos: float,
    field_size: int,
    cfg: Any,
) -> tuple[float, list[float]]:
    """Keep hypothetical team swaps from losing clear points-finishing evidence."""
    if not bool(info.get("is_hypothetical_team_assignment")):
        return position_blend_score, blended_position_samples

    top_grid_limit = int(
        cfg.get("baseline_predictor.race.final_blend.hypothetical_points_floor.top_grid_limit", 10)
    )
    if reference_grid_pos > top_grid_limit:
        return position_blend_score, blended_position_samples

    # Gate on raw extraction skill — the team-independent driver quality signal.
    raw_skill = float(info.get("raw_skill", info.get("skill", 0.5)))
    portable_skill_threshold = float(
        cfg.get(
            "baseline_predictor.race.final_blend.hypothetical_points_floor.portable_skill_threshold",
            0.70,
        )
    )
    if raw_skill < portable_skill_threshold:
        return position_blend_score, blended_position_samples

    team_strength_threshold = float(
        cfg.get(
            "baseline_predictor.race.final_blend.hypothetical_points_floor.team_strength_threshold",
            0.50,
        )
    )
    if float(info.get("team_strength", 0.0)) < team_strength_threshold:
        return position_blend_score, blended_position_samples

    dnf_probability_cap = float(
        cfg.get(
            "baseline_predictor.race.final_blend.hypothetical_points_floor.dnf_probability_cap",
            0.12,
        )
    )
    if float(info.get("dnf_probability", 1.0)) > dnf_probability_cap:
        return position_blend_score, blended_position_samples

    max_loss_positions = float(
        cfg.get(
            "baseline_predictor.race.final_blend.hypothetical_points_floor.max_loss_positions",
            0.0,
        )
    )
    capped_position = float(
        np.clip(reference_grid_pos + max_loss_positions, 1.0, float(max(1, field_size)))
    )
    capped_samples = [min(sample, capped_position) for sample in blended_position_samples]
    return min(position_blend_score, capped_position), capped_samples


@dataclass(frozen=True)
class _FinishOrderConfig:
    """Resolved config for the build_finish_order scoring loop."""

    confidence_floor: float
    confidence_cap: float
    context_confidence: float
    low_confidence_share: float
    context_confidence_adjustment: float
    weather_penalty: float
    predicted_grid_uncertainty_share: float
    grid_anchor_weight: float
    overtake_blend_scale: float
    race_adv_blend_scale: float
    skill_blend_scale: float
    elite_skill_threshold: float
    elite_driver_scale: float
    elite_driver_exponent: float
    max_driver_adjustment: float
    max_gain_base: float
    max_gain_track_scale: float
    skill_gain_scale: float
    race_adv_gain_scale: float
    max_gain_floor: float
    max_gain_ceiling: float
    racecraft_confidence_scale: float
    max_gain_confidence_scale: float
    predicted_grid_racecraft_scale: float
    predicted_grid_max_gain_scale: float
    dnf_probability_output_cap: float
    learning_position_scale: float
    track_overtaking: float


def _load_finish_order_config(
    *,
    cfg: Any,
    weather: str,
    weather_feature_modifiers: dict[str, float],
    race_params: dict[str, Any],
    input_confidence: float | None,
    is_sprint: bool,
    grid_position_samples_by_driver: dict[str, list[float]],
    field_size: int,
) -> _FinishOrderConfig:
    """Load and resolve all config for build_finish_order."""
    confidence_floor = cfg.get("baseline_predictor.race.confidence.min", 40.0)
    confidence_cap = cfg.get("baseline_predictor.race.confidence.max", 60.0)
    context_confidence_scale = float(
        cfg.get("baseline_predictor.race.confidence.context_scale", 8.0)
    )
    context_confidence = (
        0.5 if input_confidence is None else float(np.clip(input_confidence, 0.0, 1.0))
    )
    low_confidence_share = float(np.clip(1.0 - context_confidence, 0.0, 1.0))
    context_confidence_adjustment = (context_confidence - 0.5) * context_confidence_scale
    weather_penalty = (
        cfg.get("baseline_predictor.race.confidence.weather_penalty_wet", 4.0)
        if weather in ("rain", "mixed")
        else 0.0
    )
    weather_penalty += float(weather_feature_modifiers.get("confidence_adjustment", 0.0))
    predicted_grid_uncertainty_share = estimate_predicted_grid_uncertainty_share(
        grid_position_samples_by_driver=grid_position_samples_by_driver,
        field_size=field_size,
        cfg=cfg,
    )

    track_overtaking = race_params.get("track_overtaking", 0.5)
    grid_anchor_weight = np.clip(
        cfg.get("baseline_predictor.race.grid_anchor.base", 0.25)
        + (track_overtaking * cfg.get("baseline_predictor.race.grid_anchor.track_scale", 0.30)),
        0.10,
        0.80,
    )
    grid_anchor_min = cfg.get("baseline_predictor.race.grid_anchor.min", 0.35)
    main_grid_anchor_max = cfg.get("baseline_predictor.race.grid_anchor.main_max", 0.58)
    sprint_grid_anchor_min = cfg.get("baseline_predictor.race.grid_anchor.sprint_min", 0.78)
    if is_sprint:
        grid_anchor_weight = max(grid_anchor_weight, sprint_grid_anchor_min)
    else:
        grid_anchor_weight = max(grid_anchor_weight, grid_anchor_min)
        grid_anchor_weight = min(grid_anchor_weight, main_grid_anchor_max)
        grid_anchor_low_confidence_scale = float(
            cfg.get("baseline_predictor.race.grid_anchor.low_confidence_scale", 0.18)
        )
        grid_anchor_weight = float(
            np.clip(
                grid_anchor_weight + (low_confidence_share * grid_anchor_low_confidence_scale),
                grid_anchor_min,
                main_grid_anchor_max,
            )
        )
        predicted_grid_anchor_scale = float(
            cfg.get("baseline_predictor.race.predicted_grid_uncertainty.anchor_scale", 0.0)
        )
        grid_anchor_weight = float(
            np.clip(
                grid_anchor_weight
                + (predicted_grid_uncertainty_share * predicted_grid_anchor_scale),
                grid_anchor_min,
                main_grid_anchor_max,
            )
        )

    overtake_blend_scale = cfg.get(
        "baseline_predictor.race.final_blend.overtaking_skill_scale",
        1.6,
    )
    race_adv_blend_scale = cfg.get(
        "baseline_predictor.race.final_blend.race_advantage_scale",
        1.3,
    )
    skill_blend_scale = cfg.get("baseline_predictor.race.final_blend.driver_skill_scale", 1.1)
    elite_skill_threshold = cfg.get(
        "baseline_predictor.race.final_blend.elite_driver_skill_threshold",
        0.88,
    )
    elite_driver_scale = cfg.get("baseline_predictor.race.final_blend.elite_driver_scale", 0.80)
    elite_driver_exponent = cfg.get(
        "baseline_predictor.race.final_blend.elite_driver_exponent",
        1.35,
    )
    max_driver_adjustment = cfg.get(
        "baseline_predictor.race.final_blend.max_driver_adjustment_positions",
        0.9,
    )
    max_gain_base = cfg.get("baseline_predictor.race.final_blend.max_gain_base", 6.0)
    max_gain_track_scale = cfg.get(
        "baseline_predictor.race.final_blend.max_gain_track_scale",
        4.0,
    )
    skill_gain_scale = cfg.get(
        "baseline_predictor.race.final_blend.max_gain_overtaking_skill_scale",
        2.0,
    )
    race_adv_gain_scale = cfg.get(
        "baseline_predictor.race.final_blend.max_gain_race_advantage_scale",
        2.5,
    )
    max_gain_floor = cfg.get("baseline_predictor.race.final_blend.max_gain_floor", 4.0)
    max_gain_ceiling = cfg.get("baseline_predictor.race.final_blend.max_gain_ceiling", 11.0)
    low_confidence_racecraft_floor = float(
        cfg.get("baseline_predictor.race.final_blend.low_confidence_racecraft_floor", 0.72)
    )
    low_confidence_max_gain_floor = float(
        cfg.get("baseline_predictor.race.final_blend.low_confidence_max_gain_floor", 0.82)
    )
    racecraft_confidence_scale = float(
        np.clip(
            low_confidence_racecraft_floor
            + ((1.0 - low_confidence_racecraft_floor) * context_confidence),
            0.0,
            1.0,
        )
    )
    max_gain_confidence_scale = float(
        np.clip(
            low_confidence_max_gain_floor
            + ((1.0 - low_confidence_max_gain_floor) * context_confidence),
            0.0,
            1.0,
        )
    )
    predicted_grid_racecraft_scale = float(
        np.clip(
            1.0
            - (
                predicted_grid_uncertainty_share
                * float(
                    cfg.get(
                        "baseline_predictor.race.predicted_grid_uncertainty.racecraft_damp_scale",
                        0.0,
                    )
                )
            ),
            0.0,
            1.0,
        )
    )
    predicted_grid_max_gain_scale = float(
        np.clip(
            1.0
            - (
                predicted_grid_uncertainty_share
                * float(
                    cfg.get(
                        "baseline_predictor.race.predicted_grid_uncertainty.max_gain_damp_scale",
                        0.0,
                    )
                )
            ),
            0.0,
            1.0,
        )
    )
    dnf_probability_output_cap = float(cfg.get("baseline_predictor.race.dnf_rate_final_cap", 0.35))
    learning_position_scale = float(
        cfg.get("baseline_predictor.race.learning.position_adjustment_scale", 0.70)
    )

    return _FinishOrderConfig(
        confidence_floor=confidence_floor,
        confidence_cap=confidence_cap,
        context_confidence=context_confidence,
        low_confidence_share=low_confidence_share,
        context_confidence_adjustment=context_confidence_adjustment,
        weather_penalty=weather_penalty,
        predicted_grid_uncertainty_share=predicted_grid_uncertainty_share,
        grid_anchor_weight=float(grid_anchor_weight),
        overtake_blend_scale=overtake_blend_scale,
        race_adv_blend_scale=race_adv_blend_scale,
        skill_blend_scale=skill_blend_scale,
        elite_skill_threshold=elite_skill_threshold,
        elite_driver_scale=elite_driver_scale,
        elite_driver_exponent=elite_driver_exponent,
        max_driver_adjustment=max_driver_adjustment,
        max_gain_base=max_gain_base,
        max_gain_track_scale=max_gain_track_scale,
        skill_gain_scale=skill_gain_scale,
        race_adv_gain_scale=race_adv_gain_scale,
        max_gain_floor=max_gain_floor,
        max_gain_ceiling=max_gain_ceiling,
        racecraft_confidence_scale=racecraft_confidence_scale,
        max_gain_confidence_scale=max_gain_confidence_scale,
        predicted_grid_racecraft_scale=predicted_grid_racecraft_scale,
        predicted_grid_max_gain_scale=predicted_grid_max_gain_scale,
        dnf_probability_output_cap=dnf_probability_output_cap,
        learning_position_scale=learning_position_scale,
        track_overtaking=track_overtaking,
    )


def build_finish_order(
    *,
    aggregated: dict[str, Any],
    driver_info_map: dict[str, dict[str, Any]],
    grid_position_samples_by_driver: dict[str, list[float]],
    field_size: int,
    weather: str,
    is_sprint: bool,
    input_confidence: float | None,
    cfg: Any,
    race_params: dict[str, Any],
    weather_feature_modifiers: dict[str, float],
    get_learned_position_adjustment: Any,
    learned_interval_radius: float,
    enforce_non_increasing: Any,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Convert aggregated simulation outputs into the final finish-order payload."""
    fo_cfg = _load_finish_order_config(
        cfg=cfg,
        weather=weather,
        weather_feature_modifiers=weather_feature_modifiers,
        race_params=race_params,
        input_confidence=input_confidence,
        is_sprint=is_sprint,
        grid_position_samples_by_driver=grid_position_samples_by_driver,
        field_size=field_size,
    )

    finish_order: list[dict[str, Any]] = []
    blended_samples_by_driver: dict[str, list[float]] = {}
    team_to_drivers: dict[str, list[str]] = {}
    for driver_code, info in driver_info_map.items():
        team_to_drivers.setdefault(info["team"], []).append(driver_code)
    grid_reference_positions = {
        driver_code: (
            float(np.mean(grid_position_samples_by_driver.get(driver_code, [])))
            if grid_position_samples_by_driver.get(driver_code)
            else float(info["grid_pos"])
        )
        for driver_code, info in driver_info_map.items()
    }

    for driver_code, median_pos in aggregated["median_positions"].items():
        info = driver_info_map[driver_code]
        positions = aggregated["position_distributions"][driver_code]
        reference_grid_pos = float(grid_reference_positions.get(driver_code, info["grid_pos"]))
        grid_position_samples = list(grid_position_samples_by_driver.get(driver_code, []))
        if len(grid_position_samples) != len(positions):
            grid_position_samples = [reference_grid_pos for _ in positions]

        position_std = np.std(positions)
        confidence = max(
            fo_cfg.confidence_floor,
            min(
                fo_cfg.confidence_cap,
                fo_cfg.confidence_cap
                - (position_std * 3.0)
                - fo_cfg.weather_penalty
                + fo_cfg.context_confidence_adjustment,
            ),
        )

        overtake_ease = 1.0 - fo_cfg.track_overtaking
        racecraft_adjustment = (
            ((info["overtaking_skill"] - 0.5) * overtake_ease * fo_cfg.overtake_blend_scale)
            + (info["race_advantage"] * fo_cfg.race_adv_blend_scale)
            + ((info["skill"] - 0.5) * fo_cfg.skill_blend_scale)
        )
        elite_denominator = max(1e-6, 1.0 - fo_cfg.elite_skill_threshold)
        elite_driver_normalized = max(
            0.0,
            (info["skill"] - fo_cfg.elite_skill_threshold) / elite_denominator,
        )
        elite_driver_adjustment = (
            (elite_driver_normalized**fo_cfg.elite_driver_exponent)
            * fo_cfg.elite_driver_scale
            * (0.6 + (0.4 * overtake_ease))
        )
        racecraft_adjustment += elite_driver_adjustment
        racecraft_adjustment *= fo_cfg.racecraft_confidence_scale
        racecraft_adjustment *= fo_cfg.predicted_grid_racecraft_scale

        is_elite_driver = info["skill"] >= fo_cfg.elite_skill_threshold
        if reference_grid_pos <= 3.0 and not is_elite_driver:
            adjustment_cap_negative = fo_cfg.max_driver_adjustment * 0.5
            adjustment_cap_positive = fo_cfg.max_driver_adjustment
            racecraft_adjustment = np.clip(
                racecraft_adjustment,
                -adjustment_cap_negative,
                adjustment_cap_positive,
            )
        else:
            racecraft_adjustment = np.clip(
                racecraft_adjustment,
                -fo_cfg.max_driver_adjustment,
                fo_cfg.max_driver_adjustment,
            )

        max_gain = (
            fo_cfg.max_gain_base
            + (overtake_ease * fo_cfg.max_gain_track_scale)
            + ((info["overtaking_skill"] - 0.5) * fo_cfg.skill_gain_scale)
            + (max(0.0, info["race_advantage"]) * fo_cfg.race_adv_gain_scale)
        )
        max_gain *= fo_cfg.max_gain_confidence_scale
        max_gain *= fo_cfg.predicted_grid_max_gain_scale
        max_gain = np.clip(max_gain, fo_cfg.max_gain_floor, fo_cfg.max_gain_ceiling)
        min_position_score = max(1.0, reference_grid_pos - max_gain)

        position_blend_score = (
            ((1.0 - fo_cfg.grid_anchor_weight) * median_pos)
            + (fo_cfg.grid_anchor_weight * reference_grid_pos)
            - racecraft_adjustment
        )
        learned_position_adjustment = get_learned_position_adjustment(
            team=info["team"],
            driver=driver_code,
            teammates=team_to_drivers.get(info["team"], []),
            session="race",
            races_completed=info.get("season_races_completed"),
        )
        position_blend_score -= learned_position_adjustment * fo_cfg.learning_position_scale
        position_blend_score = max(position_blend_score, min_position_score)

        blended_position_samples = []
        for position_sample, grid_position_sample in zip(
            positions,
            grid_position_samples,
            strict=False,
        ):
            min_sample_position_score = max(1.0, float(grid_position_sample) - max_gain)
            blended_position_samples.append(
                max(
                    (
                        ((1.0 - fo_cfg.grid_anchor_weight) * position_sample)
                        + (fo_cfg.grid_anchor_weight * float(grid_position_sample))
                        - racecraft_adjustment
                    ),
                    min_sample_position_score,
                )
            )
        if learned_position_adjustment:
            blended_position_samples = [
                sample - (learned_position_adjustment * fo_cfg.learning_position_scale)
                for sample in blended_position_samples
            ]
        blended_position_samples = [
            max(sample_position, min_position_score) for sample_position in blended_position_samples
        ]
        position_blend_score, blended_position_samples = apply_hypothetical_points_floor(
            info=info,
            position_blend_score=float(position_blend_score),
            blended_position_samples=blended_position_samples,
            reference_grid_pos=reference_grid_pos,
            field_size=len(driver_info_map),
            cfg=cfg,
        )
        blended_samples_by_driver[driver_code] = blended_position_samples

        finish_order.append(
            {
                "driver": driver_code,
                "team": info["team"],
                "median_position": median_pos,
                "position_blend_score": round(position_blend_score, 4),
                "p5": int(np.percentile(blended_position_samples, 5)),
                "p95": int(np.percentile(blended_position_samples, 95)),
                "confidence": round(confidence, 1),
                "podium_probability": 0.0,
                "dnf_probability": round(
                    float(
                        np.clip(
                            aggregated["dnf_rates"].get(driver_code, 0.0),
                            0.0,
                            fo_cfg.dnf_probability_output_cap,
                        )
                    ),
                    3,
                ),
            }
        )

    podium_prob_by_driver, rank_samples_by_driver = _compute_podium_probabilities(
        blended_samples_by_driver=blended_samples_by_driver,
        cfg=cfg,
        base_seed=base_seed,
    )
    for row in finish_order:
        row["podium_probability"] = round(podium_prob_by_driver.get(row["driver"], 0.0), 1)
        rank_samples = rank_samples_by_driver.get(row["driver"], [])
        if rank_samples:
            row["position_blend_score"] = round(float(np.mean(rank_samples)), 4)
            row["median_position"] = int(np.median(rank_samples))
            row["p5"] = int(np.percentile(rank_samples, 5))
            row["p95"] = int(np.percentile(rank_samples, 95))

    apply_early_season_team_uncertainty_adjustments(
        finish_order=finish_order,
        driver_info_map=driver_info_map,
        cfg=cfg,
        field_size=max(1, field_size),
    )

    finish_order.sort(key=lambda item: item["position_blend_score"])
    for index, item in enumerate(finish_order, start=1):
        item["position"] = index

    if not is_sprint and blended_samples_by_driver:
        apply_main_race_movement_constraints(
            finish_order=finish_order,
            blended_samples_by_driver=blended_samples_by_driver,
            driver_info_map=driver_info_map,
            grid_reference_positions=grid_reference_positions,
            track_overtaking=float(fo_cfg.track_overtaking),
            cfg=cfg,
        )

    if cfg.get("baseline_predictor.race.podium_probability.enforce_monotonic", True):
        raw_podium_values = [float(row.get("podium_probability", 0.0)) for row in finish_order]
        smoothed_values = enforce_non_increasing(raw_podium_values)
        for row, smoothed in zip(finish_order, smoothed_values, strict=True):
            row["podium_probability"] = round(float(np.clip(smoothed, 0.0, 100.0)), 1)

    apply_learned_interval_radius(
        finish_order=finish_order,
        learned_interval_radius=learned_interval_radius,
        field_size=max(1, field_size),
    )
    apply_low_confidence_interval_floor(
        finish_order=finish_order,
        input_confidence=input_confidence,
        cfg=cfg,
        field_size=max(1, field_size),
    )
    return finish_order


def _compute_podium_probabilities(
    *,
    blended_samples_by_driver: dict[str, list[float]],
    cfg: Any,
    base_seed: int,
) -> tuple[dict[str, float], dict[str, list[int]]]:
    """Estimate podium probability from blended position samples."""
    podium_prob_by_driver: dict[str, float] = {}
    rank_samples_by_driver: dict[str, list[int]] = {
        driver: [] for driver in blended_samples_by_driver.keys()
    }
    if not blended_samples_by_driver:
        return podium_prob_by_driver, rank_samples_by_driver

    sample_lengths = [len(samples) for samples in blended_samples_by_driver.values()]
    sample_count = min(sample_lengths) if sample_lengths else 0
    if sample_count <= 0:
        return podium_prob_by_driver, rank_samples_by_driver

    podium_counts: dict[str, int] = {driver: 0 for driver in blended_samples_by_driver}
    minimum_podium_samples = int(
        cfg.get("baseline_predictor.race.podium_probability.min_sample_count", 250)
    )
    minimum_podium_samples = max(1, minimum_podium_samples)
    use_resampling = sample_count < minimum_podium_samples
    draw_count = minimum_podium_samples if use_resampling else sample_count

    resample_rng = None
    if use_resampling:
        resample_seed_offset = int(
            cfg.get("baseline_predictor.race.podium_probability.resample_seed_offset", 99173)
        )
        resample_rng = np.random.default_rng(base_seed + resample_seed_offset)

    for sample_idx in range(draw_count):
        if resample_rng is None:
            ranked_scores = [
                (driver_code, samples[sample_idx])
                for driver_code, samples in blended_samples_by_driver.items()
            ]
        else:
            ranked_scores = [
                (
                    driver_code,
                    samples[int(resample_rng.integers(0, sample_count))],
                )
                for driver_code, samples in blended_samples_by_driver.items()
            ]

        ranked = sorted(ranked_scores, key=lambda item: (item[1], item[0]))
        for rank_index, (driver_code, _) in enumerate(ranked, start=1):
            rank_samples_by_driver[driver_code].append(rank_index)
        for driver_code, _ in ranked[:3]:
            podium_counts[driver_code] += 1

    podium_prob_by_driver = {
        driver: (count / draw_count) * 100.0 for driver, count in podium_counts.items()
    }
    return podium_prob_by_driver, rank_samples_by_driver


def apply_main_race_movement_constraints(
    *,
    finish_order: list[dict[str, Any]],
    blended_samples_by_driver: dict[str, list[float]],
    driver_info_map: dict[str, dict[str, Any]],
    grid_reference_positions: dict[str, float],
    track_overtaking: float,
    cfg: Any,
) -> None:
    """Keep main-race grid movement within configured realism bounds."""
    base_movement_floor = float(cfg.get("baseline_predictor.race.main_race_movement_floor", 1.0))
    movement_quantile = float(cfg.get("baseline_predictor.race.main_race_movement_quantile", 20.0))
    movement_ceiling_base = float(
        cfg.get("baseline_predictor.race.main_race_movement_ceiling_base", 2.5)
    )
    movement_ceiling_track_scale = float(
        cfg.get("baseline_predictor.race.main_race_movement_ceiling_track_scale", 0.70)
    )
    movement_ceiling_min = float(
        cfg.get("baseline_predictor.race.main_race_movement_ceiling_min", base_movement_floor)
    )
    movement_ceiling = max(
        movement_ceiling_min,
        movement_ceiling_base
        - (float(np.clip(track_overtaking, 0.0, 1.0)) * movement_ceiling_track_scale),
    )
    movement_floor_track_scale = float(
        cfg.get("baseline_predictor.race.main_race_movement_floor_track_scale", 0.25)
    )
    overtake_ease = 1.0 - float(np.clip(track_overtaking, 0.0, 1.0))
    movement_floor = min(
        movement_ceiling,
        base_movement_floor + (overtake_ease * movement_floor_track_scale),
    )

    avg_grid_change = _avg_grid_change(
        rows=finish_order,
        driver_info_map=driver_info_map,
        grid_reference_positions=grid_reference_positions,
    )
    if avg_grid_change < movement_floor:
        quantile_candidates = [movement_quantile, 20.0, 10.0, 0.0]
        used_quantiles: set[float] = set()
        for quantile_candidate in quantile_candidates:
            quantile = float(np.clip(quantile_candidate, 0.0, 50.0))
            if quantile in used_quantiles:
                continue
            used_quantiles.add(quantile)

            quantile_scores = {
                driver: float(np.percentile(samples, quantile))
                for driver, samples in blended_samples_by_driver.items()
                if samples
            }
            _apply_score_ranking(finish_order=finish_order, scores=quantile_scores)
            avg_grid_change = _avg_grid_change(
                rows=finish_order,
                driver_info_map=driver_info_map,
                grid_reference_positions=grid_reference_positions,
            )
            if avg_grid_change >= movement_floor:
                break

        if avg_grid_change < movement_floor:
            sample_lengths = [
                len(samples) for samples in blended_samples_by_driver.values() if samples
            ]
            sample_count = min(sample_lengths) if sample_lengths else 0
            if sample_count > 0:
                selected_scores: dict[str, float] | None = None
                selected_avg: float | None = None
                strongest_scores: dict[str, float] | None = None
                strongest_avg = -1.0

                for sample_idx in range(sample_count):
                    candidate_scores = {
                        driver: float(samples[sample_idx])
                        for driver, samples in blended_samples_by_driver.items()
                        if len(samples) > sample_idx
                    }
                    candidate_avg = _avg_grid_change_for_scores(
                        finish_order=finish_order,
                        scores=candidate_scores,
                        driver_info_map=driver_info_map,
                        grid_reference_positions=grid_reference_positions,
                    )
                    if candidate_avg > strongest_avg:
                        strongest_avg = candidate_avg
                        strongest_scores = candidate_scores
                    if candidate_avg >= movement_floor and (
                        selected_avg is None or candidate_avg < selected_avg
                    ):
                        selected_avg = candidate_avg
                        selected_scores = candidate_scores

                final_scores = selected_scores or strongest_scores
                if final_scores:
                    _apply_score_ranking(finish_order=finish_order, scores=final_scores)

    avg_grid_change = _avg_grid_change(
        rows=finish_order,
        driver_info_map=driver_info_map,
        grid_reference_positions=grid_reference_positions,
    )
    if avg_grid_change <= movement_ceiling:
        return

    base_scores = {row["driver"]: float(row["position_blend_score"]) for row in finish_order}
    selected_ceiling_scores: dict[str, float] | None = None
    closest_scores: dict[str, float] | None = None
    closest_delta = float("inf")
    keep_factors = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30]

    for keep_factor in keep_factors:
        candidate_ceiling_scores: dict[str, float] = {}
        for row in finish_order:
            driver_code = row["driver"]
            info = driver_info_map.get(driver_code)
            if info is None:
                continue
            reference_grid_pos = float(grid_reference_positions.get(driver_code, info["grid_pos"]))
            candidate_ceiling_scores[driver_code] = (
                keep_factor * base_scores.get(driver_code, float(row["position_blend_score"]))
            ) + ((1.0 - keep_factor) * reference_grid_pos)

        candidate_avg = _avg_grid_change_for_scores(
            finish_order=finish_order,
            scores=candidate_ceiling_scores,
            driver_info_map=driver_info_map,
            grid_reference_positions=grid_reference_positions,
        )
        if movement_floor <= candidate_avg <= movement_ceiling:
            selected_ceiling_scores = candidate_ceiling_scores
            break
        if movement_floor <= candidate_avg < avg_grid_change:
            candidate_delta = abs(candidate_avg - movement_ceiling)
            if candidate_delta < closest_delta:
                closest_delta = candidate_delta
                closest_scores = candidate_ceiling_scores

    final_scores = selected_ceiling_scores or closest_scores
    if final_scores:
        _apply_score_ranking(finish_order=finish_order, scores=final_scores)


def _avg_grid_change(
    *,
    rows: list[dict[str, Any]],
    driver_info_map: dict[str, dict[str, Any]],
    grid_reference_positions: dict[str, float],
) -> float:
    """Return average absolute grid movement for ranked rows."""
    total_grid_change = 0.0
    total_drivers = 0
    for row in rows:
        info = driver_info_map.get(row["driver"])
        if info is None:
            continue
        reference_grid_pos = float(grid_reference_positions.get(row["driver"], info["grid_pos"]))
        total_grid_change += abs(float(row["position"]) - reference_grid_pos)
        total_drivers += 1
    return (total_grid_change / total_drivers) if total_drivers else 0.0


def _apply_score_ranking(
    *,
    finish_order: list[dict[str, Any]],
    scores: dict[str, float],
) -> None:
    """Re-rank finish order using candidate scores and preserve tie stability."""
    finish_order.sort(
        key=lambda row: (
            scores.get(row["driver"], float(row["position_blend_score"])),
            float(row["position_blend_score"]),
            row["driver"],
        )
    )
    for index, row in enumerate(finish_order, start=1):
        row["position"] = index
        if row["driver"] in scores:
            row["position_blend_score"] = round(scores[row["driver"]], 4)


def _avg_grid_change_for_scores(
    *,
    finish_order: list[dict[str, Any]],
    scores: dict[str, float],
    driver_info_map: dict[str, dict[str, Any]],
    grid_reference_positions: dict[str, float],
) -> float:
    """Return average grid movement implied by a score ranking candidate."""
    total_grid_change = 0.0
    total_drivers = 0
    ranked_rows = sorted(
        finish_order,
        key=lambda row: (
            scores.get(row["driver"], float(row["position_blend_score"])),
            float(row["position_blend_score"]),
            row["driver"],
        ),
    )
    for index, row in enumerate(ranked_rows, start=1):
        info = driver_info_map.get(row["driver"])
        if info is None:
            continue
        reference_grid_pos = float(grid_reference_positions.get(row["driver"], info["grid_pos"]))
        total_grid_change += abs(float(index) - reference_grid_pos)
        total_drivers += 1
    return (total_grid_change / total_drivers) if total_drivers else 0.0
