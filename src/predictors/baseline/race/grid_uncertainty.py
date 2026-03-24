"""Grid uncertainty helpers for race prediction flow."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.types.prediction_types import QualifyingGridEntry


def normalize_confidence_to_unit_interval(value: Any) -> float:
    """Convert confidence values expressed as 0-1 or 0-100 into a 0-1 scale."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0

    if confidence > 1.0:
        confidence /= 100.0
    return float(np.clip(confidence, 0.0, 1.0))


def coerce_grid_position_metric(row: QualifyingGridEntry, key: str, fallback: int) -> int:
    """Read an integer grid metric from a qualifying row with a safe fallback."""
    raw_value: Any = row.get(key, fallback)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(fallback)


def prepare_grid_uncertainty_profile(
    *,
    validated_grid: list[QualifyingGridEntry],
    input_confidence: float | None,
    cfg: Any,
) -> dict[str, dict[str, float]]:
    """Build per-driver grid-sampling settings from qualifying uncertainty fields."""
    if not validated_grid:
        return {}

    field_size = max(1, len(validated_grid))
    input_uncertainty = 0.0
    if input_confidence is not None:
        input_uncertainty = float(1.0 - np.clip(input_confidence, 0.0, 1.0))

    base_std = float(cfg.get("baseline_predictor.race.grid_uncertainty.base_std", 0.35))
    interval_divisor = float(
        max(
            1e-6,
            cfg.get("baseline_predictor.race.grid_uncertainty.interval_divisor", 3.29),
        )
    )
    confidence_scale = float(
        cfg.get("baseline_predictor.race.grid_uncertainty.confidence_scale", 0.90)
    )
    input_confidence_scale = float(
        cfg.get("baseline_predictor.race.grid_uncertainty.input_confidence_scale", 0.60)
    )
    position_delta_scale = float(
        cfg.get("baseline_predictor.race.grid_uncertainty.position_delta_scale", 0.35)
    )
    max_std = float(
        cfg.get(
            "baseline_predictor.race.grid_uncertainty.max_std",
            max(1.0, field_size / 4.0),
        )
    )

    profile: dict[str, dict[str, float]] = {}
    has_probabilistic_signal = False
    for row in validated_grid:
        driver = str(row["driver"])
        base_position = int(row["position"])
        center_position = coerce_grid_position_metric(row, "median_position", base_position)
        p5 = coerce_grid_position_metric(row, "p5", center_position)
        p95 = coerce_grid_position_metric(row, "p95", center_position)
        interval_width = max(0, p95 - p5)
        position_delta = abs(center_position - base_position)
        row_confidence = normalize_confidence_to_unit_interval(row.get("confidence", 1.0))
        has_row_signal = any(key in row for key in ("median_position", "p5", "p95", "confidence"))

        if not has_row_signal:
            profile[driver] = {"center": float(base_position), "std": 0.0}
            continue

        std = max(base_std, interval_width / interval_divisor)
        std += position_delta * position_delta_scale
        std *= (
            1.0
            + ((1.0 - row_confidence) * confidence_scale)
            + (input_uncertainty * input_confidence_scale)
        )
        std = float(np.clip(std, 0.0, max_std))
        has_probabilistic_signal = has_probabilistic_signal or std > 0.0
        profile[driver] = {"center": float(center_position), "std": std}

    return profile if has_probabilistic_signal else {}


def sample_probabilistic_grid_positions(
    *,
    validated_grid: list[QualifyingGridEntry],
    grid_uncertainty_profile: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Sample one coherent starting-grid permutation from qualifying uncertainty."""
    if not validated_grid:
        return {}
    if not grid_uncertainty_profile:
        return {str(row["driver"]): int(row["position"]) for row in validated_grid}

    latent_scores: list[tuple[str, float, int]] = []
    for row in validated_grid:
        driver = str(row["driver"])
        fallback_position = int(row["position"])
        uncertainty = grid_uncertainty_profile.get(driver, {})
        center = float(uncertainty.get("center", fallback_position))
        std = float(uncertainty.get("std", 0.0))
        latent_position = center if std <= 0.0 else float(rng.normal(center, std))
        latent_scores.append((driver, latent_position, fallback_position))

    ranked = sorted(latent_scores, key=lambda item: (item[1], item[2], item[0]))
    return {driver: index for index, (driver, _, _) in enumerate(ranked, start=1)}
