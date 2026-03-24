"""Testing-profile helpers for baseline predictor strength adjustments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def get_testing_characteristics_for_profile(
    *,
    resolve_team_data: Callable[[str], dict[str, Any]],
    team: str,
    profile: str,
) -> dict[str, float]:
    """Return testing characteristics for one run profile with older-file fallbacks."""
    team_data = resolve_team_data(team)

    profile_store = team_data.get("testing_characteristics_profiles")
    if isinstance(profile_store, dict):
        profile_data = profile_store.get(profile)
        if isinstance(profile_data, dict):
            return profile_data

    fallback = team_data.get("testing_characteristics")
    if not isinstance(fallback, dict):
        return {}

    fallback_profile = fallback.get("run_profile")
    if fallback_profile == profile:
        return fallback
    if profile == "balanced":
        return fallback
    return {}


def compute_testing_profile_modifier(
    *,
    team: str,
    profile: str,
    metric_weights: dict[str, float],
    scale: float,
    get_testing_characteristics_for_profile_fn: Callable[[str, str], dict[str, float]],
    cfg: Any,
) -> tuple[float, bool]:
    """Compute a bounded modifier from testing or practice profile metrics."""
    profile_metrics = get_testing_characteristics_for_profile_fn(team, profile)
    if not profile_metrics:
        return 0.0, False

    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in metric_weights.items():
        value = profile_metrics.get(metric_name)
        if value is None:
            continue
        try:
            centered = float(value) - 0.5
        except (TypeError, ValueError):
            continue
        weighted_sum += centered * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0, False

    normalized_centered = weighted_sum / total_weight
    clip_range = cfg.get(
        "baseline_predictor.race.testing_modifier_clip_range",
        [-0.04, 0.04],
    )
    if isinstance(clip_range, list) and len(clip_range) == 2 and clip_range[0] < clip_range[1]:
        min_clip, max_clip = clip_range
    else:
        min_clip, max_clip = -0.04, 0.04

    modifier = float(np.clip(normalized_centered * scale, min_clip, max_clip))
    return modifier, True
