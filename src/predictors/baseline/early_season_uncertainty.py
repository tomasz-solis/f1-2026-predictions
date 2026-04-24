"""Shared helpers for early-season team-uncertainty handling.

These helpers translate preseason team uncertainty into two runtime effects:

- wider published position intervals in the opening races
- lower published confidence while the model is still learning the field

The intent is simple: a regulation reset should begin with weaker priors and
visibly broader bands, then shrink as real race evidence arrives.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_early_season_uncertainty_share(
    *,
    team_uncertainty: float | None,
    races_completed: int | None,
    cfg: Any,
    prefix: str,
) -> float:
    """Return a 0-1 share for how much preseason uncertainty should still matter.

    The share is the product of two terms:

    - how far the team's uncertainty still sits above a calm baseline
    - how early we still are in the season

    By multiplying them, the effect fades naturally even for very uncertain
    teams once enough race weekends have been completed.
    """

    try:
        uncertainty_value = float(team_uncertainty if team_uncertainty is not None else 0.0)
    except (TypeError, ValueError):
        uncertainty_value = 0.0

    try:
        races_completed_value = int(races_completed if races_completed is not None else 0)
    except (TypeError, ValueError):
        races_completed_value = 0

    activation_floor = float(
        cfg.get(f"{prefix}.early_season_team_uncertainty.activation_floor", 0.22)
    )
    scale = float(cfg.get(f"{prefix}.early_season_team_uncertainty.scale", 0.22))
    decay_races = int(cfg.get(f"{prefix}.early_season_team_uncertainty.decay_races", 3))

    scale = max(scale, 1e-6)
    decay_races = max(decay_races, 1)
    uncertainty_share = float(np.clip((uncertainty_value - activation_floor) / scale, 0.0, 1.0))
    remaining_race_share = float(
        np.clip((decay_races - max(0, races_completed_value)) / decay_races, 0.0, 1.0)
    )
    return float(np.clip(uncertainty_share * remaining_race_share, 0.0, 1.0))


def resolve_early_season_interval_extension(
    *,
    team_uncertainty: float | None,
    races_completed: int | None,
    cfg: Any,
    prefix: str,
) -> int:
    """Return extra half-width positions to add to the published interval."""

    share = compute_early_season_uncertainty_share(
        team_uncertainty=team_uncertainty,
        races_completed=races_completed,
        cfg=cfg,
        prefix=prefix,
    )
    interval_scale = float(
        cfg.get(f"{prefix}.early_season_team_uncertainty.interval_positions_scale", 0.0)
    )
    return int(np.ceil(max(0.0, share * interval_scale)))


def resolve_early_season_confidence_penalty(
    *,
    team_uncertainty: float | None,
    races_completed: int | None,
    cfg: Any,
    prefix: str,
) -> float:
    """Return the confidence penalty, in points, from early-season uncertainty."""

    share = compute_early_season_uncertainty_share(
        team_uncertainty=team_uncertainty,
        races_completed=races_completed,
        cfg=cfg,
        prefix=prefix,
    )
    penalty_scale = float(
        cfg.get(f"{prefix}.early_season_team_uncertainty.confidence_penalty_scale", 0.0)
    )
    return float(max(0.0, share * penalty_scale))


def resolve_effective_learning_min_samples(
    *,
    configured_min_samples: int,
    races_completed: int | None,
) -> int:
    """Cap the adaptive-learning gate to evidence that could actually exist.

    Early in a season, especially after a regulation reset, waiting for three
    samples when only one or two prior races exist leaves the driver-ordering
    layer unable to react at all. We still honor the configured steady-state
    requirement, but we temporarily lower the gate to the number of completed
    races that the target prediction could genuinely have seen.
    """

    try:
        min_samples = int(configured_min_samples)
    except (TypeError, ValueError):
        min_samples = 1
    min_samples = max(min_samples, 1)

    try:
        available_races = int(races_completed) if races_completed is not None else None
    except (TypeError, ValueError):
        available_races = None

    if available_races is None or available_races <= 0:
        return min_samples

    return min(min_samples, max(available_races, 1))
