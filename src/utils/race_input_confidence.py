"""Helpers for calibrating race-input confidence from qualifying context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.utils import config_loader


def derive_race_input_confidence(
    qualifying_result: Mapping[str, Any],
    *,
    grid_source: str,
) -> float:
    """Estimate race-grid confidence from qualifying quality and grid provenance."""
    resolved_grid_source = str(grid_source).strip().upper()
    if resolved_grid_source == "ACTUAL":
        return 1.0

    try:
        base_confidence = float(qualifying_result.get("data_confidence_score", 0.5))
    except (TypeError, ValueError):
        base_confidence = 0.5
    base_confidence = float(max(0.0, min(base_confidence, 1.0)))

    data_source = str(qualifying_result.get("data_source", "")).strip().lower()
    source_adjustment = 0.0
    if "model-only" in data_source:
        source_adjustment = -0.10
    elif _uses_testing_or_checkpoint_blend(
        qualifying_result=qualifying_result,
        normalized_data_source=data_source,
    ):
        source_adjustment = -0.05

    return float(max(0.0, min(base_confidence + source_adjustment, 1.0)))


def _uses_testing_or_checkpoint_blend(
    *,
    qualifying_result: Mapping[str, Any],
    normalized_data_source: str,
) -> bool:
    """Return whether qualifying input came from a weaker inferred pace source."""
    if bool(qualifying_result.get("testing_fallback_used")):
        return True
    return (
        "checkpoint profile blend" in normalized_data_source
        or "testing short-run profile blend" in normalized_data_source
    )


def cap_predicted_main_race_input_confidence(
    input_confidence: float,
    *,
    qualifying_result: Mapping[str, Any],
    grid_source: str,
    is_sprint_weekend: bool,
    boundary_session_name: str | None = None,
    config: Any | None = None,
) -> float:
    """
    Cap Sunday-race grid confidence when sprint weekends still rely on a predicted main grid.

    Before main qualifying is actually run on a sprint weekend, the Grand Prix race input grid
    is still only an inferred forecast. Treating that grid as highly reliable makes the race
    model over-anchor to an order we do not yet know. This helper keeps those pre-Q sprint
    weekend race runs more skeptical without changing normal weekends or actual-grid runs.
    """
    resolved_confidence = float(input_confidence)
    if not is_sprint_weekend:
        return resolved_confidence

    resolved_grid_source = str(grid_source).strip().upper()
    if resolved_grid_source == "ACTUAL":
        return resolved_confidence

    qualifying_stage = str(qualifying_result.get("qualifying_stage", "")).strip().lower()
    if qualifying_stage != "main":
        return resolved_confidence

    cfg = config or config_loader
    default_cap = float(
        cfg.get("baseline_predictor.race.main_race_predicted_grid_sprint_confidence_cap", 0.55)
    )
    checkpoint_caps = cfg.get(
        "baseline_predictor.race.main_race_predicted_grid_sprint_confidence_caps_by_checkpoint",
        {},
    )
    checkpoint_name = (
        str(boundary_session_name or qualifying_result.get("boundary_session_name", ""))
        .strip()
        .upper()
    )
    if isinstance(checkpoint_caps, Mapping) and checkpoint_name:
        configured_cap = checkpoint_caps.get(checkpoint_name, default_cap)
    else:
        configured_cap = default_cap

    confidence_cap = float(configured_cap)
    return min(resolved_confidence, confidence_cap)
