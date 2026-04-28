"""Promotion gates for model challengers and component stacks."""

from __future__ import annotations

from typing import Any


def _coerce_float(value: Any) -> float | None:
    """Return a float when the value is numeric enough for gate checks."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    """Return an integer count, defaulting missing values to zero."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def evaluate_component_promotion_gate(
    *,
    deltas: dict[str, Any],
    race_delta_summary: dict[str, Any] | None = None,
    min_total_mae_improvement: float = 0.02,
    central_mae_tolerance: float = 0.02,
    top3_tolerance_pp: float = 2.0,
) -> dict[str, Any]:
    """Evaluate whether a challenger is good enough to stack or promote.

    Positive MAE deltas mean the challenger improved against the champion.
    Accuracy deltas are percentage-point deltas where positive is better.
    """
    race_delta = _coerce_float(deltas.get("race_mae_improvement"))
    qualifying_delta = _coerce_float(deltas.get("qualifying_mae_improvement"))
    top3_delta = _coerce_float(deltas.get("top3_accuracy_delta"))
    winner_delta = _coerce_float(deltas.get("winner_accuracy_delta"))

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    checks["has_central_mae_deltas"] = race_delta is not None and qualifying_delta is not None
    if not checks["has_central_mae_deltas"]:
        reasons.append("missing race or qualifying MAE delta")
    race_mae_delta = race_delta if race_delta is not None else 0.0
    qualifying_mae_delta = qualifying_delta if qualifying_delta is not None else 0.0

    total_mae_improvement = float(race_mae_delta + qualifying_mae_delta)
    checks["improves_total_central_mae"] = total_mae_improvement >= min_total_mae_improvement
    if not checks["improves_total_central_mae"]:
        reasons.append(
            f"combined race and qualifying MAE improvement is below {min_total_mae_improvement:.3f}"
        )

    checks["race_mae_not_regressed"] = race_mae_delta >= -central_mae_tolerance
    if not checks["race_mae_not_regressed"]:
        reasons.append(f"race MAE regressed by more than {central_mae_tolerance:.3f}")

    checks["qualifying_mae_not_regressed"] = qualifying_mae_delta >= -central_mae_tolerance
    if not checks["qualifying_mae_not_regressed"]:
        reasons.append(f"qualifying MAE regressed by more than {central_mae_tolerance:.3f}")

    checks["winner_accuracy_not_regressed"] = winner_delta is None or winner_delta >= 0.0
    if not checks["winner_accuracy_not_regressed"]:
        reasons.append("winner accuracy dropped")

    checks["top3_accuracy_not_broadly_regressed"] = (
        top3_delta is None or top3_delta >= -top3_tolerance_pp
    )
    if not checks["top3_accuracy_not_broadly_regressed"]:
        reasons.append(f"top-3 accuracy dropped by more than {top3_tolerance_pp:.1f} pp")

    race_delta_summary = race_delta_summary or {}
    races_compared = _coerce_int(race_delta_summary.get("races_compared"))
    race_worse = _coerce_int(race_delta_summary.get("race_worse_count"))
    race_better = _coerce_int(race_delta_summary.get("race_better_count"))
    qualifying_worse = _coerce_int(race_delta_summary.get("qualifying_worse_count"))
    qualifying_better = _coerce_int(race_delta_summary.get("qualifying_better_count"))

    has_race_level_counts = races_compared > 0
    checks["race_weekends_not_broadly_worse"] = not has_race_level_counts or race_worse <= max(
        race_better, 1
    )
    if not checks["race_weekends_not_broadly_worse"]:
        reasons.append("race MAE got worse on more weekends than it improved")

    checks["qualifying_weekends_not_broadly_worse"] = (
        not has_race_level_counts or qualifying_worse <= max(qualifying_better, 1)
    )
    if not checks["qualifying_weekends_not_broadly_worse"]:
        reasons.append("qualifying MAE got worse on more weekends than it improved")

    passed = all(checks.values())
    return {
        "passed": bool(passed),
        "reasons": reasons,
        "checks": checks,
        "thresholds": {
            "min_total_mae_improvement": float(min_total_mae_improvement),
            "central_mae_tolerance": float(central_mae_tolerance),
            "top3_tolerance_pp": float(top3_tolerance_pp),
        },
        "total_mae_improvement": total_mae_improvement,
    }
