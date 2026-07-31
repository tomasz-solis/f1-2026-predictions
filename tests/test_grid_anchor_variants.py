"""Focused coverage for opt-in post-simulation grid-anchor challengers."""

from __future__ import annotations

import pytest

from src.predictors.baseline.race.result_processing import _resolve_grid_anchor_variant


class _Config:
    def __init__(self, variant: str, overrides: dict[str, object] | None = None) -> None:
        self._values: dict[str, object] = {
            "baseline_predictor.model_variant": variant,
            **(overrides or {}),
        }

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)


def test_champion_preserves_resolved_anchor_weight() -> None:
    weight, diagnostics = _resolve_grid_anchor_variant(
        champion_weight=0.57,
        cfg=_Config("champion"),
        grid_source_detail="predicted_joint",
    )

    assert weight == pytest.approx(0.57)
    assert diagnostics == {
        "model_variant": "champion",
        "requested": "champion",
        "applied": False,
        "fallback_reason": None,
        "champion_weight": 0.57,
        "effective_weight": 0.57,
        "source_detail": "predicted_joint",
    }


def test_no_anchor_variant_sets_only_final_anchor_to_zero() -> None:
    weight, diagnostics = _resolve_grid_anchor_variant(
        champion_weight=0.57,
        cfg=_Config("r2_no_anchor"),
        grid_source_detail="actual_starting_grid",
    )

    assert weight == 0.0
    assert diagnostics["requested"] == "r2_no_anchor"
    assert diagnostics["applied"] is True
    assert diagnostics["fallback_reason"] is None
    assert diagnostics["effective_weight"] == 0.0


def test_source_anchor_uses_explicit_matching_calibration() -> None:
    weight, diagnostics = _resolve_grid_anchor_variant(
        champion_weight=0.57,
        cfg=_Config(
            "r2_source_anchor",
            {
                "baseline_predictor.race.grid_anchor.source_calibrated.predicted_joint": 0.31,
            },
        ),
        grid_source_detail="predicted_joint",
    )

    assert weight == pytest.approx(0.31)
    assert diagnostics["requested"] == "r2_source_anchor"
    assert diagnostics["applied"] is True
    assert diagnostics["effective_weight"] == pytest.approx(0.31)


@pytest.mark.parametrize(
    ("source_detail", "configured_value", "expected_reason"),
    [
        (None, 0.31, "missing_grid_source_detail"),
        ("predicted_joint", None, "missing_source_calibration"),
        ("predicted_joint", "bad", "invalid_source_calibration"),
        ("predicted_joint", float("nan"), "invalid_source_calibration"),
        ("predicted_joint", 1.1, "invalid_source_calibration"),
        ("predicted_joint", True, "invalid_source_calibration"),
    ],
)
def test_source_anchor_fails_closed_to_champion(
    source_detail: str | None,
    configured_value: object,
    expected_reason: str,
) -> None:
    overrides = {}
    if configured_value is not None:
        overrides["baseline_predictor.race.grid_anchor.source_calibrated.predicted_joint"] = (
            configured_value
        )

    weight, diagnostics = _resolve_grid_anchor_variant(
        champion_weight=0.57,
        cfg=_Config("r2_source_anchor", overrides),
        grid_source_detail=source_detail,
    )

    assert weight == pytest.approx(0.57)
    assert diagnostics["applied"] is False
    assert diagnostics["fallback_reason"] == expected_reason
    assert diagnostics["effective_weight"] == pytest.approx(0.57)
