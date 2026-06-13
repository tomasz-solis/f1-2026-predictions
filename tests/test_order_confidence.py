"""Tests for the calibrated order-confidence metric."""

from __future__ import annotations

import numpy as np

from src.models.order_confidence import compute_order_confidence, within_tolerance


def test_dominant_position_reads_high() -> None:
    """A car that almost always finishes in its slot should read far above 60%."""
    samples = [1] * 90 + [2] * 8 + [3] * 2
    confidence = compute_order_confidence(
        samples=samples,
        predicted_position=1,
        tolerance=1.0,
    )
    assert confidence > 90.0


def test_volatile_midfield_reads_low() -> None:
    """A widely scattered midfield slot should read honestly low, not floored at 40."""
    rng = np.random.default_rng(0)
    samples = rng.integers(6, 17, size=400).tolist()
    confidence = compute_order_confidence(
        samples=samples,
        predicted_position=11,
        tolerance=1.0,
    )
    assert confidence < 45.0
    assert confidence >= 2.0


def test_not_capped_at_sixty() -> None:
    """The metric must be able to exceed the legacy 60% ceiling."""
    samples = [5] * 100
    confidence = compute_order_confidence(samples=samples, predicted_position=5, tolerance=0.0)
    assert confidence > 60.0


def test_published_interval_widening_lowers_confidence() -> None:
    """Widening the published interval beyond the raw spread reduces confidence."""
    samples = ([4] * 50) + ([5] * 50)
    tight = compute_order_confidence(
        samples=samples,
        predicted_position=4,
        tolerance=1.0,
        published_p5=4,
        published_p95=5,
    )
    widened = compute_order_confidence(
        samples=samples,
        predicted_position=4,
        tolerance=1.0,
        published_p5=1,
        published_p95=12,
    )
    assert widened < tight


def test_spread_inflation_lowers_confidence() -> None:
    """A larger spread_inflation must not raise confidence."""
    samples = list(range(1, 21))
    base = compute_order_confidence(samples=samples, predicted_position=10, spread_inflation=1.0)
    inflated = compute_order_confidence(
        samples=samples, predicted_position=10, spread_inflation=2.0
    )
    assert inflated <= base


def test_clamps_respected() -> None:
    """Output stays within the configured clamp bounds."""
    samples = [3] * 100
    confidence = compute_order_confidence(
        samples=samples,
        predicted_position=3,
        tolerance=0.0,
        conf_max=88.0,
    )
    assert confidence <= 88.0


def test_empty_samples_returns_midpoint() -> None:
    """No samples falls back to the clamp midpoint rather than crashing."""
    confidence = compute_order_confidence(
        samples=[],
        predicted_position=5,
        conf_min=10.0,
        conf_max=90.0,
    )
    assert confidence == 50.0


def test_within_tolerance_matches_metric_definition() -> None:
    """The backtest event helper agrees with the metric's band definition."""
    assert within_tolerance(predicted_position=5, actual_position=6, tolerance=1.0)
    assert within_tolerance(predicted_position=5, actual_position=4, tolerance=1.0)
    assert not within_tolerance(predicted_position=5, actual_position=7, tolerance=1.0)
