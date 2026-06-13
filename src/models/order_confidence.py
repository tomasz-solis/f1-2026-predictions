"""Calibrated order-confidence metric.

Order confidence answers a concrete, calibratable question: *"What is the
probability this entrant finishes within ``tolerance`` positions of its
predicted slot?"* It is estimated directly from the Monte-Carlo finishing
-position samples that already back the published P5/P95 interval, so the
displayed number and the displayed range describe the **same** distribution.

Unlike the legacy ``cap - position_std * k`` heuristic it replaces, this value
is not squeezed into a narrow band: a dominant car can honestly read 85%+ and a
volatile midfield slot can honestly read ~30%. Because it is a genuine
probability, its calibration can be measured against replayed results (predicted
confidence vs. empirical within-tolerance hit rate) and tuned via
``spread_inflation``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_order_confidence(
    *,
    samples: Sequence[float] | np.ndarray,
    predicted_position: float,
    tolerance: float = 1.0,
    spread_inflation: float = 1.0,
    published_p5: float | None = None,
    published_p95: float | None = None,
    max_interval_scale: float = 3.0,
    conf_min: float = 2.0,
    conf_max: float = 99.0,
) -> float:
    """Return ``P(|finish - predicted_position| <= tolerance)`` as a 0-100 percent.

    Parameters
    ----------
    samples:
        Monte-Carlo finishing-position samples for one entrant (the same draws
        that produce the published P5/P95 interval).
    predicted_position:
        The position the entrant is published at; the probability is centred on
        this slot so the number answers "how likely is *this* placement right".
    tolerance:
        Half-width of the position band counted as a hit. ``1.0`` means "within
        one place"; ``0.0`` means the exact position.
    spread_inflation:
        Global multiplier on sample deviations used to calibrate the metric
        against backtest coverage. ``1.0`` leaves the empirical distribution
        untouched; values above ``1.0`` lower confidence to compensate for a
        simulation distribution that is narrower than reality.
    published_p5, published_p95:
        When provided, sample deviations are scaled so the metric reflects the
        *published* interval after any conformal / learned / early-season
        widening, keeping the displayed number consistent with the displayed
        range. Ignored when the published interval is not wider than the raw
        sample interval.
    max_interval_scale:
        Cap on the published-vs-raw interval scaling to avoid runaway widening.
    conf_min, conf_max:
        Output clamp. Defaults avoid dishonest absolutes (0% / 100%) while
        leaving the full informative range open.
    """
    sample_array = np.asarray(list(samples), dtype=float)
    sample_array = sample_array[np.isfinite(sample_array)]
    if sample_array.size == 0:
        midpoint = (float(conf_min) + float(conf_max)) / 2.0
        return float(round(np.clip(midpoint, conf_min, conf_max), 1))

    deviations = sample_array - float(predicted_position)

    interval_scale = 1.0
    if published_p5 is not None and published_p95 is not None:
        raw_half_width = (
            float(np.percentile(sample_array, 95)) - float(np.percentile(sample_array, 5))
        ) / 2.0
        published_half_width = (float(published_p95) - float(published_p5)) / 2.0
        if raw_half_width > 1e-6 and published_half_width > raw_half_width:
            interval_scale = float(
                np.clip(
                    published_half_width / raw_half_width,
                    1.0,
                    max(1.0, float(max_interval_scale)),
                )
            )

    scale = interval_scale * max(1e-6, float(spread_inflation))
    scaled_abs_deviation = np.abs(deviations) * scale

    # +0.5 keeps the integer position band [pos-tol, pos+tol] intact once the
    # deviations have been scaled into a continuous quantity.
    band = max(0.0, float(tolerance)) + 0.5
    within_fraction = float(np.mean(scaled_abs_deviation <= band)) * 100.0
    return float(round(np.clip(within_fraction, float(conf_min), float(conf_max)), 1))


def within_tolerance(
    *,
    predicted_position: float,
    actual_position: float,
    tolerance: float = 1.0,
) -> bool:
    """Return whether an actual result landed within ``tolerance`` of prediction.

    Shared by the backtest calibration metric so the scored event matches the
    definition :func:`compute_order_confidence` estimates.
    """
    return abs(float(actual_position) - float(predicted_position)) <= max(0.0, float(tolerance))
