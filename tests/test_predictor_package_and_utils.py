"""Tests for predictor package imports."""

from src.predictors import Baseline2026Predictor
from src.predictors.baseline_2026 import Baseline2026Predictor as PredictorImpl


def test_predictor_package_exports_baseline_predictor():
    """The predictor package should re-export the baseline predictor at the top level."""
    assert Baseline2026Predictor is PredictorImpl
