"""Performance scoring methods for driver ranking (vectorized)."""

import numpy as np
import pandas as pd

_SCORING_FEATURES = {
    "slow_corner": "slow_corner_speed",
    "medium_corner": "medium_corner_speed",
    "high_corner": "high_corner_speed",
    "straight": "avg_speed_full_throttle",
    "throttle_usage": "pct_full_throttle",
}


class PerformanceScoringMethod:
    """Base class for different scoring approaches."""

    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        """
        Score drivers on each characteristic.

        Parameters
        ----------
        testing_features : pd.DataFrame
            Features extracted from testing

        Returns
        -------
        pd.DataFrame
            Columns: driver_number, slow_corner_score, medium_corner_score, etc.
        """
        raise NotImplementedError


class AbsoluteDifferenceScoring(PerformanceScoringMethod):
    """Score = (value - median) in actual units.  Vectorized."""

    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        result = testing_features[["driver_number"]].copy()
        for metric_name, col in _SCORING_FEATURES.items():
            if col in testing_features.columns:
                result[f"{metric_name}_score"] = (
                    testing_features[col] - testing_features[col].median()
                )
            else:
                result[f"{metric_name}_score"] = np.nan
        return result


class RankingScoring(PerformanceScoringMethod):
    """Score = rank (1 = best, 20 = worst).  Vectorized."""

    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        result = testing_features[["driver_number"]].copy()
        for metric_name, col in _SCORING_FEATURES.items():
            if col in testing_features.columns:
                result[f"{metric_name}_score"] = testing_features[col].rank(
                    ascending=False, method="min", na_option="keep"
                )
            else:
                result[f"{metric_name}_score"] = np.nan
        return result


class QuantileScoring(PerformanceScoringMethod):
    """Score = quantile tier (3 = top 25%, 2 = middle 50%, 1 = bottom 25%).  Vectorized."""

    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        result = testing_features[["driver_number"]].copy()
        for metric_name, col in _SCORING_FEATURES.items():
            if col in testing_features.columns:
                series = testing_features[col]
                q75 = series.quantile(0.75)
                q25 = series.quantile(0.25)
                result[f"{metric_name}_score"] = np.where(
                    series.isna(),
                    np.nan,
                    np.where(series >= q75, 3, np.where(series >= q25, 2, 1)),
                )
            else:
                result[f"{metric_name}_score"] = np.nan
        return result


class ZScoreScoring(PerformanceScoringMethod):
    """Score = standardized z-score.  Vectorized."""

    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        result = testing_features[["driver_number"]].copy()
        for metric_name, col in _SCORING_FEATURES.items():
            if col in testing_features.columns:
                mean_val = testing_features[col].mean()
                std_val = testing_features[col].std()
                if std_val > 0:
                    result[f"{metric_name}_score"] = (testing_features[col] - mean_val) / std_val
                else:
                    result[f"{metric_name}_score"] = 0.0
            else:
                result[f"{metric_name}_score"] = np.nan
        return result
