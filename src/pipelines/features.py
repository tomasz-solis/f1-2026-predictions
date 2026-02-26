"""Feature pipeline orchestration."""

import logging

import numpy as np
import pandas as pd

from ..features.telemetry import LapFeatureExtractor, SessionFeatureAggregator

logger = logging.getLogger(__name__)


class RelativePerformanceCalculator:
    """Convert absolute features to relative performance vs field."""

    def __init__(self, use_median=True):
        """
        use_median: If True, normalize to median (robust to outliers).
                   If False, normalize to mean.
        """
        self.use_median = use_median

    def normalize_features(self, features_df):
        """
        Add relative features: difference from field median/mean.
        Prefix: 'fastest_lap_rel', 'avg_throttle_rel', etc.
        """
        df = features_df.copy()

        # Identify numeric columns (skip metadata like driver_code)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].notna().sum() < 2:
                continue  # Skip if not enough data

            if self.use_median:
                baseline = df[col].median()
            else:
                baseline = df[col].mean()

            df[f"{col}_rel"] = df[col] - baseline

        return df

    def add_percentile_ranks(self, features_df):
        """
        Add percentile ranks for key features.
        Example: fastest_lap_pct = 95 means faster than 95% of field.
        """
        df = features_df.copy()

        # Lower is better for lap times
        if "fastest_lap" in df.columns:
            df["fastest_lap_pct"] = df["fastest_lap"].rank(pct=True, ascending=True) * 100

        # Higher is better for speed metrics
        speed_cols = [col for col in df.columns if "speed" in col.lower() and "_rel" not in col]
        for col in speed_cols:
            if col in df.columns:
                df[f"{col}_pct"] = df[col].rank(pct=True, ascending=False) * 100

        return df


class F1FeaturePipeline:
    """
    Complete feature extraction pipeline.

    Usage:
        pipeline = F1FeaturePipeline()
        features = pipeline.process_session(session)
    """

    def __init__(self):
        self.lap_extractor = LapFeatureExtractor()
        self.session_aggregator = SessionFeatureAggregator(self.lap_extractor)
        self.rel_calculator = RelativePerformanceCalculator(use_median=True)

    def process_session(self, session, add_metadata=True):
        """
        Complete pipeline: Session → Features with relative performance.

        Returns DataFrame with one row per driver.
        """
        # Step 1: Extract raw features
        features = self.session_aggregator.extract_all_drivers(session)

        if len(features) == 0:
            return pd.DataFrame()

        # Step 2: Calculate relative performance
        normalized = self.rel_calculator.normalize_features(features)
        with_ranks = self.rel_calculator.add_percentile_ranks(normalized)

        # Step 3: Add metadata
        if add_metadata:
            with_ranks["year"] = session.event["EventDate"].year
            with_ranks["event"] = session.event["EventName"]
            with_ranks["session_type"] = session.name
            with_ranks["session_date"] = session.date

        return with_ranks

    def process_multiple_sessions(self, sessions, verbose=True):
        """Process multiple sessions and combine."""
        all_features = []

        for i, session in enumerate(sessions):
            if verbose:
                logger.info(
                    "Processing %s/%s: %s - %s",
                    i + 1,
                    len(sessions),
                    session.event["EventName"],
                    session.name,
                )

            features = self.process_session(session)
            if len(features) > 0:
                all_features.append(features)

        if len(all_features) == 0:
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=True)

        if verbose:
            logger.info("Processed %s sessions", len(all_features))
            logger.info(
                "%s total rows, %s drivers",
                len(combined),
                combined["driver_number"].nunique(),
            )

        return combined
