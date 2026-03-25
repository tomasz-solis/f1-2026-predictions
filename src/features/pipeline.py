"""Telemetry feature pipeline for driver summaries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .telemetry import LapFeatureExtractor, SessionFeatureAggregator

logger = logging.getLogger(__name__)


class RelativePerformanceCalculator:
    """Convert absolute features into field-relative deltas and percentiles."""

    def __init__(self, use_median: bool = True):
        """Choose whether relative metrics are centered on the median or mean."""
        self.use_median = use_median

    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add `<metric>_rel` columns based on the session baseline."""
        df = features_df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].notna().sum() < 2:
                continue

            baseline = df[col].median() if self.use_median else df[col].mean()
            df[f"{col}_rel"] = df[col] - baseline

        return df

    def add_percentile_ranks(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add percentile columns for lap-time and speed-style metrics."""
        df = features_df.copy()

        if "fastest_lap" in df.columns:
            df["fastest_lap_pct"] = df["fastest_lap"].rank(pct=True, ascending=True) * 100

        speed_cols = [col for col in df.columns if "speed" in col.lower() and "_rel" not in col]
        for col in speed_cols:
            if col in df.columns:
                df[f"{col}_pct"] = df[col].rank(pct=True, ascending=False) * 100

        return df


class F1FeaturePipeline:
    """Build a driver-level feature table from a FastF1 session object."""

    def __init__(self):
        """Initialize the extractor, aggregator, and relative-metric calculator."""
        self.lap_extractor = LapFeatureExtractor()
        self.session_aggregator = SessionFeatureAggregator(self.lap_extractor)
        self.rel_calculator = RelativePerformanceCalculator(use_median=True)

    def process_session(self, session, add_metadata: bool = True) -> pd.DataFrame:
        """Process one session into a feature table, optionally adding metadata."""
        features = self.session_aggregator.extract_all_drivers(session)

        if len(features) == 0:
            return pd.DataFrame()

        normalized = self.rel_calculator.normalize_features(features)
        with_ranks = self.rel_calculator.add_percentile_ranks(normalized)

        if add_metadata:
            with_ranks["year"] = session.event["EventDate"].year
            with_ranks["event"] = session.event["EventName"]
            with_ranks["session_type"] = session.name
            with_ranks["session_date"] = session.date

        return with_ranks

    def process_multiple_sessions(self, sessions, verbose: bool = True) -> pd.DataFrame:
        """Process multiple sessions and concatenate their feature tables."""
        all_features: list[pd.DataFrame] = []

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
