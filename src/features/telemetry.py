"""Feature extraction from F1 telemetry data."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LapFeatureExtractor:
    """Extract telemetry features from a single F1 lap."""

    def __init__(
        self,
        corner_speed_thresholds: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """
        Corner speed thresholds for classification.
        Default: slow <100, medium 100-200, high 200-250 km/h
        """
        if corner_speed_thresholds is None:
            self.corner_thresholds = {
                "slow": (0, 100),
                "medium": (100, 200),
                "high": (200, 250),
            }
        else:
            self.corner_thresholds = corner_speed_thresholds

    def extract_corner_speeds(self, telemetry: pd.DataFrame) -> dict[str, float]:
        """Average speed in slow/medium/high-speed corners."""
        # Corner threshold: 250 km/h separates corners from straights
        # Based on typical F1 corner exit speeds (Monza Parabolica ~230, Spa Pouhon ~270)
        corners = telemetry[telemetry["Speed"] < 250]

        speeds = {}
        for corner_type, (min_speed, max_speed) in self.corner_thresholds.items():
            mask = (corners["Speed"] >= min_speed) & (corners["Speed"] < max_speed)
            corner_data = corners[mask]

            if len(corner_data) > 0:
                speeds[f"{corner_type}_corner_speed"] = corner_data["Speed"].mean()
            else:
                speeds[f"{corner_type}_corner_speed"] = np.nan

        return speeds

    def extract_throttle_metrics(self, telemetry: pd.DataFrame) -> dict[str, float]:
        """Throttle usage - percentage at full throttle, average, smoothness."""
        throttle = telemetry["Throttle"]

        return {
            "pct_full_throttle": (throttle == 100).sum() / len(throttle) * 100,
            "avg_throttle": throttle.mean(),
            "throttle_smoothness": throttle.std(),  # lower = smoother
        }

    def extract_braking_metrics(self, telemetry: pd.DataFrame) -> dict[str, float]:
        """Braking zones and intensity."""
        brake = telemetry["Brake"]

        # Count braking zones (transitions from 0 to >0)
        braking_points = ((brake > 0) & (brake.shift(1) == 0)).sum()

        return {
            "braking_pct": (brake > 0).sum() / len(brake) * 100,
            "braking_zones": braking_points,
            "avg_brake_intensity": brake[brake > 0].mean() if (brake > 0).any() else 0,
        }

    def extract_straight_line_speed(self, telemetry: pd.DataFrame) -> dict[str, float]:
        """Top speed and speed at full throttle."""
        full_throttle = telemetry[telemetry["Throttle"] == 100]

        # Max gear (usually 8th) indicates straight-line running
        max_gear = telemetry["nGear"].max()
        top_gear = telemetry[telemetry["nGear"] == max_gear]

        return {
            "avg_speed_full_throttle": (
                full_throttle["Speed"].mean() if len(full_throttle) > 0 else np.nan
            ),
            "max_speed": telemetry["Speed"].max(),
            "pct_at_max_gear": len(top_gear) / len(telemetry) * 100,
        }

    def extract_drs_usage(self, telemetry: pd.DataFrame) -> dict[str, float]:
        """How much DRS was available and used."""
        drs = telemetry["DRS"]
        return {"drs_active_pct": (drs > 0).sum() / len(drs) * 100}

    def extract_features(self, lap: Any) -> dict[str, float]:
        """
        Extract all features from a lap.
        Returns dict of feature_name -> value.
        """
        try:
            telemetry = lap.get_telemetry()

            if telemetry is None or len(telemetry) == 0:
                return {}

            # Combine all feature extractors
            features = {}
            features.update(self.extract_corner_speeds(telemetry))
            features.update(self.extract_throttle_metrics(telemetry))
            features.update(self.extract_braking_metrics(telemetry))
            features.update(self.extract_straight_line_speed(telemetry))
            features.update(self.extract_drs_usage(telemetry))

            return features

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.warning(
                f"Telemetry feature extraction failed: {e}. This lap's telemetry features will be unavailable."
            )
            # Sometimes telemetry fails to load
            return {}


class SessionFeatureAggregator:
    """Aggregate lap-level features into session-level features for a driver."""

    def __init__(self, lap_extractor: LapFeatureExtractor) -> None:
        self.lap_extractor = lap_extractor

    def filter_clean_laps(self, laps: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers and invalid laps.
        Keep laps where: in-lap, out-lap, yellow flags, accidents filtered out.
        """
        # Basic filters
        clean = laps[(laps["IsAccurate"]) & (laps["TrackStatus"] == "1")].copy()  # Green flag

        # Remove statistical outliers (more than 3 std from median)
        if len(clean) > 5:
            lap_times = clean["LapTime"].dt.total_seconds()
            median = lap_times.median()
            std = lap_times.std()

            clean = clean[abs(lap_times - median) < 3 * std]

        return clean

    def extract_driver_session(self, laps: pd.DataFrame) -> dict[str, Any]:
        """
        Extract features for one driver's session.
        Returns aggregated features (median across clean laps).
        """
        if len(laps) == 0:
            return {}

        # Basic info
        driver_info = {
            "driver_number": str(laps.iloc[0]["DriverNumber"]),
            "driver_code": laps.iloc[0]["Driver"],
            "team": laps.iloc[0]["Team"],
            "total_laps": len(laps),
        }

        # Filter to clean laps only
        clean_laps = self.filter_clean_laps(laps)
        driver_info["clean_laps"] = len(clean_laps)

        if len(clean_laps) == 0:
            return driver_info

        # Fastest lap (key metric for practice sessions)
        fastest = clean_laps.pick_fastest()
        driver_info["fastest_lap"] = fastest["LapTime"].total_seconds()

        # Extract features from all clean laps
        lap_features = []
        for _idx, lap in clean_laps.iterrows():
            features = self.lap_extractor.extract_features(lap)
            if features:  # Skip if telemetry failed
                lap_features.append(features)

        if len(lap_features) == 0:
            return driver_info

        # Aggregate: median across all laps (ignores outliers)
        df = pd.DataFrame(lap_features)
        aggregated = df.median().to_dict()

        # Also track consistency (std)
        for col in df.columns:
            aggregated[f"{col}_std"] = df[col].std()

        # Merge with driver info
        return {**driver_info, **aggregated}

    def extract_all_drivers(self, session: Any) -> pd.DataFrame:
        """Extract features for all drivers in a session."""
        driver_features = []

        for driver in session.laps["Driver"].unique():
            driver_laps = session.laps.pick_drivers(driver)
            features = self.extract_driver_session(driver_laps)

            if features and "fastest_lap" in features:
                driver_features.append(features)

        return pd.DataFrame(driver_features)
