"""Extract track-shape features from fastest-lap telemetry."""

import logging

import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

# Legacy clustering work lives in archived `circuit.py`; metric extraction stays here.

logger = logging.getLogger(__name__)


def extract_track_metrics(session) -> dict[str, float] | None:
    """Extract speed and braking metrics from fastest lap telemetry. Returns low/medium/high speed percentages and braking event count."""
    try:
        if session.laps.empty:
            return None
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()
        # Compute speed deltas to identify heavy braking
        tel["delta_speed"] = tel["Speed"].diff()
        heavy_brakes = tel["delta_speed"] < -30
        return {
            "avg_speed": tel["Speed"].mean(),
            "top_speed": tel["Speed"].max(),
            "braking_events": int(heavy_brakes.sum()),
            "low_pct": float((tel["Speed"] < 120).mean()),
            "med_pct": float(((tel["Speed"] >= 120) & (tel["Speed"] < 200)).mean()),
            "high_pct": float((tel["Speed"] >= 200).mean()),
        }
    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.warning("Failed to extract track metrics: %s", e)
        return None


def identify_corners(telemetry: pd.DataFrame, min_speed_drop: int = 15) -> pd.DataFrame:
    """Identify corners by finding local speed minima.

    Returns entry/apex/exit speeds for each corner.
    """
    speed = telemetry["Speed"].values

    # Find local minima (corners)
    minima_idx, _ = find_peaks(-speed, prominence=min_speed_drop)

    corners = []
    for idx in minima_idx:
        # Find entry point (where braking starts)
        entry_idx = idx
        for i in range(idx - 1, max(0, idx - 50), -1):
            if speed[i] > speed[i + 1]:
                entry_idx = i
            else:
                break

        # Find exit point (back to speed)
        exit_idx = idx
        for i in range(idx + 1, min(len(speed), idx + 50)):
            if speed[i] > speed[i - 1]:
                exit_idx = i
            else:
                break

        corners.append(
            {
                "entry_speed": speed[entry_idx],
                "apex_speed": speed[idx],
                "exit_speed": speed[exit_idx],
                "speed_lost": speed[entry_idx] - speed[idx],
            }
        )

    return pd.DataFrame(corners)


def extract_corner_characteristics(session) -> dict[str, float] | None:
    """Extract corner severity, density, and speed loss metrics from session telemetry."""
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()

        corners = identify_corners(tel, min_speed_drop=15)

        if len(corners) == 0:
            return None

        # Corner severity distribution
        heavy = (corners["speed_lost"] > 60).sum()
        medium = ((corners["speed_lost"] > 30) & (corners["speed_lost"] <= 60)).sum()
        light = (corners["speed_lost"] <= 30).sum()
        total = len(corners)

        return {
            "avg_speed_loss_kmh": float(corners["speed_lost"].mean()),
            "max_speed_loss_kmh": float(corners["speed_lost"].max()),
            "min_corner_speed_kmh": float(corners["apex_speed"].min()),
            "heavy_braking_pct": heavy / total,
            "medium_braking_pct": medium / total,
            "light_braking_pct": light / total,
            "total_corners": total,
            "corner_density": total / (tel["Distance"].max() / 1000.0),
        }
    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.warning(
            "Could not extract corner characteristics: %s. Corner metrics will be unavailable for car-track matching.",
            e,
        )
        return None


def extract_full_throttle_pct(session) -> float | None:
    """Extract percentage of lap at full throttle (≥98%). Returns float 0-1 or None."""
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data()
        if "Throttle" not in tel.columns:
            return None
        return float((tel["Throttle"] >= 98).mean())
    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.warning(
            "Could not extract full throttle percentage: %s. This metric will be unavailable.", e
        )
        return None


def extract_tire_stress_proxy(session) -> float | None:
    """Extract tire stress energy score from speed and variance. Returns float or None."""
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data()
        avg_speed = tel["Speed"].mean()
        speed_variance = tel["Speed"].std()
        energy_score = (avg_speed / 100) * (1 + speed_variance / 100)
        return float(energy_score)
    except (AttributeError, KeyError, ValueError, TypeError) as e:
        logger.warning(
            "Could not extract tire stress proxy (energy score): %s. This metric will be unavailable.",
            e,
        )
        return None


def extract_track_profile(season: int, session) -> dict | None:
    """Extract complete track profile including corner distribution, severity, power, and tire stress."""
    metrics = extract_track_metrics(session)
    if metrics is None:
        return None

    # Get corner characteristics
    corner_chars = extract_corner_characteristics(session)

    profile = {
        "track_name": session.event["EventName"],
        # Corner speed distribution
        "slow_corner_pct": metrics["low_pct"],
        "medium_corner_pct": metrics["med_pct"],
        "fast_corner_pct": metrics["high_pct"],
        # Power characteristics
        "full_throttle_pct": extract_full_throttle_pct(session),
        "top_speed_kmh": metrics["top_speed"],
        # Tire stress proxy
        "energy_score": extract_tire_stress_proxy(session),
        # Complexity
        "braking_zones": metrics["braking_events"],
        "extracted_from": season,
    }

    # Add corner characteristics if available
    if corner_chars:
        profile.update(
            {
                "avg_speed_loss_kmh": corner_chars["avg_speed_loss_kmh"],
                "max_speed_loss_kmh": corner_chars["max_speed_loss_kmh"],
                "min_corner_speed_kmh": corner_chars["min_corner_speed_kmh"],
                "heavy_braking_pct": corner_chars["heavy_braking_pct"],
                "medium_braking_pct": corner_chars["medium_braking_pct"],
                "light_braking_pct": corner_chars["light_braking_pct"],
                "total_corners": corner_chars["total_corners"],
                "corner_density": corner_chars["corner_density"],
            }
        )

    return profile


def identify_street_circuits(track_name: str) -> int:
    """Identify if track is a street circuit. Returns 1 if yes, 0 otherwise."""
    STREET_CIRCUITS = {
        "Monaco",
        "Singapore",
        "Azerbaijan",
        "Las Vegas",
        "Miami",
        "Saudi Arabian",  # Jeddah
    }

    return 1 if any(sc in track_name for sc in STREET_CIRCUITS) else 0


def calculate_track_z_scores(df_tracks: pd.DataFrame, features: list) -> tuple[pd.DataFrame, dict]:
    """Calculate z-scores for track features. Returns dataframe with z-score columns and scaler parameters."""
    # Remove tracks with missing data
    df_complete = df_tracks.dropna(subset=features)

    scaler = StandardScaler()
    z_scores = scaler.fit_transform(df_complete[features])

    # Add z-scores to dataframe
    for i, feature in enumerate(features):
        df_complete[f"{feature}_z"] = z_scores[:, i]

    # Save scaler parameters for later use with car characteristics
    scaler_params = {
        "features": features,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
    }

    return df_complete, scaler_params


def describe_track_profile(row: pd.Series) -> str:
    """Generate human-readable description of track characteristics from z-scores."""
    tags = []

    # Corner speed characteristics
    if row["slow_corner_pct_z"] > 1.5:
        tags.append("EXTREME slow corners")
    elif row["slow_corner_pct_z"] > 1.0:
        tags.append("Heavy slow corners")
    elif row["slow_corner_pct_z"] > 0.5:
        tags.append("Above-average slow corners")

    if row["medium_corner_pct_z"] > 1.5:
        tags.append("EXTREME technical sections")
    elif row["medium_corner_pct_z"] > 1.0:
        tags.append("Heavy technical sections")
    elif row["medium_corner_pct_z"] > 0.5:
        tags.append("Above-average technical")

    if row["fast_corner_pct_z"] > 1.5:
        tags.append("EXTREME high-speed")
    elif row["fast_corner_pct_z"] > 1.0:
        tags.append("Heavy high-speed")
    elif row["fast_corner_pct_z"] > 0.5:
        tags.append("Above-average high-speed")

    # Corner density and tightness
    if row["corner_density_z"] > 1.5:
        tags.append("EXTREME corner density")
    elif row["corner_density_z"] > 1.0:
        tags.append("High corner density")

    if row["min_corner_speed_kmh_z"] < -1.5:
        tags.append("EXTREME tight corners")
    elif row["min_corner_speed_kmh_z"] < -1.0:
        tags.append("Very tight corners")

    # Corner severity (braking)
    if row["avg_speed_loss_kmh_z"] > 1.5:
        tags.append("EXTREME braking demands")
    elif row["avg_speed_loss_kmh_z"] > 1.0:
        tags.append("Heavy braking")
    elif row["avg_speed_loss_kmh_z"] < -1.0:
        tags.append("Flow-focused")

    if row["heavy_braking_pct_z"] > 1.0:
        tags.append("Traction-limited")

    # Power characteristics
    if row["full_throttle_pct_z"] > 1.0:
        tags.append("High throttle demand")

    # Tire stress
    if row["energy_score_z"] > 1.5:
        tags.append("EXTREME tire stress")
    elif row["energy_score_z"] > 1.0:
        tags.append("High tire stress")

    # Complexity
    if row["braking_zones_z"] > 1.0:
        tags.append("High complexity")

    # Street circuit
    if row["is_street_circuit"] == 1:
        tags.append("Street circuit")

    return ", ".join(tags) if tags else "Balanced"
