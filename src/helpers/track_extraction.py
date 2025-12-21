"""

Originally from: Notebook 05 track characteristics extraction
Adapted for 2026 Bayesian prediction system.

==============================================================================

Track characteristics extraction for F1 prediction system.

This module provides functions to extract static track geometry characteristics
from telemetry data for car-track matching in the prediction system.

Key characteristics extracted:
- Corner speed distribution (slow/medium/fast %)
- Corner density and severity (corners/km, speed loss)
- Power characteristics (full throttle %, top speed)
- Tire stress proxy (energy score)
- Track type (street circuit flag)

Example:
    >>> from helpers.track_extraction import extract_track_profile
    >>> profile = extract_track_profile(2025, session)
    >>> print(profile['corner_density'])

Author: Tomasz Solis
Date: December 2024
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional

# Import from other helpers
from .circuit_utils import extract_track_metrics


def identify_corners(telemetry: pd.DataFrame, min_speed_drop: int = 15) -> pd.DataFrame:
    """
    Identify corners by finding local speed minima.
    
    Returns corner entry/apex/exit speeds for each detected corner.
    
    Args:
        telemetry: Telemetry dataframe with Speed column
        min_speed_drop: Minimum speed drop in km/h to count as a corner
    
    Returns:
        DataFrame with columns: entry_speed, apex_speed, exit_speed, speed_lost
    """
    speed = telemetry['Speed'].values
    
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
        
        corners.append({
            'entry_speed': speed[entry_idx],
            'apex_speed': speed[idx],
            'exit_speed': speed[exit_idx],
            'speed_lost': speed[entry_idx] - speed[idx]
        })
    
    return pd.DataFrame(corners)


def extract_corner_characteristics(session) -> Optional[Dict[str, float]]:
    """
    Extract corner entry/exit characteristics from session.
    
    Calculates:
    - Average speed loss per corner
    - Corner severity distribution (heavy/medium/light braking)
    - Corner density (corners per km)
    - Minimum corner speed (tightest corner)
    
    Args:
        session: Loaded FastF1 session with telemetry
    
    Returns:
        Dictionary with corner metrics, or None if extraction fails
    """
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()
        
        corners = identify_corners(tel, min_speed_drop=15)
        
        if len(corners) == 0:
            return None
        
        # Corner severity distribution
        heavy = (corners['speed_lost'] > 60).sum()
        medium = ((corners['speed_lost'] > 30) & (corners['speed_lost'] <= 60)).sum()
        light = (corners['speed_lost'] <= 30).sum()
        total = len(corners)
        
        return {
            'avg_speed_loss_kmh': float(corners['speed_lost'].mean()),
            'max_speed_loss_kmh': float(corners['speed_lost'].max()),
            'min_corner_speed_kmh': float(corners['apex_speed'].min()),
            'heavy_braking_pct': heavy / total,
            'medium_braking_pct': medium / total,
            'light_braking_pct': light / total,
            'total_corners': total,
            'corner_density': total / (tel['Distance'].max() / 1000.0)
        }
    except Exception as e:
        print(f"  Warning: Could not extract corner characteristics: {e}")
        return None


def extract_full_throttle_pct(session) -> Optional[float]:
    """
    Extract percentage of lap at full throttle (≥98%).
    
    Different from straights - can be full throttle through fast corners.
    Monaco: ~40-50%, Monza: ~75-85%
    
    Args:
        session: Loaded FastF1 session with telemetry
    
    Returns:
        Full throttle percentage (0-1), or None if unavailable
    """
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data()
        if 'Throttle' not in tel.columns:
            return None
        return float((tel['Throttle'] >= 98).mean())
    except:
        return None


def extract_tire_stress_proxy(session) -> Optional[float]:
    """
    Extract tire stress proxy based on speed and variance.
    
    Energy score = (avg_speed / 100) * (1 + speed_variance / 100)
    Higher speed + higher variance = more tire stress
    
    High-energy: Silverstone, Suzuka (lots of lateral forces)
    Low-energy: Monaco, Hungary (slower, less variance)
    
    Args:
        session: Loaded FastF1 session with telemetry
    
    Returns:
        Energy score, or None if unavailable
    """
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data()
        avg_speed = tel['Speed'].mean()
        speed_variance = tel['Speed'].std()
        energy_score = (avg_speed / 100) * (1 + speed_variance / 100)
        return float(energy_score)
    except:
        return None


def extract_track_profile(season: int, session) -> Optional[Dict]:
    """
    Extract complete track profile from session.
    
    Includes:
    - Corner speed distribution (slow/med/fast %)
    - Corner severity (entry/exit delta)
    - Power characteristics (throttle, straights)
    - Tire stress proxy
    
    Args:
        season: Year of the season (e.g. 2025)
        session: Loaded FastF1 session with telemetry
    
    Returns:
        Dictionary with all track characteristics, or None if extraction fails
    """
    metrics = extract_track_metrics(session)
    if metrics is None:
        return None
    
    # Get corner characteristics
    corner_chars = extract_corner_characteristics(session)
    
    profile = {
        'track_name': session.event['EventName'],
        
        # Corner speed distribution
        'slow_corner_pct': metrics['low_pct'],
        'medium_corner_pct': metrics['med_pct'],
        'fast_corner_pct': metrics['high_pct'],
        
        # Power characteristics
        'full_throttle_pct': extract_full_throttle_pct(session),
        'top_speed_kmh': metrics['top_speed'],
        
        # Tire stress proxy
        'energy_score': extract_tire_stress_proxy(session),
        
        # Complexity
        'braking_zones': metrics['braking_events'],
        
        'extracted_from': season
    }
    
    # Add corner characteristics if available
    if corner_chars:
        profile.update({
            'avg_speed_loss_kmh': corner_chars['avg_speed_loss_kmh'],
            'max_speed_loss_kmh': corner_chars['max_speed_loss_kmh'],
            'min_corner_speed_kmh': corner_chars['min_corner_speed_kmh'],
            'heavy_braking_pct': corner_chars['heavy_braking_pct'],
            'medium_braking_pct': corner_chars['medium_braking_pct'],
            'light_braking_pct': corner_chars['light_braking_pct'],
            'total_corners': corner_chars['total_corners'],
            'corner_density': corner_chars['corner_density']
        })
    
    return profile


def identify_street_circuits(track_name: str) -> int:
    """
    Identify if track is a street circuit.
    
    Street circuits have unique characteristics:
    - Tight, unforgiving (no run-off)
    - Low grip early in weekend (dusty surface)
    - Often bumpy
    - Historically more incidents
    
    Args:
        track_name: Name of the track/event
    
    Returns:
        1 if street circuit, 0 otherwise
    """
    STREET_CIRCUITS = {
        'Monaco', 'Singapore', 'Azerbaijan', 'Las Vegas',
        'Miami', 'Saudi Arabian'  # Jeddah
    }
    
    return 1 if any(sc in track_name for sc in STREET_CIRCUITS) else 0


def calculate_track_z_scores(
    df_tracks: pd.DataFrame,
    features: list
) -> tuple[pd.DataFrame, Dict]:
    """
    Calculate z-scores for track characteristics.
    
    Standardizes features to mean=0, std=1 for car-track matching.
    
    Args:
        df_tracks: DataFrame with track characteristics
        features: List of feature names to standardize
    
    Returns:
        Tuple of (dataframe with z-scores added, scaler parameters dict)
    """
    # Remove tracks with missing data
    df_complete = df_tracks.dropna(subset=features)
    
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(df_complete[features])
    
    # Add z-scores to dataframe
    for i, feature in enumerate(features):
        df_complete[f'{feature}_z'] = z_scores[:, i]
    
    # Save scaler parameters for later use with car characteristics
    scaler_params = {
        'features': features,
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
    
    return df_complete, scaler_params


def describe_track_profile(row: pd.Series) -> str:
    """
    Generate human-readable description of track characteristics.
    
    Args:
        row: DataFrame row with track characteristics and z-scores
    
    Returns:
        Comma-separated string of track characteristics
    """
    tags = []
    
    # Corner speed characteristics
    if row['slow_corner_pct_z'] > 1.5:
        tags.append('EXTREME slow corners')
    elif row['slow_corner_pct_z'] > 1.0:
        tags.append('Heavy slow corners')
    elif row['slow_corner_pct_z'] > 0.5:
        tags.append('Above-average slow corners')
    
    if row['medium_corner_pct_z'] > 1.5:
        tags.append('EXTREME technical sections')
    elif row['medium_corner_pct_z'] > 1.0:
        tags.append('Heavy technical sections')
    elif row['medium_corner_pct_z'] > 0.5:
        tags.append('Above-average technical')
    
    if row['fast_corner_pct_z'] > 1.5:
        tags.append('EXTREME high-speed')
    elif row['fast_corner_pct_z'] > 1.0:
        tags.append('Heavy high-speed')
    elif row['fast_corner_pct_z'] > 0.5:
        tags.append('Above-average high-speed')
    
    # Corner density and tightness
    if row['corner_density_z'] > 1.5:
        tags.append('EXTREME corner density')
    elif row['corner_density_z'] > 1.0:
        tags.append('High corner density')
    
    if row['min_corner_speed_kmh_z'] < -1.5:
        tags.append('EXTREME tight corners')
    elif row['min_corner_speed_kmh_z'] < -1.0:
        tags.append('Very tight corners')
    
    # Corner severity (braking)
    if row['avg_speed_loss_kmh_z'] > 1.5:
        tags.append('EXTREME braking demands')
    elif row['avg_speed_loss_kmh_z'] > 1.0:
        tags.append('Heavy braking')
    elif row['avg_speed_loss_kmh_z'] < -1.0:
        tags.append('Flow-focused')
    
    if row['heavy_braking_pct_z'] > 1.0:
        tags.append('Traction-limited')
    
    # Power characteristics
    if row['full_throttle_pct_z'] > 1.0:
        tags.append('High throttle demand')
    
    # Tire stress
    if row['energy_score_z'] > 1.5:
        tags.append('EXTREME tire stress')
    elif row['energy_score_z'] > 1.0:
        tags.append('High tire stress')
    
    # Complexity
    if row['braking_zones_z'] > 1.0:
        tags.append('High complexity')
    
    # Street circuit
    if row['is_street_circuit'] == 1:
        tags.append('Street circuit')
    
    return ', '.join(tags) if tags else 'Balanced'
