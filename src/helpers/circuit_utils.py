"""

Originally from: https://github.com/tomasz-solis/formula1
Adapted for 2026 Bayesian prediction system.

==============================================================================

Circuit utilities for geometry and track-level analytics.

This module provides comprehensive circuit analysis capabilities including:
- Circuit metadata extraction (location, elevation, coordinates)
- Track performance metrics (speed profiles, braking patterns)
- Corner classification by entry speed (slow/medium/fast)
- Circuit clustering analysis using PCA and KMeans
- Visualization tools (radar charts, scatter plots)

The module handles both FastF1 and OpenF1 data sources with automatic
fallback mechanisms for robustness.

Example:
    >>> from helpers.circuit_utils import get_circuits, extract_track_metrics
    >>> circuits = get_circuits(2024)
    >>> print(circuits[['circuitName', 'altitude']].head())

Author: Tomasz Solis
Date: November 2025
"""

# Library imports
import logging
import os
import numpy as np
import pandas as pd
import fastf1 as ff1
import plotly.graph_objects as go
import plotly.express as px

from fastf1.ergast import Ergast
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


# =============================================================================
# CIRCUIT LOOKUP HELPERS
# =============================================================================

def get_elevation(latitude: float,
                  longitude: float,
                  timeout: int = 10) -> float:
    """
    Fetch ground elevation at GPS coordinates via Open-Meteo API.
    
    This is a wrapper around general_utils.get_elevation that maintains
    the same interface for backward compatibility within this module.
    
    Args:
        latitude: GPS latitude in decimal degrees (-90 to 90)
        longitude: GPS longitude in decimal degrees (-180 to 180)
        timeout: API request timeout in seconds (default: 10)
    
    Returns:
        Elevation above sea level in meters
        
    Raises:
        RuntimeError: If API returns no elevation data
        requests.HTTPError: If API request fails
        
    Example:
        >>> # Bahrain International Circuit
        >>> elev = get_elevation(26.0325, 50.5106)
        >>> print(f"Elevation: {elev}m")
        Elevation: 7.0m
    """
    from .general_utils import get_elevation as _get_elev
    return _get_elev(latitude, longitude, timeout)


def get_circuits(season: int) -> pd.DataFrame:
    """
    Retrieve circuit metadata for all tracks used in a season.
    
    Queries Ergast API for circuit info (name, location, coordinates),
    then enriches each circuit with elevation data from Open-Meteo API.
    
    Args:
        season: F1 championship year (e.g., 2024)
    
    Returns:
        DataFrame with columns:
        - circuitName: Official circuit name (str)
        - location: City/locality (str)
        - country: Country name (str)
        - lat: Latitude in decimal degrees (float)
        - lon: Longitude in decimal degrees (float)
        - altitude: Elevation above sea level in meters (float)
        
    Example:
        >>> circuits_2024 = get_circuits(2024)
        >>> print(circuits_2024[['circuitName', 'altitude']].head())
    """
    ergast = Ergast()
    racetracks = ergast.get_circuits(season)
    results = []

    for name in racetracks.circuitName:
        try:
            row = racetracks[racetracks.circuitName == name].iloc[0]
            # Extract geolocation info
            lat = row['lat']; lon = row['long']
            locality = row['locality']; country = row['country']
            # Lookup altitude via external API
            altitude = get_elevation(lat, lon)
            results.append({
                'circuitName': name,
                'location': locality,
                'country': country,
                'lat': lat,
                'lon': lon,
                'altitude': altitude
            })
        except Exception as e:
            log.warning(f"Failed to get altitude for {name}: {e}")
            continue  # skip problematic circuits

    return pd.DataFrame(results)


def get_all_circuits(start_year: int = 2020,
                     end_year: int = 2025) -> pd.DataFrame:
    """
    Aggregate unique circuits across a range of seasons.
    
    Collects circuit metadata for multiple years and deduplicates by
    circuit name, keeping the first occurrence.
    
    Args:
        start_year: First season year, inclusive (default: 2020)
        end_year: Last season year, inclusive (default: 2025)
    
    Returns:
        DataFrame of unique circuits with metadata columns:
        circuitName, location, country, lat, lon, altitude
        
    Example:
        >>> all_circuits = get_all_circuits(2020, 2024)
        >>> print(f"Total unique circuits: {len(all_circuits)}")
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        df = get_circuits(year)
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    # Deduplicate by circuitName, keep first occurrence
    return full.drop_duplicates(subset=['circuitName'], keep='first').reset_index(drop=True)


# =============================================================================
# TRACK FEATURE EXTRACTION
# =============================================================================

def extract_track_metrics(session) -> Optional[Dict[str, float]]:
    """
    Extract speed and braking characteristics from session's fastest lap.
    
    Analyzes telemetry to compute track-level metrics including speed
    profile (low/medium/high speed percentages) and braking event count.
    
    Speed classification thresholds:
        - Low speed: < 120 km/h
        - Medium speed: 120-200 km/h
        - High speed: ≥ 200 km/h
    
    Args:
        session: Loaded FastF1 session with telemetry data
    
    Returns:
        Dictionary with track metrics, or None if extraction fails:
        - avg_speed: Mean speed across lap (km/h)
        - top_speed: Maximum speed reached (km/h)
        - braking_events: Count of heavy braking (>30 km/h drop)
        - low_pct: Fraction of lap below 120 km/h (0-1)
        - med_pct: Fraction of lap between 120-200 km/h (0-1)
        - high_pct: Fraction of lap above 200 km/h (0-1)
        
    Example:
        >>> metrics = extract_track_metrics(session)
        >>> print(f"Top speed: {metrics['top_speed']:.1f} km/h")
    """
    try:
        if session.laps.empty:
            return None
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()
        # Compute speed deltas to identify heavy braking
        tel['delta_speed'] = tel['Speed'].diff()
        heavy_brakes = tel['delta_speed'] < -30
        return {
            'avg_speed': tel['Speed'].mean(),
            'top_speed': tel['Speed'].max(),
            'braking_events': int(heavy_brakes.sum()),
            'low_pct': float((tel['Speed'] < 120).mean()),
            'med_pct': float(((tel['Speed'] >= 120) & (tel['Speed'] < 200)).mean()),
            'high_pct': float((tel['Speed'] >= 200).mean())
        }
    except Exception as e:
        log.warning(f"⚠️ Failed to extract track metrics: {e}")
        return None


def get_valid_lap_with_pos(session, max_attempts: int = 5):
    """
    Find a quick lap with valid positional data (X/Y coordinates).
    
    FastF1 position data is often incomplete or missing for certain drivers.
    This iterates through fastest laps to find one with valid coordinates.
    
    Args:
        session: FastF1 session object
        max_attempts: Maximum number of quick laps to try (default: 5)
    
    Returns:
        A Lap object with valid position data, or None if not found
        
    Example:
        >>> lap = get_valid_lap_with_pos(session)
        >>> if lap:
        ...     pos = lap.get_pos_data()
        ...     print(f"Found valid lap: Driver {lap.Driver}")
    """
    fast_laps = session.laps.pick_quicklaps().sort_values('LapTime')
    for i, lap in enumerate(fast_laps.itertuples()):
        if i >= max_attempts:
            break
        drv_num = lap.DriverNumber
        try:
            _ = session.pos_data[drv_num]
            return session.laps.loc[lap.Index]
        except KeyError:
            continue  # no pos data for this driver
        except Exception as e:
            log.warning(f"Skipping lap for driver {drv_num}: {e}")
    log.warning("⚠️ No valid lap with position data found.")
    return None


def get_circuit_corner_profile(
    session,
    low_thresh: int = 100,
    med_thresh: int = 170
) -> Dict[str, int]:
    """
    Analyze circuit layout by classifying corners by entry speed.
    
    Uses spatial matching between circuit corner coordinates and telemetry
    to determine entry speed for each corner, then classifies into
    slow/medium/fast categories. Also detects chicanes by corner proximity.
    
    Algorithm:
        1. Load fastest lap with valid position data (X,Y coordinates)
        2. Merge positional data with speed telemetry via timestamp
        3. Build KD-tree from telemetry (X,Y) points
        4. Query tree to map each circuit corner to nearest telemetry point
        5. Classify corners by speed at matched point
        6. Detect chicanes: consecutive corners within 100m distance
    
    Args:
        session: Loaded FastF1 session with telemetry and circuit info
        low_thresh: Maximum speed (km/h) for slow corners (default: 100)
        med_thresh: Maximum speed (km/h) for medium corners (default: 170)
    
    Returns:
        Dictionary with corner counts:
        - slow_corners: Count with entry speed < low_thresh
        - medium_corners: Count with low_thresh ≤ speed < med_thresh
        - fast_corners: Count with speed ≥ med_thresh
        - chicanes: Count of corner pairs within 100m
        
    Raises:
        ValueError: If extraction fails or data incomplete
        
    Example:
        >>> profile = get_circuit_corner_profile(session)
        >>> print(f"Slow corners: {profile['slow_corners']}")
        
    Note:
        - Classification based on entry speed, not apex speed
        - Chicane detection uses fixed 100m threshold
    """
    try:
        session.load(telemetry=True)
        lap = get_valid_lap_with_pos(session)
        if lap is None:
            event = session.event.get("EventName", "Unknown")
            name = getattr(session, "name", "Unknown")
            log.warning(f"⚠️ Skipping {event} {name}: No valid lap with pos data")
            raise RuntimeError("⚠️ No valid lap with position data")
        pos = lap.get_pos_data().copy()
        car = lap.get_car_data().add_distance().copy()
        corners = session.get_circuit_info().corners.copy()

        if pos.empty or car.empty or corners.empty:
            raise ValueError("⚠️ Missing required telemetry or circuit data.")

        # Convert Time to seconds
        pos['Time_s'] = pos['Time'].dt.total_seconds()
        car['Time_s'] = car['Time'].dt.total_seconds()

        # Nearest merge using merge_asof
        merged = pd.merge_asof(
            pos.sort_values("Time_s"),
            car[["Time_s", "Speed", "Distance"]].sort_values("Time_s"),
            on="Time_s",
            direction="nearest"
        )

        # KD-tree to map corners to nearest telemetry point
        tree = cKDTree(merged[['X', 'Y']].dropna().values)
        corner_coords = corners[['X', 'Y']].dropna().values
        distances, indices = tree.query(corner_coords, k=1)

        matched = merged.iloc[indices].reset_index(drop=True)
        matched = matched.rename(columns={"Distance": "DriverDistance"})  # avoid conflict
        corners = corners.reset_index(drop=True)
        corners = pd.concat([corners, matched[['Speed', 'DriverDistance']]], axis=1)

        # Classify by speed
        corners['corner_type'] = pd.cut(
            corners['Speed'],
            bins=[0, low_thresh, med_thresh, 1000],
            labels=['slow', 'medium', 'fast'],
            include_lowest=True
        )
        counts = corners['corner_type'].value_counts().to_dict()

        # Chicane detection (distance between consecutive corners)
        corners = corners.sort_values(by='DriverDistance')
        corners['DistanceFromPrev'] = corners['DriverDistance'].diff().fillna(9999)
        chicanes = (corners['DistanceFromPrev'] < 100).sum()

        return {
            'slow_corners': counts.get('slow', 0),
            'medium_corners': counts.get('medium', 0),
            'fast_corners': counts.get('fast', 0),
            'chicanes': chicanes
        }
    
        
    except Exception as e:
        event = session.event.get("EventName", "Unknown")
        name = getattr(session, "name", "Unknown")
        raise ValueError(f"⚠️ Failed to compute corner profile: {event} {name} – {e}")


# =============================================================================
# HIGHER-LEVEL PROFILING PIPELINES
# =============================================================================

def build_profiles_for_season(
    year: int,
    circuit_metadata: pd.DataFrame,
    *,
    only_specific: Optional[set[Tuple[str,str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build circuit-performance profiles for one F1 season.
    
    Iterates through completed events and sessions, extracting track metrics
    (speed, corners, chicanes), telemetry features, and weather conditions.
    
    Pipeline per session:
        1. Load session via load_session() (FastF1 or OpenF1 fallback)
        2. Estimate lap length from telemetry
        3. Extract track metrics (extract_track_metrics)
        4. Extract corner profile (get_circuit_corner_profile)
        5. Extract weather data (get_weather_info)
        6. Merge with circuit altitude from metadata
        7. Combine all into single row
    
    Args:
        year: F1 championship season year (e.g., 2024)
        circuit_metadata: DataFrame from get_all_circuits() with altitude data
        only_specific: Optional filter - set of (event_name, session_code)
                      tuples to process. Example: {('Monaco GP', 'Q')}
    
    Returns:
        Tuple of (profiles_df, skipped_df):
        - profiles_df: Successfully processed sessions with columns:
          year, event, location, session, real_altitude, lap_length,
          telemetry_source, avg_speed, top_speed, braking_events,
          slow_corners, medium_corners, fast_corners, chicanes,
          air_temp_avg, track_temp_avg, rain_detected
        - skipped_df: Failed sessions with columns:
          year, event, session, reason
          
    Example:
        >>> metadata = get_all_circuits(2024)
        >>> profiles, skipped = build_profiles_for_season(2024, metadata)
        >>> print(f"Processed: {len(profiles)}, Failed: {len(skipped)}")
    """
    records: list[dict] = []
    skipped: list[dict] = []

    try:
        from .general_utils import _official_schedule, _session_list, _session_date_col, load_session, get_weather_info
        sched = _official_schedule(year)
        past  = sched[sched.Session1DateUtc < datetime.now(timezone.utc)]
    except Exception as e:
        skipped.append(
            {"year": year, "event": None, "session": None, "reason": str(e)}
        )
        return pd.DataFrame(), pd.DataFrame(skipped)
        
    # Iterate events and sessions
    for _, ev in tqdm(past.iterrows(), total=len(past), desc=f"{year} events", leave=True,colour="blue"):
        ev_name = ev["EventName"]
        raw_fmt = ev["EventFormat"]
        location = ev["Location"] 
        fmt = str(raw_fmt.item() if isinstance(raw_fmt, pd.Series) else raw_fmt)
        fmt = fmt.lower() if pd.notnull(fmt) else "conventional"
        date_map = _session_date_col(fmt, ev)
        sessions = _session_list(fmt)

        for sess in tqdm(sessions, desc=f"{year} {ev_name}", leave=True, colour="black"):
            if only_specific and (ev_name, sess) not in only_specific:
                continue
            try:
                s_info = load_session(year, ev_name, sess)
                if s_info["status"] == "error":
                    raise ValueError(s_info["reason"])

                tele_src = s_info["source"] # fastf1 / openf1
                laps = s_info["laps"]
                session = s_info["session"]  # FastF1.Session or None

                if laps is None or laps.empty:
                    raise ValueError("Lap data missing")

                # Estimate lap length
                if session is not None: # FastF1
                    try:
                        dist = session.laps.pick_fastest().get_car_data().add_distance()['Distance'].max()
                        lap_len = dist
                    except Exception:
                        lap_len = np.nan
                else: # OpenF1 fallback
                    lap_len = (
                        laps.groupby("driver_number")["lap_distance"].max().max()
                    )

                try:
                    tmet = extract_track_metrics(session) if session else None
                except Exception as e:
                    raise ValueError(f"⚠️ Failed to extract telemetry metrics: {e}")
                
                try:
                    cmet = get_circuit_corner_profile(session) if session else None
                except Exception as e:
                    raise ValueError(f"⚠️ {e}")  # already descriptive
                
                try:
                    wmet = get_weather_info(session, year, ev_name, sess)
                except Exception as e:
                    raise ValueError(f"⚠️ Failed to fetch weather data: {e}")

                if not tmet:
                    raise ValueError("⚠️ Missing telemetry metrics")

                try:
                    alt = (
                        circuit_metadata
                        .loc[circuit_metadata["location"] == location, "altitude"]
                        .iloc[0]
                    )
                except IndexError:
                    alt = np.nan

                records.append(
                    {
                        "year": year,
                        "event": ev_name,
                        "location": location,
                        "session": sess,
                        "real_altitude": alt,
                        "lap_length": lap_len,
                        "telemetry_source": tele_src,
                        **tmet, **cmet, **wmet,
                    }
                )

            except Exception as e:
                skipped.append(
                    {
                        "year": year,
                        "event": ev_name,
                        "session": sess,
                        "reason": f"{type(e).__name__}: {e}",
                    }
                )

    return pd.DataFrame(records), pd.DataFrame(skipped)
    

def _build_circuit_profile_df(
    start_year: int,
    end_year: int,
    *,
    only_specific: Optional[Dict[int, set[Tuple[str,str]]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build circuit profiles over multiple seasons.
    
    Orchestrates profile generation across years by calling
    build_profiles_for_season() for each year, then concatenating results.
    
    Args:
        start_year: First year, inclusive (e.g., 2022)
        end_year: Last year, inclusive (e.g., 2024)
        only_specific: Optional mapping of year to sessions to include.
                      Format: {year: {(event_name, session_code), ...}}
    
    Returns:
        Tuple of (all_profiles_df, all_skipped_df)
        
    Example:
        >>> profiles, skipped = _build_circuit_profile_df(2022, 2024)
        >>> print(f"Total sessions: {len(profiles)}")
    """
    all_profiles, all_skipped = [], []

    for year in range(start_year, end_year + 1):
        tqdm.write(f"\n📅 Building profiles for season {year}...")
        circuit_metadata = get_all_circuits(year)

        if only_specific and year in only_specific:
            df, skipped = build_profiles_for_season(
                year,
                circuit_metadata,
                only_specific=only_specific[year]
            )
        else:
            df, skipped = build_profiles_for_season(year, circuit_metadata)

        all_profiles.append(df)
        all_skipped.append(skipped)

    # 3) concatenate
    profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    skipped = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()

    # 4) log any skips
    if not skipped.empty:
        tqdm.write("\n⚠️ Skipped sessions:")
        for _, row in skipped.iterrows():
            tqdm.write(f"⚠️  - {row['year']} {row['event']} {row['session']} – {row['reason']}")

    return profiles, skipped
    

def fit_track_clusters(
    df_profiles: pd.DataFrame,
    group_cols: List[str] = ['event','year'],
    feat_cols: Optional[List[str]] = None,
    scaler=None,
    clusterer=None,
    do_pca: bool = False,
    n_components: int = 2
) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Cluster tracks based on performance metrics with optional PCA projection.
    
    Aggregates session-level features per track, scales/imputes data, applies
    optional PCA, then performs clustering.
    
    Args:
        df_profiles: Session-level feature DataFrame
        group_cols: Columns defining each track group (default: ['event', 'year'])
        feat_cols: Numeric features for clustering (if None, uses all numeric)
        scaler: Preprocessing scaler (default: StandardScaler)
        clusterer: Clustering estimator (default: KMeans(n_clusters=5))
        do_pca: Whether to include PCA step (default: False)
        n_components: Number of PCA components if do_pca=True (default: 2)
    
    Returns:
        Tuple of (track_profile_df, fitted_pipeline):
        - track_profile: DataFrame with cluster labels (and PC coords if PCA)
        - pipeline: Fitted sklearn Pipeline
        
    Example:
        >>> track_profile, pipeline = fit_track_clusters(profiles, do_pca=True)
        >>> print(track_profile[['event', 'cluster', 'PC1', 'PC2']].head())
    """
    # Determine features
    feat_cols = feat_cols or df_profiles.select_dtypes(include='number').columns.tolist()
    # Aggregate per track
    track_features = (
        df_profiles
        .groupby(group_cols)[feat_cols]
        .mean()
        .reset_index()
    )
    X = track_features[feat_cols]

    # Build pipeline
    steps = [
        ('imputer', SimpleImputer()),
        ('scaler', scaler or StandardScaler())
    ]
    if do_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=42)))
    steps.append(('cluster', clusterer or KMeans(n_clusters=5, random_state=42)))
    pipe = Pipeline(steps)

    # Fit clusters
    labels = pipe.fit_predict(X)
    track_profile = track_features.copy()
    track_profile['cluster'] = labels.astype(str)

    # PCA coords if requested
    if do_pca:
        X_imp = pipe.named_steps['imputer'].transform(X)
        X_scl = pipe.named_steps['scaler'].transform(X_imp)
        pcs = pipe.named_steps['pca'].transform(X_scl)
        track_profile[['PC1', 'PC2']] = pcs

    return track_profile, pipe

    
def plot_cluster_radar(
    df_profiles: pd.DataFrame,
    categories: List[str],
    cluster_col: str = 'cluster',
    normalize: bool = True
) -> go.Figure:
    """
    Create a radar chart comparing clusters on selected features.
    
    Aggregates features by cluster (mean), optionally normalizes to [0,1],
    then plots each cluster as separate trace on radar chart.
    
    Args:
        df_profiles: DataFrame with cluster labels and feature columns
        categories: List of feature column names to plot
        cluster_col: Column name for clusters (default: 'cluster')
        normalize: If True, scale features to [0,1] (default: True)
    
    Returns:
        Plotly Figure object with radar chart
        
    Example:
        >>> features = ['avg_speed', 'slow_corners', 'chicanes']
        >>> fig = plot_cluster_radar(track_profile, features)
        >>> fig.show()
    """
    # Aggregate mean feature per cluster
    agg = df_profiles.groupby(cluster_col)[categories].mean()
    if normalize:
        agg = agg.apply(lambda col: (col - col.min())/(col.max()-col.min()), axis=0)
    agg = agg.reset_index()
    fig = go.Figure()
    for _, row in agg.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].tolist(), theta=categories,
            fill='toself', name=f"Cluster {row[cluster_col]}"
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1] if normalize else None)),
        title="Driving Style Radar per Cluster"
    )
    return fig


def create_pca_for_n_clusters(
    circuits: pd.DataFrame,
    clusters: int,
    feat_cols: List[str]
) -> Tuple[px.scatter, go.Figure]:
    """
    Perform PCA and KMeans clustering, return scatter and radar plots.
    
    Convenience function combining clustering with PCA visualization
    and radar chart generation.
    
    Args:
        circuits: DataFrame with track metrics and identifiers
        clusters: Number of clusters for KMeans
        feat_cols: Feature column names for clustering
    
    Returns:
        Tuple of (scatter_plot, radar_plot)
        
    Example:
        >>> scatter, radar = create_pca_for_n_clusters(profiles, 5, features)
        >>> scatter.show()
        >>> radar.show()
    """
    # Fit on per-track metrics
    track_profile, pipeline = fit_track_clusters(
        circuits,
        group_cols=['track_id'],
        feat_cols=feat_cols,
        do_pca=True,
        clusterer=KMeans(n_clusters=clusters, random_state=42),
    )

    # Map cluster back to original circuits
    key = track_profile.set_index('track_id')['cluster']
    circuits['cluster'] = (
        circuits['event'].astype(str) + '_' + circuits['year'].astype(str)
    ).map(key)

    # Prepare ordered cluster labels
    cluster_vals = sorted(track_profile['cluster'].unique(), key=int)

    # Scatter via PCA dims
    scatter_plot = px.scatter(
        track_profile,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['track_id'],
        title=f'PCA view for k={clusters}',
        category_orders={'cluster': cluster_vals},
    )

    # Identify top varying features for radar
    stats = track_profile.groupby('cluster')[feat_cols].mean()
    spreads = (stats.max() - stats.min()).sort_values(ascending=False)
    top_features = spreads.head(8).index.tolist()
    
    # Radar chart
    radar_plot = plot_cluster_radar(
        track_profile,
        categories=top_features,
        cluster_col='cluster',
        normalize=True,
    )

    return scatter_plot, radar_plot