"""Helper functions for Bayesian prediction."""

import pandas as pd
import numpy as np
from typing import Dict
from .bayesian import DriverPrior, BayesianDriverRanking
from .scoring import AbsoluteDifferenceScoring

# From cell 16
def remove_outliers_mad(df: pd.DataFrame, feature: str, n_mad: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using Median Absolute Deviation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data
    feature : str
        Column to check for outliers
    n_mad : float
        Number of MADs from median to consider outlier
    
    Returns
    -------
    pd.DataFrame
        Data with outliers removed
    """
    values = df[feature].dropna()
    median = values.median()
    mad = median_abs_deviation(values, nan_policy='omit')
    
    # Define outlier bounds
    lower_bound = median - n_mad * mad
    upper_bound = median + n_mad * mad
    
    # Filter
    mask = (df[feature] >= lower_bound) & (df[feature] <= upper_bound)
    
    removed = len(df) - mask.sum()
    if removed > 0:
        removed_drivers = df[~mask]['driver_number'].tolist()
        print(f"  Removed {removed} outliers in {feature}: {removed_drivers}")
        print(f"    Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    
    return df[mask]

# From cell 17
def predict_race_simple(
    driver_scores: pd.DataFrame,
    track_chars: dict,
    bayesian_priors: dict
) -> pd.DataFrame:
    """
    Simple prediction: weighted sum of scores by track characteristics.
    
    track_chars should have keys like:
      'medium_corner_time_pct', 'slow_corner_time_pct', etc.
    """
    predictions = []
    
    for idx, row in driver_scores.iterrows():
        driver_num = row['driver_number']
        
        # Weighted sum of scores
        testing_signal = (
            track_chars.get('medium_corner_time_pct', 0.4) * row.get('medium_corner_score', 0) +
            track_chars.get('slow_corner_time_pct', 0.2) * row.get('slow_corner_score', 0) +
            track_chars.get('high_corner_time_pct', 0.2) * row.get('high_corner_score', 0) +
            track_chars.get('straight_time_pct', 0.2) * row.get('straight_score', 0)
        )
        
        # Get Bayesian prior
        prior_mu = bayesian_priors[driver_num].mu if driver_num in bayesian_priors else 10.0
        
        # Combine: 90% prior, 10% testing
        final_rating = 0.9 * prior_mu + 0.1 * testing_signal
        
        predictions.append({
            'driver_number': driver_num,
            'rating': final_rating
        })
    
    df = pd.DataFrame(predictions)
    df = df.sort_values('rating', ascending=False).reset_index(drop=True)
    df['predicted_position'] = range(1, len(df) + 1)
    
    return df


def predict_race_fixed(
    driver_scores: pd.DataFrame,
    track_chars: dict,
    bayesian_priors: dict
) -> pd.DataFrame:
    """
    Fixed prediction with proper z-score normalization.
    """
    # Normalize scores to z-scores
    scores_normalized = driver_scores.copy()
    
    score_cols = ['slow_corner_score', 'medium_corner_score', 
                  'high_corner_score', 'straight_score']
    
    for col in score_cols:
        if col in scores_normalized.columns:
            mean = scores_normalized[col].mean()
            std = scores_normalized[col].std()
            if std > 0:
                scores_normalized[f'{col}_z'] = (scores_normalized[col] - mean) / std
            else:
                scores_normalized[f'{col}_z'] = 0
    
    predictions = []
    
    for idx, row in scores_normalized.iterrows():
        driver_num = row['driver_number']
        
        # Track-weighted z-scores
        testing_signal = (
            track_chars.get('medium_corner_time_pct', 0.4) * row.get('medium_corner_score_z', 0) +
            track_chars.get('slow_corner_time_pct', 0.2) * row.get('slow_corner_score_z', 0) +
            track_chars.get('high_corner_time_pct', 0.2) * row.get('high_corner_score_z', 0) +
            track_chars.get('straight_time_pct', 0.2) * row.get('straight_score_z', 0)
        )
        
        # Get Bayesian prior
        prior_mu = bayesian_priors[driver_num].mu if driver_num in bayesian_priors else 10.0
        
        # Combine: 90% prior, 10% testing
        final_rating = 0.9 * (21 - prior_mu) + 0.1 * testing_signal
        
        predictions.append({
            'driver_number': driver_num,
            'rating': final_rating
        })
    
    df = pd.DataFrame(predictions)
    df = df.sort_values('rating').reset_index(drop=True)
    df['predicted_position'] = range(1, len(df) + 1)
    
    return df

# From cell 17
def predict_race_fixed(
    driver_scores: pd.DataFrame,
    track_chars: dict,
    bayesian_priors: dict
) -> pd.DataFrame:
    """
    Fixed prediction with proper z-score normalization.
    """
    # Normalize scores to z-scores
    scores_normalized = driver_scores.copy()
    
    score_cols = ['slow_corner_score', 'medium_corner_score', 
                  'high_corner_score', 'straight_score']
    
    for col in score_cols:
        if col in scores_normalized.columns:
            mean = scores_normalized[col].mean()
            std = scores_normalized[col].std()
            if std > 0:
                scores_normalized[f'{col}_z'] = (scores_normalized[col] - mean) / std
            else:
                scores_normalized[f'{col}_z'] = 0
    
    predictions = []
    
    for idx, row in scores_normalized.iterrows():
        driver_num = row['driver_number']
        
        # Track-weighted z-scores
        testing_signal = (
            track_chars.get('medium_corner_time_pct', 0.4) * row.get('medium_corner_score_z', 0) +
            track_chars.get('slow_corner_time_pct', 0.2) * row.get('slow_corner_score_z', 0) +
            track_chars.get('high_corner_time_pct', 0.2) * row.get('high_corner_score_z', 0) +
            track_chars.get('straight_time_pct', 0.2) * row.get('straight_score_z', 0)
        )
        
        # Get Bayesian prior
        prior_mu = bayesian_priors[driver_num].mu if driver_num in bayesian_priors else 10.0
        
        # Combine: 90% prior, 10% testing
        final_rating = 0.9 * (21 - prior_mu) + 0.1 * testing_signal
        
        predictions.append({
            'driver_number': driver_num,
            'rating': final_rating
        })
    
    df = pd.DataFrame(predictions)
    df = df.sort_values('rating').reset_index(drop=True)
    df['predicted_position'] = range(1, len(df) + 1)
    
    return df

# From cell 22
def initialize_2023_standings_priors() -> Dict[str, DriverPrior]:
    """
    2023 Final Championship Standings (correct priors for 2024).
    
    2023 Results:
    1. VER - 575 pts (dominant)
    2. PER - 285 pts
    3. HAM - 234 pts
    4. ALO - 206 pts
    5. SAI - 200 pts
    6. RUS - 175 pts
    7. LEC - 206 pts
    8. NOR - 205 pts
    9. PIA - 97 pts (rookie)
    ...
    """
    priors_2023 = {
        # Top tier 2023
        '1': DriverPrior('1', 'VER', 'Red Bull Racing', 'top', 'elite', mu=20, sigma=3),  # Dominant
        '11': DriverPrior('11', 'PER', 'Red Bull Racing', 'top', 'experienced', mu=16, sigma=4),
        
        # Upper midfield 2023
        '44': DriverPrior('44', 'HAM', 'Mercedes', 'midfield', 'elite', mu=15, sigma=4),
        '14': DriverPrior('14', 'ALO', 'Aston Martin', 'midfield', 'elite', mu=15, sigma=4),
        '55': DriverPrior('55', 'SAI', 'Ferrari', 'midfield', 'experienced', mu=15, sigma=4),
        '63': DriverPrior('63', 'RUS', 'Mercedes', 'midfield', 'experienced', mu=14, sigma=4),
        '16': DriverPrior('16', 'LEC', 'Ferrari', 'midfield', 'elite', mu=15, sigma=4),
        '4': DriverPrior('4', 'NOR', 'McLaren', 'midfield', 'experienced', mu=15, sigma=4),
        
        # Promising 2023
        '81': DriverPrior('81', 'PIA', 'McLaren', 'midfield', 'rookie', mu=12, sigma=5),  # Strong rookie
        
        # Lower midfield 2023
        '10': DriverPrior('10', 'GAS', 'Alpine', 'backmarker', 'experienced', mu=10, sigma=5),
        '23': DriverPrior('23', 'ALB', 'Williams', 'backmarker', 'experienced', mu=10, sigma=5),
        '18': DriverPrior('18', 'STR', 'Aston Martin', 'backmarker', 'experienced', mu=9, sigma=5),
        '31': DriverPrior('31', 'OCO', 'Alpine', 'backmarker', 'experienced', mu=9, sigma=5),
        '20': DriverPrior('20', 'MAG', 'Haas', 'backmarker', 'experienced', mu=8, sigma=6),
        '27': DriverPrior('27', 'HUL', 'Haas', 'backmarker', 'experienced', mu=8, sigma=6),
        '22': DriverPrior('22', 'TSU', 'AlphaTauri', 'backmarker', 'experienced', mu=8, sigma=6),
        '24': DriverPrior('24', 'ZHO', 'Alfa Romeo', 'backmarker', 'experienced', mu=7, sigma=6),
        '77': DriverPrior('77', 'BOT', 'Alfa Romeo', 'backmarker', 'experienced', mu=8, sigma=6),
        '3': DriverPrior('3', 'RIC', 'AlphaTauri', 'backmarker', 'experienced', mu=9, sigma=5),
        '2': DriverPrior('2', 'SAR', 'Williams', 'backmarker', 'rookie', mu=6, sigma=7),
    }
    
    return priors_2023

# Initialize correct priors
priors_2023 = initialize_2023_standings_priors()

# From cell 17
def predict_prior_only(bayesian_priors: dict) -> pd.DataFrame:
    """
    Baseline: Predict using ONLY Bayesian priors, no testing data.
    
    This is the "do nothing" baseline - just rank drivers by championship standings.
    """
    predictions = []
    
    for driver_num, prior in bayesian_priors.items():
        # Lower μ = better expected position
        predictions.append({
            'driver_number': driver_num,
            'predicted_position': int(21 - prior.mu)
        })
    
    df = pd.DataFrame(predictions)
    df = df.sort_values('predicted_position').reset_index(drop=True)
    
    return df

# From cell 3
def initialize_2026_style_priors() -> Dict[str, DriverPrior]:
    """
    Initialize priors based on 2025 championship standings.
    
    2025 Results inform tier structure:
    - Top 4: NOR(423), VER(421), PIA(410), RUS(319) - Clear elite
    - Next tier: LEC(242), HAM(156), ANT(150) - Strong but gap to top
    - Midfield: ALB(73), SAI(64), ALO(56), HUL(51), HAD(51)
    - Lower: BEA(41), LAW(38), OCO(38), STR(33), TSU(33)
    - Backmarkers: GAS(22), BOR(19), COL(0)
    
    For 2026: Keep tier structure but HIGH uncertainty (new regs)
    """
    priors = {
        # TOP TIER - 2025 Top 4 (clear elite)
        '4': DriverPrior('4', 'NOR', 'McLaren', 'top', 'elite', mu=18, sigma=4),  # Champion
        '1': DriverPrior('1', 'VER', 'Red Bull Racing', 'top', 'elite', mu=18, sigma=4),  # 2 pts behind!
        '81': DriverPrior('81', 'PIA', 'McLaren', 'top', 'elite', mu=17, sigma=4),  # Strong 3rd
        '63': DriverPrior('63', 'RUS', 'Mercedes', 'top', 'elite', mu=17, sigma=4),  # Clear 4th
        
        # UPPER MIDFIELD - Next tier (strong but gap to top)
        '16': DriverPrior('16', 'LEC', 'Ferrari', 'midfield', 'elite', mu=14, sigma=5),  # 242 pts
        '44': DriverPrior('44', 'HAM', 'Ferrari', 'midfield', 'elite', mu=13, sigma=5),  # 156 pts (struggled)
        '12': DriverPrior('12', 'ANT', 'Mercedes', 'midfield', 'rookie', mu=13, sigma=5),  # 150 pts (strong rookie)
        
        # MIDFIELD - Best of rest
        '23': DriverPrior('23', 'ALB', 'Williams', 'midfield', 'experienced', mu=11, sigma=5),  # 73 pts
        '55': DriverPrior('55', 'SAI', 'Williams', 'midfield', 'experienced', mu=11, sigma=5),  # 64 pts
        '14': DriverPrior('14', 'ALO', 'Aston Martin', 'midfield', 'elite', mu=11, sigma=5),  # 56 pts
        '27': DriverPrior('27', 'HUL', 'Kick Sauber', 'midfield', 'experienced', mu=10, sigma=6),  # 51 pts
        '6': DriverPrior('6', 'HAD', 'Racing Bulls', 'midfield', 'rookie', mu=10, sigma=6),  # 51 pts (good rookie)
        
        # LOWER MIDFIELD
        '87': DriverPrior('87', 'BEA', 'Haas F1 Team', 'backmarker', 'rookie', mu=9, sigma=6),  # 41 pts
        '30': DriverPrior('30', 'LAW', 'Racing Bulls', 'backmarker', 'experienced', mu=9, sigma=6),  # 38 pts
        '31': DriverPrior('31', 'OCO', 'Haas F1 Team', 'backmarker', 'experienced', mu=9, sigma=6),  # 38 pts
        '18': DriverPrior('18', 'STR', 'Aston Martin', 'backmarker', 'experienced', mu=8, sigma=6),  # 33 pts
        '22': DriverPrior('22', 'TSU', 'Red Bull Racing', 'backmarker', 'experienced', mu=8, sigma=6),  # 33 pts (struggled)
        
        # BACKMARKERS
        '10': DriverPrior('10', 'GAS', 'Alpine', 'backmarker', 'experienced', mu=7, sigma=6),  # 22 pts
        '5': DriverPrior('5', 'BOR', 'Kick Sauber', 'backmarker', 'rookie', mu=6, sigma=7),  # 19 pts
        '7': DriverPrior('7', 'COL', 'Alpine', 'backmarker', 'rookie', mu=5, sigma=7),  # 0 pts (replaced DOO)
    }
    
    return priors

# Initialize priors
priors = initialize_2026_style_priors()

