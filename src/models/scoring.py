"""Performance scoring methods for driver ranking."""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

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
    """Score = (value - median) in actual units."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed', 
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns and pd.notna(row[feature_col]):
                    median_val = testing_features[feature_col].median()
                    score[f'{metric_name}_score'] = row[feature_col] - median_val
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


class RankingScoring(PerformanceScoringMethod):
    """Score = rank (1 = best, 20 = worst)."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed',
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns:
                    # Rank (ascending=False: highest value = rank 1)
                    rank = testing_features[feature_col].rank(ascending=False, method='min', na_option='keep')
                    score[f'{metric_name}_score'] = rank.loc[idx]
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


class QuantileScoring(PerformanceScoringMethod):
    """Score = quantile tier (3 = top 25%, 2 = middle 50%, 1 = bottom 25%)."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed',
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns and pd.notna(row[feature_col]):
                    val = row[feature_col]
                    q75 = testing_features[feature_col].quantile(0.75)
                    q25 = testing_features[feature_col].quantile(0.25)
                    
                    if val >= q75:
                        tier = 3  # Top 25%
                    elif val >= q25:
                        tier = 2  # Middle 50%
                    else:
                        tier = 1  # Bottom 25%
                    
                    score[f'{metric_name}_score'] = tier
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


class ZScoreScoring(PerformanceScoringMethod):
    """Score = standardized z-score."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed',
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns and pd.notna(row[feature_col]):
                    mean_val = testing_features[feature_col].mean()
                    std_val = testing_features[feature_col].std()
                    
                    if std_val > 0:
                        z_score = (row[feature_col] - mean_val) / std_val
                        score[f'{metric_name}_score'] = z_score
                    else:
                        score[f'{metric_name}_score'] = 0
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)



class PerformanceScoringMethod:
    """Base class for different scoring approaches."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        """Score drivers on each characteristic."""
        raise NotImplementedError


class AbsoluteDifferenceScoring(PerformanceScoringMethod):
    """Score = (value - median) in actual units."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed', 
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns and pd.notna(row[feature_col]):
                    median_val = testing_features[feature_col].median()
                    score[f'{metric_name}_score'] = row[feature_col] - median_val
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


class RankingScoring(PerformanceScoringMethod):
    """Score = rank (1 = best, 20 = worst)."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed',
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns:
                    rank = testing_features[feature_col].rank(ascending=False, method='min', na_option='keep')
                    score[f'{metric_name}_score'] = rank.loc[idx]
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


class QuantileScoring(PerformanceScoringMethod):
    """Score = quantile tier (3 = top 25%, 2 = middle 50%, 1 = bottom 25%)."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed',
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns and pd.notna(row[feature_col]):
                    val = row[feature_col]
                    q75 = testing_features[feature_col].quantile(0.75)
                    q25 = testing_features[feature_col].quantile(0.25)
                    
                    if val >= q75:
                        tier = 3
                    elif val >= q25:
                        tier = 2
                    else:
                        tier = 1
                    
                    score[f'{metric_name}_score'] = tier
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


class ZScoreScoring(PerformanceScoringMethod):
    """Score = standardized z-score."""
    
    def score_drivers(self, testing_features: pd.DataFrame) -> pd.DataFrame:
        scores = []
        
        features = {
            'slow_corner': 'slow_corner_speed',
            'medium_corner': 'medium_corner_speed',
            'high_corner': 'high_corner_speed',
            'straight': 'avg_speed_full_throttle',
            'throttle_usage': 'pct_full_throttle'
        }
        
        for idx, row in testing_features.iterrows():
            driver = row['driver_number']
            score = {'driver_number': driver}
            
            for metric_name, feature_col in features.items():
                if feature_col in testing_features.columns and pd.notna(row[feature_col]):
                    mean_val = testing_features[feature_col].mean()
                    std_val = testing_features[feature_col].std()
                    
                    if std_val > 0:
                        z_score = (row[feature_col] - mean_val) / std_val
                        score[f'{metric_name}_score'] = z_score
                    else:
                        score[f'{metric_name}_score'] = 0
                else:
                    score[f'{metric_name}_score'] = np.nan
            
            scores.append(score)
        
        return pd.DataFrame(scores)


