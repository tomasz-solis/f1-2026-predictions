"""Bayesian ranking for F1 predictions."""

import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class DriverPrior:
    """Prior belief about driver performance."""
    driver_number: str
    driver_code: str
    team: str
    team_tier: str  # 'top', 'midfield', 'backmarker'
    driver_tier: str  # 'elite', 'experienced', 'rookie'
    mu: float  # Expected rating (higher = better)
    sigma: float  # Uncertainty

class BayesianDriverRanking:
    """
    Bayesian ranking with Gaussian priors.
    Each driver: rating ~ N(μ, σ²)
    Update via Bayesian inference.
    """
    
    def __init__(self, priors):
        """Initialize with prior beliefs (dict of driver_number -> DriverPrior)."""
        self.ratings = {}
        self.n_observations = {}
        
        for driver_num, prior in priors.items():
            self.ratings[driver_num] = (prior.mu, prior.sigma)
            self.n_observations[driver_num] = 0
        
        self.priors = priors
        self.update_history = []
    
    def predict_positions(self):
        """Predict positions based on current ratings."""
        predictions = []
        
        for driver_num, (mu, sigma) in self.ratings.items():
            # Rating to position: position = 21 - rating
            predicted_pos = 21 - mu
            ci_lower = max(1, predicted_pos - 1.96 * sigma)
            ci_upper = min(20, predicted_pos + 1.96 * sigma)
            
            predictions.append({
                'driver_number': driver_num,
                'driver_code': self.priors[driver_num].driver_code,
                'team': self.priors[driver_num].team,
                'rating_mu': mu,
                'rating_sigma': sigma,
                'predicted_position': predicted_pos,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_observations': self.n_observations[driver_num]
            })
        
        df = pd.DataFrame(predictions)
        df = df.sort_values('predicted_position')
        df['predicted_rank'] = range(1, len(df) + 1)
        
        return df
    
    def update_from_session(self, observed_positions, confidence_weight=1.0, session_name="Session"):
        """
        Update ratings from observed positions.
        
        confidence_weight: How much to trust this observation
        - 0.1 = testing (low trust)
        - 0.3 = practice (medium)
        - 0.8 = sprint quali (high)
        - 1.0 = race (full trust)
        """
        for driver_num, observed_pos in observed_positions.items():
            if driver_num not in self.ratings:
                continue
            
            prior_mu, prior_sigma = self.ratings[driver_num]
            
            # Position to rating
            observed_rating = 21 - observed_pos
            
            # Observation uncertainty (inversely proportional to confidence)
            obs_sigma = 5.0 / confidence_weight
            
            # Bayesian update
            new_sigma_sq = 1 / (1/prior_sigma**2 + 1/obs_sigma**2)
            new_sigma = np.sqrt(new_sigma_sq)
            new_mu = (prior_mu/prior_sigma**2 + observed_rating/obs_sigma**2) * new_sigma_sq
            
            self.ratings[driver_num] = (new_mu, new_sigma)
            self.n_observations[driver_num] += 1
            
            self.update_history.append({
                'session': session_name,
                'driver_number': driver_num,
                'driver_code': self.priors[driver_num].driver_code,
                'observed_position': observed_pos,
                'prior_mu': prior_mu,
                'prior_sigma': prior_sigma,
                'new_mu': new_mu,
                'new_sigma': new_sigma,
                'confidence_weight': confidence_weight
            })
    
    def get_update_summary(self):
        """Get summary of all updates."""
        return pd.DataFrame(self.update_history)
