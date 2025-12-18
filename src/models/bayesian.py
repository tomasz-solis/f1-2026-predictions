"""Bayesian ranking system for F1 predictions."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class DriverPrior:
    """Prior belief about driver performance."""
    driver_number: str
    driver_code: str
    team: str
    team_tier: str  # 'top', 'midfield', 'backmarker'
    driver_tier: str  # 'elite', 'experienced', 'rookie'
    
    # Initial rating (μ, σ)
    # μ = expected position (higher = better, 20 = best, 1 = worst)
    # σ = uncertainty (higher = more uncertain)
    mu: float
    sigma: float

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


class BayesianDriverRanking:
    """
    Bayesian ranking system with Gaussian priors.
    
    Each driver has rating ~ N(μ, σ²)
    - μ = performance rating (higher = better)
    - σ = uncertainty (decreases as we observe more)
    
    Update rule: Bayesian inference with observed positions
    """
    
    def __init__(self, priors: Dict[str, DriverPrior]):
        """
        Initialize with prior beliefs.
        
        Parameters
        ----------
        priors : dict
            driver_number -> DriverPrior
        """
        self.ratings = {}  # driver_number -> (mu, sigma)
        self.n_observations = {}  # driver_number -> count
        
        for driver_num, prior in priors.items():
            self.ratings[driver_num] = (prior.mu, prior.sigma)
            self.n_observations[driver_num] = 0
        
        self.priors = priors
        self.update_history = []  # Track all updates
    
    def predict_positions(self) -> pd.DataFrame:
        """
        Predict positions based on current ratings.
        
        Returns
        -------
        pd.DataFrame
            Columns: driver_number, predicted_position, ci_lower, ci_upper
            Sorted by predicted position (best first)
        """
        predictions = []
        
        for driver_num, (mu, sigma) in self.ratings.items():
            # Convert rating to position (invert scale)
            # Rating 18 → Position ~2-3
            # Rating 10 → Position ~10-11
            # Rating 3  → Position ~18-19
            
            # Simple linear mapping: position = 21 - rating
            predicted_pos = 21 - mu
            
            # Confidence interval (95% = ±1.96σ)
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
    
    def update_from_session(
        self,
        observed_positions: Dict[str, int],
        confidence_weight: float = 1.0,
        session_name: str = "Session"
    ):
        """
        Update ratings based on observed positions.
        
        Uses Bayesian update with Gaussian likelihood.
        
        Parameters
        ----------
        observed_positions : dict
            driver_number -> finishing position (1-20)
        confidence_weight : float
            How much to trust this observation
            - 0.1 = testing (low trust)
            - 0.3 = practice (medium trust)
            - 0.7 = qualifying (high trust)
            - 1.0 = race (full trust)
        session_name : str
            For logging
        """
        # Bayesian update: posterior = likelihood × prior
        # For Gaussian: new_sigma² = 1 / (1/prior_sigma² + 1/obs_sigma²)
        #               new_mu = (prior_mu/prior_sigma² + obs_mu/obs_sigma²) × new_sigma²
        
        for driver_num, observed_pos in observed_positions.items():
            if driver_num not in self.ratings:
                continue  # Skip unknown drivers
            
            prior_mu, prior_sigma = self.ratings[driver_num]
            
            # Convert observed position to rating
            # Position 1 → Rating ~19-20
            # Position 10 → Rating ~10-11
            # Position 20 → Rating ~1-2
            observed_rating = 21 - observed_pos
            
            # Observation uncertainty (inversely proportional to confidence)
            # Higher confidence = lower observation uncertainty
            obs_sigma = 5.0 / confidence_weight  # Range: 5 (race) to 50 (testing)
            
            # Bayesian update
            new_sigma_sq = 1 / (1/prior_sigma**2 + 1/obs_sigma**2)
            new_sigma = np.sqrt(new_sigma_sq)
            
            new_mu = (prior_mu/prior_sigma**2 + observed_rating/obs_sigma**2) * new_sigma_sq
            
            # Update ratings
            self.ratings[driver_num] = (new_mu, new_sigma)
            self.n_observations[driver_num] += 1
            
            # Log update
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
    
    def get_update_summary(self) -> pd.DataFrame:
        """Get summary of all updates."""
        return pd.DataFrame(self.update_history)