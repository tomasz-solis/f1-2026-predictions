"""Bayesian driver ranking with volatility-adjusted priors.

Uses a conjugate Normal-Normal update. The volatility parameter inflates
uncertainty when observations deviate from priors, which helps with
regulation changes and mid-season form shifts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriverPrior:
    """Configuration for a driver's initial belief state."""

    driver_number: str
    driver_code: str
    team: str
    team_tier: str  # 'top', 'midfield', 'backmarker'
    mu: float  # Expected rating (Higher = Better performance)
    sigma: float  # Uncertainty (Standard Deviation)


@dataclass
class UpdateRecord:
    """Audit trail for a single Bayesian update."""

    driver_number: str
    session_name: str
    observed_pos: int
    prior_mu: float
    prior_sigma: float
    posterior_mu: float
    posterior_sigma: float
    shock_factor: float


class BayesianDriverRanking:
    """
    Stateful engine for tracking driver ratings using Bayesian Inference.

    Model:
        Rating ~ Normal(mu, sigma^2)
        Update Rule: Conjugate Normal-Normal update.

    Key Features:
        - Dynamic Volatility: Increases sigma when prediction error is high.
        - Confidence Weighting: Trust Race results more than Practice.
    """

    def __init__(self, priors: dict[str, DriverPrior]):
        self.priors = priors
        # State: {driver_number: (mu, sigma)}
        self.ratings: dict[str, tuple[float, float]] = {
            d: (p.mu, p.sigma) for d, p in priors.items()
        }
        self.history: list[UpdateRecord] = []

    def get_current_ratings(self) -> pd.DataFrame:
        """Return current ratings as a DataFrame for analysis."""
        data = []
        for d_num, (mu, sigma) in self.ratings.items():
            prior = self.priors[d_num]
            # Convert latent rating to expected position (approximate inverse)
            # Simple heuristic: Position ~= 21 - mu (clamped 1-20)
            expected_pos = np.clip(21 - mu, 1, 20)

            data.append(
                {
                    "driver_number": d_num,
                    "driver_code": prior.driver_code,
                    "team": prior.team,
                    "rating_mu": round(mu, 2),
                    "rating_sigma": round(sigma, 2),
                    "expected_position": round(expected_pos, 1),
                    "lower_ci": round(np.clip(21 - (mu + 1.96 * sigma), 1, 20), 1),
                    "upper_ci": round(np.clip(21 - (mu - 1.96 * sigma), 1, 20), 1),
                }
            )

        return pd.DataFrame(data).sort_values("rating_mu", ascending=False)

    def update(
        self, observations: dict[str, int], session_name: str, confidence: float = 1.0
    ) -> None:
        """Update driver ratings based on observed race results with specified confidence."""
        # Load config for volatility and other parameters
        try:
            from src.utils.config_loader import get

            BASE_VOLATILITY = get("bayesian.base_volatility", 0.1)
            SHOCK_THRESHOLD = get("bayesian.shock_threshold", 2.0)
            SHOCK_MULTIPLIER = get("bayesian.shock_multiplier", 0.5)
            BASE_OBS_NOISE = get("bayesian.base_observation_noise", 2.0)
        except (ImportError, FileNotFoundError, KeyError):
            # Fallback to defaults if config not available
            BASE_VOLATILITY = 0.1
            SHOCK_THRESHOLD = 2.0
            SHOCK_MULTIPLIER = 0.5
            BASE_OBS_NOISE = 2.0

        for d_num, finish_pos in observations.items():
            if d_num not in self.ratings:
                continue

            prior_mu, prior_sigma = self.ratings[d_num]

            # --- 1. Process Noise (Volatility Injection) ---
            # Widen the prior slightly to account for development since last race
            prior_sigma = np.sqrt(prior_sigma**2 + BASE_VOLATILITY**2)

            # --- 2. Observation Logic ---
            # Convert position to latent performance rating (High rating = Low Position)
            # Map Position 1 -> Rating 20, Pos 20 -> Rating 1
            observed_rating = 21.0 - finish_pos

            # Calculate Innovation (Prediction Error)
            innovation = abs(observed_rating - prior_mu)

            # --- 3. Adaptive Shock Factor ---
            # If result is > threshold std devs away, assume Concept Drift
            # and inflate sigma further to learn faster.
            shock = 0.0
            if innovation > (SHOCK_THRESHOLD * prior_sigma):
                shock = SHOCK_MULTIPLIER * (innovation / prior_sigma)

            # Effective observation noise (High confidence = Low noise)
            obs_noise = BASE_OBS_NOISE / (confidence + 1e-6)

            # Inflate prior uncertainty if shocked
            effective_prior_sigma = prior_sigma * (1.0 + shock)

            # --- 4. Bayesian Update (Normal-Normal Conjugate) ---
            # Precision = 1 / variance
            prior_prec = 1.0 / (effective_prior_sigma**2)
            obs_prec = 1.0 / (obs_noise**2)

            posterior_sigma_sq = 1.0 / (prior_prec + obs_prec)
            posterior_mu = (prior_mu * prior_prec + observed_rating * obs_prec) * posterior_sigma_sq
            posterior_sigma = np.sqrt(posterior_sigma_sq)

            # Update State
            self.ratings[d_num] = (posterior_mu, posterior_sigma)

            # Log for audit
            self.history.append(
                UpdateRecord(
                    driver_number=d_num,
                    session_name=session_name,
                    observed_pos=finish_pos,
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    posterior_mu=posterior_mu,
                    posterior_sigma=posterior_sigma,
                    shock_factor=shock,
                )
            )

    # Alias for backward compatibility with older scripts/notebooks
    update_from_session = update

    def get_history_df(self) -> pd.DataFrame:
        """Export update history for visualization."""
        return pd.DataFrame([vars(r) for r in self.history])
