"""Bayesian driver ratings."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config_loader import get

MIN_SIGMA = 0.05


@dataclass
class DriverPrior:
    """Initial rating state for one driver."""

    driver_number: str
    driver_code: str
    team: str
    team_tier: str  # 'top', 'midfield', 'backmarker'
    mu: float  # Expected rating (Higher = Better performance)
    sigma: float  # Uncertainty (Standard Deviation)


@dataclass
class UpdateRecord:
    """One saved rating update."""

    driver_number: str
    session_name: str
    observed_pos: int
    prior_mu: float
    prior_sigma: float
    posterior_mu: float
    posterior_sigma: float
    shock_factor: float


class BayesianDriverRanking:
    """Track driver ratings across sessions."""

    def __init__(self, priors: dict[str, DriverPrior], grid_size: int = 22):
        """Initialize ratings for the current grid."""
        self.priors = priors
        self.grid_size = max(int(grid_size), 2)
        self.ratings: dict[str, tuple[float, float]] = {
            d: (p.mu, p.sigma) for d, p in priors.items()
        }
        self.history: list[UpdateRecord] = []

    def get_current_ratings(self) -> pd.DataFrame:
        """Return the current ratings table."""
        data = []
        for d_num, (mu, sigma) in self.ratings.items():
            prior = self.priors[d_num]
            expected_pos = np.clip((self.grid_size + 1) - mu, 1, self.grid_size)

            data.append(
                {
                    "driver_number": d_num,
                    "driver_code": prior.driver_code,
                    "team": prior.team,
                    "rating_mu": round(mu, 2),
                    "rating_sigma": round(sigma, 2),
                    "expected_position": round(expected_pos, 1),
                    "lower_ci": round(
                        np.clip((self.grid_size + 1) - (mu + 1.96 * sigma), 1, self.grid_size),
                        1,
                    ),
                    "upper_ci": round(
                        np.clip((self.grid_size + 1) - (mu - 1.96 * sigma), 1, self.grid_size),
                        1,
                    ),
                }
            )

        return pd.DataFrame(data).sort_values("rating_mu", ascending=False)

    @staticmethod
    def _load_update_hyperparameters() -> tuple[float, float, float, float]:
        """Read update settings from config."""
        try:
            base_volatility = float(get("bayesian.base_volatility", 0.1))
            shock_threshold = float(get("bayesian.shock_threshold", 2.0))
            shock_multiplier = float(get("bayesian.shock_multiplier", 0.5))
            base_obs_noise = float(get("bayesian.base_observation_noise", 2.0))
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            base_volatility = 0.1
            shock_threshold = 2.0
            shock_multiplier = 0.5
            base_obs_noise = 2.0

        return base_volatility, shock_threshold, shock_multiplier, base_obs_noise

    def _position_to_rating(self, finish_pos: int | float) -> float:
        """Map a finishing position to the rating scale."""
        return float(self.grid_size + 1) - float(finish_pos)

    def _rating_to_position(self, observed_rating: float) -> int:
        """Map a rating back to a finishing position."""
        projected_position = round(float(self.grid_size + 1) - float(observed_rating))
        return int(np.clip(projected_position, 1, self.grid_size))

    def _apply_rating_updates(
        self,
        *,
        observed_ratings: dict[str, float],
        session_name: str,
        confidence: float,
        observed_positions: dict[str, int] | None = None,
    ) -> None:
        """Apply one update pass from observed ratings."""
        (
            base_volatility,
            shock_threshold,
            shock_multiplier,
            base_obs_noise,
        ) = self._load_update_hyperparameters()
        for driver_number, observed_rating in observed_ratings.items():
            observed_pos = (
                observed_positions[driver_number]
                if observed_positions is not None and driver_number in observed_positions
                else self._rating_to_position(observed_rating)
            )
            self._update_single_driver_rating(
                driver_number=driver_number,
                observed_rating=observed_rating,
                observed_pos=observed_pos,
                session_name=session_name,
                confidence=confidence,
                base_volatility=base_volatility,
                shock_threshold=shock_threshold,
                shock_multiplier=shock_multiplier,
                base_obs_noise=base_obs_noise,
            )

    def update(
        self, observations: dict[str, int], session_name: str, confidence: float = 1.0
    ) -> None:
        """Update ratings from finishing positions."""
        observed_ratings = {
            driver_code: self._position_to_rating(finish_pos)
            for driver_code, finish_pos in observations.items()
        }
        self._apply_rating_updates(
            observed_ratings=observed_ratings,
            session_name=session_name,
            confidence=confidence,
            observed_positions=observations,
        )

    def update_teammate_relative(
        self,
        observations: dict[str, int],
        session_name: str,
        lineups: dict[str, list[str]],
        confidence: float = 1.0,
    ) -> None:
        """Update ratings from teammate-relative results."""
        driver_to_team: dict[str, str] = {}
        for team_name, drivers in lineups.items():
            for driver_code in drivers:
                driver_to_team[str(driver_code)] = str(team_name)

        observed_ratings = {
            driver_code: self._position_to_rating(finish_pos)
            for driver_code, finish_pos in observations.items()
        }
        team_ratings: dict[str, list[float]] = {}
        for driver_code, observed_rating in observed_ratings.items():
            driver_team = driver_to_team.get(driver_code)
            if driver_team is None:
                continue
            team_ratings.setdefault(driver_team, []).append(observed_rating)

        team_means = {
            team_name: float(np.mean(ratings))
            for team_name, ratings in team_ratings.items()
            if len(ratings) >= 2
        }
        if not team_means:
            self.update(observations=observations, session_name=session_name, confidence=confidence)
            return

        field_mean = (
            float(np.mean(list(observed_ratings.values())))
            if observed_ratings
            else float(self.grid_size + 1) / 2.0
        )
        adjusted_ratings: dict[str, float] = {}
        adjusted_positions: dict[str, int] = {}

        for driver_code, _finish_pos in observations.items():
            raw_rating = observed_ratings[driver_code]
            driver_team = driver_to_team.get(driver_code)
            adjusted_rating = raw_rating
            if driver_team is not None and driver_team in team_means:
                adjusted_rating = raw_rating - team_means[driver_team] + field_mean

            adjusted_ratings[driver_code] = adjusted_rating
            adjusted_positions[driver_code] = self._rating_to_position(adjusted_rating)

        self._apply_rating_updates(
            observed_ratings=adjusted_ratings,
            session_name=session_name,
            confidence=confidence,
            observed_positions=adjusted_positions,
        )

    def get_history_df(self) -> pd.DataFrame:
        """Export update history for visualization."""
        return pd.DataFrame([vars(r) for r in self.history])

    def _update_single_driver_rating(
        self,
        *,
        driver_number: str,
        observed_rating: float,
        observed_pos: int,
        session_name: str,
        confidence: float,
        base_volatility: float,
        shock_threshold: float,
        shock_multiplier: float,
        base_obs_noise: float,
    ) -> None:
        """Apply one conjugate Normal-Normal update for a single driver."""
        if driver_number not in self.ratings:
            return

        prior_mu, prior_sigma = self.ratings[driver_number]
        prior_sigma = np.sqrt(prior_sigma**2 + base_volatility**2)
        innovation = abs(observed_rating - prior_mu)
        shock = 0.0
        if innovation > (shock_threshold * prior_sigma):
            shock = shock_multiplier * (innovation / prior_sigma)
        obs_noise = base_obs_noise / (confidence + 1e-6)
        effective_prior_sigma = prior_sigma * (1.0 + shock)
        prior_prec = 1.0 / (effective_prior_sigma**2)
        obs_prec = 1.0 / (obs_noise**2)

        posterior_sigma_sq = 1.0 / (prior_prec + obs_prec)
        posterior_mu = (prior_mu * prior_prec + observed_rating * obs_prec) * posterior_sigma_sq
        posterior_sigma = np.sqrt(posterior_sigma_sq)
        posterior_sigma = max(posterior_sigma, MIN_SIGMA)

        self.ratings[driver_number] = (posterior_mu, posterior_sigma)

        self.history.append(
            UpdateRecord(
                driver_number=driver_number,
                session_name=session_name,
                observed_pos=observed_pos,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                posterior_mu=posterior_mu,
                posterior_sigma=posterior_sigma,
                shock_factor=shock,
            )
        )
