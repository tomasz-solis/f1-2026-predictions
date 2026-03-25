"""Apply coarse 2026 regulation adjustments to preseason priors."""

from .bayesian import DriverPrior


def apply_2026_regulations(priors: dict[str, DriverPrior]) -> dict[str, DriverPrior]:
    """Adjust priors for broad 2026 regulation scenarios."""

    ENGINE_BOOST = 1.5
    NEW_TEAM_PENALTY = -2.0
    CUSTOMER_PENALTY = -0.5

    adjusted_priors = {}

    for d_num, p in priors.items():
        new_mu = p.mu
        new_sigma = p.sigma + 1.0

        if p.team in ["Mercedes", "Ferrari"]:
            new_mu += ENGINE_BOOST
        elif p.team == "Kick Sauber":
            new_mu += NEW_TEAM_PENALTY
        elif "Customer" in p.team_tier:
            new_mu += CUSTOMER_PENALTY

        adjusted_priors[d_num] = DriverPrior(
            driver_number=p.driver_number,
            driver_code=p.driver_code,
            team=p.team,
            team_tier=p.team_tier,
            mu=new_mu,
            sigma=new_sigma,
        )

    return adjusted_priors
