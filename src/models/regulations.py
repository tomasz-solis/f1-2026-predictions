"""Apply 2026 regulation adjustments to preseason priors.

The 2026 reset changes where performance should come from. Power units shift
toward a much larger electrical contribution, active aero replaces DRS, and the
cars are meant to produce a cleaner wake than the first wave of ground-effect
rules. That does not mean we know who will be fastest, but it does justify a
small factory-program boost, a learning-curve penalty for brand-new entries,
and extra uncertainty for everyone.
"""

from __future__ import annotations

from .bayesian import DriverPrior

FACTORY_PU_TEAMS = {"Mercedes", "Ferrari", "Red Bull Racing", "Alpine", "Audi"}
NEW_ENTRY_TEAMS = {"Cadillac F1"}


def apply_2026_regulations(
    priors: dict[str, DriverPrior],
    *,
    factory_program_boost: float = 1.0,
    new_entry_penalty: float = -1.5,
    customer_penalty: float = -0.3,
    regulation_reset_sigma_bump: float = 1.0,
) -> dict[str, DriverPrior]:
    """Adjust priors for the 2026 rules reset.

    The intent is not to hard-code the finishing order. It is to admit that
    in-house power-unit programs and integrated packaging matter more than
    usual when the rules move this much, while still widening uncertainty
    enough that early race results can take over quickly.
    """
    adjusted_priors: dict[str, DriverPrior] = {}

    for driver_number, prior in priors.items():
        adjusted_mu = prior.mu
        adjusted_sigma = prior.sigma + regulation_reset_sigma_bump

        if prior.team in FACTORY_PU_TEAMS:
            adjusted_mu += factory_program_boost
        elif prior.team in NEW_ENTRY_TEAMS:
            adjusted_mu += new_entry_penalty
        else:
            adjusted_mu += customer_penalty

        adjusted_priors[driver_number] = DriverPrior(
            driver_number=prior.driver_number,
            driver_code=prior.driver_code,
            team=prior.team,
            team_tier=prior.team_tier,
            mu=adjusted_mu,
            sigma=adjusted_sigma,
        )

    return adjusted_priors
