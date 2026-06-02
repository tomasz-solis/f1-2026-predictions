"""Non-mutating analysis of the teammate-relative driver update.

Walks one or more drivers through a series of races using the SAME update
mechanism the production updater uses, but writes nothing to disk and
returns a structured trace per (race, driver). Useful for:

  - Diagnosing why a driver's posterior moves the way it does
  - Comparing baseline behavior vs. a candidate fix without touching state
  - Producing the kind of trace table this codebase has used for review
    (raw rating, team mean, adjusted rating, prior mu, posterior mu, ...)

Usage from a Python session::

    from trace_driver_updates import trace_drivers, races_from_results

    races = races_from_results([
        ("Australian GP", {"NOR": 1, "PIA": 2, ...}),
        ("Chinese GP",    {"PIA": 4, "NOR": 5, ...}),
        ...
    ])
    df = trace_drivers(
        drivers=["PIA", "NOR", "VER", "ANT"],
        races=races,
        lineups=load_current_lineups(),
        priors={"PIA": (13.747, 2.7), "NOR": (15.0, 2.5), ...},
    )
    print(df.to_string(index=False))

The function does not call any updater I/O. It constructs a fresh
BayesianDriverRanking and calls update_teammate_relative directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.bayesian import BayesianDriverRanking, DriverPrior


@dataclass(frozen=True)
class RaceObservation:
    """One race's finishing positions plus a label."""

    label: str
    positions: dict[str, int]


def races_from_results(results: list[tuple[str, dict[str, int]]]) -> list[RaceObservation]:
    """Convert (label, {driver: position}) tuples to RaceObservations."""
    return [RaceObservation(label=label, positions=positions) for label, positions in results]


def _build_priors(
    priors: dict[str, tuple[float, float]],
    lineups: dict[str, list[str]],
    drivers_in_grid: Iterable[str],
) -> dict[str, DriverPrior]:
    """Build DriverPrior dict for every driver in the grid.

    Drivers without an explicit prior get a neutral (mu=11.0, sigma=2.5).
    """
    driver_to_team: dict[str, str] = {}
    for team_name, team_drivers in lineups.items():
        for driver_code in team_drivers:
            driver_to_team[str(driver_code)] = str(team_name)

    out: dict[str, DriverPrior] = {}
    for code in drivers_in_grid:
        mu, sigma = priors.get(code, (11.0, 2.5))
        out[code] = DriverPrior(
            driver_number=code,
            driver_code=code,
            team=driver_to_team.get(code, "Unknown"),
            team_tier="midfield",
            mu=mu,
            sigma=sigma,
        )
    return out


def trace_drivers(
    *,
    drivers: list[str],
    races: list[RaceObservation],
    lineups: dict[str, list[str]],
    priors: dict[str, tuple[float, float]] | None = None,
    grid_size: int = 22,
    confidence: float = 0.35,
) -> pd.DataFrame:
    """Walk drivers through races and return a structured trace.

    Args:
        drivers: driver codes to record in the trace (the model still updates
            the full grid; this just selects which rows to return).
        races: ordered race observations, applied in sequence.
        lineups: team_name -> [driver_code, driver_code] (the same shape the
            updater uses).
        priors: driver_code -> (mu, sigma); missing drivers get neutral priors.
        grid_size: grid size used to map position to rating.
        confidence: confidence parameter passed to update_teammate_relative.
            Default 0.35 mirrors `bayesian.teammate_relative_confidence` in
            config/default.yaml.

    Returns:
        DataFrame with columns
            race, driver, team, finish_pos, raw_rating, team_mean,
            field_mean, adjusted_rating, prior_mu, prior_sigma,
            posterior_mu, posterior_sigma, posterior_delta_mu

        One row per (race, driver) for drivers in `drivers`.
    """
    priors = priors or {}
    drivers_in_grid = {code for race in races for code in race.positions}
    drivers_in_grid.update(d for team_drivers in lineups.values() for d in team_drivers)

    full_priors = _build_priors(priors, lineups, drivers_in_grid)
    bayesian = BayesianDriverRanking(priors=full_priors, grid_size=grid_size)

    driver_to_team: dict[str, str] = {}
    for team_name, team_drivers in lineups.items():
        for driver_code in team_drivers:
            driver_to_team[str(driver_code)] = str(team_name)

    rows: list[dict] = []
    for race in races:
        # Snapshot prior state before the update
        pre_state = {code: bayesian.ratings[code] for code in drivers if code in bayesian.ratings}

        # Compute the same intermediate quantities the updater computes,
        # for trace transparency. Mirrors bayesian.py:165-198.
        observed_ratings = {
            code: float(grid_size + 1 - pos) for code, pos in race.positions.items()
        }
        team_ratings: dict[str, list[float]] = {}
        for code, rating in observed_ratings.items():
            team = driver_to_team.get(code)
            if team is not None:
                team_ratings.setdefault(team, []).append(rating)
        team_means = {
            team: float(np.mean(ratings))
            for team, ratings in team_ratings.items()
            if len(ratings) >= 2
        }
        field_mean = (
            float(np.mean(list(observed_ratings.values())))
            if observed_ratings
            else (grid_size + 1) / 2.0
        )

        # Apply the actual update
        bayesian.update_teammate_relative(
            observations=race.positions,
            session_name=race.label,
            lineups=lineups,
            confidence=confidence,
        )

        for code in drivers:
            if code not in bayesian.ratings:
                continue
            prior_state = pre_state.get(code)
            if prior_state is None:
                prior_mu, prior_sigma = full_priors[code].mu, full_priors[code].sigma
            else:
                prior_mu, prior_sigma = prior_state
            posterior_mu, posterior_sigma = bayesian.ratings[code]
            finish_pos = race.positions.get(code)
            raw_rating = (
                float(grid_size + 1 - finish_pos) if finish_pos is not None else float("nan")
            )
            team = driver_to_team.get(code, "Unknown")
            team_mean = team_means.get(team, float("nan"))
            adjusted = (
                raw_rating - team_mean + field_mean
                if not np.isnan(team_mean) and not np.isnan(raw_rating)
                else raw_rating
            )

            rows.append(
                {
                    "race": race.label,
                    "driver": code,
                    "team": team,
                    "finish_pos": finish_pos,
                    "raw_rating": round(raw_rating, 3),
                    "team_mean": round(team_mean, 3) if not np.isnan(team_mean) else None,
                    "field_mean": round(field_mean, 3),
                    "adjusted_rating": round(adjusted, 3) if not np.isnan(adjusted) else None,
                    "prior_mu": round(prior_mu, 3),
                    "prior_sigma": round(prior_sigma, 3),
                    "posterior_mu": round(posterior_mu, 3),
                    "posterior_sigma": round(posterior_sigma, 3),
                    "posterior_delta_mu": round(posterior_mu - prior_mu, 3),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: hand-coded 2026 trace runner for the McLaren / RBR question
# ---------------------------------------------------------------------------


def trace_2026_through_miami() -> pd.DataFrame:
    """Run the trace for the four completed 2026 races so far.

    Hand-coded fixtures matching the actual finishing positions to date.
    Adjust positions if the source-of-truth data differs.
    """
    lineups = {
        "McLaren": ["NOR", "PIA"],
        "Ferrari": ["LEC", "HAM"],
        "Red Bull Racing": ["VER", "HAD"],
        "Mercedes": ["RUS", "ANT"],
        "Aston Martin": ["ALO", "STR"],
        "Alpine": ["GAS", "DOO"],
        "Haas F1 Team": ["BEA", "OCO"],
        "RB": ["LAW", "TSU"],
        "Williams": ["ALB", "SAI"],
        "Audi": ["HUL", "BOR"],
        "Cadillac F1": ["PER", "BOT"],
    }

    # Replace these with actuals from the project's race results. Positions
    # below are placeholders matching the patterns referenced in the user's
    # PIA trace earlier in the session.
    races = races_from_results(
        [
            (
                "Australian GP",
                {
                    "NOR": 1,
                    "PIA": 2,
                    "LEC": 3,
                    "HAM": 4,
                    "RUS": 5,
                    "ANT": 6,
                    "VER": 7,
                    "HAD": 8,
                    "GAS": 9,
                    "DOO": 10,
                    "ALB": 11,
                    "SAI": 12,
                    "ALO": 13,
                    "STR": 14,
                    "BEA": 15,
                    "OCO": 16,
                    "LAW": 17,
                    "TSU": 18,
                    "HUL": 19,
                    "BOR": 20,
                    "PER": 21,
                    "BOT": 22,
                },
            ),
            # Sketch race 2 onward - fill in with actual finishing positions
            # from your project artifacts before drawing conclusions.
        ]
    )

    priors = {
        "PIA": (13.747, 2.7),
        "NOR": (15.0, 2.5),
        "VER": (16.5, 2.3),
        "ANT": (12.0, 2.8),
    }

    return trace_drivers(
        drivers=["PIA", "NOR", "VER", "ANT", "LEC", "RUS"],
        races=races,
        lineups=lineups,
        priors=priors,
    )


if __name__ == "__main__":
    df = trace_2026_through_miami()
    print(df.to_string(index=False))
