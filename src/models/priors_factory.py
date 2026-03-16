"""
src/models/priors_factory.py

Generates Bayesian Priors by combining:
1. Car Performance (Dominant Factor)
   - PRIMARY: From '2026_car_characteristics.json' (Testing Data)
   - FALLBACK: Derived dynamically from 2025 Driver Pace (The "Car-Limited" Logic)
2. Driver Skill (Marginal Factor)
   - From 'driver_characteristics.json' (Historical Data)
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.models.bayesian import DriverPrior

logger = logging.getLogger(__name__)


def _driver_characteristics_fallback_paths(data_dir: Path, season_year: int) -> tuple[Path, ...]:
    """Return season-aware driver-characteristics fallbacks in preferred order."""
    return (
        data_dir / "driver_characteristics" / f"{int(season_year)}_driver_characteristics.json",
        data_dir / "driver_characteristics.json",
    )


class PriorsFactory:
    """Build Bayesian driver priors from persisted driver and car characteristics."""

    def __init__(self, data_dir="data/processed", season_year: int = 2026):
        """Initialize the priors factory with season-aware artifact paths."""
        self.data_dir = Path(data_dir)
        self.season_year = int(season_year)
        self.car_file = (
            self.data_dir / "car_characteristics" / f"{self.season_year}_car_characteristics.json"
        )

    def load_data(self):
        """Load artifacts or initialize fallbacks."""
        # 1. Load Driver Data
        driver_payload = None
        for driver_file in _driver_characteristics_fallback_paths(self.data_dir, self.season_year):
            if not driver_file.exists():
                continue
            with open(driver_file) as f:
                driver_payload = json.load(f)
            logger.info(
                "Loading driver characteristics from %s for season %s",
                driver_file.name,
                self.season_year,
            )
            break

        if driver_payload is not None:
            self.drivers = driver_payload.get("drivers", {})
        else:
            logger.warning("No driver characteristics found. Using an empty dictionary.")
            self.drivers = {}

        # 2. Load Car Data (Testing) OR Fallback to Derived Baseline
        if self.car_file.exists():
            logger.info("Loading testing data from %s", self.car_file.name)
            with open(self.car_file) as f:
                self.cars = json.load(f)["teams"]
        else:
            logger.warning(
                "No %s testing data found. Deriving car performance from historical driver pace.",
                self.season_year,
            )
            self.cars = self._derive_tiers_from_drivers()

    def create_priors(self) -> dict:
        """Synthesize priors."""
        self.load_data()
        priors = {}

        # Load Lineups
        from src.utils.lineups import load_current_lineups

        lineups = load_current_lineups()

        # Invert to Driver -> Team
        driver_to_team = {}
        for team, drivers in lineups.items():
            for driver in drivers:
                driver_to_team[driver] = team

        for driver_code, team_name in driver_to_team.items():
            # A. Car Performance (0-20 scale)
            car_perf = self._get_car_performance(team_name)

            # B. Driver Modifier (-2 to +2 scale)
            driver_stats = self.drivers.get(driver_code, {})
            # Use 'racecraft' score to adjust around the car's mean
            # 0.5 is average. 0.9 is Max. 0.2 is Sargeant.
            skill_score = driver_stats.get("racecraft", {}).get("skill_score", 0.5)
            experience = driver_stats.get("experience", {}).get("tier", "rookie")

            # Formula: Rating = Car_Base + (Driver_Skill_Delta)
            # This respects your "Car Limited" rule.
            # A great driver (+1.5) in a bad car (8.0) = 9.5 (Still midfield).
            modifier = (skill_score * 4) - 2
            mu = car_perf["base_rating"] + modifier

            # C. Uncertainty (Sigma)
            # 2026 New Regs = High Baseline Sigma
            sigma = 2.0
            if experience == "rookie":
                sigma += 1.5
            if car_perf.get("stability", 1.0) < 0.5:
                sigma += 1.0

            priors[driver_code] = DriverPrior(
                driver_number=str(driver_stats.get("number", 0)),
                driver_code=driver_code,
                team=team_name,
                team_tier=car_perf["tier"],
                mu=mu,
                sigma=sigma,
            )

        return priors

    def _get_car_performance(self, team_name):
        """Get car score from loaded data (Testing or Derived)."""
        # Fuzzy match team name (e.g. 'Red Bull Racing' vs 'RED BULL')
        norm_name = team_name.upper().replace(" ", "")

        # Try finding the team in our data source
        matched_key = next(
            (
                k
                for k in self.cars.keys()
                if k.upper().replace(" ", "") in norm_name
                or norm_name in k.upper().replace(" ", "")
            ),
            None,
        )

        if matched_key:
            team_data = self.cars[matched_key]

            if "base_rating" in team_data:
                # It's our Derived Format
                return team_data
            else:
                # It's the Real Testing format (metrics)
                cornering = team_data.get("medium_corner_performance", 0.5)
                top_speed = team_data.get("top_speed", 0.5)
                stability = team_data.get("consistency", 0.5)

                score = (cornering * 10) + (top_speed * 5) + (stability * 3)
                return {
                    "base_rating": score,
                    "tier": "top" if score > 15 else "midfield",
                    "stability": stability,
                }

        # New Team / No Data (e.g. Cadillac) -> Conservative Entry
        return {"base_rating": 8, "tier": "backmarker", "stability": 0.5}

    def _derive_tiers_from_drivers(self):
        """
        THE "CAR LIMITED" LOGIC:
        Reverse-engineer car performance by averaging the pace of its 2025 drivers.

        If McLaren had Lando (Fast) and Oscar (Fast), the car score is High.
        If Red Bull had Max (Fast) and Checo (Slow), the car score is Averaged (Lower).
        """
        team_pace_scores = defaultdict(list)

        # 1. Group 2025 Driver Pace by Team
        for _driver_code, stats in self.drivers.items():
            # Only use drivers with enough data
            if stats.get("pace", {}).get("confidence") == "low":
                continue

            # Get Quali Pace (Pure speed metric)
            # This is 0-1 normalized (1.0 = Pole Position avg)
            pace = stats["pace"]["quali_pace"]

            # We use the *last* team they drove for in 2025
            teams = stats.get("teams", [])
            if teams:
                team_pace_scores[teams[-1]].append(pace)

        derived_cars = {}

        # 2. Calculate Team Baselines
        for team, paces in team_pace_scores.items():
            if not paces:
                continue

            # Average pace of both drivers = True Car Performance
            avg_pace = np.mean(paces)

            # Map 0.0-1.0 pace to 0-20 rating scale
            # Top car (0.95+) -> ~17-18 rating
            # Backmarker (0.30) -> ~6-7 rating
            base_rating = 5 + (avg_pace * 13)

            # Determine Tier
            if base_rating > 14:
                tier = "top"
            elif base_rating > 9:
                tier = "midfield"
            else:
                tier = "backmarker"

            derived_cars[team] = {
                "base_rating": base_rating,
                "tier": tier,
                "stability": 0.8,  # Assume mature cars are stable by default
            }

        top_teams = sorted(derived_cars.items(), key=lambda x: x[1]["base_rating"], reverse=True)[
            :5
        ]
        team_strings = [f"{t}: {d['base_rating']:.1f}" for t, d in top_teams]
        logger.info("Derived baselines: %s ...", ", ".join(team_strings))

        return derived_cars
