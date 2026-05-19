"""Build Bayesian driver priors from stored car and driver signals."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.models.bayesian import DriverPrior
from src.models.regulations import apply_2026_regulations

logger = logging.getLogger(__name__)


def _driver_characteristics_fallback_paths(data_dir: Path, season_year: int) -> tuple[Path, ...]:
    """Return season-aware driver-characteristics fallbacks in preferred order."""
    return (
        data_dir / "driver_characteristics" / f"{int(season_year)}_driver_characteristics.json",
        data_dir / "driver_characteristics.json",
    )


def _experience_tier_for_season(driver_stats: dict, season_year: int) -> str:
    """Resolve a driver's experience tier for the requested season."""
    experience = driver_stats.get("experience", {}) if isinstance(driver_stats, dict) else {}
    stored_tier = str(experience.get("tier", "rookie") or "rookie").strip().lower()
    if stored_tier == "sophomore":
        stored_tier = "second_year"

    debut_year = experience.get("debut_year")
    stored_years = experience.get("years_of_experience")
    effective_years: int | None = None
    try:
        effective_years = int(stored_years)
    except (TypeError, ValueError):
        effective_years = None

    try:
        debut_year_int = int(debut_year) if debut_year is not None else None
    except (TypeError, ValueError):
        debut_year_int = None

    if debut_year_int is not None and int(season_year) >= debut_year_int:
        computed_years = int(season_year) - debut_year_int
        effective_years = (
            computed_years
            if effective_years is None
            else max(
                effective_years,
                computed_years,
            )
        )

    if effective_years is None:
        return stored_tier
    if effective_years <= 0:
        return "rookie"
    if effective_years == 1:
        return "second_year"
    if effective_years <= 3:
        return "developing"
    if effective_years <= 6:
        return "established"
    if effective_years <= 14:
        return "veteran"
    return "sunset"


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
        """Synthesize priors and apply the 2026 regulation-reset adjustment."""
        self.load_data()
        priors = {}

        from src.utils.lineups import load_current_lineups

        lineups = load_current_lineups()
        if lineups is None:
            raise ValueError("Current lineups could not be loaded")

        driver_to_team = {}
        for team, drivers in lineups.items():
            for driver in drivers:
                driver_to_team[driver] = team

        for driver_code, team_name in driver_to_team.items():
            car_perf = self._get_car_performance(team_name)
            driver_stats = self.drivers.get(driver_code, {})
            skill_score = driver_stats.get("racecraft", {}).get("skill_score", 0.5)
            experience = _experience_tier_for_season(driver_stats, self.season_year)

            # Driver skill nudges the team baseline without overpowering it.
            modifier = (skill_score * 4) - 2
            mu = car_perf["base_rating"] + modifier

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

        return apply_2026_regulations(priors)

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
        """Infer a preseason car baseline from the recent pace of each team's drivers."""
        team_pace_scores = defaultdict(list)

        for _driver_code, stats in self.drivers.items():
            if stats.get("pace", {}).get("confidence") == "low":
                continue

            pace = stats["pace"]["quali_pace"]
            teams = stats.get("teams", [])
            if teams:
                team_pace_scores[teams[-1]].append(pace)

        derived_cars = {}

        for team, paces in team_pace_scores.items():
            if not paces:
                continue

            avg_pace = np.mean(paces)
            base_rating = 5 + (avg_pace * 13)
            if base_rating > 14:
                tier = "top"
            elif base_rating > 9:
                tier = "midfield"
            else:
                tier = "backmarker"

            derived_cars[team] = {
                "base_rating": base_rating,
                "tier": tier,
                "stability": 0.8,
            }

        top_teams = sorted(derived_cars.items(), key=lambda x: x[1]["base_rating"], reverse=True)[
            :5
        ]
        team_strings = [f"{t}: {d['base_rating']:.1f}" for t, d in top_teams]
        logger.info("Derived baselines: %s ...", ", ".join(team_strings))

        return derived_cars
