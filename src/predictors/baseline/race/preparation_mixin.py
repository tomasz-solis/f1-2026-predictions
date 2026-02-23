"""Race preparation helpers for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.types.prediction_types import DriverRaceInfo, QualifyingGridEntry
from src.utils import config_loader
from src.utils.schema_validation import validate_track_characteristics

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePreparationMixin:
    """Race preparation methods for Baseline2026Predictor."""

    def _resolve_effective_experience_tier_for_race(self, driver_data: dict) -> str:
        """Resolve experience tier for the current prediction year."""
        experience = driver_data.get("experience", {}) if isinstance(driver_data, dict) else {}
        stored_tier = str(experience.get("tier", "unknown"))
        if stored_tier == "sophomore":
            stored_tier = "second_year"

        current_year = int(getattr(self, "year", 2026))
        stored_years = experience.get("years_of_experience", 0)
        debut_year = experience.get("debut_year")

        try:
            effective_years = int(stored_years)
        except (TypeError, ValueError):
            effective_years = None

        try:
            debut_year_int = int(debut_year) if debut_year is not None else None
        except (TypeError, ValueError):
            debut_year_int = None

        if debut_year_int is not None and current_year >= debut_year_int:
            computed_years = current_year - debut_year_int
            if effective_years is None:
                effective_years = computed_years
            else:
                effective_years = max(effective_years, computed_years)

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

    def _is_known_lineup_driver(self, driver_code: str, team: str) -> bool:
        """Return True if driver is in configured active lineups."""
        try:
            from src.utils.lineups import load_current_lineups

            current_lineups = load_current_lineups() or {}
        except (FileNotFoundError, OSError, ValueError, TypeError) as e:
            logger.warning(f"Could not load current lineups while checking {driver_code}: {e}")
            return False

        team_drivers = current_lineups.get(team, [])
        if driver_code in team_drivers:
            return True
        return any(driver_code in drivers for drivers in current_lineups.values())

    def _get_driver_data_or_fallback(self, driver_code: str, team: str) -> dict:
        """Return driver data, using defaults for known active-lineup drivers."""
        driver_data = self.drivers.get(driver_code)
        if driver_data:
            return driver_data

        if self._is_known_lineup_driver(driver_code, team):
            fallback = self._build_missing_driver_fallback(driver_code, team)
            self.drivers[driver_code] = fallback
            return fallback

        raise ValueError(f"Driver {driver_code} not found in loaded characteristics")

    def _get_teammate_driver_data(self, driver_code: str, team: str) -> tuple[str, dict] | None:
        """Return teammate data from configured current lineups when available."""
        try:
            from src.utils.lineups import load_current_lineups

            current_lineups = load_current_lineups() or {}
        except (FileNotFoundError, OSError, ValueError, TypeError) as e:
            logger.warning(
                f"Could not load current lineups while building fallback for {driver_code}: {e}"
            )
            return None

        team_drivers = current_lineups.get(team, [])
        for teammate_code in team_drivers:
            if teammate_code == driver_code:
                continue
            teammate_data = self.drivers.get(teammate_code)
            if teammate_data:
                return teammate_code, teammate_data
        return None

    def _load_driver_debut_years(self) -> dict[str, int]:
        """Load and cache driver debut years from artifact store, then CSV fallback."""
        cached = getattr(self, "_driver_debut_years_cache", None)
        if isinstance(cached, dict):
            return cached

        store = getattr(self, "artifact_store", None)
        if store is not None and hasattr(store, "load_artifact"):
            try:
                payload = store.load_artifact(
                    artifact_type="driver_debuts",
                    artifact_key="driver_debuts",
                )
            except Exception as e:
                logger.warning(f"Could not load driver debuts artifact: {e}")
                payload = None

            if isinstance(payload, dict):
                raw_debuts = payload.get("driver_debuts", payload)
                if isinstance(raw_debuts, dict):
                    debuts_from_store: dict[str, int] = {}
                    for code, year in raw_debuts.items():
                        try:
                            debuts_from_store[str(code)] = int(year)
                        except (TypeError, ValueError):
                            continue
                    if debuts_from_store:
                        logger.info(
                            f"Loaded {len(debuts_from_store)} driver debuts from artifact store"
                        )
                        self._driver_debut_years_cache = debuts_from_store
                        return debuts_from_store

        debut_csv = Path("data/driver_debuts.csv")
        if not debut_csv.exists():
            logger.warning(
                "Driver debuts CSV not found; missing drivers will be treated as rookies"
            )
            self._driver_debut_years_cache = {}
            return {}

        try:
            from src.features.driver_experience import load_driver_debuts_from_csv

            debut_years = load_driver_debuts_from_csv(debut_csv)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not load driver debuts CSV: {e}")
            debut_years = {}

        if debut_years:
            logger.info(f"Loaded {len(debut_years)} driver debuts from CSV fallback")

        self._driver_debut_years_cache = debut_years
        return debut_years

    def _infer_missing_driver_experience_tier(self, driver_code: str) -> str:
        """Infer tier for missing driver profiles from debut CSV and current prediction year."""
        debut_years = self._load_driver_debut_years()
        debut_year = debut_years.get(driver_code)
        if debut_year is None:
            return "rookie"

        current_year = int(getattr(self, "year", 2026))
        years_experience = max(0, current_year - int(debut_year))
        if years_experience == 0:
            return "rookie"
        if years_experience == 1:
            return "second_year"
        if years_experience <= 3:
            return "developing"
        if years_experience <= 6:
            return "established"
        if years_experience <= 14:
            return "veteran"
        return "sunset"

    def _build_missing_driver_fallback(self, driver_code: str, team: str) -> dict:
        """Build a synthetic profile for known active-lineup drivers missing characteristics."""
        cfg = getattr(self, "config", config_loader)
        default_skill = cfg.get("baseline_predictor.qualifying.default_skill", 0.5)
        inferred_tier = self._infer_missing_driver_experience_tier(driver_code)
        teammate_weight = cfg.get("baseline_predictor.race.missing_driver_teammate_weight", 0.75)
        teammate_weight = float(np.clip(teammate_weight, 0.0, 1.0))
        default_dnf = cfg.get("baseline_predictor.race.missing_driver_default_dnf_rate", 0.10)
        rookie_dnf_penalty = cfg.get(
            "baseline_predictor.race.missing_driver_rookie_dnf_penalty", 0.02
        )
        rookie_quali_penalty = cfg.get(
            "baseline_predictor.race.missing_driver_rookie_quali_penalty", 0.08
        )
        rookie_race_penalty = cfg.get(
            "baseline_predictor.race.missing_driver_rookie_race_penalty", 0.07
        )
        rookie_skill_penalty = cfg.get(
            "baseline_predictor.race.missing_driver_rookie_skill_penalty", 0.08
        )
        rookie_overtaking_penalty = cfg.get(
            "baseline_predictor.race.missing_driver_rookie_overtaking_penalty", 0.06
        )
        second_year_penalty_scale = float(
            cfg.get(
                "baseline_predictor.race.missing_driver_second_year_penalty_scale",
                cfg.get("baseline_predictor.race.missing_driver_sophomore_penalty_scale", 0.55),
            )
        )
        second_year_penalty_scale = float(np.clip(second_year_penalty_scale, 0.0, 1.0))

        if inferred_tier == "sophomore":
            inferred_tier = "second_year"

        teammate_entry = self._get_teammate_driver_data(driver_code, team)
        if teammate_entry:
            teammate_code, teammate_data = teammate_entry
            teammate_pace = teammate_data.get("pace", {})
            teammate_racecraft = teammate_data.get("racecraft", {})
            teammate_dnf = teammate_data.get("dnf_risk", {}).get("dnf_rate", default_dnf)

            # Regress toward neutral defaults so a missing profile isn't a hard clone of teammate.
            quali_pace = (
                teammate_weight * teammate_pace.get("quali_pace", 0.5)
                + (1.0 - teammate_weight) * 0.5
            )
            race_pace = (
                teammate_weight * teammate_pace.get("race_pace", 0.5)
                + (1.0 - teammate_weight) * 0.5
            )
            skill_score = (
                teammate_weight * teammate_racecraft.get("skill_score", default_skill)
                + (1.0 - teammate_weight) * default_skill
            )
            overtaking_skill = (
                teammate_weight * teammate_racecraft.get("overtaking_skill", default_skill)
                + (1.0 - teammate_weight) * default_skill
            )
            dnf_rate = teammate_dnf

            if inferred_tier == "rookie":
                # Keep rookies anchored below teammate baseline in model-only conditions.
                quali_pace -= rookie_quali_penalty
                race_pace -= rookie_race_penalty
                skill_score -= rookie_skill_penalty
                overtaking_skill -= rookie_overtaking_penalty
                dnf_rate += rookie_dnf_penalty
            elif inferred_tier == "second_year":
                # Year-2 drivers get a reduced penalty profile versus pure rookies.
                quali_pace -= rookie_quali_penalty * second_year_penalty_scale
                race_pace -= rookie_race_penalty * second_year_penalty_scale
                skill_score -= rookie_skill_penalty * second_year_penalty_scale
                overtaking_skill -= rookie_overtaking_penalty * second_year_penalty_scale
                dnf_rate += rookie_dnf_penalty * second_year_penalty_scale

            logger.info(
                f"Driver {driver_code} missing characteristics; using teammate-informed fallback from "
                f"{teammate_code} for {team} (tier={inferred_tier})"
            )
            return {
                "pace": {
                    "quali_pace": float(np.clip(quali_pace, 0.0, 1.0)),
                    "race_pace": float(np.clip(race_pace, 0.0, 1.0)),
                },
                "racecraft": {
                    "skill_score": float(np.clip(skill_score, 0.0, 1.0)),
                    "overtaking_skill": float(np.clip(overtaking_skill, 0.0, 1.0)),
                },
                "dnf_risk": {
                    "dnf_rate": float(np.clip(max(teammate_dnf, dnf_rate), 0.0, 0.35)),
                },
                "experience": {"tier": inferred_tier},
            }

        logger.warning(
            f"Driver {driver_code} missing characteristics; using neutral fallback for {team}"
        )
        neutral_pace = 0.5
        neutral_race = 0.5
        neutral_skill = default_skill
        neutral_overtaking = default_skill
        neutral_dnf = default_dnf
        if inferred_tier == "rookie":
            neutral_pace -= rookie_quali_penalty
            neutral_race -= rookie_race_penalty
            neutral_skill -= rookie_skill_penalty
            neutral_overtaking -= rookie_overtaking_penalty
            neutral_dnf += rookie_dnf_penalty
        elif inferred_tier == "second_year":
            neutral_pace -= rookie_quali_penalty * second_year_penalty_scale
            neutral_race -= rookie_race_penalty * second_year_penalty_scale
            neutral_skill -= rookie_skill_penalty * second_year_penalty_scale
            neutral_overtaking -= rookie_overtaking_penalty * second_year_penalty_scale
            neutral_dnf += rookie_dnf_penalty * second_year_penalty_scale
        return {
            "pace": {
                "quali_pace": float(np.clip(neutral_pace, 0.0, 1.0)),
                "race_pace": float(np.clip(neutral_race, 0.0, 1.0)),
            },
            "racecraft": {
                "skill_score": float(np.clip(neutral_skill, 0.0, 1.0)),
                "overtaking_skill": float(np.clip(neutral_overtaking, 0.0, 1.0)),
            },
            "dnf_risk": {"dnf_rate": float(np.clip(neutral_dnf, 0.0, 0.35))},
            "experience": {"tier": inferred_tier},
        }

    def _load_track_overtaking_difficulty(self, race_name: str | None) -> float:
        """Load track overtaking difficulty from characteristics file."""
        if not race_name:
            return 0.5

        try:
            track_file = Path(
                "data/processed/track_characteristics/2026_track_characteristics.json"
            )
            with open(track_file) as f:
                track_data = json.load(f)
                validate_track_characteristics(track_data)
                tracks = track_data["tracks"]
                return tracks.get(race_name, {}).get("overtaking_difficulty", 0.5)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not load track characteristics: {e}. Using default 0.5.")
            return 0.5

    def _prepare_driver_info(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        race_name: str | None,
        race_compound: str = "MEDIUM",
    ) -> tuple[dict[str, DriverRaceInfo], int]:
        """Build driver info map with team strength, profile modifiers, skills, and DNF probabilities."""
        cfg = getattr(self, "config", config_loader)
        driver_info_map: dict[str, DriverRaceInfo] = {}
        teams_with_long_profile = set()

        dnf_rate_historical_cap = cfg.get("baseline_predictor.race.dnf_rate_historical_cap", 0.20)
        dnf_rate_final_cap = cfg.get("baseline_predictor.race.dnf_rate_final_cap", 0.35)
        long_profile_scale = cfg.get(
            "baseline_predictor.race.testing_long_run_modifier_scale", 0.05
        )
        long_profile_weights = cfg.get(
            "baseline_predictor.race.testing_profile_weights.long_run",
            {
                "overall_pace": 0.50,
                "tire_deg_performance": 0.35,
                "consistency": 0.15,
            },
        )
        defensive_skill_weights = cfg.get(
            "baseline_predictor.race.defensive_skill_weights",
            {
                "overtaking_component": 0.65,
                "skill_component": 0.35,
            },
        )
        team_uncertainty_dnf_multiplier = cfg.get(
            "baseline_predictor.race.team_uncertainty_dnf_multiplier", 0.20
        )

        for entry in qualifying_grid:
            driver_code = entry["driver"]
            team = entry["team"]
            grid_pos = entry["position"]

            # Get base team strength with compound adjustment
            if race_name:
                team_strength = self.get_compound_adjusted_team_strength(
                    team, race_name, race_compound
                )
            else:
                team_strength = self.teams.get(team, {}).get("overall_performance", 0.50)

            # Add long-run testing profile modifier
            long_modifier, has_long_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="long_run",
                metric_weights=long_profile_weights,
                scale=long_profile_scale,
            )
            team_strength = np.clip(team_strength + long_modifier, 0.0, 1.0)
            if has_long_profile:
                teams_with_long_profile.add(team)

            driver_data = self._get_driver_data_or_fallback(driver_code, team)

            pace_data = driver_data.get("pace", {})
            quali_pace = pace_data.get("quali_pace", 0.5)
            race_pace = pace_data.get("race_pace", 0.5)
            race_advantage = race_pace - quali_pace

            racecraft = driver_data.get("racecraft", {})
            skill = racecraft.get("skill_score", 0.5)
            overtaking_skill = racecraft.get("overtaking_skill", 0.5)
            defensive_skill = racecraft.get("defensive_skill")
            if defensive_skill is None:
                # Backward-compatible fallback: reuse core racecraft signal when
                # explicit defensive skill is not present in extracted data.
                defensive_skill = (
                    defensive_skill_weights.get("overtaking_component", 0.65) * overtaking_skill
                    + defensive_skill_weights.get("skill_component", 0.35) * skill
                )
            defensive_skill = np.clip(defensive_skill, 0.0, 1.0)

            dnf_rate = min(
                driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10),
                dnf_rate_historical_cap,
            )

            experience_tier = self._resolve_effective_experience_tier_for_race(driver_data)
            experience_modifiers = {
                "rookie": 0.05,
                "second_year": 0.03,
                "developing": 0.02,
                "established": 0.00,
                "veteran": -0.01,
                "sunset": -0.005,
            }
            experience_dnf_modifier = experience_modifiers.get(experience_tier, 0.0)

            team_uncertainty = self.teams.get(team, {}).get("uncertainty", 0.30)
            if team_uncertainty >= 0.40:
                adjusted_dnf = (
                    dnf_rate
                    + experience_dnf_modifier
                    + (team_uncertainty * team_uncertainty_dnf_multiplier)
                )
            else:
                adjusted_dnf = dnf_rate + experience_dnf_modifier

            driver_info_map[driver_code] = {
                "driver": driver_code,
                "team": team,
                "grid_pos": grid_pos,
                "team_strength": team_strength,
                "skill": skill,
                "race_advantage": race_advantage,
                "overtaking_skill": overtaking_skill,
                "defensive_skill": defensive_skill,
                "dnf_probability": max(0.0, min(adjusted_dnf, dnf_rate_final_cap)),
            }

        return driver_info_map, len(teams_with_long_profile)

    def _prepare_driver_info_with_compounds(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        race_name: str | None,
    ) -> tuple[dict[str, DriverRaceInfo], int]:
        """Build driver info map with per-compound team strengths for lap-by-lap simulation."""
        cfg = getattr(self, "config", config_loader)
        from src.utils.compound_performance import get_compound_performance_modifier

        driver_info_map: dict[str, DriverRaceInfo] = {}
        teams_with_long_profile = set()

        dnf_rate_historical_cap = cfg.get("baseline_predictor.race.dnf_rate_historical_cap", 0.20)
        dnf_rate_final_cap = cfg.get("baseline_predictor.race.dnf_rate_final_cap", 0.35)
        long_profile_scale = cfg.get(
            "baseline_predictor.race.testing_long_run_modifier_scale", 0.05
        )
        long_profile_weights = cfg.get(
            "baseline_predictor.race.testing_profile_weights.long_run",
            {
                "overall_pace": 0.50,
                "tire_deg_performance": 0.35,
                "consistency": 0.15,
            },
        )
        default_tire_deg_slope = cfg.get(
            "baseline_predictor.race.tire_physics.default_deg_slope", 0.15
        )
        defensive_skill_weights = cfg.get(
            "baseline_predictor.race.defensive_skill_weights",
            {
                "overtaking_component": 0.65,
                "skill_component": 0.35,
            },
        )
        team_uncertainty_dnf_multiplier = cfg.get(
            "baseline_predictor.race.team_uncertainty_dnf_multiplier", 0.20
        )

        for entry in qualifying_grid:
            driver_code = entry["driver"]
            team = entry["team"]
            grid_pos = entry["position"]

            # Use weekend-aware team strength so race simulation starts from the same
            # blended baseline logic as qualifying.
            if race_name:
                base_team_strength = self.get_blended_team_strength(team, race_name)
            else:
                base_team_strength = self.teams.get(team, {}).get("overall_performance", 0.50)

            # Add long-run testing profile modifier
            long_modifier, has_long_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="long_run",
                metric_weights=long_profile_weights,
                scale=long_profile_scale,
            )
            base_team_strength = np.clip(base_team_strength + long_modifier, 0.0, 1.0)
            if has_long_profile:
                teams_with_long_profile.add(team)

            # Pre-compute per-compound team strengths
            team_compound_chars = self.teams.get(team, {}).get("compound_characteristics", {})

            team_strength_by_compound = {}
            tire_deg_by_compound = {}

            for compound in ["SOFT", "MEDIUM", "HARD"]:
                if compound in team_compound_chars:
                    # Compound-specific modifier
                    modifier = get_compound_performance_modifier(team_compound_chars, compound)
                    adjusted_strength = base_team_strength + modifier

                    # Tire degradation slope
                    tire_deg_slope = team_compound_chars[compound].get(
                        "tire_deg_slope", default_tire_deg_slope
                    )
                else:
                    # Fallback: no compound data available
                    adjusted_strength = base_team_strength
                    tire_deg_slope = default_tire_deg_slope

                team_strength_by_compound[compound] = np.clip(adjusted_strength, 0.0, 1.0)
                tire_deg_by_compound[compound] = tire_deg_slope

            # Get driver characteristics
            driver_data = self._get_driver_data_or_fallback(driver_code, team)

            pace_data = driver_data.get("pace", {})
            quali_pace = pace_data.get("quali_pace", 0.5)
            race_pace = pace_data.get("race_pace", 0.5)
            race_advantage = race_pace - quali_pace

            racecraft = driver_data.get("racecraft", {})
            skill = racecraft.get("skill_score", 0.5)
            overtaking_skill = racecraft.get("overtaking_skill", 0.5)
            defensive_skill = racecraft.get("defensive_skill")
            if defensive_skill is None:
                defensive_skill = (
                    defensive_skill_weights.get("overtaking_component", 0.65) * overtaking_skill
                    + defensive_skill_weights.get("skill_component", 0.35) * skill
                )
            defensive_skill = np.clip(defensive_skill, 0.0, 1.0)

            # DNF probability
            dnf_rate = min(
                driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10),
                dnf_rate_historical_cap,
            )

            experience_tier = self._resolve_effective_experience_tier_for_race(driver_data)
            experience_modifiers = {
                "rookie": 0.05,
                "second_year": 0.03,
                "developing": 0.02,
                "established": 0.00,
                "veteran": -0.01,
                "sunset": -0.005,
            }
            experience_dnf_modifier = experience_modifiers.get(experience_tier, 0.0)

            team_uncertainty = self.teams.get(team, {}).get("uncertainty", 0.30)
            if team_uncertainty >= 0.40:
                adjusted_dnf = (
                    dnf_rate
                    + experience_dnf_modifier
                    + (team_uncertainty * team_uncertainty_dnf_multiplier)
                )
            else:
                adjusted_dnf = dnf_rate + experience_dnf_modifier

            driver_info_map[driver_code] = {
                "driver": driver_code,
                "team": team,
                "grid_pos": grid_pos,
                "team_strength": base_team_strength,  # Base strength
                "team_strength_by_compound": team_strength_by_compound,  # Per-compound
                "tire_deg_by_compound": tire_deg_by_compound,  # Per-compound deg slopes
                "skill": skill,
                "race_advantage": race_advantage,
                "overtaking_skill": overtaking_skill,
                "defensive_skill": defensive_skill,
                "dnf_probability": max(0.0, min(adjusted_dnf, dnf_rate_final_cap)),
            }

        return driver_info_map, len(teams_with_long_profile)
