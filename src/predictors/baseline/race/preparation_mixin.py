"""Race preparation helpers for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.types.prediction_types import DriverRaceInfo, QualifyingGridEntry
from src.utils import config_loader
from src.utils.schema_validation import validate_track_characteristics

from .preparation_flow import (
    build_missing_driver_fallback,
    infer_missing_driver_experience_tier,
    prepare_driver_info_core,
    prepare_driver_info_with_compounds_core,
    resolve_effective_experience_tier_for_race,
)

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePreparationMixin:
    """Race preparation methods for Baseline2026Predictor."""

    def _resolve_effective_experience_tier_for_race(self, driver_data: dict) -> str:
        """Resolve experience tier for the current prediction year."""
        current_year = int(getattr(self, "year", 2026))
        return resolve_effective_experience_tier_for_race(
            driver_data=driver_data,
            current_year=current_year,
        )

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
        current_year = int(getattr(self, "year", 2026))
        return infer_missing_driver_experience_tier(
            driver_code=driver_code,
            current_year=current_year,
            load_driver_debut_years_fn=self._load_driver_debut_years,
        )

    def _build_missing_driver_fallback(self, driver_code: str, team: str) -> dict:
        """Build a synthetic profile for known active-lineup drivers missing characteristics."""
        return build_missing_driver_fallback(
            driver_code=driver_code,
            team=team,
            config=getattr(self, "config", config_loader),
            infer_missing_driver_experience_tier_fn=self._infer_missing_driver_experience_tier,
            get_teammate_driver_data_fn=self._get_teammate_driver_data,
            logger=logger,
        )

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
        return prepare_driver_info_core(
            qualifying_grid=qualifying_grid,
            race_name=race_name,
            race_compound=race_compound,
            teams=self.teams,
            config=getattr(self, "config", config_loader),
            get_compound_adjusted_team_strength_fn=self.get_compound_adjusted_team_strength,
            compute_testing_profile_modifier_fn=self._compute_testing_profile_modifier,
            get_driver_data_or_fallback_fn=self._get_driver_data_or_fallback,
            resolve_effective_experience_tier_for_race_fn=self._resolve_effective_experience_tier_for_race,
        )

    def _prepare_driver_info_with_compounds(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        race_name: str | None,
    ) -> tuple[dict[str, DriverRaceInfo], int]:
        """Build driver info map with per-compound team strengths for lap-by-lap simulation."""
        from src.utils.compound_performance import get_compound_performance_modifier

        return prepare_driver_info_with_compounds_core(
            qualifying_grid=qualifying_grid,
            race_name=race_name,
            teams=self.teams,
            config=getattr(self, "config", config_loader),
            get_blended_team_strength_fn=self.get_blended_team_strength,
            compute_testing_profile_modifier_fn=self._compute_testing_profile_modifier,
            get_driver_data_or_fallback_fn=self._get_driver_data_or_fallback,
            resolve_effective_experience_tier_for_race_fn=self._resolve_effective_experience_tier_for_race,
            get_compound_performance_modifier_fn=get_compound_performance_modifier,
        )
