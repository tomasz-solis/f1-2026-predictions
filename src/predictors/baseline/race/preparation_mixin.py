"""Race preparation helpers for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.persistence.artifact_store import ArtifactStore
from src.types.prediction_types import DriverRaceInfo, QualifyingGridEntry
from src.utils import config_loader
from src.utils.schema_validation import validate_track_characteristics

from .preparation_flow import (
    _blend_race_skill_with_bayesian_form,
    build_missing_driver_fallback,
    infer_missing_driver_experience_tier,
    prepare_driver_info_core,
    prepare_driver_info_with_compounds_core,
    resolve_effective_experience_tier_for_race,
)

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePreparationMixin:
    """Race preparation methods for Baseline2026Predictor."""

    if TYPE_CHECKING:
        artifact_store: ArtifactStore | None
        config: Any
        drivers: dict[str, dict[str, Any]]
        season_year: int
        teams: dict[str, dict[str, Any]]
        year: int

        def _compute_testing_profile_modifier(
            self,
            team: str,
            profile: str,
            metric_weights: dict[str, float],
            scale: float,
        ) -> tuple[float, bool]: ...

        def get_blended_team_strength(self, team: str, race_name: str) -> float: ...

        def get_compound_adjusted_team_strength(
            self,
            team: str,
            race_name: str,
            compound: str = "MEDIUM",
        ) -> float: ...

    def _load_current_lineups_for_preparation(self) -> dict[str, list[str]]:
        """Load and cache current lineups for custom-grid context checks."""
        cached = getattr(self, "_current_lineups_cache", None)
        if isinstance(cached, dict):
            return cached

        try:
            from src.utils.lineups import load_current_lineups

            current_lineups = load_current_lineups() or {}
        except (FileNotFoundError, OSError, ValueError, TypeError) as e:
            logger.warning("Could not load current lineups for race preparation: %s", e)
            current_lineups = {}

        self._current_lineups_cache = current_lineups
        return current_lineups

    def _resolve_current_lineup_team(self, driver_code: str) -> str | None:
        """Return the active-lineup team for a driver when it is known."""
        current_lineups = self._load_current_lineups_for_preparation()
        for team, drivers in current_lineups.items():
            if driver_code in drivers:
                return str(team)
        return None

    def _build_portable_skill_signal(self, driver_code: str, base_skill: float) -> float:
        """Build a portable driver-skill signal for hypothetical team swaps."""
        driver_data = self.drivers.get(driver_code, {})
        if not isinstance(driver_data, dict):
            return float(base_skill)

        experience_tier = str(driver_data.get("experience", {}).get("tier", "")).strip().lower()
        if experience_tier not in {"established", "veteran", "sunset"}:
            return float(base_skill)

        cfg = getattr(self, "config", None) or config_loader
        configured_grid_size = int(cfg.get("grid.size", len(self.drivers) or 22))
        return _blend_race_skill_with_bayesian_form(
            driver_data=driver_data,
            base_skill=float(base_skill),
            races_completed=int(getattr(self, "races_completed", 0)),
            grid_size=max(configured_grid_size, len(self.drivers) or 0, 2),
            config=cfg,
        )

    def _annotate_driver_assignment_context(
        self,
        driver_info_map: dict[str, DriverRaceInfo],
    ) -> dict[str, DriverRaceInfo]:
        """Annotate race driver info with lineup-mismatch context for custom grids."""
        for driver_code, info in driver_info_map.items():
            assigned_team = str(info.get("team", "")).strip()
            current_lineup_team = self._resolve_current_lineup_team(driver_code)
            is_hypothetical_team_assignment = bool(
                assigned_team and current_lineup_team and assigned_team != current_lineup_team
            )
            info["current_lineup_team"] = current_lineup_team or assigned_team
            info["portable_skill"] = self._build_portable_skill_signal(
                driver_code,
                float(info.get("skill", 0.5)),
            )
            info["is_hypothetical_team_assignment"] = is_hypothetical_team_assignment
        return driver_info_map

    def _resolve_effective_experience_tier_for_race(self, driver_data: dict) -> str:
        """Resolve experience tier for the current prediction year."""
        current_year = int(getattr(self, "year", 2026))
        return resolve_effective_experience_tier_for_race(
            driver_data=driver_data,
            current_year=current_year,
        )

    def _is_known_lineup_driver(self, driver_code: str, team: str) -> bool:
        """Return True if driver is in configured active lineups."""
        current_lineups = self._load_current_lineups_for_preparation()

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
        current_lineups = self._load_current_lineups_for_preparation()

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
                logger.warning("Could not load driver debuts artifact: %s", e)
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
                            "Loaded %s driver debuts from artifact store", len(debuts_from_store)
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
            logger.warning("Could not load driver debuts CSV: %s", e)
            debut_years = {}

        if debut_years:
            logger.info("Loaded %s driver debuts from CSV fallback", len(debut_years))

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
            season_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
            track_file = Path(
                f"data/processed/track_characteristics/{season_year}_track_characteristics.json"
            )
            with open(track_file) as f:
                track_data = json.load(f)
                validate_track_characteristics(track_data, expected_year=season_year)
                tracks = track_data["tracks"]
                return tracks.get(race_name, {}).get("overtaking_difficulty", 0.5)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Could not load track characteristics: %s. Using default 0.5.", e)
            return 0.5

    def _prepare_driver_info(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        race_name: str | None,
        race_compound: str = "MEDIUM",
    ) -> tuple[dict[str, DriverRaceInfo], int]:
        """Build driver info map with team strength, profile modifiers, skills, and DNF probabilities."""
        driver_info_map, profile_count = prepare_driver_info_core(
            qualifying_grid=qualifying_grid,
            race_name=race_name,
            race_compound=race_compound,
            races_completed=int(getattr(self, "races_completed", 0)),
            teams=self.teams,
            config=getattr(self, "config", config_loader),
            get_compound_adjusted_team_strength_fn=self.get_compound_adjusted_team_strength,
            compute_testing_profile_modifier_fn=self._compute_testing_profile_modifier,
            get_driver_data_or_fallback_fn=self._get_driver_data_or_fallback,
            resolve_effective_experience_tier_for_race_fn=self._resolve_effective_experience_tier_for_race,
        )
        return self._annotate_driver_assignment_context(driver_info_map), profile_count

    def _prepare_driver_info_with_compounds(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        race_name: str | None,
    ) -> tuple[dict[str, DriverRaceInfo], int]:
        """Build driver info map with per-compound team strengths for lap-by-lap simulation."""
        from src.data.compound_performance import get_compound_performance_modifier

        driver_info_map, profile_count = prepare_driver_info_with_compounds_core(
            qualifying_grid=qualifying_grid,
            race_name=race_name,
            races_completed=int(getattr(self, "races_completed", 0)),
            teams=self.teams,
            config=getattr(self, "config", config_loader),
            get_blended_team_strength_fn=self.get_blended_team_strength,
            compute_testing_profile_modifier_fn=self._compute_testing_profile_modifier,
            get_driver_data_or_fallback_fn=self._get_driver_data_or_fallback,
            resolve_effective_experience_tier_for_race_fn=self._resolve_effective_experience_tier_for_race,
            get_compound_performance_modifier_fn=get_compound_performance_modifier,
        )
        return self._annotate_driver_assignment_context(driver_info_map), profile_count
