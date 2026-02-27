"""Data-loading and team-strength mixin for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.persistence.artifact_store import ArtifactStore
from src.systems.weight_schedule import (
    SCHEDULES,
    ScheduleType,
    calculate_blended_performance,
    get_recommended_schedule,
)
from src.utils import config_loader
from src.utils.compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
)

logger = logging.getLogger("src.predictors.baseline_2026")


def _driver_characteristics_fallback_paths(data_dir: Path, year: int) -> tuple[Path, ...]:
    """Return season-aware driver-characteristics fallback candidates."""
    return (
        data_dir / "driver_characteristics" / f"{year}_driver_characteristics.json",
        data_dir / "driver_characteristics.json",
    )


class BaselineDataMixin:
    """Shared data and team-strength methods for Baseline2026Predictor."""

    def __init__(self):
        """Initialize data mixin with compound extraction cache."""
        if not hasattr(self, "_compound_cache"):
            self._compound_cache = {}

    def load_data(self) -> None:
        """Load season data and driver characteristics with schema validation."""
        target_year = int(getattr(self, "season_year", 2026))
        # Use injected artifact store or create new one
        store = getattr(self, "artifact_store", None) or ArtifactStore(data_root=self.data_dir)

        # Load and validate season car characteristics
        data = store.load_artifact(
            artifact_type="car_characteristics",
            artifact_key=f"{target_year}::car_characteristics",
        )

        if not data:
            # Fallback to file for backward compatibility
            logger.warning("Could not load car characteristics from DB, falling back to file")
            car_file = (
                self.data_dir / "car_characteristics" / f"{target_year}_car_characteristics.json"
            )
            with open(car_file) as f:
                data = json.load(f)

        # Validate team characteristics before using
        try:
            validate_team_characteristics(data)
        except ValueError as e:
            logger.error(f"Failed to load team characteristics: {e}")
            raise

        self.teams = data["teams"]

        # Check data freshness and warn if stale
        data_freshness = data.get("data_freshness", "UNKNOWN")
        races_completed = data.get("races_completed", 0)
        data.get("last_updated")

        if data_freshness == "BASELINE_PRESEASON":
            logger.warning(
                "Using pre-season baseline data; team performance remains uncertain until races complete."
            )
        elif data_freshness == "LIVE_UPDATED":
            logger.info(f"Using live-updated data from {races_completed} race(s)")
        else:
            logger.warning(
                f"Data freshness unknown ({data_freshness}); predictions may be outdated"
            )

        # Load and validate driver characteristics
        driver_data = store.load_artifact(
            artifact_type="driver_characteristics",
            artifact_key=f"{target_year}::driver_characteristics",
        )

        if not driver_data:
            # Fallback to file for backward compatibility
            logger.warning("Could not load driver characteristics from DB, falling back to file")
            driver_data = None
            for driver_file in _driver_characteristics_fallback_paths(self.data_dir, target_year):
                if not driver_file.exists():
                    continue
                with open(driver_file) as f:
                    driver_data = json.load(f)
                logger.info(
                    "Loaded driver characteristics fallback from %s for season %s",
                    driver_file,
                    target_year,
                )
                break
            if driver_data is None:
                raise FileNotFoundError(
                    f"Could not locate driver characteristics fallback for season {target_year} "
                    f"under {self.data_dir}"
                )

        # Validate driver characteristics before using
        try:
            validate_driver_characteristics(driver_data)
        except ValueError as e:
            logger.error(f"Failed to load driver characteristics: {e}")
            raise

        # ERROR DETECTION: Check for extraction bugs (does NOT correct)
        from src.utils.driver_validation import validate_driver_data

        errors = validate_driver_data(driver_data["drivers"])
        if errors:
            logger.warning(
                f"Driver data has {len(errors)} validation errors. "
                "Consider re-running extraction: python scripts/extract_driver_characteristics.py --years 2023,2024,2025"
            )

        self.drivers = driver_data["drivers"]

        # Load track characteristics for weight schedule system
        track_data = store.load_artifact(
            artifact_type="track_characteristics",
            artifact_key=f"{target_year}::track_characteristics",
        )

        if not track_data:
            # Fallback to file for backward compatibility
            logger.warning("Could not load track characteristics from DB, falling back to file")
            track_file = (
                self.data_dir
                / "track_characteristics"
                / f"{target_year}_track_characteristics.json"
            )
            try:
                with open(track_file) as f:
                    track_data = json.load(f)
            except FileNotFoundError:
                logger.warning("Track characteristics not found")
                track_data = {}

        self.tracks = track_data.get("tracks", {})
        if self.tracks:
            logger.info(f"Loaded track characteristics for {len(self.tracks)} circuits")

        # Store races completed and year for weight schedule (from car characteristics)
        self.races_completed = data.get("races_completed", 0)
        self.year = data.get("year", target_year)

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Calculate track-car suitability modifier (-0.1 to +0.1) based on car directionality vs track composition."""
        team_data = self.teams.get(team, {})
        directionality = team_data.get("directionality", {})

        # If no directionality data, return neutral
        if not directionality:
            return 0.0

        track_profile = self.tracks.get(race_name, {})

        # If track has no telemetry data, return neutral
        if "straights_pct" not in track_profile:
            return 0.0

        # Calculate weighted suitability based on track composition
        total_pct = (
            track_profile.get("straights_pct", 0)
            + track_profile.get("slow_corners_pct", 0)
            + track_profile.get("medium_corners_pct", 0)
            + track_profile.get("high_corners_pct", 0)
        )

        if total_pct == 0:
            return 0.0

        # Weighted combination of car strengths × track demands
        suitability = (
            directionality.get("max_speed", 0) * (track_profile.get("straights_pct", 0) / total_pct)
            + directionality.get("slow_corner_speed", 0)
            * (track_profile.get("slow_corners_pct", 0) / total_pct)
            + directionality.get("medium_corner_speed", 0)
            * (track_profile.get("medium_corners_pct", 0) / total_pct)
            + directionality.get("high_corner_speed", 0)
            * (track_profile.get("high_corners_pct", 0) / total_pct)
        )

        return suitability

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        """
        Calculate blended team strength using weight schedule system.

        Combines:
        1. Baseline (2025 standings) - decreases over season
        2. Testing directionality (track suitability) - decreases over season
        3. Current season (running average) - increases over season
        """
        team_data = self.teams.get(team, {})

        # 1. Baseline from 2025 standings
        baseline = team_data.get("overall_performance", 0.5)

        # 2. Testing/practice-informed track suitability signal (relative modifier)
        testing_modifier = self.calculate_track_suitability(team, race_name)
        testing_score = float(np.clip(baseline + testing_modifier, 0.0, 1.0))

        # 3. Current season running average
        current_season_performance = team_data.get("current_season_performance", [])
        if current_season_performance:
            current = np.mean(current_season_performance)
        else:
            # Pre-season: use baseline as fallback for current
            current = baseline

        # 4. Apply weight schedule
        race_number = self.races_completed + 1  # Next race

        # Use configured schedule when provided; default to regulation-change recommendation.
        cfg = getattr(self, "config", config_loader)
        configured_schedule = cfg.get("baseline_predictor.team_strength_schedule", None)
        schedule: ScheduleType
        if isinstance(configured_schedule, str) and configured_schedule in SCHEDULES:
            schedule = cast(ScheduleType, configured_schedule)
        else:
            schedule = get_recommended_schedule(is_regulation_change=True)

        blended = calculate_blended_performance(
            baseline_score=baseline,
            testing_modifier=testing_score,
            current_score=current,
            race_number=race_number,
            schedule=schedule,
        )

        return blended

    def _select_race_compound(self, race_name: str) -> str:
        """Select primary race compound based on track tire stress characteristics."""
        cfg = getattr(self, "config", config_loader)
        try:
            season_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
            candidate_years = [season_year]
            if season_year > 2020:
                candidate_years.append(season_year - 1)
            if 2025 not in candidate_years:
                candidate_years.append(2025)

            pirelli_file = next(
                (
                    Path("data") / f"{candidate_year}_pirelli_info.json"
                    for candidate_year in candidate_years
                ),
                None,
            )
            if pirelli_file is None:
                return "MEDIUM"  # Default fallback
            if not pirelli_file.exists():
                fallback_file = next(
                    (
                        Path("data") / f"{candidate_year}_pirelli_info.json"
                        for candidate_year in candidate_years[1:]
                        if (Path("data") / f"{candidate_year}_pirelli_info.json").exists()
                    ),
                    None,
                )
                if fallback_file is None:
                    return "MEDIUM"
                pirelli_file = fallback_file

            with open(pirelli_file) as f:
                pirelli_data = json.load(f)

            # Normalize race name to match keys (lowercase, underscores)
            race_key = race_name.lower().replace(" ", "_").replace("-", "_")
            track_info = pirelli_data.get(race_key, {})

            if not track_info or "tyre_stress" not in track_info:
                return "MEDIUM"

            tyre_stress = track_info["tyre_stress"]

            # Load thresholds from config
            high_threshold = cfg.get(
                "baseline_predictor.compound_selection.high_stress_threshold", 3.5
            )
            low_threshold = cfg.get(
                "baseline_predictor.compound_selection.low_stress_threshold", 2.5
            )
            default_stress = cfg.get(
                "baseline_predictor.compound_selection.default_stress_fallback", 3.0
            )

            # Calculate total tire stress score (higher = more demanding)
            stress_score = (
                tyre_stress.get("traction", default_stress)
                + tyre_stress.get("braking", default_stress)
                + tyre_stress.get("lateral", default_stress)
                + tyre_stress.get("asphalt_abrasion", default_stress)
            ) / 4.0

            # Apply thresholds from config
            if stress_score > high_threshold:
                return "HARD"
            elif stress_score < low_threshold:
                return "SOFT"
            else:
                return "MEDIUM"

        except Exception as e:
            logger.debug(f"Could not determine race compound for {race_name}: {e}")
            return "MEDIUM"

    def get_compound_adjusted_team_strength(
        self, team: str, race_name: str, compound: str = "MEDIUM"
    ) -> float:
        """Get team strength (0-1) adjusted for tire compound performance."""
        cfg = getattr(self, "config", config_loader)
        # Get base blended team strength
        base_strength = self.get_blended_team_strength(team, race_name)

        # Get compound characteristics
        team_data = self.teams.get(team, {})
        compound_chars = team_data.get("compound_characteristics", {})

        # Check if we have reliable compound data
        min_laps_threshold = cfg.get("baseline_predictor.race.min_laps_for_compound_data", 10)
        if not should_use_compound_adjustments(
            compound_chars, min_laps_threshold=min_laps_threshold
        ):
            return base_strength

        # Calculate compound modifier
        compound_modifier = get_compound_performance_modifier(compound_chars, compound)

        # Apply modifier and clip to valid range
        adjusted_strength = np.clip(base_strength + compound_modifier, 0.0, 1.0)

        logger.debug(
            f"  {team} on {compound}: base={base_strength:.3f} + "
            f"compound={compound_modifier:+.3f} = {adjusted_strength:.3f}"
        )

        return adjusted_strength

    def _get_testing_characteristics_for_profile(self, team: str, profile: str) -> dict[str, float]:
        """Get testing/practice characteristics for a profile with backward-compatible fallbacks."""
        team_data = self.teams.get(team, {})

        profile_store = team_data.get("testing_characteristics_profiles")
        if isinstance(profile_store, dict):
            profile_data = profile_store.get(profile)
            if isinstance(profile_data, dict):
                return profile_data

        fallback = team_data.get("testing_characteristics")
        if not isinstance(fallback, dict):
            return {}

        fallback_profile = fallback.get("run_profile")
        if fallback_profile == profile:
            return fallback

        # Older files may only store one profile in testing_characteristics.
        if profile == "balanced":
            return fallback

        return {}

    def _compute_testing_profile_modifier(
        self,
        team: str,
        profile: str,
        metric_weights: dict[str, float],
        scale: float,
    ) -> tuple[float, bool]:
        """
        Compute a small team-strength modifier from testing/practice characteristics.

        Returns (modifier, has_profile_data). Modifier is bounded to avoid overpowering
        the existing baseline + track-suitability + season-performance logic.
        """
        cfg = getattr(self, "config", config_loader)
        profile_metrics = self._get_testing_characteristics_for_profile(team, profile)
        if not profile_metrics:
            return 0.0, False

        weighted_sum = 0.0
        total_weight = 0.0
        for metric_name, weight in metric_weights.items():
            value = profile_metrics.get(metric_name)
            if value is None:
                continue
            centered = value - 0.5
            weighted_sum += centered * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0, False

        normalized_centered = weighted_sum / total_weight
        clip_range = cfg.get("baseline_predictor.race.testing_modifier_clip_range", [-0.04, 0.04])
        if isinstance(clip_range, list) and len(clip_range) == 2 and clip_range[0] < clip_range[1]:
            min_clip, max_clip = clip_range
        else:
            min_clip, max_clip = -0.04, 0.04
        modifier = np.clip(normalized_centered * scale, min_clip, max_clip)
        return modifier, True

    def _update_compound_characteristics_from_session(
        self,
        session_laps: pd.DataFrame,
        race_name: str,
        year: int,
        is_sprint: bool,
    ) -> None:
        """Extract compound characteristics with session-level caching."""
        cfg = getattr(self, "config", config_loader)
        # Check cache first (keyed by race + session lap count as freshness indicator)
        cache_key = (race_name, year, len(session_laps))
        if cache_key in self._compound_cache:
            logger.debug(
                f"Using cached compound metrics for {race_name} ({len(session_laps)} laps)"
            )
            cached_compounds = self._compound_cache[cache_key]
            for team_name, compounds in cached_compounds.items():
                if team_name in self.teams:
                    self.teams[team_name]["compound_characteristics"] = compounds
            return

        from src.systems.compound_analyzer import (
            aggregate_compound_samples,
            extract_compound_metrics,
            normalize_compound_metrics_across_teams,
        )
        from src.utils.team_mapping import map_team_to_characteristics

        logger.info(f"Extracting compound metrics from session for {race_name}...")

        # Extract compound metrics per team
        race_compound_metrics = {}
        known_teams = set(self.teams.keys())

        for raw_team in session_laps["Team"].unique():
            if pd.isna(raw_team):
                continue

            canonical_team = map_team_to_characteristics(str(raw_team), known_teams=known_teams)
            if not canonical_team:
                continue

            team_laps = session_laps[session_laps["Team"] == raw_team]
            compound_data = extract_compound_metrics(team_laps, canonical_team, race_name)

            if compound_data:
                race_compound_metrics[canonical_team] = compound_data

        # Normalize compound metrics across teams (track-specific)
        if race_compound_metrics:
            normalized_compound_metrics = normalize_compound_metrics_across_teams(
                race_compound_metrics, race_name
            )

            # Get blend weight from config based on session type
            # Practice sessions are exploratory (lower weight), sprint/race are competitive (higher weight)
            if is_sprint:
                blend_weight = cfg.get("baseline_predictor.compound_blend_weights.sprint", 0.50)
            else:
                blend_weight = cfg.get("baseline_predictor.compound_blend_weights.practice", 0.30)

            # Update in-memory team data with blended compound characteristics
            for team_name, new_compounds in normalized_compound_metrics.items():
                if team_name not in self.teams:
                    continue

                existing_compound_chars = self.teams[team_name].get("compound_characteristics", {})
                if not isinstance(existing_compound_chars, dict):
                    existing_compound_chars = {}

                # Blend with existing compound data
                blended_compounds = aggregate_compound_samples(
                    existing_compound_chars,
                    new_compounds,
                    blend_weight=blend_weight,
                    race_name=race_name,
                )

                self.teams[team_name]["compound_characteristics"] = blended_compounds

            # Cache the blended results
            self._compound_cache[cache_key] = {
                team: self.teams[team].get("compound_characteristics", {})
                for team in normalized_compound_metrics
                if team in self.teams
            }

            # Persist updated compound characteristics to DB if using artifact store
            store = getattr(self, "artifact_store", None)
            storage_mode = getattr(store, "storage_mode", "file_only") if store else "file_only"
            if store and storage_mode in {"db_only", "fallback", "dual_write"}:
                try:
                    season_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
                    artifact_key = f"{season_year}::car_characteristics"
                    car_data = store.load_artifact("car_characteristics", artifact_key)
                    if car_data:
                        for team_name in normalized_compound_metrics:
                            if team_name in car_data.get("teams", {}):
                                car_data["teams"][team_name]["compound_characteristics"] = (
                                    self.teams[team_name].get("compound_characteristics", {})
                                )
                        store.save_artifact("car_characteristics", artifact_key, car_data)
                        logger.debug(
                            "Persisted compound characteristics for "
                            f"{len(normalized_compound_metrics)} teams to DB"
                        )
                except Exception as e:
                    logger.warning(f"Failed to persist compound characteristics to DB: {e}")
            else:
                logger.debug("Skipping DB persistence (file-only mode or no artifact store)")

            logger.info(
                f"Updated and cached compound characteristics for {len(normalized_compound_metrics)} teams "
                f"(blend_weight={blend_weight:.0%})"
            )
        else:
            logger.debug("No compound metrics extracted from session")
