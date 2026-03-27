"""Data-loading and team-strength mixin for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.data.compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline.compound_updates import update_compound_characteristics_from_session
from src.predictors.baseline.data_support import (
    canonicalize_team_payload_keys,
    coerce_non_negative_int,
    driver_characteristics_fallback_paths,
    extract_target_actual_rows,
    sanitize_performance_observations,
    score_teams_from_actual_rows,
)
from src.predictors.baseline.team_strength import (
    calculate_track_suitability as calculate_track_suitability_helper,
)
from src.predictors.baseline.team_strength import (
    get_blended_team_strength as get_blended_team_strength_helper,
)
from src.predictors.baseline.team_strength import (
    get_compound_adjusted_team_strength as get_compound_adjusted_team_strength_helper,
)
from src.predictors.baseline.team_strength import (
    resolve_team_data as resolve_team_data_helper,
)
from src.predictors.baseline.team_strength import (
    select_race_compound,
)
from src.predictors.baseline.testing_profiles import (
    compute_testing_profile_modifier,
    get_testing_characteristics_for_profile,
)
from src.systems.weight_schedule import (
    SCHEDULES,
    calculate_blended_performance,
    get_recommended_schedule,
)
from src.utils import config_loader
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
    validate_track_characteristics,
)

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineDataMixin:
    """Shared data and team-strength methods for Baseline2026Predictor."""

    if TYPE_CHECKING:
        artifact_store: ArtifactStore | None
        car_characteristics_snapshot: dict[str, Any]
        config: Any
        data_dir: Path
        drivers: dict[str, dict[str, Any]]
        season_year: int
        teams: dict[str, dict[str, Any]]
        tracks: dict[str, dict[str, Any]]
        year: int

    def __init__(self):
        """Initialize data mixin with compound extraction cache."""
        if not hasattr(self, "_compound_cache"):
            self._compound_cache = {}

    def _resolve_predictions_data_root(self) -> Path:
        """Return the data root used when reading saved prediction artifacts."""
        store = getattr(self, "artifact_store", None)
        store_root = getattr(store, "data_root", None)
        if isinstance(store_root, Path):
            return store_root
        if isinstance(store_root, str) and store_root:
            return Path(store_root)
        return self.data_dir.parent if self.data_dir.name == "processed" else self.data_dir

    def _get_race_order_map(self, target_year: int) -> dict[str, int]:
        """Return season race order for contextual current-form cutoffs."""
        cache = getattr(self, "_race_order_map_cache", {})
        if target_year in cache:
            return cache[target_year]

        try:
            from src.utils.weekend import get_schedule_rows

            schedule_rows = get_schedule_rows(target_year)
        except Exception as exc:
            logger.debug("Could not load schedule rows for %s: %s", target_year, exc)
            schedule_rows = ()

        race_order_map: dict[str, int] = {}
        race_index = 0
        for raw_race_name, raw_event_format in schedule_rows:
            race_name = str(raw_race_name).strip()
            event_format = str(raw_event_format).strip().lower()
            if not race_name:
                continue
            if "testing" in race_name.lower() or "testing" in event_format:
                continue
            race_index += 1
            race_order_map.setdefault(race_name, race_index)

        cache[target_year] = race_order_map
        self._race_order_map_cache = cache
        return race_order_map

    def _prediction_race_sort_key(
        self,
        prediction: dict[str, object],
        *,
        race_order_map: dict[str, int],
    ) -> tuple[int, str, str]:
        """Build a stable race ordering key for saved predictions."""
        metadata = prediction.get("metadata", {})
        if not isinstance(metadata, dict):
            return (10_000, "", "")

        race_name = str(metadata.get("race_name", "")).strip()
        predicted_at = str(metadata.get("predicted_at", "")).strip()
        return (race_order_map.get(race_name, 10_000), predicted_at, race_name)

    def _blend_saved_actual_team_scores(
        self,
        *,
        qualifying_rows: list[dict[str, object]],
        race_rows: list[dict[str, object]],
        known_teams: set[str],
    ) -> dict[str, float]:
        """Blend saved qualifying and race actuals into one team-form snapshot."""
        qualifying_scores = (
            score_teams_from_actual_rows(qualifying_rows, known_teams=known_teams)
            if qualifying_rows
            else {}
        )
        race_scores = (
            score_teams_from_actual_rows(race_rows, known_teams=known_teams) if race_rows else {}
        )
        if race_scores and not qualifying_scores:
            return race_scores
        if qualifying_scores and not race_scores:
            return qualifying_scores
        if not qualifying_scores and not race_scores:
            return {}

        cfg = getattr(self, "config", config_loader)
        race_weight = float(
            cfg.get("baseline_predictor.current_season_form.saved_actual_race_weight", 0.70)
        )
        race_weight = float(np.clip(race_weight, 0.0, 1.0))
        qualifying_weight = 1.0 - race_weight

        blended_scores: dict[str, float] = {}
        for team_name in set(qualifying_scores) | set(race_scores):
            qualifying_score = qualifying_scores.get(team_name)
            race_score = race_scores.get(team_name)
            if qualifying_score is None and race_score is None:
                continue
            if qualifying_score is None:
                if race_score is None:
                    continue
                blended_scores[team_name] = race_score
                continue
            if race_score is None:
                blended_scores[team_name] = qualifying_score
                continue
            blended_scores[team_name] = float(
                np.clip(
                    (qualifying_score * qualifying_weight) + (race_score * race_weight),
                    0.0,
                    1.0,
                )
            )

        return blended_scores

    def _load_saved_actual_race_scores(
        self,
        target_year: int,
    ) -> list[dict[str, object]]:
        """Load one team-score snapshot per completed race from saved actuals."""
        cache = getattr(self, "_saved_actual_race_scores_cache", {})
        if target_year in cache:
            cached_records = cache[target_year]
            return cached_records if isinstance(cached_records, list) else []

        cfg = getattr(self, "config", config_loader)
        if not bool(
            cfg.get("baseline_predictor.current_season_form.infer_from_saved_actuals", True)
        ):
            return []

        try:
            from src.utils.prediction_logger import PredictionLogger
        except Exception as exc:
            logger.debug("Could not import PredictionLogger for current-form inference: %s", exc)
            return []

        predictions_dir = self._resolve_predictions_data_root() / "predictions"
        try:
            prediction_logger = PredictionLogger(predictions_dir=str(predictions_dir))
            predictions = prediction_logger.get_all_predictions(target_year)
        except Exception as exc:
            logger.debug("Could not load saved predictions for %s: %s", target_year, exc)
            return []

        if not predictions:
            return []

        known_teams = set(self.teams.keys())
        race_order_map = self._get_race_order_map(target_year)
        seen_races: set[str] = set()
        race_records: list[dict[str, object]] = []

        for prediction in sorted(
            predictions,
            key=lambda payload: self._prediction_race_sort_key(
                payload,
                race_order_map=race_order_map,
            ),
        ):
            metadata = prediction.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            race_name = str(metadata.get("race_name", "")).strip()
            if not race_name or race_name in seen_races:
                continue

            qualifying_rows = extract_target_actual_rows(
                prediction,
                target_key="main_qualifying",
            )
            race_rows = extract_target_actual_rows(
                prediction,
                target_key="grand_prix_race",
            )
            team_scores = self._blend_saved_actual_team_scores(
                qualifying_rows=qualifying_rows,
                race_rows=race_rows,
                known_teams=known_teams,
            )
            if not team_scores:
                continue

            seen_races.add(race_name)
            race_records.append(
                {
                    "race_name": race_name,
                    "team_scores": team_scores,
                }
            )

        cache[target_year] = race_records
        self._saved_actual_race_scores_cache = cache
        return race_records

    def _resolve_saved_actual_races_completed(
        self,
        *,
        target_year: int,
        data_freshness: str,
        races_completed: int,
    ) -> int:
        """Resolve available completed-race count from live state or saved actuals."""
        has_live_current_form = any(
            sanitize_performance_observations(team_data.get("current_season_performance"))
            for team_data in self.teams.values()
        )
        if data_freshness == "LIVE_UPDATED" or has_live_current_form:
            return races_completed

        race_records = self._load_saved_actual_race_scores(target_year)
        inferred_races = len(race_records)
        if inferred_races <= 0:
            return races_completed

        teams_with_scores = set()
        for record in race_records:
            team_scores = record.get("team_scores", {})
            if not isinstance(team_scores, dict):
                continue
            for team_name in team_scores:
                if team_name in self.teams:
                    teams_with_scores.add(str(team_name))

        logger.info(
            "Recovered current-season team form from saved actuals: %s race(s), %s team(s)",
            inferred_races,
            len(teams_with_scores),
        )
        return max(races_completed, inferred_races)

    def _count_known_prior_races(self, target_year: int, race_name: str | None) -> int | None:
        """Return number of scheduled races before the target race when known."""
        normalized_race_name = str(race_name or "").strip()
        if not normalized_race_name:
            return None

        race_order_map = self._get_race_order_map(target_year)
        target_order = race_order_map.get(normalized_race_name)
        if target_order is None:
            return None
        return max(target_order - 1, 0)

    def _race_precedes_target(
        self,
        *,
        target_year: int,
        candidate_race_name: str,
        target_race_name: str | None,
    ) -> bool:
        """Return True when one race is known to be earlier than the target race."""
        normalized_target = str(target_race_name or "").strip()
        normalized_candidate = str(candidate_race_name).strip()
        if not normalized_target:
            return True
        if normalized_candidate == normalized_target:
            return False

        race_order_map = self._get_race_order_map(target_year)
        target_order = race_order_map.get(normalized_target)
        candidate_order = race_order_map.get(normalized_candidate)
        if target_order is None:
            return normalized_candidate != normalized_target
        if candidate_order is None:
            return False
        return candidate_order < target_order

    def _get_saved_actual_observations(
        self,
        *,
        team_name: str,
        target_year: int,
        race_name: str | None,
    ) -> list[float]:
        """Return saved-actual observations for one team before the target race."""
        observations: list[float] = []
        for race_record in self._load_saved_actual_race_scores(target_year):
            candidate_race_name = str(race_record.get("race_name", "")).strip()
            if not self._race_precedes_target(
                target_year=target_year,
                candidate_race_name=candidate_race_name,
                target_race_name=race_name,
            ):
                continue
            team_scores = race_record.get("team_scores", {})
            if not isinstance(team_scores, dict):
                continue
            score = team_scores.get(team_name)
            if score is None:
                continue
            observations.append(float(np.clip(float(score), 0.0, 1.0)))
        return observations

    def _get_current_season_observations(
        self,
        *,
        team_name: str,
        team_data: dict[str, object],
        race_name: str | None,
    ) -> list[float]:
        """Return race-context-aware current-season observations for one team."""
        target_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
        live_observations = sanitize_performance_observations(
            team_data.get("current_season_performance")
        )
        prior_race_limit = coerce_non_negative_int(
            self._count_known_prior_races(target_year, race_name)
        )
        saved_actual_observations = self._get_saved_actual_observations(
            team_name=team_name,
            target_year=target_year,
            race_name=race_name,
        )
        if prior_race_limit is not None:
            saved_actual_observations = saved_actual_observations[:prior_race_limit]

        if live_observations:
            if prior_race_limit is None:
                if len(saved_actual_observations) > len(live_observations):
                    return saved_actual_observations
                return live_observations
            limited_live_observations = live_observations[:prior_race_limit]
            if len(saved_actual_observations) > len(limited_live_observations):
                return saved_actual_observations
            return limited_live_observations

        return saved_actual_observations

    def _get_contextual_races_completed(self, race_name: str | None) -> int:
        """Return completed-race count capped to what the target race could have known."""
        target_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
        available_races = max(
            coerce_non_negative_int(getattr(self, "races_completed", 0)) or 0,
            len(self._load_saved_actual_race_scores(target_year)),
        )
        prior_race_limit = self._count_known_prior_races(target_year, race_name)
        if prior_race_limit is None:
            return available_races
        return min(available_races, prior_race_limit)

    def _get_current_season_score(
        self,
        team_name: str,
        team_data: dict[str, object],
        *,
        fallback: float,
        race_name: str | None,
    ) -> float:
        """Return the current-season score using a recency-weighted average."""
        observations = self._get_current_season_observations(
            team_name=team_name,
            team_data=team_data,
            race_name=race_name,
        )
        if not observations:
            return float(fallback)

        cfg = getattr(self, "config", config_loader)
        recency_exponent = float(
            cfg.get("baseline_predictor.current_season_form.recency_exponent", 1.5)
        )
        recency_exponent = max(0.0, recency_exponent)
        if len(observations) == 1 or recency_exponent == 0.0:
            weighted_score = float(np.mean(observations))
        else:
            weights = np.power(np.arange(1, len(observations) + 1, dtype=float), recency_exponent)
            weighted_score = float(np.average(observations, weights=weights))

        stabilization_strength = float(
            cfg.get("baseline_predictor.current_season_form.stabilization_strength", 1.5)
        )
        stabilization_strength = max(0.0, stabilization_strength)
        if stabilization_strength == 0.0:
            return weighted_score

        observation_weight = len(observations) / (len(observations) + stabilization_strength)
        stabilized_score = float(fallback) + (
            (weighted_score - float(fallback)) * observation_weight
        )
        return float(np.clip(stabilized_score, 0.0, 1.0))

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
            validate_team_characteristics(data, expected_year=target_year)
        except ValueError as e:
            logger.error(f"Failed to load team characteristics: {e}")
            raise

        raw_teams = data.get("teams", {})
        if not isinstance(raw_teams, dict):
            raise ValueError("Team characteristics payload is missing a valid `teams` mapping")
        self.teams = canonicalize_team_payload_keys(raw_teams)
        checkpoint_snapshot = data.get("checkpoint_snapshot", {})
        self.car_characteristics_snapshot = (
            checkpoint_snapshot if isinstance(checkpoint_snapshot, dict) else {}
        )

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

        try:
            races_completed_value = int(races_completed)
        except (TypeError, ValueError):
            races_completed_value = 0
        races_completed = self._resolve_saved_actual_races_completed(
            target_year=target_year,
            data_freshness=str(data_freshness),
            races_completed=max(0, races_completed_value),
        )
        if data_freshness == "BASELINE_PRESEASON" and races_completed > races_completed_value:
            logger.info(
                "Pre-season payload will use contextual saved-actual fallback from %s completed race(s).",
                races_completed,
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
            for driver_file in driver_characteristics_fallback_paths(self.data_dir, target_year):
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
            validate_driver_characteristics(driver_data, expected_year=target_year)
        except ValueError as e:
            logger.error(f"Failed to load driver characteristics: {e}")
            raise

        # ERROR DETECTION: Check for extraction bugs (does NOT correct)
        from src.utils.driver_validation import validate_driver_data

        errors = validate_driver_data(driver_data["drivers"])
        if errors:
            logger.warning(
                f"Driver data has {len(errors)} validation errors. "
                "Consider re-running extraction: python scripts/extract_driver_characteristics.py --years 2023,2024,2025,2026"
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

        if track_data:
            try:
                validate_track_characteristics(track_data, expected_year=target_year)
            except ValueError as e:
                logger.error(f"Failed to load track characteristics: {e}")
                raise

        self.tracks = track_data.get("tracks", {})
        if self.tracks:
            logger.info(f"Loaded track characteristics for {len(self.tracks)} circuits")

        # Store races completed and year for weight schedule (from car characteristics)
        self.races_completed = races_completed
        self.year = data.get("year", target_year)

    def _resolve_team_data(self, team: str) -> dict:
        """Resolve team payload using alias-aware mapping before fallback."""
        return resolve_team_data_helper(teams=self.teams, team=team)

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Calculate track-car suitability from directionality and layout mix."""
        return calculate_track_suitability_helper(
            team_data=self._resolve_team_data(team),
            track_profile=self.tracks.get(race_name, {}),
        )

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        """Blend baseline, track suitability, and current-season form into one score."""
        cfg = getattr(self, "config", config_loader)
        return get_blended_team_strength_helper(
            context=self,
            team=team,
            race_name=race_name,
            cfg=cfg,
            schedules=SCHEDULES,
            get_recommended_schedule_fn=get_recommended_schedule,
            calculate_blended_performance_fn=calculate_blended_performance,
        )

    def _select_race_compound(self, race_name: str) -> str:
        """Select the likely primary race compound from Pirelli tire-stress data."""
        cfg = getattr(self, "config", config_loader)
        season_year = int(getattr(self, "season_year", getattr(self, "year", 2026)))
        return select_race_compound(race_name=race_name, season_year=season_year, cfg=cfg)

    def get_compound_adjusted_team_strength(
        self, team: str, race_name: str, compound: str = "MEDIUM"
    ) -> float:
        """Return team strength adjusted for compound-specific performance data."""
        cfg = getattr(self, "config", config_loader)
        return get_compound_adjusted_team_strength_helper(
            context=self,
            team=team,
            race_name=race_name,
            compound=compound,
            cfg=cfg,
            should_use_compound_adjustments_fn=should_use_compound_adjustments,
            get_compound_performance_modifier_fn=get_compound_performance_modifier,
        )

    def _get_testing_characteristics_for_profile(self, team: str, profile: str) -> dict[str, float]:
        """Return testing characteristics for one run profile with older-file fallbacks."""
        return get_testing_characteristics_for_profile(
            resolve_team_data=self._resolve_team_data,
            team=team,
            profile=profile,
        )

    def _get_checkpoint_driver_delta_seconds(
        self,
        team: str,
        driver: str,
        preferred_profiles: tuple[str, ...] = ("short_run", "balanced", "long_run"),
    ) -> float | None:
        """Return one checkpoint-backed driver lap-time delta when a snapshot provides it."""
        team_data = self._resolve_team_data(team)
        checkpoint_driver_deltas = team_data.get("checkpoint_driver_deltas_seconds")
        if not isinstance(checkpoint_driver_deltas, dict) or not checkpoint_driver_deltas:
            return None

        driver_code = str(driver).strip().upper()
        if not driver_code:
            return None

        for profile_name in preferred_profiles:
            profile_deltas = checkpoint_driver_deltas.get(profile_name)
            if not isinstance(profile_deltas, dict):
                continue
            raw_delta = profile_deltas.get(driver_code)
            if not isinstance(raw_delta, int | float | str):
                continue
            try:
                delta_seconds = float(raw_delta)
            except (TypeError, ValueError):
                continue
            if np.isfinite(delta_seconds):
                return delta_seconds

        return None

    def _compute_testing_profile_modifier(
        self,
        team: str,
        profile: str,
        metric_weights: dict[str, float],
        scale: float,
    ) -> tuple[float, bool]:
        """Compute a small bounded modifier from testing or practice profile data."""
        cfg = getattr(self, "config", config_loader)
        return compute_testing_profile_modifier(
            team=team,
            profile=profile,
            metric_weights=metric_weights,
            scale=scale,
            get_testing_characteristics_for_profile_fn=self._get_testing_characteristics_for_profile,
            cfg=cfg,
        )

    def _update_compound_characteristics_from_session(
        self,
        session_laps: pd.DataFrame,
        race_name: str,
        year: int,
        is_sprint: bool,
    ) -> None:
        """Extract compound characteristics with session-level caching and persistence."""
        cfg = getattr(self, "config", config_loader)
        update_compound_characteristics_from_session(
            context=self,
            session_laps=session_laps,
            race_name=race_name,
            year=year,
            is_sprint=is_sprint,
            cfg=cfg,
        )
