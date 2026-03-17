"""Data-loading and team-strength mixin for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
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
from src.utils.accuracy_targets import explicit_target_actuals, synthesize_legacy_actuals
from src.utils.compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
)
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger("src.predictors.baseline_2026")


def _driver_characteristics_fallback_paths(data_dir: Path, year: int) -> tuple[Path, ...]:
    """Return season-aware driver-characteristics fallback candidates."""
    return (
        data_dir / "driver_characteristics" / f"{year}_driver_characteristics.json",
        data_dir / "driver_characteristics.json",
    )


def _is_missing_payload_value(value: object) -> bool:
    """Return True when payload value should be treated as missing during merge."""
    if value is None:
        return True
    if isinstance(value, float):
        return not np.isfinite(value)
    return False


def _merge_team_payload(existing: object, incoming: object) -> object:
    """Merge team payload fragments while preserving existing non-missing values."""
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = deepcopy(existing)
        for key, incoming_value in incoming.items():
            if key not in merged:
                merged[key] = deepcopy(incoming_value)
                continue
            merged[key] = _merge_team_payload(merged[key], incoming_value)
        return merged

    if _is_missing_payload_value(existing) and not _is_missing_payload_value(incoming):
        return deepcopy(incoming)
    return deepcopy(existing)


def _canonicalize_team_payload_keys(teams_payload: dict[str, object]) -> dict[str, dict]:
    """
    Canonicalize team payload keys (for example Sauber lineage -> Audi) with safe merges.

    Returns a dictionary keyed by characteristics team labels used across predictors.
    """
    canonical_payload: dict[str, dict] = {}
    for raw_team_name, raw_team_data in teams_payload.items():
        if not isinstance(raw_team_data, dict):
            continue

        mapped_name = map_team_to_characteristics(str(raw_team_name))
        team_name = (
            mapped_name if isinstance(mapped_name, str) and mapped_name else str(raw_team_name)
        )

        existing = canonical_payload.get(team_name)
        if existing is None:
            canonical_payload[team_name] = deepcopy(raw_team_data)
            continue

        merged = _merge_team_payload(existing, raw_team_data)
        canonical_payload[team_name] = merged if isinstance(merged, dict) else existing

    return canonical_payload


def _sanitize_performance_observations(observations: object) -> list[float]:
    """Return a finite 0-1 performance series from a raw observations payload."""
    if not isinstance(observations, list):
        return []

    sanitized: list[float] = []
    for value in observations:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        sanitized.append(float(np.clip(numeric_value, 0.0, 1.0)))
    return sanitized


def _extract_target_actual_rows(
    prediction_data: dict[str, object],
    *,
    target_key: str,
) -> list[dict[str, object]]:
    """Return canonical actual rows for one target from a saved prediction payload."""
    explicit_targets = explicit_target_actuals(prediction_data)
    explicit_rows = explicit_targets.get(target_key)
    if explicit_rows:
        return explicit_rows

    metadata = prediction_data.get("metadata", {})
    weekend_format = ""
    if isinstance(metadata, dict):
        weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
    synthesized_targets = synthesize_legacy_actuals(
        prediction_data,
        is_sprint=weekend_format == "sprint",
    )
    return synthesized_targets.get(target_key, [])


def _score_teams_from_actual_rows(
    actual_rows: list[dict[str, object]],
    *,
    known_teams: set[str],
) -> dict[str, float]:
    """Convert actual classified positions into normalized team-form scores."""
    team_positions: dict[str, list[int]] = {}
    field_size = 0

    for row in actual_rows:
        raw_team = row.get("team")
        if not isinstance(raw_team, str) or not raw_team.strip():
            continue

        canonical_team = map_team_to_characteristics(raw_team, known_teams=known_teams)
        team_name = canonical_team if canonical_team else raw_team.strip()

        position = _coerce_non_negative_int(row.get("position"))
        if position is None:
            continue
        if position < 1:
            continue

        team_positions.setdefault(team_name, []).append(position)
        field_size = max(field_size, position)

    if field_size < 2:
        return {}

    scored_teams: dict[str, float] = {}
    denominator = float(field_size - 1)
    for team_name, positions in team_positions.items():
        if not positions:
            continue
        average_position = float(np.mean(positions))
        normalized_score = 1.0 - ((average_position - 1.0) / denominator)
        scored_teams[team_name] = float(np.clip(normalized_score, 0.0, 1.0))

    return scored_teams


def _coerce_non_negative_int(value: object) -> int | None:
    """Convert an int-like value into a non-negative integer when possible."""
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int | float | np.integer | np.floating):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, str | bytes | bytearray):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return max(parsed, 0)


class BaselineDataMixin:
    """Shared data and team-strength methods for Baseline2026Predictor."""

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
            _score_teams_from_actual_rows(qualifying_rows, known_teams=known_teams)
            if qualifying_rows
            else {}
        )
        race_scores = (
            _score_teams_from_actual_rows(race_rows, known_teams=known_teams) if race_rows else {}
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

            qualifying_rows = _extract_target_actual_rows(
                prediction,
                target_key="main_qualifying",
            )
            race_rows = _extract_target_actual_rows(
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
            _sanitize_performance_observations(team_data.get("current_season_performance"))
            for team_data in self.teams.values()
        )
        if data_freshness == "LIVE_UPDATED" or has_live_current_form:
            return races_completed

        race_records = self._load_saved_actual_race_scores(target_year)
        inferred_races = len(race_records)
        if inferred_races <= 0:
            return races_completed

        teams_with_scores = {
            str(team_name)
            for record in race_records
            for team_name in (record.get("team_scores", {}) or {}).keys()
            if team_name in self.teams
        }

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
        live_observations = _sanitize_performance_observations(
            team_data.get("current_season_performance")
        )
        prior_race_limit = _coerce_non_negative_int(
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
            _coerce_non_negative_int(getattr(self, "races_completed", 0)) or 0,
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
            validate_team_characteristics(data)
        except ValueError as e:
            logger.error(f"Failed to load team characteristics: {e}")
            raise

        raw_teams = data.get("teams", {})
        if not isinstance(raw_teams, dict):
            raise ValueError("Team characteristics payload is missing a valid `teams` mapping")
        self.teams = _canonicalize_team_payload_keys(raw_teams)

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

        self.tracks = track_data.get("tracks", {})
        if self.tracks:
            logger.info(f"Loaded track characteristics for {len(self.tracks)} circuits")

        # Store races completed and year for weight schedule (from car characteristics)
        self.races_completed = races_completed
        self.year = data.get("year", target_year)

    def _resolve_team_data(self, team: str) -> dict:
        """Resolve team payload using alias-aware mapping before fallback."""
        team_data = self.teams.get(team)
        if isinstance(team_data, dict):
            return team_data

        known_teams = set(self.teams.keys())
        mapped_team = map_team_to_characteristics(team, known_teams=known_teams)
        if not isinstance(mapped_team, str) or not mapped_team:
            return {}

        mapped_team_data = self.teams.get(mapped_team)
        return mapped_team_data if isinstance(mapped_team_data, dict) else {}

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Calculate track-car suitability modifier (-0.1 to +0.1) based on car directionality vs track composition."""
        team_data = self._resolve_team_data(team)
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
        3. Current season form (recency-weighted) - increases over season
        """
        team_data = self._resolve_team_data(team)

        # 1. Baseline from 2025 standings
        baseline = team_data.get("overall_performance", 0.5)

        # 2. Testing/practice-informed track suitability signal (relative modifier)
        testing_modifier = self.calculate_track_suitability(team, race_name)
        testing_score = float(np.clip(baseline + testing_modifier, 0.0, 1.0))

        # 3. Current season form with a strong recency bias.
        current = self._get_current_season_score(
            team,
            team_data,
            fallback=baseline,
            race_name=race_name,
        )

        # 4. Apply weight schedule
        race_number = self._get_contextual_races_completed(race_name) + 1

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
        team_data = self._resolve_team_data(team)
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
        team_data = self._resolve_team_data(team)

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
