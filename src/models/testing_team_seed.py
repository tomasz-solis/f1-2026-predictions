"""Learn preseason team seeds from cached testing telemetry.

This module keeps the experimental 2026 rule-reset path separate from the
current standings-seeded baseline. The goal is narrow on purpose:

- build team-level preseason feature rows from cached testing and early FP
- learn a mapping from those rows to early-season team pace
- emit a normal team-characteristics artifact the predictor already understands

Driver priors stay on the existing path in v1.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
from scripts.calculate_team_performance import calculate_team_performance_from_races
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.systems.compound_analyzer import aggregate_compound_samples
from src.systems.testing_updater import (
    _PROFILES_FOR_STORAGE,
    _TESTING_CHARACTERISTIC_METRICS,
    _collect_session_metrics,
    _count_team_selected_laps,
    _extract_session_compound_metrics,
    _extract_session_driver_deltas,
    _load_sessions_for_event,
    _resolve_testing_backends,
    _resolve_testing_cache_dir,
)
from src.systems.testing_updater_flow import (
    apply_team_updates,
    collect_sessions_for_events,
    raise_if_no_loaded_sessions,
)
from src.systems.testing_updater_metrics import (
    _aggregate_metric_samples,
    _blend_directionality,
    _build_directionality_from_metrics,
)
from src.utils.normalization import rank_normalize
from src.utils.schema_validation import validate_team_characteristics
from src.utils.team_mapping import CHARACTERISTICS_TEAM_MAP, map_team_to_characteristics
from src.utils.weekend import get_schedule_rows

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/raw/.fastf1_cache")
_DEFAULT_FEATURE_CACHE_DIR = Path("data/processed/model_artifacts/testing_team_seed/features")
_DEFAULT_TESTING_BACKEND = "auto"
_DEFAULT_TEAM_SET = tuple(sorted(set(CHARACTERISTICS_TEAM_MAP.values())))
_PRESEASON_SESSION_CANDIDATES = (
    "Day 1",
    "Day 2",
    "Day 3",
    "Practice 1",
    "Practice 2",
    "Practice 3",
)
_AUXILIARY_SESSION_CANDIDATES = ("FP1", "FP2", "Practice 1", "Practice 2")
_PRESEASON_SAMPLE_WEIGHT = 1.0
_AUXILIARY_FP_SAMPLE_WEIGHT = 0.35
_REGULATION_RESET_YEARS = tuple(sorted({2014, 2022, 2026}))
_MIN_TRANSFER_WEIGHT = 0.35
_RESET_TARGET_MATURE_STAGE_CAP = 0.45
_RANK_SEED_ANCHORS = (0.85, 0.75, 0.74, 0.70, 0.55, 0.48, 0.47, 0.43, 0.40, 0.38, 0.35)
_DEFAULT_CHAMPION_PRIOR_ORDER = (
    "McLaren",
    "Mercedes",
    "Red Bull Racing",
    "Ferrari",
    "Williams",
    "RB",
    "Aston Martin",
    "Haas F1 Team",
    "Alpine",
    "Audi",
    "Cadillac F1",
)
_MIN_SOURCE_RANKED_TEAMS = 5
_MODEL_ALPHA_GRID = tuple(float(alpha) for alpha in np.logspace(-3, 3, 25))
_PROFILE_METRICS = (
    "overall_pace",
    "slow_corner_performance",
    "medium_corner_performance",
    "fast_corner_performance",
    "braking_performance",
    "top_speed",
    "consistency",
    "tire_deg_performance",
    "tire_deg_slope",
)
_PROFILE_GAP_METRICS = (
    "overall_pace",
    "top_speed",
    "consistency",
    "tire_deg_performance",
)
_DIRECTIONALITY_KEYS = (
    "max_speed",
    "slow_corner_speed",
    "medium_corner_speed",
    "high_corner_speed",
)
_FEATURE_CACHE_VERSION = 1
_PACE_GLIMPSE_SCALE = 0.28
_MAX_TESTING_DELTA_FROM_CHAMPION = 0.08
_MAX_TESTING_DELTA_MULTIPLIER = 0.45


def _build_feature_columns() -> tuple[str, ...]:
    """Return the fixed feature list used by the testing team model."""
    feature_columns: list[str] = []

    for profile_name in _PROFILES_FOR_STORAGE:
        for metric_name in _PROFILE_METRICS:
            feature_columns.append(f"{profile_name}_{metric_name}")

    for metric_name in _PROFILE_GAP_METRICS:
        feature_columns.append(f"short_run_minus_balanced_{metric_name}")
        feature_columns.append(f"long_run_minus_balanced_{metric_name}")
        feature_columns.append(f"short_run_minus_long_run_{metric_name}")

    for directionality_key in _DIRECTIONALITY_KEYS:
        feature_columns.append(f"directionality_{directionality_key}")

    feature_columns.extend(
        (
            "coverage_sessions_used",
            "coverage_sessions_blended",
            "coverage_circuits_observed",
            "coverage_loaded_sessions",
            "coverage_selected_lap_weight",
        )
    )

    return tuple(feature_columns)


FEATURE_COLUMNS = _build_feature_columns()


@dataclass(frozen=True)
class TestingSeedSnapshot:
    """Aggregated team-level testing snapshot for one source window."""

    year: int
    event_names: tuple[str, ...]
    source_kind: str
    loaded_sessions: tuple[str, ...]
    updated_teams: tuple[str, ...]
    teams: dict[str, dict[str, Any]]
    selected_lap_weights: dict[str, float]
    team_session_counts: dict[str, int]


@dataclass(frozen=True)
class LeaveOneSeasonOutSummary:
    """Cross-validated summary of preseason-only team-seed predictions."""

    rows: list[dict[str, Any]]
    seasons: list[dict[str, Any]]
    mae: float
    rmse: float


@dataclass
class FittedTestingTeamSeedModel:
    """Small wrapper around the fitted preprocessors and RidgeCV regressor."""

    feature_names: tuple[str, ...]
    training_years: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    regressor: RidgeCV

    def predict_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """Predict raw team-strength scores for feature rows."""
        missing_columns = [column for column in self.feature_names if column not in rows.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns for prediction: {missing_columns}")

        transformed = self.imputer.transform(rows.loc[:, self.feature_names])
        scaled = self.scaler.transform(transformed)
        return np.asarray(self.regressor.predict(scaled), dtype=float)


def discover_preseason_event_names(
    year: int,
    *,
    cache_dirs: Iterable[str | Path] | None = None,
) -> list[str]:
    """Discover preseason event labels from cached FastF1 directories.

    The cache uses slightly different names across eras. This helper intentionally
    accepts both `Pre-Season Testing` and the older `Pre-Season Track Session`.
    """
    discovered: dict[str, str] = {}
    candidate_dirs = cache_dirs or (_DEFAULT_CACHE_DIR,)

    for raw_cache_dir in candidate_dirs:
        year_root = Path(raw_cache_dir) / str(int(year))
        if not year_root.exists():
            continue

        for event_dir in year_root.iterdir():
            if not event_dir.is_dir():
                continue

            raw_name = str(event_dir.name).strip()
            if "_" not in raw_name:
                continue

            raw_date, raw_label = raw_name.split("_", 1)
            event_label = raw_label.replace("_", " ").strip()
            normalized_label = "".join(char for char in event_label.lower() if char.isalnum())
            is_preseason = "preseason" in normalized_label and (
                "testing" in normalized_label or "tracksession" in normalized_label
            )
            if not is_preseason:
                continue

            try:
                has_payload = any(
                    child.is_dir() or child.is_file() for child in event_dir.iterdir()
                )
            except OSError:
                has_payload = False
            if not has_payload:
                continue

            best_date = discovered.get(event_label)
            if best_date is None or raw_date < best_date:
                discovered[event_label] = raw_date

    return [
        label for label, _date in sorted(discovered.items(), key=lambda item: (item[1], item[0]))
    ]


def discover_auxiliary_race_events(year: int, *, max_events: int = 2) -> list[str]:
    """Return the first non-testing race weekends used as auxiliary FP samples."""
    rows = [event_name for event_name, _event_format in get_schedule_rows(year)]
    if max_events <= 0:
        return []
    return rows[:max_events]


def _empty_characteristics_payload(teams: Iterable[str]) -> dict[str, Any]:
    """Build a minimal characteristics payload for in-memory testing updates."""
    return {
        "year": 0,
        "data_freshness": "BASELINE_PRESEASON",
        "teams": {team_name: {"overall_performance": 0.5} for team_name in teams},
    }


def _coerce_float(value: Any) -> float | None:
    """Convert one scalar to a finite float when possible."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _flatten_team_features(
    team_payload: dict[str, Any],
    *,
    loaded_session_count: int,
    selected_lap_weight: float,
) -> dict[str, float | None]:
    """Flatten one team testing payload into a fixed model-feature row."""
    feature_values: dict[str, float | None] = {column: None for column in FEATURE_COLUMNS}

    profiles = team_payload.get("testing_characteristics_profiles")
    if not isinstance(profiles, dict):
        profiles = {}

    for profile_name in _PROFILES_FOR_STORAGE:
        profile_payload = profiles.get(profile_name)
        if not isinstance(profile_payload, dict):
            profile_payload = {}
        for metric_name in _PROFILE_METRICS:
            feature_values[f"{profile_name}_{metric_name}"] = _coerce_float(
                profile_payload.get(metric_name)
            )

    for metric_name in _PROFILE_GAP_METRICS:
        balanced_value = feature_values.get(f"balanced_{metric_name}")
        short_value = feature_values.get(f"short_run_{metric_name}")
        long_value = feature_values.get(f"long_run_{metric_name}")
        if balanced_value is not None and short_value is not None:
            feature_values[f"short_run_minus_balanced_{metric_name}"] = short_value - balanced_value
        if balanced_value is not None and long_value is not None:
            feature_values[f"long_run_minus_balanced_{metric_name}"] = long_value - balanced_value
        if short_value is not None and long_value is not None:
            feature_values[f"short_run_minus_long_run_{metric_name}"] = short_value - long_value

    directionality = team_payload.get("directionality")
    if isinstance(directionality, dict):
        for directionality_key in _DIRECTIONALITY_KEYS:
            feature_values[f"directionality_{directionality_key}"] = _coerce_float(
                directionality.get(directionality_key)
            )

    testing_characteristics = team_payload.get("testing_characteristics")
    if not isinstance(testing_characteristics, dict):
        testing_characteristics = {}

    circuits_observed = testing_characteristics.get("circuits_observed")
    circuit_count = 0
    if isinstance(circuits_observed, list):
        circuit_count = len([circuit for circuit in circuits_observed if str(circuit).strip()])

    feature_values["coverage_sessions_used"] = _coerce_float(
        testing_characteristics.get("sessions_used")
    )
    feature_values["coverage_sessions_blended"] = _coerce_float(
        testing_characteristics.get("sessions_blended")
    )
    feature_values["coverage_circuits_observed"] = float(circuit_count)
    feature_values["coverage_loaded_sessions"] = float(max(loaded_session_count, 0))
    feature_values["coverage_selected_lap_weight"] = float(max(selected_lap_weight, 0.0))
    return feature_values


def _canonicalize_team_metric_map(raw_team_values: dict[str, float]) -> dict[str, float]:
    """Map raw FastF1 team labels onto the predictor's shared team keys."""
    canonical_values: dict[str, list[float]] = {}
    known_teams = set(_DEFAULT_TEAM_SET)

    for raw_team_name, raw_value in raw_team_values.items():
        canonical_team = map_team_to_characteristics(raw_team_name, known_teams=known_teams)
        if canonical_team is None:
            continue
        canonical_values.setdefault(canonical_team, []).append(float(raw_value))

    return {
        team_name: float(np.mean(team_values))
        for team_name, team_values in canonical_values.items()
        if team_values
    }


def build_target_label_map(year: int, *, race_limit: int = 3) -> dict[str, float]:
    """Build early-season team labels from the first completed races of a season."""
    race_payload = calculate_team_performance_from_races(year, max_races=race_limit)
    raw_values = {
        team_name: float(team_payload["overall_performance"])
        for team_name, team_payload in race_payload.items()
        if isinstance(team_payload, dict) and team_payload.get("overall_performance") is not None
    }
    canonical_values = _canonicalize_team_metric_map(raw_values)
    if not canonical_values:
        return {}
    return rank_normalize(canonical_values, higher_is_better=True)


def collect_team_testing_snapshot(
    year: int,
    *,
    event_names: Iterable[str],
    session_candidates: Iterable[str],
    source_kind: str,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    testing_backend: str | None = _DEFAULT_TESTING_BACKEND,
    session_aggregation: str = "laps_weighted",
) -> TestingSeedSnapshot:
    """Collect one aggregated team snapshot from cached testing or FP sessions."""
    event_names = tuple(str(event_name) for event_name in event_names if str(event_name).strip())
    if not event_names:
        raise ValueError("At least one event name is required for testing snapshot collection.")

    known_teams = set(_DEFAULT_TEAM_SET)
    cache_path = _resolve_testing_cache_dir(str(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))

    collection = collect_sessions_for_events(
        year=int(year),
        events=list(event_names),
        session_candidates=[str(candidate) for candidate in session_candidates],
        testing_backends=_resolve_testing_backends(testing_backend),
        known_teams=known_teams,
        run_profile="balanced",
        profiles_for_storage=_PROFILES_FOR_STORAGE,
        load_sessions_for_event=_load_sessions_for_event,
        collect_session_metrics=_collect_session_metrics,
        extract_session_driver_deltas=_extract_session_driver_deltas,
        count_team_selected_laps=_count_team_selected_laps,
        extract_session_compound_metrics=_extract_session_compound_metrics,
        logger=logger,
    )
    raise_if_no_loaded_sessions(
        discovered_sessions=collection.discovered_sessions,
        loaded_sessions=collection.loaded_sessions,
        extraction_diagnostics=collection.extraction_diagnostics,
        load_errors=collection.load_errors,
    )

    characteristics = _empty_characteristics_payload(known_teams)
    updated_teams = apply_team_updates(
        characteristics=characteristics,
        metric_samples=collection.metric_samples,
        profile_metric_samples=collection.profile_metric_samples,
        team_sessions_used=collection.team_sessions_used,
        team_profile_sessions_used=collection.team_profile_sessions_used,
        compound_metrics_by_session=collection.compound_metrics_by_session,
        now_iso=datetime.now(UTC).isoformat(),
        session_aggregation=session_aggregation,
        run_profile="balanced",
        directionality_scale=0.10,
        new_weight=1.0,
        profiles_for_storage=_PROFILES_FOR_STORAGE,
        testing_characteristic_metrics=_TESTING_CHARACTERISTIC_METRICS,
        aggregate_metric_samples=_aggregate_metric_samples,
        build_directionality_from_metrics=_build_directionality_from_metrics,
        blend_directionality=_blend_directionality,
        aggregate_compound_samples=aggregate_compound_samples,
    )

    selected_lap_weights: dict[str, float] = {}
    for team_name, metric_samples in collection.metric_samples.items():
        overall_pace_samples = metric_samples.get("overall_pace", [])
        selected_lap_weights[team_name] = float(
            sum(max(0.0, float(weight)) for _value, weight in overall_pace_samples)
        )

    team_session_counts = {
        team_name: len(session_ids)
        for team_name, session_ids in collection.team_sessions_used.items()
    }

    return TestingSeedSnapshot(
        year=int(year),
        event_names=event_names,
        source_kind=str(source_kind),
        loaded_sessions=tuple(collection.loaded_sessions),
        updated_teams=tuple(sorted(updated_teams)),
        teams=characteristics["teams"],
        selected_lap_weights=selected_lap_weights,
        team_session_counts=team_session_counts,
    )


def _rows_from_snapshot(
    snapshot: TestingSeedSnapshot,
    *,
    label_map: dict[str, float],
    sample_weight: float,
) -> list[dict[str, Any]]:
    """Convert one collected snapshot into supervised training rows."""
    rows: list[dict[str, Any]] = []

    for team_name in snapshot.updated_teams:
        if team_name not in label_map:
            continue
        team_payload = snapshot.teams.get(team_name, {})
        if not isinstance(team_payload, dict):
            continue

        feature_values = _flatten_team_features(
            team_payload,
            loaded_session_count=snapshot.team_session_counts.get(team_name, 0),
            selected_lap_weight=snapshot.selected_lap_weights.get(team_name, 0.0),
        )
        rows.append(
            {
                "season_year": int(snapshot.year),
                "team_name": team_name,
                "source_kind": snapshot.source_kind,
                "source_events": list(snapshot.event_names),
                "sample_weight": float(sample_weight),
                "target_team_strength": float(label_map[team_name]),
                **feature_values,
            }
        )

    return rows


def _feature_cache_path(
    year: int,
    *,
    feature_cache_dir: str | Path = _DEFAULT_FEATURE_CACHE_DIR,
) -> Path:
    """Return the cache file path for one season feature snapshot."""
    return Path(feature_cache_dir) / f"{int(year)}_team_seed_features.json"


def _load_cached_training_rows(
    year: int,
    *,
    auxiliary_event_limit: int,
    cache_dir: str | Path,
    feature_cache_dir: str | Path = _DEFAULT_FEATURE_CACHE_DIR,
) -> list[dict[str, Any]] | None:
    """Load cached training rows for one season when settings match."""
    cache_path = _feature_cache_path(year, feature_cache_dir=feature_cache_dir)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    expected_cache_dir = str(Path(cache_dir))
    if int(metadata.get("cache_version", -1)) != _FEATURE_CACHE_VERSION:
        return None
    if int(metadata.get("auxiliary_event_limit", -1)) != int(auxiliary_event_limit):
        return None
    if str(metadata.get("fastf1_cache_dir", "")) != expected_cache_dir:
        return None

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return None
    return [row for row in rows if isinstance(row, dict)]


def _save_cached_training_rows(
    *,
    year: int,
    rows: list[dict[str, Any]],
    auxiliary_event_limit: int,
    cache_dir: str | Path,
    feature_cache_dir: str | Path = _DEFAULT_FEATURE_CACHE_DIR,
) -> Path:
    """Persist training rows for one season so repeated runs can reuse them."""
    cache_path = _feature_cache_path(year, feature_cache_dir=feature_cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "cache_version": _FEATURE_CACHE_VERSION,
            "season_year": int(year),
            "auxiliary_event_limit": int(auxiliary_event_limit),
            "fastf1_cache_dir": str(Path(cache_dir)),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return cache_path


def _regulation_cycle_stage(year: int) -> int:
    """Return how many seasons a year sits beyond the most recent rules reset."""
    resolved_year = int(year)
    reset_year = max(
        (candidate for candidate in _REGULATION_RESET_YEARS if candidate <= resolved_year),
        default=_REGULATION_RESET_YEARS[0],
    )
    return max(0, resolved_year - int(reset_year))


def compute_target_year_transfer_weight(
    *,
    source_year: int,
    target_year: int,
) -> float:
    """Estimate how relevant one training season is for a target season.

    The intent is narrow and practical: for a regulation-reset target like 2026,
    another reset-year season like 2022, and future 2014-derived data if present,
    should count more than a mature carryover season such as 2025. Recency still
    matters, but regulation-cycle similarity matters first.
    """

    source_year_value = int(source_year)
    target_year_value = int(target_year)
    source_stage = _regulation_cycle_stage(source_year_value)
    target_stage = _regulation_cycle_stage(target_year_value)

    stage_similarity = float(
        np.clip(
            1.0 - (0.12 * abs(source_stage - target_stage)),
            _MIN_TRANSFER_WEIGHT,
            1.0,
        )
    )
    recency_multiplier = float(
        np.clip(
            1.0 - (0.03 * abs(target_year_value - source_year_value)),
            0.82,
            1.0,
        )
    )
    transfer_weight = float(
        np.clip(
            stage_similarity * recency_multiplier,
            _MIN_TRANSFER_WEIGHT,
            1.0,
        )
    )
    if target_stage == 0 and source_stage >= 3:
        transfer_weight = min(transfer_weight, _RESET_TARGET_MATURE_STAGE_CAP)
    return transfer_weight


def apply_target_year_transfer_weights(
    dataset: pd.DataFrame,
    *,
    target_year: int,
) -> pd.DataFrame:
    """Return a copy of the dataset with regulation-aware effective weights.

    Base sample weights still express how much we trust preseason versus
    auxiliary FP rows. This helper adds a second layer: how much one source
    season should transfer to the target season's regulation context.
    """

    weighted = dataset.copy()
    base_sample_weight = (
        weighted["sample_weight"].astype(float)
        if "sample_weight" in weighted.columns
        else pd.Series(dtype=float)
    )
    transfer_weight = (
        weighted["season_year"]
        .astype(int)
        .map(
            lambda year: compute_target_year_transfer_weight(
                source_year=year, target_year=target_year
            )
        )
        if "season_year" in weighted.columns
        else pd.Series(dtype=float)
    )
    weighted["base_sample_weight"] = base_sample_weight
    weighted["transfer_weight"] = transfer_weight
    weighted["effective_sample_weight"] = base_sample_weight * transfer_weight
    return weighted


def summarize_target_year_transfer_weights(
    dataset: pd.DataFrame,
    *,
    target_year: int,
) -> list[dict[str, Any]]:
    """Summarize how much each training season contributes to one target year."""

    if dataset.empty:
        return []

    weighted = apply_target_year_transfer_weights(dataset, target_year=target_year)
    total_effective_weight = float(weighted["effective_sample_weight"].sum())
    summaries: list[dict[str, Any]] = []
    for season_year in sorted(
        int(year) for year in weighted["season_year"].astype(int).unique().tolist()
    ):
        season_rows = weighted[weighted["season_year"].astype(int) == season_year].reset_index(
            drop=True
        )
        effective_weight_total = float(season_rows["effective_sample_weight"].sum())
        summaries.append(
            {
                "season_year": season_year,
                "regulation_cycle_stage": _regulation_cycle_stage(season_year),
                "row_count": int(len(season_rows)),
                "transfer_weight_mean": round(
                    float(season_rows["transfer_weight"].astype(float).mean()),
                    4,
                ),
                "base_sample_weight_total": round(
                    float(season_rows["base_sample_weight"].astype(float).sum()),
                    4,
                ),
                "effective_sample_weight_total": round(effective_weight_total, 4),
                "effective_sample_weight_share": round(
                    (effective_weight_total / total_effective_weight)
                    if total_effective_weight > 0.0
                    else 0.0,
                    4,
                ),
            }
        )
    return summaries


def build_training_dataset(
    training_years: Iterable[int],
    *,
    auxiliary_event_limit: int = 2,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    feature_cache_dir: str | Path = _DEFAULT_FEATURE_CACHE_DIR,
) -> pd.DataFrame:
    """Build the training table for the testing-derived team prior model."""
    rows: list[dict[str, Any]] = []

    for raw_year in training_years:
        year = int(raw_year)
        cached_rows = _load_cached_training_rows(
            year,
            auxiliary_event_limit=auxiliary_event_limit,
            cache_dir=cache_dir,
            feature_cache_dir=feature_cache_dir,
        )
        if cached_rows is not None:
            rows.extend(cached_rows)
            continue

        label_map = build_target_label_map(year, race_limit=3)
        if not label_map:
            logger.warning(
                "Skipping %s training rows because no target labels were available.", year
            )
            continue

        year_rows: list[dict[str, Any]] = []
        preseason_events = discover_preseason_event_names(year, cache_dirs=(cache_dir,))
        if preseason_events:
            try:
                preseason_snapshot = collect_team_testing_snapshot(
                    year,
                    event_names=preseason_events,
                    session_candidates=_PRESEASON_SESSION_CANDIDATES,
                    source_kind="preseason",
                    cache_dir=cache_dir,
                )
            except ValueError as exc:
                logger.warning("Skipping %s preseason rows: %s", year, exc)
            else:
                year_rows.extend(
                    _rows_from_snapshot(
                        preseason_snapshot,
                        label_map=label_map,
                        sample_weight=_PRESEASON_SAMPLE_WEIGHT,
                    )
                )

        for event_name in discover_auxiliary_race_events(year, max_events=auxiliary_event_limit):
            try:
                fp_snapshot = collect_team_testing_snapshot(
                    year,
                    event_names=(event_name,),
                    session_candidates=_AUXILIARY_SESSION_CANDIDATES,
                    source_kind="auxiliary_fp",
                    cache_dir=cache_dir,
                )
            except ValueError as exc:
                logger.warning("Skipping %s auxiliary FP rows for %s: %s", year, event_name, exc)
            else:
                year_rows.extend(
                    _rows_from_snapshot(
                        fp_snapshot,
                        label_map=label_map,
                        sample_weight=_AUXILIARY_FP_SAMPLE_WEIGHT,
                    )
                )

        if year_rows:
            _save_cached_training_rows(
                year=year,
                rows=year_rows,
                auxiliary_event_limit=auxiliary_event_limit,
                cache_dir=cache_dir,
                feature_cache_dir=feature_cache_dir,
            )
            rows.extend(year_rows)

    return pd.DataFrame(rows)


def fit_testing_team_seed_model(
    dataset: pd.DataFrame,
    *,
    alpha_grid: Iterable[float] = _MODEL_ALPHA_GRID,
) -> FittedTestingTeamSeedModel:
    """Fit the RidgeCV preseason team model on the provided dataset."""
    if dataset.empty:
        raise ValueError("Cannot fit testing team seed model on an empty dataset.")

    feature_frame = dataset.loc[:, FEATURE_COLUMNS].copy()
    target = dataset["target_team_strength"].astype(float).to_numpy()
    sample_weight_column = (
        "effective_sample_weight"
        if "effective_sample_weight" in dataset.columns
        else "sample_weight"
    )
    sample_weight = dataset[sample_weight_column].astype(float).to_numpy()
    training_years = tuple(sorted(int(year) for year in dataset["season_year"].unique().tolist()))

    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(feature_frame)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    regressor = RidgeCV(alphas=tuple(float(alpha) for alpha in alpha_grid))
    regressor.fit(scaled, target, sample_weight=sample_weight)

    return FittedTestingTeamSeedModel(
        feature_names=FEATURE_COLUMNS,
        training_years=training_years,
        alpha_grid=tuple(float(alpha) for alpha in alpha_grid),
        imputer=imputer,
        scaler=scaler,
        regressor=regressor,
    )


def run_leave_one_season_out_validation(
    dataset: pd.DataFrame,
    *,
    use_regulation_transfer_weights: bool = False,
) -> LeaveOneSeasonOutSummary:
    """Run leave-one-season-out validation using preseason rows for scoring."""
    if dataset.empty:
        raise ValueError("Cannot run leave-one-season-out validation on an empty dataset.")

    row_summaries: list[dict[str, Any]] = []
    season_summaries: list[dict[str, Any]] = []

    for holdout_year in sorted(int(year) for year in dataset["season_year"].unique().tolist()):
        training_rows = dataset[dataset["season_year"] != holdout_year].reset_index(drop=True)
        holdout_rows = dataset[
            (dataset["season_year"] == holdout_year) & (dataset["source_kind"] == "preseason")
        ].reset_index(drop=True)
        if training_rows.empty or holdout_rows.empty:
            continue

        if use_regulation_transfer_weights:
            training_rows = apply_target_year_transfer_weights(
                training_rows,
                target_year=holdout_year,
            )
        model = fit_testing_team_seed_model(training_rows)
        raw_predictions = model.predict_rows(holdout_rows)
        normalized_predictions = rank_normalize(
            {
                str(team_name): float(prediction)
                for team_name, prediction in zip(
                    holdout_rows["team_name"].tolist(),
                    raw_predictions.tolist(),
                    strict=False,
                )
            },
            higher_is_better=True,
        )

        absolute_errors: list[float] = []
        for team_name, raw_prediction, actual_value in zip(
            holdout_rows["team_name"].tolist(),
            raw_predictions.tolist(),
            holdout_rows["target_team_strength"].astype(float).tolist(),
            strict=False,
        ):
            normalized_prediction = float(normalized_predictions[str(team_name)])
            absolute_error = abs(normalized_prediction - float(actual_value))
            absolute_errors.append(absolute_error)
            row_summaries.append(
                {
                    "season_year": holdout_year,
                    "team_name": str(team_name),
                    "predicted_raw": float(raw_prediction),
                    "predicted_strength": normalized_prediction,
                    "actual_strength": float(actual_value),
                    "absolute_error": absolute_error,
                }
            )

        season_summaries.append(
            {
                "season_year": holdout_year,
                "teams_scored": len(absolute_errors),
                "mae": float(np.mean(absolute_errors)),
                "rmse": float(np.sqrt(np.mean(np.square(absolute_errors)))),
            }
        )

    all_errors = [float(row["absolute_error"]) for row in row_summaries]
    if not all_errors:
        raise ValueError("No preseason rows were available for leave-one-season-out validation.")

    return LeaveOneSeasonOutSummary(
        rows=row_summaries,
        seasons=season_summaries,
        mae=float(np.mean(all_errors)),
        rmse=float(np.sqrt(np.mean(np.square(all_errors)))),
    )


def _fallback_validation_summary(dataset: pd.DataFrame) -> LeaveOneSeasonOutSummary:
    """Return a conservative validation summary when LOSO scoring is impossible.

    Small smoke runs or narrow diagnostics may intentionally train on a single
    season. In that case there is no honest leave-one-season-out split, but we
    still need a stable uncertainty estimate so the builder can run.
    """
    if dataset.empty:
        raise ValueError("Cannot build a fallback validation summary from an empty dataset.")

    target_values = dataset["target_team_strength"].astype(float).to_numpy()
    target_spread = float(np.std(target_values))
    conservative_error = float(np.clip(max(0.10, target_spread * 0.35), 0.10, 0.20))
    training_years = sorted(
        int(year) for year in dataset["season_year"].astype(int).unique().tolist()
    )
    return LeaveOneSeasonOutSummary(
        rows=[],
        seasons=[
            {
                "season_year": training_years[0],
                "teams_scored": int(len(dataset)),
                "mae": conservative_error,
                "rmse": conservative_error,
                "method": "fallback_single_season",
            }
        ],
        mae=conservative_error,
        rmse=conservative_error,
    )


def _build_rank_seed_map(
    ranked_teams: list[str],
    *,
    minimum_anchor: float = _RANK_SEED_ANCHORS[-1],
) -> dict[str, float]:
    """Map a ranked team list onto the current standings-style preseason anchors."""
    seeded: dict[str, float] = {}
    for index, team_name in enumerate(ranked_teams):
        anchor_index = min(index, len(_RANK_SEED_ANCHORS) - 1)
        seeded[team_name] = float(_RANK_SEED_ANCHORS[anchor_index])
    for team_name in ranked_teams[len(_RANK_SEED_ANCHORS) :]:
        seeded[team_name] = float(minimum_anchor)
    return seeded


def _build_champion_prior_seed_map(
    *,
    source_year: int,
    target_teams: Iterable[str],
) -> dict[str, float]:
    """Return standings-style champion anchors for the requested target teams.

    The live champion path depends on a stable prior-year standings story. When
    the source standings are missing or suspiciously sparse, fall back to the
    default champion order instead of letting partial source data reshuffle the
    whole grid.
    """

    normalized_target_teams = {
        str(team_name).strip() for team_name in target_teams if str(team_name).strip()
    }

    source_scores: dict[str, float] = {}
    try:
        source_team_payload = calculate_team_performance_from_races(source_year)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not load source-year ranking for %s; using default champion prior order: %s",
            source_year,
            exc,
        )
        source_team_payload = {}

    if isinstance(source_team_payload, dict):
        source_scores = _canonicalize_team_metric_map(
            {
                team_name: float(team_payload["overall_performance"])
                for team_name, team_payload in source_team_payload.items()
                if isinstance(team_payload, dict)
                and team_payload.get("overall_performance") is not None
            }
        )

    if len(source_scores) >= _MIN_SOURCE_RANKED_TEAMS:
        ranked_teams = [
            team_name
            for team_name, _value in sorted(
                source_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
    else:
        if source_scores:
            logger.warning(
                "Source year %s returned only %s ranked teams; preserving the default champion order.",
                source_year,
                len(source_scores),
            )
        ranked_teams = list(_DEFAULT_CHAMPION_PRIOR_ORDER)

    for team_name in _DEFAULT_CHAMPION_PRIOR_ORDER:
        if team_name not in ranked_teams:
            ranked_teams.append(team_name)
    for team_name in sorted(normalized_target_teams):
        if team_name not in ranked_teams:
            ranked_teams.append(team_name)

    seeded_performance = _build_rank_seed_map(ranked_teams)
    resolved_teams = (
        sorted(normalized_target_teams)
        if normalized_target_teams
        else sorted(seeded_performance.keys())
    )
    return {
        team_name: float(seeded_performance.get(team_name, _RANK_SEED_ANCHORS[-1]))
        for team_name in resolved_teams
    }


def build_neutral_team_seed_payload(
    *,
    target_year: int,
    generated_at: str | None = None,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Build a neutral comparison payload while preserving target-year telemetry."""
    preseason_events = discover_preseason_event_names(target_year, cache_dirs=(cache_dir,))
    preseason_snapshot = collect_team_testing_snapshot(
        target_year,
        event_names=preseason_events,
        session_candidates=_PRESEASON_SESSION_CANDIDATES,
        source_kind="preseason",
        cache_dir=cache_dir,
    )
    generated_at = generated_at or datetime.now(UTC).isoformat()

    teams_payload: dict[str, Any] = {}
    for team_name in sorted(preseason_snapshot.updated_teams):
        base_team_payload = preseason_snapshot.teams.get(team_name, {})
        if not isinstance(base_team_payload, dict):
            base_team_payload = {}
        team_entry: dict[str, Any] = {
            "overall_performance": 0.5,
            "preseason_overall_performance": 0.5,
            "uncertainty": 0.30,
            "note": "Neutral preseason seed with target-year testing telemetry preserved.",
            "last_updated": generated_at,
            "races_completed": 0,
            "current_season_performance": [],
        }
        for optional_key in (
            "directionality",
            "testing_characteristics",
            "testing_characteristics_profiles",
            "compound_characteristics",
        ):
            optional_value = base_team_payload.get(optional_key)
            if optional_value is not None:
                team_entry[optional_key] = optional_value
        teams_payload[team_name] = team_entry

    payload = {
        "year": int(target_year),
        "data_freshness": "BASELINE_PRESEASON",
        "note": f"Neutral preseason comparison seed for {target_year}.",
        "generated_at": generated_at,
        "last_updated": generated_at,
        "races_completed": 0,
        "directionality_source": "neutral_team_seed",
        "directionality_last_updated": generated_at,
        "directionality_meta": {
            "seed_mode": "neutral",
            "target_year": int(target_year),
            "target_testing_events": list(preseason_snapshot.event_names),
            "loaded_sessions": list(preseason_snapshot.loaded_sessions),
        },
        "teams": teams_payload,
    }
    validate_team_characteristics(payload, expected_year=target_year)
    return payload


def build_prior_year_ranking_seed_payload(
    *,
    target_year: int,
    source_year: int | None = None,
    generated_at: str | None = None,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Build a comparison preseason payload using prior-year team ranking anchors.

    The payload preserves target-year testing telemetry while keeping the
    standings-seeded `overall_performance` story for comparison experiments.
    """
    actual_source_year = int(source_year if source_year is not None else (target_year - 1))
    preseason_events = discover_preseason_event_names(target_year, cache_dirs=(cache_dir,))
    preseason_snapshot = collect_team_testing_snapshot(
        target_year,
        event_names=preseason_events,
        session_candidates=_PRESEASON_SESSION_CANDIDATES,
        source_kind="preseason",
        cache_dir=cache_dir,
    )

    generated_at = generated_at or datetime.now(UTC).isoformat()
    target_teams = set(preseason_snapshot.updated_teams)
    seeded_performance = _build_champion_prior_seed_map(
        source_year=actual_source_year,
        target_teams=target_teams,
    )

    teams_payload: dict[str, Any] = {}
    for team_name in sorted(seeded_performance):
        base_team_payload = preseason_snapshot.teams.get(team_name, {})
        if not isinstance(base_team_payload, dict):
            base_team_payload = {}
        preseason_performance = float(seeded_performance.get(team_name, _RANK_SEED_ANCHORS[-1]))
        team_entry: dict[str, Any] = {
            "overall_performance": preseason_performance,
            "preseason_overall_performance": preseason_performance,
            "uncertainty": 0.30,
            "note": (
                f"{actual_source_year} ranking seed with target-year preseason testing telemetry "
                "preserved for comparison."
            ),
            "last_updated": generated_at,
            "races_completed": 0,
            "current_season_performance": [],
        }
        for optional_key in (
            "directionality",
            "testing_characteristics",
            "testing_characteristics_profiles",
            "compound_characteristics",
        ):
            optional_value = base_team_payload.get(optional_key)
            if optional_value is not None:
                team_entry[optional_key] = optional_value
        teams_payload[team_name] = team_entry

    payload = {
        "year": int(target_year),
        "data_freshness": "BASELINE_PRESEASON",
        "note": (
            f"Preseason {target_year} comparison seed built from {actual_source_year} team ranking "
            "and target-year testing telemetry."
        ),
        "generated_at": generated_at,
        "last_updated": generated_at,
        "races_completed": 0,
        "directionality_source": "prior_year_ranking_seed",
        "directionality_last_updated": generated_at,
        "directionality_meta": {
            "seed_mode": "prior_year_ranking",
            "source_year": actual_source_year,
            "target_year": int(target_year),
            "target_testing_events": list(preseason_snapshot.event_names),
            "loaded_sessions": list(preseason_snapshot.loaded_sessions),
        },
        "teams": teams_payload,
    }
    validate_team_characteristics(payload, expected_year=target_year)
    return payload


def _coverage_penalty(
    team_payload: dict[str, Any],
    *,
    selected_lap_weight: float,
) -> float:
    """Return an uncertainty penalty for sparse or thin testing coverage."""
    testing_characteristics = team_payload.get("testing_characteristics")
    if not isinstance(testing_characteristics, dict):
        return 0.08

    sessions_used = int(_coerce_float(testing_characteristics.get("sessions_used")) or 0)
    circuits_observed = testing_characteristics.get("circuits_observed")
    circuit_count = len(circuits_observed) if isinstance(circuits_observed, list) else 0
    lap_weight = float(max(selected_lap_weight, 0.0))

    penalty = 0.0
    if sessions_used < 3:
        penalty += 0.03
    if circuit_count < 1:
        penalty += 0.03
    if lap_weight < 12.0:
        penalty += 0.02
    return penalty


def _extract_profile_metric(
    team_payload: dict[str, Any],
    *,
    profile_name: str,
    metric_name: str,
) -> float | None:
    """Extract one finite testing-profile metric when present."""
    profiles = team_payload.get("testing_characteristics_profiles")
    if not isinstance(profiles, dict):
        return None
    profile_payload = profiles.get(profile_name)
    if not isinstance(profile_payload, dict):
        return None
    return _coerce_float(profile_payload.get(metric_name))


def _compute_testing_signal_disagreement(team_payload: dict[str, Any]) -> float:
    """Estimate how noisy or program-dependent the testing signal looks.

    The goal is not to infer intent. It is simply to detect when the short-run,
    long-run, and speed-shape story disagree enough that the pace signal should
    be trusted less as a preseason ranking prior.
    """

    disagreement_components: list[float] = []

    short_overall = _extract_profile_metric(
        team_payload,
        profile_name="short_run",
        metric_name="overall_pace",
    )
    balanced_overall = _extract_profile_metric(
        team_payload,
        profile_name="balanced",
        metric_name="overall_pace",
    )
    long_overall = _extract_profile_metric(
        team_payload,
        profile_name="long_run",
        metric_name="overall_pace",
    )
    short_top_speed = _extract_profile_metric(
        team_payload,
        profile_name="short_run",
        metric_name="top_speed",
    )
    balanced_consistency = _extract_profile_metric(
        team_payload,
        profile_name="balanced",
        metric_name="consistency",
    )
    long_consistency = _extract_profile_metric(
        team_payload,
        profile_name="long_run",
        metric_name="consistency",
    )

    if short_overall is not None and balanced_overall is not None:
        disagreement_components.append(abs(short_overall - balanced_overall))
    if short_overall is not None and long_overall is not None:
        disagreement_components.append(abs(short_overall - long_overall))
    if short_top_speed is not None and short_overall is not None:
        disagreement_components.append(abs(short_top_speed - short_overall))
    if balanced_consistency is not None and long_consistency is not None:
        disagreement_components.append(abs(long_consistency - balanced_consistency))

    if not disagreement_components:
        return 0.5

    mean_disagreement = float(np.mean(disagreement_components))
    return float(np.clip(mean_disagreement / 0.14, 0.0, 1.0))


def _compute_pace_glimpse_confidence(
    team_payload: dict[str, Any],
    *,
    selected_lap_weight: float,
    signal_disagreement: float,
) -> float:
    """Estimate how much trust to place in the testing pace glimpse.

    Coverage matters more than any single pretty lap. Even with decent coverage,
    a noisy cross-profile picture should keep the preseason pace shift small.
    """

    testing_characteristics = team_payload.get("testing_characteristics")
    if not isinstance(testing_characteristics, dict):
        return 0.20

    sessions_used = int(_coerce_float(testing_characteristics.get("sessions_used")) or 0)
    circuits_observed = testing_characteristics.get("circuits_observed")
    circuit_count = len(circuits_observed) if isinstance(circuits_observed, list) else 0

    sessions_score = float(np.clip(sessions_used / 6.0, 0.0, 1.0))
    circuit_score = float(np.clip(circuit_count / 2.0, 0.0, 1.0))
    lap_weight_score = float(np.clip(max(selected_lap_weight, 0.0) / 24.0, 0.0, 1.0))

    confidence = (
        0.15
        + (0.35 * sessions_score)
        + (0.20 * circuit_score)
        + (0.30 * lap_weight_score)
        - (0.28 * float(np.clip(signal_disagreement, 0.0, 1.0)))
    )
    return float(np.clip(confidence, 0.20, 0.80))


def _shrink_testing_prediction_toward_neutral(
    *,
    normalized_prediction: float,
    pace_glimpse_confidence: float,
) -> float:
    """Shrink model ranking into a weak preseason pace prior around neutral."""

    centered_prediction = float(np.clip(normalized_prediction, 0.0, 1.0)) - 0.5
    shrunk_score = 0.5 + (
        centered_prediction
        * _PACE_GLIMPSE_SCALE
        * float(np.clip(pace_glimpse_confidence, 0.0, 1.0))
    )
    return float(np.clip(shrunk_score, 0.0, 1.0))


def _apply_bounded_testing_delta_to_champion_prior(
    *,
    champion_prior_overall_performance: float,
    testing_glimpse_overall_performance: float,
    pace_glimpse_confidence: float,
    signal_disagreement: float,
    coverage_penalty: float,
) -> tuple[float, float, float, float]:
    """Blend testing onto the champion prior as a small disagreement-aware delta."""

    champion_prior = float(np.clip(champion_prior_overall_performance, 0.0, 1.0))
    testing_glimpse = float(np.clip(testing_glimpse_overall_performance, 0.0, 1.0))
    bounded_gap = float(
        np.clip(
            testing_glimpse - champion_prior,
            -_MAX_TESTING_DELTA_FROM_CHAMPION,
            _MAX_TESTING_DELTA_FROM_CHAMPION,
        )
    )

    disagreement_factor = 1.0 - (0.90 * float(np.clip(signal_disagreement, 0.0, 1.0)) ** 1.35)
    normalized_coverage_penalty = min(max(float(coverage_penalty), 0.0), 0.08) / 0.08
    coverage_factor = 1.0 - (normalized_coverage_penalty * 0.40)
    confidence_factor = 0.12 + (0.58 * float(np.clip(pace_glimpse_confidence, 0.0, 1.0)))
    delta_multiplier = float(
        np.clip(
            confidence_factor * max(0.0, disagreement_factor) * max(0.0, coverage_factor),
            0.0,
            _MAX_TESTING_DELTA_MULTIPLIER,
        )
    )
    applied_delta = float(bounded_gap * delta_multiplier)
    conservative_overall_performance = float(np.clip(champion_prior + applied_delta, 0.0, 1.0))
    return conservative_overall_performance, applied_delta, bounded_gap, delta_multiplier


def _build_preseason_uncertainty(
    *,
    validation_rmse: float,
    coverage_penalty: float,
    signal_disagreement: float,
    pace_glimpse_confidence: float,
) -> float:
    """Combine validation, coverage, and signal quality into preseason uncertainty."""

    base_uncertainty = float(np.clip(0.24 + (validation_rmse * 0.45), 0.26, 0.34))
    confidence_drag = (1.0 - float(np.clip(pace_glimpse_confidence, 0.0, 1.0))) * 0.12
    uncertainty = (
        base_uncertainty
        + float(max(0.0, coverage_penalty))
        + (float(np.clip(signal_disagreement, 0.0, 1.0)) * 0.14)
        + confidence_drag
    )
    return float(np.clip(uncertainty, 0.26, 0.55))


def build_testing_model_team_payload(
    *,
    target_year: int,
    training_years: Iterable[int],
    generated_at: str | None = None,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Build one preseason team artifact from learned testing-to-pace mapping."""
    training_years = tuple(sorted(int(year) for year in training_years))
    if not training_years:
        raise ValueError("At least one training year is required for the testing team seed model.")

    dataset = build_training_dataset(training_years, cache_dir=cache_dir)
    weighted_dataset = apply_target_year_transfer_weights(dataset, target_year=target_year)
    training_year_relevance = summarize_target_year_transfer_weights(
        dataset,
        target_year=target_year,
    )
    unique_training_years = (
        dataset["season_year"].astype(int).unique().tolist() if not dataset.empty else []
    )
    if len(unique_training_years) >= 2:
        validation = run_leave_one_season_out_validation(
            dataset,
            use_regulation_transfer_weights=True,
        )
    else:
        validation = _fallback_validation_summary(weighted_dataset)
    model = fit_testing_team_seed_model(weighted_dataset)
    generated_at = generated_at or datetime.now(UTC).isoformat()

    preseason_events = discover_preseason_event_names(target_year, cache_dirs=(cache_dir,))
    preseason_snapshot = collect_team_testing_snapshot(
        target_year,
        event_names=preseason_events,
        session_candidates=_PRESEASON_SESSION_CANDIDATES,
        source_kind="preseason",
        cache_dir=cache_dir,
    )

    prediction_rows: list[dict[str, Any]] = []
    for team_name in preseason_snapshot.updated_teams:
        team_payload = preseason_snapshot.teams.get(team_name, {})
        if not isinstance(team_payload, dict):
            continue
        flattened = _flatten_team_features(
            team_payload,
            loaded_session_count=preseason_snapshot.team_session_counts.get(team_name, 0),
            selected_lap_weight=preseason_snapshot.selected_lap_weights.get(team_name, 0.0),
        )
        prediction_rows.append({"team_name": team_name, **flattened})

    prediction_frame = pd.DataFrame(prediction_rows)
    if prediction_frame.empty:
        raise ValueError(f"No preseason testing features could be built for {target_year}.")

    raw_predictions = model.predict_rows(prediction_frame)
    normalized_predictions = rank_normalize(
        {
            str(team_name): float(prediction)
            for team_name, prediction in zip(
                prediction_frame["team_name"].tolist(),
                raw_predictions.tolist(),
                strict=False,
            )
        },
        higher_is_better=True,
    )
    champion_prior_scores = _build_champion_prior_seed_map(
        source_year=target_year - 1,
        target_teams=prediction_frame["team_name"].tolist(),
    )
    champion_prior_ranks = rank_normalize(champion_prior_scores, higher_is_better=True)

    teams_payload: dict[str, Any] = {}
    team_diagnostics: dict[str, Any] = {}
    for team_name, raw_prediction in zip(
        prediction_frame["team_name"].tolist(),
        raw_predictions.tolist(),
        strict=False,
    ):
        team_payload = preseason_snapshot.teams.get(str(team_name), {})
        if not isinstance(team_payload, dict):
            team_payload = {}
        selected_lap_weight = preseason_snapshot.selected_lap_weights.get(str(team_name), 0.0)
        normalized_prediction = float(normalized_predictions[str(team_name)])
        coverage_penalty = _coverage_penalty(
            team_payload,
            selected_lap_weight=selected_lap_weight,
        )
        signal_disagreement = _compute_testing_signal_disagreement(team_payload)
        pace_glimpse_confidence = _compute_pace_glimpse_confidence(
            team_payload,
            selected_lap_weight=selected_lap_weight,
            signal_disagreement=signal_disagreement,
        )
        testing_glimpse_overall_performance = _shrink_testing_prediction_toward_neutral(
            normalized_prediction=normalized_prediction,
            pace_glimpse_confidence=pace_glimpse_confidence,
        )
        champion_prior_overall_performance = float(
            champion_prior_scores.get(str(team_name), _RANK_SEED_ANCHORS[-1])
        )
        (
            overall_performance,
            applied_testing_delta,
            bounded_testing_gap,
            delta_multiplier,
        ) = _apply_bounded_testing_delta_to_champion_prior(
            champion_prior_overall_performance=champion_prior_overall_performance,
            testing_glimpse_overall_performance=testing_glimpse_overall_performance,
            pace_glimpse_confidence=pace_glimpse_confidence,
            signal_disagreement=signal_disagreement,
            coverage_penalty=coverage_penalty,
        )
        uncertainty = _build_preseason_uncertainty(
            validation_rmse=validation.rmse,
            coverage_penalty=coverage_penalty,
            signal_disagreement=signal_disagreement,
            pace_glimpse_confidence=pace_glimpse_confidence,
        )
        team_entry: dict[str, Any] = {
            "overall_performance": round(overall_performance, 3),
            "preseason_overall_performance": round(overall_performance, 3),
            "uncertainty": round(uncertainty, 3),
            "note": (
                "Champion prior adjusted by a bounded testing delta "
                f"(raw model score {float(raw_prediction):.3f}; applied delta "
                f"{applied_testing_delta:+.3f})."
            ),
            "last_updated": generated_at,
            "races_completed": 0,
            "current_season_performance": [],
        }
        for optional_key in (
            "directionality",
            "testing_characteristics",
            "testing_characteristics_profiles",
            "compound_characteristics",
        ):
            optional_value = team_payload.get(optional_key)
            if optional_value is not None:
                team_entry[optional_key] = optional_value
        teams_payload[str(team_name)] = team_entry
        team_diagnostics[str(team_name)] = {
            "raw_model_score": round(float(raw_prediction), 4),
            "normalized_prediction": round(normalized_prediction, 4),
            "champion_prior_overall_performance": round(champion_prior_overall_performance, 4),
            "champion_prior_normalized_rank": round(
                float(champion_prior_ranks.get(str(team_name), 0.0)),
                4,
            ),
            "pace_glimpse_confidence": round(pace_glimpse_confidence, 4),
            "signal_disagreement": round(signal_disagreement, 4),
            "coverage_penalty": round(float(coverage_penalty), 4),
            "testing_glimpse_overall_performance": round(
                testing_glimpse_overall_performance,
                4,
            ),
            "bounded_gap_from_champion_prior": round(bounded_testing_gap, 4),
            "delta_multiplier": round(delta_multiplier, 4),
            "applied_testing_delta": round(applied_testing_delta, 4),
            "conservative_overall_performance": round(overall_performance, 4),
        }

    payload = {
        "year": int(target_year),
        "data_freshness": "BASELINE_PRESEASON",
        "note": (
            f"Preseason {target_year} team seed built as a bounded testing delta on top of the "
            f"champion prior, validated on seasons {list(training_years)} with regulation-aware "
            "transfer weights."
        ),
        "learning_note": (
            "Experimental reset-year team seed. Testing can only nudge the champion prior by a "
            "small, disagreement-aware amount, while training seasons are weighted by "
            "regulation-cycle similarity before uncertainty yields to real race evidence."
        ),
        "generated_at": generated_at,
        "last_updated": generated_at,
        "races_completed": 0,
        "directionality_source": "testing_team_seed_model",
        "directionality_last_updated": generated_at,
        "directionality_meta": {
            "seed_mode": "testing_model",
            "blend_strategy": "bounded_champion_delta",
            "champion_prior_source_year": int(target_year - 1),
            "target_year": int(target_year),
            "training_years": list(training_years),
            "regulation_reset_years": list(_REGULATION_RESET_YEARS),
            "target_testing_events": list(preseason_snapshot.event_names),
            "loaded_sessions": list(preseason_snapshot.loaded_sessions),
            "model_type": "RidgeCV",
            "feature_count": len(FEATURE_COLUMNS),
            "selected_alpha": float(model.regressor.alpha_),
            "training_year_relevance": training_year_relevance,
            "validation": {
                "mae": validation.mae,
                "rmse": validation.rmse,
                "season_summaries": validation.seasons,
            },
            "team_modeling_diagnostics": team_diagnostics,
        },
        "teams": teams_payload,
    }
    validate_team_characteristics(payload, expected_year=target_year)
    return payload


def write_validation_report(
    *,
    payload: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a compact JSON report for one generated testing-model payload."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "year": payload.get("year"),
        "generated_at": payload.get("generated_at"),
        "seed_mode": payload.get("directionality_meta", {}).get("seed_mode"),
        "directionality_meta": payload.get("directionality_meta", {}),
        "teams_ranked": sorted(
            (
                {
                    "team_name": team_name,
                    "overall_performance": team_payload.get("overall_performance"),
                    "uncertainty": team_payload.get("uncertainty"),
                }
                for team_name, team_payload in payload.get("teams", {}).items()
                if isinstance(team_payload, dict)
            ),
            key=lambda item: float(item.get("overall_performance") or 0.0),
            reverse=True,
        ),
    }
    output_file.write_text(json.dumps(report, indent=2))
    return output_file
