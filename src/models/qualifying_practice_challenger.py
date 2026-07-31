"""Interpretable Bradley--Terry / Plackett--Luce qualifying challenger.

This module is deliberately independent from the champion predictor.  It fits
pairwise utilities from versioned practice-evidence rows and emits coherent grid
permutations.  Runtime integration must fail closed when no fitted artifact is
available.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_SCHEMA_VERSION = 2
EVENT_RELATIVE_FEATURE_TRANSFORM = "event_centered_seconds_v1"
EVENT_LEVEL_PACE_FEATURES = frozenset(
    {
        "best_adjusted_lap_s",
        "best2_mean_s",
        "adjusted_q20_s",
        "theoretical_lap_s",
    }
)
DEFAULT_FEATURE_COLUMNS = (
    "prior_utility",
    "best_adjusted_lap_s",
    "best2_mean_s",
    "adjusted_q20_s",
    "theoretical_lap_s",
    "execution_loss_s",
    "consistency_mad_s",
    "session_improvement_s",
    "teammate_gap_s",
    "compound_adjustment_se_s",
    "measurement_se_s",
    "clean_lap_count",
    "quali_run_count",
    "evidence_session_count",
    "direct_soft_flag",
    "evidence_quality_score",
)


@dataclass(frozen=True)
class FittedQualifyingPracticeModel:
    """JSON-serializable pairwise utility model for one checkpoint."""

    checkpoint: str
    feature_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    feature_medians: tuple[float, ...]
    feature_scales: tuple[float, ...]
    temperature: float
    training_events: int
    generated_at: str
    feature_transform: str = EVENT_RELATIVE_FEATURE_TRANSFORM
    schema_version: int = MODEL_SCHEMA_VERSION

    def utility_values(self, rows: pd.DataFrame) -> np.ndarray:
        """Return aligned latent utilities, retaining duplicate driver rows."""

        matrix = _feature_matrix(
            rows,
            feature_columns=self.feature_columns,
            medians=np.asarray(self.feature_medians, dtype=float),
            scales=np.asarray(self.feature_scales, dtype=float),
        )
        return matrix @ np.asarray(self.coefficients, dtype=float)

    def utilities(self, rows: pd.DataFrame) -> dict[str, float]:
        """Return one latent utility per driver; higher means faster."""

        if "driver" not in rows.columns:
            raise ValueError("Qualifying practice rows require a driver column")
        values = self.utility_values(rows)
        return {
            str(driver): float(utility)
            for driver, utility in zip(rows["driver"], values, strict=True)
        }

    def to_dict(self) -> dict[str, Any]:
        """Return stable artifact data without an executable pickle."""

        return {
            "artifact_type": "qualifying_practice_model",
            "schema_version": int(self.schema_version),
            "checkpoint": self.checkpoint,
            "feature_columns": list(self.feature_columns),
            "coefficients": list(self.coefficients),
            "feature_medians": list(self.feature_medians),
            "feature_scales": list(self.feature_scales),
            "temperature": float(self.temperature),
            "training_events": int(self.training_events),
            "generated_at": self.generated_at,
            "feature_transform": self.feature_transform,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FittedQualifyingPracticeModel:
        """Validate and construct a model from JSON artifact data."""

        if payload.get("artifact_type") != "qualifying_practice_model":
            raise ValueError("Not a qualifying_practice_model artifact")
        if int(payload.get("schema_version", -1)) != MODEL_SCHEMA_VERSION:
            raise ValueError("Unsupported qualifying practice model schema")
        feature_transform = str(payload.get("feature_transform", "")).strip()
        if feature_transform != EVENT_RELATIVE_FEATURE_TRANSFORM:
            raise ValueError("Unsupported qualifying practice feature transform")
        columns = tuple(str(value) for value in payload["feature_columns"])
        coefficients = tuple(float(value) for value in payload["coefficients"])
        medians = tuple(float(value) for value in payload["feature_medians"])
        scales = tuple(float(value) for value in payload["feature_scales"])
        if not columns or not (len(columns) == len(coefficients) == len(medians) == len(scales)):
            raise ValueError("Qualifying practice model arrays must have equal non-zero lengths")
        if any(not np.isfinite(value) for value in (*coefficients, *medians, *scales)):
            raise ValueError("Qualifying practice model arrays must be finite")
        if any(value <= 0 for value in scales):
            raise ValueError("Qualifying practice feature scales must be positive")
        temperature = float(payload["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Qualifying practice temperature must be positive")
        return cls(
            checkpoint=str(payload["checkpoint"]).strip().upper(),
            feature_columns=columns,
            coefficients=coefficients,
            feature_medians=medians,
            feature_scales=scales,
            temperature=temperature,
            training_events=int(payload["training_events"]),
            generated_at=str(payload["generated_at"]),
            feature_transform=feature_transform,
        )


def fit_bradley_terry_model(
    dataset: pd.DataFrame,
    *,
    checkpoint: str,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    regularization_c: float = 1.0,
    temperature: float = 1.0,
) -> FittedQualifyingPracticeModel:
    """Fit a symmetric no-intercept pairwise model with equal event weight."""

    required = {"event_id", "driver", "actual_position", *feature_columns}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Missing qualifying practice training columns: {missing}")
    if dataset.empty:
        raise ValueError("Cannot fit qualifying practice model on an empty dataset")

    training = dataset.reset_index(drop=True)
    numeric = _event_relative_numeric_features(
        training,
        feature_columns=feature_columns,
    )
    medians = numeric.median(axis=0).fillna(0.0).to_numpy(dtype=float)
    imputed = numeric.fillna(pd.Series(medians, index=feature_columns))
    scales = imputed.std(axis=0, ddof=0).to_numpy(dtype=float)
    scales[~np.isfinite(scales) | (scales <= 1e-9)] = 1.0
    standardized = (imputed.to_numpy(dtype=float) - medians) / scales

    pair_x: list[np.ndarray] = []
    pair_y: list[int] = []
    pair_weight: list[float] = []
    for _event_id, indexes in training.groupby("event_id", sort=False).groups.items():
        event_indexes = list(indexes)
        unordered_pairs = len(event_indexes) * (len(event_indexes) - 1) // 2
        if unordered_pairs <= 0:
            continue
        weight = 1.0 / (2.0 * unordered_pairs)
        for left_offset, left_index in enumerate(event_indexes):
            for right_index in event_indexes[left_offset + 1 :]:
                left_pos = float(training.loc[left_index, "actual_position"])
                right_pos = float(training.loc[right_index, "actual_position"])
                if left_pos == right_pos:
                    continue
                difference = standardized[int(left_index)] - standardized[int(right_index)]
                left_won = int(left_pos < right_pos)
                pair_x.extend((difference, -difference))
                pair_y.extend((left_won, 1 - left_won))
                pair_weight.extend((weight, weight))

    if not pair_x or len(set(pair_y)) < 2:
        raise ValueError("Qualifying practice training data has no usable ordered pairs")

    classifier = LogisticRegression(
        fit_intercept=False,
        C=max(1e-6, float(regularization_c)),
        solver="lbfgs",
        max_iter=10_000,
        random_state=42,
    )
    classifier.fit(
        np.vstack(pair_x),
        np.asarray(pair_y, dtype=int),
        sample_weight=np.asarray(pair_weight, dtype=float),
    )
    resolved_temperature = float(temperature)
    if not np.isfinite(resolved_temperature) or resolved_temperature <= 0:
        raise ValueError("temperature must be positive")
    return FittedQualifyingPracticeModel(
        checkpoint=str(checkpoint).strip().upper(),
        feature_columns=tuple(feature_columns),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        feature_medians=tuple(float(value) for value in medians),
        feature_scales=tuple(float(value) for value in scales),
        temperature=resolved_temperature,
        training_events=int(training["event_id"].nunique()),
        generated_at=datetime.now(UTC).isoformat(),
        feature_transform=EVENT_RELATIVE_FEATURE_TRANSFORM,
    )


def calibrate_temperature(
    *,
    utility_differences: np.ndarray,
    outcomes: np.ndarray,
    candidates: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
) -> float:
    """Choose temperature by validation log loss, never training-set accuracy."""

    differences = np.asarray(utility_differences, dtype=float)
    labels = np.asarray(outcomes, dtype=float)
    if differences.shape != labels.shape or differences.size == 0:
        raise ValueError("Temperature calibration requires aligned non-empty arrays")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("Temperature calibration outcomes must be in [0, 1]")

    best_temperature: float | None = None
    best_loss = float("inf")
    for raw_temperature in candidates:
        temperature = float(raw_temperature)
        if temperature <= 0:
            continue
        probabilities = _sigmoid(differences / temperature)
        probabilities = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
        loss = float(
            -np.mean(
                (labels * np.log(probabilities)) + ((1.0 - labels) * np.log(1.0 - probabilities))
            )
        )
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    if best_temperature is None:
        raise ValueError("Temperature candidates must include a positive value")
    return best_temperature


def simulate_plackett_luce_grids(
    *,
    utilities: Mapping[str, float],
    n_simulations: int,
    rng: np.random.Generator,
    temperature: float,
    utility_sigma_by_driver: Mapping[str, float] | None = None,
    utility_candidates_by_driver: Mapping[str, Sequence[float] | np.ndarray] | None = None,
) -> tuple[dict[str, list[int]], list[list[str]]]:
    """Sample coherent grids and aligned marginal position records.

    When research run-level candidates are supplied, one candidate utility is
    bootstrapped independently for each driver and simulation before measurement and
    checkpoint-calibrated execution noise are added.  Drivers without candidates keep
    their central utility.
    """

    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    if not utilities:
        raise ValueError("utilities must not be empty")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    drivers = sorted(utilities)
    sigma_map = utility_sigma_by_driver or {}
    unknown_candidate_drivers = sorted(set(utility_candidates_by_driver or {}).difference(drivers))
    if unknown_candidate_drivers:
        raise ValueError(f"Utility candidates contain unknown drivers: {unknown_candidate_drivers}")
    candidate_map: dict[str, np.ndarray] = {}
    for driver, raw_candidates in (utility_candidates_by_driver or {}).items():
        candidates = np.asarray(tuple(raw_candidates), dtype=float)
        if candidates.ndim != 1:
            raise ValueError(f"Utility candidates for {driver} must be one-dimensional")
        if np.any(~np.isfinite(candidates)):
            raise ValueError(f"Utility candidates for {driver} must be finite")
        if candidates.size:
            candidate_map[str(driver)] = np.sort(candidates)
    position_records: dict[str, list[int]] = {driver: [] for driver in drivers}
    scenarios: list[list[str]] = []
    for _ in range(n_simulations):
        sampled: list[tuple[str, float]] = []
        for driver in drivers:
            driver_candidates = candidate_map.get(driver)
            base_utility = (
                float(driver_candidates[int(rng.integers(0, len(driver_candidates)))])
                if driver_candidates is not None
                else float(utilities[driver])
            )
            sigma = max(0.0, float(sigma_map.get(driver, 0.0)))
            evidence_noise = float(rng.normal(0.0, sigma)) if sigma > 0 else 0.0
            execution_noise = float(rng.gumbel(0.0, temperature))
            sampled.append((driver, base_utility + evidence_noise + execution_noise))
        ordered = [driver for driver, _ in sorted(sampled, key=lambda item: (-item[1], item[0]))]
        scenarios.append(ordered)
        for position, driver in enumerate(ordered, start=1):
            position_records[driver].append(position)
    return position_records, scenarios


def save_qualifying_practice_model(
    model: FittedQualifyingPracticeModel,
    path: str | Path,
) -> Path:
    """Write a deterministic JSON model artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n")
    return destination


def load_qualifying_practice_model(
    path: str | Path,
) -> FittedQualifyingPracticeModel | None:
    """Load a model artifact, returning None only when it does not exist."""

    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Qualifying practice model artifact must be an object")
    return FittedQualifyingPracticeModel.from_dict(payload)


def _feature_matrix(
    rows: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    missing = sorted(set(feature_columns).difference(rows.columns))
    if missing:
        raise ValueError(f"Missing qualifying practice inference columns: {missing}")
    numeric = _event_relative_numeric_features(rows, feature_columns=feature_columns)
    imputed = numeric.fillna(pd.Series(medians, index=feature_columns))
    return (imputed.to_numpy(dtype=float) - medians) / scales


def _event_relative_numeric_features(
    rows: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return circuit-safe features with lap levels centred inside each event.

    Absolute lap time is mostly circuit length, not qualifying potential.  The
    Bradley--Terry model therefore consumes each driver's delta from the event
    median for level-like pace features.  Missing drivers remain missing here and
    are imputed to the fitted neutral median later, rather than to a global raw
    lap time from a different circuit.
    """

    numeric = rows.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    level_features = [
        feature for feature in feature_columns if feature in EVENT_LEVEL_PACE_FEATURES
    ]
    if not level_features:
        return numeric

    if "event_id" in rows.columns:
        event_keys = rows["event_id"].astype(str)
        for feature in level_features:
            event_median = numeric[feature].groupby(event_keys, sort=False).transform("median")
            numeric[feature] = numeric[feature] - event_median
    else:
        for feature in level_features:
            non_missing = numeric[feature].dropna()
            if not non_missing.empty:
                numeric[feature] = numeric[feature] - float(non_missing.median())
    return numeric


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
