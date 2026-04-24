"""Learn clipped qualifying residual corrections from historical weekends."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.actual_results_fetcher import fetch_actual_session_results
from src.predictors.baseline.qualifying_simulation import build_deterministic_qualifying_ranking
from src.utils.backtesting import (
    NestedDictConfig,
    apply_config_overrides,
    get_races_for_year,
    load_config_dict,
)
from src.utils.weekend import get_weekend_type

logger = logging.getLogger(__name__)

NUMERIC_FEATURE_COLUMNS = (
    "team_strength",
    "track_suitability",
    "testing_short_run_modifier",
    "fp_blend_weight",
    "fp_driver_delta",
    "data_confidence_score",
    "skill",
    "quali_pace",
    "raw_quali_pace",
    "bayesian_skill_score",
    "bayesian_pace_blend_weight",
    "baseline_position",
    "baseline_score",
    "teammate_quali_gap",
    "teammate_skill_gap",
    "races_completed",
    "is_sprint",
)
CAT_FEATURE_COLUMNS = (
    "experience_tier",
    "weather",
    "data_source_mode",
    "weekend_format",
)


def default_qualifying_residual_artifact_path(*, data_root: str | Path = "data") -> Path:
    """Return the default artifact path for the qualifying residual model."""
    return (
        Path(data_root)
        / "processed"
        / "model_artifacts"
        / "qualifying_residual"
        / "qualifying_residual_model.pkl"
    )


def default_qualifying_residual_summary_path(*, data_root: str | Path = "data") -> Path:
    """Return the default metadata summary path for the qualifying residual model."""
    return (
        Path(data_root)
        / "processed"
        / "model_artifacts"
        / "qualifying_residual"
        / "qualifying_residual_model.summary.json"
    )


def _build_one_hot_encoder() -> OneHotEncoder:
    """Return a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    """Convert a scalar to a finite float with a stable default."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


@dataclass
class FittedQualifyingResidualModel:
    """Persisted sklearn pipeline plus metadata for clipped residual inference."""

    pipeline: Pipeline
    training_years: tuple[int, ...]
    clip_positions: float
    generated_at: str
    numeric_columns: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS
    categorical_columns: tuple[str, ...] = CAT_FEATURE_COLUMNS

    def predict_adjustments(self, rows: pd.DataFrame) -> np.ndarray:
        """Predict clipped residual adjustments in position units."""
        required_columns = list(self.numeric_columns) + list(self.categorical_columns)
        missing_columns = [column for column in required_columns if column not in rows.columns]
        if missing_columns:
            raise ValueError(
                f"Missing qualifying residual feature columns: {sorted(missing_columns)}"
            )
        predictions = np.asarray(self.pipeline.predict(rows.loc[:, required_columns]), dtype=float)
        return np.clip(predictions, -self.clip_positions, self.clip_positions)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable metadata summary."""
        return {
            "generated_at": self.generated_at,
            "training_years": list(self.training_years),
            "clip_positions": float(self.clip_positions),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
        }


def fit_qualifying_residual_model(
    dataset: pd.DataFrame,
    *,
    clip_positions: float = 2.0,
) -> FittedQualifyingResidualModel:
    """Fit the qualifying residual model on one driver-weekend dataset."""
    if dataset.empty:
        raise ValueError("Cannot fit qualifying residual model on an empty dataset.")

    required_columns = set(NUMERIC_FEATURE_COLUMNS).union(CAT_FEATURE_COLUMNS)
    required_columns.add("target_residual_positions")
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing qualifying residual columns: {sorted(missing_columns)}"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURE_COLUMNS),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _build_one_hot_encoder()),
                    ]
                ),
                list(CAT_FEATURE_COLUMNS),
            ),
        ]
    )
    regressor = ElasticNetCV(
        l1_ratio=[0.10, 0.30, 0.50, 0.80, 0.95],
        alphas=np.logspace(-3, 1, 25),
        cv=5,
        max_iter=20_000,
        random_state=42,
        selection="cyclic",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    feature_frame = dataset.loc[:, list(NUMERIC_FEATURE_COLUMNS) + list(CAT_FEATURE_COLUMNS)].copy()
    target = dataset["target_residual_positions"].astype(float).to_numpy()
    pipeline.fit(feature_frame, target)

    training_years = tuple(
        sorted(
            int(year_value) for year_value in dataset["season_year"].astype(int).unique().tolist()
        )
    )
    return FittedQualifyingResidualModel(
        pipeline=pipeline,
        training_years=training_years,
        clip_positions=float(max(0.0, clip_positions)),
        generated_at=datetime.now(UTC).isoformat(),
    )


def load_qualifying_residual_model(path: str | Path) -> FittedQualifyingResidualModel | None:
    """Load a fitted residual artifact from disk when present."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    try:
        with open(artifact_path, "rb") as handle:
            loaded = pickle.load(handle)
    except (OSError, pickle.PickleError):
        return None
    return loaded if isinstance(loaded, FittedQualifyingResidualModel) else None


def save_qualifying_residual_model(
    *,
    model: FittedQualifyingResidualModel,
    artifact_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Persist the fitted model artifact and optional JSON summary."""
    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_file, "wb") as handle:
        pickle.dump(model, handle)

    written_summary: Path | None = None
    if summary_path is not None:
        written_summary = Path(summary_path)
        written_summary.parent.mkdir(parents=True, exist_ok=True)
        written_summary.write_text(json.dumps(model.summary(), indent=2))

    return artifact_file, written_summary


def _resolve_weekend_format(*, year: int, race_name: str) -> str:
    """Return a stable weekend-format label."""
    try:
        return "sprint" if get_weekend_type(year, race_name) == "sprint" else "normal"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return "unknown"


def _build_teammate_gap_maps(
    all_drivers: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build teammate-relative pace and skill gaps for one qualifying driver list."""
    team_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in all_drivers:
        team_to_rows.setdefault(str(row["team"]), []).append(row)

    quali_gap_by_driver: dict[str, float] = {}
    skill_gap_by_driver: dict[str, float] = {}
    for team_rows in team_to_rows.values():
        for row in team_rows:
            driver_code = str(row["driver"])
            teammate_rows = [candidate for candidate in team_rows if candidate is not row]
            if not teammate_rows:
                quali_gap_by_driver[driver_code] = 0.0
                skill_gap_by_driver[driver_code] = 0.0
                continue

            teammate_quali_mean = float(
                np.mean(
                    [
                        _coerce_float(candidate.get("raw_quali_pace"), default=0.5)
                        for candidate in teammate_rows
                    ]
                )
            )
            teammate_skill_mean = float(
                np.mean(
                    [
                        _coerce_float(candidate.get("skill"), default=0.5)
                        for candidate in teammate_rows
                    ]
                )
            )
            quali_gap_by_driver[driver_code] = (
                _coerce_float(row.get("raw_quali_pace"), default=0.5) - teammate_quali_mean
            )
            skill_gap_by_driver[driver_code] = (
                _coerce_float(row.get("skill"), default=0.5) - teammate_skill_mean
            )

    return quali_gap_by_driver, skill_gap_by_driver


def build_feature_frame_from_context(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    weather: str,
    all_drivers: list[dict[str, Any]],
    is_sprint: bool,
    data_confidence_score: float,
    data_source_mode: str,
    fp_blend_weight: float,
    driver_fp_modifiers: dict[str, float] | None,
    baseline_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build one feature frame from prepared qualifying inputs.

    The predictor is expected to expose the same helper methods the runtime uses
    during live prediction, so the training and inference feature pipelines stay
    aligned.
    """
    baseline_by_driver = {
        str(row["driver"]): row
        for row in baseline_rows
        if isinstance(row, dict) and row.get("driver")
    }
    short_profile_weights = predictor._get_testing_profile_weights(  # noqa: SLF001
        "short_run",
        {
            "overall_pace": 0.55,
            "top_speed": 0.20,
            "medium_corner_performance": 0.15,
            "fast_corner_performance": 0.10,
        },
    )
    short_profile_scale = float(
        predictor.config.get("baseline_predictor.qualifying.testing_short_run_modifier_scale", 0.04)
    )
    races_completed = int(predictor._get_contextual_races_completed(race_name))  # noqa: SLF001
    teammate_quali_gap, teammate_skill_gap = _build_teammate_gap_maps(all_drivers)
    fp_driver_adjustments = driver_fp_modifiers or {}

    team_track_suitability: dict[str, float] = {}
    team_testing_modifier: dict[str, float] = {}
    for driver_info in all_drivers:
        team_name = str(driver_info["team"])
        if team_name in team_track_suitability:
            continue
        team_track_suitability[team_name] = float(
            predictor.calculate_track_suitability(team_name, race_name)
        )
        modifier, _has_profile = predictor._compute_testing_profile_modifier(  # noqa: SLF001
            team_name,
            "short_run",
            short_profile_weights,
            short_profile_scale,
        )
        team_testing_modifier[team_name] = float(modifier)

    rows: list[dict[str, Any]] = []
    for driver_info in all_drivers:
        driver_code = str(driver_info["driver"])
        baseline_row = baseline_by_driver.get(driver_code, {})
        rows.append(
            {
                "season_year": int(year),
                "race_name": race_name,
                "driver": driver_code,
                "team": str(driver_info["team"]),
                "team_strength": _coerce_float(driver_info.get("team_strength"), default=0.5),
                "track_suitability": team_track_suitability.get(str(driver_info["team"]), 0.5),
                "testing_short_run_modifier": team_testing_modifier.get(
                    str(driver_info["team"]), 0.0
                ),
                "fp_blend_weight": _coerce_float(fp_blend_weight),
                "fp_driver_delta": _coerce_float(
                    fp_driver_adjustments.get(driver_code), default=0.0
                ),
                "data_confidence_score": _coerce_float(data_confidence_score, default=0.0),
                "skill": _coerce_float(driver_info.get("skill"), default=0.5),
                "quali_pace": _coerce_float(driver_info.get("quali_pace"), default=0.5),
                "raw_quali_pace": _coerce_float(driver_info.get("raw_quali_pace"), default=0.5),
                "bayesian_skill_score": _coerce_float(
                    driver_info.get("bayesian_skill_score"), default=0.5
                ),
                "bayesian_pace_blend_weight": _coerce_float(
                    driver_info.get("bayesian_pace_blend_weight"),
                    default=0.0,
                ),
                "baseline_position": _coerce_float(
                    baseline_row.get("baseline_position"),
                    default=len(all_drivers),
                ),
                "baseline_score": _coerce_float(baseline_row.get("baseline_score"), default=0.0),
                "teammate_quali_gap": teammate_quali_gap.get(driver_code, 0.0),
                "teammate_skill_gap": teammate_skill_gap.get(driver_code, 0.0),
                "races_completed": float(races_completed),
                "is_sprint": float(1 if is_sprint else 0),
                "experience_tier": str(driver_info.get("experience_tier", "unknown")),
                "weather": str(weather).strip().lower() or "dry",
                "data_source_mode": str(data_source_mode).strip().lower() or "model_only",
                "weekend_format": _resolve_weekend_format(year=year, race_name=race_name),
            }
        )

    return pd.DataFrame(rows)


def build_qualifying_residual_dataset(
    years: list[int] | tuple[int, ...],
    *,
    max_races: int | None = None,
    config_path: str = "config/default.yaml",
    data_root: str | Path = "data",
    weather: str = "dry",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a historical driver-weekend dataset for residual training."""
    from src.persistence.artifact_store import ArtifactStore
    from src.predictors import Baseline2026Predictor

    base_config = load_config_dict(config_path)
    training_config = apply_config_overrides(
        base_config,
        {
            "baseline_predictor.qualifying.qualifying_residual_model.enabled": False,
            "baseline_predictor.race.race_residual_model.enabled": False,
            "baseline_predictor.conformal_calibration.enabled": False,
        },
    )

    dataset_rows: list[dict[str, Any]] = []
    data_root_path = Path(data_root)

    for raw_year in years:
        year = int(raw_year)
        predictor = Baseline2026Predictor(
            data_dir=str(data_root_path / "processed"),
            season_year=year,
            seed=seed,
            config=cast(Any, NestedDictConfig(training_config)),
            artifact_store=ArtifactStore(data_root=data_root_path),
        )
        reset_state = getattr(getattr(predictor, "calibration_system", None), "reset_state", None)
        if callable(reset_state):
            reset_state(season=year)

        for race_name in get_races_for_year(year=year, max_races=max_races):
            actual_results = fetch_actual_session_results(year, race_name, "Q")
            if not actual_results:
                continue

            try:
                prepared = predictor._prepare_qualifying_prediction_inputs(  # noqa: SLF001
                    year=year,
                    race_name=race_name,
                    qualifying_stage="main",
                    practice_signal_mode="auto",
                    checkpoint_session_name=None,
                    weather=weather,
                )
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Skipping qualifying residual dataset row for %s %s: %s", year, race_name, exc
                )
                continue

            baseline_rows = build_deterministic_qualifying_ranking(
                all_drivers=prepared["all_drivers"],
                is_sprint=bool(prepared["is_sprint"]),
                has_practice_data=bool(prepared["has_practice_like_data"]),
                has_testing_fallback_data=bool(prepared["testing_fallback_used"]),
                cfg=predictor.config,
                weather=weather,
            )
            feature_frame = build_feature_frame_from_context(
                predictor=predictor,
                year=year,
                race_name=race_name,
                weather=weather,
                all_drivers=prepared["all_drivers"],
                is_sprint=bool(prepared["is_sprint"]),
                data_confidence_score=float(prepared["data_confidence_score"]),
                data_source_mode=str(prepared["data_source_mode"]),
                fp_blend_weight=float(prepared["effective_fp_blend_weight"]),
                driver_fp_modifiers=prepared.get("driver_fp_modifiers"),
                baseline_rows=baseline_rows,
            )
            actual_by_driver = {
                str(row["driver"]): int(row["position"])
                for row in actual_results
                if isinstance(row, dict) and row.get("driver") and row.get("position") is not None
            }
            for row in feature_frame.to_dict(orient="records"):
                driver_code = str(row["driver"])
                actual_position = actual_by_driver.get(driver_code)
                if actual_position is None:
                    continue
                baseline_position = int(
                    round(
                        _coerce_float(row.get("baseline_position"), default=len(actual_by_driver))
                    )
                )
                row["actual_position"] = int(actual_position)
                row["target_residual_positions"] = float(baseline_position - actual_position)
                dataset_rows.append(row)

    return pd.DataFrame(dataset_rows)


def build_qualifying_residual_model(
    *,
    years: list[int] | tuple[int, ...],
    artifact_path: str | Path,
    summary_path: str | Path | None = None,
    clip_positions: float = 2.0,
    max_races: int | None = None,
    config_path: str = "config/default.yaml",
    data_root: str | Path = "data",
    weather: str = "dry",
    seed: int = 42,
) -> tuple[FittedQualifyingResidualModel, pd.DataFrame]:
    """Build and persist a fitted qualifying residual artifact from historical years."""
    dataset = build_qualifying_residual_dataset(
        years=years,
        max_races=max_races,
        config_path=config_path,
        data_root=data_root,
        weather=weather,
        seed=seed,
    )
    model = fit_qualifying_residual_model(dataset, clip_positions=clip_positions)
    save_qualifying_residual_model(
        model=model,
        artifact_path=artifact_path,
        summary_path=summary_path,
    )
    return model, dataset


def apply_qualifying_residual_model(
    *,
    model: FittedQualifyingResidualModel,
    feature_frame: pd.DataFrame,
    all_drivers: list[dict[str, Any]],
) -> dict[str, float]:
    """Apply a fitted residual model to driver rows in place."""
    adjustments = model.predict_adjustments(feature_frame)
    adjustments_by_driver: dict[str, float] = {}
    for driver_info, predicted_adjustment in zip(
        all_drivers,
        adjustments.tolist(),
        strict=False,
    ):
        driver_code = str(driver_info["driver"])
        clipped_adjustment = float(
            np.clip(predicted_adjustment, -model.clip_positions, model.clip_positions)
        )
        driver_info["qualifying_residual_adjustment"] = clipped_adjustment
        adjustments_by_driver[driver_code] = clipped_adjustment
    return adjustments_by_driver


def summarize_qualifying_residual_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """Return compact dataset diagnostics for reporting scripts."""
    if dataset.empty:
        return {"rows": 0, "races": 0, "years": []}

    return {
        "rows": int(len(dataset)),
        "races": int(dataset[["season_year", "race_name"]].drop_duplicates().shape[0]),
        "years": sorted(int(year) for year in dataset["season_year"].astype(int).unique().tolist()),
        "mean_absolute_target": float(
            np.mean(np.abs(dataset["target_residual_positions"].astype(float).to_numpy()))
        ),
    }
