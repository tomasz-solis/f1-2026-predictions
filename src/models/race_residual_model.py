"""Learn clipped race-pace corrections from historical qualifying-to-race deltas."""

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
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.actual_results_fetcher import (
    fetch_actual_session_results,
    fetch_actual_starting_grid,
)
from src.models.conformal_calibration import resolve_race_data_regime
from src.types.prediction_types import DriverRaceInfo, QualifyingGridEntry
from src.utils.backtesting import (
    NestedDictConfig,
    apply_config_overrides,
    get_races_for_year,
    load_config_dict,
)
from src.utils.weekend import get_weekend_type

logger = logging.getLogger(__name__)

NUMERIC_FEATURE_COLUMNS = (
    "grid_position",
    "team_strength",
    "race_advantage",
    "skill",
    "overtaking_skill",
    "track_overtaking_difficulty",
    "tire_stress_score",
    "long_run_modifier",
    "input_confidence",
    "mean_grid_confidence",
    "is_sprint",
)
CAT_FEATURE_COLUMNS = (
    "weather",
    "grid_source_mode",
    "weekend_format",
    "data_regime",
)


def default_race_residual_artifact_path(*, data_root: str | Path = "data") -> Path:
    """Return the default artifact path for the race residual model."""
    return (
        Path(data_root)
        / "processed"
        / "model_artifacts"
        / "race_residual"
        / "race_residual_model.pkl"
    )


def default_race_residual_summary_path(*, data_root: str | Path = "data") -> Path:
    """Return the default summary path for the race residual model."""
    return (
        Path(data_root)
        / "processed"
        / "model_artifacts"
        / "race_residual"
        / "race_residual_model.summary.json"
    )


def _build_one_hot_encoder() -> OneHotEncoder:
    """Return a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    """Return one finite float with a stable default."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


@dataclass
class FittedRaceResidualModel:
    """Persisted sklearn pipeline plus clipped positions-gained metadata."""

    pipeline: Pipeline
    training_years: tuple[int, ...]
    clip_positions_gained: float
    generated_at: str
    numeric_columns: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS
    categorical_columns: tuple[str, ...] = CAT_FEATURE_COLUMNS

    def predict_positions_gained(self, rows: pd.DataFrame) -> np.ndarray:
        """Predict clipped positions-gained adjustments."""
        required_columns = list(self.numeric_columns) + list(self.categorical_columns)
        missing_columns = [column for column in required_columns if column not in rows.columns]
        if missing_columns:
            raise ValueError(f"Missing race residual feature columns: {sorted(missing_columns)}")
        predictions = np.asarray(self.pipeline.predict(rows.loc[:, required_columns]), dtype=float)
        return np.clip(
            predictions,
            -self.clip_positions_gained,
            self.clip_positions_gained,
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON summary."""
        return {
            "generated_at": self.generated_at,
            "training_years": list(self.training_years),
            "clip_positions_gained": float(self.clip_positions_gained),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
        }


def fit_race_residual_model(
    dataset: pd.DataFrame,
    *,
    clip_positions_gained: float = 2.5,
) -> FittedRaceResidualModel:
    """Fit a clipped RidgeCV race residual model on one dataset."""
    if dataset.empty:
        raise ValueError("Cannot fit race residual model on an empty dataset.")

    required_columns = set(NUMERIC_FEATURE_COLUMNS).union(CAT_FEATURE_COLUMNS)
    required_columns.add("target_positions_gained")
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing race residual columns: {sorted(missing_columns)}")

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
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    feature_frame = dataset.loc[:, list(NUMERIC_FEATURE_COLUMNS) + list(CAT_FEATURE_COLUMNS)].copy()
    target = dataset["target_positions_gained"].astype(float).to_numpy()
    pipeline.fit(feature_frame, target)

    training_years = tuple(
        sorted(
            int(year_value) for year_value in dataset["season_year"].astype(int).unique().tolist()
        )
    )
    return FittedRaceResidualModel(
        pipeline=pipeline,
        training_years=training_years,
        clip_positions_gained=float(max(0.0, clip_positions_gained)),
        generated_at=datetime.now(UTC).isoformat(),
    )


def load_race_residual_model(path: str | Path) -> FittedRaceResidualModel | None:
    """Load a persisted race residual model when present."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    try:
        with open(artifact_path, "rb") as handle:
            loaded = pickle.load(handle)
    except (OSError, pickle.PickleError):
        return None
    return loaded if isinstance(loaded, FittedRaceResidualModel) else None


def save_race_residual_model(
    *,
    model: FittedRaceResidualModel,
    artifact_path: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Persist the fitted race residual model and optional summary."""
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


def build_feature_frame_from_context(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    weather: str,
    qualifying_grid: list[QualifyingGridEntry],
    driver_info_map: dict[str, DriverRaceInfo],
    input_confidence: float | None,
    mean_grid_confidence: float | None,
    grid_source_mode: str,
    is_sprint: bool,
) -> pd.DataFrame:
    """Build one race residual feature frame from race-prep inputs."""
    track_payload = getattr(predictor, "tracks", {}).get(race_name, {})
    track_overtaking = _coerce_float(track_payload.get("overtaking_difficulty"), default=0.5)
    long_profile_weights = predictor._get_testing_profile_weights(  # noqa: SLF001
        "long_run",
        {
            "overall_pace": 0.50,
            "tire_deg_performance": 0.35,
            "consistency": 0.15,
        },
    )
    long_profile_scale = float(
        predictor.config.get("baseline_predictor.race.testing_long_run_modifier_scale", 0.05)
    )
    try:
        from src.data.track_data_loader import get_tire_stress_score

        tire_stress_score = _coerce_float(get_tire_stress_score(race_name, year=year), default=3.0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        tire_stress_score = 3.0

    rows: list[dict[str, Any]] = []
    grid_rows_by_driver = {
        str(row["driver"]): row
        for row in qualifying_grid
        if isinstance(row, dict) and row.get("driver")
    }
    data_regime = resolve_race_data_regime(
        input_confidence=input_confidence,
        mean_grid_confidence=mean_grid_confidence,
    )
    for driver_code, info in driver_info_map.items():
        long_modifier, _has_profile = predictor._compute_testing_profile_modifier(  # noqa: SLF001
            str(info["team"]),
            "long_run",
            long_profile_weights,
            long_profile_scale,
        )
        grid_row: QualifyingGridEntry = grid_rows_by_driver.get(
            driver_code,
            {"driver": driver_code, "team": str(info["team"]), "position": int(info["grid_pos"])},
        )
        rows.append(
            {
                "season_year": int(year),
                "race_name": race_name,
                "driver": driver_code,
                "team": str(info["team"]),
                "grid_position": _coerce_float(info.get("grid_pos"), default=22.0),
                "team_strength": _coerce_float(info.get("team_strength"), default=0.5),
                "race_advantage": _coerce_float(info.get("race_advantage"), default=0.0),
                "skill": _coerce_float(info.get("skill"), default=0.5),
                "overtaking_skill": _coerce_float(info.get("overtaking_skill"), default=0.5),
                "track_overtaking_difficulty": track_overtaking,
                "tire_stress_score": tire_stress_score,
                "long_run_modifier": float(long_modifier),
                "input_confidence": _coerce_float(input_confidence, default=0.0),
                "mean_grid_confidence": _coerce_float(
                    mean_grid_confidence,
                    default=_coerce_float(grid_row.get("confidence"), default=0.0),
                ),
                "is_sprint": float(1 if is_sprint else 0),
                "weather": str(weather).strip().lower() or "dry",
                "grid_source_mode": str(grid_source_mode).strip().lower() or "predicted",
                "weekend_format": _resolve_weekend_format(year=year, race_name=race_name),
                "data_regime": data_regime,
            }
        )

    return pd.DataFrame(rows)


def build_race_residual_dataset(
    years: list[int] | tuple[int, ...],
    *,
    max_races: int | None = None,
    config_path: str = "config/default.yaml",
    data_root: str | Path = "data",
    weather: str = "dry",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a historical race residual dataset from qualifying-to-race deltas."""
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
            # The label is positions gained from the start, so the start has to be the
            # grid the cars lined up on. Qualifying classification carries no penalties:
            # in 2026 it disagrees with the real grid on 15% of driver-races, and on 20
            # of 22 rows at Spa, which is exactly where the residual signal is largest.
            starting_grid = fetch_actual_starting_grid(year, race_name)
            race_actual = fetch_actual_session_results(year, race_name, "R")
            if not starting_grid or not race_actual:
                continue

            try:
                driver_info_map, _profile_count = predictor._prepare_driver_info_with_compounds(  # noqa: SLF001
                    starting_grid,
                    race_name,
                )
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Skipping race residual dataset row for %s %s: %s", year, race_name, exc
                )
                continue

            feature_frame = build_feature_frame_from_context(
                predictor=predictor,
                year=year,
                race_name=race_name,
                weather=weather,
                qualifying_grid=starting_grid,
                driver_info_map=driver_info_map,
                input_confidence=1.0,
                mean_grid_confidence=1.0,
                grid_source_mode="actual",
                is_sprint=False,
            )
            actual_by_driver = {
                str(row["driver"]): int(row["position"])
                for row in race_actual
                if isinstance(row, dict) and row.get("driver") and row.get("position") is not None
            }
            for row in feature_frame.to_dict(orient="records"):
                driver_code = str(row["driver"])
                actual_position = actual_by_driver.get(driver_code)
                if actual_position is None:
                    continue
                grid_position = int(round(_coerce_float(row.get("grid_position"), default=22.0)))
                row["actual_position"] = int(actual_position)
                row["target_positions_gained"] = float(grid_position - actual_position)
                dataset_rows.append(row)

    return pd.DataFrame(dataset_rows)


def build_race_residual_model(
    *,
    years: list[int] | tuple[int, ...],
    artifact_path: str | Path,
    summary_path: str | Path | None = None,
    clip_positions_gained: float = 2.5,
    max_races: int | None = None,
    config_path: str = "config/default.yaml",
    data_root: str | Path = "data",
    weather: str = "dry",
    seed: int = 42,
) -> tuple[FittedRaceResidualModel, pd.DataFrame]:
    """Build and persist a fitted race residual artifact from historical years."""
    dataset = build_race_residual_dataset(
        years=years,
        max_races=max_races,
        config_path=config_path,
        data_root=data_root,
        weather=weather,
        seed=seed,
    )
    model = fit_race_residual_model(dataset, clip_positions_gained=clip_positions_gained)
    save_race_residual_model(
        model=model,
        artifact_path=artifact_path,
        summary_path=summary_path,
    )
    return model, dataset


def apply_race_residual_model(
    *,
    model: FittedRaceResidualModel,
    feature_frame: pd.DataFrame,
    driver_info_map: dict[str, DriverRaceInfo],
    positions_to_race_advantage_scale: float,
) -> dict[str, float]:
    """Apply a fitted race residual model in place as a race-advantage correction."""
    adjustments = model.predict_positions_gained(feature_frame)
    adjustments_by_driver: dict[str, float] = {}
    for row, predicted_gain in zip(
        feature_frame.to_dict(orient="records"),
        adjustments.tolist(),
        strict=False,
    ):
        driver_code = str(row["driver"])
        info = driver_info_map.get(driver_code)
        if info is None:
            continue
        clipped_gain = float(
            np.clip(predicted_gain, -model.clip_positions_gained, model.clip_positions_gained)
        )
        info["race_residual_adjustment"] = clipped_gain
        info["race_advantage"] = float(
            np.clip(
                _coerce_float(info.get("race_advantage"), default=0.0)
                + (clipped_gain * positions_to_race_advantage_scale),
                -1.0,
                1.0,
            )
        )
        adjustments_by_driver[driver_code] = clipped_gain
    return adjustments_by_driver


def summarize_race_residual_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    """Return compact dataset diagnostics for reporting."""
    if dataset.empty:
        return {"rows": 0, "races": 0, "years": []}

    return {
        "rows": int(len(dataset)),
        "races": int(dataset[["season_year", "race_name"]].drop_duplicates().shape[0]),
        "years": sorted(int(year) for year in dataset["season_year"].astype(int).unique().tolist()),
        "mean_absolute_target": float(
            np.mean(np.abs(dataset["target_positions_gained"].astype(float).to_numpy()))
        ),
    }
