"""Race prediction method for Baseline2026Predictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_non_competitive_weather_features,
    resolve_race_distance_laps,
    resolve_track_temperature_c,
    resolve_track_temperature_profile,
)
from src.predictors.baseline.early_season_uncertainty import (
    resolve_effective_learning_min_samples,
)
from src.simulation.pit_strategy import generate_pit_strategy
from src.types.prediction_types import DriverRaceInfo, QualifyingGridEntry
from src.utils import config_loader
from src.utils.grid_validation import validate_qualifying_grid
from src.utils.lap_by_lap_simulator import (
    aggregate_simulation_results,
    simulate_race_lap_by_lap,
)
from src.utils.prediction_context import PredictionContext, activate_prediction_runtime
from src.utils.validation_helpers import validate_enum, validate_positive_int

from .race_simulation import RaceSimulationDeps, predict_race_core

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePredictionMixin:
    """Race prediction method implementation for Baseline2026Predictor."""

    if TYPE_CHECKING:
        calibration_system: Any
        config: Any
        season_year: int
        seed: int
        year: int

        def _load_race_params(self) -> dict: ...

        def _prepare_driver_info_with_compounds(
            self,
            qualifying_grid: list[QualifyingGridEntry],
            race_name: str | None,
        ) -> tuple[dict[str, Any], int]: ...

    @staticmethod
    def _enforce_non_increasing(values: list[float]) -> list[float]:
        """Apply isotonic-style smoothing so sequence is non-increasing."""
        if not values:
            return values

        # Pool adjacent violators for monotone decreasing fit.
        blocks: list[list[float]] = []  # [start, end, avg, count]
        for idx, value in enumerate(values):
            blocks.append([float(idx), float(idx), float(value), 1.0])
            while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
                right = blocks.pop()
                left = blocks.pop()
                merged_count = left[3] + right[3]
                merged_avg = ((left[2] * left[3]) + (right[2] * right[3])) / merged_count
                blocks.append([left[0], right[1], merged_avg, merged_count])

        smoothed = [0.0 for _ in values]
        for start, end, avg, _ in blocks:
            for idx in range(int(start), int(end) + 1):
                smoothed[idx] = float(avg)
        return smoothed

    def _get_learned_position_adjustment(
        self,
        *,
        team: str,
        driver: str,
        teammates: list[str],
        session: str = "race",
        races_completed: int | None = None,
    ) -> float:
        """Return learned position adjustment from systematic calibration state."""
        calibration_system = getattr(self, "calibration_system", None)
        if calibration_system is None:
            return 0.0

        getter = getattr(calibration_system, "get_combined_position_adjustment", None)
        if not callable(getter):
            return 0.0

        cfg = getattr(self, "config", config_loader)
        configured_min_samples = int(
            cfg.get(
                "learning.min_samples",
                cfg.get("baseline_predictor.learning.min_samples", 1),
            )
        )
        min_samples = resolve_effective_learning_min_samples(
            configured_min_samples=configured_min_samples,
            races_completed=races_completed,
        )
        driver_error_scale = float(
            cfg.get(
                "learning.driver_error_scale",
                cfg.get("baseline_predictor.learning.driver_error_scale", 0.18),
            )
        )
        teammate_gap_scale = float(
            cfg.get(
                "learning.teammate_gap_scale",
                cfg.get("baseline_predictor.learning.teammate_gap_scale", 0.10),
            )
        )
        max_adjustment = float(
            cfg.get(
                "learning.max_adjustment",
                cfg.get("baseline_predictor.learning.max_adjustment", 2.5),
            )
        )

        try:
            return float(
                getter(
                    team=team,
                    driver=driver,
                    teammates=teammates,
                    session=session,
                    min_samples=max(1, min_samples),
                    driver_error_scale=driver_error_scale,
                    teammate_gap_scale=teammate_gap_scale,
                    max_adjustment=max_adjustment,
                )
            )
        except Exception as exc:
            logger.debug("Could not load learned race adjustment for %s: %s", driver, exc)
            return 0.0

    def _get_learned_interval_radius(self, *, session: str = "race") -> float:
        """Return learned interval radius floor from systematic calibration state."""
        calibration_system = getattr(self, "calibration_system", None)
        if calibration_system is None:
            return 0.0

        getter = getattr(calibration_system, "get_interval_radius", None)
        if not callable(getter):
            return 0.0

        cfg = getattr(self, "config", config_loader)
        min_samples = int(cfg.get("learning.interval_min_samples", 20))
        target_coverage = float(cfg.get("learning.interval_target_coverage", 0.90))
        max_adjustment = float(cfg.get("learning.interval_max_adjustment", 6.0))

        try:
            return float(
                getter(
                    session=session,
                    min_samples=max(1, min_samples),
                    target_coverage=target_coverage,
                    max_adjustment=max_adjustment,
                )
            )
        except Exception as exc:
            logger.debug("Could not load learned race interval radius: %s", exc)
            return 0.0

    def _load_race_residual_model(self) -> Any | None:
        """Load the persisted race residual model when enabled."""
        cfg = getattr(self, "config", config_loader)
        enabled = bool(cfg.get("baseline_predictor.race.race_residual_model.enabled", False))
        if not enabled:
            return None
        uses_testing_seed = getattr(self, "_uses_testing_model_team_seed", None)
        if callable(uses_testing_seed) and uses_testing_seed():
            allow_with_testing_seed = bool(
                cfg.get(
                    "baseline_predictor.race.race_residual_model.allow_with_testing_seed", False
                )
            )
            if not allow_with_testing_seed:
                logger.info(
                    "Skipping race residual model because the active team seed is testing_model."
                )
                return None

        cached = getattr(self, "_race_residual_model_cache", None)
        if cached is not None:
            return cached

        from src.models.race_residual_model import load_race_residual_model

        resolver = getattr(self, "_resolve_predictions_data_root", None)
        data_root = resolver() if callable(resolver) else Path("data")
        artifact_path = cfg.get(
            "baseline_predictor.race.race_residual_model.artifact_path",
            str(
                Path(data_root)
                / "processed"
                / "model_artifacts"
                / "race_residual"
                / "race_residual_model.pkl"
            ),
        )
        loaded = load_race_residual_model(artifact_path)
        self._race_residual_model_cache = loaded
        return loaded

    def _get_conformal_interval_radius(self, *, session: str, regime: str) -> float:
        """Return the conformal interval radius for one session/regime bucket."""
        cfg = getattr(self, "config", config_loader)
        enabled = bool(cfg.get("baseline_predictor.conformal_calibration.enabled", False))
        if not enabled:
            return 0.0

        cached = getattr(self, "_conformal_calibration_artifact_cache", None)
        if cached is None:
            from src.models.conformal_calibration import load_conformal_calibration_artifact

            resolver = getattr(self, "_resolve_predictions_data_root", None)
            data_root = resolver() if callable(resolver) else Path("data")
            artifact_path = cfg.get(
                "baseline_predictor.conformal_calibration.artifact_path",
                str(
                    Path(data_root)
                    / "processed"
                    / "model_artifacts"
                    / "conformal_calibration"
                    / "conformal_calibration.json"
                ),
            )
            cached = load_conformal_calibration_artifact(artifact_path)
            self._conformal_calibration_artifact_cache = cached

        if cached is None:
            return 0.0
        try:
            return float(cached.get_radius(session=session, regime=regime))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _apply_race_residual_model(
        self,
        *,
        driver_info_map: dict[str, DriverRaceInfo],
        qualifying_grid: list[QualifyingGridEntry],
        race_name: str | None,
        weather: str,
        input_confidence: float | None,
        is_sprint: bool,
        year: int,
    ) -> dict[str, float]:
        """Apply the persisted race residual model as a small race-advantage correction."""
        if not race_name:
            return {}

        model = self._load_race_residual_model()
        if model is None:
            return {}

        from src.models.race_residual_model import (
            apply_race_residual_model,
            build_feature_frame_from_context,
        )

        confidence_values = [
            float(row["confidence"]) / 100.0
            for row in qualifying_grid
            if isinstance(row.get("confidence"), int | float)
        ]
        mean_grid_confidence = (
            float(sum(confidence_values) / len(confidence_values)) if confidence_values else None
        )
        grid_source_mode = "predicted" if confidence_values else "actual"
        feature_frame = build_feature_frame_from_context(
            predictor=self,
            year=year,
            race_name=race_name,
            weather=weather,
            qualifying_grid=qualifying_grid,
            driver_info_map=driver_info_map,
            input_confidence=input_confidence,
            mean_grid_confidence=mean_grid_confidence,
            grid_source_mode=grid_source_mode,
            is_sprint=is_sprint,
        )
        scale = float(
            getattr(self, "config", config_loader).get(
                "baseline_predictor.race.race_residual_model.positions_to_race_advantage_scale",
                0.05,
            )
        )
        return apply_race_residual_model(
            model=model,
            feature_frame=feature_frame,
            driver_info_map=driver_info_map,
            positions_to_race_advantage_scale=scale,
        )

    def predict_race(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 300,
        is_sprint: bool = False,
        race_compound: str = "MEDIUM",
        year: int | None = None,
        input_confidence: float | None = None,
        prediction_context: PredictionContext | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Predict race result using lap-by-lap Monte Carlo simulation with tire deg and pit stops.

        ``location`` (the schedule venue) makes circuit resolution authoritative for track
        params; callers that know it (warmup, dashboard prediction flow) pass it. When it is
        omitted, circuit lookups fall back to name/year resolution.
        """
        cfg = getattr(self, "config", config_loader)
        resolved_year = (
            year
            if year is not None
            else int(getattr(self, "season_year", getattr(self, "year", 2026)))
        )
        with activate_prediction_runtime(config=cfg, prediction_context=prediction_context):
            validate_enum(weather, "weather", ["dry", "rain", "mixed"])
            validate_positive_int(n_simulations, "n_simulations", min_val=1)
            validated_grid = validate_qualifying_grid(qualifying_grid)
            return predict_race_core(
                validated_grid=validated_grid,
                weather=weather,
                race_name=race_name,
                n_simulations=n_simulations,
                is_sprint=is_sprint,
                race_compound=race_compound,
                input_confidence=input_confidence,
                location=location,
                year=resolved_year,
                cfg=cfg,
                base_seed=int(getattr(self, "seed", 42)),
                deps=RaceSimulationDeps(
                    load_race_params=self._load_race_params,
                    prepare_driver_info_with_compounds=self._prepare_driver_info_with_compounds,
                    get_learned_position_adjustment=self._get_learned_position_adjustment,
                    get_learned_interval_radius=self._get_learned_interval_radius,
                    apply_race_residual_model=self._apply_race_residual_model,
                    get_conformal_interval_radius=self._get_conformal_interval_radius,
                    enforce_non_increasing=self._enforce_non_increasing,
                    load_track_specific_params=load_track_specific_params,
                    get_tire_stress_score=get_tire_stress_score,
                    get_available_compounds=get_available_compounds,
                    resolve_track_temperature_c=resolve_track_temperature_c,
                    resolve_track_temperature_profile=resolve_track_temperature_profile,
                    resolve_non_competitive_weather_features=resolve_non_competitive_weather_features,
                    resolve_race_distance_laps=resolve_race_distance_laps,
                    generate_pit_strategy=generate_pit_strategy,
                    simulate_race_lap_by_lap=simulate_race_lap_by_lap,
                    aggregate_simulation_results=aggregate_simulation_results,
                ),
            )
