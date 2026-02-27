"""Race prediction method for Baseline2026Predictor."""

from __future__ import annotations

import logging
from typing import Any

from src.types.prediction_types import QualifyingGridEntry
from src.utils import config_loader
from src.utils.grid_validation import validate_qualifying_grid
from src.utils.lap_by_lap_simulator import (
    aggregate_simulation_results,
    simulate_race_lap_by_lap,
)
from src.utils.pit_strategy import generate_pit_strategy
from src.utils.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_race_distance_laps,
)
from src.utils.validation_helpers import validate_enum, validate_positive_int

from .prediction_flow import predict_race_core

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePredictionMixin:
    """Race prediction method implementation for Baseline2026Predictor."""

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
    ) -> float:
        """Return learned position adjustment from systematic calibration state."""
        calibration_system = getattr(self, "calibration_system", None)
        if calibration_system is None:
            return 0.0

        getter = getattr(calibration_system, "get_combined_position_adjustment", None)
        if not callable(getter):
            return 0.0

        cfg = getattr(self, "config", config_loader)
        min_samples = int(cfg.get("baseline_predictor.learning.min_samples", 1))
        driver_error_scale = float(cfg.get("baseline_predictor.learning.driver_error_scale", 0.18))
        teammate_gap_scale = float(cfg.get("baseline_predictor.learning.teammate_gap_scale", 0.10))
        max_adjustment = float(cfg.get("baseline_predictor.learning.max_adjustment", 2.5))

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
            logger.debug(f"Could not load learned race adjustment for {driver}: {exc}")
            return 0.0

    def predict_race(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
        is_sprint: bool = False,
        race_compound: str = "MEDIUM",
        year: int | None = None,
        input_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Predict race result using lap-by-lap Monte Carlo simulation with tire deg and pit stops."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        cfg = getattr(self, "config", config_loader)
        validated_grid = validate_qualifying_grid(qualifying_grid)
        return predict_race_core(
            validated_grid=validated_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=is_sprint,
            race_compound=race_compound,
            input_confidence=input_confidence,
            year=year
            if year is not None
            else int(getattr(self, "season_year", getattr(self, "year", 2026))),
            cfg=cfg,
            base_seed=int(getattr(self, "seed", 42)),
            load_race_params=self._load_race_params,
            prepare_driver_info_with_compounds=self._prepare_driver_info_with_compounds,
            get_learned_position_adjustment=self._get_learned_position_adjustment,
            enforce_non_increasing=self._enforce_non_increasing,
            load_track_specific_params=load_track_specific_params,
            get_tire_stress_score=get_tire_stress_score,
            get_available_compounds=get_available_compounds,
            resolve_race_distance_laps=resolve_race_distance_laps,
            generate_pit_strategy=generate_pit_strategy,
            simulate_race_lap_by_lap=simulate_race_lap_by_lap,
            aggregate_simulation_results=aggregate_simulation_results,
        )
