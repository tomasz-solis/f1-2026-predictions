"""Qualifying and sprint-race mixin for Baseline2026Predictor."""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

import numpy as np

from src.types.prediction_types import QualifyingGridEntry
from src.utils import config_loader
from src.utils.fp_blending import blend_team_strength, get_best_fp_performance
from src.utils.lineups import get_lineups
from src.utils.validation_helpers import (
    validate_enum,
    validate_positive_int,
    validate_year,
)
from src.utils.weekend import is_sprint_weekend

from .qualifying_preparation import (
    apply_testing_fallback_adjustment as _apply_testing_fallback_adjustment_impl,
)
from .qualifying_preparation import (
    build_driver_list_with_strengths_core,
)
from .qualifying_preparation import (
    build_testing_short_run_fallback as _build_testing_short_run_fallback_impl,
)
from .qualifying_preparation import (
    extract_experience_total_races as _extract_experience_total_races_impl,
)
from .qualifying_preparation import (
    resolve_effective_experience_tier as _resolve_effective_experience_tier_impl,
)
from .qualifying_simulation import run_qualifying_simulations

logger = logging.getLogger("src.predictors.baseline_2026")

_DEFAULT_TESTING_SHORT_RUN_WEIGHTS = {
    "overall_pace": 0.55,
    "top_speed": 0.20,
    "medium_corner_performance": 0.15,
    "fast_corner_performance": 0.10,
}


class BaselineQualifyingMixin:
    """Shared qualifying and sprint-race methods for Baseline2026Predictor."""

    def _get_testing_profile_weights(
        self, profile: str, defaults: dict[str, float]
    ) -> dict[str, float]:
        """Get configured testing profile weights with safe fallback."""
        cfg = getattr(self, "config", config_loader)
        weights = cfg.get(f"baseline_predictor.race.testing_profile_weights.{profile}", defaults)
        return weights if isinstance(weights, dict) and weights else defaults

    def _resolve_effective_experience_tier(
        self, driver_data: dict[str, Any], prediction_year: int | None
    ) -> str:
        """Resolve experience tier at prediction time to avoid stale preseason labels."""
        return _resolve_effective_experience_tier_impl(
            driver_data=driver_data,
            prediction_year=prediction_year,
        )

    def _extract_experience_total_races(self, driver_data: dict[str, Any]) -> int | None:
        """Extract total races from driver profile when available."""
        return _extract_experience_total_races_impl(driver_data)

    def _build_testing_short_run_fallback(
        self,
        lineups: dict[str, list[str]],
        metric_weights: dict[str, float],
    ) -> dict[str, float] | None:
        """Build a team-pace fallback from stored short-run testing profiles."""
        cfg = getattr(self, "config", config_loader)
        return _build_testing_short_run_fallback_impl(
            lineups=lineups,
            metric_weights=metric_weights,
            cfg=cfg,
            get_testing_characteristics_for_profile=self._get_testing_characteristics_for_profile,
        )

    def _apply_testing_fallback_adjustment(
        self,
        model_strengths: dict[str, float],
        testing_fallback_performance: dict[str, float] | None,
    ) -> dict[str, float]:
        """
        Apply a conservative testing-derived adjustment on top of model strengths.

        Testing programs are noisy (fuel/load/run-plan effects), so we use them as a
        bounded relative nudge instead of treating them as direct pace replacement.
        """
        cfg = getattr(self, "config", config_loader)
        return _apply_testing_fallback_adjustment_impl(
            model_strengths=model_strengths,
            testing_fallback_performance=testing_fallback_performance,
            cfg=cfg,
        )

    def _get_learned_position_adjustment(
        self,
        *,
        team: str,
        driver: str,
        teammates: list[str],
        session: str = "qualifying",
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
            logger.debug(f"Could not load learned qualifying adjustment for {driver}: {exc}")
            return 0.0

    def _build_driver_list_with_strengths(
        self,
        lineups: dict[str, list[str]],
        fp_performance: dict[str, float] | None,
        testing_fallback_performance: dict[str, float] | None,
        race_name: str,
        is_sprint: bool,
        prediction_year: int | None = None,
    ) -> tuple[list[dict], int]:
        """Build driver list with blended team/driver strengths and testing modifiers."""
        _ = is_sprint
        cfg = getattr(self, "config", config_loader)
        short_profile_weights = self._get_testing_profile_weights(
            "short_run",
            _DEFAULT_TESTING_SHORT_RUN_WEIGHTS,
        )
        fallback_loader = getattr(self, "_get_driver_data_or_fallback", None)
        return build_driver_list_with_strengths_core(
            lineups=lineups,
            fp_performance=fp_performance,
            testing_fallback_performance=testing_fallback_performance,
            race_name=race_name,
            prediction_year=prediction_year,
            drivers=self.drivers,
            cfg=cfg,
            short_profile_weights=short_profile_weights,
            get_blended_team_strength_fn=self.get_blended_team_strength,
            compute_testing_profile_modifier_fn=self._compute_testing_profile_modifier,
            blend_team_strength_fn=blend_team_strength,
            apply_testing_fallback_adjustment_fn=self._apply_testing_fallback_adjustment,
            resolve_effective_experience_tier_fn=self._resolve_effective_experience_tier,
            extract_experience_total_races_fn=self._extract_experience_total_races,
            get_learned_position_adjustment_fn=self._get_learned_position_adjustment,
            get_driver_data_or_fallback_fn=(fallback_loader if callable(fallback_loader) else None),
        )

    def _run_qualifying_simulations(
        self,
        all_drivers: list[dict],
        n_simulations: int,
        is_sprint: bool,
        has_practice_data: bool,
        rng: np.random.Generator,
    ) -> dict[str, list[int]]:
        """Run Monte Carlo qualifying simulations and return position records."""
        cfg = getattr(self, "config", config_loader)
        return run_qualifying_simulations(
            all_drivers=all_drivers,
            n_simulations=n_simulations,
            is_sprint=is_sprint,
            has_practice_data=has_practice_data,
            rng=rng,
            cfg=cfg,
            logger=logger,
        )

    def _aggregate_grid_results(
        self, position_records: dict[str, list[int]], all_drivers: list[dict]
    ) -> list[QualifyingGridEntry]:
        """Aggregate simulation results into final grid with confidence intervals."""
        cfg = getattr(self, "config", config_loader)
        grid: list[QualifyingGridEntry] = []
        confidence_std_multiplier = cfg.get(
            "baseline_predictor.qualifying.confidence_std_multiplier", 5.0
        )
        confidence_cap = cfg.get("baseline_predictor.qualifying.confidence_cap", 60)
        confidence_min = cfg.get("baseline_predictor.qualifying.confidence_min", 40)

        for driver_info in all_drivers:
            positions = position_records[driver_info["driver"]]
            median_pos = int(np.median(positions))
            mean_pos = float(np.mean(positions))
            p5 = int(np.percentile(positions, 5))
            p95 = int(np.percentile(positions, 95))

            position_std = np.std(positions)
            confidence = max(
                confidence_min,
                min(confidence_cap, confidence_cap - (position_std * confidence_std_multiplier)),
            )

            grid.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "position": median_pos,
                    "median_position": median_pos,
                    "_mean_position": mean_pos,
                    "p5": p5,
                    "p95": p95,
                    "confidence": float(round(confidence, 1)),
                }
            )

        # Resolve median ties with the underlying simulation mean so teammate order
        # does not collapse into insertion-order blocks when medians are equal.
        grid.sort(key=lambda x: (x["median_position"], x["_mean_position"], x["driver"]))

        for i, item in enumerate(grid):
            item["position"] = i + 1
            item.pop("_mean_position", None)

        return grid

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
    ) -> dict[str, Any]:
        """Predict qualifying with Monte Carlo simulation (sprint/normal weekends)."""
        cfg = getattr(self, "config", config_loader)

        validate_year(year, "year", min_year=2020, max_year=2030)
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        validate_enum(qualifying_stage, "qualifying_stage", ["auto", "sprint", "main"])

        try:
            is_sprint = is_sprint_weekend(year, race_name)
        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Could not determine sprint weekend for {race_name}: {e}")
            is_sprint = False

        seed_material = f"{self.seed}:{year}:{race_name}:{qualifying_stage}:{int(is_sprint)}"
        seed = int(sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)

        lineups = get_lineups(year, race_name)

        session_name, fp_performance, session_laps = get_best_fp_performance(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            qualifying_stage=qualifying_stage,
        )

        if session_laps is not None:
            self._update_compound_characteristics_from_session(
                session_laps, race_name, year, is_sprint
            )

        short_profile_weights = self._get_testing_profile_weights(
            "short_run",
            _DEFAULT_TESTING_SHORT_RUN_WEIGHTS,
        )
        testing_fallback_performance = None
        if session_name is None and fp_performance is None:
            testing_fallback_performance = self._build_testing_short_run_fallback(
                lineups=lineups,
                metric_weights=short_profile_weights,
            )
        testing_fallback_used = testing_fallback_performance is not None

        all_drivers, teams_with_short_profile = self._build_driver_list_with_strengths(
            lineups,
            fp_performance,
            testing_fallback_performance,
            race_name,
            is_sprint,
            prediction_year=year,
        )
        if cfg.get("baseline_predictor.qualifying.enable_driver_fp_adjustment", True):
            from src.utils.driver_fp_adjustment import calculate_driver_fp_modifiers

            fp_session_types = ["FP1"] if is_sprint else ["FP1", "FP2", "FP3"]
            modifier_scale = cfg.get(
                "baseline_predictor.qualifying.driver_fp_adjustment_scale", 0.10
            )
            smoothing_seconds = cfg.get(
                "baseline_predictor.qualifying.driver_fp_adjustment_smoothing", 0.50
            )
            driver_fp_modifiers = calculate_driver_fp_modifiers(
                year=year,
                race_name=race_name,
                session_types=fp_session_types,
                scale=modifier_scale,
                smoothing_seconds=smoothing_seconds,
            )
            for driver_info in all_drivers:
                fp_modifier = driver_fp_modifiers.get(driver_info["driver"], 0.0)
                if fp_modifier == 0.0:
                    continue
                driver_info["skill"] = np.clip(driver_info["skill"] + fp_modifier, 0.01, 0.99)

        position_records = self._run_qualifying_simulations(
            all_drivers, n_simulations, is_sprint, session_name is not None, rng
        )

        grid = self._aggregate_grid_results(position_records, all_drivers)

        if session_name is not None:
            data_source = session_name
        elif testing_fallback_used:
            data_source = "Testing short-run profile blend (no weekend practice data)"
        else:
            data_source = "Model-only (no practice/testing data)"

        return {
            "grid": grid,
            "data_source": data_source,
            "blend_used": session_name is not None,
            "testing_fallback_used": testing_fallback_used,
            "qualifying_stage": qualifying_stage,
            "characteristics_profile_used": "short_run",
            "teams_with_characteristics_profile": teams_with_short_profile,
        }

    def predict_sprint_race(
        self,
        sprint_quali_grid: list[dict],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
    ) -> dict[str, Any]:
        """Predict Sprint Race with reduced chaos and increased grid influence."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

        result = self.predict_race(
            qualifying_grid=sprint_quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=True,
        )

        return result
