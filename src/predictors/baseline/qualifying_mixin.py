"""Qualifying and sprint-race mixin for Baseline2026Predictor."""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

import numpy as np

from src.types.prediction_types import QualifyingGridEntry
from src.utils import config_loader
from src.utils.fp_blending import (
    blend_team_strength,
    get_best_fp_performance_with_session_laps,
)
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

    def _resolve_data_confidence_score(
        self,
        session_name: str | None,
        *,
        testing_fallback_used: bool,
    ) -> float:
        """
        Estimate information quality for the current qualifying context.

        The score intentionally increases as the weekend progresses from FP1 to FP3
        (or sprint sessions), and is lowest for model-only runs.
        """
        cfg = getattr(self, "config", config_loader)
        model_only_confidence = float(
            cfg.get("baseline_predictor.qualifying.data_confidence.model_only", 0.25)
        )
        testing_fallback_confidence = float(
            cfg.get("baseline_predictor.qualifying.data_confidence.testing_fallback", 0.45)
        )
        fp1_confidence = float(cfg.get("qualifying.session_confidence.fp1", 0.2))
        fp2_confidence = float(cfg.get("qualifying.session_confidence.fp2", 0.5))
        fp3_confidence = float(cfg.get("qualifying.session_confidence.fp3", 0.9))
        sprint_quali_confidence = float(cfg.get("qualifying.session_confidence.sprint_quali", 0.85))
        sprint_race_confidence = float(
            cfg.get("baseline_predictor.qualifying.data_confidence.sprint_race", 0.70)
        )

        if session_name is None:
            return float(
                np.clip(
                    testing_fallback_confidence if testing_fallback_used else model_only_confidence,
                    0.0,
                    1.0,
                )
            )

        normalized_name = session_name.lower()
        # Priority-based matching avoids overlapping token double-counting
        # (e.g., "Sprint Qualifying" containing both "sprint qualifying" and "sprint").
        if "sprint qualifying" in normalized_name:
            return float(np.clip(sprint_quali_confidence, 0.0, 1.0))
        if "fp3" in normalized_name:
            return float(np.clip(fp3_confidence, 0.0, 1.0))
        if "fp2" in normalized_name:
            return float(np.clip(fp2_confidence, 0.0, 1.0))
        if "fp1" in normalized_name:
            return float(np.clip(fp1_confidence, 0.0, 1.0))
        if "sprint pace signal" in normalized_name or "sprint" in normalized_name:
            return float(np.clip(sprint_race_confidence, 0.0, 1.0))

        return float(np.clip(testing_fallback_confidence, 0.0, 1.0))

    def _resolve_fp_blend_weight(self, data_confidence_score: float) -> float:
        """Scale FP blend weight by weekend data confidence."""
        cfg = getattr(self, "config", config_loader)
        base_weight = float(cfg.get("baseline_predictor.qualifying.fp_blend_weight", 0.70))
        blend_scale = float(
            cfg.get("baseline_predictor.qualifying.fp_blend_confidence_scale", 0.30)
        )
        min_weight = float(cfg.get("baseline_predictor.qualifying.fp_blend_weight_min", 0.45))
        max_weight = float(cfg.get("baseline_predictor.qualifying.fp_blend_weight_max", 0.85))
        lower = min(min_weight, max_weight)
        upper = max(min_weight, max_weight)

        adjusted_weight = base_weight + ((float(data_confidence_score) - 0.5) * blend_scale)
        return float(np.clip(adjusted_weight, lower, upper))

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
        fp_blend_weight: float,
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
            fp_blend_weight=fp_blend_weight,
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
        has_testing_fallback_data: bool = False,
    ) -> dict[str, list[int]]:
        """Run Monte Carlo qualifying simulations and return position records."""
        cfg = getattr(self, "config", config_loader)
        return run_qualifying_simulations(
            all_drivers=all_drivers,
            n_simulations=n_simulations,
            is_sprint=is_sprint,
            has_practice_data=has_practice_data,
            has_testing_fallback_data=has_testing_fallback_data,
            rng=rng,
            cfg=cfg,
            logger=logger,
        )

    def _aggregate_grid_results(
        self,
        position_records: dict[str, list[int]],
        all_drivers: list[dict],
        *,
        data_confidence_score: float | None = None,
    ) -> list[QualifyingGridEntry]:
        """Aggregate simulation results into final grid with confidence intervals."""
        cfg = getattr(self, "config", config_loader)
        grid: list[QualifyingGridEntry] = []
        mean_positions: dict[str, float] = {}
        confidence_std_multiplier = cfg.get(
            "baseline_predictor.qualifying.confidence_std_multiplier", 5.0
        )
        session_confidence_scale = float(
            cfg.get("baseline_predictor.qualifying.session_confidence_scale", 10.0)
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
            if data_confidence_score is not None:
                confidence += (float(np.clip(data_confidence_score, 0.0, 1.0)) - 0.5) * (
                    session_confidence_scale
                )
                confidence = max(confidence_min, min(confidence_cap, confidence))

            grid.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "position": median_pos,
                    "median_position": median_pos,
                    "p5": p5,
                    "p95": p95,
                    "confidence": float(round(confidence, 1)),
                }
            )
            mean_positions[driver_info["driver"]] = mean_pos

        # Resolve median ties with the underlying simulation mean so teammate order
        # does not collapse into insertion-order blocks when medians are equal.
        grid.sort(
            key=lambda x: (
                x["median_position"],
                mean_positions.get(x["driver"], float(x["median_position"])),
                x["driver"],
            )
        )

        for i, item in enumerate(grid):
            item["position"] = i + 1

        return grid

    def _build_teammate_head_to_head_probabilities(
        self,
        *,
        position_records: dict[str, list[int]],
        all_drivers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build simulation-based teammate head-to-head probabilities by team."""
        drivers_by_team: dict[str, list[str]] = {}
        for driver_info in all_drivers:
            team_name = str(driver_info.get("team", "")).strip()
            driver_code = str(driver_info.get("driver", "")).strip()
            if not team_name or not driver_code:
                continue
            drivers_by_team.setdefault(team_name, [])
            if driver_code not in drivers_by_team[team_name]:
                drivers_by_team[team_name].append(driver_code)

        probabilities: list[dict[str, Any]] = []
        for team_name, team_drivers in drivers_by_team.items():
            if len(team_drivers) < 2:
                continue

            ranked_teammates = sorted(
                team_drivers,
                key=lambda driver: (
                    float(np.mean(position_records.get(driver, [999]))),
                    driver,
                ),
            )
            driver_a, driver_b = ranked_teammates[0], ranked_teammates[1]
            positions_a = position_records.get(driver_a, [])
            positions_b = position_records.get(driver_b, [])
            n_samples = min(len(positions_a), len(positions_b))
            if n_samples <= 0:
                continue

            wins_a = 0
            wins_b = 0
            ties = 0
            for position_a, position_b in zip(
                positions_a[:n_samples], positions_b[:n_samples], strict=True
            ):
                if position_a < position_b:
                    wins_a += 1
                elif position_b < position_a:
                    wins_b += 1
                else:
                    ties += 1

            p_a_ahead = wins_a / n_samples
            p_b_ahead = wins_b / n_samples
            p_tie = ties / n_samples
            probabilities.append(
                {
                    "team": team_name,
                    "driver_a": driver_a,
                    "driver_b": driver_b,
                    "p_driver_a_ahead": float(p_a_ahead),
                    "p_driver_b_ahead": float(p_b_ahead),
                    "p_tie": float(p_tie),
                    "n_samples": int(n_samples),
                    "decision_margin": float(abs(p_a_ahead - p_b_ahead)),
                }
            )

        probabilities.sort(key=lambda item: float(item.get("decision_margin", 1.0)))
        return probabilities

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

        session_name, fp_performance, session_laps, session_laps_by_type = (
            get_best_fp_performance_with_session_laps(
                year=year,
                race_name=race_name,
                is_sprint=is_sprint,
                qualifying_stage=qualifying_stage,
            )
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
        data_confidence_score = self._resolve_data_confidence_score(
            session_name,
            testing_fallback_used=testing_fallback_used,
        )
        effective_fp_blend_weight = self._resolve_fp_blend_weight(data_confidence_score)

        all_drivers, teams_with_short_profile = self._build_driver_list_with_strengths(
            lineups,
            fp_performance,
            testing_fallback_performance,
            race_name,
            is_sprint,
            effective_fp_blend_weight,
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
                preloaded_session_laps=session_laps_by_type,
            )
            for driver_info in all_drivers:
                fp_modifier = driver_fp_modifiers.get(driver_info["driver"], 0.0)
                if fp_modifier == 0.0:
                    continue
                driver_info["skill"] = np.clip(driver_info["skill"] + fp_modifier, 0.01, 0.99)

        position_records = self._run_qualifying_simulations(
            all_drivers,
            n_simulations,
            is_sprint,
            session_name is not None,
            rng,
            testing_fallback_used,
        )

        grid = self._aggregate_grid_results(
            position_records,
            all_drivers,
            data_confidence_score=data_confidence_score,
        )

        if session_name is not None:
            data_source = session_name
        elif testing_fallback_used:
            data_source = "Testing short-run profile blend (no weekend practice data)"
        else:
            data_source = "Model-only (no practice/testing data)"

        teammate_head_to_head = self._build_teammate_head_to_head_probabilities(
            position_records=position_records,
            all_drivers=all_drivers,
        )

        return {
            "grid": grid,
            "data_source": data_source,
            "blend_used": session_name is not None,
            "testing_fallback_used": testing_fallback_used,
            "data_confidence_score": round(float(data_confidence_score), 3),
            "fp_blend_weight_used": round(float(effective_fp_blend_weight), 3),
            "qualifying_stage": qualifying_stage,
            "characteristics_profile_used": "short_run",
            "teams_with_characteristics_profile": teams_with_short_profile,
            "teammate_head_to_head": teammate_head_to_head,
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
