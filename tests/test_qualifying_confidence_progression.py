"""Tests for weekend-stage confidence and FP blend adaptation in qualifying flow."""

from __future__ import annotations

import numpy as np

from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin


class _ConfigStub:
    def __init__(self):
        self._values = {
            "baseline_predictor.qualifying.fp_blend_weight": 0.70,
            "baseline_predictor.qualifying.fp_blend_weight_min": 0.45,
            "baseline_predictor.qualifying.fp_blend_weight_max": 0.85,
            "baseline_predictor.qualifying.fp_blend_confidence_scale": 0.30,
            "baseline_predictor.qualifying.session_confidence_scale": 10.0,
            "baseline_predictor.qualifying.confidence_std_multiplier": 5.0,
            "baseline_predictor.qualifying.confidence_cap": 60,
            "baseline_predictor.qualifying.confidence_min": 40,
            "baseline_predictor.qualifying.early_season_team_uncertainty.activation_floor": 0.22,
            "baseline_predictor.qualifying.early_season_team_uncertainty.scale": 0.20,
            "baseline_predictor.qualifying.early_season_team_uncertainty.decay_races": 3,
            "baseline_predictor.qualifying.early_season_team_uncertainty.interval_positions_scale": 2.0,
            "baseline_predictor.qualifying.early_season_team_uncertainty.confidence_penalty_scale": 6.0,
            "baseline_predictor.qualifying.data_confidence.model_only": 0.25,
            "baseline_predictor.qualifying.data_confidence.testing_fallback": 0.45,
            "baseline_predictor.qualifying.data_confidence.sprint_race": 0.70,
            "qualifying.session_confidence.fp1": 0.2,
            "qualifying.session_confidence.fp2": 0.5,
            "qualifying.session_confidence.fp3": 0.9,
            "qualifying.session_confidence.sprint_quali": 0.85,
        }

    def get(self, key: str, default):
        return self._values.get(key, default)


class _Predictor(BaselineQualifyingMixin):
    def __init__(self):
        self.config = _ConfigStub()


class _IntervalCalibrationStub:
    def get_interval_radius(
        self,
        *,
        session: str,
        min_samples: int,
        target_coverage: float,
        max_adjustment: float,
    ) -> float:
        _ = (session, min_samples, target_coverage, max_adjustment)
        return 2.0


def test_weekend_data_confidence_increases_with_session_maturity():
    predictor = _Predictor()

    model_only = predictor._resolve_data_confidence_score(None, testing_fallback_used=False)
    testing_fallback = predictor._resolve_data_confidence_score(None, testing_fallback_used=True)
    fp1 = predictor._resolve_data_confidence_score("FP1 short-stint", testing_fallback_used=False)
    fp3 = predictor._resolve_data_confidence_score("FP3 short-stint", testing_fallback_used=False)

    assert model_only < testing_fallback < fp3
    assert fp1 < fp3


def test_sprint_quali_confidence_uses_precedence_without_double_counting():
    predictor = _Predictor()

    sprint_quali = predictor._resolve_data_confidence_score(
        "Sprint Qualifying short-stint",
        testing_fallback_used=False,
    )
    sprint_session = predictor._resolve_data_confidence_score(
        "Sprint pace signal",
        testing_fallback_used=False,
    )

    assert sprint_quali == 0.85
    assert sprint_session == 0.70
    assert sprint_quali > sprint_session


def test_fp_blend_weight_scales_with_data_confidence():
    predictor = _Predictor()

    low_weight = predictor._resolve_fp_blend_weight(0.2)
    high_weight = predictor._resolve_fp_blend_weight(0.9)

    assert 0.45 <= low_weight <= 0.85
    assert 0.45 <= high_weight <= 0.85
    assert high_weight > low_weight


def test_aggregate_grid_confidence_gets_weekend_stage_boost():
    predictor = _Predictor()
    position_records = {
        "VER": [1, 1, 2, 1, 2, 1, 1, 2],
        "NOR": [2, 2, 1, 2, 1, 2, 2, 1],
    }
    all_drivers = [
        {"driver": "VER", "team": "Red Bull Racing"},
        {"driver": "NOR", "team": "McLaren"},
    ]

    low_conf_grid = predictor._aggregate_grid_results(
        position_records,
        all_drivers,
        data_confidence_score=0.2,
    )
    high_conf_grid = predictor._aggregate_grid_results(
        position_records,
        all_drivers,
        data_confidence_score=0.9,
    )

    low_avg_conf = np.mean([entry["confidence"] for entry in low_conf_grid])
    high_avg_conf = np.mean([entry["confidence"] for entry in high_conf_grid])

    assert high_avg_conf > low_avg_conf


def test_aggregate_grid_applies_learned_interval_radius_floor():
    predictor = _Predictor()
    predictor.calibration_system = _IntervalCalibrationStub()
    position_records = {
        "VER": [1, 1, 1, 2, 1, 1],
        "NOR": [2, 2, 2, 1, 2, 2],
    }
    all_drivers = [
        {"driver": "VER", "team": "Red Bull Racing"},
        {"driver": "NOR", "team": "McLaren"},
    ]

    grid = predictor._aggregate_grid_results(
        position_records, all_drivers, data_confidence_score=0.9
    )
    by_driver = {entry["driver"]: entry for entry in grid}

    assert by_driver["VER"]["p5"] == 1
    assert by_driver["VER"]["p95"] == 2
    assert by_driver["NOR"]["p5"] <= 1
    assert by_driver["NOR"]["p95"] == 2


def test_aggregate_grid_uses_team_uncertainty_to_widen_opening_weekends():
    predictor = _Predictor()
    position_records = {
        "VER": [1, 1, 1, 1, 1, 1],
        "NOR": [2, 2, 2, 2, 2, 2],
        "PIA": [3, 3, 3, 3, 3, 3],
        "RUS": [4, 4, 4, 4, 4, 4],
    }
    early_drivers = [
        {
            "driver": "VER",
            "team": "Red Bull Racing",
            "team_uncertainty": 0.42,
            "season_races_completed": 0,
        },
        {"driver": "NOR", "team": "McLaren", "team_uncertainty": 0.42, "season_races_completed": 0},
        {"driver": "PIA", "team": "McLaren", "team_uncertainty": 0.42, "season_races_completed": 0},
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_uncertainty": 0.42,
            "season_races_completed": 0,
        },
    ]
    late_drivers = [
        {
            "driver": "VER",
            "team": "Red Bull Racing",
            "team_uncertainty": 0.42,
            "season_races_completed": 4,
        },
        {"driver": "NOR", "team": "McLaren", "team_uncertainty": 0.42, "season_races_completed": 4},
        {"driver": "PIA", "team": "McLaren", "team_uncertainty": 0.42, "season_races_completed": 4},
        {
            "driver": "RUS",
            "team": "Mercedes",
            "team_uncertainty": 0.42,
            "season_races_completed": 4,
        },
    ]

    early_grid = predictor._aggregate_grid_results(
        position_records,
        early_drivers,
        data_confidence_score=0.45,
    )
    late_grid = predictor._aggregate_grid_results(
        position_records,
        late_drivers,
        data_confidence_score=0.45,
    )

    early_by_driver = {entry["driver"]: entry for entry in early_grid}
    late_by_driver = {entry["driver"]: entry for entry in late_grid}

    assert early_by_driver["VER"]["p95"] > late_by_driver["VER"]["p95"]
    assert early_by_driver["NOR"]["confidence"] < late_by_driver["NOR"]["confidence"]
