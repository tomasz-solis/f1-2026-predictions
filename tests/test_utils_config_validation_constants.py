"""Tests for production config, constants, and validation metrics utilities."""

import json

import numpy as np
import pytest

from src.utils import constants, validation
from src.utils.config import ProductionConfig, get_best_method, load_production_config


def _sample_config():
    return {
        "qualifying_methods": {
            "sprint_weekends": {
                "method": "session",
                "session": "FP1",
                "expected_mae": 2.5,
                "confidence": "high",
            },
            "conventional_weekends": {
                "method": "blend",
                "blend_weight": 0.7,
                "expected_mae": 2.0,
                "confidence": "high",
            },
        },
        "race_methods": {"default": {"expected_mae": 3.1}},
        "notes": {
            "comprehensive_testing_notebook": "21B",
            "total_races_analyzed": 24,
            "last_updated": "2026-02-01",
            "performance_ranking_2025": {"1": "McLaren", "2": "Ferrari"},
        },
    }


def test_production_config_load_and_strategy(tmp_path):
    config_file = tmp_path / "production_config.json"
    config_file.write_text(json.dumps(_sample_config()))

    cfg = ProductionConfig(str(config_file))

    sprint = cfg.get_qualifying_strategy("sprint")
    conv = cfg.get_qualifying_strategy("conventional")

    assert sprint["method"] == "session"
    assert conv["method"] == "blend"
    assert cfg.get_expected_mae("qualifying", weekend_type="sprint") == 2.5
    assert cfg.get_expected_mae("race") == 3.1
    assert cfg.get_performance_ranking()["1"] == "McLaren"
    assert "PRODUCTION CONFIGURATION" in str(cfg)


def test_production_config_weighted_mae_and_fallback(tmp_path):
    config_file = tmp_path / "production_config.json"
    config_file.write_text(json.dumps(_sample_config()))
    cfg = ProductionConfig(str(config_file))

    weighted = cfg.get_expected_mae("qualifying")
    assert weighted == (2.5 * 6 + 2.0 * 18) / 24
    assert cfg.get_expected_mae("unknown") == 4.0


def test_load_production_config_and_get_best_method(patcher, tmp_path):
    config_file = tmp_path / "production_config.json"
    config_file.write_text(json.dumps(_sample_config()))

    loaded = load_production_config(str(config_file))
    assert isinstance(loaded, ProductionConfig)

    patcher.setattr("src.utils.config.load_production_config", lambda: loaded)
    best = get_best_method("conventional")
    assert best["method"] == "blend"


def test_production_config_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        ProductionConfig(str(missing_path))


def test_constants_module_contains_expected_ranges():
    assert constants.DNF_RATE_HISTORICAL_CAP <= constants.DNF_RATE_FINAL_CAP
    assert 0 < constants.QUALI_TEAM_WEIGHT < 1
    assert 0 < constants.QUALI_SKILL_WEIGHT < 1
    assert constants.GRID_WEIGHT_MIN < constants.GRID_WEIGHT_MAX
    assert constants.CORNER_SPEED_THRESHOLD == 250
    assert constants.FUEL_LOAD_MAX_KG == 110


def test_compare_rankings_and_aggregate_metrics():
    metrics = validation.compare_rankings(
        predicted=["Mercedes", "Red Bull", "McLaren", "Ferrari", "Alpine"],
        actual=["Red Bull", "Mercedes", "Ferrari", "McLaren", "Alpine"],
    )
    assert "winner_correct" in metrics
    assert "top3_accuracy" in metrics
    assert "spearman" in metrics
    assert "mae_positions" in metrics

    aggregated = validation.aggregate_metrics(
        [metrics, {"winner_correct": np.nan, "top3_accuracy": 1.0, "spearman": 0.5}]
    )
    assert "top3_accuracy" in aggregated


def test_confidence_calibration_and_grouping_helpers():
    calibration = validation.confidence_calibration(
        [(0.9, True), (0.8, True), (0.2, False), (0.4, False)]
    )
    assert "brier_score" in calibration
    assert "bins" in calibration

    by_track = validation.analyze_by_track_type(
        {
            "Monaco Grand Prix": {"mae_positions": 2.5},
            "Monza Grand Prix": {"mae_positions": 1.8},
        },
        {
            "Monaco Grand Prix": "street",
            "Monza Grand Prix": "permanent",
        },
    )
    assert by_track["street"]["count"] == 1
    assert by_track["permanent"]["count"] == 1

    by_stage = validation.analyze_by_stage(
        {
            "RaceA": {"FP3": {"mae_positions": 2.0}, "Q": {"mae_positions": 1.5}},
            "RaceB": {"FP3": {"mae_positions": 2.4}, "Q": {"mae_positions": 1.2}},
        }
    )
    assert by_stage["FP3"]["count"] == 2
    assert by_stage["Q"]["count"] == 2
