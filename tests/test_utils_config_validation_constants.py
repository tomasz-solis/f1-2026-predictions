"""Tests for production config utilities."""

import json

import pytest

from src.utils.config import ProductionConfig, load_production_config


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
    assert "PRODUCTION CONFIGURATION" in str(cfg)


def test_load_production_config_returns_production_config(tmp_path):
    config_file = tmp_path / "production_config.json"
    config_file.write_text(json.dumps(_sample_config()))

    loaded = load_production_config(str(config_file))
    assert isinstance(loaded, ProductionConfig)


def test_production_config_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        ProductionConfig(str(missing_path))
