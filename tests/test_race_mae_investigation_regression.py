"""Regression guard for the race-MAE investigation and position-weighted metrics work.

This research task (see CLAUDE_CODE_HANDOFF.md) is explicitly forbidden from changing
production configuration or the served champion default. This test proves that stayed
true after the metric-helper and report-builder extensions: ``config/production_config.json``
keeps its known baseline digest, and ``config/default.yaml`` still resolves
``baseline_predictor.model_variant`` to ``champion``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Baseline digest recorded before this research task began. Any drift here means
# config/production_config.json changed, which the task's hard boundaries forbid.
_PRODUCTION_CONFIG_SHA256 = "c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1"


def test_production_config_json_matches_known_baseline_digest():
    production_config_path = REPO_ROOT / "config" / "production_config.json"
    digest = hashlib.sha256(production_config_path.read_bytes()).hexdigest()
    assert digest == _PRODUCTION_CONFIG_SHA256, (
        "config/production_config.json changed during research work; this file must stay untouched."
    )


def test_default_yaml_model_variant_stays_champion():
    default_config_path = REPO_ROOT / "config" / "default.yaml"
    with default_config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    model_variant = loaded["baseline_predictor"]["model_variant"]
    assert model_variant == "champion", (
        "baseline_predictor.model_variant in config/default.yaml must remain "
        "'champion'; research challengers are opt-in overlays, never the default."
    )
