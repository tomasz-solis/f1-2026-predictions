"""Regression tests against fixed-seed prediction outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.predictors.baseline_2026 import Baseline2026Predictor

GOLDEN_DIR = Path("data/test")


def _build_test_predictor() -> Baseline2026Predictor:
    """Create the deterministic predictor used for golden fixtures."""
    return Baseline2026Predictor(seed=42)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Round numeric output so the golden files track model behavior, not float noise."""
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            normalized[key] = round(value, 4)
        else:
            normalized[key] = value
    return normalized


def _qualifying_payload() -> dict[str, Any]:
    """Build the fixed-seed qualifying payload used for regression checks."""
    predictor = _build_test_predictor()
    result = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=40)
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Bahrain Grand Prix",
        "n_simulations": 40,
        "grid": [_normalize_row(row) for row in result["grid"]],
    }


def _race_payload() -> dict[str, Any]:
    """Build the fixed-seed race payload used for regression checks."""
    predictor = _build_test_predictor()
    qualifying = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=40)
    result = predictor.predict_race(
        qualifying["grid"],
        weather="dry",
        race_name="Bahrain Grand Prix",
        n_simulations=40,
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Bahrain Grand Prix",
        "weather": "dry",
        "n_simulations": 40,
        "finish_order": [_normalize_row(row) for row in result["finish_order"]],
    }


def _assert_matches_or_update(
    *,
    golden_path: Path,
    payload: dict[str, Any],
    update_golden_files: bool,
) -> None:
    """Compare one payload to disk or rewrite the golden fixture if requested."""
    if update_golden_files or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if update_golden_files:
            return

    golden_payload = json.loads(golden_path.read_text())
    assert payload == golden_payload, f"Prediction output changed: {golden_path}"


def test_qualifying_regression(update_golden_files):
    """Fixed-seed Bahrain qualifying output should stay stable."""
    payload = _qualifying_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_qualifying_bahrain.json",
        payload=payload,
        update_golden_files=update_golden_files,
    )


def test_race_regression(update_golden_files):
    """Fixed-seed Bahrain race output should stay stable."""
    payload = _race_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_race_bahrain.json",
        payload=payload,
        update_golden_files=update_golden_files,
    )
