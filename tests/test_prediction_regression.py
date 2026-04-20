"""Regression tests against fixed-seed prediction outputs."""

from __future__ import annotations

import json
import numbers
import sys
from pathlib import Path
from typing import Any

from src.predictors.baseline_2026 import Baseline2026Predictor
from src.utils.prediction_context import build_historical_prediction_context

GOLDEN_DIR = Path("data/test")
REFERENCE_REGRESSION_ENV = sys.platform == "darwin" and sys.version_info[:2] == (3, 11)


def _build_test_predictor() -> Baseline2026Predictor:
    """Create the deterministic predictor used for golden fixtures."""
    return Baseline2026Predictor(seed=42)


def _historical_context(race_name: str, session_name: str) -> dict[str, Any]:
    """Build deterministic historical context kwargs for regression predictions."""
    return {
        "prediction_context": build_historical_prediction_context(
            year=2026,
            race_name=race_name,
            target_session_name=session_name,
            seed=42,
        )
    }


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Round numeric output so the golden files track model behavior, not float noise."""
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, numbers.Integral):
            normalized[key] = int(value)
        elif isinstance(value, numbers.Real):
            normalized[key] = round(value, 4)
        else:
            normalized[key] = value
    return normalized


def _assert_payload_shape(payload: dict[str, Any], *, row_key: str) -> None:
    """Validate the basic structure of one stored prediction payload."""
    rows = payload[row_key]
    assert isinstance(rows, list)
    assert len(rows) == 22

    positions = [int(row["position"]) for row in rows]
    assert positions == list(range(1, 23))

    drivers = [str(row["driver"]) for row in rows]
    assert len(set(drivers)) == len(rows)


def _assert_cross_environment_regression(
    *,
    payload: dict[str, Any],
    golden_payload: dict[str, Any],
    row_key: str,
    max_position_delta: int,
    mean_position_delta: float,
) -> None:
    """Allow bounded drift when the same seeded model runs on a different environment.

    The Monte Carlo pipeline is stable on the reference environment used to
    generate the goldens, but minor platform/interpreter differences can still
    reshuffle near-tied ranks. Outside that reference environment we still want
    strong regression protection without requiring byte-for-byte identity.
    """
    metadata_keys = sorted(set(payload.keys()) - {row_key})
    assert metadata_keys == sorted(set(golden_payload.keys()) - {row_key})
    for key in metadata_keys:
        assert payload[key] == golden_payload[key]

    _assert_payload_shape(payload, row_key=row_key)
    _assert_payload_shape(golden_payload, row_key=row_key)

    payload_rows = {str(row["driver"]): row for row in payload[row_key]}
    golden_rows = {str(row["driver"]): row for row in golden_payload[row_key]}
    assert payload_rows.keys() == golden_rows.keys()

    position_deltas = [
        abs(int(payload_rows[driver]["position"]) - int(golden_rows[driver]["position"]))
        for driver in sorted(payload_rows)
    ]
    assert max(position_deltas) <= max_position_delta
    assert (sum(position_deltas) / len(position_deltas)) <= mean_position_delta

    assert payload[row_key][0]["driver"] == golden_payload[row_key][0]["driver"]


def _qualifying_payload() -> dict[str, Any]:
    """Build the fixed-seed qualifying payload used for regression checks."""
    predictor = _build_test_predictor()
    result = predictor.predict_qualifying(
        2026,
        "Australian Grand Prix",
        n_simulations=40,
        **_historical_context("Australian Grand Prix", "Q"),
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Australian Grand Prix",
        "n_simulations": 40,
        "grid": [_normalize_row(row) for row in result["grid"]],
    }


def _race_payload() -> dict[str, Any]:
    """Build the fixed-seed race payload used for regression checks."""
    predictor = _build_test_predictor()
    qualifying = predictor.predict_qualifying(
        2026,
        "Australian Grand Prix",
        n_simulations=40,
        **_historical_context("Australian Grand Prix", "Q"),
    )
    result = predictor.predict_race(
        qualifying["grid"],
        weather="dry",
        race_name="Australian Grand Prix",
        n_simulations=40,
        **_historical_context("Australian Grand Prix", "R"),
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Australian Grand Prix",
        "weather": "dry",
        "n_simulations": 40,
        "finish_order": [_normalize_row(row) for row in result["finish_order"]],
    }


def _sprint_qualifying_payload() -> dict[str, Any]:
    """Build the fixed-seed sprint qualifying payload used for regression checks."""
    predictor = _build_test_predictor()
    result = predictor.predict_qualifying(
        2026,
        "Chinese Grand Prix",
        n_simulations=40,
        qualifying_stage="sprint",
        **_historical_context("Chinese Grand Prix", "SQ"),
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Chinese Grand Prix",
        "n_simulations": 40,
        "qualifying_stage": "sprint",
        "grid": [_normalize_row(row) for row in result["grid"]],
    }


def _sprint_race_payload() -> dict[str, Any]:
    """Build the fixed-seed sprint race payload used for regression checks."""
    predictor = _build_test_predictor()
    sprint_qualifying = predictor.predict_qualifying(
        2026,
        "Chinese Grand Prix",
        n_simulations=40,
        qualifying_stage="sprint",
        **_historical_context("Chinese Grand Prix", "SQ"),
    )
    result = predictor.predict_sprint_race(
        sprint_qualifying["grid"],
        weather="dry",
        race_name="Chinese Grand Prix",
        n_simulations=40,
        **_historical_context("Chinese Grand Prix", "SPRINT"),
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Chinese Grand Prix",
        "weather": "dry",
        "n_simulations": 40,
        "finish_order": [_normalize_row(row) for row in result["finish_order"]],
    }


def _china_main_qualifying_payload() -> dict[str, Any]:
    """Build the fixed-seed main qualifying payload for a sprint weekend."""
    predictor = _build_test_predictor()
    result = predictor.predict_qualifying(
        2026,
        "Chinese Grand Prix",
        n_simulations=40,
        qualifying_stage="main",
        **_historical_context("Chinese Grand Prix", "Q"),
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Chinese Grand Prix",
        "n_simulations": 40,
        "qualifying_stage": "main",
        "grid": [_normalize_row(row) for row in result["grid"]],
    }


def _china_main_race_payload() -> dict[str, Any]:
    """Build the fixed-seed main race payload for a sprint weekend."""
    predictor = _build_test_predictor()
    main_qualifying = predictor.predict_qualifying(
        2026,
        "Chinese Grand Prix",
        n_simulations=40,
        qualifying_stage="main",
        **_historical_context("Chinese Grand Prix", "Q"),
    )
    result = predictor.predict_race(
        main_qualifying["grid"],
        weather="dry",
        race_name="Chinese Grand Prix",
        n_simulations=40,
        **_historical_context("Chinese Grand Prix", "R"),
    )
    return {
        "seed": 42,
        "year": 2026,
        "race_name": "Chinese Grand Prix",
        "weather": "dry",
        "n_simulations": 40,
        "finish_order": [_normalize_row(row) for row in result["finish_order"]],
    }


def _assert_matches_or_update(
    *,
    golden_path: Path,
    payload: dict[str, Any],
    update_golden_files: bool,
    row_key: str,
    max_position_delta: int,
    mean_position_delta: float,
) -> None:
    """Compare one payload to disk or rewrite the golden fixture if requested."""
    if update_golden_files or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if update_golden_files:
            return

    golden_payload = json.loads(golden_path.read_text())
    if REFERENCE_REGRESSION_ENV:
        assert payload == golden_payload, f"Prediction output changed: {golden_path}"
        return

    _assert_cross_environment_regression(
        payload=payload,
        golden_payload=golden_payload,
        row_key=row_key,
        max_position_delta=max_position_delta,
        mean_position_delta=mean_position_delta,
    )


def test_qualifying_regression(update_golden_files):
    """Fixed-seed Australia qualifying output should stay stable."""
    payload = _qualifying_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_qualifying_australia.json",
        payload=payload,
        update_golden_files=update_golden_files,
        row_key="grid",
        max_position_delta=5,
        mean_position_delta=2.0,
    )


def test_race_regression(update_golden_files):
    """Fixed-seed Australia race output should stay stable."""
    payload = _race_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_race_australia.json",
        payload=payload,
        update_golden_files=update_golden_files,
        row_key="finish_order",
        max_position_delta=5,
        mean_position_delta=2.5,
    )


def test_sprint_qualifying_regression(update_golden_files):
    """Fixed-seed China sprint qualifying output should stay stable."""
    payload = _sprint_qualifying_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_sprint_qualifying_china.json",
        payload=payload,
        update_golden_files=update_golden_files,
        row_key="grid",
        max_position_delta=5,
        mean_position_delta=2.0,
    )


def test_sprint_race_regression(update_golden_files):
    """Fixed-seed China sprint race output should stay stable."""
    payload = _sprint_race_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_sprint_race_china.json",
        payload=payload,
        update_golden_files=update_golden_files,
        row_key="finish_order",
        max_position_delta=5,
        mean_position_delta=2.5,
    )


def test_china_main_qualifying_regression(update_golden_files):
    """Fixed-seed China main qualifying output should stay stable."""
    payload = _china_main_qualifying_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_qualifying_china.json",
        payload=payload,
        update_golden_files=update_golden_files,
        row_key="grid",
        max_position_delta=5,
        mean_position_delta=2.0,
    )


def test_china_main_race_regression(update_golden_files):
    """Fixed-seed China main race output should stay stable."""
    payload = _china_main_race_payload()
    _assert_matches_or_update(
        golden_path=GOLDEN_DIR / "golden_race_china.json",
        payload=payload,
        update_golden_files=update_golden_files,
        row_key="finish_order",
        max_position_delta=5,
        mean_position_delta=2.5,
    )
