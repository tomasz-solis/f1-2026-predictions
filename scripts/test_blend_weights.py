"""Evaluate FP blend-weight candidates against real qualifying results."""

from __future__ import annotations

import logging
from typing import Any

from src.data.actual_results_fetcher import fetch_actual_session_results
from src.predictors import Baseline2026Predictor

logger = logging.getLogger(__name__)


def _set_nested_config_value(config_obj: Any, dotted_key: str, value: Any) -> bool:
    """Set a nested config value for Config-like objects."""
    storage = None
    for attr in ("_config", "_data"):
        candidate = getattr(config_obj, attr, None)
        if isinstance(candidate, dict):
            storage = candidate
            break

    if storage is None:
        return False

    keys = dotted_key.split(".")
    cursor = storage
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value
    return True


def _calculate_grid_mae(predicted: list[dict], actual: list[dict]) -> float:
    """Calculate mean absolute position error for shared drivers."""
    actual_positions = {entry["driver"]: entry["position"] for entry in actual}
    errors = []
    for prediction in predicted:
        driver = prediction.get("driver")
        if driver not in actual_positions:
            continue
        errors.append(abs(prediction["position"] - actual_positions[driver]))

    return (sum(errors) / len(errors)) if errors else 0.0


def test_blend_weights(
    predictor: Any,
    year: int,
    race_name: str,
    actual_quali_grid: list[dict],
    blend_weights: list[float] | None = None,
) -> dict[float, float]:
    """Evaluate candidate FP blend weights and return MAE for each."""
    if not actual_quali_grid:
        raise ValueError("actual_quali_grid cannot be empty")

    if blend_weights is None:
        blend_weights = [0.5, 0.6, 0.7, 0.8, 0.9]

    config_key = "baseline_predictor.qualifying.fp_blend_weight"
    original_weight = predictor.config.get(config_key, 0.7)
    results: dict[float, float] = {}

    for weight in blend_weights:
        updated = _set_nested_config_value(predictor.config, config_key, weight)
        if not updated:
            logger.warning("Could not set predictor blend weight dynamically; stopping sweep")
            break

        prediction = predictor.predict_qualifying(
            year=year,
            race_name=race_name,
            n_simulations=50,
        )
        mae = _calculate_grid_mae(prediction["grid"], actual_quali_grid)
        results[weight] = mae
        logger.info("Blend weight %.2f: MAE %.2f", weight, mae)

    _set_nested_config_value(predictor.config, config_key, original_weight)
    return results


def main() -> None:
    predictor = Baseline2026Predictor(seed=42)
    predictor.load_data()

    races = [
        "Australian Grand Prix",
        "Chinese Grand Prix",
        "Japanese Grand Prix",
    ]

    for race in races:
        print(f"\n=== {race} ===")
        actual_grid = fetch_actual_session_results(2026, race, "Q")
        if not actual_grid:
            print("No actual qualifying grid available yet.")
            continue

        results = test_blend_weights(
            predictor=predictor,
            year=2026,
            race_name=race,
            actual_quali_grid=actual_grid,
        )
        if not results:
            print("No blend-weight results were produced.")
            continue

        print("Blend Weight | MAE")
        print()
        for weight, mae in sorted(results.items()):
            print(f"{weight:>11.2f} | {mae:.2f}")

        best_weight = min(results, key=results.get)
        print(f"Best weight: {best_weight:.2f} (MAE {results[best_weight]:.2f})")


if __name__ == "__main__":
    main()
