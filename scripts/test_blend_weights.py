"""Evaluate FP blend-weight candidates against real qualifying results."""

from src.data.actual_results_fetcher import fetch_actual_session_results
from src.predictors import Baseline2026Predictor
from src.utils.blend_weight_tester import test_blend_weights


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
        print("------------------")
        for weight, mae in sorted(results.items()):
            print(f"{weight:>11.2f} | {mae:.2f}")

        best_weight = min(results, key=results.get)
        print(f"Best weight: {best_weight:.2f} (MAE {results[best_weight]:.2f})")


if __name__ == "__main__":
    main()
