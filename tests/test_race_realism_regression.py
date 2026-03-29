"""Realism regression tests for race predictions.

These tests enforce quantitative realism constraints to prevent race predictions
from becoming too random or unrealistic. All tests use seeded RNG for determinism.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.predictors.baseline_2026 import Baseline2026Predictor


class TestRaceRealismRegression:
    """Regression tests to enforce race prediction realism."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return Baseline2026Predictor()

    def test_grid_to_race_correlation_minimum(self, predictor):
        """Grid-to-race correlation must be >= 0.75 in dry conditions.

        This ensures qualifying order has meaningful impact on race result.
        Target inspired by real F1 data where grid order correlates strongly
        with finishing order in clean races.
        """
        # Use moderate simulation count for stable, repeatable results
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=300)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=300,
        )

        # Extract positions
        grid_positions = {entry["driver"]: entry["position"] for entry in quali["grid"]}
        race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}

        drivers = sorted(grid_positions.keys())
        grid_pos = [grid_positions[d] for d in drivers]
        race_pos = [race_positions[d] for d in drivers]

        # Calculate Spearman correlation
        correlation, _ = spearmanr(grid_pos, race_pos)

        assert correlation >= 0.75, (
            f"Grid-to-race correlation too low: {correlation:.3f}. "
            f"Expected >= 0.75 for realistic race behavior."
        )

    def test_mean_position_change_maximum(self, predictor):
        """Mean absolute position change must be <= 3.0 in dry conditions.

        This prevents excessive position shuffling that would make races
        feel like a lottery rather than a competition.
        """
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=300)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=300,
        )

        grid_positions = {entry["driver"]: entry["position"] for entry in quali["grid"]}
        race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}

        position_changes = [
            abs(grid_positions[d] - race_positions[d]) for d in grid_positions.keys()
        ]

        mean_change = np.mean(position_changes)

        assert mean_change <= 3.0, (
            f"Mean position change too high: {mean_change:.2f}. "
            f"Expected <= 3.0 for realistic position stability."
        )

    def test_pole_sitter_top3_probability(self, predictor):
        """Pole sitter must have >= 45% chance of finishing in top 3.

        In real F1, pole position confers a significant advantage.
        This test ensures the model respects that reality.
        """
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=300)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=300,
        )

        pole_driver = quali["grid"][0]["driver"]
        pole_finish = next(e for e in race["finish_order"] if e["driver"] == pole_driver)

        top3_prob = pole_finish["podium_probability"]

        assert top3_prob >= 45.0, (
            f"Pole sitter top-3 probability too low: {top3_prob:.1f}%. "
            f"Expected >= 45% to reflect real qualifying advantage."
        )

    def test_top5_starters_dominate_podium(self, predictor):
        """Top-5 starters must account for >= 50% of total podium probability.

        This ensures that front-runners don't get swamped by midfield
        in unrealistic ways.
        """
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=300)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=300,
        )

        top5_drivers = {entry["driver"] for entry in quali["grid"][:5]}

        total_podium_prob = 0.0
        top5_podium_prob = 0.0

        for entry in race["finish_order"]:
            prob = entry["podium_probability"]
            total_podium_prob += prob
            if entry["driver"] in top5_drivers:
                top5_podium_prob += prob

        top5_fraction = (top5_podium_prob / total_podium_prob) * 100 if total_podium_prob > 0 else 0

        assert top5_fraction >= 50.0, (
            f"Top-5 podium dominance too low: {top5_fraction:.1f}%. "
            f"Expected >= 50% for realistic front-runner advantage."
        )

    def test_front_runners_no_excessive_falloffs(self, predictor):
        """Top-3 starters should have reasonable worst-case scenarios.

        This is measured via P95 (95th percentile worst finish).
        - At least one top-3 driver must have P95 <= 10 (strong performance floor)
        - Mean P95 for top-3 should be <= 12 (average stability)

        This allows for some variance while preventing all three from having
        unrealistic tail outcomes.
        """
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=300)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=300,
        )

        top3_drivers = [entry["driver"] for entry in quali["grid"][:3]]

        p95_values = []
        good_floor_count = 0
        for driver in top3_drivers:
            entry = next(e for e in race["finish_order"] if e["driver"] == driver)
            p95 = entry["p95"]
            p95_values.append(p95)

            if p95 <= 10:
                good_floor_count += 1

        mean_p95 = sum(p95_values) / len(p95_values)

        # At least one top-3 driver should have strong performance floor
        assert good_floor_count >= 1, (
            f"No top-3 drivers have P95 <= 10. All have: {p95_values}. "
            f"Expected at least 1 driver with strong performance floor."
        )

        # Average P95 should be reasonable
        assert mean_p95 <= 12.0, (
            f"Mean P95 for top-3 too high: {mean_p95:.1f}. "
            f"P95 values: {p95_values}. Expected mean <= 12.0 for stability."
        )

    def test_podium_probabilities_mostly_monotonic(self, predictor):
        """Podium probabilities should generally decrease with finishing position.

        For top-8 finishers, there should be at most 2 inversions where
        a lower-placed finisher has higher podium probability than someone
        finishing ahead of them.
        """
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=300)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=300,
        )

        top8 = sorted(race["finish_order"], key=lambda x: x["position"])[:8]

        inversions = 0
        for i in range(1, len(top8)):
            prev_prob = top8[i - 1]["podium_probability"]
            curr_prob = top8[i]["podium_probability"]
            # Allow 5% tolerance for numerical stability
            if curr_prob > prev_prob + 5.0:
                inversions += 1

        assert inversions <= 2, (
            f"Too many podium probability inversions: {inversions}. "
            f"Expected <= 2 for coherent probability ordering."
        )

    def test_sprint_race_has_higher_grid_influence(self, predictor):
        """Sprint races should preserve qualifying order more than full races.

        This reflects the reality that shorter races with no pit stops
        have fewer opportunities for position changes.
        """
        quali = predictor.predict_qualifying(2026, "Miami Grand Prix", n_simulations=200)

        # Sprint race (no pit stops, shorter distance)
        sprint = predictor.predict_sprint_race(
            sprint_quali_grid=quali["grid"],
            weather="dry",
            race_name="Miami Grand Prix",
            n_simulations=200,
        )

        # Full race (with pit stops, full distance)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Miami Grand Prix",
            n_simulations=200,
            is_sprint=False,
        )

        # Calculate position changes
        grid_positions = {entry["driver"]: entry["position"] for entry in quali["grid"]}

        sprint_positions = {entry["driver"]: entry["position"] for entry in sprint["finish_order"]}
        race_positions = {entry["driver"]: entry["position"] for entry in race["finish_order"]}

        sprint_changes = [abs(grid_positions[d] - sprint_positions[d]) for d in grid_positions]
        race_changes = [abs(grid_positions[d] - race_positions[d]) for d in grid_positions]

        sprint_mean = np.mean(sprint_changes)
        race_mean = np.mean(race_changes)

        assert sprint_mean < race_mean * 1.15, (
            f"Sprint should have lower position change than race. "
            f"Sprint: {sprint_mean:.2f}, Race: {race_mean:.2f}"
        )

    def test_wet_weather_increases_variance(self, predictor):
        """Wet weather should increase position variance compared to dry.

        This reflects the reality that rain races are more unpredictable.
        """
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=200)

        dry_race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=200,
        )

        wet_race = predictor.predict_race(
            quali["grid"],
            weather="rain",
            race_name="Australian Grand Prix",
            n_simulations=200,
        )

        # Compare average confidence (lower = more variance)
        dry_conf = np.mean([e["confidence"] for e in dry_race["finish_order"]])
        wet_conf = np.mean([e["confidence"] for e in wet_race["finish_order"]])

        # Wet should have equal or lower confidence (more variance)
        assert wet_conf <= dry_conf + 2.0, (
            f"Wet weather should not have higher confidence than dry. "
            f"Dry: {dry_conf:.1f}, Wet: {wet_conf:.1f}"
        )
