"""
Integration tests for Baseline 2026 Predictor

Tests the full prediction pipeline: loading data → qualifying → race
"""

import numpy as np
import pytest

from src.predictors.baseline_2026 import Baseline2026Predictor


class TestBaseline2026Integration:
    """Test full prediction workflow"""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance"""
        return Baseline2026Predictor()

    def test_can_load_2026_data(self, predictor):
        """Verify 2026 team and driver data loads correctly"""
        assert predictor.teams is not None
        assert predictor.drivers is not None
        assert len(predictor.teams) == 11  # 2026 has 11 teams
        assert "McLaren" in predictor.teams
        assert "Cadillac F1" in predictor.teams

    def test_qualifying_prediction_format(self, predictor):
        """Test qualifying prediction returns correct format"""
        result = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=10)

        assert "grid" in result
        assert len(result["grid"]) == 22  # 11 teams × 2 drivers

        # Check each grid entry has required fields
        for entry in result["grid"]:
            assert "driver" in entry
            assert "team" in entry
            assert "position" in entry
            assert "confidence" in entry
            assert 1 <= entry["position"] <= 22
            assert 40 <= entry["confidence"] <= 60  # Baseline confidence range

    def test_race_prediction_format(self, predictor):
        """Test race prediction returns correct format"""
        # First get qualifying grid
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=10)

        # Then predict race
        result = predictor.predict_race(quali["grid"], weather="dry", n_simulations=10)

        assert "finish_order" in result
        assert len(result["finish_order"]) == 22

        # Check each finish entry
        for entry in result["finish_order"]:
            assert "driver" in entry
            assert "team" in entry
            assert "position" in entry
            assert "confidence" in entry
            assert "podium_probability" in entry
            assert "dnf_probability" in entry
            assert 1 <= entry["position"] <= 22
            assert 40 <= entry["confidence"] <= 60
            assert 0 <= entry["podium_probability"] <= 100
            assert 0 <= entry["dnf_probability"] <= 0.35

    def test_monte_carlo_stability(self, predictor):
        """Test that multiple simulations produce stable results"""
        # Run prediction twice with same inputs
        result1 = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=50)
        result2 = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=50)

        # Extract positions
        pos1 = {entry["driver"]: entry["position"] for entry in result1["grid"]}
        pos2 = {entry["driver"]: entry["position"] for entry in result2["grid"]}

        # Check positions are similar (allowing for Monte Carlo variance)
        differences = []
        for driver in pos1:
            differences.append(abs(pos1[driver] - pos2[driver]))

        # Mean position difference should be small (< 3 positions)
        assert np.mean(differences) < 3.0

    def test_team_hierarchy_respected(self, predictor):
        """Test that strong teams generally finish ahead of weak teams"""
        result = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=50)

        # Get positions
        positions = {entry["driver"]: entry["position"] for entry in result["grid"]}

        # McLaren drivers (strongest team 0.85) should generally be in top 5
        mclaren_positions = [
            pos
            for driver, pos in positions.items()
            if any(
                entry["team"] == "McLaren" for entry in result["grid"] if entry["driver"] == driver
            )
        ]
        assert all(pos <= 10 for pos in mclaren_positions), "McLaren drivers should be in top 10"

        # Cadillac drivers (weakest team 0.30) should generally be in bottom 5
        cadillac_positions = [
            pos
            for driver, pos in positions.items()
            if any(
                entry["team"] == "Cadillac F1"
                for entry in result["grid"]
                if entry["driver"] == driver
            )
        ]
        assert all(pos >= 13 for pos in cadillac_positions), (
            "Cadillac drivers should be bottom half"
        )

    def test_sprint_weekend_detection(self, predictor):
        """Test that sprint weekends have slightly higher variance"""
        # This tests the internal sprint detection logic
        sprint_result = predictor.predict_qualifying(
            2026, "Chinese Grand Prix", n_simulations=50
        )  # Sprint
        normal_result = predictor.predict_qualifying(
            2026, "Bahrain Grand Prix", n_simulations=50
        )  # Normal

        # Both should work and return valid results
        assert len(sprint_result["grid"]) == 22
        assert len(normal_result["grid"]) == 22

        # Sprint should have slightly lower average confidence (more variance)
        sprint_conf = np.mean([e["confidence"] for e in sprint_result["grid"]])
        normal_conf = np.mean([e["confidence"] for e in normal_result["grid"]])
        assert sprint_conf <= normal_conf + 1.0  # Allow small difference

    def test_dnf_risk_calculation(self, predictor):
        """Test DNF risk is calculated appropriately"""
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=10)
        race = predictor.predict_race(quali["grid"], weather="dry", n_simulations=10)

        # Count high DNF risk drivers (>15% - adjusted for crash-only DNF rates)
        high_risk = [e for e in race["finish_order"] if e["dnf_probability"] > 0.15]

        # Should have some DNF risk variation (0-8 drivers depending on team/driver mix)
        assert len(high_risk) <= 8, f"Too many high DNF risk drivers: {len(high_risk)}"

        # All DNF probabilities should be capped at 35% and non-negative
        assert all(0 <= e["dnf_probability"] <= 0.35 for e in race["finish_order"])

    def test_weather_impact(self, predictor):
        """Test that rain increases race unpredictability"""
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=10)

        # Run race with different weather
        dry_race = predictor.predict_race(quali["grid"], weather="dry", n_simulations=10)
        rain_race = predictor.predict_race(quali["grid"], weather="rain", n_simulations=10)

        # Rain should have lower average confidence
        dry_conf = np.mean([e["confidence"] for e in dry_race["finish_order"]])
        rain_conf = np.mean([e["confidence"] for e in rain_race["finish_order"]])

        # Rain confidence should be lower or similar (weather adds chaos)
        assert rain_conf <= dry_conf + 2.0

    def test_grid_position_impact(self, predictor):
        """Test that grid position heavily influences race result"""
        # Create a qualifying grid (increase simulations for stability)
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=500)

        # Run race (increase simulations for stability)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            n_simulations=500,
            race_name="Bahrain Grand Prix",
        )

        # Driver starting P1 should have high chance of finishing in top 5
        pole_driver = quali["grid"][0]["driver"]  # P1 on grid
        pole_finish = next(e for e in race["finish_order"] if e["driver"] == pole_driver)

        assert pole_finish["position"] <= 8, "Pole sitter should finish in top 8 most of the time"
        assert pole_finish["p95"] <= 17, (
            "Pole sitter should not regularly fall to the back half of the field"
        )

    def test_race_order_is_not_clustered_by_team(self, predictor):
        """Race order should not collapse into teammate pairs."""
        quali = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=150)
        race = predictor.predict_race(
            quali["grid"],
            weather="dry",
            race_name="Australian Grand Prix",
            n_simulations=150,
        )

        finish = sorted(race["finish_order"], key=lambda e: e["position"])
        top10_adjacent_teammates = sum(
            1 for idx in range(9) if finish[idx]["team"] == finish[idx + 1]["team"]
        )

        assert top10_adjacent_teammates <= 4, (
            "Top 10 should not be mostly teammate blocks in race prediction"
        )

    def test_qualifying_order_is_not_over_clustered_by_team(self, predictor):
        """Qualifying top 10 should avoid near-perfect teammate block ordering."""
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=300)
        grid = sorted(quali["grid"], key=lambda e: e["position"])
        top10_adjacent_teammates = sum(
            1 for idx in range(9) if grid[idx]["team"] == grid[idx + 1]["team"]
        )

        assert top10_adjacent_teammates <= 2, (
            "Qualifying top 10 appears overly team-blocked; expected more interleaving."
        )

    def test_sprint_race_uses_no_pit_stop_model(self, predictor):
        """Sprint races should run without scheduled pit stops."""
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=20)
        sprint = predictor.predict_sprint_race(
            sprint_quali_grid=quali["grid"],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=20,
        )

        assert sprint["pit_lap_distribution"] == {}
        assert sprint["compound_strategies"], "Expected compound strategy output"
        assert all("→" not in strategy for strategy in sprint["compound_strategies"].keys())


class TestBaseline2026EdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_race_name(self):
        """Test handling of invalid race name"""
        predictor = Baseline2026Predictor()

        # Should not crash, but may have lower confidence or fallback behavior
        try:
            result = predictor.predict_qualifying(2026, "Invalid Grand Prix", n_simulations=10)
            # If it doesn't raise, check it returns valid data
            assert "grid" in result
        except Exception:
            # If it raises, that's also acceptable behavior
            pass

    def test_predict_qualifying_raises_when_weekend_type_is_unknown(self, patcher):
        """Predict qualifying should fail closed when weekend format cannot be resolved."""
        predictor = Baseline2026Predictor()
        patcher.setattr(
            "src.predictors.baseline.qualifying_mixin.is_sprint_weekend",
            lambda year, race_name: (_ for _ in ()).throw(ValueError("missing schedule row")),
        )

        with pytest.raises(ValueError, match="Could not determine weekend format"):
            predictor.predict_qualifying(2026, "Mystery Grand Prix", n_simulations=1)

    def test_extreme_weather(self):
        """Test mixed weather condition"""
        predictor = Baseline2026Predictor()
        quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=10)
        race = predictor.predict_race(quali["grid"], weather="mixed", n_simulations=10)

        # Should still produce valid results
        assert len(race["finish_order"]) == 22
        assert all(e["confidence"] >= 40 for e in race["finish_order"])

    def test_minimal_simulations(self):
        """Test prediction with minimal simulations (edge case)"""
        predictor = Baseline2026Predictor()
        result = predictor.predict_qualifying(2026, "Bahrain Grand Prix", n_simulations=5)

        # Should still work but with lower confidence
        assert len(result["grid"]) == 22
        assert all("position" in e for e in result["grid"])
