"""Unit tests for pit strategy generation and validation."""

from unittest.mock import patch

import numpy as np

from src.utils.pit_strategy import (
    _get_default_strategy,
    _sample_compound_sequence,
    generate_pit_strategy,
    validate_strategy,
)


class TestGeneratePitStrategy:
    """Test pit strategy generation."""

    def test_low_stress_favors_one_stop(self):
        """Low tire stress (Monaco) should favor 1-stop."""
        rng = np.random.default_rng(seed=42)
        one_stop_count = 0

        # Run 100 simulations
        for _ in range(100):
            strategy = generate_pit_strategy(
                race_distance=60,
                tire_stress_score=2.0,  # Low stress
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                rng=rng,
            )
            if strategy["num_stops"] == 1:
                one_stop_count += 1

        # Should be >90% 1-stop (per config: 95%)
        assert one_stop_count > 85

    def test_high_stress_favors_two_stop(self):
        """High tire stress (Bahrain) should favor 2-stop."""
        rng = np.random.default_rng(seed=42)
        two_stop_count = 0

        # Run 100 simulations
        for _ in range(100):
            strategy = generate_pit_strategy(
                race_distance=60,
                tire_stress_score=4.0,  # High stress
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                rng=rng,
            )
            if strategy["num_stops"] == 2:
                two_stop_count += 1

        # Should be >75% 2-stop (per config: 80%)
        assert two_stop_count > 70

    def test_strategy_structure(self):
        """Generated strategy should have correct structure."""
        rng = np.random.default_rng(seed=42)

        strategy = generate_pit_strategy(
            race_distance=60,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=rng,
        )

        # Check required fields
        assert "num_stops" in strategy
        assert "pit_laps" in strategy
        assert "compound_sequence" in strategy
        assert "stint_lengths" in strategy

        # Check field relationships
        assert len(strategy["pit_laps"]) == strategy["num_stops"]
        assert len(strategy["compound_sequence"]) == strategy["num_stops"] + 1
        assert sum(strategy["stint_lengths"]) == 60

    def test_fia_rule_enforcement(self):
        """Generated strategies must use ≥2 different compounds."""
        rng = np.random.default_rng(seed=42)

        for _ in range(50):
            strategy = generate_pit_strategy(
                race_distance=60,
                tire_stress_score=3.0,
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                rng=rng,
            )

            unique_compounds = set(strategy["compound_sequence"])
            assert len(unique_compounds) >= 2, "FIA rule: must use ≥2 compounds"

    def test_pit_lap_within_safety_margins(self):
        """Pit laps should respect min/max constraints."""
        rng = np.random.default_rng(seed=42)

        for _ in range(50):
            strategy = generate_pit_strategy(
                race_distance=60,
                tire_stress_score=3.0,
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                rng=rng,
            )

            for pit_lap in strategy["pit_laps"]:
                assert pit_lap >= 5, "Min pit lap = 5"
                assert pit_lap <= 55, "Max pit lap = race_distance - 5"

    def test_sprint_race_scaling(self):
        """Sprint race (20 laps) should have earlier pit windows."""
        rng = np.random.default_rng(seed=42)

        strategy = generate_pit_strategy(
            race_distance=20,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=rng,
        )

        # Pit windows should scale: 60-lap [23-37] → 20-lap [8-12]
        if strategy["num_stops"] == 1:
            pit_lap = strategy["pit_laps"][0]
            assert 5 <= pit_lap <= 15, "Sprint pit window should be early"

    def test_mixed_weather_can_select_intermediate(self):
        """Mixed-weather strategies should sometimes include INTERMEDIATE tires."""
        rng = np.random.default_rng(seed=42)
        intermediate_count = 0

        for _ in range(200):
            strategy = generate_pit_strategy(
                race_distance=57,
                tire_stress_score=3.0,
                available_compounds=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"],
                rng=rng,
                enforce_two_compound_rule=True,
            )
            intermediate_count += strategy["compound_sequence"].count("INTERMEDIATE")

        assert intermediate_count > 0


class TestValidateStrategy:
    """Test strategy validation."""

    def test_valid_one_stop_strategy(self):
        """Valid 1-stop strategy passes validation."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [30, 30],
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is True

    def test_valid_two_stop_strategy(self):
        """Valid 2-stop strategy passes validation."""
        strategy = {
            "num_stops": 2,
            "pit_laps": [20, 40],
            "compound_sequence": ["SOFT", "MEDIUM", "HARD"],
            "stint_lengths": [20, 20, 20],
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is True

    def test_missing_field_fails(self):
        """Strategy missing required field fails validation."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [30],
            # Missing compound_sequence and stint_lengths
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is False

    def test_fia_rule_violation_fails(self):
        """Strategy with <2 unique compounds fails."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "SOFT"],  # Same compound!
            "stint_lengths": [30, 30],
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is False

    def test_stint_lengths_mismatch_fails(self):
        """Stint lengths not summing to race distance fails."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [30, 25],  # Only 55 laps!
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is False

    def test_pit_lap_too_early_fails(self):
        """Pit lap before lap 5 fails."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [3],  # Too early!
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [3, 57],
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is False

    def test_pit_lap_too_late_fails(self):
        """Pit lap in last 5 laps fails."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [58],  # Too late!
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [58, 2],
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is False

    def test_unavailable_compound_fails(self):
        """Using unavailable compound fails."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "INTERMEDIATE"],  # WET not available
            "stint_lengths": [30, 30],
        }

        result = validate_strategy(strategy, 60, ["SOFT", "MEDIUM", "HARD"])
        assert result is False

    def test_single_compound_allowed_when_rule_disabled(self):
        """Wet-race validation can allow one compound throughout."""
        strategy = {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["INTERMEDIATE", "INTERMEDIATE"],
            "stint_lengths": [30, 30],
        }

        assert (
            validate_strategy(
                strategy,
                race_distance=60,
                available_compounds=["INTERMEDIATE", "WET"],
                enforce_two_compound_rule=False,
            )
            is True
        )


class TestSampleCompoundSequence:
    """Test compound sequence sampling."""

    def test_high_stress_prefers_hard(self):
        """High tire stress should prefer HARD compound."""
        rng = np.random.default_rng(seed=42)

        # Run 50 samples
        hard_starts = 0
        for _ in range(50):
            sequence = _sample_compound_sequence(
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                num_stops=1,  # 2 compounds needed
                tire_stress_score=4.0,  # High stress
                rng=rng,
            )
            if sequence[0] == "HARD" or sequence[1] == "HARD":
                hard_starts += 1

        # Should frequently include HARD
        assert hard_starts > 30

    def test_low_stress_prefers_soft(self):
        """Low tire stress should prefer SOFT compound."""
        rng = np.random.default_rng(seed=42)

        # Run 50 samples
        soft_starts = 0
        for _ in range(50):
            sequence = _sample_compound_sequence(
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                num_stops=1,  # 2 compounds needed
                tire_stress_score=2.0,  # Low stress
                rng=rng,
            )
            if sequence[0] == "SOFT":
                soft_starts += 1

        # Should frequently start with SOFT
        assert soft_starts > 25


class TestGetDefaultStrategy:
    """Test default fallback strategy."""

    def test_default_is_one_stop(self):
        """Default strategy should be 1-stop."""
        strategy = _get_default_strategy(60, ["SOFT", "MEDIUM", "HARD"])

        assert strategy["num_stops"] == 1
        assert len(strategy["pit_laps"]) == 1

    def test_default_pit_at_halfway(self):
        """Default pit should be ~50% race distance."""
        strategy = _get_default_strategy(60, ["SOFT", "MEDIUM", "HARD"])

        pit_lap = strategy["pit_laps"][0]
        assert 25 <= pit_lap <= 35  # Around lap 30

    def test_default_uses_two_compounds(self):
        """Default should use 2 different compounds."""
        strategy = _get_default_strategy(60, ["SOFT", "MEDIUM", "HARD"])

        unique_compounds = set(strategy["compound_sequence"])
        assert len(unique_compounds) >= 2


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_only_two_compounds_available(self):
        """Strategy generation with only 2 compounds."""
        rng = np.random.default_rng(seed=42)

        strategy = generate_pit_strategy(
            race_distance=60,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM"],  # Only 2
            rng=rng,
        )

        # Should still work
        assert len(set(strategy["compound_sequence"])) >= 2

    def test_very_short_race(self):
        """Strategy for very short race (e.g., 10 laps)."""
        rng = np.random.default_rng(seed=42)

        strategy = generate_pit_strategy(
            race_distance=10,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=rng,
        )

        # Should generate valid strategy even for short race
        assert sum(strategy["stint_lengths"]) == 10
        assert all(lap >= 5 and lap <= 5 for lap in strategy["pit_laps"])  # [5, 5]

    def test_reproducibility_with_seed(self):
        """Same seed should generate same strategy."""
        rng1 = np.random.default_rng(seed=123)
        strategy1 = generate_pit_strategy(
            race_distance=60,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=rng1,
        )

        rng2 = np.random.default_rng(seed=123)
        strategy2 = generate_pit_strategy(
            race_distance=60,
            tire_stress_score=3.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=rng2,
        )

        assert strategy1 == strategy2

    def test_wet_strategy_generation_allows_single_compound(self):
        """Wet strategy generation should not enforce dry FIA two-compound rule."""
        rng = np.random.default_rng(seed=42)
        strategy = generate_pit_strategy(
            race_distance=60,
            tire_stress_score=3.0,
            available_compounds=["INTERMEDIATE", "WET"],
            rng=rng,
            enforce_two_compound_rule=False,
        )

        assert validate_strategy(
            strategy,
            race_distance=60,
            available_compounds=["INTERMEDIATE", "WET"],
            enforce_two_compound_rule=False,
        )

    def test_monaco_like_track_biases_later_than_monza_like(self):
        """Hard-to-pass tracks should bias pit timing later than easy-to-pass tracks."""
        import src.utils.pit_strategy as pit_strategy_module

        def _config_get(key: str, default=None):
            overrides = {
                "baseline_predictor.race.tire_strategy.stop_probability.medium_stress_1stop": 1.0,
                "baseline_predictor.race.strategy_constraints.pit_lap_variance.one_stop": 0.0,
                "baseline_predictor.race.tire_strategy.windows.one_stop": [23, 37],
                "baseline_predictor.race.strategy_constraints.min_pit_lap": 5,
                "baseline_predictor.race.strategy_constraints.max_pit_lap_from_end": 5,
            }
            return overrides.get(key, default)

        with patch.object(pit_strategy_module.config_loader, "get", _config_get):
            monaco_strategy = generate_pit_strategy(
                race_distance=60,
                tire_stress_score=3.0,
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                rng=np.random.default_rng(7),
                track_overtaking=0.95,  # hard overtaking -> overcut tendency
                grid_position=5,
            )
            monza_strategy = generate_pit_strategy(
                race_distance=60,
                tire_stress_score=3.0,
                available_compounds=["SOFT", "MEDIUM", "HARD"],
                rng=np.random.default_rng(7),
                track_overtaking=0.20,  # easy overtaking -> undercut tendency
                grid_position=5,
            )

        assert monaco_strategy["num_stops"] == 1
        assert monza_strategy["num_stops"] == 1
        assert monaco_strategy["pit_laps"][0] > monza_strategy["pit_laps"][0]
