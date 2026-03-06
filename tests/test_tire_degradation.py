"""Unit tests for tire degradation and fuel effect modeling."""

import pytest

from src.utils.tire_degradation import (
    calculate_fuel_delta,
    calculate_tire_deg_delta,
    estimate_stint_pace_degradation,
    get_effective_tire_deg_slope,
    get_fresh_tire_advantage,
)


class TestCalculateTireDegDelta:
    """Test tire degradation calculation."""

    def test_zero_laps_returns_zero(self):
        """No degradation on lap 0."""
        result = calculate_tire_deg_delta(
            tire_deg_slope=0.15,
            laps_on_tire=0,
            fuel_load_kg=100.0,
        )
        assert result == 0.0

    def test_negative_slope_returns_zero(self):
        """Negative deg slope should return 0 (Mercedes case)."""
        result = calculate_tire_deg_delta(
            tire_deg_slope=-0.03,
            laps_on_tire=10,
            fuel_load_kg=100.0,
        )
        assert result == 0.0

    def test_linear_degradation(self):
        """Degradation increases linearly with laps."""
        # 0.15 s/lap slope, 10 laps, full fuel (110kg → ~1.1x multiplier)
        result = calculate_tire_deg_delta(
            tire_deg_slope=0.15,
            laps_on_tire=10,
            fuel_load_kg=110.0,
            initial_fuel_kg=110.0,
        )
        # Expected: 0.15 * 10 * 1.1 = 1.65s
        assert result == pytest.approx(1.65, abs=0.01)

    def test_fuel_effect_on_degradation(self):
        """Heavier car degrades tires faster."""
        # Full tank
        result_full = calculate_tire_deg_delta(
            tire_deg_slope=0.15,
            laps_on_tire=10,
            fuel_load_kg=110.0,
            initial_fuel_kg=110.0,
        )

        # Empty tank
        result_empty = calculate_tire_deg_delta(
            tire_deg_slope=0.15,
            laps_on_tire=10,
            fuel_load_kg=0.0,
            initial_fuel_kg=110.0,
        )

        # Full tank should have more degradation
        assert result_full > result_empty
        assert result_empty == pytest.approx(1.5, abs=0.01)  # 0.15 * 10 * 1.0

    def test_track_temperature_increases_degradation_on_hot_track(self):
        """Higher track temperature should increase degradation penalty."""
        cool_track = calculate_tire_deg_delta(
            tire_deg_slope=0.15,
            laps_on_tire=10,
            fuel_load_kg=80.0,
            initial_fuel_kg=110.0,
            track_temp=20.0,
        )
        hot_track = calculate_tire_deg_delta(
            tire_deg_slope=0.15,
            laps_on_tire=10,
            fuel_load_kg=80.0,
            initial_fuel_kg=110.0,
            track_temp=50.0,
        )

        assert hot_track > cool_track


class TestCalculateFuelDelta:
    """Test fuel weight effect on lap time."""

    def test_zero_laps_remaining_returns_zero(self):
        """No fuel penalty when race is done."""
        result = calculate_fuel_delta(laps_remaining=0)
        assert result == 0.0

    def test_fuel_penalty_decreases_over_race(self):
        """Fuel penalty should decrease as fuel burns."""
        # Start of race (60 laps remaining)
        result_start = calculate_fuel_delta(laps_remaining=60)

        # Mid race (30 laps remaining)
        result_mid = calculate_fuel_delta(laps_remaining=30)

        # End of race (5 laps remaining)
        result_end = calculate_fuel_delta(laps_remaining=5)

        assert result_start > result_mid > result_end
        # Start: 60*1.5 = 90kg → 9 * 0.035 = 0.315s
        assert result_start == pytest.approx(0.315, abs=0.01)


class TestGetFreshTireAdvantage:
    """Test fresh tire advantage calculation."""

    def test_soft_fresh_tire_advantage(self):
        """SOFT tires have 0.5s advantage on lap 1."""
        result = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=0)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_medium_fresh_tire_advantage(self):
        """MEDIUM tires have 0.3s advantage on lap 1."""
        result = get_fresh_tire_advantage(compound="MEDIUM", laps_on_tire=0)
        assert result == pytest.approx(0.3, abs=0.01)

    def test_hard_fresh_tire_advantage(self):
        """HARD tires have 0.1s advantage on lap 1."""
        result = get_fresh_tire_advantage(compound="HARD", laps_on_tire=0)
        assert result == pytest.approx(0.1, abs=0.01)

    def test_advantage_decays_linearly(self):
        """Advantage decreases linearly over fresh tire window."""
        # SOFT: 3-lap window
        lap0 = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=0)
        lap1 = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=1)
        lap2 = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=2)
        lap3 = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=3)

        assert lap0 > lap1 > lap2
        assert lap3 == 0.0  # After window

    def test_case_insensitive_compound(self):
        """Compound name should be case-insensitive."""
        result_upper = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=0)
        result_lower = get_fresh_tire_advantage(compound="soft", laps_on_tire=0)
        result_mixed = get_fresh_tire_advantage(compound="Soft", laps_on_tire=0)

        assert result_upper == result_lower == result_mixed

    def test_track_temperature_reduces_fresh_tire_advantage_away_from_optimal(self):
        """Fresh-tire gain should fall when track temperature is far from optimum."""
        optimal_track = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=0, track_temp=30.0)
        very_hot_track = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=0, track_temp=55.0)
        very_cold_track = get_fresh_tire_advantage(compound="SOFT", laps_on_tire=0, track_temp=5.0)

        assert optimal_track > very_hot_track
        assert optimal_track > very_cold_track


class TestGetEffectiveTireDegSlope:
    """Test traffic-based tire degradation adjustment."""

    def test_clean_air_bonus_for_leaders(self):
        """P1-P5 get 5% better tire life."""
        base_slope = 0.15

        # Leader (P1)
        result_p1 = get_effective_tire_deg_slope(base_slope, traffic_position=1)
        assert result_p1 < base_slope
        assert result_p1 == pytest.approx(0.1425, abs=0.001)  # 0.15 * 0.95

        # P5
        result_p5 = get_effective_tire_deg_slope(base_slope, traffic_position=5)
        assert result_p5 == result_p1

    def test_midfield_neutral(self):
        """P6-P15 have neutral tire life."""
        base_slope = 0.15

        result = get_effective_tire_deg_slope(base_slope, traffic_position=10)
        assert result == base_slope

    def test_traffic_penalty_for_backmarkers(self):
        """P16+ get 5% worse tire life."""
        base_slope = 0.15

        result = get_effective_tire_deg_slope(base_slope, traffic_position=18)
        assert result > base_slope
        assert result == pytest.approx(0.1575, abs=0.001)  # 0.15 * 1.05

    def test_zero_cars_returns_base(self):
        """Edge case: 0 total cars returns base slope."""
        base_slope = 0.15
        result = get_effective_tire_deg_slope(base_slope, traffic_position=1, total_cars=0)
        assert result == base_slope

    def test_missing_base_slope_uses_config_default(self, monkeypatch):
        """Missing slope should fall back to configured default instead of crashing."""
        import src.utils.tire_degradation as tire_degradation

        def _mock_config_get(key, default=None):
            if key == "baseline_predictor.race.tire_physics.default_deg_slope":
                return 0.21
            return default

        monkeypatch.setattr(tire_degradation.config_loader, "get", _mock_config_get)

        result = get_effective_tire_deg_slope(None, traffic_position=10)
        assert result == pytest.approx(0.21, abs=0.001)


class TestIntegrationScenarios:
    """Integration tests combining multiple effects."""

    def test_full_stint_degradation(self):
        """Test realistic stint progression."""
        tire_deg_slope = 0.15
        initial_fuel = 110.0

        # Lap 1: fresh tires, full fuel
        lap1_deg = calculate_tire_deg_delta(tire_deg_slope, 1, 110.0, initial_fuel)
        lap1_fresh = get_fresh_tire_advantage("SOFT", 0)
        lap1_fuel = calculate_fuel_delta(60)

        # Lap 1 should have low net penalty (fresh tire advantage > deg + fuel)
        lap1_net = lap1_deg - lap1_fresh + lap1_fuel
        assert lap1_net < 0.5  # Faster early in stint

        # Lap 20: worn tires, less fuel
        lap20_deg = calculate_tire_deg_delta(tire_deg_slope, 20, 50.0, initial_fuel)
        lap20_fresh = get_fresh_tire_advantage("SOFT", 20)  # 0 (past window)
        lap20_fuel = calculate_fuel_delta(40)

        # Lap 20 should have higher penalty (no fresh advantage, high deg)
        lap20_net = lap20_deg - lap20_fresh + lap20_fuel
        assert lap20_net > 2.0  # Slower late in stint

    def test_red_bull_high_deg_scenario(self):
        """Test Red Bull's high SOFT degradation (0.421 s/lap)."""
        tire_deg_slope = 0.421

        # After 15 laps
        result = calculate_tire_deg_delta(
            tire_deg_slope=tire_deg_slope,
            laps_on_tire=15,
            fuel_load_kg=80.0,
            initial_fuel_kg=110.0,
        )

        # 0.421 * 15 * ~1.07 = ~6.75s penalty
        assert result > 6.0
        assert result < 7.5

    def test_mercedes_negative_deg_scenario(self):
        """Test Mercedes' negative degradation (tires improve)."""
        tire_deg_slope = -0.030

        # Should always return 0 (handled by function)
        result = calculate_tire_deg_delta(
            tire_deg_slope=tire_deg_slope,
            laps_on_tire=15,
            fuel_load_kg=80.0,
        )

        assert result == 0.0

    def test_stint_estimator_uses_lap_zero_tire_age(self):
        """Stint estimator should include lap-0 fresh-tire behavior."""
        short_stint = estimate_stint_pace_degradation(
            tire_deg_slope=0.15,
            stint_length=1,
            compound="SOFT",
            fuel_load_start_kg=110.0,
            track_temp=30.0,
        )
        longer_stint = estimate_stint_pace_degradation(
            tire_deg_slope=0.15,
            stint_length=5,
            compound="SOFT",
            fuel_load_start_kg=110.0,
            track_temp=30.0,
        )

        # One-lap stint should carry near-zero degradation due to fresh tire lap.
        assert short_stint == pytest.approx(0.0, abs=0.001)
        assert longer_stint > short_stint
