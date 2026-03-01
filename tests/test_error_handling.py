"""Negative-path tests for validation and error handling."""

import numpy as np
import pytest

from src.extractors.validation import (
    validate_fp_team_order,
    validate_session_positions,
    validate_team_pace_data,
)
from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap
from src.utils.tire_degradation import (
    calculate_tire_deg_delta,
    get_fresh_tire_advantage,
)


# ---------------------------------------------------------------------------
# Original test (kept -- uses lazy import to avoid supabase chain in CI)
# ---------------------------------------------------------------------------
def test_prepare_driver_info_with_unknown_driver_raises():
    from src.predictors.baseline.race.preparation_mixin import BaselineRacePreparationMixin

    class _DummyPreparation(BaselineRacePreparationMixin):
        def __init__(self):
            self.teams = {"McLaren": {"uncertainty": 0.2, "compound_characteristics": {}}}
            self.drivers = {}

        def get_blended_team_strength(self, team: str, race_name: str) -> float:
            return 0.5

        def _compute_testing_profile_modifier(
            self,
            team: str,
            profile: str,
            metric_weights: dict[str, float],
            scale: float,
        ) -> tuple[float, bool]:
            return 0.0, False

    prep = _DummyPreparation()

    with pytest.raises(ValueError, match="Driver .* not found"):
        prep._prepare_driver_info_with_compounds(
            qualifying_grid=[{"driver": "UNKNOWN", "team": "McLaren", "position": 1}],
            race_name="Bahrain Grand Prix",
        )


# ---------------------------------------------------------------------------
# Tire degradation edge cases
# ---------------------------------------------------------------------------
class TestTireDegEdgeCases:
    def test_zero_slope_returns_zero(self):
        assert calculate_tire_deg_delta(0.0, 10, 80.0) == 0.0

    def test_negative_slope_returns_zero(self):
        assert calculate_tire_deg_delta(-0.1, 10, 80.0) == 0.0

    def test_zero_laps_returns_zero(self):
        assert calculate_tire_deg_delta(0.15, 0, 80.0) == 0.0

    def test_negative_laps_returns_zero(self):
        assert calculate_tire_deg_delta(0.15, -5, 80.0) == 0.0

    def test_cliff_increases_degradation(self):
        """Degradation beyond max age should be greater than at max age."""
        within_cliff = calculate_tire_deg_delta(0.15, 20, 80.0, compound="SOFT")
        beyond_cliff = calculate_tire_deg_delta(0.15, 30, 80.0, compound="SOFT")
        assert beyond_cliff > within_cliff

    def test_cliff_without_compound_is_linear(self):
        """Without compound info, degradation stays linear (backwards compatible)."""
        at_25 = calculate_tire_deg_delta(0.15, 25, 80.0)
        at_30 = calculate_tire_deg_delta(0.15, 30, 80.0)
        # Linear: ratio should be 30/25 = 1.2
        assert pytest.approx(at_30 / at_25, rel=0.01) == 30 / 25

    def test_fresh_tire_advantage_unknown_compound(self):
        """Unknown compound should return zero advantage."""
        assert get_fresh_tire_advantage("ULTRASOFT", 0) == 0.0

    def test_fresh_tire_advantage_past_window(self):
        """Past the fresh tire window, advantage should be zero."""
        assert get_fresh_tire_advantage("SOFT", 10) == 0.0


# ---------------------------------------------------------------------------
# Simulator edge cases
# ---------------------------------------------------------------------------
class TestSimulatorEdgeCases:
    def _base_params(self):
        return {
            "fuel": {
                "initial_load_kg": 100.0,
                "effect_per_lap": 0.0,
                "burn_rate_kg_per_lap": 1.5,
            },
            "lap_time": {
                "reference_base": 90.0,
                "team_pace_penalty_range": 1.0,
                "skill_improvement_max": 0.0,
                "bounds": [70.0, 120.0],
            },
            "team_strength_compression": 1.0,
            "race_advantage_lap_impact": 0.0,
            "start_grid_gap_seconds": 0.4,
            "base_chaos": {"dry": 0.0, "wet": 0.0},
            "lap1_chaos": {
                "front_row": 0.0,
                "upper_midfield": 0.0,
                "midfield": 0.0,
                "back_field": 0.0,
            },
            "pit_stops": {
                "loss_duration": 22.0,
                "overtake_loss_range": [0.0, 0.0],
            },
            "sc_probability": 0.0,
            "safety_car_luck_range": 0.0,
            "teammate_variance_std": 0.0,
            "track_overtaking": 0.5,
            "overtake_model": {
                "dirty_air_window_s": 1.8,
                "dirty_air_penalty_base": 0.0,
                "dirty_air_penalty_track_scale": 0.0,
                "pass_window_s": 1.2,
                "pass_threshold_base": 0.1,
                "pass_threshold_track_scale": 0.0,
                "pass_probability_base": 0.0,
                "pass_probability_scale": 0.0,
                "pass_time_bonus_range": [0.1, 0.1],
                "pace_diff_scale": 0.5,
                "skill_scale": 0.2,
                "race_adv_scale": 0.2,
                "track_ease_scale": 0.2,
            },
        }

    def _driver_info(self, grid_pos=1, team_strength=0.5):
        return {
            "grid_pos": grid_pos,
            "dnf_probability": 0.0,
            "team_strength": team_strength,
            "team_strength_by_compound": {"MEDIUM": team_strength},
            "tire_deg_by_compound": {"MEDIUM": 0.0},
            "skill": 0.5,
            "race_advantage": 0.0,
            "overtaking_skill": 0.5,
        }

    def _strategy(self):
        return {
            "num_stops": 0,
            "pit_laps": [],
            "compound_sequence": ["MEDIUM"],
            "stint_lengths": [60],
        }

    def test_single_lap_race_completes(self):
        """A 1-lap race should produce valid results."""
        result = simulate_race_lap_by_lap(
            driver_info_map={"A": self._driver_info()},
            strategies={"A": self._strategy()},
            race_params=self._base_params(),
            race_distance=1,
            weather="dry",
            rng=np.random.default_rng(0),
        )
        assert result["finish_order"] == ["A"]
        assert result["dnf_drivers"] == []

    def test_high_dnf_race(self):
        """With high DNF probability, some drivers should DNF and all should appear in finish_order."""
        info = {"A": self._driver_info(), "B": self._driver_info(grid_pos=2)}
        # DNF probability is spread per-lap (p/race_distance per lap), so 1.0
        # doesn't guarantee DNF on every seed. Use a long race to make it likely.
        info["A"]["dnf_probability"] = 1.0
        info["B"]["dnf_probability"] = 1.0
        strategies = {"A": self._strategy(), "B": self._strategy()}

        dnf_counts = 0
        n_trials = 20
        for seed in range(n_trials):
            result = simulate_race_lap_by_lap(
                driver_info_map=info,
                strategies=strategies,
                race_params=self._base_params(),
                race_distance=30,
                weather="dry",
                rng=np.random.default_rng(seed),
            )
            # All drivers must always appear in finish_order regardless of DNF
            assert set(result["finish_order"]) == {"A", "B"}
            dnf_counts += len(result["dnf_drivers"])

        # With p=1.0 over 30 laps, most drivers should DNF most of the time
        assert dnf_counts > n_trials, "Expected frequent DNFs with probability 1.0"

    def test_wet_weather_accepted(self):
        """Wet weather flag should be accepted without error."""
        result = simulate_race_lap_by_lap(
            driver_info_map={"A": self._driver_info()},
            strategies={"A": self._strategy()},
            race_params=self._base_params(),
            race_distance=5,
            weather="rain",
            rng=np.random.default_rng(0),
        )
        assert len(result["finish_order"]) == 1


# ---------------------------------------------------------------------------
# Extraction validation edge cases
# ---------------------------------------------------------------------------
class TestExtractionValidation:
    def test_empty_positions_warns(self):
        warnings = validate_session_positions({})
        assert len(warnings) == 1
        assert "No positions" in warnings[0]

    def test_out_of_range_position_warns(self):
        warnings = validate_session_positions({"VER": 0, "HAM": 25})
        assert len(warnings) == 2

    def test_valid_positions_no_warnings(self):
        positions = {f"D{i}": i for i in range(1, 21)}
        warnings = validate_session_positions(positions)
        assert warnings == []

    def test_negative_degradation_warns(self):
        data = {"McLaren": {"avg_pace": 90.0, "degradation": -0.05}}
        warnings = validate_team_pace_data(data)
        assert any("negative" in w for w in warnings)

    def test_extreme_pace_warns(self):
        data = {"McLaren": {"avg_pace": 200.0}}
        warnings = validate_team_pace_data(data)
        assert any("outside" in w for w in warnings)

    def test_none_pace_data_is_valid(self):
        warnings = validate_team_pace_data(None)
        assert warnings == []

    def test_duplicate_ranks_warns(self):
        warnings = validate_fp_team_order({"A": 1, "B": 1, "C": 3})
        assert any("Duplicate" in w for w in warnings)

    def test_valid_ranks_no_warnings(self):
        ranks = {f"Team{i}": i for i in range(1, 11)}
        warnings = validate_fp_team_order(ranks)
        assert warnings == []

    def test_non_contiguous_ranks_warns(self):
        warnings = validate_fp_team_order({"A": 1, "B": 3, "C": 5})
        assert any("Non-contiguous" in w for w in warnings)
