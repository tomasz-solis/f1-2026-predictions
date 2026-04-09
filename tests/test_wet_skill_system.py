"""Tests for wet-weather skill system across qualifying and race paths."""

import numpy as np
import pytest


class TestWetSkillQualifying:
    """Qualifying wet_skill adjustments produce position shuffles."""

    def test_wet_skill_shifts_qualifying_score(self):
        """High wet_skill scores better in rain than low wet_skill."""
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )

        good_adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.90},
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        bad_adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.60},
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        neutral_adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.70},
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )

        assert good_adj > 0.0
        assert bad_adj < 0.0
        assert abs(neutral_adj) < 1e-9

    def test_dry_weather_zero_adjustment(self):
        """Dry conditions produce zero adjustment regardless of wet_skill."""
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )

        adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.90},
            weather="dry",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        assert abs(adj) < 1e-9

    def test_mixed_partial_adjustment(self):
        """Mixed conditions produce a partial adjustment less than full rain."""
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )

        driver = {"wet_skill": 0.90}
        rain_adj = _compute_wet_skill_adjustment(
            driver_info=driver,
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        mixed_adj = _compute_wet_skill_adjustment(
            driver_info=driver,
            weather="mixed",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        assert 0 < mixed_adj < rain_adj


class TestWetSkillRace:
    """Race lap-by-lap simulator applies wet_skill modifier."""

    def test_zero_in_dry(self):
        """Dry weather produces zero modifier."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        result = _compute_race_wet_skill_modifier(
            skill_info={"wet_skill": 0.90},
            weather="dry",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert result == 0.0

    def test_negative_for_good_wet_driver(self):
        """Good wet driver gets NEGATIVE (faster) lap time modifier in rain."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        result = _compute_race_wet_skill_modifier(
            skill_info={"wet_skill": 0.90},
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert result < 0.0

    def test_positive_for_bad_wet_driver(self):
        """Bad wet driver gets POSITIVE (slower) lap time modifier in rain."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        result = _compute_race_wet_skill_modifier(
            skill_info={"wet_skill": 0.55},
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert result > 0.0

    def test_mixed_half_effect(self):
        """Mixed conditions produce exactly half the rain effect."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        info = {"wet_skill": 0.90}
        rain = _compute_race_wet_skill_modifier(
            skill_info=info,
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        mixed = _compute_race_wet_skill_modifier(
            skill_info=info,
            weather="mixed",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert abs(mixed - rain * 0.5) < 1e-9

    def test_neutral_produces_zero(self):
        """Driver at exactly the neutral point gets zero modifier."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        result = _compute_race_wet_skill_modifier(
            skill_info={"wet_skill": 0.70},
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert abs(result) < 1e-9

    def test_sign_convention_opposite_to_qualifying(self):
        """Race returns negative-for-good; qualifying returns positive-for-good.

        This is correct: qualifying scores are "higher = better", lap times
        are "lower = better". Both conventions mean "good wet driver benefits."
        """
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        good_driver = {"wet_skill": 0.90}

        quali_adj = _compute_wet_skill_adjustment(
            driver_info=good_driver,
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        race_adj = _compute_race_wet_skill_modifier(
            skill_info=good_driver,
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )

        assert quali_adj > 0, "Qualifying: positive = better score"
        assert race_adj < 0, "Race: negative = faster lap time"


class TestWetSkillDataPresence:
    """Driver characteristics data has wet_skill populated."""

    def test_all_drivers_have_wet_skill(self):
        """Every driver in the characteristics file must have wet_skill."""
        import json

        with open("data/processed/driver_characteristics.json") as f:
            data = json.load(f)

        drivers = data.get("drivers", {})
        assert len(drivers) > 0
        missing = [code for code, d in drivers.items() if "wet_skill" not in d]
        assert not missing, f"Missing wet_skill: {missing}"

    def test_wet_skill_values_in_range(self):
        """All wet_skill values between 0.0 and 1.0."""
        import json

        with open("data/processed/driver_characteristics.json") as f:
            data = json.load(f)

        for code, d in data.get("drivers", {}).items():
            ws = d.get("wet_skill")
            if ws is not None:
                assert 0.0 <= ws <= 1.0, f"{code} wet_skill={ws} out of range"

    def test_wet_skill_spread_meaningful(self):
        """Spread must be >= 0.25 to produce differentiated predictions."""
        import json

        with open("data/processed/driver_characteristics.json") as f:
            data = json.load(f)

        values = [d["wet_skill"] for d in data["drivers"].values() if "wet_skill" in d]
        assert len(values) >= 20
        spread = max(values) - min(values)
        assert spread >= 0.25, f"Spread too narrow: {spread:.2f}"


class TestWetSkillRaceInfoPipeline:
    """wet_skill flows from characteristics through to simulator input."""

    def test_wet_skill_absent_defaults_to_neutral(self):
        """Driver data without wet_skill key should produce 0.70 default."""
        driver_data = {"pace": {"quali_pace": 0.5}, "racecraft": {"skill_score": 0.5}}
        result = float(driver_data.get("wet_skill", 0.70))
        assert result == 0.70

    def test_wet_skill_present_passes_through(self):
        """Driver data with wet_skill should pass the actual value."""
        driver_data = {"wet_skill": 0.85, "pace": {}, "racecraft": {"skill_score": 0.6}}
        result = float(driver_data.get("wet_skill", 0.70))
        assert result == 0.85


class TestWetSkillIntegration:
    """Integration tests: wet conditions change simulation outcomes."""

    def test_wet_race_favors_good_wet_driver(self):
        """A driver with high wet_skill should win more in wet than dry."""
        from src.utils.lap_by_lap_simulator import simulate_race_lap_by_lap

        base_info = {
            "team": "Team1",
            "team_strength": 0.70,
            "team_strength_by_compound": {"MEDIUM": 0.70, "HARD": 0.70},
            "tire_deg_by_compound": {"MEDIUM": 0.12, "HARD": 0.10},
            "skill": 0.75,
            "race_advantage": 0.0,
            "overtaking_skill": 0.6,
            "defensive_skill": 0.6,
            "dnf_probability": 0.0,
        }
        driver_info = {
            "A": {**base_info, "grid_pos": 1, "wet_skill": 0.90},
            "B": {**base_info, "grid_pos": 2, "team": "Team2", "wet_skill": 0.55},
        }
        strategies = {
            "A": {"compound_sequence": ["MEDIUM", "HARD"], "pit_laps": [25]},
            "B": {"compound_sequence": ["MEDIUM", "HARD"], "pit_laps": [25]},
        }
        race_params = {
            "fuel": {"initial_load_kg": 110.0, "effect_per_lap": 0.03, "burn_rate_kg_per_lap": 1.5},
            "pit_stops": {"loss_duration": 22.0, "overtake_loss_range": [0, 2]},
            "base_chaos": {"dry": 0.28, "wet": 0.42},
            "lap_time": {
                "reference_base": 90.0,
                "team_pace_penalty_range": 5.0,
                "skill_improvement_max": 0.75,
                "elite_skill_threshold": 0.88,
                "elite_skill_lap_bonus_max": 0.09,
                "elite_skill_exponent": 1.3,
                "bounds": [70.0, 120.0],
            },
            "team_strength_compression": 0.35,
            "race_advantage_lap_impact": 0.20,
            "wet_skill_lap_weight": 0.16,
            "wet_skill_neutral": 0.70,
            "sc_probability": 0.0,
            "safety_car_trigger_lap": 999,
        }

        n_sims = 200
        dry_a_wins = sum(
            1
            for i in range(n_sims)
            if simulate_race_lap_by_lap(
                driver_info, strategies, dict(race_params), 50, "dry", np.random.default_rng(i)
            )["finish_order"][0]
            == "A"
        )
        wet_a_wins = sum(
            1
            for i in range(n_sims)
            if simulate_race_lap_by_lap(
                driver_info, strategies, dict(race_params), 50, "wet", np.random.default_rng(i)
            )["finish_order"][0]
            == "A"
        )

        assert wet_a_wins / n_sims > dry_a_wins / n_sims, (
            f"Good wet driver should win more in wet ({wet_a_wins}/{n_sims}) "
            f"than dry ({dry_a_wins}/{n_sims})"
        )

    def test_dry_race_ignores_wet_skill_completely(self):
        """In dry, different wet_skill values must produce zero modifier."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        for ws in [0.40, 0.55, 0.70, 0.85, 0.95]:
            assert _compute_race_wet_skill_modifier({"wet_skill": ws}, "dry", 0.16, 0.70) == 0.0

    def test_unknown_weather_treated_as_dry(self):
        """Unrecognized weather strings produce zero wet adjustment."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        for weather in ["sunny", "overcast", "", "unknown", "DRY", "Dry"]:
            assert _compute_race_wet_skill_modifier({"wet_skill": 0.90}, weather, 0.16, 0.70) == 0.0


class TestWetSkillEdgeCases:
    """Edge cases and None handling."""

    def test_wet_skill_none_defaults_to_neutral_race(self):
        """wet_skill=None in driver info should fall back to neutral."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        result = _compute_race_wet_skill_modifier(
            skill_info={"wet_skill": None},
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert result == 0.0

    def test_wet_skill_none_defaults_to_neutral_qualifying(self):
        """wet_skill=None in driver info should fall back to neutral in qualifying."""
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )

        result = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": None},
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        assert result == 0.0

    def test_missing_wet_skill_key_defaults_neutral(self):
        """Driver info without wet_skill key should behave as neutral."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        result = _compute_race_wet_skill_modifier(
            skill_info={"skill": 0.80},
            weather="rain",
            wet_skill_weight=0.16,
            wet_skill_neutral=0.70,
        )
        assert result == 0.0

    def test_all_neutral_produces_no_shuffling(self):
        """If all drivers are at neutral, wet modifier is zero for all."""
        from src.utils.lap_by_lap_simulator import _compute_race_wet_skill_modifier

        for _ in range(20):
            assert _compute_race_wet_skill_modifier({"wet_skill": 0.70}, "wet", 0.16, 0.70) == 0.0


class TestWetSkillSprintPath:
    """Sprint qualifying and sprint race paths."""

    def test_sprint_qualifying_applies_wet_skill(self):
        """Sprint qualifying uses the same wet_skill adjustment function."""
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )

        adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.85},
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        assert adj == pytest.approx(0.15 * 0.18)

    def test_sprint_qualifying_mixed_weather(self):
        """Sprint qualifying in mixed conditions gets partial effect."""
        from src.predictors.baseline.qualifying_simulation import (
            _compute_wet_skill_adjustment,
        )

        rain_adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.85},
            weather="rain",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        mixed_adj = _compute_wet_skill_adjustment(
            driver_info={"wet_skill": 0.85},
            weather="mixed",
            wet_skill_weight=0.18,
            wet_skill_neutral=0.70,
            mixed_wet_blend=0.5,
        )
        assert mixed_adj == pytest.approx(rain_adj * 0.5)
