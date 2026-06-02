"""Tests for wet-weather skill system across qualifying and race paths."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

_DRIVER_CHARACTERISTICS_PATHS = (
    Path("data/processed/driver_characteristics/2026_driver_characteristics.json"),
    Path("data/processed/driver_characteristics.json"),
)


def _load_committed_driver_characteristics() -> dict[str, object]:
    """Load committed driver characteristics from the current data layout."""
    for path in _DRIVER_CHARACTERISTICS_PATHS:
        if path.exists():
            return json.loads(path.read_text())

    searched = ", ".join(str(path) for path in _DRIVER_CHARACTERISTICS_PATHS)
    pytest.fail(f"No committed driver characteristics file found; checked: {searched}")


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
        data = _load_committed_driver_characteristics()

        drivers = data.get("drivers", {})
        assert len(drivers) > 0
        missing = [code for code, d in drivers.items() if "wet_skill" not in d]
        assert not missing, f"Missing wet_skill: {missing}"

    def test_wet_skill_values_in_range(self):
        """All wet_skill values between 0.0 and 1.0."""
        data = _load_committed_driver_characteristics()

        for code, d in data.get("drivers", {}).items():
            ws = d.get("wet_skill")
            if ws is not None:
                assert 0.0 <= ws <= 1.0, f"{code} wet_skill={ws} out of range"

    def test_wet_skill_spread_meaningful(self):
        """Spread must be >= 0.25 to produce differentiated predictions."""
        data = _load_committed_driver_characteristics()

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
            "wet_skill_lap_weight": 0.80,
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
            assert _compute_race_wet_skill_modifier({"wet_skill": 0.70}, "rain", 0.16, 0.70) == 0.0


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


def _build_sprint_weather_session(rainfall_samples: list[bool]) -> MagicMock:
    """Create a sprint session stub with controllable rainfall samples."""
    session = MagicMock()
    session.weather_data = pd.DataFrame({"Rainfall": rainfall_samples})
    return session


def _run_sprint_wet_skill_update(
    *,
    patcher,
    tmp_path,
    rainfall_samples: list[bool],
    initial_wet_skills: dict[str, float],
    wet_skill_update_blend: float = 0.15,
) -> dict[str, dict[str, object]]:
    """Run the sprint updater with controlled weather and return saved drivers."""
    from src.models.bayesian import DriverPrior
    from src.systems import updater

    patcher.chdir(tmp_path)

    sprint_results = pd.DataFrame(
        {
            "Abbreviation": ["LEC", "HAM"],
            "Position": [1, 20],
            "Status": ["Finished", "Finished"],
            "race_name": ["China Grand Prix"] * 2,
            "year": [2026] * 2,
        }
    )
    sprint_session = _build_sprint_weather_session(rainfall_samples)

    patcher.setattr("src.utils.weekend.is_sprint_weekend", lambda year, race_name: True)
    patcher.setattr(
        updater,
        "load_competitive_session",
        lambda year, race_name, session_name, load_laps=False: (sprint_results, sprint_session),
    )
    patcher.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda config_path="data/current_lineups.json": {"Ferrari": ["LEC", "HAM"]},
    )

    priors = {
        "LEC": DriverPrior("16", "LEC", "Ferrari", "top", mu=16.5, sigma=2.0),
        "HAM": DriverPrior("44", "HAM", "Ferrari", "top", mu=16.0, sigma=2.2),
    }
    patcher.setattr("src.models.priors_factory.PriorsFactory.create_priors", lambda self: priors)

    driver_payload = {
        "version": 1,
        "drivers": {
            driver_code: {
                "racecraft": {"skill_score": 0.60},
                "pace": {"race_pace": 0.50},
                "wet_skill": wet_skill,
            }
            for driver_code, wet_skill in initial_wet_skills.items()
        },
    }

    class _Store:
        """Minimal artifact store stub for sprint wet-skill tests."""

        def __init__(self, data_root):
            self.data_root = data_root

        def load_artifact(self, artifact_type, artifact_key):
            if artifact_type == "driver_characteristics":
                return driver_payload
            return None

        def get_latest_version(self, artifact_type, artifact_key):
            return 1

        def save_artifact(self, artifact_type, artifact_key, data, version):
            """Accept persistence writes without hitting a real backend."""

    patcher.setattr(updater, "ArtifactStore", _Store)

    def _config_get(key, default=None):
        """Return stable config values for the sprint wet-skill tests."""
        overrides = {
            "bayesian.sprint_race_confidence": 0.20,
            "baseline_predictor.driver_form.race_pace_update_blend": 0.25,
            "baseline_predictor.driver_form.wet_skill_update_blend": wet_skill_update_blend,
            "baseline_predictor.driver_form.wet_skill_observation_scale": 0.40,
            "baseline_predictor.race.lap_time.wet_skill_neutral": 0.70,
            "grid.size": 22,
        }
        return overrides.get(key, default)

    patcher.setattr(updater.config_loader, "get", _config_get)

    updater.update_from_sprint_race(2026, "China Grand Prix", data_root=str(tmp_path))

    fallback_file = (
        tmp_path
        / "data"
        / "processed"
        / "driver_characteristics"
        / "2026_driver_characteristics.json"
    )
    saved_payload = json.loads(fallback_file.read_text())
    return saved_payload["drivers"]


class TestSprintWetSkillUpdate:
    """Sprint race should update wet_skill in wet and mixed conditions."""

    def test_sprint_wet_updates_wet_skill(self, patcher, tmp_path):
        """Wet sprint race should update wet_skill without moving dry state."""
        initial_wet_skills = {"LEC": 0.724, "HAM": 0.726}

        drivers = _run_sprint_wet_skill_update(
            patcher=patcher,
            tmp_path=tmp_path,
            rainfall_samples=[True, True, False, True],
            initial_wet_skills=initial_wet_skills,
        )

        assert drivers["LEC"]["wet_skill"] != initial_wet_skills["LEC"], (
            "Sprint wet update should change LEC"
        )
        assert drivers["HAM"]["wet_skill"] != initial_wet_skills["HAM"], (
            "Sprint wet update should change HAM"
        )
        assert drivers["LEC"]["wet_skill"] > initial_wet_skills["LEC"], (
            "LEC won sprint, should increase"
        )
        assert drivers["HAM"]["wet_skill"] < initial_wet_skills["HAM"], (
            "HAM lost sprint, should decrease"
        )
        assert drivers["LEC"]["pace"]["race_pace"] == 0.50
        assert drivers["HAM"]["pace"]["race_pace"] == 0.50
        assert "bayesian" not in drivers["LEC"]
        assert "bayesian" not in drivers["HAM"]

    def test_sprint_dry_leaves_wet_skill_unchanged(self, patcher, tmp_path):
        """Dry sprint race should leave wet_skill untouched."""
        initial_wet_skills = {"LEC": 0.724, "HAM": 0.726}

        drivers = _run_sprint_wet_skill_update(
            patcher=patcher,
            tmp_path=tmp_path,
            rainfall_samples=[False, False, False, False],
            initial_wet_skills=initial_wet_skills,
        )

        assert drivers["LEC"]["wet_skill"] == initial_wet_skills["LEC"]
        assert drivers["HAM"]["wet_skill"] == initial_wet_skills["HAM"]

    def test_sprint_wet_blend_is_quarter_of_main_race(self, patcher, tmp_path):
        """Sprint wet-skill blend should stay at one quarter of the main-race blend."""
        initial_wet_skills = {"LEC": 0.70, "HAM": 0.70}

        drivers = _run_sprint_wet_skill_update(
            patcher=patcher,
            tmp_path=tmp_path,
            rainfall_samples=[True, True, False, True],
            initial_wet_skills=initial_wet_skills,
        )

        observed_signal = 0.95
        main_race_blend = 0.15
        sprint_blend = main_race_blend * 0.25
        expected_sprint = (1.0 - sprint_blend) * initial_wet_skills[
            "LEC"
        ] + sprint_blend * observed_signal
        full_race_counterfactual = (1.0 - main_race_blend) * initial_wet_skills[
            "LEC"
        ] + main_race_blend * observed_signal
        assert abs(drivers["LEC"]["wet_skill"] - round(expected_sprint, 3)) < 0.002
        assert full_race_counterfactual > expected_sprint, (
            "Full race blend should move more than sprint"
        )


class TestNormalizeWeatherKey:
    """normalize_weather_key maps all variants to canonical forms."""

    def test_wet_maps_to_rain(self):
        from src.utils.validation_helpers import normalize_weather_key

        assert normalize_weather_key("wet") == "rain"

    def test_rain_stays_rain(self):
        from src.utils.validation_helpers import normalize_weather_key

        assert normalize_weather_key("rain") == "rain"

    def test_dry_stays_dry(self):
        from src.utils.validation_helpers import normalize_weather_key

        assert normalize_weather_key("dry") == "dry"

    def test_mixed_stays_mixed(self):
        from src.utils.validation_helpers import normalize_weather_key

        assert normalize_weather_key("mixed") == "mixed"

    def test_case_insensitive(self):
        from src.utils.validation_helpers import normalize_weather_key

        assert normalize_weather_key("WET") == "rain"
        assert normalize_weather_key("Rain") == "rain"
        assert normalize_weather_key(" DRY ") == "dry"

    def test_unknown_passes_through(self):
        from src.utils.validation_helpers import normalize_weather_key

        assert normalize_weather_key("sunny") == "sunny"
        assert normalize_weather_key("") == ""
