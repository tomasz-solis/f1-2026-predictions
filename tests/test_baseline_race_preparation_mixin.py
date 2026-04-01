from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

import src.predictors.baseline.race.preparation_mixin as prep_module
from src.predictors.baseline.race.preparation_flow import (
    _blend_race_skill_with_bayesian_form,
)
from src.predictors.baseline.race.preparation_mixin import BaselineRacePreparationMixin


class DummyPreparation(BaselineRacePreparationMixin):
    def __init__(self):
        self.teams = {}
        self.drivers = {}
        self.compound_strength = 0.9
        self.blended_strength = 0.7
        self.profile_modifier = (0.0, False)

    def get_compound_adjusted_team_strength(
        self, team: str, race_name: str, race_compound: str
    ) -> float:
        return self.compound_strength

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        return self.blended_strength

    def _compute_testing_profile_modifier(
        self, team: str, profile: str, metric_weights: dict, scale: float
    ):
        return self.profile_modifier


class DummyConfig:
    def __init__(self, overrides: dict[str, object] | None = None):
        self._overrides = overrides or {}

    def get(self, key: str, default=None):
        return self._overrides.get(key, default)


def _portable_skill_config() -> DummyConfig:
    """Return the shared blend config used by portable-skill tests."""
    return DummyConfig(
        {
            "grid.size": 22,
            "baseline_predictor.driver_form.bayesian_pace_blend_per_race": 0.20,
            "baseline_predictor.driver_form.bayesian_pace_blend_cap": 0.60,
        }
    )


def _established_driver_with_rating(rating_mu: float) -> dict[str, object]:
    """Build a minimal established-driver payload with Bayesian form."""
    return {
        "experience": {"tier": "established"},
        "bayesian": {
            "rating_mu": rating_mu,
            "normalized_skill_score": 0.99,
        },
    }


@contextmanager
def _working_directory(path: Path):
    current = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


def test_load_track_overtaking_difficulty_from_file_and_fallbacks(tmp_path):
    track_dir = tmp_path / "data" / "processed" / "track_characteristics"
    track_dir.mkdir(parents=True)
    track_file = track_dir / "2026_track_characteristics.json"
    track_file.write_text(
        json.dumps({"tracks": {"Bahrain Grand Prix": {"overtaking_difficulty": 0.82}}})
    )

    prep = DummyPreparation()
    with _working_directory(tmp_path):
        with patch.object(
            prep_module,
            "validate_track_characteristics",
            lambda payload, **kwargs: None,
        ):
            assert prep._load_track_overtaking_difficulty(None) == 0.5
            assert prep._load_track_overtaking_difficulty("Bahrain Grand Prix") == 0.82
            assert prep._load_track_overtaking_difficulty("Unknown Race") == 0.5

            track_file.write_text("{bad json")
            assert prep._load_track_overtaking_difficulty("Bahrain Grand Prix") == 0.5


def test_load_track_overtaking_difficulty_handles_schema_validation_error(tmp_path):
    track_dir = tmp_path / "data" / "processed" / "track_characteristics"
    track_dir.mkdir(parents=True)
    (track_dir / "2026_track_characteristics.json").write_text(json.dumps({"tracks": {}}))

    prep = DummyPreparation()
    with _working_directory(tmp_path):
        with patch.object(
            prep_module,
            "validate_track_characteristics",
            lambda payload, **kwargs: (_ for _ in ()).throw(ValueError("schema error")),
        ):
            assert prep._load_track_overtaking_difficulty("Bahrain Grand Prix") == 0.5


def test_prepare_driver_info_applies_caps_and_profile_modifiers():
    prep = DummyPreparation()
    prep.compound_strength = 0.9
    prep.profile_modifier = (0.25, True)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.33,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
        }
    )
    prep.teams = {"McLaren": {"overall_performance": 0.55, "uncertainty": 0.50}}
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.6, "race_pace": 0.7},
            "racecraft": {"skill_score": 0.8, "overtaking_skill": 0.7},
            "dnf_risk": {"dnf_rate": 0.6},
            "experience": {"tier": "rookie"},
        }
    }

    info_map, long_profile_count = prep._prepare_driver_info(
        qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        race_name="Bahrain Grand Prix",
        race_compound="SOFT",
    )

    info = info_map["NOR"]
    assert long_profile_count == 1
    assert info["team_strength"] == 1.0
    assert info["race_advantage"] == pytest.approx(0.1)
    assert info["defensive_skill"] == pytest.approx(0.735)
    assert info["dnf_probability"] == pytest.approx(0.33)


def test_prepare_driver_info_with_compounds_builds_per_compound_strengths():
    prep = DummyPreparation()
    prep.blended_strength = 0.7
    prep.profile_modifier = (0.1, True)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.tire_physics.default_deg_slope": 0.12,
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
        }
    )
    prep.teams = {
        "McLaren": {
            "uncertainty": 0.2,
            "compound_characteristics": {
                "SOFT": {
                    "tire_deg_slope": 0.20,
                }
            },
        }
    }
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.6, "race_pace": 0.65},
            "racecraft": {"skill_score": 0.7, "overtaking_skill": 0.6},
            "dnf_risk": {"dnf_rate": 0.12},
            "experience": {"tier": "established"},
        }
    }
    with patch(
        "src.data.compound_performance.get_compound_performance_modifier",
        lambda team_compound_chars, compound: 0.05 if compound == "SOFT" else 0.0,
    ):
        info_map, long_profile_count = prep._prepare_driver_info_with_compounds(
            qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 2}],
            race_name="Bahrain Grand Prix",
        )

    info = info_map["NOR"]
    assert long_profile_count == 1
    assert info["team_strength"] == pytest.approx(0.8)
    assert info["team_strength_by_compound"]["SOFT"] == pytest.approx(0.85)
    assert info["team_strength_by_compound"]["MEDIUM"] == pytest.approx(0.8)
    assert info["team_strength_by_compound"]["HARD"] == pytest.approx(0.8)
    assert info["tire_deg_by_compound"]["SOFT"] == pytest.approx(0.2)
    assert info["tire_deg_by_compound"]["MEDIUM"] == pytest.approx(0.12)


def test_prepare_driver_info_with_compounds_defaults_missing_tire_deg_slope():
    prep = DummyPreparation()
    prep.blended_strength = 0.7
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.tire_physics.default_deg_slope": 0.12,
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
        }
    )
    prep.teams = {
        "McLaren": {
            "uncertainty": 0.2,
            "compound_characteristics": {"SOFT": {"tire_deg_slope": None}},
        }
    }
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.6, "race_pace": 0.65},
            "racecraft": {"skill_score": 0.7, "overtaking_skill": 0.6},
            "dnf_risk": {"dnf_rate": 0.12},
            "experience": {"tier": "established"},
        }
    }

    with patch(
        "src.data.compound_performance.get_compound_performance_modifier",
        lambda team_compound_chars, compound: 0.0,
    ):
        info_map, _ = prep._prepare_driver_info_with_compounds(
            qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 2}],
            race_name="Bahrain Grand Prix",
        )

    info = info_map["NOR"]
    assert info["tire_deg_by_compound"]["SOFT"] == pytest.approx(0.12)


def test_prepare_driver_info_with_compounds_blends_race_skill_with_bayesian_form():
    prep = DummyPreparation()
    prep.races_completed = 2
    prep.blended_strength = 0.7
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "grid.size": 22,
            "baseline_predictor.race.tire_physics.default_deg_slope": 0.12,
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.driver_form.bayesian_pace_blend_per_race": 0.20,
            "baseline_predictor.driver_form.bayesian_pace_blend_cap": 0.60,
        }
    )
    prep.teams = {"Mercedes": {"uncertainty": 0.2}}
    prep.drivers = {
        "ANT": {
            "pace": {"quali_pace": 0.45, "race_pace": 0.48},
            "racecraft": {"skill_score": 0.30, "overtaking_skill": 0.55},
            "dnf_risk": {"dnf_rate": 0.12},
            "experience": {"tier": "rookie"},
            "bayesian": {
                "rating_mu": 20.0,
                "normalized_skill_score": 0.99,
            },
        }
    }

    with patch(
        "src.data.compound_performance.get_compound_performance_modifier",
        lambda team_compound_chars, compound: 0.0,
    ):
        info_map, _ = prep._prepare_driver_info_with_compounds(
            qualifying_grid=[{"driver": "ANT", "team": "Mercedes", "position": 2}],
            race_name="Bahrain Grand Prix",
        )

    expected_skill = (0.60 * 0.30) + (0.40 * (19.0 / 21.0))
    assert info_map["ANT"]["skill"] == pytest.approx(expected_skill)


def test_prepare_driver_info_resolves_team_alias_for_uncertainty():
    prep = DummyPreparation()
    prep.compound_strength = 0.6
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.race.team_uncertainty_dnf_multiplier": 0.20,
        }
    )
    prep.teams = {"Sauber": {"overall_performance": 0.38, "uncertainty": 0.50}}
    prep.drivers = {
        "HUL": {
            "pace": {"quali_pace": 0.58, "race_pace": 0.60},
            "racecraft": {"skill_score": 0.66, "overtaking_skill": 0.62},
            "dnf_risk": {"dnf_rate": 0.12},
            "experience": {"tier": "established", "years_of_experience": 5},
        }
    }

    info_map, _ = prep._prepare_driver_info(
        qualifying_grid=[{"driver": "HUL", "team": "Audi", "position": 10}],
        race_name="Bahrain Grand Prix",
        race_compound="SOFT",
    )

    assert info_map["HUL"]["team_strength"] == pytest.approx(0.6)
    assert info_map["HUL"]["dnf_probability"] == pytest.approx(0.22)


def test_build_portable_skill_signal_recomputes_bayesian_normalization():
    prep = DummyPreparation()
    prep.races_completed = 3
    prep.config = _portable_skill_config()
    prep.drivers = {
        "HAM": _established_driver_with_rating(11.0),
    }

    portable_skill = prep._build_portable_skill_signal("HAM", base_skill=0.40)

    expected = _blend_race_skill_with_bayesian_form(
        driver_data=prep.drivers["HAM"],
        base_skill=0.40,
        races_completed=3,
        grid_size=22,
        config=prep.config,
    )

    stale_cached_blend = (0.40 * 0.40) + (0.60 * 0.95)

    assert portable_skill == pytest.approx(expected)
    assert portable_skill != pytest.approx(stale_cached_blend)


def test_portable_skill_signal_zero_races_returns_base_skill():
    """Portable skill should stay at the prior until there is race evidence to blend."""
    prep = DummyPreparation()
    prep.races_completed = 0
    prep.config = _portable_skill_config()
    prep.drivers = {"HAM": _established_driver_with_rating(18.0)}

    portable_skill = prep._build_portable_skill_signal("HAM", base_skill=0.50)

    assert portable_skill == pytest.approx(0.50)


def test_portable_skill_signal_moves_toward_positive_bayesian_form():
    """Portable skill should move up when established-driver form is better than the prior."""
    prep = DummyPreparation()
    prep.races_completed = 5
    prep.config = _portable_skill_config()
    prep.drivers = {"HAM": _established_driver_with_rating(18.0)}

    portable_skill = prep._build_portable_skill_signal("HAM", base_skill=0.30)

    assert portable_skill > 0.30
    assert portable_skill == pytest.approx(
        _blend_race_skill_with_bayesian_form(
            driver_data=prep.drivers["HAM"],
            base_skill=0.30,
            races_completed=5,
            grid_size=22,
            config=prep.config,
        )
    )


def test_portable_skill_signal_moves_toward_negative_bayesian_form():
    """Portable skill should move down when established-driver form is below the prior."""
    prep = DummyPreparation()
    prep.races_completed = 5
    prep.config = _portable_skill_config()
    prep.drivers = {"HAM": _established_driver_with_rating(3.0)}

    portable_skill = prep._build_portable_skill_signal("HAM", base_skill=0.80)

    assert portable_skill < 0.80


def test_portable_skill_and_blend_race_skill_agree():
    """Portable-skill and race-skill blending should share one live formula."""
    prep = DummyPreparation()
    prep.races_completed = 4
    prep.config = _portable_skill_config()
    prep.drivers = {"HAM": _established_driver_with_rating(14.0)}

    portable_skill = prep._build_portable_skill_signal("HAM", base_skill=0.55)
    blended_skill = _blend_race_skill_with_bayesian_form(
        driver_data=prep.drivers["HAM"],
        base_skill=0.55,
        races_completed=4,
        grid_size=22,
        config=prep.config,
    )

    assert portable_skill == pytest.approx(blended_skill)


def test_prepare_driver_info_with_compounds_resolves_team_alias_for_compound_data():
    prep = DummyPreparation()
    prep.blended_strength = 0.7
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.tire_physics.default_deg_slope": 0.12,
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
        }
    )
    prep.teams = {
        "Sauber": {
            "uncertainty": 0.2,
            "compound_characteristics": {"SOFT": {"tire_deg_slope": 0.21}},
        }
    }
    prep.drivers = {
        "HUL": {
            "pace": {"quali_pace": 0.58, "race_pace": 0.60},
            "racecraft": {"skill_score": 0.66, "overtaking_skill": 0.62},
            "dnf_risk": {"dnf_rate": 0.12},
            "experience": {"tier": "established", "years_of_experience": 5},
        }
    }

    with patch(
        "src.data.compound_performance.get_compound_performance_modifier",
        lambda team_compound_chars, compound: 0.05 if compound == "SOFT" else 0.0,
    ):
        info_map, _ = prep._prepare_driver_info_with_compounds(
            qualifying_grid=[{"driver": "HUL", "team": "Audi", "position": 10}],
            race_name="Bahrain Grand Prix",
        )

    info = info_map["HUL"]
    assert info["team_strength_by_compound"]["SOFT"] == pytest.approx(0.75)
    assert info["tire_deg_by_compound"]["SOFT"] == pytest.approx(0.21)


def test_get_driver_data_or_fallback_uses_teammate_profile_for_missing_lineup_driver():
    prep = DummyPreparation()
    prep.config = DummyConfig(
        {
            "baseline_predictor.qualifying.default_skill": 0.5,
            "baseline_predictor.race.missing_driver_teammate_weight": 0.80,
            "baseline_predictor.race.missing_driver_default_dnf_rate": 0.10,
            "baseline_predictor.race.missing_driver_rookie_dnf_penalty": 0.02,
            "baseline_predictor.race.missing_driver_rookie_quali_penalty": 0.08,
            "baseline_predictor.race.missing_driver_rookie_race_penalty": 0.07,
            "baseline_predictor.race.missing_driver_rookie_skill_penalty": 0.08,
            "baseline_predictor.race.missing_driver_rookie_overtaking_penalty": 0.06,
        }
    )
    prep.drivers = {
        "LAW": {
            "pace": {"quali_pace": 0.70, "race_pace": 0.65},
            "racecraft": {"skill_score": 0.62, "overtaking_skill": 0.60},
            "dnf_risk": {"dnf_rate": 0.03},
            "experience": {"tier": "developing"},
        }
    }
    with patch("src.utils.lineups.load_current_lineups", return_value={"RB": ["LAW", "LIN"]}):
        fallback = prep._get_driver_data_or_fallback("LIN", "RB")

    assert fallback["pace"]["quali_pace"] == pytest.approx(0.58)
    assert fallback["pace"]["race_pace"] == pytest.approx(0.55)
    assert fallback["racecraft"]["skill_score"] == pytest.approx(0.516)
    assert fallback["racecraft"]["overtaking_skill"] == pytest.approx(0.52)
    assert fallback["dnf_risk"]["dnf_rate"] == pytest.approx(0.05)
    assert fallback["experience"]["tier"] == "rookie"
    assert prep.drivers["LIN"] == fallback


def test_missing_driver_fallback_applies_reduced_second_year_penalties():
    prep = DummyPreparation()
    prep.config = DummyConfig(
        {
            "baseline_predictor.qualifying.default_skill": 0.5,
            "baseline_predictor.race.missing_driver_teammate_weight": 0.80,
            "baseline_predictor.race.missing_driver_default_dnf_rate": 0.10,
            "baseline_predictor.race.missing_driver_rookie_dnf_penalty": 0.02,
            "baseline_predictor.race.missing_driver_rookie_quali_penalty": 0.08,
            "baseline_predictor.race.missing_driver_rookie_race_penalty": 0.07,
            "baseline_predictor.race.missing_driver_rookie_skill_penalty": 0.08,
            "baseline_predictor.race.missing_driver_rookie_overtaking_penalty": 0.06,
            "baseline_predictor.race.missing_driver_second_year_penalty_scale": 0.50,
        }
    )
    prep.drivers = {
        "LAW": {
            "pace": {"quali_pace": 0.70, "race_pace": 0.65},
            "racecraft": {"skill_score": 0.62, "overtaking_skill": 0.60},
            "dnf_risk": {"dnf_rate": 0.03},
            "experience": {"tier": "developing"},
        }
    }
    with patch("src.utils.lineups.load_current_lineups", return_value={"RB": ["LAW", "LIN"]}):
        with patch.object(
            prep, "_infer_missing_driver_experience_tier", return_value="second_year"
        ):
            fallback = prep._get_driver_data_or_fallback("LIN", "RB")

    assert fallback["pace"]["quali_pace"] == pytest.approx(0.62)
    assert fallback["pace"]["race_pace"] == pytest.approx(0.585)
    assert fallback["racecraft"]["skill_score"] == pytest.approx(0.556)
    assert fallback["racecraft"]["overtaking_skill"] == pytest.approx(0.55)
    assert fallback["dnf_risk"]["dnf_rate"] == pytest.approx(0.04)
    assert fallback["experience"]["tier"] == "second_year"


def test_infer_missing_driver_experience_tier_uses_debuts_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "driver_debuts.csv").write_text(
        "Driver,First F1 season,Notes\n"
        "Lewis Hamilton,2007,McLaren debut\n"
        "Arvid Lindblad,2026,Racing Bulls debut\n"
    )

    prep = DummyPreparation()
    prep.year = 2026

    with _working_directory(tmp_path):
        assert prep._infer_missing_driver_experience_tier("HAM") == "sunset"
        assert prep._infer_missing_driver_experience_tier("LIN") == "rookie"
        assert prep._infer_missing_driver_experience_tier("ZZZ") == "rookie"


def test_infer_missing_driver_experience_tier_prefers_artifact_store():
    class StubStore:
        def load_artifact(self, artifact_type: str, artifact_key: str):
            assert artifact_type == "driver_debuts"
            assert artifact_key == "driver_debuts"
            return {"driver_debuts": {"HAM": 2007, "LIN": 2026}}

    prep = DummyPreparation()
    prep.year = 2026
    prep.artifact_store = StubStore()

    assert prep._infer_missing_driver_experience_tier("HAM") == "sunset"
    assert prep._infer_missing_driver_experience_tier("LIN") == "rookie"
    assert prep._infer_missing_driver_experience_tier("ZZZ") == "rookie"


def test_resolve_effective_experience_tier_for_race_upgrades_second_year_driver():
    prep = DummyPreparation()
    prep.year = 2026

    driver_data = {
        "experience": {
            "tier": "rookie",
            "years_of_experience": 0,
            "debut_year": 2025,
        }
    }

    assert prep._resolve_effective_experience_tier_for_race(driver_data) == "second_year"


def test_prepare_driver_info_uses_effective_experience_tier_for_dnf_modifier():
    prep = DummyPreparation()
    prep.year = 2026
    prep.compound_strength = 0.75
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
            "baseline_predictor.race.team_uncertainty_dnf_multiplier": 0.20,
        }
    )
    prep.teams = {"Mercedes": {"overall_performance": 0.75, "uncertainty": 0.20}}
    prep.drivers = {
        "ANT": {
            "pace": {"quali_pace": 0.50, "race_pace": 0.50},
            "racecraft": {"skill_score": 0.50, "overtaking_skill": 0.50},
            "dnf_risk": {"dnf_rate": 0.10},
            "experience": {
                "tier": "rookie",
                "years_of_experience": 0,
                "debut_year": 2025,
            },
        }
    }

    info_map, _ = prep._prepare_driver_info(
        qualifying_grid=[{"driver": "ANT", "team": "Mercedes", "position": 10}],
        race_name="Bahrain Grand Prix",
        race_compound="MEDIUM",
    )

    # Second-year driver should receive "second_year" (+0.03) modifier, not rookie (+0.05).
    assert info_map["ANT"]["dnf_probability"] == pytest.approx(0.13)


def test_prepare_driver_info_applies_configured_dnf_floor():
    prep = DummyPreparation()
    prep.year = 2026
    prep.compound_strength = 0.70
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.race.dnf_rate_floor": 0.02,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
            "baseline_predictor.race.team_uncertainty_dnf_multiplier": 0.20,
        }
    )
    prep.teams = {"McLaren": {"overall_performance": 0.70, "uncertainty": 0.05}}
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.75, "race_pace": 0.78},
            "racecraft": {"skill_score": 0.78, "overtaking_skill": 0.76},
            "dnf_risk": {"dnf_rate": 0.0},
            "experience": {"tier": "established", "years_of_experience": 6},
        }
    }

    info_map, _ = prep._prepare_driver_info(
        qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        race_name="Australian Grand Prix",
        race_compound="SOFT",
    )

    assert info_map["NOR"]["dnf_probability"] == pytest.approx(0.02)


def test_prepare_driver_info_coerces_invalid_dnf_rate_values():
    prep = DummyPreparation()
    prep.year = 2026
    prep.compound_strength = 0.70
    prep.profile_modifier = (0.0, False)
    prep.config = DummyConfig(
        {
            "baseline_predictor.race.dnf_rate_historical_cap": 0.20,
            "baseline_predictor.race.dnf_rate_final_cap": 0.35,
            "baseline_predictor.race.dnf_rate_floor": 0.02,
            "baseline_predictor.race.testing_long_run_modifier_scale": 0.05,
            "baseline_predictor.race.team_uncertainty_dnf_multiplier": 0.20,
        }
    )
    prep.teams = {"McLaren": {"overall_performance": 0.70, "uncertainty": 0.05}}
    prep.drivers = {
        "NOR": {
            "pace": {"quali_pace": 0.75, "race_pace": 0.78},
            "racecraft": {"skill_score": 0.78, "overtaking_skill": 0.76},
            "dnf_risk": {"dnf_rate": None},
            "experience": {"tier": "established", "years_of_experience": 6},
        }
    }

    info_map, _ = prep._prepare_driver_info(
        qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        race_name="Australian Grand Prix",
        race_compound="SOFT",
    )

    # Invalid values should fallback to the neutral baseline (0.10) and remain bounded.
    assert info_map["NOR"]["dnf_probability"] == pytest.approx(0.10)
