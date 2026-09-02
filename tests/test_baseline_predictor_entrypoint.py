"""Tests for the public Baseline2026Predictor entry point."""

from __future__ import annotations

from pathlib import Path

import src.predictors.baseline_2026 as predictor_module
from src.predictors.baseline.data_mixin import BaselineDataMixin
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.race.prediction_mixin import BaselineRacePredictionMixin
from src.predictors.baseline.race.preparation_mixin import BaselineRacePreparationMixin
from src.predictors.baseline_2026 import Baseline2026Predictor


def test_predictor_public_methods_are_inherited_from_mixins():
    """The entry-point class should expose the mixin API without pass-through wrappers."""
    assert Baseline2026Predictor.load_data is BaselineDataMixin.load_data
    assert (
        Baseline2026Predictor.calculate_track_suitability
        is BaselineDataMixin.calculate_track_suitability
    )
    assert (
        Baseline2026Predictor.get_blended_team_strength
        is BaselineDataMixin.get_blended_team_strength
    )
    assert Baseline2026Predictor._select_race_compound is BaselineDataMixin._select_race_compound
    assert (
        Baseline2026Predictor.get_compound_adjusted_team_strength
        is BaselineDataMixin.get_compound_adjusted_team_strength
    )
    assert Baseline2026Predictor.predict_qualifying is BaselineQualifyingMixin.predict_qualifying
    assert Baseline2026Predictor.predict_sprint_race is BaselineQualifyingMixin.predict_sprint_race
    assert Baseline2026Predictor.predict_race is BaselineRacePredictionMixin.predict_race


def test_predictor_helpers_are_inherited_without_magic_getattr():
    """Helper methods should still come from the mixins through normal inheritance."""
    assert "__getattr__" not in Baseline2026Predictor.__dict__
    assert (
        Baseline2026Predictor._resolve_predictions_data_root
        is BaselineDataMixin._resolve_predictions_data_root
    )
    assert (
        Baseline2026Predictor._get_testing_profile_weights
        is BaselineQualifyingMixin._get_testing_profile_weights
    )
    assert (
        Baseline2026Predictor._prepare_driver_info_with_compounds
        is BaselineRacePreparationMixin._prepare_driver_info_with_compounds
    )


def test_predictor_inherited_learned_adjustment_supports_race_session():
    """The shared learned-adjustment helper should still accept explicit race sessions."""
    predictor = Baseline2026Predictor.__new__(Baseline2026Predictor)
    calls: dict[str, object] = {}

    class _Calibration:
        def get_combined_position_adjustment(self, **kwargs):
            calls.update(kwargs)
            return 0.37

    class _Config:
        def get(self, key, default=None):
            return default

    predictor.calibration_system = _Calibration()
    predictor.config = _Config()

    adjustment = predictor._get_learned_position_adjustment(
        team="McLaren",
        driver="NOR",
        teammates=["NOR", "PIA"],
        session="race",
    )

    assert adjustment == 0.37
    assert calls["session"] == "race"


def test_predictor_initialization_loads_data(patcher, tmp_path):
    """The predictor constructor should prepare storage, learning state, and baseline data."""
    calls: dict[str, object] = {}

    class _StubArtifactStore:
        def __init__(self, data_root):
            self.data_root = data_root

    class _StubLearningSystem:
        def __init__(self, state_file):
            calls["state_file"] = state_file

    def _fake_load_data(self):
        calls["load_data_called"] = True
        self.teams = {}
        self.drivers = {}
        self.tracks = {}

    patcher.setattr(
        predictor_module,
        "create_baseline_if_missing",
        lambda data_dir: calls.setdefault("data_dir", data_dir),
    )
    patcher.setattr(predictor_module, "ArtifactStore", _StubArtifactStore)
    patcher.setattr(predictor_module, "SystematicLearningSystem", _StubLearningSystem)
    patcher.setattr(predictor_module, "Config", lambda: object())
    patcher.setattr(BaselineDataMixin, "load_data", _fake_load_data)

    predictor = Baseline2026Predictor(data_dir=str(tmp_path / "processed"), seed=99)

    assert calls["data_dir"] == Path(tmp_path / "processed")
    assert calls["load_data_called"] is True
    assert calls["state_file"] == Path(tmp_path) / "learning_state.json"
    assert predictor.seed == 99
