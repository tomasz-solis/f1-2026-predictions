from __future__ import annotations

from pathlib import Path

import src.predictors.baseline_2026 as predictor_module
from src.predictors.baseline.components import (
    BaselineDataLoader,
    BaselineQualifyingEngine,
    BaselineRaceEngine,
    BaselineStrengthCalculator,
)
from src.predictors.baseline.data_mixin import BaselineDataMixin
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.race_mixin import BaselineRaceMixin
from src.predictors.baseline_2026 import Baseline2026Predictor


def test_data_loader_delegates_to_data_mixin(monkeypatch):
    predictor = object()
    loader = BaselineDataLoader(predictor)
    calls: dict[str, object] = {}

    def _fake_load_data(self):
        calls["self"] = self

    monkeypatch.setattr(BaselineDataMixin, "load_data", _fake_load_data)

    loader.load_data()

    assert calls["self"] is predictor


def test_strength_calculator_delegates_to_data_mixin(monkeypatch):
    predictor = object()
    calculator = BaselineStrengthCalculator(predictor)

    monkeypatch.setattr(
        BaselineDataMixin,
        "calculate_track_suitability",
        lambda self, team, race_name: 0.12,
    )
    monkeypatch.setattr(
        BaselineDataMixin,
        "get_blended_team_strength",
        lambda self, team, race_name: 0.77,
    )
    monkeypatch.setattr(
        BaselineDataMixin,
        "_select_race_compound",
        lambda self, race_name: "SOFT",
    )
    monkeypatch.setattr(
        BaselineDataMixin,
        "get_compound_adjusted_team_strength",
        lambda self, team, race_name, compound: 0.81,
    )

    assert calculator.calculate_track_suitability("McLaren", "Bahrain Grand Prix") == 0.12
    assert calculator.get_blended_team_strength("McLaren", "Bahrain Grand Prix") == 0.77
    assert calculator.select_race_compound("Bahrain Grand Prix") == "SOFT"
    assert (
        calculator.get_compound_adjusted_team_strength(
            "McLaren",
            "Bahrain Grand Prix",
            compound="MEDIUM",
        )
        == 0.81
    )


def test_engines_delegate_to_mixins(monkeypatch):
    predictor = object()
    qualifying_engine = BaselineQualifyingEngine(predictor)
    race_engine = BaselineRaceEngine(predictor)
    calls: dict[str, object] = {}

    def _fake_predict_qualifying(self, **kwargs):
        calls["qualifying_self"] = self
        calls["qualifying_kwargs"] = kwargs
        return {"grid": []}

    def _fake_predict_sprint_race(self, **kwargs):
        calls["sprint_self"] = self
        calls["sprint_kwargs"] = kwargs
        return {"finish_order": []}

    def _fake_predict_race(self, **kwargs):
        calls["race_self"] = self
        calls["race_kwargs"] = kwargs
        return {"finish_order": []}

    monkeypatch.setattr(BaselineQualifyingMixin, "predict_qualifying", _fake_predict_qualifying)
    monkeypatch.setattr(BaselineQualifyingMixin, "predict_sprint_race", _fake_predict_sprint_race)
    monkeypatch.setattr(BaselineRaceMixin, "predict_race", _fake_predict_race)

    qualifying_result = qualifying_engine.predict(
        year=2026,
        race_name="Bahrain Grand Prix",
        n_simulations=5,
        qualifying_stage="main",
    )
    sprint_result = qualifying_engine.predict_sprint_race(
        sprint_quali_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        weather="dry",
        race_name="Chinese Grand Prix",
        n_simulations=5,
    )
    race_result = race_engine.predict(
        qualifying_grid=[{"driver": "NOR", "team": "McLaren", "position": 1}],
        weather="dry",
        race_name="Bahrain Grand Prix",
        n_simulations=5,
        is_sprint=False,
        race_compound="MEDIUM",
    )

    assert calls["qualifying_self"] is predictor
    assert calls["qualifying_kwargs"] == {
        "year": 2026,
        "race_name": "Bahrain Grand Prix",
        "n_simulations": 5,
        "qualifying_stage": "main",
    }
    assert calls["sprint_self"] is predictor
    assert calls["sprint_kwargs"]["race_name"] == "Chinese Grand Prix"
    assert calls["race_self"] is predictor
    assert calls["race_kwargs"]["race_name"] == "Bahrain Grand Prix"
    assert qualifying_result == {"grid": []}
    assert sprint_result == {"finish_order": []}
    assert race_result == {"finish_order": []}


def test_baseline_predictor_facade_methods_delegate_to_components():
    predictor = Baseline2026Predictor.__new__(Baseline2026Predictor)

    class _Loader:
        def __init__(self):
            self.called = False

        def load_data(self):
            self.called = True

    class _Strength:
        def __init__(self):
            self.calls = []

        def calculate_track_suitability(self, team, race_name):
            self.calls.append(("suitability", team, race_name))
            return 0.05

        def get_blended_team_strength(self, team, race_name):
            self.calls.append(("blended", team, race_name))
            return 0.72

        def select_race_compound(self, race_name):
            self.calls.append(("compound", race_name))
            return "HARD"

        def get_compound_adjusted_team_strength(self, team, race_name, compound="MEDIUM"):
            self.calls.append(("compound_adjusted", team, race_name, compound))
            return 0.75

    class _Qualifying:
        def predict(self, **kwargs):
            return {"kind": "qualifying", "kwargs": kwargs}

        def predict_sprint_race(self, **kwargs):
            return {"kind": "sprint", "kwargs": kwargs}

    class _Race:
        def predict(self, **kwargs):
            return {"kind": "race", "kwargs": kwargs}

    predictor.data_loader = _Loader()
    predictor.strength_calculator = _Strength()
    predictor.qualifying_engine = _Qualifying()
    predictor.race_engine = _Race()

    predictor.load_data()
    assert predictor.data_loader.called is True
    assert predictor.calculate_track_suitability("Ferrari", "Monza Grand Prix") == 0.05
    assert predictor.get_blended_team_strength("Ferrari", "Monza Grand Prix") == 0.72
    assert predictor._select_race_compound("Monza Grand Prix") == "HARD"
    assert (
        predictor.get_compound_adjusted_team_strength(
            "Ferrari",
            "Monza Grand Prix",
            compound="SOFT",
        )
        == 0.75
    )
    assert predictor.predict_qualifying(2026, "Monza Grand Prix", n_simulations=5)["kind"] == (
        "qualifying"
    )
    assert (
        predictor.predict_sprint_race([], weather="dry", race_name="Miami Grand Prix")["kind"]
        == "sprint"
    )
    assert predictor.predict_race([], weather="dry", race_name="Monza Grand Prix")["kind"] == (
        "race"
    )


def test_predictor_initialization_uses_components(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class _StubDataLoader:
        def __init__(self, predictor):
            self._predictor = predictor
            calls["loader_predictor"] = predictor

        def load_data(self):
            calls["load_data_called"] = True
            self._predictor.teams = {}
            self._predictor.drivers = {}
            self._predictor.tracks = {}

    class _StubStrengthCalculator:
        def __init__(self, predictor):
            calls["strength_predictor"] = predictor

    class _StubQualifyingEngine:
        def __init__(self, predictor):
            calls["qualifying_predictor"] = predictor

    class _StubRaceEngine:
        def __init__(self, predictor):
            calls["race_predictor"] = predictor

    class _StubArtifactStore:
        def __init__(self, data_root):
            self.data_root = data_root

    monkeypatch.setattr(predictor_module, "BaselineDataLoader", _StubDataLoader)
    monkeypatch.setattr(predictor_module, "BaselineStrengthCalculator", _StubStrengthCalculator)
    monkeypatch.setattr(predictor_module, "BaselineQualifyingEngine", _StubQualifyingEngine)
    monkeypatch.setattr(predictor_module, "BaselineRaceEngine", _StubRaceEngine)
    monkeypatch.setattr(
        predictor_module,
        "create_baseline_if_missing",
        lambda data_dir: calls.setdefault("data_dir", data_dir),
    )
    monkeypatch.setattr(predictor_module, "ArtifactStore", _StubArtifactStore)
    monkeypatch.setattr(predictor_module, "Config", lambda: object())

    predictor = Baseline2026Predictor(data_dir=str(tmp_path / "processed"), seed=99)

    assert calls["data_dir"] == Path(tmp_path / "processed")
    assert calls["load_data_called"] is True
    assert calls["loader_predictor"] is predictor
    assert calls["strength_predictor"] is predictor
    assert calls["qualifying_predictor"] is predictor
    assert calls["race_predictor"] is predictor
