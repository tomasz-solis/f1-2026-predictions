"""Import checks for executable helper scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module(module_name: str, relative_path: str):
    """Load a repo script as a module so tests can inspect its imports."""
    script_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_predict_weekend_run_uses_shared_predictor(monkeypatch):
    """The weekend helper should build predictions through the shared predictor entry point."""
    module = _load_module("predict_weekend_script", "predict_weekend.py")

    class FakeLearner:
        def get_optimal_blend_weight(self, default=0.7):
            return default

    class FakePredictor:
        instances: list[FakePredictor] = []

        def __init__(self, season_year=2026):
            self.season_year = season_year
            self.qualifying_calls: list[dict] = []
            self.race_calls: list[dict] = []
            FakePredictor.instances.append(self)

        def predict_qualifying(self, **kwargs):
            self.qualifying_calls.append(kwargs)
            return {
                "grid": [
                    {
                        "position": 1,
                        "driver": "Lando Norris",
                        "team": "McLaren",
                        "confidence": 62.0,
                    }
                ]
            }

        def predict_race(self, **kwargs):
            self.race_calls.append(kwargs)
            return {
                "finish_order": [
                    {
                        "position": 1,
                        "driver": "Lando Norris",
                        "team": "McLaren",
                        "confidence": 64.0,
                        "podium_probability": 0.71,
                    }
                ]
            }

    monkeypatch.setattr(module, "LearningSystem", FakeLearner)
    monkeypatch.setattr(module, "Baseline2026Predictor", FakePredictor)
    monkeypatch.setattr(module, "auto_catchup_history", lambda year, learner: None)
    monkeypatch.setattr(module, "get_weekend_type", lambda year, race_name: "conventional")
    monkeypatch.setattr(
        module,
        "get_available_data",
        lambda year, race_name, weekend_type: {
            "fp1": None,
            "fp2": None,
            "fp3": None,
            "quali": None,
            "sprint_quali": None,
        },
    )
    monkeypatch.setattr(module, "_print_table", lambda df, columns: None)

    module.run_weekend_predictions(2026, "Australian Grand Prix")

    predictor = FakePredictor.instances[0]
    assert predictor.season_year == 2026
    assert predictor.qualifying_calls == [{"year": 2026, "race_name": "Australian Grand Prix"}]
    assert predictor.race_calls == [
        {
            "qualifying_grid": [
                {
                    "position": 1,
                    "driver": "Lando Norris",
                    "team": "McLaren",
                    "confidence": 62.0,
                }
            ],
            "weather": "dry",
            "race_name": "Australian Grand Prix",
            "year": 2026,
        }
    ]


def test_simulator_run_loop_uses_shared_predictor(monkeypatch):
    """The season simulator should call the shared predictor entry point."""
    module = _load_module("season_simulator_script", "scripts/simulator.py")

    class FakeFactory:
        def create_priors(self):
            return {"baseline": 1.0}

    class FakeRanker:
        def __init__(self, priors):
            self.priors = priors

        def update(self, podium, session_name, confidence):
            return None

        def get_current_ratings(self):
            return pd.DataFrame([{"driver_code": "NOR", "rating_mu": 1.23}])

    class FakePredictor:
        instances: list[FakePredictor] = []

        def __init__(self, season_year=2026):
            self.season_year = season_year
            self.calls: list[dict] = []
            FakePredictor.instances.append(self)

        def predict_race(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "finish_order": [
                    {
                        "driver": "Lando Norris",
                        "confidence": 62.0,
                    }
                ]
            }

    class FakeLearner:
        def get_recommended_method(self, weekend_type):
            return {"method": "blend"}

        def update_after_race(self, **kwargs):
            return {"recommendations": []}

    import src.models.bayesian as bayesian_module
    import src.models.priors_factory as priors_factory_module
    import src.models.regulations as regulations_module
    import src.predictors as predictors_module
    import src.systems.learning as learning_module
    import src.utils.lineups as lineups_module
    import src.utils.weekend as weekend_module

    monkeypatch.setattr(bayesian_module, "BayesianDriverRanking", FakeRanker)
    monkeypatch.setattr(priors_factory_module, "PriorsFactory", FakeFactory)
    monkeypatch.setattr(regulations_module, "apply_2026_regulations", lambda priors: priors)
    monkeypatch.setattr(predictors_module, "Baseline2026Predictor", FakePredictor)
    monkeypatch.setattr(learning_module, "LearningSystem", FakeLearner)
    monkeypatch.setattr(
        lineups_module,
        "get_lineups",
        lambda year, race_name: {
            "McLaren": ["Lando Norris", "Oscar Piastri"],
            "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
        },
    )
    monkeypatch.setattr(weekend_module, "get_weekend_type", lambda year, race_name: "conventional")
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index=False: None)

    module.run_simulation_loop(year=2026)

    predictor = FakePredictor.instances[0]
    assert predictor.season_year == 2026
    assert len(predictor.calls) == 4
    assert predictor.calls[0]["race_name"] == "Australian Grand Prix"
    assert predictor.calls[0]["year"] == 2026
    assert predictor.calls[0]["weather"] == "dry"


def test_update_prediction_actuals_uses_shared_results_fetcher():
    """The prediction-actuals script should import the shared results fetcher."""
    module = _load_module(
        "update_prediction_actuals_script",
        "scripts/update_prediction_actuals.py",
    )

    assert module.fetch_actual_session_results.__module__ == "src.data.actual_results_fetcher"


def test_update_from_race_script_reconciles_saved_prediction_actuals(monkeypatch):
    """The race-update script should refresh saved prediction actuals after learning updates."""
    module = _load_module("update_from_race_script", "scripts/update_from_race.py")
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "update_from_race",
        lambda year, race_name, data_dir: calls.update(
            update=(int(year), str(race_name), str(data_dir))
        ),
    )

    class _Logger:
        def reconcile_completed_prediction_actuals(self, year: int) -> int:
            calls["reconciled_year"] = int(year)
            return 3

    monkeypatch.setattr(module, "PredictionLogger", lambda: _Logger())
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["update_from_race.py", "Australian Grand Prix", "--year", "2026"],
    )

    module.main()

    assert calls["update"] == (2026, "Australian Grand Prix", "data/processed")
    assert calls["reconciled_year"] == 2026


def test_blend_weight_script_uses_shared_results_fetcher():
    """The blend-weight helper should import the shared results fetcher."""
    module = _load_module("blend_weight_script", "scripts/test_blend_weights.py")

    assert module.fetch_actual_session_results.__module__ == "src.data.actual_results_fetcher"


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("blend_weight_script_predictor", "scripts/test_blend_weights.py"),
        ("predictor_db_script", "scripts/test_predictor_with_db.py"),
        ("backtest_script", "scripts/backtest_2025_season.py"),
    ],
)
def test_helper_scripts_import_predictor_from_package_root(
    monkeypatch,
    module_name: str,
    relative_path: str,
):
    """Helper scripts should import the predictor from the package root."""
    import src.predictors as predictors_module

    class SentinelPredictor:
        """Marker class used to confirm which predictor object the script imported."""

    monkeypatch.setattr(predictors_module, "Baseline2026Predictor", SentinelPredictor)

    module = _load_module(module_name, relative_path)

    assert module.Baseline2026Predictor is SentinelPredictor
