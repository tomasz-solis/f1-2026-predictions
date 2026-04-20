"""Tests for prediction-time runtime context and injected config overrides."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.track_data_loader import get_tire_stress_score
from src.simulation.pit_strategy import generate_pit_strategy
from src.utils.backtesting import NestedDictConfig
from src.utils.fp_blending import get_fp_team_performance
from src.utils.prediction_context import PredictionContext, activate_prediction_runtime


def test_historical_context_prevents_wall_clock_staleness_rejection():
    """Historical prediction context should anchor staleness to the target session."""
    laps = pd.DataFrame(
        {
            "Driver": ["NOR"] * 5 + ["PIA"] * 5,
            "Team": ["McLaren"] * 10,
            "LapTime": [pd.Timedelta(seconds=90 + (lap * 0.1)) for lap in range(10)],
            "Compound": ["SOFT"] * 10,
        }
    )
    mock_session = MagicMock()
    mock_session.laps = laps
    mock_session.date = datetime(2026, 3, 13, 10, tzinfo=UTC)

    with patch("src.utils.fp_blending.ff1.get_session", return_value=mock_session):
        with activate_prediction_runtime(
            prediction_context=PredictionContext(
                mode="historical",
                target_session_datetime=datetime(2026, 3, 13, 16, tzinfo=UTC),
            )
        ):
            perf, session_laps, error = get_fp_team_performance(
                2026,
                "Australian Grand Prix",
                "FP1",
                max_data_age_hours=12.0,
            )

    assert perf is not None
    assert session_laps is not None
    assert error is None


def test_active_runtime_config_overrides_pit_strategy_defaults():
    """Pit-strategy helpers should honor injected config instead of global defaults."""
    override_config = NestedDictConfig(
        {
            "baseline_predictor": {
                "race": {
                    "tire_strategy": {
                        "stop_probability": {
                            "high_stress_2stop": 1.0,
                            "medium_stress_1stop": 0.0,
                            "low_stress_1stop": 0.0,
                        },
                        "windows": {
                            "one_stop": [20, 22],
                            "two_stop_first": [12, 14],
                            "two_stop_second": [32, 34],
                        },
                    },
                    "strategy_constraints": {
                        "pit_lap_variance": {"one_stop": 0.1, "two_stop": 0.1},
                        "min_pit_lap": 5,
                        "max_pit_lap_from_end": 5,
                        "min_laps_between_stops": 8,
                        "strategy_optimality": 1.0,
                    },
                },
                "compound_selection": {
                    "high_stress_threshold": 3.0,
                    "low_stress_threshold": 2.0,
                },
            }
        }
    )

    with activate_prediction_runtime(config=override_config):
        strategy = generate_pit_strategy(
            race_distance=57,
            tire_stress_score=4.0,
            available_compounds=["SOFT", "MEDIUM", "HARD"],
            rng=np.random.default_rng(7),
        )

    assert strategy["num_stops"] == 2
    assert len(strategy["pit_laps"]) == 2


def test_active_runtime_config_overrides_track_loader_defaults():
    """Track-data helpers should read injected config during prediction runs."""
    override_config = NestedDictConfig(
        {
            "baseline_predictor": {
                "compound_selection": {
                    "default_stress_fallback": 4.2,
                }
            }
        }
    )

    with activate_prediction_runtime(config=override_config):
        stress_score = get_tire_stress_score(race_name=None, year=2026)

    assert stress_score == 4.2
