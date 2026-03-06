from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

import src.predictors.baseline.race.prediction_mixin as prediction_module
from src.predictors.baseline.race.prediction_mixin import BaselineRacePredictionMixin


class DummyRacePredictor(BaselineRacePredictionMixin):
    seed = 7

    def _load_race_params(self) -> dict:
        return {}

    def _prepare_driver_info_with_compounds(
        self, qualifying_grid: list[dict], race_name: str | None
    ) -> tuple[dict, int]:
        return (
            {
                "DRV": {
                    "driver": "DRV",
                    "team": "Team",
                    "grid_pos": 1,
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 0.5,
                    "race_advantage": 0.0,
                    "overtaking_skill": 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            },
            0,
        )


def _stub_prediction_dependencies(stack: ExitStack):
    stack.enter_context(
        patch.object(prediction_module, "load_track_specific_params", lambda _race_name: {})
    )
    stack.enter_context(
        patch.object(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "get_available_compounds",
            lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"],
        )
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "resolve_track_temperature_c",
            lambda *args, **kwargs: 31.0,
        )
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "resolve_track_temperature_profile",
            lambda *args, **kwargs: {
                "track_temperature_c": 31.0,
                "source": "session_weather_blend",
                "reason": "session_signal_available",
                "weather_bucket": "dry",
                "session_name": "Q",
                "session_track_temperature_c": 32.0,
                "session_temperature_source": "track_temp",
                "session_air_temperature_c": None,
                "forecast_track_temperature_c": 29.0,
                "session_weight": 0.7,
                "forecast_weight": 0.3,
                "blend_enabled": True,
            },
        )
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "resolve_non_competitive_weather_features",
            lambda *args, **kwargs: {
                "available": True,
                "source_session": "FP3",
                "reason": "session_weather_available",
                "practice_weather_bucket": "dry",
                "track_temperature_c": 33.0,
                "air_temperature_c": 24.0,
                "wind_speed_kph": 18.0,
                "humidity_pct": 52.0,
                "rainfall_signal": 0.0,
            },
        )
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "resolve_race_distance_laps",
            lambda year, race_name, is_sprint: 60,
        )
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "simulate_race_lap_by_lap",
            lambda **kwargs: {
                "finish_order": ["DRV"],
                "dnf_drivers": [],
                "strategies_used": kwargs["strategies"],
            },
        )
    )
    stack.enter_context(
        patch.object(
            prediction_module,
            "aggregate_simulation_results",
            lambda _simulation_results: {
                "median_positions": {"DRV": 1},
                "position_distributions": {"DRV": [1]},
                "dnf_rates": {"DRV": 0.0},
                "compound_strategy_distribution": {"SOFT→MEDIUM": 1.0},
                "pit_lap_distribution": {"lap_30-35": 1.0},
            },
        )
    )


def test_predict_race_enforces_two_compounds_for_mixed_weather():
    predictor = DummyRacePredictor()

    enforce_flags: list[bool] = []

    def _fake_generate_pit_strategy(**kwargs):
        enforce_flags.append(bool(kwargs["enforce_two_compound_rule"]))
        return {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [30, 30],
        }

    with ExitStack() as stack:
        _stub_prediction_dependencies(stack)
        stack.enter_context(
            patch.object(prediction_module, "generate_pit_strategy", _fake_generate_pit_strategy)
        )

        predictor.predict_race(
            qualifying_grid=[{"driver": "DRV", "team": "Team", "position": 1}],
            weather="mixed",
            race_name="Bahrain Grand Prix",
            n_simulations=1,
        )

    assert enforce_flags == [True]


def test_predict_race_exposes_track_temperature_context():
    predictor = DummyRacePredictor()

    def _fake_generate_pit_strategy(**_kwargs):
        return {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["SOFT", "MEDIUM"],
            "stint_lengths": [30, 30],
        }

    with ExitStack() as stack:
        _stub_prediction_dependencies(stack)
        stack.enter_context(
            patch.object(prediction_module, "generate_pit_strategy", _fake_generate_pit_strategy)
        )
        result = predictor.predict_race(
            qualifying_grid=[{"driver": "DRV", "team": "Team", "position": 1}],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=1,
        )

    context = result["track_temperature_context"]
    assert context["source"] == "session_weather_blend"
    assert context["session_name"] == "Q"
    assert context["track_temperature_c"] == pytest.approx(31.0)
    assert context["session_weight"] == pytest.approx(0.7)
    assert context["forecast_weight"] == pytest.approx(0.3)

    weather_context = result["weather_feature_context"]
    assert weather_context["available"] is True
    assert weather_context["source_session"] == "FP3"
    assert weather_context["selected_weather"] == "dry"
    assert weather_context["practice_weather_bucket"] == "dry"
    assert weather_context["chaos_multiplier"] == pytest.approx(1.0)
    assert weather_context["teammate_variance_multiplier"] == pytest.approx(1.0)
    assert weather_context["confidence_adjustment"] == pytest.approx(0.0)


def test_predict_race_allows_single_compound_rule_override_for_rain():
    predictor = DummyRacePredictor()

    enforce_flags: list[bool] = []

    def _fake_generate_pit_strategy(**kwargs):
        enforce_flags.append(bool(kwargs["enforce_two_compound_rule"]))
        return {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["INTERMEDIATE", "INTERMEDIATE"],
            "stint_lengths": [30, 30],
        }

    with ExitStack() as stack:
        _stub_prediction_dependencies(stack)
        stack.enter_context(
            patch.object(prediction_module, "generate_pit_strategy", _fake_generate_pit_strategy)
        )

        predictor.predict_race(
            qualifying_grid=[{"driver": "DRV", "team": "Team", "position": 1}],
            weather="rain",
            race_name="Bahrain Grand Prix",
            n_simulations=1,
        )

    assert enforce_flags == [False]


def test_predict_race_weather_feature_context_reflects_bucket_mismatch():
    predictor = DummyRacePredictor()

    def _fake_generate_pit_strategy(**_kwargs):
        return {
            "num_stops": 1,
            "pit_laps": [30],
            "compound_sequence": ["INTERMEDIATE", "WET"],
            "stint_lengths": [30, 30],
        }

    with ExitStack() as stack:
        _stub_prediction_dependencies(stack)
        stack.enter_context(
            patch.object(prediction_module, "generate_pit_strategy", _fake_generate_pit_strategy)
        )
        result = predictor.predict_race(
            qualifying_grid=[{"driver": "DRV", "team": "Team", "position": 1}],
            weather="rain",
            race_name="Bahrain Grand Prix",
            n_simulations=1,
        )

    weather_context = result["weather_feature_context"]
    assert weather_context["selected_weather"] == "rain"
    assert weather_context["practice_weather_bucket"] == "dry"
    assert weather_context["weather_mismatch_score"] == pytest.approx(1.0)
    assert weather_context["chaos_multiplier"] > 1.0
    assert weather_context["teammate_variance_multiplier"] > 1.0
    assert weather_context["confidence_adjustment"] > 0.0


def test_predict_race_caps_extreme_backmarker_recovery():
    class ExtremeRecoveryPredictor(BaselineRacePredictionMixin):
        seed = 11

        def _load_race_params(self) -> dict:
            return {}

        def _prepare_driver_info_with_compounds(
            self, qualifying_grid: list[dict], race_name: str | None
        ) -> tuple[dict, int]:
            info_map = {}
            for entry in qualifying_grid:
                driver = entry["driver"]
                is_extreme_recovery = driver == "D22"
                info_map[driver] = {
                    "driver": driver,
                    "team": entry["team"],
                    "grid_pos": entry["position"],
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 1.0 if is_extreme_recovery else 0.5,
                    "race_advantage": 0.35 if is_extreme_recovery else 0.0,
                    "overtaking_skill": 1.0 if is_extreme_recovery else 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            return info_map, 0

    predictor = ExtremeRecoveryPredictor()

    strategy = {
        "num_stops": 0,
        "pit_laps": [],
        "compound_sequence": ["MEDIUM"],
        "stint_lengths": [60],
    }

    def _fake_aggregate(_simulation_results):
        median_positions = {}
        position_distributions = {}
        for idx in range(1, 23):
            driver = f"D{idx:02d}"
            median = 1 if driver == "D22" else idx
            median_positions[driver] = median
            position_distributions[driver] = [median, median, median]
        return {
            "median_positions": median_positions,
            "position_distributions": position_distributions,
            "dnf_rates": {f"D{idx:02d}": 0.0 for idx in range(1, 23)},
            "compound_strategy_distribution": {"MEDIUM": 1.0},
            "pit_lap_distribution": {},
        }

    qualifying_grid = [
        {"driver": f"D{idx:02d}", "team": f"Team{idx:02d}", "position": idx} for idx in range(1, 23)
    ]

    original_config_get = prediction_module.config_loader.get
    overrides = {
        "baseline_predictor.race.grid_anchor.base": 0.0,
        "baseline_predictor.race.grid_anchor.track_scale": 0.0,
        "baseline_predictor.race.grid_anchor.min": 0.0,
        "baseline_predictor.race.grid_anchor.sprint_min": 0.0,
        "baseline_predictor.race.final_blend.overtaking_skill_scale": 6.0,
        "baseline_predictor.race.final_blend.race_advantage_scale": 6.0,
        "baseline_predictor.race.final_blend.driver_skill_scale": 6.0,
        "baseline_predictor.race.final_blend.max_driver_adjustment_positions": 20.0,
    }

    def _config_get(key, default=None):
        if key in overrides:
            return overrides[key]
        return original_config_get(key, default)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                prediction_module,
                "load_track_specific_params",
                lambda _race_name: {"track_overtaking": 0.05},
            )
        )
        stack.enter_context(
            patch.object(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "get_available_compounds",
                lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD"],
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "resolve_race_distance_laps",
                lambda year, race_name, is_sprint: 60,
            )
        )
        stack.enter_context(
            patch.object(prediction_module, "generate_pit_strategy", lambda **kwargs: strategy)
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "simulate_race_lap_by_lap",
                lambda **kwargs: {
                    "finish_order": [
                        entry["driver"] for entry in kwargs["driver_info_map"].values()
                    ],
                    "dnf_drivers": [],
                    "strategies_used": kwargs["strategies"],
                },
            )
        )
        stack.enter_context(
            patch.object(prediction_module, "aggregate_simulation_results", _fake_aggregate)
        )
        stack.enter_context(patch.object(prediction_module.config_loader, "get", _config_get))

        result = predictor.predict_race(
            qualifying_grid=qualifying_grid,
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=1,
        )

    positions = {entry["driver"]: entry["position"] for entry in result["finish_order"]}
    assert positions["D22"] > 1
    assert positions["D22"] >= 10


def test_predict_race_podium_probability_matches_ranked_outcomes():
    class PodiumProbabilityPredictor(BaselineRacePredictionMixin):
        seed = 21

        def _load_race_params(self) -> dict:
            return {}

        def _prepare_driver_info_with_compounds(
            self, qualifying_grid: list[dict], race_name: str | None
        ) -> tuple[dict, int]:
            _ = race_name
            info_map = {}
            for row in qualifying_grid:
                driver = row["driver"]
                info_map[driver] = {
                    "driver": driver,
                    "team": row["team"],
                    "grid_pos": row["position"],
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 0.5,
                    "race_advantage": 0.0,
                    "overtaking_skill": 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            return info_map, 0

    predictor = PodiumProbabilityPredictor()

    qualifying_grid = [
        {"driver": "A", "team": "TeamA", "position": 10},
        {"driver": "B", "team": "TeamB", "position": 11},
        {"driver": "C", "team": "TeamC", "position": 12},
        {"driver": "D", "team": "TeamD", "position": 13},
    ]

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(prediction_module, "load_track_specific_params", lambda _race_name: {})
        )
        stack.enter_context(
            patch.object(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "get_available_compounds",
                lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD"],
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "resolve_race_distance_laps",
                lambda year, race_name, is_sprint: 60,
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "generate_pit_strategy",
                lambda **kwargs: {
                    "num_stops": 1,
                    "pit_laps": [30],
                    "compound_sequence": ["SOFT", "MEDIUM"],
                    "stint_lengths": [30, 30],
                },
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "simulate_race_lap_by_lap",
                lambda **kwargs: {
                    "finish_order": [
                        entry["driver"] for entry in kwargs["driver_info_map"].values()
                    ],
                    "dnf_drivers": [],
                    "strategies_used": kwargs["strategies"],
                },
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "aggregate_simulation_results",
                lambda _simulation_results: {
                    "median_positions": {"A": 10, "B": 11, "C": 12, "D": 13},
                    "position_distributions": {
                        "A": [10, 10, 10],
                        "B": [11, 11, 11],
                        "C": [12, 12, 12],
                        "D": [13, 13, 13],
                    },
                    "dnf_rates": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
                    "compound_strategy_distribution": {"SOFT→MEDIUM": 1.0},
                    "pit_lap_distribution": {"lap_30-35": 1.0},
                },
            )
        )

        result = predictor.predict_race(
            qualifying_grid=qualifying_grid,
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=3,
        )

    by_position = sorted(result["finish_order"], key=lambda row: row["position"])
    assert by_position[0]["podium_probability"] == 100.0
    assert by_position[1]["podium_probability"] == 100.0
    assert by_position[2]["podium_probability"] == 100.0
    assert by_position[3]["podium_probability"] == 0.0


def test_predict_race_widens_top_interval_when_input_confidence_is_low():
    class IntervalFloorPredictor(BaselineRacePredictionMixin):
        seed = 31

        def _load_race_params(self) -> dict:
            return {}

        def _prepare_driver_info_with_compounds(
            self, qualifying_grid: list[dict], race_name: str | None
        ) -> tuple[dict, int]:
            _ = race_name
            info_map = {}
            for row in qualifying_grid:
                info_map[row["driver"]] = {
                    "driver": row["driver"],
                    "team": row["team"],
                    "grid_pos": row["position"],
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 0.5,
                    "race_advantage": 0.0,
                    "overtaking_skill": 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            return info_map, 0

    predictor = IntervalFloorPredictor()

    qualifying_grid = [
        {"driver": "A", "team": "TeamA", "position": 1},
        {"driver": "B", "team": "TeamB", "position": 2},
        {"driver": "C", "team": "TeamC", "position": 3},
    ]

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(prediction_module, "load_track_specific_params", lambda _race_name: {})
        )
        stack.enter_context(
            patch.object(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "get_available_compounds",
                lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD"],
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "resolve_race_distance_laps",
                lambda year, race_name, is_sprint: 60,
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "generate_pit_strategy",
                lambda **kwargs: {
                    "num_stops": 0,
                    "pit_laps": [],
                    "compound_sequence": ["MEDIUM"],
                    "stint_lengths": [60],
                },
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "simulate_race_lap_by_lap",
                lambda **kwargs: {
                    "finish_order": [
                        entry["driver"] for entry in kwargs["driver_info_map"].values()
                    ],
                    "dnf_drivers": [],
                    "strategies_used": kwargs["strategies"],
                },
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "aggregate_simulation_results",
                lambda _simulation_results: {
                    "median_positions": {"A": 1, "B": 2, "C": 3},
                    "position_distributions": {"A": [1, 1, 1], "B": [2, 2, 2], "C": [3, 3, 3]},
                    "dnf_rates": {"A": 0.0, "B": 0.0, "C": 0.0},
                    "compound_strategy_distribution": {"MEDIUM": 1.0},
                    "pit_lap_distribution": {},
                },
            )
        )

        result = predictor.predict_race(
            qualifying_grid=qualifying_grid,
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=3,
            input_confidence=0.35,
        )

    by_position = sorted(result["finish_order"], key=lambda row: row["position"])
    assert by_position[0]["driver"] == "A"
    assert by_position[0]["p5"] == 1
    assert by_position[0]["p95"] >= 2


def test_predict_race_applies_learned_position_adjustment():
    class _CalibrationStub:
        def get_combined_position_adjustment(
            self,
            *,
            team,
            driver,
            teammates,
            session,
            min_samples,
            driver_error_scale,
            teammate_gap_scale,
            max_adjustment,
        ):
            _ = (
                team,
                teammates,
                session,
                min_samples,
                driver_error_scale,
                teammate_gap_scale,
                max_adjustment,
            )
            return 2.0 if driver == "A" else -1.5

    class _Config:
        def get(self, key, default=None):
            overrides = {
                "baseline_predictor.race.learning.position_adjustment_scale": 1.0,
                "baseline_predictor.race.grid_anchor.base": 0.4,
                "baseline_predictor.race.grid_anchor.track_scale": 0.0,
                "baseline_predictor.race.grid_anchor.min": 0.4,
                "baseline_predictor.race.grid_anchor.sprint_min": 0.4,
            }
            return overrides.get(key, default)

    class LearnedAdjustmentPredictor(BaselineRacePredictionMixin):
        seed = 77

        def __init__(self):
            self.calibration_system = _CalibrationStub()
            self.config = _Config()

        def _load_race_params(self) -> dict:
            return {}

        def _prepare_driver_info_with_compounds(
            self, qualifying_grid: list[dict], race_name: str | None
        ) -> tuple[dict, int]:
            _ = race_name
            info_map = {}
            for row in qualifying_grid:
                info_map[row["driver"]] = {
                    "driver": row["driver"],
                    "team": row["team"],
                    "grid_pos": row["position"],
                    "team_strength": 0.5,
                    "team_strength_by_compound": {"SOFT": 0.5, "MEDIUM": 0.5, "HARD": 0.5},
                    "tire_deg_by_compound": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
                    "skill": 0.5,
                    "race_advantage": 0.0,
                    "overtaking_skill": 0.5,
                    "defensive_skill": 0.5,
                    "dnf_probability": 0.0,
                }
            return info_map, 0

    predictor = LearnedAdjustmentPredictor()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(prediction_module, "load_track_specific_params", lambda _race_name: {})
        )
        stack.enter_context(
            patch.object(prediction_module, "get_tire_stress_score", lambda _race_name: 3.0)
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "get_available_compounds",
                lambda _race_name, weather="dry": ["SOFT", "MEDIUM", "HARD"],
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "resolve_race_distance_laps",
                lambda year, race_name, is_sprint: 60,
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "generate_pit_strategy",
                lambda **kwargs: {
                    "num_stops": 0,
                    "pit_laps": [],
                    "compound_sequence": ["MEDIUM"],
                    "stint_lengths": [60],
                },
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "simulate_race_lap_by_lap",
                lambda **kwargs: {
                    "finish_order": [
                        entry["driver"] for entry in kwargs["driver_info_map"].values()
                    ],
                    "dnf_drivers": [],
                    "strategies_used": kwargs["strategies"],
                },
            )
        )
        stack.enter_context(
            patch.object(
                prediction_module,
                "aggregate_simulation_results",
                lambda _simulation_results: {
                    "median_positions": {"A": 2, "B": 1},
                    "position_distributions": {"A": [2, 2, 2], "B": [1, 1, 1]},
                    "dnf_rates": {"A": 0.0, "B": 0.0},
                    "compound_strategy_distribution": {"MEDIUM": 1.0},
                    "pit_lap_distribution": {},
                },
            )
        )

        result = predictor.predict_race(
            qualifying_grid=[
                {"driver": "A", "team": "TeamX", "position": 2},
                {"driver": "B", "team": "TeamX", "position": 1},
            ],
            weather="dry",
            race_name="Bahrain Grand Prix",
            n_simulations=3,
        )

    by_position = sorted(result["finish_order"], key=lambda row: row["position"])
    assert by_position[0]["driver"] == "A"
    assert by_position[1]["driver"] == "B"


@pytest.mark.parametrize(
    ("input_values", "expected_first", "expected_last"),
    [
        ([100.0, 2.0, 10.0, 0.0], 100.0, 0.0),
        ([75.0, 60.0, 62.0, 20.0], 75.0, 20.0),
    ],
)
def test_enforce_non_increasing_podium_probabilities(input_values, expected_first, expected_last):
    smoothed = BaselineRacePredictionMixin._enforce_non_increasing(input_values)

    assert smoothed[0] == expected_first
    assert smoothed[-1] == expected_last
    assert all(smoothed[idx] >= smoothed[idx + 1] for idx in range(len(smoothed) - 1))
