"""Tests for track parameter helpers."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError

from src.data.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_non_competitive_weather_features,
    resolve_race_distance_laps,
    resolve_track_temperature_c,
    resolve_track_temperature_profile,
)


def setup_function():
    resolve_non_competitive_weather_features.cache_clear()
    resolve_track_temperature_profile.cache_clear()
    resolve_track_temperature_c.cache_clear()
    resolve_race_distance_laps.cache_clear()


def test_get_available_compounds_is_weather_aware():
    assert get_available_compounds("Bahrain Grand Prix", weather="dry") == [
        "SOFT",
        "MEDIUM",
        "HARD",
    ]
    assert get_available_compounds("Bahrain Grand Prix", weather="rain") == ["INTERMEDIATE", "WET"]
    assert get_available_compounds("Bahrain Grand Prix", weather="mixed") == [
        "SOFT",
        "MEDIUM",
        "HARD",
        "INTERMEDIATE",
    ]


def test_resolve_race_distance_uses_known_track_mapping():
    assert resolve_race_distance_laps(2026, "Monaco Grand Prix", is_sprint=False) == 78
    assert resolve_race_distance_laps(2026, "British Grand Prix", is_sprint=True) == 17


def test_resolve_race_distance_uses_fastf1_metadata_for_unknown_tracks():
    mock_session = MagicMock()
    mock_session.total_laps = 63

    with patch("src.data.track_data_loader.fastf1.get_session", return_value=mock_session):
        laps = resolve_race_distance_laps(2026, "Imaginary Grand Prix", is_sprint=False)

    assert laps == 63
    mock_session.load.assert_not_called()


def test_resolve_race_distance_falls_back_when_fastf1_fails():
    with patch("src.data.track_data_loader.fastf1.get_session", side_effect=RuntimeError("boom")):
        assert resolve_race_distance_laps(2026, "Unknown Grand Prix", is_sprint=False) == 60
        assert resolve_race_distance_laps(2026, "Unknown Grand Prix", is_sprint=True) == 20


def test_load_track_specific_params_uses_requested_year_payload(tmp_path):
    processed_root = tmp_path / "processed"
    track_path = processed_root / "track_characteristics" / "2027_track_characteristics.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Bahrain Grand Prix": {
                        "pit_stop_loss": 22.5,
                        "safety_car_prob": 0.41,
                        "overtaking_difficulty": 0.36,
                    }
                }
            }
        )
    )

    with patch(
        "src.data.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Bahrain Grand Prix", year=2027)

    assert params["pit_stops"]["loss_duration"] == 22.5
    assert params["sc_probability"] == 0.41
    assert params["track_overtaking"] == 0.36


def test_load_track_specific_params_falls_back_to_previous_available_year(tmp_path):
    processed_root = tmp_path / "processed"
    fallback_path = processed_root / "track_characteristics" / "2026_track_characteristics.json"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Bahrain Grand Prix": {
                        "pit_stop_loss": 20.0,
                        "safety_car_prob": 0.33,
                        "overtaking_difficulty": 0.44,
                    }
                }
            }
        )
    )

    with patch(
        "src.data.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Bahrain Grand Prix", year=2028)

    assert params["pit_stops"]["loss_duration"] == 20.0
    assert params["sc_probability"] == 0.33
    assert params["track_overtaking"] == 0.44


def test_load_track_specific_params_normalizes_underscaled_overtaking_values(tmp_path):
    processed_root = tmp_path / "processed"
    track_path = processed_root / "track_characteristics" / "2026_track_characteristics.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Monaco Grand Prix": {
                        "pit_stop_loss": 19.0,
                        "safety_car_prob": 0.7,
                        "overtaking_difficulty": 0.03,
                    }
                }
            }
        )
    )

    with patch(
        "src.data.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Monaco Grand Prix", year=2026)

    assert params["track_overtaking"] == 0.95


def test_load_track_specific_params_accepts_categorical_overtaking_difficulty(tmp_path):
    processed_root = tmp_path / "processed"
    track_path = processed_root / "track_characteristics" / "2026_track_characteristics.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Monaco Grand Prix": {
                        "pit_stop_loss": 19.0,
                        "safety_car_prob": 0.7,
                        "overtaking_difficulty": "very_hard",
                    }
                }
            }
        )
    )

    with patch(
        "src.data.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Monaco Grand Prix", year=2026)

    assert params["track_overtaking"] == pytest.approx(0.95)


def test_load_track_specific_params_infers_overtaking_from_likelihood_when_needed(tmp_path):
    processed_root = tmp_path / "processed"
    track_path = processed_root / "track_characteristics" / "2026_track_characteristics.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Australian Grand Prix": {
                        "pit_stop_loss": 18.2,
                        "safety_car_prob": 0.3,
                        "overtaking_difficulty": 0.0,
                        "overtaking_likelihood": 0.62,
                    }
                }
            }
        )
    )

    with patch(
        "src.data.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Australian Grand Prix", year=2026)

    assert params["track_overtaking"] == pytest.approx(0.38)


def test_load_track_specific_params_uses_default_baseline_for_unknown_under_scaled_track(tmp_path):
    processed_root = tmp_path / "processed"
    track_path = processed_root / "track_characteristics" / "2026_track_characteristics.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Imaginary Grand Prix": {
                        "pit_stop_loss": 22.0,
                        "safety_car_prob": 0.3,
                        "overtaking_difficulty": 0.01,
                    }
                }
            }
        )
    )

    def _cfg_get(key, default=None):
        if key == "paths.processed":
            return str(processed_root)
        if key == "track_defaults.overtaking_difficulty":
            return 0.46
        return default

    with patch("src.data.track_data_loader.config_loader.get", side_effect=_cfg_get):
        params = load_track_specific_params("Imaginary Grand Prix", year=2026)

    assert params["track_overtaking"] == pytest.approx(0.46)


def test_load_track_specific_params_blends_observed_overtaking_gradually(tmp_path):
    processed_root = tmp_path / "processed"
    track_path = processed_root / "track_characteristics" / "2026_track_characteristics.json"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        json.dumps(
            {
                "tracks": {
                    "Australian Grand Prix": {
                        "pit_stop_loss": 18.2,
                        "safety_car_prob": 0.3,
                        "overtaking_difficulty": 0.2,
                        "overtaking_observed_races": 1,
                    }
                }
            }
        )
    )

    with patch(
        "src.data.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Australian Grand Prix", year=2026)

    assert params["track_overtaking"] == pytest.approx(0.45, abs=0.02)
    assert 0.2 < params["track_overtaking"] < 0.5


def test_get_tire_stress_score_uses_resolved_year_file(tmp_path):
    pirelli_path = tmp_path / "2027_pirelli_info.json"
    pirelli_path.write_text(
        json.dumps(
            {
                "bahrain_grand_prix": {
                    "tyre_stress": {
                        "traction": 4.0,
                        "braking": 3.0,
                        "lateral": 2.0,
                        "asphalt_abrasion": 5.0,
                    }
                }
            }
        )
    )

    with patch("src.data.track_data_loader._resolve_pirelli_path", return_value=pirelli_path):
        stress = get_tire_stress_score("Bahrain Grand Prix", year=2027)

    assert stress == 3.5


def test_resolve_track_temperature_prefers_latest_completed_session_weather():
    now_utc = datetime.now(UTC)
    event = MagicMock()
    event.get_session_date.side_effect = lambda session_name: {
        "R": now_utc + timedelta(hours=2),
        "Q": now_utc - timedelta(hours=3),
        "FP3": now_utc - timedelta(hours=6),
        "FP2": now_utc - timedelta(hours=9),
        "FP1": now_utc - timedelta(hours=12),
    }.get(session_name)

    quali_session = MagicMock()
    quali_session.weather_data = pd.DataFrame({"TrackTemp": [36.0, 37.0, 38.0]})
    quali_session.session_status = pd.DataFrame({"Status": ["Finished"]})

    with (
        patch("src.data.track_data_loader.fastf1.get_event", return_value=event),
        patch(
            "src.data.track_data_loader.fastf1.get_session", return_value=quali_session
        ) as get_session,
    ):
        resolved = resolve_track_temperature_c(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="dry",
            is_sprint=False,
        )

    assert resolved == pytest.approx(36.8)
    get_session.assert_called_once_with(2026, "Bahrain Grand Prix", "Q")


def test_resolve_track_temperature_uses_air_temp_when_track_temp_missing():
    now_utc = datetime.now(UTC)
    event = MagicMock()
    event.get_session_date.side_effect = lambda session_name: {
        "R": now_utc + timedelta(hours=2),
        "Q": now_utc + timedelta(hours=1),
        "FP3": now_utc + timedelta(hours=1),
        "FP2": now_utc + timedelta(hours=1),
        "FP1": now_utc - timedelta(hours=3),
    }.get(session_name)

    fp1_session = MagicMock()
    fp1_session.weather_data = pd.DataFrame({"AirTemp": [21.0, 22.0, 23.0]})
    fp1_session.session_status = pd.DataFrame({"Status": ["Finished"]})

    def _cfg_get(key, default=None):
        if key == "baseline_predictor.race.track_temperature.air_to_track_offset_c":
            return 8.0
        return default

    with (
        patch("src.data.track_data_loader.fastf1.get_event", return_value=event),
        patch(
            "src.data.track_data_loader.fastf1.get_session", return_value=fp1_session
        ) as get_session,
        patch("src.data.track_data_loader.config_loader.get", side_effect=_cfg_get),
    ):
        resolved = resolve_track_temperature_c(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="mixed",
            is_sprint=False,
        )

    assert resolved == pytest.approx(29.6)
    get_session.assert_called_once_with(2026, "Bahrain Grand Prix", "FP1")


def test_resolve_track_temperature_falls_back_when_fastf1_unavailable():
    with patch(
        "src.data.track_data_loader.fastf1.get_event",
        side_effect=RuntimeError("network unavailable"),
    ):
        resolved = resolve_track_temperature_c(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="rain",
            is_sprint=False,
        )

    assert resolved == pytest.approx(23.0)


def test_resolve_track_temperature_handles_unloaded_session_status():
    """Unloaded FastF1 session status should not block weather temperature."""
    now_utc = datetime.now(UTC)
    event = MagicMock()
    event.get_session_date.side_effect = lambda session_name: {
        "R": now_utc + timedelta(hours=2),
        "Q": now_utc - timedelta(hours=3),
        "FP3": now_utc - timedelta(hours=6),
        "FP2": now_utc - timedelta(hours=9),
        "FP1": now_utc - timedelta(hours=12),
    }.get(session_name)

    class _WeatherOnlySession:
        """Weather-only FastF1 stub that does not expose session status."""

        def __init__(self) -> None:
            self.weather_data = pd.DataFrame({"TrackTemp": [36.0, 37.0, 38.0]})

        def load(self, **_kwargs) -> None:
            """Simulate a successful weather-only load."""

        @property
        def session_status(self):
            """Match FastF1 behavior when laps were not loaded."""
            raise DataNotLoadedError("The data you are trying to access has not been loaded yet.")

    with (
        patch("src.data.track_data_loader.fastf1.get_event", return_value=event),
        patch("src.data.track_data_loader.fastf1.get_session", return_value=_WeatherOnlySession()),
    ):
        resolved = resolve_track_temperature_c(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="dry",
            is_sprint=False,
        )

    assert resolved == pytest.approx(36.8)


def test_resolve_track_temperature_profile_skips_fastf1_when_session_weather_disabled():
    """Disabled session weather should use forecast fallback without FastF1 lookup."""
    with (
        patch(
            "src.data.track_data_loader._cfg_get",
            side_effect=lambda key, default=None: (
                False
                if key == "baseline_predictor.race.track_temperature.session_weather_enabled"
                else default
            ),
        ),
        patch("src.data.track_data_loader.fastf1.get_event") as get_event,
    ):
        profile = resolve_track_temperature_profile(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="dry",
            is_sprint=False,
        )

    assert profile["source"] == "forecast_fallback"
    assert profile["reason"] == "session_weather_disabled"
    assert profile["track_temperature_c"] == pytest.approx(36.0)
    get_event.assert_not_called()


def test_resolve_track_temperature_profile_reports_session_blend_metadata():
    now_utc = datetime.now(UTC)
    event = MagicMock()
    event.get_session_date.side_effect = lambda session_name: {
        "R": now_utc + timedelta(hours=2),
        "Q": now_utc - timedelta(hours=3),
        "FP3": now_utc - timedelta(hours=6),
        "FP2": now_utc - timedelta(hours=9),
        "FP1": now_utc - timedelta(hours=12),
    }.get(session_name)

    quali_session = MagicMock()
    quali_session.weather_data = pd.DataFrame({"TrackTemp": [36.0, 37.0, 38.0]})
    quali_session.session_status = pd.DataFrame({"Status": ["Finished"]})

    with (
        patch("src.data.track_data_loader.fastf1.get_event", return_value=event),
        patch("src.data.track_data_loader.fastf1.get_session", return_value=quali_session),
    ):
        profile = resolve_track_temperature_profile(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="dry",
            is_sprint=False,
        )

    assert profile["source"] == "session_weather_blend"
    assert profile["session_name"] == "Q"
    assert profile["session_temperature_source"] == "track_temp"
    assert profile["session_track_temperature_c"] == pytest.approx(37.0)
    assert profile["forecast_track_temperature_c"] == pytest.approx(36.0)
    assert profile["track_temperature_c"] == pytest.approx(36.8)
    assert profile["session_weight"] == pytest.approx(0.8)
    assert profile["forecast_weight"] == pytest.approx(0.2)


def test_resolve_track_temperature_profile_reports_air_temp_inferred_source():
    now_utc = datetime.now(UTC)
    event = MagicMock()
    event.get_session_date.side_effect = lambda session_name: {
        "R": now_utc + timedelta(hours=2),
        "Q": now_utc + timedelta(hours=1),
        "FP3": now_utc + timedelta(hours=1),
        "FP2": now_utc + timedelta(hours=1),
        "FP1": now_utc - timedelta(hours=3),
    }.get(session_name)

    fp1_session = MagicMock()
    fp1_session.weather_data = pd.DataFrame({"AirTemp": [21.0, 22.0, 23.0]})
    fp1_session.session_status = pd.DataFrame({"Status": ["Finished"]})

    def _cfg_get(key, default=None):
        if key == "baseline_predictor.race.track_temperature.air_to_track_offset_c":
            return 8.0
        return default

    with (
        patch("src.data.track_data_loader.fastf1.get_event", return_value=event),
        patch("src.data.track_data_loader.fastf1.get_session", return_value=fp1_session),
        patch("src.data.track_data_loader.config_loader.get", side_effect=_cfg_get),
    ):
        profile = resolve_track_temperature_profile(
            year=2026,
            race_name="Bahrain Grand Prix",
            weather="mixed",
            is_sprint=False,
        )

    assert profile["source"] == "session_weather_blend"
    assert profile["session_name"] == "FP1"
    assert profile["session_temperature_source"] == "air_temp_inferred"
    assert profile["session_air_temperature_c"] == pytest.approx(22.0)
    assert profile["session_track_temperature_c"] == pytest.approx(30.0)
    assert profile["forecast_track_temperature_c"] == pytest.approx(29.0)
    assert profile["track_temperature_c"] == pytest.approx(29.6)


def test_resolve_non_competitive_weather_features_uses_latest_completed_practice():
    now_utc = datetime.now(UTC)
    event = MagicMock()
    event.get_session_date.side_effect = lambda session_name: {
        "FP3": now_utc - timedelta(hours=2),
        "FP2": now_utc - timedelta(hours=6),
        "FP1": now_utc - timedelta(hours=10),
    }.get(session_name)

    fp3_session = MagicMock()
    fp3_session.weather_data = pd.DataFrame(
        {
            "TrackTemp": [33.0, 34.0, 35.0],
            "AirTemp": [23.0, 24.0, 25.0],
            "WindSpeed": [17.0, 18.0, 19.0],
            "Humidity": [50.0, 52.0, 54.0],
            "Rainfall": [0.0, 0.0, 0.0],
        }
    )
    fp3_session.session_status = pd.DataFrame({"Status": ["Finished"]})

    with (
        patch("src.data.track_data_loader.fastf1.get_event", return_value=event),
        patch(
            "src.data.track_data_loader.fastf1.get_session", return_value=fp3_session
        ) as get_session,
    ):
        features = resolve_non_competitive_weather_features(
            year=2026,
            race_name="Bahrain Grand Prix",
            is_sprint=False,
        )

    assert features["available"] is True
    assert features["source_session"] == "FP3"
    assert features["practice_weather_bucket"] == "dry"
    assert features["track_temperature_c"] == pytest.approx(34.0)
    assert features["air_temperature_c"] == pytest.approx(24.0)
    assert features["wind_speed_kph"] == pytest.approx(18.0)
    assert features["humidity_pct"] == pytest.approx(52.0)
    assert features["rainfall_signal"] == pytest.approx(0.0)
    get_session.assert_called_once_with(2026, "Bahrain Grand Prix", "FP3")


def test_resolve_non_competitive_weather_features_falls_back_when_event_unavailable():
    """Unavailable event metadata should return an unavailable weather-feature payload."""
    with patch(
        "src.data.track_data_loader.fastf1.get_event",
        side_effect=RuntimeError("network unavailable"),
    ):
        features = resolve_non_competitive_weather_features(
            year=2026,
            race_name="Bahrain Grand Prix",
            is_sprint=False,
        )

    assert features["available"] is False
    assert features["reason"] == "event_load_failed"


def test_resolve_non_competitive_weather_features_skips_fastf1_when_disabled():
    """Disabled practice weather should return unavailable features without lookup."""
    with (
        patch(
            "src.data.track_data_loader._cfg_get",
            side_effect=lambda key, default=None: (
                False
                if key == "baseline_predictor.race.weather_features.session_weather_enabled"
                else default
            ),
        ),
        patch("src.data.track_data_loader.fastf1.get_event") as get_event,
    ):
        features = resolve_non_competitive_weather_features(
            year=2026,
            race_name="Bahrain Grand Prix",
            is_sprint=False,
        )

    assert features["available"] is False
    assert features["reason"] == "session_weather_disabled"
    get_event.assert_not_called()
