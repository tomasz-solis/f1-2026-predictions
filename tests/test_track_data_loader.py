"""Tests for track parameter helpers."""

import json
from unittest.mock import MagicMock, patch

from src.utils.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_race_distance_laps,
)


def setup_function():
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

    with patch("src.utils.track_data_loader.fastf1.get_session", return_value=mock_session):
        laps = resolve_race_distance_laps(2026, "Imaginary Grand Prix", is_sprint=False)

    assert laps == 63
    mock_session.load.assert_not_called()


def test_resolve_race_distance_falls_back_when_fastf1_fails():
    with patch("src.utils.track_data_loader.fastf1.get_session", side_effect=RuntimeError("boom")):
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
        "src.utils.track_data_loader.config_loader.get",
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
        "src.utils.track_data_loader.config_loader.get",
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
        "src.utils.track_data_loader.config_loader.get",
        side_effect=lambda key, default=None: (
            str(processed_root) if key == "paths.processed" else default
        ),
    ):
        params = load_track_specific_params("Monaco Grand Prix", year=2026)

    assert params["track_overtaking"] == 0.95


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

    with patch("src.utils.track_data_loader._resolve_pirelli_path", return_value=pirelli_path):
        stress = get_tire_stress_score("Bahrain Grand Prix", year=2027)

    assert stress == 3.5
