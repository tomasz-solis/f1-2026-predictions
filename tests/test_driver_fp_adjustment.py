"""Tests for driver-level FP adjustment helpers."""

from unittest.mock import patch

import pandas as pd

from src.utils.driver_fp_adjustment import calculate_driver_fp_modifiers


def _build_laps(driver_a: str, driver_b: str) -> pd.DataFrame:
    """Create a minimal two-driver lap DataFrame for one team."""
    return pd.DataFrame(
        {
            "Team": ["TeamX", "TeamX"],
            "Driver": [driver_a, driver_b],
            "LapTime": pd.to_timedelta([90.0, 90.6], unit="s"),
        }
    )


def test_calculate_driver_fp_modifiers_uses_preloaded_laps_without_fetching() -> None:
    """Preloaded laps should bypass external FP session loading."""
    laps = _build_laps("DRV1", "DRV2")

    with patch("src.utils.driver_fp_adjustment.get_fp_team_performance") as mocked_fetch:
        modifiers = calculate_driver_fp_modifiers(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_types=["FP1"],
            preloaded_session_laps={"FP1": laps},
        )

    mocked_fetch.assert_not_called()
    assert modifiers["DRV1"] > 0.0
    assert modifiers["DRV2"] < 0.0


def test_calculate_driver_fp_modifiers_fetches_only_missing_sessions() -> None:
    """Only sessions absent from preloaded laps should trigger FP fetches."""
    fp1_laps = _build_laps("DRV1", "DRV2")
    fp2_laps = _build_laps("DRV1", "DRV2")

    with patch(
        "src.utils.driver_fp_adjustment.get_fp_team_performance",
        return_value=(None, fp2_laps, None),
    ) as mocked_fetch:
        modifiers = calculate_driver_fp_modifiers(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_types=["FP1", "FP2"],
            preloaded_session_laps={"FP1": fp1_laps},
        )

    mocked_fetch.assert_called_once_with(2026, "Bahrain Grand Prix", "FP2")
    assert modifiers
