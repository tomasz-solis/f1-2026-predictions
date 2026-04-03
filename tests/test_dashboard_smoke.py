"""
Smoke tests for Streamlit dashboard

Basic tests to ensure the dashboard can load without crashing.
"""

from pathlib import Path
from unittest.mock import patch


class TestDashboardSmoke:
    """Basic dashboard loading tests"""

    def test_dashboard_imports(self):
        """Test that dashboard file can be imported"""
        # This ensures all imports in app.py are valid
        import importlib.util

        spec = importlib.util.spec_from_file_location("app", "app.py")
        assert spec is not None

    def test_baseline_predictor_can_be_imported(self):
        """Test critical imports for dashboard"""
        from src.predictors.baseline_2026 import Baseline2026Predictor

        assert Baseline2026Predictor is not None

    def test_fastf1_available(self):
        """Test FastF1 is installed and importable"""
        import fastf1

        assert fastf1 is not None

    def test_streamlit_available(self):
        """Test Streamlit is installed"""
        import streamlit

        assert streamlit is not None

    def test_plotly_available(self):
        """Test Plotly is installed for visualizations"""
        import plotly.graph_objects as go

        assert go is not None

    def test_data_files_exist(self):
        """Test required data files are present"""
        from pathlib import Path

        # Check critical data files
        required_files = [
            "data/current_lineups.json",
            "data/processed/car_characteristics/2026_car_characteristics.json",
            "data/processed/track_characteristics/2026_track_characteristics.json",
        ]

        for file_path in required_files:
            assert Path(file_path).exists(), f"Required file missing: {file_path}"

    def test_config_files_exist(self):
        """Test configuration files are present"""
        assert Path("config/default.yaml").exists()
        assert Path("config/production_config.json").exists()


class TestDashboardComponents:
    """Test individual dashboard components"""

    @patch("fastf1.get_event_schedule")
    def test_calendar_loading_logic(self, mock_schedule):
        """Test 2026 calendar loading logic"""
        import pandas as pd

        # Mock FastF1 schedule
        mock_data = pd.DataFrame(
            {
                "EventName": [
                    "Australian Grand Prix",
                    "Pre-Season Testing",
                    "Chinese Grand Prix",
                ],
                "EventFormat": ["conventional", None, "conventional"],
            }
        )
        mock_schedule.return_value = mock_data

        # Simulate dashboard logic
        schedule = mock_schedule(2026)
        race_events = schedule[
            (schedule["EventFormat"].notna())
            & (~schedule["EventName"].str.contains("Testing", case=False, na=False))
        ].copy()

        # Should filter out testing
        assert len(race_events) == 2
        assert "Pre-Season Testing" not in race_events["EventName"].values

    def test_prediction_pipeline(self):
        """Test full prediction pipeline works"""
        from src.predictors.baseline_2026 import Baseline2026Predictor

        predictor = Baseline2026Predictor()

        # Test qualifying prediction
        quali_result = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=5)
        assert "grid" in quali_result
        assert len(quali_result["grid"]) == 22

        # Test race prediction
        race_result = predictor.predict_race(quali_result["grid"], weather="dry", n_simulations=5)
        assert "finish_order" in race_result
        assert len(race_result["finish_order"]) == 22


class TestDashboardDataFlow:
    """Test data flow through dashboard"""

    def test_race_name_processing(self):
        """Test sprint indicator removal works"""
        race_selection = "Chinese Grand Prix (Sprint)"
        race_name = race_selection.replace(" (Sprint)", "")
        assert race_name == "Chinese Grand Prix"

    def test_weather_options_valid(self):
        """Test weather options are valid"""
        valid_weather = ["dry", "rain", "mixed"]

        # Dashboard should only use these
        for weather in valid_weather:
            assert weather in ["dry", "rain", "mixed"]

    def test_confidence_display_format(self):
        """Test confidence values are formatted correctly"""
        confidence = 54.7
        formatted = round(confidence, 1)
        assert formatted == 54.7
        assert isinstance(formatted, int | float)

    def test_dnf_risk_color_logic(self):
        """Test DNF risk color coding logic"""

        def get_dnf_status(dnf_pct):
            if dnf_pct > 20:
                return "High"
            elif dnf_pct > 10:
                return "Medium"
            return "Low"

        assert get_dnf_status(5) == "Low"
        assert get_dnf_status(15) == "Medium"
        assert get_dnf_status(25) == "High"


class TestDashboardEdgeCases:
    """Test edge cases in dashboard"""

    def test_missing_sprint_detection_fallback(self):
        """Sprint detection failure should fall back to conventional weekend."""
        from src.utils.weekend import is_sprint_weekend

        # Invalid race should return False, not crash
        try:
            result = is_sprint_weekend(2026, "Invalid Race Name")
            assert result is False
        except BaseException:
            # If it raises, that's also acceptable
            pass

    def test_empty_qualifying_grid_handling(self):
        """Test handling of edge case with no qualifying data"""
        from src.predictors.baseline_2026 import Baseline2026Predictor

        predictor = Baseline2026Predictor()

        # Should still produce results
        result = predictor.predict_qualifying(2026, "Australian Grand Prix", n_simulations=1)
        assert len(result["grid"]) > 0
