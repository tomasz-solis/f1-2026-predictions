"""Integration checks for weight-schedule wiring in baseline predictions."""

from src.predictors.baseline_2026 import Baseline2026Predictor
from src.systems.weight_schedule import get_schedule_weights


def test_weight_schedule_integration():
    """Weight schedule and predictor stack should execute end-to-end."""

    # 1. Test weight schedule module
    weights = get_schedule_weights(race_number=1, schedule="extreme")
    assert weights["baseline"] == 0.30
    assert weights["testing"] == 0.20
    assert weights["current"] == 0.50

    # 2. Test predictor initialization
    predictor = Baseline2026Predictor()
    assert len(predictor.teams) > 0
    assert len(predictor.tracks) > 0

    # 3. Test track suitability calculation
    suitability = predictor.calculate_track_suitability("McLaren", "Bahrain Grand Prix")
    assert isinstance(suitability, float)

    # 4. Test blended team strength
    for team in list(predictor.teams.keys())[:3]:  # Test first 3 teams
        baseline = predictor.teams[team].get("overall_performance", 0.5)
        blended = predictor.get_blended_team_strength(team, "Bahrain Grand Prix")
        assert isinstance(baseline, float)
        assert isinstance(blended, float)

    # 5. Test qualifying prediction (full integration test)
    result = predictor.predict_qualifying(
        year=2026,
        race_name="Bahrain Grand Prix",
        n_simulations=10,  # Fast test
    )
    assert len(result["grid"]) >= 20
    assert all("driver" in entry and "team" in entry for entry in result["grid"][:3])
