"""Verify that in-season Bayesian form blending affects race predictions."""

from src.predictors.baseline_2026 import Baseline2026Predictor


def test_bayesian_blend_changes_race_skill_after_races():
    """Race skill should differ from base skill_score once races_completed > 0.

    Before this was fixed, the race path did not apply Bayesian blending at
    all. Now both paths share the same logic and the race should respond to
    in-season form evidence.
    """
    predictor = Baseline2026Predictor(seed=42)

    ver_data = predictor.drivers.get("VER")
    if ver_data is None:
        return

    bayesian_block = ver_data.get("bayesian", {})
    if not bayesian_block or bayesian_block.get("rating_mu") is None:
        return
    bayesian_block["rating_mu"] = 12.0

    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
    ]

    predictor.races_completed = 0
    info_0, _ = predictor._prepare_driver_info(grid, "Australian Grand Prix", "MEDIUM")
    skill_at_0 = info_0["VER"]["skill"]

    predictor.races_completed = 5
    info_5, _ = predictor._prepare_driver_info(grid, "Australian Grand Prix", "MEDIUM")
    skill_at_5 = info_5["VER"]["skill"]

    assert skill_at_0 != skill_at_5, (
        f"VER race skill at 0 races ({skill_at_0:.4f}) equals skill at 5 races "
        f"({skill_at_5:.4f}). Bayesian race blend is not active."
    )


def test_bayesian_blend_does_not_apply_at_zero_races():
    """At races_completed=0 the blend weight is 0 so race skill equals base skill."""
    predictor = Baseline2026Predictor(seed=42)
    predictor.races_completed = 0

    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
    ]
    info, _ = predictor._prepare_driver_info(grid, "Australian Grand Prix", "MEDIUM")

    ver_data = predictor.drivers.get("VER", {})
    base_skill = ver_data.get("racecraft", {}).get("skill_score", 0.5)
    race_skill = info["VER"]["skill"]

    assert abs(race_skill - base_skill) < 0.001, (
        f"VER race skill ({race_skill:.4f}) differs from base "
        f"({base_skill:.4f}) at races_completed=0."
    )
