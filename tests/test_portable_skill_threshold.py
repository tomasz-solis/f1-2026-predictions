"""Verify the portable skill gate uses raw extraction skill and is reachable."""

from src.predictors.baseline_2026 import Baseline2026Predictor
from src.utils.config_loader import Config

ACTIVE_2026 = {
    "HAM",
    "LEC",
    "ANT",
    "RUS",
    "NOR",
    "PIA",
    "VER",
    "HAD",
    "GAS",
    "COL",
    "STR",
    "ALO",
    "BEA",
    "OCO",
    "LAW",
    "LIN",
    "SAI",
    "ALB",
    "BOT",
    "PER",
    "BOR",
    "HUL",
}


def test_raw_skill_stored_in_driver_info():
    """Race driver info should carry the raw extraction skill_score."""
    predictor = Baseline2026Predictor(seed=42)
    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
    ]
    driver_info, _ = predictor._prepare_driver_info(grid, "Australian Grand Prix", "MEDIUM")
    ver_info = driver_info["VER"]
    assert "raw_skill" in ver_info, "raw_skill field missing from driver info"

    ver_data = predictor.drivers.get("VER", {})
    expected = ver_data.get("racecraft", {}).get("skill_score", 0.5)
    assert abs(ver_info["raw_skill"] - expected) < 0.001, (
        f"raw_skill ({ver_info['raw_skill']:.4f}) should match "
        f"extraction skill_score ({expected:.4f})"
    )


def test_portable_skill_not_double_blended():
    """Portable skill should be a single Bayesian blend from raw skill, not double.

    Before this was fixed, _build_portable_skill_signal received the already-blended
    info["skill"] and blended it toward Bayesian again. Now it reads raw skill_score
    directly from driver_data, so the blend is applied exactly once.
    """
    predictor = Baseline2026Predictor(seed=42)
    predictor.races_completed = 3

    ver_data = predictor.drivers.get("VER", {})
    if not ver_data:
        return
    raw_skill = ver_data.get("racecraft", {}).get("skill_score", 0.5)

    # Passing an absurd base_skill should not affect the result - the function
    # must read from driver_data["racecraft"]["skill_score"] for eligible drivers.
    portable = predictor._build_portable_skill_signal("VER", 999.0)

    assert portable < 1.0, (
        f"portable_skill={portable}, function is using the passed-in base_skill "
        "instead of raw skill_score from driver_data"
    )
    assert portable <= raw_skill, (
        f"portable ({portable:.4f}) exceeds raw skill ({raw_skill:.4f}) - "
        "the Bayesian blend should pull toward the mean, not above"
    )


def test_threshold_gate_uses_raw_skill():
    """At least VER and LEC should have raw skill >= 0.70 threshold."""
    predictor = Baseline2026Predictor(seed=42)

    cfg = Config()
    threshold = float(
        cfg.get(
            "baseline_predictor.race.final_blend.hypothetical_points_floor.portable_skill_threshold",
            0.70,
        )
    )

    above = []
    for code in ACTIVE_2026:
        driver_data = predictor.drivers.get(code, {})
        raw = driver_data.get("racecraft", {}).get("skill_score", 0.0)
        if raw >= threshold:
            above.append(code)

    assert len(above) >= 2, (
        f"Only {len(above)} drivers have raw skill >= {threshold}: {above}. "
        "The hypothetical floor gate is unreachable."
    )
    expected_elite = {"VER", "LEC"}
    missing = expected_elite - set(above)
    assert not missing, (
        f"{missing} should pass the raw skill threshold of {threshold} "
        f"but did not. Raw skills may need review."
    )
