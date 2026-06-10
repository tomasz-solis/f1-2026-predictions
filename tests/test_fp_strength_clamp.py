"""Tests for the live-FP single-session strength-move clamp."""

from src.predictors.baseline.qualifying_preparation import _blend_strengths, _clamp_strength_move
from src.utils.fp_blending import blend_team_strength


def test_clamp_bounds_move_from_model():
    model = {"McLaren": 0.84, "Sauber": 0.30}
    blended = {"McLaren": 0.25, "Sauber": 0.55}  # McLaren collapsed, Sauber inflated
    clamped = _clamp_strength_move(blended, model, max_move=0.25)
    assert clamped["McLaren"] == 0.84 - 0.25  # floored at model - limit
    assert clamped["Sauber"] == 0.30 + 0.25  # capped at model + limit


def test_clamp_noop_when_within_limit():
    model = {"A": 0.6, "B": 0.5}
    blended = {"A": 0.55, "B": 0.52}
    assert _clamp_strength_move(blended, model, max_move=0.25) == blended


def test_clamp_disabled_with_none():
    model = {"A": 0.6}
    blended = {"A": 0.1}
    assert _clamp_strength_move(blended, model, max_move=None) == blended


def test_fp_branch_applies_clamp():
    """A compromised FP session cannot move a strong team past the clamp."""
    model = {
        "McLaren": 0.84,
        "Mercedes": 0.80,
        "Ferrari": 0.74,
        "Red Bull": 0.70,
        "Williams": 0.45,
        "Sauber": 0.30,
    }
    # McLaren session-slowest (FP wants to bury them); others ordered normally.
    fp = {
        "McLaren": 0.00,
        "Mercedes": 1.00,
        "Ferrari": 0.60,
        "Red Bull": 0.55,
        "Williams": 0.25,
        "Sauber": 0.40,
    }
    blended = _blend_strengths(
        model_strengths=model,
        fp_performance=fp,
        testing_fallback_performance=None,
        uses_checkpoint_practice_profiles=False,
        checkpoint_practice_blend_weight=None,
        checkpoint_testing_fallback_performance=None,
        fp_blend_weight=0.80,
        practice_like_profile_label=None,
        practice_like_blend_weight=None,
        blend_team_strength_fn=blend_team_strength,
        apply_testing_fallback_adjustment_fn=lambda **_k: model,
        fp_max_strength_move=0.25,
    )
    # Clamp guarantees McLaren cannot fall more than 0.25 below its prior.
    assert blended["McLaren"] >= model["McLaren"] - 0.25 - 1e-9
    assert blended["McLaren"] > blended["Williams"]
    assert blended["McLaren"] > blended["Sauber"]
