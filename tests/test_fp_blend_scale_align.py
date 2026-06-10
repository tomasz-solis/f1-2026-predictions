"""Tests for robust FP normalization and scale-aligned team-strength blending.

These cover the regression where a single compromised practice session could drag
a strong team toward the 0.0 min-max anchor and, blended on a mismatched scale,
collapse it down the grid.
"""

import numpy as np

from src.utils.fp_blending import _scale_align_fp_to_model, blend_team_strength
from src.utils.fp_blending_flow import normalize_team_pace, robust_spread


def test_robust_spread_resists_single_outlier():
    """One extreme value must not blow up the spread the way (max-min) does."""
    base = [10.0, 10.1, 10.2, 9.9, 10.05]
    with_outlier = [*base, 25.0]
    assert robust_spread(np.array(with_outlier)) < 1.0
    # Range-based scaling would report ~15 here; robust stays tight.


def test_robust_normalization_does_not_force_slowest_to_zero():
    """A team only marginally off pace should sit near 0.5, not at the 0.0 anchor."""
    team_medians = {"A": 90.0, "B": 90.1, "C": 90.2, "D": 90.3, "E": 95.0}
    minmax = normalize_team_pace(team_medians, normalization="minmax")
    robust = normalize_team_pace(team_medians, normalization="robust")

    # Under min-max the slow outlier (E) defines 0.0 and crushes B/C/D toward 1.0.
    assert minmax["D"] == 0.0 or minmax["E"] == 0.0
    # Under robust, the marginally-slower D stays well above the floor.
    assert robust["D"] > 0.3
    # Fastest team is rewarded but not pinned to a hard 1.0 anchor.
    assert 0.5 < robust["A"] <= 1.0


def test_robust_normalization_outlier_does_not_redefine_scale():
    """Adding a sandbagging slow team should barely move the other teams' scores."""
    tight = {"A": 90.0, "B": 90.1, "C": 90.2, "D": 90.3}
    robust_tight = normalize_team_pace(tight, normalization="robust")
    robust_with_sandbag = normalize_team_pace({**tight, "E": 96.0}, normalization="robust")
    for team in ("A", "B", "C", "D"):
        assert abs(robust_tight[team] - robust_with_sandbag[team]) < 0.2

    minmax_tight = normalize_team_pace(tight, normalization="minmax")
    minmax_with_sandbag = normalize_team_pace({**tight, "E": 96.0}, normalization="minmax")
    # Min-max collapses A-D together once the outlier stretches the denominator.
    assert abs(minmax_tight["D"] - minmax_with_sandbag["D"]) > 0.5


def test_scale_align_keeps_fp_within_model_band():
    """A full-range FP signal is compressed onto the model's calibrated band."""
    model = {
        "McLaren": 0.84,
        "Mercedes": 0.80,
        "Ferrari": 0.74,
        "Red Bull": 0.70,
        "Williams": 0.45,
        "Alpine": 0.37,
        "Sauber": 0.30,
    }
    # FP spans the whole 0-1 range (min-max style) and disagrees on ordering.
    fp = {
        "McLaren": 0.00,  # compromised session -> session-slowest
        "Mercedes": 1.00,
        "Ferrari": 0.55,
        "Red Bull": 0.62,
        "Williams": 0.20,
        "Alpine": 0.90,
        "Sauber": 0.40,
    }
    aligned = _scale_align_fp_to_model(model, fp, missing_from_fp=set())
    model_lo, model_hi = min(model.values()), max(model.values())
    # Aligned FP must sit within (a small margin of) the model's own band.
    for team, value in aligned.items():
        assert model_lo - 0.1 <= value <= model_hi + 0.1, (team, value)


def test_compromised_top_team_does_not_collapse_after_blend():
    """McLaren slowest in one session should not crater below the field after blending."""
    model = {
        "McLaren": 0.84,
        "Mercedes": 0.80,
        "Ferrari": 0.74,
        "Red Bull": 0.70,
        "Williams": 0.45,
        "Alpine": 0.37,
        "Sauber": 0.30,
    }
    fp = {
        "McLaren": 0.00,
        "Mercedes": 1.00,
        "Ferrari": 0.55,
        "Red Bull": 0.62,
        "Williams": 0.20,
        "Alpine": 0.90,
        "Sauber": 0.40,
    }
    blended = blend_team_strength(model, fp, blend_weight=0.70)

    # Legacy behaviour would have given McLaren 0.7*0.0 + 0.3*0.84 = 0.252 (last).
    # With scale-alignment, McLaren stays clearly above the genuine backmarkers.
    assert blended["McLaren"] > blended["Sauber"]
    assert blended["McLaren"] > blended["Williams"]
    assert blended["McLaren"] > 0.45


def test_blend_missing_team_uses_model_only():
    """Teams absent from FP keep their model strength."""
    model = {"A": 0.8, "B": 0.6, "C": 0.5, "D": 0.4}
    fp = {"A": 0.9, "B": 0.5, "C": 0.4}  # D missing
    blended = blend_team_strength(model, fp, blend_weight=0.7)
    assert blended["D"] == 0.4


def test_scale_align_disabled_returns_raw_fp(monkeypatch):
    """With alignment disabled, the raw FP values are used unchanged."""
    import src.utils.fp_blending as fb

    monkeypatch.setattr(
        fb,
        "get_config_value",
        lambda key, default=None: False if key.endswith("fp_scale_align") else default,
    )
    model = {"A": 0.8, "B": 0.6, "C": 0.5}
    fp = {"A": 0.9, "B": 0.5, "C": 0.4}
    aligned = fb._scale_align_fp_to_model(model, fp, missing_from_fp=set())
    assert aligned == fp
