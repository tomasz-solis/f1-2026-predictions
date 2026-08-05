"""Freshness check for the committed team-strength seconds mapping.

The mapping converts a team-strength rank into a time gap, which is a property of
the current field. It goes stale the obvious way: calibration rows gain rounds,
nobody refreezes, and the committed slope stops describing the data it claims to
be fitted on.

This reads the *current* calibration rows rather than anything recorded in the
artifact, because a stale artifact reports the healthy numbers it was frozen with.

The comparison is against per-round scatter rather than a fixed tolerance: the
slope moves round to round anyway, and only a move beating that noise means
anything.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pandas as pd
import pytest

from src.models.team_strength_mapping import fit_linear_team_strength_mapping

_MAPPING_DIR = Path("data/processed/team_strength_seconds_mapping")
_MAPPING_FILE = _MAPPING_DIR / "latest.json"
_OBSERVATIONS_FILE = _MAPPING_DIR / "calibration_observations.csv"

# A refit is called for past two standard errors. Fail at three, so ordinary
# round-to-round movement does not break the build but a real shift does.
_STALENESS_STANDARD_ERRORS = 3.0


def _load() -> tuple[dict, pd.DataFrame]:
    """Load the committed mapping and its calibration rows, or skip."""
    if not _MAPPING_FILE.exists() or not _OBSERVATIONS_FILE.exists():
        pytest.skip("Team-strength mapping artifacts not present")
    with open(_MAPPING_FILE, encoding="utf-8") as handle:
        mapping = json.load(handle)
    return mapping, pd.read_csv(_OBSERVATIONS_FILE)


@pytest.mark.parametrize("session_kind", ["race", "qualifying"])
def test_shipped_slope_still_matches_its_calibration_rows(session_kind: str) -> None:
    """Refitting on the current rows must reproduce the frozen slope.

    This is the guard against a silently stale artifact: when the season adds
    rounds, the fit moves and the committed mapping no longer describes the data
    it claims to be fitted on.
    """
    mapping, observations = _load()
    policy = str(mapping["policy"])
    training_years = tuple(int(year) for year in mapping["training_years"])

    shipped = float(mapping["mappings"][session_kind]["slope_s_per_unit"])
    refit = fit_linear_team_strength_mapping(
        observations,
        session_kind=session_kind,
        policy=policy,
        training_years=training_years,
    ).slope_s_per_unit

    # Judge the gap against how much the slope moves round to round anyway. Fitting
    # each round on its own gives that scatter; a round with too few rows to fit is
    # skipped rather than contributing a wild estimate.
    rows = observations[
        observations["session_kind"].eq(session_kind) & observations["year"].isin(training_years)
    ]
    per_round: list[float] = []
    for _, one_round in rows.groupby(["year", "race_name"]):
        if len(one_round) < 12:
            continue
        try:
            per_round.append(
                fit_linear_team_strength_mapping(
                    one_round,
                    session_kind=session_kind,
                    policy=policy,
                    training_years=training_years,
                ).slope_s_per_unit
            )
        except ValueError:
            continue

    round_sd = statistics.stdev(per_round) if len(per_round) >= 4 else None
    if not round_sd:
        # No scatter estimate means no band to judge against. A refit on unchanged
        # rows should reproduce exactly, so fall back to that.
        assert math.isclose(shipped, refit, rel_tol=1e-6), (
            f"{session_kind}: shipped slope {shipped:.5f} != refit {refit:.5f}"
        )
        return

    standard_error = round_sd / math.sqrt(len(per_round))

    assert abs(shipped - refit) <= _STALENESS_STANDARD_ERRORS * standard_error, (
        f"{session_kind}: committed mapping is stale. Shipped slope {shipped:.5f}, "
        f"refit on current calibration rows {refit:.5f}, "
        f"{abs(shipped - refit) / standard_error:.1f} standard errors apart. "
        "Re-run scripts/freeze_team_strength_seconds_mapping.py."
    )


# --- the mapping must also agree with what consumes it -----------------------
#
# The freshness checks above guard the artifact against its own calibration rows.
# They cannot see the other way this mapping goes wrong: a consumer that hardcodes
# a slope-derived constant and then does not follow a refit.
#
# That is not hypothetical. `team_strength_seconds_score_scale` was pinned to
# 1.9707717329051126 -- the frozen *race* slope -- while being used on the
# *qualifying* path, and it stayed pinned through the 2026 refit until 7 of 22
# drivers saturated the [0, 1] clip and their learned driver ratings were erased.


class _StubConfig:
    """Minimal config stand-in exposing the dotted ``get`` the simulator uses."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self._values = values or {}

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)


def test_qualifying_score_scale_tracks_the_live_qualifying_slope() -> None:
    """The seconds-to-score divisor must be the qualifying slope, not a frozen copy."""
    from src.models.team_strength_mapping import load_live_team_strength_mappings
    from src.predictors.baseline.qualifying_simulation import _resolve_seconds_score_scale

    mappings = load_live_team_strength_mappings()
    qualifying = mappings.get("qualifying")
    if qualifying is None:
        pytest.skip("Live qualifying mapping not present")

    resolved = _resolve_seconds_score_scale(_StubConfig())

    assert math.isclose(resolved, qualifying.slope_s_per_unit, rel_tol=1e-9), (
        f"qualifying score scale {resolved:.9f} does not track the live qualifying "
        f"slope {qualifying.slope_s_per_unit:.9f}. Do not hardcode this value -- "
        "delta = slope * (team_strength - 0.5), so only the qualifying slope inverts "
        "the conversion and keeps the projected signal inside [0, 1]."
    )

    race = mappings.get("race")
    if race is not None and not math.isclose(
        race.slope_s_per_unit, qualifying.slope_s_per_unit, rel_tol=1e-9
    ):
        assert not math.isclose(resolved, race.slope_s_per_unit, rel_tol=1e-9), (
            "qualifying score scale is using the RACE slope. That was the original "
            "defect: a different session's calibration on the qualifying path."
        )


def test_qualifying_team_signal_never_saturates_across_the_strength_range() -> None:
    """Team strength alone must not pin the projected signal to the clip bounds."""
    from src.models.team_strength_mapping import load_live_team_strength_mappings
    from src.predictors.baseline.qualifying_simulation import _resolve_seconds_score_scale

    qualifying = load_live_team_strength_mappings().get("qualifying")
    if qualifying is None:
        pytest.skip("Live qualifying mapping not present")

    scale = _resolve_seconds_score_scale(_StubConfig())
    # 0.02 and 0.98 stand in for the strongest and weakest car the field can show.
    for team_strength in (0.02, 0.25, 0.5, 0.75, 0.98):
        raw = 0.5 + (qualifying.predict_delta_one(team_strength) / scale)
        assert 0.0 < raw < 1.0, (
            f"team_strength {team_strength} projects to {raw:.4f}, which the clip "
            "pins to a bound. Saturated drivers lose their learned quali_rating_mu_s "
            "entirely and teammates become indistinguishable."
        )


def test_explicit_config_value_still_overrides_the_derived_scale() -> None:
    """An A/B arm must still be able to vary the scale without a code change."""
    from src.predictors.baseline.qualifying_simulation import _resolve_seconds_score_scale

    cfg = _StubConfig({"baseline_predictor.qualifying.team_strength_seconds_score_scale": 2.6060})
    assert math.isclose(_resolve_seconds_score_scale(cfg), 2.6060, rel_tol=1e-9)
