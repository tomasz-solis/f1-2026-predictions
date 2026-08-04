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
