"""Tests for the debut-season driver seconds fallback builder."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.build_driver_seconds_rookie_fallback import build_driver_seconds_rookie_fallback


def _observation(
    *,
    year: int,
    session_kind: str,
    reference: str,
    comparison: str,
    gap_s: float,
) -> dict[str, object]:
    """Return one valid aggregate matched-lap observation."""
    return {
        "reference_driver_code": reference,
        "comparison_driver_code": comparison,
        "year": year,
        "session_kind": session_kind,
        "matched_gap_median_s": gap_s,
        "n_matched_pairs": 12,
        "weather_bucket": "dry",
        "skip_reason": None,
    }


def _prior_payload() -> dict[str, object]:
    """Return a small teammate-network prior with anchored teammate values."""
    drivers = {
        "V1": {"mu_s": 0.1, "sigma_s": 0.15},
        "V2": {"mu_s": 0.3, "sigma_s": 0.16},
        "R1": {"mu_s": -0.1, "sigma_s": 0.51},
        "R2": {"mu_s": 0.2, "sigma_s": 0.61},
    }
    return {
        "config": {
            "historical_scope": {"start": 2023, "end": 2025},
            "min_driver_observations": 24,
        },
        "race_network": {"drivers": drivers},
        "quali_network": {"drivers": drivers},
    }


def test_fallback_uses_median_of_debut_season_rookie_medians() -> None:
    """Future rows should not leak into the debut-season rookie fallback."""
    observations = pd.DataFrame(
        [
            # R1 implied mean: -0.2 + V1(0.1) = -0.1.
            _observation(
                year=2023,
                session_kind="race",
                reference="R1",
                comparison="V1",
                gap_s=-0.2,
            ),
            # R2 implied mean: V2(0.3) - 0.1 = 0.2.
            _observation(
                year=2024,
                session_kind="race",
                reference="V2",
                comparison="R2",
                gap_s=0.1,
            ),
            # R1's second season must not shape the rookie fallback.
            _observation(
                year=2024,
                session_kind="race",
                reference="R1",
                comparison="V1",
                gap_s=9.0,
            ),
            _observation(
                year=2023,
                session_kind="qualifying",
                reference="R1",
                comparison="V1",
                gap_s=-0.2,
            ),
            _observation(
                year=2024,
                session_kind="qualifying",
                reference="V2",
                comparison="R2",
                gap_s=0.1,
            ),
        ]
    )

    artifact = build_driver_seconds_rookie_fallback(
        observations=observations,
        teammate_prior=_prior_payload(),
        driver_debuts={"R1": 2023, "R2": 2024},
        built_at="2026-05-21T00:00:00+00:00",
    )

    assert artifact["race"]["mu_s"] == pytest.approx(0.05)
    assert artifact["race"]["sigma_s"] == pytest.approx(0.56)
    assert artifact["race"]["n_rookies"] == 2
    assert artifact["race"]["n_implied_observations"] == 2
    assert artifact["qualifying"]["mu_s"] == pytest.approx(0.05)
    assert artifact["promotion_policy"]["min_observations"] == 24
