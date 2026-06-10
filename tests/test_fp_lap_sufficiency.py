"""Tests for the per-driver FP lap-sufficiency gate."""

import pandas as pd

from src.utils.fp_blending import _extract_representative_lap_time
from src.utils.fp_blending_flow import extract_team_performance_from_laps
from src.utils.team_mapping import map_team_to_characteristics


def _laps(rows):
    """rows: list of (driver, team, n_laps, base_seconds)."""
    data = []
    for driver, team, n_laps, base_seconds in rows:
        for i in range(n_laps):
            data.append(
                {
                    "Driver": driver,
                    "Team": team,
                    "LapTime": pd.Timedelta(seconds=base_seconds + 0.01 * i),
                    "Compound": "SOFT",
                    "TyreLife": 1 + (i % 5),
                }
            )
    return pd.DataFrame(data)


def _extract(laps, gate):
    return extract_team_performance_from_laps(
        laps=laps,
        run_focus="short",
        min_long_run_laps=12,
        preferred_short_run_compound="SOFT",
        long_run_outlier_threshold=1.5,
        long_run_trim_ends=True,
        extract_representative_lap_time_fn=_extract_representative_lap_time,
        map_team_to_characteristics_fn=map_team_to_characteristics,
        normalization="robust",
        spread_k=2.0,
        min_driver_laps=gate,
    )


def test_thin_driver_excluded_so_teammate_carries_team():
    """A 3-lap DNF run is dropped so the team reflects the teammate's real pace."""
    laps = _laps(
        [
            ("NOR", "McLaren", 3, 95.0),  # stopped early - slow, unrepresentative
            ("PIA", "McLaren", 12, 90.0),
            ("VER", "Red Bull Racing", 12, 90.5),
            ("PER", "Red Bull Racing", 12, 90.6),
            ("LEC", "Ferrari", 12, 90.2),
            ("HAM", "Ferrari", 12, 90.3),
        ]
    )
    with_gate = _extract(laps, 4)
    no_gate = _extract(laps, 0)
    # Without the gate NOR's slow run drags McLaren's median; with it, PIA carries the team.
    assert with_gate["McLaren"] > no_gate["McLaren"]
    assert with_gate["McLaren"] >= max(with_gate.values()) - 1e-9  # McLaren back on top


def test_fully_thin_team_is_omitted():
    """If both cars are too thin, the team is dropped (model-only fallback downstream)."""
    laps = _laps(
        [
            ("NOR", "McLaren", 2, 95.0),
            ("PIA", "McLaren", 3, 95.0),
            ("VER", "Red Bull Racing", 12, 90.5),
            ("PER", "Red Bull Racing", 12, 90.6),
            ("LEC", "Ferrari", 12, 90.2),
            ("HAM", "Ferrari", 12, 90.3),
        ]
    )
    result = _extract(laps, 4)
    assert "McLaren" not in result
    assert "Red Bull Racing" in result and "Ferrari" in result


def test_gate_disabled_keeps_all_drivers():
    laps = _laps([("NOR", "McLaren", 1, 95.0), ("PIA", "McLaren", 12, 90.0)])
    result = _extract(laps, 0)
    assert "McLaren" in result
