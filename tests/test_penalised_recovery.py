"""A penalised driver's recovery is anchored to his qualifying pace, not his grid slot.

Grid position normally proxies pace, which is why the finish-order blend anchors to it.
A steward's penalty breaks that assumption: the qualifying position is the pace evidence,
the grid slot is only where the simulation starts him. See docs/MODEL_LEDGER.md for the
2026 Italian GP probe this fixes (ANT P22 -> reported P18 despite a raw simulation median
of P5).
"""

from src.utils.grid_penalties import apply_grid_penalties
from src.utils.grid_validation import validate_qualifying_grid


class _Cfg:
    """Config stub exposing one grid.penalties payload, matching test_grid_penalties.py."""

    def __init__(self, penalties):
        self._penalties = penalties

    def get(self, key, default=None):
        assert key == "grid.penalties"
        return self._penalties


def _grid(*drivers):
    return [
        {"driver": driver, "team": "Team", "position": position}
        for position, driver in enumerate(drivers, start=1)
    ]


def test_a_penalised_row_carries_its_qualifying_position():
    cfg = _Cfg({"R": {"ANT": 3}})

    result, _applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR", "LEC"), race_name="R", cfg=cfg
    )

    penalised = next(row for row in result if row["driver"] == "ANT")
    assert penalised["qualifying_position"] == 2  # qualified P2, moved to P4


def test_an_unpenalised_row_carries_no_qualifying_position():
    cfg = _Cfg({"R": {"ANT": 3}})

    result, _applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR", "LEC"), race_name="R", cfg=cfg
    )

    clean = next(row for row in result if row["driver"] == "VER")
    assert "qualifying_position" not in clean


def test_validation_preserves_the_qualifying_position():
    grid = [
        {"driver": "ANT", "team": "Mercedes", "position": 22, "qualifying_position": 2},
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
    ]

    validated = validate_qualifying_grid(grid, min_entries=1)

    assert validated[0]["qualifying_position"] == 2
    assert "qualifying_position" not in validated[1]


def test_the_anchor_resolves_to_qualifying_position_for_a_penalised_driver():
    from src.predictors.baseline.race.result_processing import resolve_pace_anchor

    pace_anchor_pos, pace_anchor_gap = resolve_pace_anchor(
        reference_grid_pos=22.0, qualifying_pos=2
    )

    assert pace_anchor_pos == 2.0
    assert pace_anchor_gap == 20.0


def test_the_anchor_resolves_to_grid_position_when_there_is_no_qualifying_position():
    from src.predictors.baseline.race.result_processing import resolve_pace_anchor

    pace_anchor_pos, pace_anchor_gap = resolve_pace_anchor(
        reference_grid_pos=20.0, qualifying_pos=None
    )

    assert pace_anchor_pos == 20.0
    assert pace_anchor_gap == 0.0
