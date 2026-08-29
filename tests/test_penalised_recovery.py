"""A penalised grid row records where the driver qualified, and starts him where he was put.

``qualifying_position`` survives grid validation so the page can show what the penalty
cost. It no longer feeds the finish-order blend: ``resolve_pace_anchor`` used it to
restore a penalised driver's qualifying position as his anchor, which erased the penalty
outright (2026 Italian GP: ANT P22 reported P3). The blend now anchors to the grid slot
the driver actually starts from. See docs/OVERTAKING_CALIBRATION_PLAN.md.
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


def _finish_order_for(is_penalised: bool) -> dict:
    """Run the finish-order blend over a full field with one back-of-grid quick car.

    Twenty-one drivers line up in grid order and the simulation keeps them there. The
    twenty-second starts P22 with a simulated median of P4 -- the shape of a penalised
    quick car, where the simulation has already carried him forward and the only question
    is whether the blend hands that answer through or overrides it.
    """
    from src.predictors.baseline.race.result_processing import build_finish_order

    class _Cfg:
        def get(self, _key, default=None):
            return default

    samples = 40
    field = {}
    grid_samples = {}
    medians = {}
    distributions = {}
    for position in range(1, 22):
        code = f"D{position:02d}"
        field[code] = {
            "team": f"Team{position}",
            "grid_pos": position,
            "skill": 0.5,
            "overtaking_skill": 0.5,
            "race_advantage": 0.0,
        }
        grid_samples[code] = [float(position)] * samples
        medians[code] = float(position)
        distributions[code] = [float(position)] * samples

    back = {
        "team": "TeamBack",
        "grid_pos": 22,
        "skill": 0.5,
        "overtaking_skill": 0.5,
        "race_advantage": 0.0,
    }
    if is_penalised:
        back["is_penalised"] = True
    field["BACK"] = back
    grid_samples["BACK"] = [22.0] * samples
    medians["BACK"] = 4.0
    distributions["BACK"] = [4.0] * samples

    order = build_finish_order(
        aggregated={
            "median_positions": medians,
            "position_distributions": distributions,
            "dnf_rates": dict.fromkeys(field, 0.0),
        },
        driver_info_map=field,
        grid_position_samples_by_driver=grid_samples,
        field_size=22,
        weather="dry",
        is_sprint=False,
        input_confidence=None,
        cfg=_Cfg(),
        race_params={},
        weather_feature_modifiers={},
        get_learned_position_adjustment=lambda **_: 0.0,
        learned_interval_radius=0.0,
        enforce_non_increasing=lambda values: values,
        base_seed=42,
    )
    return {row["driver"]: row for row in order}


def test_a_penalised_driver_keeps_the_recovery_the_simulation_gave_him():
    """The blend must not re-anchor a penalised driver to the slot he was moved to.

    He is started from the penalised slot already, so anchoring him there a second time
    charges the penalty twice -- the defect that reported ANT P17 from P22 when the
    simulation had him at P4. See docs/MODEL_LEDGER.md, 2026-08-29.
    """
    penalised = _finish_order_for(True)["BACK"]
    unpenalised = _finish_order_for(False)["BACK"]

    assert penalised["position"] < unpenalised["position"], (
        f"A penalised driver was reported P{penalised['position']} against P"
        f"{unpenalised['position']} for an unflagged driver with the identical simulated "
        "race. The blend is still anchoring him to his penalised grid slot."
    )


def test_an_unpenalised_driver_stays_anchored_to_his_grid_slot():
    """Only a flagged driver changes; the anchor and max_gain floor are otherwise intact.

    ``min_position_score`` is ``max(1, grid - max_gain)`` with max_gain capped at 11, so
    an ordinary driver starting P22 cannot be reported ahead of about P11 however quick
    the simulation says he was. This is the control for the test above: if it ever passes
    trivially, the anchor has stopped applying to everyone.
    """
    unpenalised = _finish_order_for(False)["BACK"]

    assert unpenalised["position"] >= 8, (
        f"An unflagged driver starting P22 with a P4 simulated median was reported "
        f"P{unpenalised['position']}. The grid anchor is no longer damping ordinary "
        "drivers, so the penalised-driver test above proves nothing."
    )
