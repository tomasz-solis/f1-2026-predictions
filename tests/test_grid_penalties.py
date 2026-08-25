"""Post-qualifying grid penalties applied to the race grid."""

import pytest

from src.predictors.baseline.race.grid_uncertainty import prepare_grid_uncertainty_profile
from src.utils.grid_penalties import apply_grid_penalties, save_penalties


class _Cfg:
    """Minimal config stub exposing one grid.penalties payload."""

    def __init__(self, penalties):
        self._penalties = penalties

    def get(self, key, default=None):
        assert key == "grid.penalties"
        return self._penalties


def _grid(*drivers):
    """Build a qualifying grid whose rows carry a realistic prediction spread."""
    return [
        {
            "driver": driver,
            "team": "Team",
            "position": position,
            "median_position": position,
            "p5": max(1, position - 2),
            "p95": position + 2,
            "confidence": 0.6,
        }
        for position, driver in enumerate(drivers, start=1)
    ]


def test_penalised_driver_drops_and_the_field_closes_up():
    cfg = _Cfg({"Italian Grand Prix": {"ANT": 3}})

    result, _applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR", "LEC", "RUS"), race_name="Italian Grand Prix", cfg=cfg
    )

    # Qualified P2, three-place drop: he starts P5 and the three cars behind move up.
    assert [row["driver"] for row in result] == ["VER", "NOR", "LEC", "RUS", "ANT"]
    assert [row["position"] for row in result] == [1, 2, 3, 4, 5]


def test_a_tied_slot_puts_the_penalised_car_behind_the_clean_one():
    """A drop bigger than the field starts him last; it does not push a clean driver back."""
    cfg = _Cfg({"Italian Grand Prix": {"VER": 10}})

    result, _applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR"), race_name="Italian Grand Prix", cfg=cfg
    )

    assert [row["driver"] for row in result] == ["ANT", "NOR", "VER"]


def test_the_penalised_row_loses_its_qualifying_spread():
    """The simulation samples from the spread, so a stale spread undoes the penalty."""
    cfg = _Cfg({"Italian Grand Prix": {"D02": 20}})

    result, _applied = apply_grid_penalties(
        _grid(*[f"D{i:02d}" for i in range(1, 23)]), race_name="Italian Grand Prix", cfg=cfg
    )
    penalised = next(row for row in result if row["driver"] == "D02")
    assert penalised["position"] == 22
    assert (penalised["p5"], penalised["p95"], penalised["median_position"]) == (22, 22, 22)

    profile = prepare_grid_uncertainty_profile(
        validated_grid=result, input_confidence=1.0, cfg=_ProfileCfg()
    )
    # Centred on the penalised slot, not on the qualifying one.
    assert profile["D02"]["center"] == 22.0


class _ProfileCfg:
    """Config stub returning the documented grid-uncertainty defaults."""

    def get(self, key, default=None):
        return default


def test_a_drop_past_the_back_starts_last_and_back_beats_a_place_drop():
    cfg = _Cfg({"Italian Grand Prix": {"ANT": 20, "NOR": "back"}})

    result, _applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR", "LEC"), race_name="Italian Grand Prix", cfg=cfg
    )

    assert [row["driver"] for row in result] == ["VER", "LEC", "ANT", "NOR"]


def test_a_pit_lane_start_is_marked_as_one():
    cfg = _Cfg({"Italian Grand Prix": {"ANT": "pit"}})

    result, _applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR"), race_name="Italian Grand Prix", cfg=cfg
    )

    assert result[-1]["driver"] == "ANT"
    assert result[-1]["start_type"] == "pit_lane"
    assert "start_type" not in result[0]


def test_other_races_and_an_empty_config_leave_the_grid_untouched():
    grid = _grid("VER", "ANT")

    assert (
        apply_grid_penalties(
            grid, race_name="Dutch Grand Prix", cfg=_Cfg({"Italian Grand Prix": {"ANT": 5}})
        ).grid
        is grid
    )
    assert apply_grid_penalties(grid, race_name="Italian Grand Prix", cfg=_Cfg({})).grid is grid


def test_a_penalty_for_a_driver_who_is_not_on_the_grid_fails_closed():
    cfg = _Cfg({"Italian Grand Prix": {"HAM": 5}})

    with pytest.raises(ValueError, match="not on the grid"):
        apply_grid_penalties(_grid("VER", "ANT"), race_name="Italian Grand Prix", cfg=cfg)


def test_a_malformed_penalty_fails_closed():
    cfg = _Cfg({"Italian Grand Prix": {"ANT": "twenty"}})

    with pytest.raises(ValueError, match="place count"):
        apply_grid_penalties(_grid("VER", "ANT"), race_name="Italian Grand Prix", cfg=cfg)


def test_the_applied_penalties_are_reported_for_display():
    """The page cannot label a penalty it was never told about."""
    cfg = _Cfg({"Italian Grand Prix": {"ANT": "pit"}})

    _grid_rows, applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR"), race_name="Italian Grand Prix", cfg=cfg
    )

    assert [penalty.to_dict() for penalty in applied] == [
        {"driver": "ANT", "qualified": 2, "starts": 3, "penalty": "pit"}
    ]


class _Store:
    """Artifact store stub recording one saved payload."""

    def __init__(self, stored=None):
        self.stored = stored
        self.saved = None

    def load_artifact(self, artifact_type, artifact_key, *_args, **_kwargs):
        assert artifact_type == "grid_penalties"
        return self.stored

    def save_artifact(self, **kwargs):
        self.saved = kwargs
        return {"version": 1}


def test_a_stored_penalty_wins_over_the_config_file():
    """The dashboard writes to the store, so a Saturday-night entry needs no deploy."""
    store = _Store({"penalties": {"ANT": 20}})

    _grid_rows, applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR"),
        race_name="Italian Grand Prix",
        year=2026,
        cfg=_Cfg({"Italian Grand Prix": {"NOR": 5}}),
        store=store,
    )

    assert [penalty.driver for penalty in applied] == ["ANT"]


def test_the_config_file_still_applies_when_nothing_is_stored():
    store = _Store(None)

    _grid_rows, applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR"),
        race_name="Italian Grand Prix",
        year=2026,
        cfg=_Cfg({"Italian Grand Prix": {"NOR": 5}}),
        store=store,
    )

    assert [penalty.driver for penalty in applied] == ["NOR"]


def test_a_store_outage_falls_back_to_the_config_rather_than_failing():
    class _BrokenStore:
        def load_artifact(self, *_args, **_kwargs):
            raise RuntimeError("supabase unavailable")

    _grid_rows, applied = apply_grid_penalties(
        _grid("VER", "ANT", "NOR"),
        race_name="Italian Grand Prix",
        year=2026,
        cfg=_Cfg({"Italian Grand Prix": {"ANT": 5}}),
        store=_BrokenStore(),
    )

    assert [penalty.driver for penalty in applied] == ["ANT"]


def test_saving_rejects_a_malformed_penalty_before_it_reaches_the_store():
    store = _Store()

    with pytest.raises(ValueError, match="place count"):
        save_penalties(race_name="Italian GP", year=2026, penalties={"ANT": "soon"}, store=store)

    assert store.saved is None


def test_two_simultaneous_penalties_match_the_real_hungarian_grid():
    """2026 Hungary: HAM (P2) and ANT (P4) both took three places.

    Sorting the field by "qualifying position + places dropped" puts HAM ahead of VER.
    The stewards instead remove the penalised cars, let the rest close up, and put each
    one back at the slot his drop earns — which is how HAM ends up behind VER.
    """
    qualifying = _grid("NOR", "HAM", "LEC", "ANT", "PIA", "VER", "RUS", "HAD")
    cfg = _Cfg({"Hungarian Grand Prix": {"HAM": 3, "ANT": 3}})

    result, applied = apply_grid_penalties(qualifying, race_name="Hungarian Grand Prix", cfg=cfg)

    assert [row["driver"] for row in result] == [
        "NOR",
        "LEC",
        "PIA",
        "VER",
        "HAM",
        "RUS",
        "ANT",
        "HAD",
    ]
    # Both keep their full three places here; an overlap can leave a smaller net drop.
    assert [(p.driver, p.qualified, p.starts) for p in applied] == [("HAM", 2, 5), ("ANT", 4, 7)]


def test_a_penalty_a_backmarker_cannot_serve_just_puts_him_last():
    """Ten places from P20 in a 22-car field is a two-place drop, and from P22 it is none."""
    field = _grid(*[f"D{i:02d}" for i in range(1, 23)])

    from_twenty, _ = apply_grid_penalties(field, race_name="R", cfg=_Cfg({"R": {"D20": 10}}))
    from_last, _ = apply_grid_penalties(field, race_name="R", cfg=_Cfg({"R": {"D22": 10}}))

    assert next(row["position"] for row in from_twenty if row["driver"] == "D20") == 22
    assert next(row["position"] for row in from_last if row["driver"] == "D22") == 22
    # The cars he cannot drop below simply move up one.
    assert next(row["position"] for row in from_twenty if row["driver"] == "D21") == 20


def test_two_drivers_sent_to_the_back_keep_qualifying_order_between_them():
    field = _grid(*[f"D{i:02d}" for i in range(1, 23)])

    result, _ = apply_grid_penalties(
        field, race_name="R", cfg=_Cfg({"R": {"D10": "back", "D21": "back"}})
    )

    assert [row["driver"] for row in result][-2:] == ["D10", "D21"]
