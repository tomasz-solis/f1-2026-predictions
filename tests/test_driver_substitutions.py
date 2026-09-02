"""Race-scoped driver substitutions applied to the configured lineups."""

import pytest

from src.utils.driver_substitutions import apply_substitutions, save_substitutions


class _Cfg:
    """Minimal config stub exposing one grid.substitutions payload."""

    def __init__(self, substitutions):
        self._substitutions = substitutions

    def get(self, key, default=None):
        assert key == "grid.substitutions"
        return self._substitutions


class _Store:
    """Artifact store stub that records what was saved and serves what was seeded."""

    def __init__(self, stored=None):
        self.stored = stored
        self.saved = None

    def load_artifact(self, artifact_type, artifact_key):
        return self.stored

    def save_artifact(self, *, artifact_type, artifact_key, data):
        self.saved = data
        return data


def _lineups():
    """Two teams from the 2026 grid, enough to exercise a substitution chain."""
    return {"Red Bull Racing": ["VER", "HAD"], "RB": ["LAW", "LIN"]}


def test_a_chain_moves_both_drivers_in_one_pass():
    """HAD is injured, LAW takes the Red Bull, TSU takes the seat LAW left."""
    cfg = _Cfg({"Dutch Grand Prix": {"HAD": "LAW", "LAW": "TSU"}})

    result = apply_substitutions(_lineups(), race_name="Dutch Grand Prix", cfg=cfg)

    assert result == {"Red Bull Racing": ["VER", "LAW"], "RB": ["TSU", "LIN"]}


def test_another_races_substitutions_do_not_leak():
    cfg = _Cfg({"Dutch Grand Prix": {"HAD": "LAW", "LAW": "TSU"}})

    assert apply_substitutions(_lineups(), race_name="Italian Grand Prix", cfg=cfg) == _lineups()


def test_a_stale_entry_is_skipped_rather_than_raised():
    """A driver who has left the grid must not take a whole weekend's predictions down."""
    cfg = _Cfg({"Dutch Grand Prix": {"BOT": "TSU"}})

    assert apply_substitutions(_lineups(), race_name="Dutch Grand Prix", cfg=cfg) == _lineups()


def test_a_swap_that_would_seat_one_driver_twice_is_skipped():
    """Substituting HAD for VER's seatmate would put VER in the car alongside himself."""
    cfg = _Cfg({"Dutch Grand Prix": {"HAD": "VER"}})

    assert apply_substitutions(_lineups(), race_name="Dutch Grand Prix", cfg=cfg) == _lineups()


def test_the_stored_artifact_wins_over_the_config_file():
    cfg = _Cfg({"Dutch Grand Prix": {"HAD": "LIN"}})
    store = _Store({"substitutions": {"HAD": "LAW", "LAW": "TSU"}})

    result = apply_substitutions(
        _lineups(), race_name="Dutch Grand Prix", year=2026, cfg=cfg, store=store
    )

    assert result["Red Bull Racing"] == ["VER", "LAW"]


def test_a_store_outage_falls_back_to_the_config():
    class _BrokenStore:
        def load_artifact(self, artifact_type, artifact_key):
            raise RuntimeError("store unavailable")

    cfg = _Cfg({"Dutch Grand Prix": {"HAD": "LAW", "LAW": "TSU"}})

    result = apply_substitutions(
        _lineups(), race_name="Dutch Grand Prix", year=2026, cfg=cfg, store=_BrokenStore()
    )

    assert result["Red Bull Racing"] == ["VER", "LAW"]


def test_saving_normalises_and_records_the_race():
    store = _Store()

    save_substitutions(
        race_name="Dutch Grand Prix",
        year=2026,
        substitutions={" had ": "law", "law": " tsu "},
        lineups=_lineups(),
        store=store,
    )

    assert store.saved["substitutions"] == {"HAD": "LAW", "LAW": "TSU"}
    assert store.saved["race_name"] == "Dutch Grand Prix"


def test_saving_rejects_half_a_chain():
    """Moving LAW up without freeing his own seat would put him in two cars."""
    store = _Store()

    with pytest.raises(ValueError, match="twice"):
        save_substitutions(
            race_name="Dutch Grand Prix",
            year=2026,
            substitutions={"HAD": "LAW"},
            lineups=_lineups(),
            store=store,
        )


@pytest.mark.parametrize(
    "substitutions",
    [
        {"BOT": "TSU"},  # Not in the lineup.
        {"HAD": "VER"},  # Would seat VER twice.
        {"HAD": "HAD"},  # A no-op typed in by mistake.
        {"HAD": ""},  # Half-filled form.
    ],
)
def test_saving_rejects_a_substitution_that_cannot_hold(substitutions):
    store = _Store()

    with pytest.raises(ValueError):
        save_substitutions(
            race_name="Dutch Grand Prix",
            year=2026,
            substitutions=substitutions,
            lineups=_lineups(),
            store=store,
        )

    assert store.saved is None
