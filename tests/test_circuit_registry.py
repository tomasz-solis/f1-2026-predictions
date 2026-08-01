"""Tests for the circuit-identity registry and resolver.

The headline guarantee: the 2026 Spanish GP (Madrid) and Barcelona GP (Catalunya) are
two distinct circuits, so neither inherits the other's data, and unknown circuits
hard-fail rather than silently borrowing characteristics.
"""

import pytest

from src.data.circuit_registry import (
    CircuitResolutionError,
    circuit_aggregation_key,
    resolve_circuit,
    resolve_track_data_key,
)

# EventNames as the 2026 FastF1 schedule reports them (the names the warmup must resolve).
SCHEDULE_2026 = [
    "Australian Grand Prix",
    "Chinese Grand Prix",
    "Japanese Grand Prix",
    "Miami Grand Prix",
    "Canadian Grand Prix",
    "Monaco Grand Prix",
    "Barcelona Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Spanish Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix",
]


def test_barcelona_and_madrid_are_distinct_circuits():
    """The 2026 Spain split must never collapse into one circuit/data source."""
    barcelona = resolve_circuit("Barcelona Grand Prix", year=2026, location="Barcelona")
    madrid = resolve_circuit("Spanish Grand Prix", year=2026, location="Madrid")

    assert barcelona.circuit_id == "barcelona_catalunya"
    assert madrid.circuit_id == "madrid_madring"
    assert barcelona.circuit_id != madrid.circuit_id
    # Barcelona keeps the historical Catalunya characteristics; Madrid has none yet.
    assert barcelona.data_key == "Spanish Grand Prix"
    assert madrid.data_key is None


def test_madrid_does_not_inherit_barcelona_track_data():
    """The 2026 'Spanish Grand Prix' (Madrid) must not resolve to Barcelona's data key."""
    assert resolve_track_data_key("Spanish Grand Prix", year=2026, location="Madrid") is None
    # Even without a location, the year rule routes the 2026 Spanish GP to Madrid.
    assert resolve_track_data_key("Spanish Grand Prix", year=2026) is None
    # Barcelona GP resolves to the Catalunya data regardless of year.
    assert resolve_track_data_key("Barcelona Grand Prix", year=2026) == "Spanish Grand Prix"


def test_spanish_gp_year_rule_disambiguates_history():
    """'Spanish Grand Prix' is Barcelona through 2025 and Madrid from 2026."""
    assert resolve_circuit("Spanish Grand Prix", year=2024).circuit_id == "barcelona_catalunya"
    assert resolve_circuit("Spanish Grand Prix", year=2025).circuit_id == "barcelona_catalunya"
    assert resolve_circuit("Spanish Grand Prix", year=2026).circuit_id == "madrid_madring"
    assert resolve_circuit("Spanish Grand Prix", year=2027).circuit_id == "madrid_madring"


def test_location_takes_precedence_over_name():
    """A schedule location is authoritative even if the GP name has migrated."""
    # Hypothetical historical row: name 'Spanish Grand Prix' but run at Barcelona.
    assert (
        resolve_circuit("Spanish Grand Prix", year=2026, location="Barcelona").circuit_id
        == "barcelona_catalunya"
    )
    # Accent/spelling normalization (Montréal -> villeneuve, São Paulo -> interlagos).
    assert resolve_circuit("Canadian Grand Prix", location="Montréal").circuit_id == "villeneuve"
    assert resolve_circuit("São Paulo Grand Prix", year=2026).circuit_id == "interlagos"


def test_ambiguous_name_without_year_or_location_hard_fails():
    with pytest.raises(CircuitResolutionError, match="Ambiguous"):
        resolve_circuit("Spanish Grand Prix")


def test_unknown_race_hard_fails():
    with pytest.raises(CircuitResolutionError, match="Unrecognised race"):
        resolve_circuit("Kyalami Grand Prix", year=2026)


def test_unknown_location_falls_back_to_name_for_unambiguous_race():
    """A stale/unknown location must not break an unambiguous race - resolve by name."""
    assert (
        resolve_circuit("Barcelona Grand Prix", year=2026, location="Atlantis").circuit_id
        == "barcelona_catalunya"
    )


def test_unknown_location_hard_fails_for_ambiguous_race():
    """An ambiguous race with an unknown location cannot be safely disambiguated."""
    with pytest.raises(CircuitResolutionError, match="Unrecognised location"):
        resolve_circuit("Spanish Grand Prix", year=2026, location="Atlantis")


def test_registered_location_aliases_resolve():
    """Location-string variants for the same circuit all resolve (the user's caveat)."""
    # Yas Marina: FastF1 emits both 'Yas Marina' and 'Yas Island' across years.
    assert resolve_circuit("Abu Dhabi Grand Prix", location="Yas Island").circuit_id == "yas_marina"
    assert resolve_circuit("Abu Dhabi Grand Prix", location="Yas Marina").circuit_id == "yas_marina"
    # Barcelona: Barcelona / Montmeló / Catalunya all point at the same track.
    for loc in ("Barcelona", "Montmeló", "Catalunya"):
        assert (
            resolve_circuit("Spanish Grand Prix", location=loc).circuit_id == "barcelona_catalunya"
        )
    # Monaco / Monte Carlo, Miami / Miami Gardens.
    assert resolve_circuit("Monaco Grand Prix", location="Monte Carlo").circuit_id == "monaco"
    assert resolve_circuit("Miami Grand Prix", location="Miami Gardens").circuit_id == "miami"


@pytest.mark.parametrize("race_name", SCHEDULE_2026)
def test_every_2026_schedule_race_resolves(race_name):
    """Audit guarantee: every 2026 calendar race resolves (hard-fail would break warmup)."""
    circuit = resolve_circuit(race_name, year=2026)
    assert circuit.circuit_id
    assert circuit.country


def test_aggregation_key_groups_barcelona_history_but_splits_madrid():
    """Cross-year grouping: Barcelona history unifies; Madrid (2026 Spanish GP) splits off."""
    barcelona_2026 = circuit_aggregation_key("Barcelona Grand Prix", year=2026)
    spanish_2025 = circuit_aggregation_key("Spanish Grand Prix", year=2025)
    spanish_2026_madrid = circuit_aggregation_key("Spanish Grand Prix", year=2026)

    # Barcelona GP (2026) and the historical Spanish GP (Barcelona) share one key...
    assert barcelona_2026 == spanish_2025 == "Spanish Grand Prix"
    # ...but the 2026 Spanish GP at Madrid must NOT share that key.
    assert spanish_2026_madrid != barcelona_2026
    assert spanish_2026_madrid == "madrid_madring"
    # Location is honoured too.
    assert circuit_aggregation_key("Spanish Grand Prix", year=2026, location="Madrid") == (
        "madrid_madring"
    )


def test_aggregation_key_falls_back_to_raw_name_for_unregistered():
    assert circuit_aggregation_key("Kyalami Grand Prix", year=2026) == "Kyalami Grand Prix"


def test_no_two_circuits_share_a_data_key():
    """Two distinct circuits must never point at the same characteristics data."""
    from src.data.circuit_registry import all_circuits

    seen: dict[str, str] = {}
    for circuit in all_circuits():
        if circuit.data_key is None:
            continue
        assert circuit.data_key not in seen, (
            f"data_key {circuit.data_key!r} shared by {seen.get(circuit.data_key)} and "
            f"{circuit.circuit_id}"
        )
        seen[circuit.data_key] = circuit.circuit_id


def test_every_data_key_resolves_in_the_files_it_keys():
    """A ``data_key`` that matches no table silently degrades a race to config defaults.

    This is exactly how Interlagos regressed: the registry spelled it ``Sao Paulo`` while
    every table is keyed on FastF1's accented ``São Paulo``, so pit loss, SC/VSC, lap
    count, overtaking prior and tyre stress all fell back to defaults with no error.
    """
    import json
    from pathlib import Path

    from src.data.circuit_registry import all_circuits, pirelli_key
    from src.data.track_data_loader import KNOWN_MAIN_RACE_LAPS
    from src.utils.track_overtaking import TRACK_OVERTAKING_BASELINES

    track_chars = json.loads(
        Path("data/processed/track_characteristics/2026_track_characteristics.json").read_text(
            encoding="utf-8"
        )
    )["tracks"]
    pirelli = json.loads(Path("data/2025_pirelli_info.json").read_text(encoding="utf-8"))

    # Circuits that are registered but genuinely have no data yet resolve to None and are
    # covered by the defaults path, so only keyed circuits are checked here.
    missing: list[str] = []
    for circuit in all_circuits():
        key = circuit.data_key
        if key is None:
            continue
        for table_name, present in (
            ("2026_track_characteristics.json", key in track_chars),
            ("2025_pirelli_info.json", pirelli_key(key) in pirelli),
            ("TRACK_OVERTAKING_BASELINES", key in TRACK_OVERTAKING_BASELINES),
            ("KNOWN_MAIN_RACE_LAPS", key in KNOWN_MAIN_RACE_LAPS),
        ):
            if not present:
                missing.append(f"{circuit.circuit_id}: {key!r} missing from {table_name}")

    assert not missing, "Track data keys with no matching entry:\n  " + "\n  ".join(missing)
