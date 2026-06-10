"""Circuit-identity registry: resolve Grand Prix names to stable physical circuits.

Grand Prix *names* are political and migrate between circuits, so keying track data,
tyre data, and cross-year history by GP name risks comparing different physical
circuits. The clearest live example is Spain in 2026:

- Round 7 ``Barcelona Grand Prix`` runs at the Circuit de Barcelona-Catalunya - the
  venue every historical "Spanish Grand Prix" used through 2025.
- Round 14 ``Spanish Grand Prix`` runs at Madrid (Madring) - a different circuit.

A naive name match would feed Barcelona's track/tyre/overtaking history into the Madrid
race (and miss Barcelona entirely). This module resolves ``(race_name, year, location)``
to a stable :class:`Circuit` so callers compare like-for-like, and it **hard-fails** on
any race it cannot confidently identify rather than borrowing another circuit's data.

``location`` (the FastF1 schedule ``Location`` field) is the most reliable signal and is
preferred when available; otherwise an explicit, auditable name/year mapping is used.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitResolutionError(ValueError):
    """Raised when a race cannot be confidently mapped to a known circuit."""


def _normalize(value: str | None) -> str:
    """Lower-case, strip accents/whitespace so 'Montréal'/'Montreal' compare equal."""
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.strip().lower().split())


@dataclass(frozen=True)
class Circuit:
    """One stable physical circuit and the names/venues that identify it.

    ``data_key`` is the key under which this circuit's track/tyre characteristics are
    stored today (``None`` means the circuit is recognised but has no characteristics
    data yet - callers should use neutral defaults, never another circuit's data).
    """

    circuit_id: str
    display_name: str
    country: str
    data_key: str | None
    locations: frozenset[str] = field(default_factory=frozenset)
    race_names: frozenset[str] = field(default_factory=frozenset)


# --- Circuit registry -------------------------------------------------------------
# One entry per physical circuit on (or recently on) the calendar. ``data_key`` points
# at the existing track_characteristics / pirelli key; for most circuits the GP name and
# the data key coincide. The Spain split is the deliberate exception.
def _c(
    circuit_id: str,
    display_name: str,
    country: str,
    data_key: str | None,
    locations: tuple[str, ...],
    race_names: tuple[str, ...],
) -> Circuit:
    return Circuit(
        circuit_id=circuit_id,
        display_name=display_name,
        country=country,
        data_key=data_key,
        locations=frozenset(_normalize(item) for item in locations),
        race_names=frozenset(_normalize(item) for item in race_names),
    )


_CIRCUITS: tuple[Circuit, ...] = (
    _c(
        "albert_park",
        "Albert Park",
        "Australia",
        "Australian Grand Prix",
        ("Melbourne",),
        ("Australian Grand Prix",),
    ),
    _c(
        "shanghai",
        "Shanghai International",
        "China",
        "Chinese Grand Prix",
        ("Shanghai",),
        ("Chinese Grand Prix",),
    ),
    _c("suzuka", "Suzuka", "Japan", "Japanese Grand Prix", ("Suzuka",), ("Japanese Grand Prix",)),
    _c(
        "miami",
        "Miami International Autodrome",
        "United States",
        "Miami Grand Prix",
        ("Miami", "Miami Gardens"),
        ("Miami Grand Prix",),
    ),
    _c(
        "villeneuve",
        "Circuit Gilles Villeneuve",
        "Canada",
        "Canadian Grand Prix",
        ("Montreal",),
        ("Canadian Grand Prix",),
    ),
    _c(
        "monaco",
        "Circuit de Monaco",
        "Monaco",
        "Monaco Grand Prix",
        ("Monte Carlo", "Monaco"),
        ("Monaco Grand Prix",),
    ),
    # Spain: two distinct circuits. Barcelona-Catalunya holds the historical "Spanish
    # Grand Prix" data; Madrid/Madring is a new circuit with no characteristics yet.
    _c(
        "barcelona_catalunya",
        "Circuit de Barcelona-Catalunya",
        "Spain",
        "Spanish Grand Prix",
        ("Barcelona", "Montmelo", "Catalunya"),
        ("Barcelona Grand Prix",),
    ),
    _c("madrid_madring", "Madring", "Spain", None, ("Madrid",), ("Madrid Grand Prix",)),
    _c(
        "red_bull_ring",
        "Red Bull Ring",
        "Austria",
        "Austrian Grand Prix",
        ("Spielberg",),
        ("Austrian Grand Prix",),
    ),
    _c(
        "silverstone",
        "Silverstone",
        "United Kingdom",
        "British Grand Prix",
        ("Silverstone",),
        ("British Grand Prix",),
    ),
    _c(
        "spa",
        "Spa-Francorchamps",
        "Belgium",
        "Belgian Grand Prix",
        ("Spa-Francorchamps", "Spa"),
        ("Belgian Grand Prix",),
    ),
    _c(
        "hungaroring",
        "Hungaroring",
        "Hungary",
        "Hungarian Grand Prix",
        ("Budapest",),
        ("Hungarian Grand Prix",),
    ),
    _c(
        "zandvoort",
        "Zandvoort",
        "Netherlands",
        "Dutch Grand Prix",
        ("Zandvoort",),
        ("Dutch Grand Prix",),
    ),
    _c("monza", "Monza", "Italy", "Italian Grand Prix", ("Monza",), ("Italian Grand Prix",)),
    _c(
        "imola",
        "Imola",
        "Italy",
        "Emilia Romagna Grand Prix",
        ("Imola",),
        ("Emilia Romagna Grand Prix",),
    ),
    _c(
        "baku",
        "Baku City Circuit",
        "Azerbaijan",
        "Azerbaijan Grand Prix",
        ("Baku",),
        ("Azerbaijan Grand Prix",),
    ),
    _c(
        "marina_bay",
        "Marina Bay",
        "Singapore",
        "Singapore Grand Prix",
        ("Marina Bay", "Singapore"),
        ("Singapore Grand Prix",),
    ),
    _c(
        "cota",
        "Circuit of the Americas",
        "United States",
        "United States Grand Prix",
        ("Austin",),
        ("United States Grand Prix",),
    ),
    _c(
        "rodriguez",
        "Autodromo Hermanos Rodriguez",
        "Mexico",
        "Mexico City Grand Prix",
        ("Mexico City",),
        ("Mexico City Grand Prix", "Mexican Grand Prix"),
    ),
    _c(
        "interlagos",
        "Interlagos",
        "Brazil",
        "Sao Paulo Grand Prix",
        ("Sao Paulo",),
        ("Sao Paulo Grand Prix", "Brazilian Grand Prix"),
    ),
    _c(
        "las_vegas",
        "Las Vegas Strip Circuit",
        "United States",
        "Las Vegas Grand Prix",
        ("Las Vegas",),
        ("Las Vegas Grand Prix",),
    ),
    _c("losail", "Lusail", "Qatar", "Qatar Grand Prix", ("Lusail", "Doha"), ("Qatar Grand Prix",)),
    _c(
        "yas_marina",
        "Yas Marina",
        "United Arab Emirates",
        "Abu Dhabi Grand Prix",
        ("Yas Marina", "Yas Island", "Abu Dhabi"),
        ("Abu Dhabi Grand Prix",),
    ),
    _c(
        "sakhir",
        "Bahrain International",
        "Bahrain",
        "Bahrain Grand Prix",
        ("Sakhir",),
        ("Bahrain Grand Prix",),
    ),
    _c(
        "jeddah",
        "Jeddah Corniche",
        "Saudi Arabia",
        "Saudi Arabian Grand Prix",
        ("Jeddah",),
        ("Saudi Arabian Grand Prix",),
    ),
    # Recently-dropped circuit kept for historical (2022) generation; no current data.
    _c(
        "le_castellet",
        "Circuit Paul Ricard",
        "France",
        None,
        ("Le Castellet",),
        ("French Grand Prix",),
    ),
)

# Ambiguous GP names whose circuit depends on year/venue. The resolver requires a
# ``location`` or ``year`` to disambiguate these and hard-fails otherwise, so a name
# migration can never silently pick the wrong circuit.
#   "Spanish Grand Prix": Barcelona-Catalunya through 2025; Madrid (Madring) from 2026.
_AMBIGUOUS_NAME_YEAR_RULES: dict[str, tuple[tuple[int, str], ...]] = {
    # name -> ordered (first_year_inclusive, circuit_id); the last matching rule wins.
    _normalize("Spanish Grand Prix"): ((0, "barcelona_catalunya"), (2026, "madrid_madring")),
}

_BY_ID: dict[str, Circuit] = {circuit.circuit_id: circuit for circuit in _CIRCUITS}
_BY_LOCATION: dict[str, Circuit] = {
    location: circuit for circuit in _CIRCUITS for location in circuit.locations
}


def _unambiguous_name_index() -> dict[str, Circuit]:
    """Map each race name to its circuit, excluding names flagged ambiguous."""
    index: dict[str, list[Circuit]] = {}
    for circuit in _CIRCUITS:
        for name in circuit.race_names:
            index.setdefault(name, []).append(circuit)
    resolved: dict[str, Circuit] = {}
    for name, circuits in index.items():
        if name in _AMBIGUOUS_NAME_YEAR_RULES:
            continue
        if len(circuits) == 1:
            resolved[name] = circuits[0]
    return resolved


_BY_NAME: dict[str, Circuit] = _unambiguous_name_index()


def _resolve_ambiguous(name: str, year: int | None) -> Circuit | None:
    """Resolve a year-dependent ambiguous GP name; None if year is missing/unmapped."""
    rules = _AMBIGUOUS_NAME_YEAR_RULES.get(name)
    if not rules or year is None:
        return None
    chosen: str | None = None
    for first_year, circuit_id in sorted(rules, key=lambda item: item[0]):
        if year >= first_year:
            chosen = circuit_id
    return _BY_ID.get(chosen) if chosen else None


def resolve_circuit(
    race_name: str,
    *,
    year: int | None = None,
    location: str | None = None,
) -> Circuit:
    """Resolve a race to its physical circuit, or raise ``CircuitResolutionError``.

    The physical ``location`` (FastF1 schedule ``Location``) is authoritative when it is
    registered. If a location is supplied but not registered, an *unambiguous* GP name is
    still resolved by name (the location was a stale/unknown hint), while an *ambiguous*
    name hard-fails - so a FastF1 location-string change can never silently mis-resolve a
    migrating circuit (e.g. the Spanish GP). With no location, resolution falls back to an
    unambiguous name, then a year rule for known-ambiguous names, then hard-fail.
    """
    normalized_name = _normalize(race_name)
    name_is_ambiguous = normalized_name in _AMBIGUOUS_NAME_YEAR_RULES

    normalized_location = _normalize(location)
    if normalized_location:
        circuit = _BY_LOCATION.get(normalized_location)
        if circuit is not None:
            return circuit
        if name_is_ambiguous:
            raise CircuitResolutionError(
                f"Unrecognised location {location!r} for ambiguous race {race_name!r}; "
                "cannot safely disambiguate (it has run at different circuits across years). "
                "Register the location in src/data/circuit_registry.py before warming it."
            )
        logger.warning(
            "Unrecognised circuit location %r for %r; falling back to the GP name. "
            "Consider registering the location in src/data/circuit_registry.py.",
            location,
            race_name,
        )

    if name_is_ambiguous:
        circuit = _resolve_ambiguous(normalized_name, year)
        if circuit is not None:
            return circuit
        raise CircuitResolutionError(
            f"Ambiguous race name {race_name!r}: it has run at different circuits across "
            "years (e.g. Spanish GP = Barcelona <=2025, Madrid >=2026). Provide a "
            "schedule location or year so the correct circuit is used."
        )

    circuit = _BY_NAME.get(normalized_name)
    if circuit is not None:
        return circuit

    raise CircuitResolutionError(
        f"Unrecognised race {race_name!r}: no circuit is registered for it. Add it to "
        "src/data/circuit_registry.py (with its physical circuit) before warming it."
    )


def resolve_track_data_key(
    race_name: str,
    *,
    year: int | None = None,
    location: str | None = None,
) -> str | None:
    """Return the track/tyre data key for a race's circuit (None = use defaults).

    Hard-fails (via :func:`resolve_circuit`) when the circuit cannot be identified, so a
    migrating GP name (Spanish GP -> Madrid) never resolves to another circuit's data.
    """
    return resolve_circuit(race_name, year=year, location=location).data_key


def circuit_aggregation_key(
    race_name: str,
    *,
    year: int | None = None,
    location: str | None = None,
) -> str:
    """Return a stable per-circuit key for grouping cross-year history.

    Resolves the physical circuit and returns its ``data_key`` (or ``circuit_id`` when
    the circuit has no characteristics data yet), so results from the same circuit group
    together across GP-name changes (Barcelona GP and the pre-2026 Spanish GP both key on
    the Catalunya data) and never merge with a different circuit (the 2026 Madrid Spanish
    GP keys separately). Unregistered races fall back to the raw name (legacy behaviour),
    so historical generation is unchanged.
    """
    try:
        circuit = resolve_circuit(race_name, year=year, location=location)
    except CircuitResolutionError:
        return str(race_name)
    return circuit.data_key or circuit.circuit_id


def all_circuits() -> tuple[Circuit, ...]:
    """Return every registered circuit (for audits/diagnostics)."""
    return _CIRCUITS
