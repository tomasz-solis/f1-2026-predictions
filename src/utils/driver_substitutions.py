"""Swap a race-weekend driver into another team's seat before prediction runs.

A driver misses a weekend — injury, illness, a mid-season drop — and a reserve or a
junior-team driver takes the seat, which usually pushes a second driver up the chain.
``data/current_lineups.json`` is one static season-wide file, so editing it means a
commit and a redeploy for a change that lasts one or two races, and it retroactively
mislabels driver-to-team attribution for every result already scored against it.

Substitutions are typed in per race instead, exactly like ``grid.penalties``: the
dashboard writes them to the runtime artifact store, the config file stays the fallback,
and the entry stops applying by itself once the next race is selected.

The map is flat and applied in one simultaneous pass, so a chain needs no ordering:
``{"HAD": "LAW", "LAW": "TSU"}`` over ``Red Bull: [VER, HAD]`` and ``RB: [LAW, LIN]``
gives ``Red Bull: [VER, LAW]`` and ``RB: [TSU, LIN]``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils import config_loader

logger = logging.getLogger(__name__)

ARTIFACT_TYPE = "driver_substitutions"


def substitution_artifact_key(year: int | None, race_name: str | None) -> str:
    """Return the artifact key one race's substitutions are stored under."""
    return f"{int(year) if year is not None else 0}::{str(race_name).strip()}"


def load_configured_substitutions(
    *,
    race_name: str | None,
    year: int | None = None,
    cfg: Any = config_loader,
    store: Any = None,
) -> dict[str, str]:
    """Return one race's substitutions, preferring the runtime store over the config file.

    A storage failure falls back to the config rather than breaking serving, which is the
    same trade ``grid_penalties`` makes: a missing substitution predicts the wrong driver,
    a raised exception predicts nothing at all.
    """
    if year is not None:
        try:
            artifact_store = store if store is not None else _default_store()
            stored = artifact_store.load_artifact(
                ARTIFACT_TYPE, substitution_artifact_key(year, race_name)
            )
            if stored:
                substitutions = stored.get("substitutions") or {}
                if isinstance(substitutions, dict):
                    return _normalise(substitutions)
        except Exception as exc:  # noqa: BLE001 - a store outage must not break predicting
            logger.warning(
                "Could not read stored driver substitutions for %s %s: %s", race_name, year, exc
            )

    configured = cfg.get("grid.substitutions", {}) or {}
    substitutions = configured.get(str(race_name).strip(), {}) or {}
    return _normalise(substitutions) if isinstance(substitutions, dict) else {}


def save_substitutions(
    *,
    race_name: str,
    year: int,
    substitutions: dict[str, str],
    lineups: dict[str, list[str]] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Persist one race's substitutions to the runtime store, validating them first.

    Validation happens here, where an operator sees the error, rather than at apply time
    deep inside a prediction run.
    """
    cleaned = _normalise(substitutions)
    for driver_out, driver_in in cleaned.items():
        if not driver_out or not driver_in:
            raise ValueError("A substitution needs both a driver out and a driver in")
        if driver_out == driver_in:
            raise ValueError(f"{driver_out} cannot be substituted for himself")

    if lineups is None:
        from src.utils.lineups import load_current_lineups

        lineups = load_current_lineups() or {}
    if lineups:
        _validate_against_lineups(cleaned, lineups)

    artifact_store = store if store is not None else _default_store()
    return artifact_store.save_artifact(
        artifact_type=ARTIFACT_TYPE,
        artifact_key=substitution_artifact_key(year, race_name),
        data={
            "race_name": str(race_name).strip(),
            "year": int(year),
            "substitutions": cleaned,
        },
    )


def apply_substitutions(
    lineups: dict[str, list[str]],
    *,
    race_name: str | None,
    year: int | None = None,
    cfg: Any = config_loader,
    store: Any = None,
) -> dict[str, list[str]]:
    """Return the lineups with this race's substitutions applied.

    Anything that does not hold — a driver who is no longer in the lineup, a swap that
    would seat one driver twice — is logged and skipped, leaving the configured lineup
    intact. A stale entry must not take a weekend's predictions down.
    """
    substitutions = load_configured_substitutions(
        race_name=race_name, year=year, cfg=cfg, store=store
    )
    if not lineups or not substitutions:
        return lineups

    try:
        _validate_against_lineups(substitutions, lineups)
    except ValueError as exc:
        logger.warning("Ignoring driver substitutions for %s %s: %s", race_name, year, exc)
        return lineups

    swapped = {
        team: [substitutions.get(str(driver).strip().upper(), driver) for driver in drivers]
        for team, drivers in lineups.items()
    }
    logger.info(
        "Applied driver substitutions for %s %s: %s",
        race_name,
        year,
        ", ".join(f"{out} -> {into}" for out, into in sorted(substitutions.items())),
    )
    return swapped


def _normalise(substitutions: dict[Any, Any]) -> dict[str, str]:
    """Upper-case and strip both sides of every swap."""
    return {
        str(driver_out).strip().upper(): str(driver_in).strip().upper()
        for driver_out, driver_in in substitutions.items()
    }


def _validate_against_lineups(
    substitutions: dict[str, str],
    lineups: dict[str, list[str]],
) -> None:
    """Raise when a swap names a driver who is not racing or would seat one driver twice."""
    seated = {str(driver).strip().upper() for drivers in lineups.values() for driver in drivers}
    unknown = sorted(set(substitutions) - seated)
    if unknown:
        raise ValueError(f"Substitutions name drivers who are not in the lineup: {unknown}")

    resulting = [substitutions.get(driver, driver) for driver in sorted(seated)]
    duplicated = sorted({code for code in resulting if resulting.count(code) > 1})
    if duplicated:
        raise ValueError(f"Substitutions would seat one driver twice: {duplicated}")


def _default_store() -> Any:
    """Build the artifact store lazily so importing this module stays cheap."""
    from src.persistence.artifact_store import ArtifactStore

    return ArtifactStore(data_root="data")
