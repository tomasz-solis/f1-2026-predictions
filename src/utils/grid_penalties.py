"""Apply post-qualifying grid penalties to the grid handed to race prediction.

Qualifying classification is not the starting grid: a driver who qualifies P3 and
takes a power-unit penalty is still classified P3 by the timing feed. Once the race
has run, ``fetch_actual_starting_grid`` reads the real grid from FastF1 and this
module stops mattering. Before it, there is no automated source — the stewards
publish decisions as PDFs on a Saturday night — so the drop is typed in by hand
against ``grid.penalties`` in the config.

The drop is applied here, to the race grid, and never to the qualifying prediction.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from src.types.prediction_types import QualifyingGridEntry
from src.utils import config_loader

logger = logging.getLogger(__name__)

ARTIFACT_TYPE = "grid_penalties"

# Both sort behind every place-drop while keeping penalised drivers in qualifying
# order among themselves, which is how the stewards break that tie in practice.
_BACK_OF_GRID = 1_000
_PIT_LANE = 2_000


class AppliedPenalty(NamedTuple):
    """One penalty as it was actually applied, for display and for the record."""

    driver: str
    qualified: int
    starts: int
    penalty: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row for checkpoint payloads."""
        return {
            "driver": self.driver,
            "qualified": self.qualified,
            "starts": self.starts,
            "penalty": self.penalty,
        }


class PenalisedGrid(NamedTuple):
    """The grid to race from, plus what was done to it."""

    grid: list[QualifyingGridEntry]
    applied: list[AppliedPenalty]


def penalty_artifact_key(year: int | None, race_name: str | None) -> str:
    """Return the artifact key one race's penalties are stored under."""
    return f"{int(year) if year is not None else 0}::{str(race_name).strip()}"


def load_configured_penalties(
    *,
    race_name: str | None,
    year: int | None = None,
    cfg: Any = config_loader,
    store: Any = None,
) -> dict[str, Any]:
    """Return one race's penalties, preferring the runtime store over the config file.

    The stored artifact is what the dashboard writes, so a penalty can be entered on a
    Saturday night without a deploy. The config file stays authoritative when nothing is
    stored, which keeps local runs and committed history working. A storage failure falls
    back to the config rather than breaking serving.
    """
    if year is not None:
        try:
            artifact_store = store if store is not None else _default_store()
            stored = artifact_store.load_artifact(
                ARTIFACT_TYPE, penalty_artifact_key(year, race_name)
            )
            if stored:
                penalties = stored.get("penalties") or {}
                if isinstance(penalties, dict):
                    return dict(penalties)
        except Exception as exc:  # noqa: BLE001 - a store outage must not break predicting
            logger.warning(
                "Could not read stored grid penalties for %s %s: %s", race_name, year, exc
            )

    configured = cfg.get("grid.penalties", {}) or {}
    penalties = configured.get(str(race_name).strip(), {}) or {}
    return dict(penalties) if isinstance(penalties, dict) else {}


def save_penalties(
    *,
    race_name: str,
    year: int,
    penalties: dict[str, Any],
    store: Any = None,
) -> dict[str, Any]:
    """Persist one race's penalties to the runtime store, validating them first."""
    cleaned: dict[str, Any] = {}
    for driver, penalty in penalties.items():
        code = str(driver).strip().upper()
        if not code:
            raise ValueError("Grid penalty driver code cannot be empty")
        _places_dropped(penalty)  # Reject a malformed value before it reaches the store.
        cleaned[code] = penalty if isinstance(penalty, int) else str(penalty).strip().lower()

    artifact_store = store if store is not None else _default_store()
    return artifact_store.save_artifact(
        artifact_type=ARTIFACT_TYPE,
        artifact_key=penalty_artifact_key(year, race_name),
        data={"race_name": str(race_name).strip(), "year": int(year), "penalties": cleaned},
    )


def _default_store() -> Any:
    """Build the artifact store lazily so importing this module stays cheap."""
    from src.persistence.artifact_store import ArtifactStore

    return ArtifactStore(data_root="data")


def _places_dropped(penalty: Any) -> int:
    """Read one configured penalty as a number of places."""
    text = str(penalty).strip().lower()
    if text == "back":
        return _BACK_OF_GRID
    if text == "pit":
        return _PIT_LANE
    try:
        places = int(penalty)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Grid penalty must be a place count, 'back' or 'pit': {penalty!r}"
        ) from exc
    if places < 1:
        raise ValueError(f"Grid penalty must be at least one place, got {penalty!r}")
    return places


def apply_grid_penalties(
    grid: list[QualifyingGridEntry],
    *,
    race_name: str | None,
    year: int | None = None,
    cfg: Any = config_loader,
    store: Any = None,
) -> PenalisedGrid:
    """Return the grid reordered by this race's penalties, plus what was applied."""
    penalties = load_configured_penalties(race_name=race_name, year=year, cfg=cfg, store=store)
    if not grid or not penalties:
        return PenalisedGrid(grid, [])

    raw_by_code = {str(driver).strip().upper(): penalty for driver, penalty in penalties.items()}
    drops = {code: _places_dropped(penalty) for code, penalty in raw_by_code.items()}
    unknown = sorted(drops.keys() - {str(row["driver"]).strip().upper() for row in grid})
    if unknown:
        raise ValueError(f"Grid penalties name drivers who are not on the grid: {unknown}")

    # The stewards do not re-sort the classification by "position + places". They take the
    # penalised cars out, let everyone behind close up, and put each penalised car back in
    # at the slot his drop earns him. Those give different grids as soon as two drivers are
    # penalised: at the 2026 Hungarian GP, HAM (P2) and ANT (P4) both took three places, and
    # the sort-by-total rule puts HAM ahead of VER where the real grid has him behind.
    clean: list[QualifyingGridEntry] = []
    targets: list[tuple[int, int, QualifyingGridEntry]] = []
    for row in sorted(grid, key=lambda entry: int(entry["position"])):
        code = str(row["driver"]).strip().upper()
        drop = drops.get(code)
        if drop:
            targets.append((int(row["position"]) + drop, int(row["position"]), row))
        else:
            clean.append(row)

    ordered: list[QualifyingGridEntry] = list(clean)
    for target_slot, _qualified, row in sorted(targets, key=lambda item: (item[0], item[1])):
        # A drop bigger than the field simply reaches the back; it cannot push anyone off.
        ordered.insert(min(max(target_slot - 1, 0), len(ordered)), row)

    penalised: list[QualifyingGridEntry] = []
    applied: list[AppliedPenalty] = []
    for position, row in enumerate(ordered, start=1):
        new_row: dict[str, Any] = dict(row)
        new_row["position"] = position
        driver_code = str(row["driver"]).strip().upper()
        drop = drops.get(driver_code)
        if drop:
            applied.append(
                AppliedPenalty(
                    driver=str(row["driver"]),
                    qualified=int(row["position"]),
                    starts=position,
                    penalty=str(raw_by_code[driver_code]),
                )
            )
            # A steward's grid slot is certain. The race simulation samples starting
            # positions from median_position/p5/p95, not from position, so the
            # qualifying spread has to collapse with the move or the driver is sampled
            # straight back to where he qualified and the penalty does nothing.
            new_row.update(median_position=position, p5=position, p95=position, confidence=1.0)
            # Record where pace put the driver before the drop. The grid slot below
            # governs where the race simulation starts him; this is the separate pace
            # evidence a penalty breaks the "grid proxies pace" assumption for.
            new_row["qualifying_position"] = int(row["position"])
            if drop == _PIT_LANE:
                new_row["start_type"] = "pit_lane"
        penalised.append(new_row)  # type: ignore[arg-type]
    return PenalisedGrid(penalised, applied)
