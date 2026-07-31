#!/usr/bin/env python3
"""Build the curated street/permanent track-class binding for the 2026 research catalog.

User-approved taxonomy: exactly the repo's existing ``track_type`` classification (the
same field ``generate_evaluation_report.py``'s ``segment_breakdown.track_type`` uses),
resolved through the circuit registry (``src.data.circuit_registry``) so a migrating GP
name (e.g. the 2026 Barcelona GP, which keys the pre-2026 "Spanish Grand Prix" data
under the physical Circuit de Barcelona-Catalunya) still finds its real classification
instead of a name-mismatch gap. No new taxonomy, no invented per-event guess: every
event either resolves to the repo's own "street"/"permanent" label or is recorded as
unmapped with the exact reason, and the script fails closed (non-zero exit, no file
written) if any 2026 catalog event is unmapped.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.circuit_registry import CircuitResolutionError, resolve_track_data_key  # noqa: E402

CATALOG_PATH = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "event_catalog.json"
TRACK_CHARACTERISTICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "track_characteristics"
    / "2026_track_characteristics.json"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "track_class_by_event.json"
ALLOWED_CLASSES = frozenset({"street", "permanent"})


def _track_type(race_name: str, *, year: int, tracks: dict[str, Any]) -> tuple[str | None, str]:
    """Return (track_class, reason) -- track_class is None only when unmapped."""
    try:
        data_key = resolve_track_data_key(race_name, year=year)
    except CircuitResolutionError as exc:
        return None, f"circuit_unresolved: {exc}"
    if not data_key:
        return None, "circuit resolved with no data key (e.g. a not-yet-raced new circuit)"
    track_info = tracks.get(data_key)
    if not isinstance(track_info, dict):
        return None, f"resolved data_key {data_key!r} not present in track_characteristics"
    raw_type = str(track_info.get("type") or "").strip().lower()
    if raw_type not in ALLOWED_CLASSES:
        return (
            None,
            f"track_characteristics type {raw_type!r} is not one of {sorted(ALLOWED_CLASSES)}",
        )
    return raw_type, f"resolved via circuit_registry to data_key={data_key!r}"


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    track_data = json.loads(TRACK_CHARACTERISTICS_PATH.read_text(encoding="utf-8"))
    tracks = track_data.get("tracks", {})

    bindings: dict[str, Any] = {}
    unmapped: list[dict[str, str]] = []
    for event in catalog["events"]:
        event_id = event["event_id"]
        race_name = event["race_name"]
        year = int(event.get("year", 2026))
        track_class, reason = _track_type(race_name, year=year, tracks=tracks)
        bindings[event_id] = {
            "race_name": race_name,
            "track_class": track_class,
            "resolution": reason,
        }
        if track_class is None:
            unmapped.append({"event_id": event_id, "race_name": race_name, "reason": reason})

    if unmapped:
        print("FAILED CLOSED: the following catalog events have no resolvable track_class:")
        for row in unmapped:
            print(f"  {row['event_id']} ({row['race_name']}): {row['reason']}")
        print("Refusing to write a partial/guessed binding file. Fix the data, then re-run.")
        return 1

    payload = {
        "artifact_type": "research_track_class_by_event",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provenance": {
            "source": "repo track_type classification (track_characteristics + circuit_registry)",
            "approved_by_user": "2026-07-19",
            "research_only": True,
            "taxonomy": sorted(ALLOWED_CLASSES),
        },
        "bindings": bindings,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(bindings)} events, all resolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
