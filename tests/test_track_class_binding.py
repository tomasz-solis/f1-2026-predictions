"""Tests for the curated street/permanent track-class binding.

Covers the two things this round is about: the binding builder fails closed (never
writes a partial/guessed file) when a catalog event has no resolvable track_type, and
the backend's Q1 eligibility check reproduces the expected street/permanent split on a
synthetic catalog shaped like the real 2026 one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import scripts.build_track_class_binding as binding_builder

from src.analysis.challenger_research_backend import ProductionReplayBackend
from src.analysis.challenger_walk_forward import _normalise_catalog


def test_track_type_resolution_fails_closed_for_an_unmapped_circuit() -> None:
    """No repo track_type entry for the resolved circuit -> (None, reason), never a guess."""
    track_class, reason = binding_builder._track_type(
        "Totally Unknown Grand Prix",
        year=2026,
        tracks={"Some Other Grand Prix": {"type": "street"}},
    )
    assert track_class is None
    assert reason


def test_binding_builder_refuses_to_write_a_partial_file_when_one_event_is_unmapped(
    tmp_path, monkeypatch
) -> None:
    catalog = {
        "events": [
            {"event_id": "e1", "race_name": "Australian Grand Prix", "year": 2026},
            {"event_id": "e2", "race_name": "Nonexistent Grand Prix", "year": 2026},
        ]
    }
    tracks_payload = {"tracks": {"Australian Grand Prix": {"type": "street"}}}
    catalog_path = tmp_path / "event_catalog.json"
    tracks_path = tmp_path / "tracks.json"
    output_path = tmp_path / "track_class_by_event.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    tracks_path.write_text(json.dumps(tracks_payload), encoding="utf-8")

    monkeypatch.setattr(binding_builder, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(binding_builder, "TRACK_CHARACTERISTICS_PATH", tracks_path)
    monkeypatch.setattr(binding_builder, "OUTPUT_PATH", output_path)

    exit_code = binding_builder.main()

    assert exit_code == 1
    assert not output_path.exists()


def _synthetic_catalog() -> list[dict[str, Any]]:
    """Shaped like the real 2026 catalog's street/permanent/dry pattern: street events
    (rounds 1, 6) never accumulate 4 same-class dry priors; permanent events (rounds
    2, 3, 7, 8, 9) do, first at round 9."""
    rounds = [
        (1, "street", True),
        (2, "permanent", True),
        (3, "permanent", True),
        (4, "street", False),  # wet, excluded from prior counts
        (5, "permanent", False),  # wet, excluded from prior counts
        (6, "street", True),
        (7, "permanent", True),
        (8, "permanent", True),
        (9, "permanent", True),
    ]
    events = []
    bindings = {}
    for index, (round_number, track_class, is_dry) in enumerate(rounds):
        event_id = f"2026_{round_number:02d}_event"
        start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index * 14)
        events.append(
            {
                "event_id": event_id,
                "race_name": f"Round {round_number}",
                "year": 2026,
                "event_start_at": start.isoformat(),
                "qualifying_start_at": (start + timedelta(days=1, hours=5)).isoformat(),
                "session_kind": "main",
                "is_dry": is_dry,
                "checkpoint_payloads": {
                    "PRE": {"information_cutoff_at": start.isoformat(), "sessions_available": []}
                },
                "actual_qualifying_grid": [{"driver": "D1", "team": "T1", "position": 1}],
                "actual_race_finish_order": [
                    {"driver": "D1", "team": "T1", "position": 1, "dnf": False}
                ],
                "actual_starting_grid": [
                    {"driver": "D1", "team": "T1", "position": 1, "start_type": "grid"}
                ],
                "input_snapshot_ids": [f"snap-{index}"],
                "fastf1_cache_dir": "data/raw/.fastf1_cache",
            }
        )
        bindings[event_id] = {"track_class": track_class}
    return events, bindings


def test_q1_eligibility_matches_street_never_permanent_from_round_nine(tmp_path) -> None:
    events, bindings = _synthetic_catalog()
    binding_path = tmp_path / "track_class_by_event.json"
    binding_path.write_text(json.dumps({"bindings": bindings}), encoding="utf-8")

    backend = ProductionReplayBackend(
        events=events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        track_class_binding_path=binding_path,
    )
    catalog = _normalise_catalog(events)
    by_round = {event.event_id: event for event in catalog}

    # Street: round 6 is the only street event with any prior street event (round 1),
    # and that is 1 << the relaxed floor of 4 -- must always refuse.
    _, reason_street = backend._q1_track_class_eligibility(
        by_round["2026_06_event"], relaxed_floor=4
    )
    assert reason_street is not None
    assert "1 prior dry" in reason_street

    # Permanent: round 8 has 3 prior dry permanent events (2, 3, 7) -- still short.
    _, reason_round8 = backend._q1_track_class_eligibility(
        by_round["2026_08_event"], relaxed_floor=4
    )
    assert reason_round8 is not None
    assert "3 prior dry" in reason_round8

    # Round 9 has 4 prior dry permanent events (2, 3, 7, 8) -- first eligible fold.
    matching, reason_round9 = backend._q1_track_class_eligibility(
        by_round["2026_09_event"], relaxed_floor=4
    )
    assert reason_round9 is None
    assert matching == ["2026_02_event", "2026_03_event", "2026_07_event", "2026_08_event"]
