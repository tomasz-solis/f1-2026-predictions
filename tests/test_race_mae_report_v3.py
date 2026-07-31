"""Focused tests for the phase-4 report's two new pieces of non-trivial logic:

- the generalized structural-identity guard (retroactive, cache-mined) and its
  legitimate-architectural-invariance vs pathological classification rule;
- the matched-subset checkpoint progression (only averages over the event set
  every compared checkpoint actually scored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.build_race_mae_investigation_report_v3 as report_v3


def _write_cache_entry(
    root: Path,
    *,
    kind: str,
    checkpoint: str,
    event_id: str,
    variant_id: str,
    seed: int,
    value: dict,
) -> None:
    key = {
        "kind": kind,
        "event_id": event_id,
        "checkpoint": checkpoint,
        "variant_id": variant_id,
        "seed": seed,
        "source_digest": "irrelevant",
    }
    payload = {"key": key, "value": value}
    subdir = root / variant_id[:2]
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"{kind}-{checkpoint}-{event_id}-{variant_id}-{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _grid(order: list[str]) -> dict[str, Any]:
    return {"grid": [{"driver": d} for d in order]}


def _race(order: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "conditional_actual_grid": {"finish_order": [{"driver": d} for d in order], **extra},
        "end_to_end_predicted_grid": {"finish_order": [{"driver": d} for d in order], **extra},
    }


def _replay_for(event_id: str, checkpoint: str) -> dict[str, Any]:
    return {
        "scored_events": [
            {"event_id": event_id, "checkpoints": {checkpoint: {}}},
        ]
    }


def test_grid_only_component_conditional_view_identity_is_architectural(
    tmp_path, monkeypatch
) -> None:
    """q0 (grid-only) identical to champion on conditional_actual_grid -> legitimate."""
    monkeypatch.setattr(report_v3, "PREDICTION_CACHE_ROOT", tmp_path)
    _write_cache_entry(
        tmp_path,
        kind="race_views",
        checkpoint="FP2",
        event_id="e1",
        variant_id="champion",
        seed=17,
        value=_race(["A", "B"]),
    )
    _write_cache_entry(
        tmp_path,
        kind="race_views",
        checkpoint="FP2",
        event_id="e1",
        variant_id="q0_driver_state",
        seed=17,
        value=_race(["A", "B"]),
    )
    flags = report_v3._structural_identity_scan("q0_driver_state", _replay_for("e1", "FP2"))
    conditional_flags = [f for f in flags if f["view"] == "conditional_actual_grid"]
    assert len(conditional_flags) == 1
    assert conditional_flags[0]["classification"] == "legitimate_architectural_invariance"


def test_race_affecting_component_identity_is_flagged_pathological_with_its_own_reason(
    tmp_path, monkeypatch
) -> None:
    """r0 (race-only) identical to champion on BOTH race views is structural_identity,
    with the reason mined from race_practice_challenger (r0's own disclosure field),
    not the unrelated q1 field that's always present-but-irrelevant on every payload."""
    monkeypatch.setattr(report_v3, "PREDICTION_CACHE_ROOT", tmp_path)
    _write_cache_entry(
        tmp_path,
        kind="race_views",
        checkpoint="FP2",
        event_id="e1",
        variant_id="champion",
        seed=17,
        value=_race(["A", "B"]),
    )
    _write_cache_entry(
        tmp_path,
        kind="race_views",
        checkpoint="FP2",
        event_id="e1",
        variant_id="r0_long_run",
        seed=17,
        value=_race(
            ["A", "B"],
            race_practice_challenger={
                "applied": False,
                "fallback_reason": "insufficient_field_evidence_coverage",
            },
            qualifying_practice_challenger={"used": False, "fallback_reason": None},
        ),
    )
    flags = report_v3._structural_identity_scan("r0_long_run", _replay_for("e1", "FP2"))
    assert len(flags) == 2  # both race views
    assert all(f["classification"] == "structural_identity" for f in flags)
    assert all(f["reason"] == "insufficient_field_evidence_coverage" for f in flags)


def test_non_qualifying_component_qualifying_identity_is_not_scanned(tmp_path, monkeypatch) -> None:
    """r0 has no qualifying component -- an identical qualifying grid is expected by
    construction and must not be flagged at all (nothing meaningful to say)."""
    monkeypatch.setattr(report_v3, "PREDICTION_CACHE_ROOT", tmp_path)
    _write_cache_entry(
        tmp_path,
        kind="qualifying",
        checkpoint="FP2",
        event_id="e1",
        variant_id="champion",
        seed=17,
        value=_grid(["A", "B"]),
    )
    _write_cache_entry(
        tmp_path,
        kind="qualifying",
        checkpoint="FP2",
        event_id="e1",
        variant_id="r0_long_run",
        seed=17,
        value=_grid(["A", "B"]),
    )
    flags = report_v3._structural_identity_scan("r0_long_run", _replay_for("e1", "FP2"))
    assert flags == []


def test_differing_predictions_are_never_flagged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(report_v3, "PREDICTION_CACHE_ROOT", tmp_path)
    _write_cache_entry(
        tmp_path,
        kind="race_views",
        checkpoint="FP2",
        event_id="e1",
        variant_id="champion",
        seed=17,
        value=_race(["A", "B"]),
    )
    _write_cache_entry(
        tmp_path,
        kind="race_views",
        checkpoint="FP2",
        event_id="e1",
        variant_id="r1_joint_grid",
        seed=17,
        value=_race(["B", "A"]),
    )
    flags = report_v3._structural_identity_scan("r1_joint_grid", _replay_for("e1", "FP2"))
    assert flags == []


def _matched_replay(
    event_ids_by_checkpoint: dict[str, list[str]], *, mae_by_event: dict[str, float]
) -> dict[str, Any]:
    """Build one scored_events list merging every checkpoint's event set."""
    events: dict[str, dict[str, Any]] = {}
    for checkpoint, event_ids in event_ids_by_checkpoint.items():
        for event_id in event_ids:
            events.setdefault(
                event_id, {"event_id": event_id, "session_kind": "main", "checkpoints": {}}
            )
            mae = mae_by_event[(event_id, checkpoint)]
            events[event_id]["checkpoints"][checkpoint] = {
                "champion": {},
                "challenger": {},
                "race_views": {
                    "conditional_actual_grid": {
                        "champion": {"finisher_mae": mae},
                        "challenger": {"finisher_mae": mae},
                    },
                    "end_to_end_predicted_grid": {
                        "champion": {"finisher_mae": mae},
                        "challenger": {"finisher_mae": mae},
                    },
                },
            }
    return {
        "scored_events": list(events.values()),
        "skipped_events": [],
        "leakage_audit": {"passed": True},
    }


def test_matched_subset_progression_excludes_unmatched_events(tmp_path) -> None:
    """PRE has 3 events, FP2/FP3 only have 2 of them scored -- the matched table
    must average PRE over only the 2 events FP2/FP3 also scored, not all 3.
    (main_dry_pre_fp2_fp3 requires PRE+FP2+FP3 all present -- FP3 mirrors FP2's
    event set here so the test isolates the "unmatched event excluded" behavior
    from the separate "a whole checkpoint never ran" case.)"""
    mae_by_event = {
        ("e1", "PRE"): 2.0,
        ("e2", "PRE"): 4.0,
        ("e3", "PRE"): 100.0,  # e3 unmatched outlier
        ("e1", "FP2"): 3.0,
        ("e2", "FP2"): 5.0,
        ("e1", "FP3"): 3.0,
        ("e2", "FP3"): 5.0,
    }
    replay = _matched_replay(
        {"PRE": ["e1", "e2", "e3"], "FP2": ["e1", "e2"], "FP3": ["e1", "e2"]},
        mae_by_event=mae_by_event,
    )
    runs = {"v": {"replay": replay}}
    catalog = {
        "e1": {"session_kind": "main"},
        "e2": {"session_kind": "main"},
        "e3": {"session_kind": "main"},
    }

    result = report_v3._matched_subset_progression(runs, catalog)
    table = result["main_dry_pre_fp2_fp3"]
    assert table["n_events"] == 2
    assert set(table["matched_event_ids"]) == {"e1", "e2"}
    # PRE's matched-subset mean must be (2.0 + 4.0) / 2 = 3.0, NOT diluted/skewed
    # by e3's 100.0 outlier which FP2/FP3 never scored.
    assert table["by_checkpoint"]["PRE"]["conditional_actual_grid"] == 3.0
    assert table["by_checkpoint"]["FP2"]["conditional_actual_grid"] == 4.0


def test_matched_subset_progression_reports_unavailable_when_no_overlap(tmp_path) -> None:
    replay = _matched_replay(
        {"PRE": ["e1"], "FP2": ["e2"]}, mae_by_event={("e1", "PRE"): 1.0, ("e2", "FP2"): 1.0}
    )
    runs = {"v": {"replay": replay}}
    catalog = {"e1": {"session_kind": "main"}, "e2": {"session_kind": "main"}}

    result = report_v3._matched_subset_progression(runs, catalog)
    table = result["main_dry_pre_fp2_fp3"]
    assert table["n_events"] == 0
    assert table["by_checkpoint"]["PRE"]["available"] is False
