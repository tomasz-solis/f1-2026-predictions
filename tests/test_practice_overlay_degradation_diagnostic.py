"""Focused tests for the practice-overlay degradation diagnostic's two pieces
of non-trivial logic: matched-event intersection and seed-spread recomputation
from raw cached predictions (the raw cache has no pre-computed metric fields --
this script must recompute them, unlike the aggregated walk-forward output)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.build_practice_overlay_degradation_diagnostic as diag


def test_matched_event_ids_requires_every_checkpoint_present() -> None:
    index = {
        "e1": {"PRE": {}, "FP2": {}, "FP3": {}},
        "e2": {"PRE": {}, "FP2": {}},  # missing FP3 -- must be excluded
        "e3": {"PRE": {}, "FP2": {}, "FP3": {}},
    }
    matched = diag._matched_event_ids(index, ["PRE", "FP2", "FP3"], {"e1", "e2", "e3"})
    assert matched == ["e1", "e3"]


def test_matched_event_ids_empty_when_a_whole_checkpoint_never_ran() -> None:
    index = {"e1": {"PRE": {}}}
    assert diag._matched_event_ids(index, ["PRE", "FP2"], {"e1"}) == []


def _write_qualifying(
    root: Path, *, checkpoint: str, event_id: str, seed: int, grid: list[dict[str, Any]]
) -> None:
    key = {
        "kind": "qualifying",
        "event_id": event_id,
        "checkpoint": checkpoint,
        "variant_id": "champion",
        "seed": seed,
    }
    payload = {"key": key, "value": {"grid": grid}}
    subdir = root / "ab"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"q-{checkpoint}-{event_id}-{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_seed_level_metrics_recomputes_grid_mae_from_raw_grid_not_a_cached_field(
    tmp_path, monkeypatch
) -> None:
    """The raw cache entry only has a driver grid (no pre-computed grid_mae) --
    this must recompute it via _qualifying_grid_metrics, matching the real
    walk-forward scorer's own metric."""
    monkeypatch.setattr(diag, "PREDICTION_CACHE_ROOT", tmp_path)
    grid = [
        {"driver": "A", "team": "T1", "position": 1, "p5": 1, "p95": 1},
        {"driver": "B", "team": "T1", "position": 2, "p5": 2, "p95": 2},
    ]
    _write_qualifying(tmp_path, checkpoint="FP2", event_id="e1", seed=17, grid=grid)
    catalog = {
        "e1": {
            "actual_qualifying_grid": [
                {"driver": "A", "team": "T1", "position": 1},
                {"driver": "B", "team": "T1", "position": 2},
            ]
        }
    }
    index = diag._seed_level_metrics(catalog, ["e1"], ["FP2"])
    # driver A predicted at grid-order position 1 (matches p5/p95=1) vs actual 1 -> error 0;
    # driver B predicted position 2 vs actual 2 -> error 0 -- exact match, grid_mae == 0.0.
    assert index["e1"]["FP2"][17]["qualifying"] == 0.0


def test_seed_spread_zero_when_every_seed_agrees() -> None:
    seed_index = {
        "e1": {"FP2": {17: {"qualifying": 3.0}, 42: {"qualifying": 3.0}, 91: {"qualifying": 3.0}}}
    }
    spread = diag._seed_spread(seed_index, ["e1"], ["FP2"], field="qualifying", metric=None)
    assert spread["FP2"]["mean_per_event_seed_std"] == 0.0
    assert spread["FP2"]["n_events_with_multiple_seeds"] == 1


def test_seed_spread_nonzero_when_seeds_disagree() -> None:
    seed_index = {"e1": {"FP2": {17: {"qualifying": 2.0}, 42: {"qualifying": 4.0}}}}
    spread = diag._seed_spread(seed_index, ["e1"], ["FP2"], field="qualifying", metric=None)
    assert spread["FP2"]["mean_per_event_seed_std"] == 1.0  # pstdev([2, 4]) == 1.0


def test_per_event_delta_sorted_worst_first() -> None:
    index = {
        "e1": {"PRE": {"qualifying": {"m": 1.0}}, "FP3": {"qualifying": {"m": 2.0}}},
        "e2": {"PRE": {"qualifying": {"m": 1.0}}, "FP3": {"qualifying": {"m": 5.0}}},
    }
    rows = diag._per_event_delta(
        index,
        ["e1", "e2"],
        start_checkpoint="PRE",
        end_checkpoint="FP3",
        kind="qualifying",
        metric="m",
    )
    assert [r["event_id"] for r in rows] == ["e2", "e1"]
    assert rows[0]["delta"] == 4.0
