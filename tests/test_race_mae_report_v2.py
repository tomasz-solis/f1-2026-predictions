"""Focused test for the phase-3 report's checkpoint-scoped cache mining.

``_driver_cohort_decomposition_scoped`` is the one piece of new, non-trivial logic in
the v2 report: it must only pool prediction-cache entries whose checkpoint belongs to
the replay being reported, so a higher-sim FP2/FP3 rerun is never silently diluted by
an older PRE-checkpoint (lower-sim) cache entry that happens to share a variant_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.build_race_mae_investigation_report_v2 as report_v2


def _write_cache_entry(
    root: Path, *, checkpoint: str, event_id: str, seed: int, position: int
) -> None:
    key = {
        "kind": "race_views",
        "event_id": event_id,
        "checkpoint": checkpoint,
        "variant_id": "champion",
        "seed": seed,
        "source_digest": "irrelevant",
    }
    payload = {
        "key": key,
        "value": {
            "conditional_actual_grid": {
                "finish_order": [{"driver": "D1", "position": position}],
            }
        },
    }
    subdir = root / "ab"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"{checkpoint}-{event_id}-{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_cohort_scoping_ignores_cache_entries_outside_the_replays_own_checkpoints(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(report_v2, "PREDICTION_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(report_v2, "DRIVER_DEBUTS_PATH", tmp_path / "debuts.json")
    (tmp_path / "debuts.json").write_text(
        json.dumps({"driver_debuts": {"D1": 2020}}), encoding="utf-8"
    )

    catalog: dict[str, dict[str, Any]] = {
        "event-1": {"actual_race_finish_order": [{"driver": "D1", "position": 2, "dnf": False}]}
    }

    # One entry inside the replay's own checkpoint (FP2), one from an unrelated PRE
    # cache entry that must be excluded even though it shares event/variant.
    _write_cache_entry(tmp_path, checkpoint="FP2", event_id="event-1", seed=17, position=2)
    _write_cache_entry(tmp_path, checkpoint="PRE", event_id="event-1", seed=17, position=99)

    replay = {"scored_events": [{"checkpoints": {"FP2": {}}}]}
    result = report_v2._driver_cohort_decomposition_scoped(replay, catalog)

    assert result["computed"] is True
    # Only the FP2 row (predicted 2, actual 2 -> error 0) contributed; the PRE row
    # (predicted 99, actual 2 -> error 97) must not have leaked in.
    assert result["by_cohort"]["established"]["finisher_mae"] == 0.0
    assert result["by_cohort"]["established"]["n_driver_observations"] == 1
