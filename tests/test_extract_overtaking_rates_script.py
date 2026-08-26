"""Tests for the overtaking-rate extraction script's aggregation and write logic.

The lap-counting rule itself lives in and is tested by
``src/extractors/overtaking.py`` / ``tests/test_overtaking_extractor.py`` (this
script reuses ``extract_overtakes_from_race`` rather than reimplementing it, so the
same definition -- both cars in a swap, pit-out laps dropped, sub-5-driver laps
skipped -- applies here too).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module():
    """Load the overtaking-rate extraction script as a module."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_overtaking_rates.py"
    spec = importlib.util.spec_from_file_location("extract_overtaking_rates_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_by_data_key_rounds_to_three_decimals():
    module = _load_script_module()
    measurements = [
        module.Measurement(
            race_name="Race A",
            data_key="Circuit",
            avg_changes_per_lap=2.29849,
            laps_analyzed=57,
        )
    ]

    aggregated = module.aggregate_by_data_key(measurements)

    assert aggregated == {"Circuit": (2.298, 1)}


def test_apply_measurements_only_touches_matched_tracks_and_preserves_other_fields(tmp_path):
    module = _load_script_module()
    track_path = tmp_path / "2026_track_characteristics.json"
    track_path.write_text(
        json.dumps(
            {
                "year": 2026,
                "tracks": {
                    "Australian Grand Prix": {
                        "type": "permanent",
                        "pit_stop_loss": 18.2,
                        "overtaking_difficulty": 0.5,
                    },
                    "Monaco Grand Prix": {
                        "type": "street",
                        "overtaking_difficulty": 0.95,
                    },
                },
            }
        )
    )

    data = module.apply_measurements(track_path, {"Australian Grand Prix": (2.298, 1)})

    australia = data["tracks"]["Australian Grand Prix"]
    monaco = data["tracks"]["Monaco Grand Prix"]
    assert australia["overtaking_avg_changes_per_lap"] == 2.298
    assert australia["overtaking_observed_races"] == 1
    assert australia["type"] == "permanent"
    assert australia["pit_stop_loss"] == 18.2
    # Untouched track keeps exactly its original fields.
    assert monaco == {"type": "street", "overtaking_difficulty": 0.95}


def test_apply_measurements_skips_unknown_track_key(tmp_path):
    module = _load_script_module()
    track_path = tmp_path / "2026_track_characteristics.json"
    track_path.write_text(json.dumps({"tracks": {"Monaco Grand Prix": {"type": "street"}}}))

    data = module.apply_measurements(track_path, {"Imaginary Grand Prix": (5.0, 1)})

    assert "Imaginary Grand Prix" not in data["tracks"]
    assert data["tracks"]["Monaco Grand Prix"] == {"type": "street"}
