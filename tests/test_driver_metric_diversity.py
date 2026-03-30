"""Verify driver characteristics keep separate pace and racecraft dimensions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_driver_metrics_are_not_identical():
    """Most drivers should have distinct race pace, skill, and overtaking metrics."""
    char_file = Path("data/processed/driver_characteristics.json")
    if not char_file.exists():
        pytest.skip("Driver characteristics artifact not present")

    with open(char_file) as handle:
        drivers = json.load(handle).get("drivers", {})

    identical_count = 0
    for _code, info in drivers.items():
        race_pace = info.get("pace", {}).get("race_pace")
        skill_score = info.get("racecraft", {}).get("skill_score")
        overtaking_skill = info.get("racecraft", {}).get("overtaking_skill")
        if race_pace is not None and race_pace == skill_score == overtaking_skill:
            identical_count += 1

    max_allowed = max(3, len(drivers) // 4)
    assert identical_count <= max_allowed, (
        f"{identical_count}/{len(drivers)} drivers still have collapsed pace and racecraft axes"
    )
