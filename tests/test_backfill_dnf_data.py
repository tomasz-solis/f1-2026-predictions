"""Tests for the DNF backfill script's merge logic (non-destructive actuals update)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_backfill_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_dnf_data.py"
    spec = importlib.util.spec_from_file_location("backfill_dnf_data_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_dnf_preserves_order_and_only_adds_flag():
    module = _load_backfill_module()
    existing = [
        {"position": 1, "driver": "ANT", "team": "Mercedes"},
        {"position": 2, "driver": "RUS", "team": "Mercedes"},
        {"position": 20, "driver": "VER", "team": "Red Bull Racing"},
    ]
    # Fresh fetch may differ in order; only the per-driver DNF flag should be used.
    fetched_dnf = {"ANT": False, "RUS": False, "VER": True}
    labelled = module._merge_dnf_into_block(existing, fetched_dnf)

    assert labelled == 1
    # Positions/drivers/teams untouched; dnf attached.
    assert [r["driver"] for r in existing] == ["ANT", "RUS", "VER"]
    assert existing[0] == {"position": 1, "driver": "ANT", "team": "Mercedes", "dnf": False}
    assert existing[2]["dnf"] is True


def test_match_session_for_block_picks_best_order_match():
    module = _load_backfill_module()
    block = [
        {"position": 1, "driver": "RUS"},
        {"position": 2, "driver": "PIA"},
        {"position": 3, "driver": "NOR"},
    ]
    fetched = {
        "R": [{"driver": "ANT"}, {"driver": "RUS"}, {"driver": "HAM"}],  # GP order
        "Sprint": [{"driver": "RUS"}, {"driver": "PIA"}, {"driver": "NOR"}],  # matches block
    }
    assert module._match_session_for_block(block, fetched) == "Sprint"


def test_backfill_actuals_labels_targets_and_legacy_block():
    module = _load_backfill_module()
    prediction = {
        "actuals": {
            "race": [  # sprint weekend: legacy race mirrors the Sprint
                {"position": 1, "driver": "RUS", "team": "Mercedes"},
                {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
            ],
            "targets": {
                "grand_prix_race": [
                    {"position": 1, "driver": "ANT", "team": "Mercedes"},
                    {"position": 2, "driver": "NOR", "team": "McLaren"},
                ],
                "sprint_race": [
                    {"position": 1, "driver": "RUS", "team": "Mercedes"},
                    {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
                ],
            },
        }
    }
    fetched = {
        "R": [
            {"position": 1, "driver": "ANT", "team": "Mercedes", "dnf": False},
            {"position": 2, "driver": "NOR", "team": "McLaren", "dnf": True},
        ],
        "Sprint": [
            {"position": 1, "driver": "RUS", "team": "Mercedes", "dnf": False},
            {"position": 2, "driver": "VER", "team": "Red Bull Racing", "dnf": True},
        ],
    }
    labelled = module._backfill_actuals(prediction, fetched=fetched)

    # 1 DNF in the GP target + 1 in the sprint target.
    assert labelled == 2
    gpr = prediction["actuals"]["targets"]["grand_prix_race"]
    assert gpr[1]["driver"] == "NOR" and gpr[1]["dnf"] is True
    assert gpr[0]["dnf"] is False
    # Legacy race block (the sprint) gets VER's DNF from the matched Sprint session.
    legacy = prediction["actuals"]["race"]
    assert legacy[1]["driver"] == "VER" and legacy[1]["dnf"] is True


def test_merge_is_noop_without_fetched_data():
    module = _load_backfill_module()
    rows = [{"position": 1, "driver": "VER", "team": "RBR"}]
    assert module._merge_dnf_into_block(rows, {}) == 0
    assert "dnf" not in rows[0]
