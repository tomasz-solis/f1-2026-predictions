from __future__ import annotations

import pytest

from src.analysis.challenger_movements import build_full_field_movement_audit


def _grid(order: list[str]) -> list[dict[str, object]]:
    return [
        {"driver": driver, "team": f"Team {driver}", "position": position}
        for position, driver in enumerate(order, start=1)
    ]


def _h2h(probability: float) -> list[dict[str, object]]:
    return [
        {
            "team": "Team Pair",
            "driver_a": "A",
            "driver_b": "B",
            "p_driver_a_ahead": probability,
            "p_driver_b_ahead": 1.0 - probability,
        }
    ]


def test_movement_audit_covers_field_and_flags_strict_thresholds() -> None:
    audit = build_full_field_movement_audit(
        champion_grid=_grid(["A", "B", "C", "D"]),
        challenger_grid=_grid(["D", "B", "C", "A"]),
        champion_teammate_h2h=_h2h(0.50),
        challenger_teammate_h2h=_h2h(0.61),
    )

    assert audit["field_size"] == 4
    assert len(audit["grid_movements"]) == 4
    assert audit["review_required_count"] == 3
    assert audit["reviewed_count"] == 0
    assert audit["review_complete"] is False
    assert len(audit["audit_sha256"]) == 64


def test_exact_thresholds_do_not_require_review_and_pair_orientation_is_stable() -> None:
    champion_h2h = _h2h(0.50)
    challenger_h2h = [
        {
            "team": "Team Pair",
            "driver_a": "B",
            "driver_b": "A",
            "p_driver_a_ahead": 0.40,
            "p_driver_b_ahead": 0.60,
        }
    ]
    audit = build_full_field_movement_audit(
        champion_grid=_grid(["A", "B", "C"]),
        challenger_grid=_grid(["C", "B", "A"]),
        champion_teammate_h2h=champion_h2h,
        challenger_teammate_h2h=challenger_h2h,
    )

    assert audit["review_required_count"] == 0
    assert audit["review_complete"] is True


def test_material_reviews_require_an_explicit_decision() -> None:
    kwargs = {
        "champion_grid": _grid(["A", "B", "C", "D"]),
        "challenger_grid": _grid(["D", "B", "C", "A"]),
        "champion_teammate_h2h": _h2h(0.50),
        "challenger_teammate_h2h": _h2h(0.61),
    }
    partial = build_full_field_movement_audit(
        **kwargs,
        reviews={"grid:A": {"decision": "accepted", "note": "evidence agrees"}},
    )
    complete = build_full_field_movement_audit(
        **kwargs,
        reviews={
            "grid:A": {"decision": "accepted"},
            "grid:D": {"decision": "accepted"},
            "h2h:Team Pair:A:B": {"decision": "accepted"},
        },
    )

    assert partial["reviewed_count"] == 1
    assert partial["review_complete"] is False
    assert complete["reviewed_count"] == 3
    assert complete["review_complete"] is True


def test_movement_audit_rejects_different_driver_sets() -> None:
    with pytest.raises(ValueError, match="same driver set"):
        build_full_field_movement_audit(
            champion_grid=_grid(["A", "B"]),
            challenger_grid=_grid(["A", "C"]),
            champion_teammate_h2h=[],
            challenger_teammate_h2h=[],
        )
