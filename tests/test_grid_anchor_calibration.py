"""Tests for deterministic offline source-specific grid-anchor calibration."""

from __future__ import annotations

import math

import pytest

from src.analysis.grid_anchor_calibration import (
    fit_source_specific_grid_anchors,
    validate_grid_anchor_event_separation,
)
from src.utils.config_schema import GridAnchorSourceCalibratedConfig

_EVENT_AT = "2026-01-01T12:00:00Z"
_CUTOFF_AT = "2026-02-01T12:00:00Z"


def _row(
    event: str,
    simulated: float,
    grid: float,
    actual: float,
    source: str = "predicted_joint",
    *,
    event_at: str = _EVENT_AT,
    driver_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": event,
        "event_at": event_at,
        "grid_source_detail": source,
        "simulated_position": simulated,
        "grid_position": grid,
        "actual_position": actual,
    }
    if driver_id is not None:
        row["driver_id"] = driver_id
    return row


def test_fits_exact_piecewise_linear_minimum() -> None:
    rows = [
        _row("A", 1, 3, 2),
        _row("A", 3, 1, 2),
        _row("B", 2, 4, 3),
        _row("B", 4, 2, 3),
    ]

    result = fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=2)

    assert result["source_calibrated"] == {"predicted_joint": 0.5}
    diagnostic = result["diagnostics"]["predicted_joint"]
    assert diagnostic["mean_event_mae"] == pytest.approx(0.0)
    assert diagnostic["pre_anchor_mean_event_mae"] == pytest.approx(1.0)
    GridAnchorSourceCalibratedConfig(**result["source_calibrated"])


def test_events_receive_equal_total_weight_and_tie_break_is_conservative() -> None:
    rows = [_row("many", 1, 2, 2, driver_id=f"D{index:02d}") for index in range(20)]
    rows.append(_row("single", 1, 2, 1, driver_id="ONLY"))

    result = fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=2)

    # The two event MAEs are 1-w and w: the event-balanced objective is flat.
    # A pooled-row objective would incorrectly choose w=1 for the larger event.
    assert result["source_calibrated"]["predicted_joint"] == 0.0
    diagnostic = result["diagnostics"]["predicted_joint"]
    assert diagnostic["tied_minimizer_count"] == 2
    assert diagnostic["event_row_counts"] == {"many": 20, "single": 1}


def test_sources_are_independent_and_insufficient_source_is_not_emitted() -> None:
    rows = [
        _row("A", 1, 3, 2),
        _row("B", 1, 3, 2),
        _row("C", 2, 4, 4, "actual_starting_grid"),
    ]

    result = fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=2)

    assert result["source_calibrated"] == {"predicted_joint": 0.5}
    assert result["diagnostics"]["actual_starting_grid"]["status"] == "insufficient_events"
    assert result["diagnostics"]["actual_qualifying"]["status"] == "no_rows"
    assert result["runtime_activation_allowed"] is False


def test_output_and_digest_are_invariant_to_input_order() -> None:
    rows = [
        _row("B", 4, 2, 3),
        _row("A", 1, 3, 2),
        _row("B", 2, 4, 3),
        _row("A", 3, 1, 2),
    ]

    forward = fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=2)
    reverse = fit_source_specific_grid_anchors(
        list(reversed(rows)),
        cutoff_at=_CUTOFF_AT,
        min_events=2,
    )

    assert forward == reverse
    assert len(forward["provenance"]["input_sha256"]) == 64


def test_provenance_records_canonical_cutoff_and_sorted_training_timestamps() -> None:
    rows = [
        _row("late", 1, 3, 2, event_at="2026-01-20T13:00:00+01:00"),
        _row("early", 3, 1, 2, event_at="2026-01-10T12:00:00Z"),
    ]

    result = fit_source_specific_grid_anchors(
        rows,
        cutoff_at="2026-02-01T13:00:00+01:00",
        min_events=2,
    )

    provenance = result["provenance"]
    assert result["schema_version"] == 2
    assert provenance["cutoff_at"] == "2026-02-01T12:00:00Z"
    assert provenance["training_event_ids"] == ["early", "late"]
    assert provenance["training_event_timestamps"] == [
        "2026-01-10T12:00:00Z",
        "2026-01-20T12:00:00Z",
    ]
    assert provenance["input_event_count"] == 2
    assert provenance["input_row_count"] == 2


def test_input_digest_binds_canonical_cutoff_and_event_timestamps() -> None:
    rows = [_row("A", 1, 3, 2, event_at="2026-01-01T13:00:00+01:00")]

    utc = fit_source_specific_grid_anchors(
        rows,
        cutoff_at="2026-02-01T12:00:00Z",
        min_events=1,
    )
    equivalent_offset = fit_source_specific_grid_anchors(
        rows,
        cutoff_at="2026-02-01T13:00:00+01:00",
        min_events=1,
    )
    later_cutoff = fit_source_specific_grid_anchors(
        rows,
        cutoff_at="2026-02-02T12:00:00Z",
        min_events=1,
    )

    assert utc == equivalent_offset
    assert utc["provenance"]["input_sha256"] != later_cutoff["provenance"]["input_sha256"]


@pytest.mark.parametrize(
    "event_at",
    ["2026-02-01T12:00:00Z", "2026-02-01T12:00:01Z"],
)
def test_event_timestamp_must_be_strictly_before_cutoff(event_at: str) -> None:
    with pytest.raises(ValueError, match="strictly before cutoff_at"):
        fit_source_specific_grid_anchors(
            [_row("A", 1, 2, 1, event_at=event_at)],
            cutoff_at=_CUTOFF_AT,
            min_events=1,
        )


def test_cutoff_and_event_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="cutoff_at must include a timezone"):
        fit_source_specific_grid_anchors([], cutoff_at="2026-02-01T12:00:00")
    with pytest.raises(ValueError, match="event_at must include a timezone"):
        fit_source_specific_grid_anchors(
            [_row("A", 1, 2, 1, event_at="2026-01-01T12:00:00")],
            cutoff_at=_CUTOFF_AT,
            min_events=1,
        )


def test_each_event_requires_one_consistent_timestamp() -> None:
    rows = [
        _row("A", 1, 3, 2, event_at="2026-01-01T12:00:00Z"),
        _row("A", 3, 1, 2, event_at="2026-01-02T12:00:00Z"),
    ]

    with pytest.raises(ValueError, match="inconsistent event_at timestamps"):
        fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=1)


def test_equivalent_timezone_offsets_are_one_consistent_event_timestamp() -> None:
    rows = [
        _row("A", 1, 3, 2, event_at="2026-01-01T12:00:00Z"),
        _row("A", 3, 1, 2, event_at="2026-01-01T13:00:00+01:00"),
    ]

    result = fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=1)

    assert result["provenance"]["training_event_timestamps"] == ["2026-01-01T12:00:00Z"]


def test_exact_duplicate_replay_rows_are_rejected() -> None:
    replay_row = _row("A", 1, 3, 2)

    with pytest.raises(ValueError, match="duplicates replay row 0"):
        fit_source_specific_grid_anchors(
            [replay_row, dict(replay_row)],
            cutoff_at=_CUTOFF_AT,
            min_events=1,
        )


def test_duplicate_driver_source_observations_are_rejected() -> None:
    rows = [
        _row("A", 1, 3, 2, driver_id="VER"),
        _row("A", 2, 4, 3, driver_id="VER"),
    ]

    with pytest.raises(ValueError, match="duplicates driver/source observation"):
        fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=1)


def test_driver_identifier_aliases_cannot_bypass_duplicate_detection() -> None:
    first = _row("A", 1, 3, 2)
    first["driver_id"] = "VER"
    second = _row("A", 2, 4, 3)
    second["driver"] = "VER"

    with pytest.raises(ValueError, match="duplicates driver/source observation"):
        fit_source_specific_grid_anchors([first, second], cutoff_at=_CUTOFF_AT, min_events=1)


def test_conflicting_driver_identifier_aliases_are_rejected() -> None:
    replay_row = _row("A", 1, 3, 2)
    replay_row.update({"driver_id": "VER", "driver": "HAD"})

    with pytest.raises(ValueError, match="conflicting driver identifiers"):
        fit_source_specific_grid_anchors([replay_row], cutoff_at=_CUTOFF_AT, min_events=1)


def test_same_driver_may_have_one_row_per_distinct_source() -> None:
    rows = [
        _row("A", 1, 3, 2, driver_id="VER"),
        _row("A", 2, 4, 3, "actual_starting_grid", driver_id="VER"),
    ]

    result = fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=1)

    assert set(result["source_calibrated"]) == {
        "predicted_joint",
        "actual_starting_grid",
    }


def test_event_separation_accepts_disjoint_held_out_events() -> None:
    validate_grid_anchor_event_separation(
        ["A", "B"],
        promotion_event_ids=["C"],
        evaluation_event_ids=["D", "E"],
    )


@pytest.mark.parametrize(
    ("promotion", "evaluation", "message"),
    [
        (["A"], ["C"], "promotion=\\['A'\\]"),
        (["C"], ["B"], "evaluation=\\['B'\\]"),
        (["A"], ["B"], "promotion=\\['A'\\].*evaluation=\\['B'\\]"),
    ],
)
def test_event_separation_rejects_training_overlap(
    promotion: list[str],
    evaluation: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_grid_anchor_event_separation(
            ["A", "B"],
            promotion_event_ids=promotion,
            evaluation_event_ids=evaluation,
        )


def test_event_separation_rejects_ambiguous_event_id_sequences() -> None:
    with pytest.raises(ValueError, match="must not contain duplicate"):
        validate_grid_anchor_event_separation(["A", "A"])
    with pytest.raises(ValueError, match="must not be blank"):
        validate_grid_anchor_event_separation(["A"], evaluation_event_ids=[" "])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"event_id": "A"}], "missing required fields"),
        ([_row("A", 1, 2, 1, "unknown")], "not allowed"),
        ([_row("A", math.nan, 2, 1)], "finite positive"),
        ([_row("", 1, 2, 1)], "must not be blank"),
    ],
)
def test_invalid_replay_rows_fail_closed(rows: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_source_specific_grid_anchors(rows, cutoff_at=_CUTOFF_AT, min_events=1)


def test_invalid_fit_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fit_source_specific_grid_anchors([], cutoff_at=_CUTOFF_AT, min_events=0)
    with pytest.raises(ValueError, match="duplicates"):
        fit_source_specific_grid_anchors(
            [],
            cutoff_at=_CUTOFF_AT,
            allowed_source_details=["predicted_joint", "predicted_joint"],
        )
    with pytest.raises(ValueError, match="unsupported"):
        fit_source_specific_grid_anchors(
            [],
            cutoff_at=_CUTOFF_AT,
            allowed_source_details=["future_source"],
        )
